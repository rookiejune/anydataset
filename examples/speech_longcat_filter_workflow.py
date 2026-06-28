"""Recipe for materializing LongCat views and filtering a speech dataset.

This example is intentionally written as a readable workflow instead of a
configurable command-line tool. It assumes a GPU environment with local
LongCat weights and anytrain evaluators available.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from anydataset.provider.longcat import LongCatProvider
from anytrain.evaluator.speech import UTMOSEvaluator, WhisperASREvaluator
from anytrain.evaluator.text import TextComparisonEvaluator

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    FilterDecision,
    FilterRule,
    Modality,
    Preset,
    Role,
    Sample,
    TextItem,
    TextView,
    ViewMaterializer,
)

SPLIT = "train"
CONFIG = "en_us"

ROOT = Path("/mnt/pami202/zhuyin/datasets/anydataset/speech-longcat-filter")
DELTA = ROOT / "fleurs-longcat-delta"
BATCH_SIZE = 8
BATCH_DELTA = ROOT / f"fleurs-longcat-delta-bs{BATCH_SIZE}"
FILTER_CACHE = ROOT / "filter-cache"
DATASET_CACHE = ROOT / "dataset-cache"
COMPARE_LIMIT: int | None = None

DEVICE = "cuda:0"
MIN_UTMOS = 3.0
MAX_WER = 0.4
MIN_CHRF = 50.0


class Quality:
    def __init__(self) -> None:
        device = _filter_device()
        self.asr = WhisperASREvaluator(
            model_name="large-v3",
            device=device,
            decode_options={"language": "en", "temperature": 0.0},
        )
        self.text = TextComparisonEvaluator()
        self.utmos = UTMOSEvaluator(
            device=device,
            backend_load_options={"trust_repo": True},
        )

    def __call__(self, sample: Sample) -> FilterDecision:
        audio = cast(AudioItem, sample[Role.DEFAULT, Modality.AUDIO])
        transcript = cast(TextItem, sample[Role.DEFAULT, Modality.TEXT])
        waveform, sample_rate = audio.views[AudioView.WAVEFORM]
        reference = transcript.views[TextView.TEXT]
        prediction = self.asr.transcribe(waveform, sample_rate)
        metrics = self.text.evaluate(prediction, reference)
        score = float(self.utmos.evaluate(waveform, sample_rate)["utmos"])
        bad_text = metrics["wer"] > MAX_WER or metrics["chrf"] < MIN_CHRF
        reject = score < MIN_UTMOS and bad_text
        return FilterDecision(
            label="reject" if reject else "accept",
            metrics={
                "utmos": score,
                "asr_text": str(prediction),
                "reference_text": reference,
                "wer": float(metrics["wer"]),
                "chrf": float(metrics["chrf"]),
                "bleu": float(metrics["bleu"]),
            },
        )


def quality_factory():
    return Quality()


def _filter_device() -> str:
    import os

    return os.environ.get("ANYDATASET_FILTER_DEVICE", DEVICE)


rule = FilterRule("reject_utmos_lt_3_and_bad_asr_text_v1", quality_factory)


def fleurs_dataset():
    return Preset.FLEURS.create(split=SPLIT, config_name=CONFIG, streaming=True)


def longcat_provider(device: str):
    return LongCatProvider(
        decoders=("16k_4codebooks",),
        device=device,
        local_files_only=True,
    )


def materialize_longcat_delta(output_dir: Path, *, batch_size: int) -> dict[str, Any]:
    if output_dir.exists():
        return {
            "path": str(output_dir),
            "batch_size": batch_size,
            "materialized": False,
            "seconds": None,
        }

    start = time.perf_counter()
    ViewMaterializer(output_dir, split=SPLIT, batch_size=batch_size).write(
        dataset_factory=fleurs_dataset,
        provider_factory=longcat_provider,
        devices="auto",
    )
    return {
        "path": str(output_dir),
        "batch_size": batch_size,
        "materialized": True,
        "seconds": time.perf_counter() - start,
    }


def compare_longcat_deltas(
    baseline_dir: Path,
    batch_dir: Path,
    *,
    limit: int | None,
) -> dict[str, Any]:
    baseline = AnyDataset(f"store://{baseline_dir}:{SPLIT}", cache_root=DATASET_CACHE)
    batched = AnyDataset(f"store://{batch_dir}:{SPLIT}", cache_root=DATASET_CACHE)
    if len(baseline) != len(batched):
        raise ValueError(
            f"LongCat delta sample counts differ: {len(baseline)} != {len(batched)}."
        )

    checked = len(baseline) if limit is None else min(limit, len(baseline))
    start = time.perf_counter()
    for index in range(checked):
        _assert_same_codes(
            _longcat_codes(baseline[index]),
            _longcat_codes(batched[index]),
            index,
        )
    return {
        "baseline": str(baseline_dir),
        "batch": str(batch_dir),
        "checked": checked,
        "matches": True,
        "seconds": time.perf_counter() - start,
    }


def _longcat_codes(sample: Sample) -> Mapping[str, Any]:
    audio = cast(AudioItem, sample[Role.DEFAULT, Modality.AUDIO])
    codes = audio.views[AudioView.LONGCAT]
    if not isinstance(codes, Mapping):
        raise TypeError("LongCat view must be a mapping of code tensors.")
    return codes


def _assert_same_codes(
    baseline: Mapping[str, Any],
    batched: Mapping[str, Any],
    index: int,
) -> None:
    if set(baseline) != set(batched):
        raise ValueError(
            f"LongCat code keys differ at sample {index}: "
            f"{sorted(baseline)} != {sorted(batched)}."
        )
    for name in sorted(baseline):
        left = baseline[name]
        right = batched[name]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if left.dtype != right.dtype or left.shape != right.shape:
                raise ValueError(
                    f"LongCat code {name!r} metadata differs at sample {index}: "
                    f"{left.dtype}/{tuple(left.shape)} != "
                    f"{right.dtype}/{tuple(right.shape)}."
                )
            if not torch.equal(left, right):
                raise ValueError(f"LongCat code {name!r} differs at sample {index}.")
            continue
        if left != right:
            raise ValueError(f"LongCat code {name!r} differs at sample {index}.")


# Materialize both the original single-sample path and the batched path, then
# compare their LongCat outputs before reusing the batched delta downstream.
baseline_timing = materialize_longcat_delta(DELTA, batch_size=1)
batch_timing = materialize_longcat_delta(BATCH_DELTA, batch_size=BATCH_SIZE)
longcat_comparison = compare_longcat_deltas(
    DELTA,
    BATCH_DELTA,
    limit=COMPARE_LIMIT,
)

# Merge the original FLEURS waveform/text views with the generated LongCat view.
merged = AnyDataset(f"store://{BATCH_DELTA}:{SPLIT}", cache_root=DATASET_CACHE)
sample = merged[0]
audio = cast(AudioItem, sample[Role.DEFAULT, Modality.AUDIO])
transcript = cast(TextItem | None, sample.get((Role.DEFAULT, Modality.TEXT)))
if AudioView.WAVEFORM not in audio.views or (
    transcript is None or TextView.TEXT not in transcript.views
):
    merged = merged.merge(
        Preset.FLEURS.create(split=SPLIT, config_name=CONFIG, streaming=True),
    )

# Apply the named quality rule and select the accepted partition.
result = rule.apply(merged, metrics=True, device="auto", cache_root=FILTER_CACHE)
filtered = result.select("accept")
sample = filtered[0]
audio = cast(AudioItem, sample[Role.DEFAULT, Modality.AUDIO])

summary = {
    "delta": str(BATCH_DELTA),
    "filter_cache": str(FILTER_CACHE),
    "longcat_batch": batch_timing,
    "longcat_baseline": baseline_timing,
    "longcat_comparison": longcat_comparison,
    "metrics": None if result.metrics_path is None else str(result.metrics_path),
    "accepted": len(filtered),
    "labels": result.labels,
    "first_views": sorted(view.value for view in audio.views),
}
print(json.dumps(summary, indent=2, sort_keys=True))
