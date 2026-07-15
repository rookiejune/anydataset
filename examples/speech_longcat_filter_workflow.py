"""Recipe for materializing LongCat views and filtering a speech dataset.

This example is intentionally written as a readable workflow instead of a
configurable command-line tool. It assumes a GPU environment with local
LongCat weights and anytrain evaluators available.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
import unicodedata

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
    IterableAnyDataset,
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
ANYDATASET_HOME = ROOT / "home"
COMPARE_LIMIT: int | None = None

DEVICE = "cuda:0"
MIN_UTMOS = 2.8
MIN_CHRF = 50.0
MAX_SECONDS_PER_TEXT_UNIT = 4.0
MIN_PEAK_AMPLITUDE = 0.05

os.environ.setdefault("ANYDATASET_HOME", str(ANYDATASET_HOME))


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
        wave = torch.as_tensor(waveform)
        duration_seconds = float(wave.shape[-1]) / float(sample_rate)
        peak_amplitude = (
            float(wave.detach().abs().max().cpu().item()) if wave.numel() else 0.0
        )
        text_units = _text_units(str(reference))
        seconds_per_text_unit = duration_seconds / float(text_units)
        reject = (
            score < MIN_UTMOS
            or metrics["chrf"] < MIN_CHRF
            or seconds_per_text_unit > MAX_SECONDS_PER_TEXT_UNIT
            or peak_amplitude < MIN_PEAK_AMPLITUDE
        )
        return FilterDecision(
            label="reject" if reject else "accept",
            metrics={
                "utmos": score,
                "asr_text": str(prediction),
                "reference_text": reference,
                "wer": float(metrics["wer"]),
                "chrf": float(metrics["chrf"]),
                "bleu": float(metrics["bleu"]),
                "duration_seconds": duration_seconds,
                "peak_amplitude": peak_amplitude,
                "text_units": text_units,
                "seconds_per_text_unit": seconds_per_text_unit,
            },
        )


def quality_factory():
    return Quality()


def _filter_device() -> str:
    import os

    return os.environ.get("ANYDATASET_FILTER_DEVICE", DEVICE)


def _text_units(text: str) -> int:
    count = 0
    in_word = False
    for char in text:
        if _is_cjk(char):
            count += 1
            in_word = False
        elif char.isalnum():
            if not in_word:
                count += 1
                in_word = True
        elif unicodedata.category(char).startswith("M"):
            continue
        else:
            in_word = False
    return max(count, 1)


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x3134F
        or 0x31350 <= codepoint <= 0x323AF
    )


rule = FilterRule("speech_quality_v2_utmos28_chrf50_len4_peak005", quality_factory)


def fleurs_dataset():
    return IterableAnyDataset.preset(
        "fleurs", split=SPLIT, config_name=CONFIG, streaming=True
    )


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
    baseline = AnyDataset(f"store://{baseline_dir}:{SPLIT}")
    batched = AnyDataset(f"store://{batch_dir}:{SPLIT}")
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
merged = AnyDataset(f"store://{BATCH_DELTA}:{SPLIT}")
sample = merged[0]
audio = cast(AudioItem, sample[Role.DEFAULT, Modality.AUDIO])
transcript = cast(TextItem | None, sample.get((Role.DEFAULT, Modality.TEXT)))
if AudioView.WAVEFORM not in audio.views or (
    transcript is None or TextView.TEXT not in transcript.views
):
    merged = merged.merge(
        IterableAnyDataset.preset(
            "fleurs", split=SPLIT, config_name=CONFIG, streaming=True
        ),
    )

# Apply the named quality rule and select the accepted partition.
result = rule.apply(merged, metrics=True, device="cpu")
filtered = result.select("accept")
sample = filtered[0]
audio = cast(AudioItem, sample[Role.DEFAULT, Modality.AUDIO])

summary = {
    "delta": str(BATCH_DELTA),
    "filter_cache": str(result.cache_path),
    "longcat_batch": batch_timing,
    "longcat_baseline": baseline_timing,
    "longcat_comparison": longcat_comparison,
    "metrics": None if result.metrics_path is None else str(result.metrics_path),
    "accepted": len(filtered),
    "labels": result.labels,
    "first_views": sorted(view.value for view in audio.views),
}
print(json.dumps(summary, indent=2, sort_keys=True))
