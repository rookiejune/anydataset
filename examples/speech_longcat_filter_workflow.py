"""Materialize LongCat views and filter FLEURS speech samples.

The workflow writes streaming FLEURS rows to a map-style base store, writes a
LongCat-only delta store, merges the two stores logically, and builds cached
speech-quality partitions. It requires anytrain's LongCat and speech extras at
execution time, but importing this module does not load data or models.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from anydataset.dataset import AnyDataset, IterableAnyDataset, MergedDataset
from anydataset.filter import FilterRule
from anydataset.provider import LongCatProvider
from anydataset.quality.speech import SpeechQuality, SpeechQualityProfile
from anydataset.store import ViewMaterializer
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Preset,
    Role,
    Sample,
    Source,
    Spec,
    TextItem,
    TextView,
)

OUTPUT_ROOT_ENV = "ANYDATASET_SPEECH_WORKFLOW_ROOT"
DEFAULT_OUTPUT_ROOT = Path("storage/speech-longcat-filter")

SPLIT = "train"
CONFIG = "en_us"
BASE_DATASET_ID = f"fleurs-{CONFIG}"

LONGCAT_DECODER = "16k_4codebooks"
LONGCAT_BATCH_SIZE = 8
LONGCAT_DEVICES_ENV = "ANYDATASET_LONGCAT_DEVICES"
LONGCAT_CACHE_ENV = "ANYDATASET_LONGCAT_CACHE_DIR"
LONGCAT_LOCAL_ONLY_ENV = "ANYDATASET_LONGCAT_LOCAL_FILES_ONLY"
COMPARE_LIMIT: int | None = None

QUALITY_DEVICES_ENV = "ANYDATASET_SPEECH_QUALITY_DEVICES"
WHISPER_MODEL = "large-v3"
RULE_NAME = "speech_quality_v1_whisper_large_v3_utmos28_chrf50_len4_peak005"


def output_root() -> Path:
    value = os.environ.get(OUTPUT_ROOT_ENV, str(DEFAULT_OUTPUT_ROOT))
    return Path(value).expanduser()


def base_store_path() -> Path:
    return output_root() / "fleurs-base"


def delta_store_path(batch_size: int) -> Path:
    return output_root() / f"fleurs-longcat-bs{batch_size}"


def fleurs_dataset() -> IterableAnyDataset:
    return IterableAnyDataset.preset(
        Preset.FLEURS,
        split=SPLIT,
        config_name=CONFIG,
        streaming=True,
    )


def base_dataset() -> AnyDataset:
    return store_dataset(base_store_path())


def batched_delta_dataset() -> AnyDataset:
    return store_dataset(delta_store_path(LONGCAT_BATCH_SIZE))


def merged_dataset() -> MergedDataset:
    return base_dataset().merge(batched_delta_dataset())


def store_dataset(path: Path) -> AnyDataset:
    return AnyDataset(
        Spec(
            source=Source.STORE,
            path=str(path),
            split=SPLIT,
        )
    )


def longcat_provider(device: str) -> LongCatProvider:
    return LongCatProvider(
        cache_dir=os.environ.get(LONGCAT_CACHE_ENV),
        decoder=LONGCAT_DECODER,
        device=device,
        local_files_only=env_flag(LONGCAT_LOCAL_ONLY_ENV, default=True),
    )


def quality_factory() -> SpeechQuality:
    from anytrain.evaluator.speech import (
        SpeechEvaluator,
        UTMOSEvaluator,
        WhisperASREvaluator,
    )

    device = os.environ.get("ANYDATASET_FILTER_DEVICE")
    if device is None:
        raise RuntimeError("ANYDATASET_FILTER_DEVICE is not set by the filter runtime.")
    evaluator = SpeechEvaluator(
        asr=WhisperASREvaluator(
            model_name=WHISPER_MODEL,
            device=device,
        ),
        utmos=UTMOSEvaluator(
            device=device,
            backend_load_options={"trust_repo": True},
        ),
    )
    return SpeechQuality(
        profile=SpeechQualityProfile(
            min_utmos=2.8,
            min_chrf=50.0,
            max_seconds_per_text_unit=4.0,
            min_peak_amplitude=0.05,
        ),
        evaluator=evaluator,
        decode_options={"language": "en", "temperature": 0.0},
    )


def materialize_base_store() -> dict[str, Any]:
    path = base_store_path()
    if path.exists():
        count = len(base_dataset())
        return store_result(path, count=count, materialized=False, seconds=None)

    start = time.perf_counter()
    fleurs_dataset().write(
        path,
        dataset_id=BASE_DATASET_ID,
        split=SPLIT,
    )
    return store_result(
        path,
        count=len(base_dataset()),
        materialized=True,
        seconds=time.perf_counter() - start,
    )


def materialize_longcat_delta(path: Path, *, batch_size: int) -> dict[str, Any]:
    if path.exists():
        count = len(store_dataset(path))
        return store_result(path, count=count, materialized=False, seconds=None)

    start = time.perf_counter()
    ViewMaterializer(
        path,
        split=SPLIT,
        batch_size=batch_size,
        input_id=f"{BASE_DATASET_ID}-{SPLIT}-v1",
        provider_id=f"longcat-{LONGCAT_DECODER}-v1",
    ).write(
        dataset_factory=base_dataset,
        provider_factory=longcat_provider,
        devices=os.environ.get(LONGCAT_DEVICES_ENV, "auto"),
    )
    return store_result(
        path,
        count=len(store_dataset(path)),
        materialized=True,
        seconds=time.perf_counter() - start,
    )


def compare_longcat_deltas(
    baseline_path: Path,
    batched_path: Path,
    *,
    limit: int | None,
) -> dict[str, Any]:
    baseline = store_dataset(baseline_path)
    batched = store_dataset(batched_path)
    if len(baseline) != len(batched):
        raise ValueError(
            f"LongCat delta sample counts differ: {len(baseline)} != {len(batched)}."
        )

    checked = len(baseline) if limit is None else min(limit, len(baseline))
    start = time.perf_counter()
    for index in range(checked):
        assert_same_codes(
            longcat_codes(baseline[index]),
            longcat_codes(batched[index]),
            index=index,
        )
    return {
        "baseline": str(baseline_path),
        "batched": str(batched_path),
        "checked": checked,
        "matches": True,
        "seconds": time.perf_counter() - start,
    }


def longcat_codes(sample: Sample) -> torch.Tensor:
    item = sample.get((Role.DEFAULT, Modality.AUDIO))
    if not isinstance(item, AudioItem):
        raise TypeError("LongCat delta sample must contain the default audio item.")
    codes = item.views.get(AudioView.LONGCAT)
    if not isinstance(codes, torch.Tensor):
        raise TypeError("LongCat view must be a Tensor.")
    if codes.ndim != 2:
        raise ValueError("LongCat view must have shape [frame, codebook].")
    if codes.shape[1] == 0:
        raise ValueError("LongCat view must have a non-empty codebook axis.")
    if codes.dtype == torch.bool or codes.is_floating_point() or codes.is_complex():
        raise TypeError("LongCat view must contain integer code ids.")
    return codes


def assert_same_codes(
    baseline: torch.Tensor,
    batched: torch.Tensor,
    *,
    index: int,
) -> None:
    if baseline.dtype != batched.dtype or baseline.shape != batched.shape:
        raise ValueError(
            f"LongCat code metadata differs at sample {index}: "
            f"{baseline.dtype}/{tuple(baseline.shape)} != "
            f"{batched.dtype}/{tuple(batched.shape)}."
        )
    if not torch.equal(baseline, batched):
        raise ValueError(f"LongCat codes differ at sample {index}.")


def merged_views(dataset: MergedDataset) -> list[str]:
    if len(dataset) == 0:
        raise ValueError("FLEURS base store is empty.")
    sample = dataset[0]
    audio = sample.get((Role.DEFAULT, Modality.AUDIO))
    text = sample.get((Role.DEFAULT, Modality.TEXT))
    if not isinstance(audio, AudioItem):
        raise TypeError("Merged sample must contain the default audio item.")
    if not isinstance(text, TextItem) or TextView.TEXT not in text.views:
        raise TypeError("Merged sample must contain the default text view.")
    if AudioView.WAVEFORM not in audio.views or AudioView.LONGCAT not in audio.views:
        raise ValueError("Merged audio must contain waveform and LongCat views.")
    longcat_codes(sample)
    return sorted(view.value for view in audio.views)


def store_result(
    path: Path,
    *,
    count: int,
    materialized: bool,
    seconds: float | None,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "samples": count,
        "materialized": materialized,
        "seconds": seconds,
    }


def env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean environment value.")


def main() -> None:
    root = output_root().resolve()
    os.environ[OUTPUT_ROOT_ENV] = str(root)
    os.environ.setdefault("ANYDATASET_HOME", str(root / "home"))

    base = materialize_base_store()
    baseline = materialize_longcat_delta(delta_store_path(1), batch_size=1)
    batched = materialize_longcat_delta(
        delta_store_path(LONGCAT_BATCH_SIZE),
        batch_size=LONGCAT_BATCH_SIZE,
    )
    comparison = compare_longcat_deltas(
        delta_store_path(1),
        delta_store_path(LONGCAT_BATCH_SIZE),
        limit=COMPARE_LIMIT,
    )

    views = merged_views(merged_dataset())
    result = FilterRule(RULE_NAME, quality_factory).apply(
        dataset_factory=merged_dataset,
        metrics=True,
        device=os.environ.get(QUALITY_DEVICES_ENV, "auto"),
    )
    accepted = result.select_by("accept")

    summary = {
        "root": str(root),
        "base": base,
        "longcat_baseline": baseline,
        "longcat_batched": batched,
        "longcat_comparison": comparison,
        "merged_audio_views": views,
        "filter_cache": str(result.cache_path),
        "metrics": None if result.metrics_path is None else str(result.metrics_path),
        "counts": dict(result.counts),
        "accepted": len(accepted),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
