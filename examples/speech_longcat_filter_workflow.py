"""Recipe for materializing LongCat views and filtering a speech dataset.

This example is intentionally written as a readable workflow instead of a
configurable command-line tool. It assumes a GPU environment with local
LongCat weights and anytrain evaluators available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from anydataset.provider.longcat import LongCatViewProvider
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
FILTER_CACHE = ROOT / "filter-cache"
DATASET_CACHE = ROOT / "dataset-cache"

DEVICE = "cuda:0"
MIN_UTMOS = 3.0
MAX_WER = 0.4
MIN_CHRF = 50.0

# Evaluators used by the filter predicate.
asr = WhisperASREvaluator(
    model_name="large-v3",
    device=DEVICE,
    decode_options={"language": "en", "temperature": 0.0},
)
text = TextComparisonEvaluator()
utmos = UTMOSEvaluator(device=DEVICE, backend_load_options={"trust_repo": True})


def quality(sample: Sample) -> FilterDecision:
    audio = cast(AudioItem, sample[Role.DEFAULT, Modality.AUDIO])
    transcript = cast(TextItem, sample[Role.DEFAULT, Modality.TEXT])
    waveform, sample_rate = audio.views[AudioView.WAVEFORM]
    reference = transcript.views[TextView.TEXT]
    prediction = asr.transcribe(waveform, sample_rate)
    metrics = text.evaluate(prediction, reference)
    score = float(utmos.evaluate(waveform, sample_rate)["utmos"])
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


rule = FilterRule("reject_utmos_lt_3_and_bad_asr_text_v1", quality)


def fleurs_dataset():
    return Preset.FLEURS.create(split=SPLIT, config_name=CONFIG, streaming=True)


def longcat_provider(device: str):
    return LongCatViewProvider(
        decoders=("16k_4codebooks",),
        device=device,
        local_files_only=True,
    )


# Materialize LongCat codes into a delta store once, then reuse it.
if not DELTA.exists():
    ViewMaterializer(DELTA, split=SPLIT).write(
        dataset_factory=fleurs_dataset,
        provider_factory=longcat_provider,
        devices="auto",
    )

# Merge the original FLEURS waveform/text views with the generated LongCat view.
merged = AnyDataset(f"store://{DELTA}:{SPLIT}", cache_root=DATASET_CACHE)
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
result = rule.apply(merged, metrics=True, cache_root=FILTER_CACHE)
filtered = result.select("accept")
sample = filtered[0]
audio = cast(AudioItem, sample[Role.DEFAULT, Modality.AUDIO])

summary = {
    "delta": str(DELTA),
    "filter_cache": str(FILTER_CACHE),
    "metrics": None if result.metrics_path is None else str(result.metrics_path),
    "accepted": len(filtered),
    "labels": result.labels,
    "first_views": sorted(view.value for view in audio.views),
}
print(json.dumps(summary, indent=2, sort_keys=True))
