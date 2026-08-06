# Speech Quality Filter

`anydataset.quality.speech` provides `SpeechQuality` for cached speech
quality partitions. The rule works on canonical `Sample` objects and is
meant to be used with `FilterRule`.

## Boundary

- Dataset loading stays in `Spec`, `Preset`, `AnyDataset`, or `Source.STORE`.
- Cache construction stays in `anydataset.filter`.
- `SpeechQuality` reads every `(role, Modality.AUDIO)` item in a sample.
- Without a codec provider, each checked audio item must expose
  `AudioView.WAVEFORM` and same-role `(role, Modality.TEXT)` with
  `TextView.TEXT`.
- With `codec_provider=CodecProvider(...)`, `SpeechQuality` reads only
  `codec_provider.output`, decodes those frame codes with
  `codec_provider.codec.decode(...)`, and evaluates the reconstructed audio at
  `codec_provider.codec.sample_rate`.
- The codec path never falls back to an original waveform. A missing codec
  view or malformed codec tensor is an input-contract error.
- Missing waveform or same-role text is recorded as an audit warning. It does
  not reject the sample by itself.
- A waveform containing NaN or infinity is rejected before evaluator execution.

## Labels

`SpeechQuality` returns two labels:

- `accept`: no checked audio item failed the configured thresholds.
- `reject`: at least one checked audio item failed a threshold.

The default thresholds are:

- `min_utmos=2.8`
- `max_wer=None`
- `min_chrf=50.0`
- `max_duration_seconds=None`
- `max_seconds_per_text_unit=4.0`
- `min_peak_amplitude=0.05`
- `min_bleu=None`

WER is still recorded for audit, but is not a rejection threshold by default
because ASR output may omit word separators in languages such as Chinese. Enable
WER rejection by setting `SpeechQualityProfile(max_wer=...)`. Enable BLEU
rejection by setting `SpeechQualityProfile(min_bleu=...)`. Set
`max_duration_seconds` when the generated-audio pipeline needs an absolute
utterance-duration ceiling in addition to the text-normalized duration check.

## Metrics

`SpeechQuality` returns `FilterDecision`, so callers should apply the rule with
`metrics=True` when they want audit rows:

```python
from anydataset import FilterRule
from anydataset.quality.speech import SpeechQuality, SpeechQualityProfile

def factory():
    return SpeechQuality(
        profile=SpeechQualityProfile(min_utmos=3.2, min_chrf=55.0),
        decode_options={"language": "en", "temperature": 0.0},
    )

result = FilterRule("speech_quality_v1_en", factory).apply(
    dataset_factory=dataset_factory,
    metrics=True,
)
accepted = result.select_by("accept")
```

To evaluate one materialized codec view, construct the existing
`CodecProvider` in the filter predicate factory and pass it directly to
`SpeechQuality`:

```python
import os

from anydataset.provider import CodecProvider
from anydataset.quality.speech import SpeechQuality
from anydataset.types import AudioView

def codec_speech_factory():
    codec = load_codec(device=os.environ["ANYDATASET_FILTER_DEVICE"])
    provider = CodecProvider(codec, AudioView.LONGCAT)
    return SpeechQuality(codec_provider=provider)
```

When filter `batch_size` is greater than one, codec inputs with equal frame
length are stacked for `codec.decode(...)`. Reconstructed waveforms are then
restored to sample order and passed to the speech evaluator one at a time.
Text and speech predicates can therefore be chained against the same codec
store without materializing reconstructed waveform views.

Each metrics payload includes:

- `decision`: normalized label.
- `flags`: role-prefixed threshold failures such as `default_utmos_low` or
  `default_duration_high`.
- `flags` also records hard input failures such as `default_non_finite_waveform`.
- `warnings`: role-prefixed skipped-input warnings such as
  `source_missing_text`.
- `audio_count`: number of audio items in the sample.
- `checked_count`: number of audio items evaluated by the speech evaluator.
- `items`: per-audio audit rows with reference text, UTMOS, WER, chrF, BLEU,
  duration, peak amplitude, text units, seconds per text unit, and unprefixed
  item flags.

## Evaluator

By default `SpeechQuality` loads `anytrain.evaluator.speech.SpeechEvaluator`.
Pass `evaluator=...` to inject a test double, a preloaded evaluator, or a custom
backend. The evaluator must be callable as:

```python
evaluator(audio, sample_rate, reference_text=reference_text, **decode_options)
```

and must return finite scalar metrics named `utmos`, `wer`, `chrf`, and `bleu`.

`FilterRule.name` is the human-readable name and remains part of cache identity
for compatibility. Put evaluator model, decode options, threshold, parser, and
transform versions in `rule_id`/`version`; changing any identity field selects a
different cache. For codec filtering, include the selected `AudioView`, codec
checkpoint, and decoder configuration in that caller-managed version. The
legacy name-only form remains supported.
