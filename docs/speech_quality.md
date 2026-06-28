# Speech Quality Filter

`anydataset.quality.speech` provides a reusable predicate for cached speech
quality partitions. The predicate works on canonical `Sample` objects and is
meant to be used with `FilterRule`.

## Boundary

- Dataset loading stays in `Spec`, `Preset`, `AnyDataset`, or `Source.STORE`.
- Cache construction stays in `anydataset.filter`.
- The predicate reads every `(role, Modality.AUDIO)` item in a sample.
- Each checked audio item must expose `AudioView.WAVEFORM` and same-role
  `(role, Modality.TEXT)` with `TextView.TEXT`.
- Missing waveform or same-role text is recorded as an audit warning. It does
  not reject the sample by itself.

## Labels

The predicate returns two labels:

- `accept`: no checked audio item failed the configured thresholds.
- `reject`: at least one checked audio item failed a threshold.

The default thresholds are:

- `min_utmos=3.0`
- `max_wer=None`
- `min_chrf=50.0`
- `max_seconds_per_text_unit=4.0`
- `min_peak_amplitude=0.05`
- `min_bleu=None`

WER is still recorded for audit, but is not a rejection threshold by default
because ASR output may omit word separators in languages such as Chinese. Enable
WER rejection by setting `Profile(max_wer=...)`. Enable BLEU rejection by
setting `Profile(min_bleu=...)`.

## Metrics

The predicate returns `FilterDecision`, so callers should apply the rule with
`metrics=True` when they want audit rows:

```python
from anydataset import FilterRule
from anydataset.quality.speech import Predicate, Profile

def factory():
    return Predicate(
        profile=Profile(min_utmos=3.2, min_chrf=55.0),
        decode_options={"language": "en", "temperature": 0.0},
    )

result = FilterRule("speech_quality_v1_en", factory).apply(
    dataset,
    metrics=True,
)
accepted = result.select("accept")
```

Each metrics payload includes:

- `decision`: normalized label.
- `flags`: role-prefixed threshold failures such as `default_utmos_low`.
- `warnings`: role-prefixed skipped-input warnings such as
  `source_missing_text`.
- `audio_count`: number of audio items in the sample.
- `checked_count`: number of audio items evaluated by the speech evaluator.
- `items`: per-audio audit rows with reference text, UTMOS, WER, chrF, BLEU,
  duration, peak amplitude, text units, seconds per text unit, and unprefixed
  item flags.

## Evaluator

By default the predicate loads `anytrain.evaluator.speech.SpeechEvaluator`.
Pass `evaluator=...` to inject a test double, a preloaded evaluator, or a custom
backend. The evaluator must be callable as:

```python
evaluator(audio, sample_rate, reference_text=reference_text, **decode_options)
```

and must return finite scalar metrics named `utmos`, `wer`, `chrf`, and `bleu`.

`FilterRule.name` remains the cache contract. Include any evaluator model,
decode options, threshold, parser, and transform versions in the rule name when
cache reuse should change.
