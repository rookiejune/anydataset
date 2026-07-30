# Online Reject Replace

`RejectReplaceDataset` is a map-style wrapper that skips rejected primary
samples at read time and returns a nearby accept, or an accept from a
worker-local buffer. It is a **low reject-rate CPU safety net**, not a
replacement for cached `FilteredDataset` partitions.

```python
from anydataset.filter import RejectReplaceDataset

def cheap_cpu_rule(sample):
    return is_finite_waveform(sample) and duration_ok(sample)

train = RejectReplaceDataset(
    accepted_store,  # usually FilteredDataset.select_by("accept")
    cheap_cpu_rule,
    name="tts_safety_v1",
)
```

## Boundary

- Offline quality rules (GPU ASR, UTMOS, translation models, full-corpus
  scans) stay on `FilterRule` / `FilteredDataset`.
- The online predicate must be cheap CPU logic over an already materialized
  canonical `Sample`. Do not attach CUDA models, remote providers, or
  `RemoteFilterFactory`.
- The wrapper does **not** enter filter cache identity and does not change
  `Spec` or store provenance.
- `__len__` equals the wrapped dataset; rejects are replaced, not dropped.
- Prefer offline `select_by("accept")` first, then wrap with online rules for
  rare hard failures (empty audio, non-finite waveform, extreme duration).

## Replacement order

1. If the primary index accepts, optionally update the accept buffer and
   return it.
2. Otherwise probe `index+1, index+2, ...` (modulo length) up to `max_probe`
   sequential neighbors.
3. If probes fail and the buffer has at least `min_buffer` accepts, return a
   random buffer sample.
4. Otherwise raise. Do not return a rejected sample silently.

Accepted samples fill the buffer until `buffer_size`. When full, a new accept
replaces a random slot with probability `update_prob`. Each DataLoader worker
keeps its own buffer; there is no cross-process shared queue.

## Warnings and hard failure

Primary-index accept/reject decisions are tracked in a sliding
`stats_window` (default 200):

| Signal | Default | Behavior |
| --- | --- | --- |
| `warn_reject_ratio` | `0.05` | Log a warning (with cooldown) |
| `max_reject_ratio` | `0.20` | Raise `RuntimeError` |
| cold start / `max_probe` miss | — | Raise when buffer is still below `min_buffer` |

Frequent online rejects mean the rule belongs in `FilteredDataset`, not a
larger buffer.

## Defaults

```text
buffer_size=64
min_buffer=8
update_prob=0.25
max_probe=32
warn_reject_ratio=0.05
max_reject_ratio=0.20
stats_window=200
warn_cooldown_s=30.0
```

## Predicate labels

Return values reuse the filter label contract:

- `True` / `"accept"` keep the sample
- `False`, `"reject"`, `"review"`, or any other non-`accept` label triggers
  replacement
- `FilterDecision` is supported; online mode does not persist metrics
