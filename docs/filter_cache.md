# Cached Filter Partitions

`FilteredDataset(dataset, rule, labels=...)` applies a named rule to a
map-style `AnyDataset` and caches original dataset indices under the dataset's
physical cache path. A rule does not copy sample payloads and does not change
`Spec` identity.
The public filter API lives under `anydataset.filter` and is also re-exported
from `anydataset`.

```python
from anydataset.filter import FilterDecision, FilterRule, FilteredDataset

rule = FilterRule(
    name="quality_v1_parse_v3_transform_none",
    predicate=lambda sample: "review" if needs_review(sample) else is_good(sample),
)

train = FilteredDataset(dataset, rule, labels="accept", num_workers=4)
audit = FilteredDataset(dataset, rule, labels=("reject", "review"))
```

The predicate receives the full canonical `Sample` produced by the dataset.
`FilteredDataset` checks whether the named rule has a ready cache for the base
dataset; if not, it builds one before exposing the requested labels. Callers are
responsible for passing the labels they want to read.

Predicate return values are normalized to string labels:

- `True` becomes `"accept"`.
- `False` becomes `"reject"`.
- `str` values are used directly.
- `Enum` values use their string value, or the enum name when the value is not a
  string.
- `FilterDecision` carries a label plus optional per-sample metrics.

`FilterRule` uses `name` as its cache contract. The callable `predicate`,
dataset `parse_fn`, and dataset transforms are deliberately not inspected by
the library. Callers should include those semantic versions in `name` when
cache reuse must change.

Cache layout:

```text
cache_path/
  filters/
    <rule_hash>/
      rule.json
      partitions.json
      partitions/
        <label_hash>/
          part-000000.parquet
          part-000001.parquet
      metrics/
        metrics.json
        shards/
          part-000000.parquet
      .ready
```

`rule.json` stores the base `Spec` id, base sample count, and rule name. When
those values do not match, the rule is recomputed. `partitions.json` stores
labels, counts, and shard parquet file names. Each parquet file stores original
dataset indices for one label shard. `FilterRule.apply(..., max_shard_samples=...)`
controls the maximum number of indices written to one shard; the default is
1,000,000. `FilterRule.apply(..., commit_samples=...)` controls how many
samples are scanned before one in-memory label batch is committed to the shard
writer; the default is 100,000. Cache construction writes those bounded batches
incrementally, so it does not need to hold every accepted index in one Python
object before writing.

`FilterRule.apply(..., num_workers=...)` parallelizes cache construction across
map-style index ranges. Keep the default `num_workers=1` when the dataset or
predicate is not safe to use from worker processes.

## Metrics Side Output

When the predicate should produce audit scores or other lightweight diagnostics,
return `FilterDecision` and pass `metrics=True`:

```python
rule = FilterRule(
    name="quality_v2",
    predicate=lambda sample: FilterDecision(
        label=is_good(sample),
        metrics={"score": quality_score(sample)},
    ),
)

result = rule.apply(dataset, metrics=True)
for row in result.iter_metrics():
    ...
```

Metrics rows include the original dataset `index`, normalized `label`, and the
user metrics payload. Metrics payloads must be JSON-serializable mappings with
string keys; NaN and infinity are rejected. The evaluator logic that computes
those metrics stays in user code or a higher-level evaluator package.

Metrics are stored as parquet shards with fixed columns:

- `index`: original dataset index.
- `label`: normalized filter label.
- `metrics`: canonical JSON text.

`metrics=True` is part of the cache readiness check. If a partition cache exists
without `metrics/metrics.json`, the rule is rebuilt so `FilterResult.metrics_path`
and `FilterResult.iter_metrics()` are valid.

`FilterRule.apply(...)` returns a `FilterResult` for callers that need partition
counts, cache metadata, or metrics. `FilterResult.select(...)` is a convenience
wrapper for selecting labels from that result. `FilteredDataset` preserves
map-style indexing and exposes `iter_shard(num_shards, shard_id)` over the
remapped order.
