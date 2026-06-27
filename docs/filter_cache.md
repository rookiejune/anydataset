# Cached Filter Partitions

`FilterRule.apply(dataset)` applies a named rule to a map-style `AnyDataset`
and caches original dataset indices under the dataset's physical cache path. A
rule does not copy sample payloads and does not change `Spec` identity.

```python
from anydataset import AudioMeta, AudioReq, Modality, Role, FilterRule

schema = {
    (Role.DEFAULT, Modality.AUDIO): AudioReq(
        meta=frozenset({AudioMeta.LABEL}),
    )
}

rule = FilterRule(
    name="quality_v1_parse_v3_transform_none",
    schema=schema,
    predicate=lambda sample: "review" if needs_review(sample) else is_good(sample),
)

result = rule.apply(dataset, num_workers=4)
train = result.select("accept")
audit = result.select("reject", "review")
```

The predicate receives only the `Sample` subset selected by `schema`.

Predicate return values are normalized to string labels:

- `True` becomes `"accept"`.
- `False` becomes `"reject"`.
- `str` values are used directly.
- `Enum` values use their string value, or the enum name when the value is not a
  string.

`FilterRule` identity is derived from `name` and `schema`. The callable
`predicate`, dataset `parse_fn`, and dataset transforms are deliberately not
inspected by the library. Callers should include those semantic versions in
`name` when cache reuse must change.

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
      .ready
```

`rule.json` stores the base `Spec` id, base sample count, and rule identity.
When those values do not match, the rule is recomputed. `partitions.json` stores
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

`FilterResult.select(...)` merges one or more labels into a `FilteredDataset`.
`FilteredDataset` preserves map-style indexing and exposes
`iter_shard(num_shards, shard_id)` over the remapped order.
