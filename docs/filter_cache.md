# Cached Filter Views

`cached_filter(dataset, rule)` builds a map-style filtered view over an
`AnyDataset`. It keeps only remap indices under the dataset's physical cache
path and does not change `Spec` identity.

```python
from anydataset import FilterRule, cached_filter

rule = FilterRule(
    name="duration_range",
    version="1",
    config=(("min_seconds", 1.0), ("max_seconds", 10.0)),
    predicate=lambda sample: 1.0 <= duration(sample) <= 10.0,
)

filtered = cached_filter(dataset, rule)
```

The rule cache key is derived from `name`, `version`, and JSON-serializable
`config`. The callable `predicate` is deliberately not part of the cache key, so
callers should bump `version` when predicate semantics change.

Cache layout:

```text
cache_path/
  filters/
    <rule_hash>/
      rule.json
      indices.parquet
      .ready
```

`rule.json` stores the base `Spec` id, base sample count, and rule identity.
When those values do not match, the filter is recomputed. `indices.parquet`
stores original dataset indices, so the view is cheap to create and does not
copy sample payloads.

`FilteredDataset` preserves map-style indexing and exposes
`iter_shard(num_shards, shard_id)` over the remapped order.
