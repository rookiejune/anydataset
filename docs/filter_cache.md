# Cached Filter Partitions

`FilteredDataset(name, factory, dataset_factory=..., ...)` applies a named rule
to a map-style sample dataset and caches global row indices in that dataset's
sample space. A rule does not copy sample payloads and does not change physical
`Spec` identity. The public filter API lives under `anydataset.filter` and is
also re-exported from `anydataset`.

```python
from anydataset.filter import FilterDecision, FilteredDataset, FilterRule

def quality_factory():
    return lambda sample: "review" if needs_review(sample) else is_good(sample)


def dataset_factory():
    return build_dataset()


filtered = FilteredDataset(
    "quality_v1_parse_v3_transform_none",
    quality_factory,
    dataset_factory=dataset_factory,
    device="cpu",
)
train = filtered.select_by("accept")
audit = filtered.select_by("reject", "review")

rule = FilterRule("quality_v1_parse_v3_transform_none", quality_factory)
again = rule.apply(dataset_factory=dataset_factory, labels="accept", device="cpu")
```

The rule factory is called inside the process that executes the predicate. The
predicate receives the full canonical `Sample` produced by the dataset.
`FilteredDataset(...)` checks whether the named rule has a ready cache for the
base dataset; if not, it builds one. It selects all available labels by default.
`FilteredDataset.select_by(...)` creates a label view over the same cache
without rerunning the predicate. `FilterRule.apply(...)` is a convenience
wrapper that forwards its `name` and `factory` to `FilteredDataset`.

`FilteredDataset` is itself a map-style sample dataset, so filters can be
chained:

```python
clean_text = text_rule.apply(
    dataset_factory=dataset_factory,
).select_by("clean", "usable")
clean_both = speech_rule.apply(
    dataset_factory=clean_text.dataset_factory,
).select_by("accept")
```

When a rule is applied to a filtered view, the predicate only scans the selected
rows, but the written partition indices still refer to the original sample
space. Cache metadata records the upstream rule and selected labels, and the
downstream cache key is separated from the same rule applied to the physical
dataset.

Predicate return values are normalized to string labels:

- `True` becomes `"accept"`.
- `False` becomes `"reject"`.
- `str` values are used directly.
- `Enum` values use their string value, or the enum name when the value is not a
  string.
- `FilterDecision` carries a label plus optional per-sample metrics.

`FilterRule` uses `name` as its cache contract. The factory, predicate, dataset
`parse_fn`, and dataset transforms are deliberately not inspected by the
library. Callers should include those semantic versions in `name` when cache
reuse must change.

Cache layout:

```text
$ANYDATASET_HOME/
  cache/
    filters/
      <dataset_hash>/
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
    sources/
      <spec_id>/
        metadata.json
        .ready
```

`dataset_hash` is derived from the dataset class and physical `Spec` id for a
single physical dataset. For a merged map-style dataset it is derived from the
sorted child identities, so merge input order does not split the cache. A
`MultipleAnyDataset` is not a filter cache identity; filter or cache the child
datasets independently before combining them.

`rule.json` stores the base physical `Spec` id or filtered-view lineage, scanned
sample count, and rule name. When those values do not match, the rule is
recomputed. `partitions.json` stores labels, counts, and shard parquet file
names. Each parquet file stores global sample-space indices for one label shard.
`FilterRule.apply(..., max_shard_samples=...)`
controls the maximum number of indices written to one shard; the default is
1,000,000. `FilterRule.apply(..., commit_samples=...)` controls how many
samples are scanned before one in-memory label batch is committed to the shard
writer; the default is 100,000. Cache construction writes those bounded batches
incrementally, so it does not need to hold every accepted index in one Python
object before writing. Cache construction keeps completed chunks in a hidden
resume directory and replays them into the final cache after all samples are
covered. `write_workers` controls background fragment writer processes; the
default is one writer so predicate execution can overlap with parquet writes.
`write_prefetch` bounds pending write jobs.

`FilterRule.apply(..., device="auto")` uses one spawned process per visible CUDA
device, or CPU single-process execution when CUDA is unavailable. Pass
`device="cpu"` to force single-process CPU execution. Pass an iterable such as
`("cpu", "cpu")` or `("cuda:0", "cuda:1")` to explicitly parallelize cache
construction across map-style index ranges.
Pass `num_workers` to let each device process read samples through a PyTorch
`DataLoader`; `batch_size` controls that loader's sample batch size. This gives
one process per device, and optional DataLoader workers inside each process.
`dataset_factory` is the only dataset entry point. This keeps single-device,
DataLoader-worker, multi-device, and chained filtering on the same contract.

Multi-device filtering uses Python `spawn`, so workers must receive factories
instead of an already constructed dataset instance:

```python
def dataset_factory():
    return build_dataset()


filtered = rule.apply(
    dataset_factory=dataset_factory,
    device=("cuda:0", "cuda:1"),
)
```

Both `dataset_factory` and the predicate factory stored in `FilterRule` should
be module-level picklable callables. Multi-device filtering sets DDP-style
`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`
environment variables before calling the factories.

## Metrics Side Output

When the predicate should produce audit scores or other lightweight diagnostics,
return `FilterDecision` and pass `metrics=True`:

```python
def metric_factory():
    return lambda sample: FilterDecision(
        label=is_good(sample),
        metrics={"score": quality_score(sample)},
    )


rule = FilterRule("quality_v2", metric_factory)

filtered = rule.apply(dataset_factory=dataset_factory, metrics=True, device="cpu")
for row in filtered.iter_metrics():
    ...
```

Metrics rows include the global sample-space `index`, normalized `label`, and the
user metrics payload. Metrics payloads must be JSON-serializable mappings with
string keys; NaN and infinity are rejected. The evaluator logic that computes
those metrics stays in user code or a higher-level evaluator package.

Metrics are stored as parquet shards with fixed columns:

- `index`: global sample-space index.
- `label`: normalized filter label.
- `metrics`: canonical JSON text.

`metrics=True` is part of the cache readiness check. If a partition cache exists
without `metrics/metrics.json`, the rule is rebuilt so
`FilteredDataset.metrics_path` and `FilteredDataset.iter_metrics()` are valid.

`FilteredDataset` exposes `labels` and `counts` for the current selection, and
`available_labels` and `available_counts` for every label in the cache. It also
exposes `cache_path`, preserves map-style indexing, and provides
`iter_shard(num_shards, shard_id)` over the selected global-index order.
