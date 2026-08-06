# Filter Decisions and Cached Partitions

A filter materializes one decision for every sample in a complete
`DatasetUniverse`. It does not copy payloads, mutate physical `Spec` identity, or
shrink the input of a later transform. Selected labels become a `SelectionView`
only at the returned dataset boundary.

The dynamic entrance is `FilterRule.open()`. It returns a `FilterRun` whose
`dataset` is usable while the full-universe predicate scan continues in the
background:

```python
from anydataset.filter import FilterDecision, FilterRule, FilterRunStatus

def quality_factory():
    return lambda sample: "review" if needs_review(sample) else is_good(sample)


def dataset_factory():
    return build_dataset()


rule = FilterRule(
    "quality_v1",
    quality_factory,
    version="parse-v3",
)
run = rule.open(
    dataset_factory=dataset_factory,
    labels="accept",
    device="cpu",
)
train = run.dataset
first = train[0]  # waits only for decisions needed to resolve this position
train = run.wait()  # waits for complete decision coverage
assert run.status is FilterRunStatus.COMPLETE
run.close()
```

The rule factory is called inside the process that executes the predicate. The
predicate receives the full canonical `Sample` produced by the current stage's
complete universe. A cold `open()` starts the scan immediately. A ready
compatible cache returns a complete run without constructing the predicate.

`FilterRun` exposes:

- `dataset`: the live ordered intersection of upstream selections and the new
  selected labels;
- `status`: `RUNNING`, `COMPLETE`, or `FAILED`;
- `wait()`: wait for complete decision coverage and return `dataset`;
- `close()`: wait, release the generation lease, and re-raise any sticky error;
- context-manager lifecycle equivalent to `close()`.

Unknown decisions are unresolved state, not reject. Positive indexed access
waits only for the decisions needed to identify that returned position. `len()`,
negative indexing, complete selected-index listing, and shuffle planning require
the complete ordered selection and therefore wait for the full scan.

The existing blocking API remains available for compatibility:

```python
filtered = rule.apply(
    dataset_factory=dataset_factory,
    labels="accept",
    device="cpu",
)
```

`FilterRule.apply()` blocks until the canonical Parquet generation is ready and
returns the historical `FilteredDataset` surface, including metrics and apply
reports. New dynamic pipelines should use `open()`, because `FilterRun` makes the
complete-universe operation and the returned selection separate objects.

## Full-universe and selection semantics

If `dataset_factory()` returns a `SelectionView`, `open()` unwraps its
`DatasetUniverse`, evaluates the predicate on every universe sample, then appends
the new decision selection to the existing ordered intersection. Upstream labels
or generations do not enter predicate coverage or operation identity.

For example, two callers can select disjoint subsets of the same universe and
reuse one ready decision cache for the same rule. Each caller gets a different
returned intersection without rerunning the predicate. A later synthesize or
tokenize operation also consumes the complete universe rather than the selected
rows.

Across a one-to-one transform, decisions are rebased with stable `sample_id`.
The current universe's `sample_index` is only a dense ordinal and is not used as
a cross-stage lineage key. Duplicate or missing `sample_id` values are errors.

Filtered views preserve the universe's lightweight cost and shuffle contracts.
Callable dataloader costs receive the underlying `cost_row(...)` metadata for
the selected universe index instead of materializing the full sample. Shuffle
keeps physical index groups, maps selected rows into returned positions, and
distributes that selected stream across DDP ranks.

Predicate return values are normalized to string labels:

- `True` becomes `"accept"`.
- `False` becomes `"reject"`.
- `str` values are used directly.
- `Enum` values use their string value, or the enum name when the value is not a
  string.
- `FilterDecision` carries a label plus optional per-sample metrics.

`FilterRule.name` is the human-readable rule name and remains part of the cache
identity. Optional `rule_id`, `version`, and `content_id` fields add explicit
semantic identity; changing any of them selects a different decision cache.
When omitted, `rule_id` defaults to `name` and the original name-only path remains
compatible. The library does not infer semantic changes in a model, predicate,
parse function, or preprocessing recipe, so callers must update `version` or
`content_id` when those outputs can change.

The other identity input is the complete universe snapshot. Selection labels,
upstream filter generations, returned sample count, and selection order are
excluded. For application-owned logical universes, a non-empty `input_id` is
required and must change when universe content, order, or sample lineage changes.
`rebuild=True` forces a new generation under the same declared identity.

Cache layout:

```text
$ANYDATASET_HOME/
  cache/
    filters/
      <dataset_hash>/
        <rule_hash>/
          current.json
          generations/
            <generation_id>/
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
              .lease
              .ready
        .<rule_hash>.resume/
          filter/
            decisions/
              <chunk-digest>.arrow
    sources/
      <spec_id>/
        metadata.json
        .ready
```

`dataset_hash` identifies the complete universe. For a physical dataset it is
derived from the dataset class, physical `Spec`, sample count, and store
provenance. A `SelectionView` contributes its universe, not its selected labels.
Consequently a materializer input or provider version change selects a new
filter cache through universe provenance, while a different selection does not.

For a mutable or application-owned input, pass a non-empty `input_id`:

```python
run = rule.open(
    dataset_factory=dataset_factory,
    input_id="base-plus-local-annotations-v3",
)
run.close()
```

`input_id` is the caller-managed semantic version of the complete universe
snapshot. It supplements automatic class, physical `Spec`, store provenance,
sample-count, and lineage identity; it does not replace them. Change it when
universe content, ordering, or `sample_id` membership changes. The same ID may be
used with different selections over that unchanged universe.

`rule.json` stores the complete-universe identity, scanned sample count, and rule
identity. When those values do not match, the rule is recomputed.
`partitions.json` stores labels, counts, and Parquet shard names. Each partition
row stores an index in the current universe's coordinate system; it is not a
cross-transform lineage key. Selection rebasing uses `sample_id` instead.
`FilterRule.open/apply(..., max_shard_samples=...)`
controls the maximum number of indices written to one shard; the default is
1,000,000. `FilterRule.open/apply(..., commit_samples=...)` controls how many
samples are scanned before one in-memory label batch is committed to the shard
writer; the default is 100,000. Cache construction writes those bounded batches
incrementally, so it does not need to hold every accepted index in one Python
object before writing. Cache construction keeps completed chunks in a hidden
resume directory and replays them into the final cache after all samples are
covered. `write_workers` controls background fragment writer threads; the
default is one writer so predicate execution can overlap with parquet writes.
`write_prefetch` bounds pending write jobs.

During a live `FilterRun`, every completed decision chunk is also written as an
immutable Arrow IPC file before it becomes visible to the live selection. These
Arrow files contain only compact `(index, label)` control-plane rows. They are
not canonical sample payloads and do not replace the final Parquet partition and
metrics generation. A write or scan failure is sticky; completed Arrow chunks
remain available for diagnosis. Canonical resume and publication still use the
versioned filter-resume metadata and Parquet generation contract.

Each completed cache is published as a new immutable generation. The writer
first atomically renames the completed generation into `generations/`, then
atomically replaces `current.json`. Readers resolve that pointer once and expose
the resolved generation as `FilteredDataset.cache_path`; metrics and partition
reads therefore stay on the same snapshot even when another call publishes a
new generation.

Every live filtered cache and exported `dataset_factory` holds a shared OS file
lock on the generation's `.lease` file. Publication and
`cleanup_filter_generations(filtered.cache_path)` take the rule writer lock and
only delete non-current generations whose lease can be locked exclusively.
Process exit releases a lease in the kernel, so cleanup does not depend on a PID
file, heartbeat, or timeout. Cleanup also runs after a successful publication;
leased generations are retained and can be collected by a later publication or
an explicit cleanup call.

```python
from anydataset.filter import cleanup_filter_generations

removed = cleanup_filter_generations(filtered.cache_path)
```

A serialized pickle is a reference to its exact generation, not a persistent
lease. A live object or exported factory pins the generation, including while it
is being pickled or sent to a worker, but bytes stored for later restoration do
not. Callers that retain dormant pickles must defer explicit cleanup. Direct
mutation of generation files is outside the cache contract; shard fingerprints
still detect such mutation, and actual shard row counts are checked against the
manifest.

Single-label selections retain their shard-lazy file index. Multiple selected
labels use a lazy sorted merge: constructing the view and reading its length do
not load partition shards, while indexed access incrementally loads and memoizes
only the merge prefix needed so far.

Caches written by the earlier same-directory layout have no `current.json` and
are treated as cache misses. They are not silently opened as generations;
generation cleanup only manages children of `generations/`.

Cache construction reports scan and fragment-writer progress on stdout. In
multi-device runs, each worker also writes lifecycle and failure details under
`$ANYDATASET_HOME/logs/<timestamp>-<pid>/filter/part-xxxxx.log`; rank 0 mirrors
ordinary lifecycle logs to stdout and errors to stderr.

`FilterRule.open/apply(..., device="auto")` resolves all visible CUDA devices, or CPU
when CUDA is unavailable. One resolved device runs in the calling process;
multiple devices start one worker per device using
`Runtime.process_start_method`, which defaults to `"spawn"`. Pass `device="cpu"`
to force CPU execution. Pass an iterable such as `("cpu", "cpu")` or
`("cuda:0", "cuda:1")` to explicitly parallelize cache construction across
map-style index ranges.
After every ordered row and worker completion marker has been received, the
caller gives completed workers a bounded grace period to exit. A worker that
remains in framework or accelerator finalization is terminated without
invalidating the already completed filter result; this fallback is recorded in
the run log.
Pass `num_workers` to let each execution process read samples through a PyTorch
`DataLoader`; `batch_size` controls that loader's sample batch size.
`dataset_factory` is the only dataset entry point. This keeps single-device,
DataLoader-worker, multi-device, and chained filtering on the same contract.

A predicate may implement `call_batch(samples)` to process one loader batch at
once. It must return an ordered sequence containing exactly one filter output
for each input sample; outputs are matched to inputs by position. Predicates
without `call_batch` keep the per-sample `__call__` path. Resume indexes are
removed before either path runs, so a batched predicate never receives rows
that are already committed.

With the default `"spawn"` start method, multi-device workers must receive
picklable factories instead of an already constructed dataset instance:

```python
def dataset_factory():
    return build_dataset()


run = rule.open(
    dataset_factory=dataset_factory,
    device=("cuda:0", "cuda:1"),
)
run.close()
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

Metrics rows include the current universe `index`, normalized `label`, and the
user metrics payload. That index is a dense/local decision coordinate, not the
stable sample-lineage identifier. Metrics payloads must be JSON-serializable
mappings with string keys; NaN and infinity are rejected. The evaluator logic
that computes those metrics stays in user code or a higher-level evaluator
package.

Metrics are stored as parquet shards with fixed columns:

- `index`: global sample-space index.
- `label`: normalized filter label.
- `metrics`: canonical JSON text.

This three-column schema is the public cache contract. The JSON payload remains
the extension point; metric-specific promoted columns would couple cache schema
versions to application rules. `iter_metrics()` is also the export boundary for
databases, object stores, or custom analytics. Filter construction does not call
pluggable external sinks because retries and resume replay would otherwise need
an exactly-once side-effect protocol. Add a versioned cache schema only when a
concrete query workload cannot be served by this representation.

`metrics=True` is part of the cache readiness check. If a partition cache exists
without `metrics/metrics.json`, the rule is rebuilt so
`FilteredDataset.metrics_path` and `FilteredDataset.iter_metrics()` are valid.

`FilteredDataset` exposes `labels` and `counts` for the current selection, and
`available_labels` and `available_counts` for every label in the cache. It also
exposes `cache_path`, preserves map-style indexing, and provides
`iter_shard(num_shards, shard_id)` as `(sample_index, sample)` pairs over the
selected global-index order.

Runtime observability is explicit rather than stored on the dataset object:

```python
applied = rule.apply_with_report(dataset_factory=dataset_factory, device="cpu")
filtered = applied.dataset
report = applied.report
```

`FilterApplyReport` contains wall-clock elapsed seconds, per-phase timings
(`dataset_seconds`, `cache_lookup_seconds`, `cache_build_seconds`, and
`partition_read_seconds`), sample count, cache path, and whether the call reused
a ready cache. `logs_dir` is populated only for calls that actually build or
rebuild a cache; hot cache hits leave it as `None`. Its `samples_per_second`
property is apply-call throughput, so cache hits measure cache lookup and
partition loading rather than predicate speed.

## Online safety net

For rare CPU-only rejects after an accept partition is already selected, wrap
the map-style dataset with `RejectReplaceDataset`. It does not write filter
cache, does not support GPU predicates, and hard-fails when reject rates are
high. See [`online_filter.md`](online_filter.md).
