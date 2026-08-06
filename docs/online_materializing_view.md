# Online Materializing View

## Status

This document defines the dynamic transform contract used when a complete
canonical store is not available yet. The mode decision is dataset-wide:

1. a compatible ready store is opened after strict operation-identity validation;
   an explicit `input_id` keeps this fully lazy, while `input_id_factory`
   constructs only the input identity object; or
2. the provider runs against the complete logical dataset universe, returns
   requested values online, and persists complete coverage in background.

This is not a sparse read-through cache. Incomplete staging fragments are never
opened as a dataset, and one returned batch never mixes physical-store rows with
online rows.

Workspace packages own dataset names, chain methods such as `synthesize()` and
`tokenize()`, model lookup, and path policy. Anydataset owns the complete-universe
execution, lineage, staging, and publication contracts.

## Universe, operation, and selection

Dynamic pipelines separate three concepts:

- `DatasetUniverse` is the complete map-style sample space at one pipeline stage.
- A transform or filter operation is materialized over every sample in that
  universe, independent of the caller's current selection.
- `SelectionView` applies an ordered intersection only when samples are returned
  to the caller.

For a one-to-one transform, the model is:

```text
SelectionView(U0, S0)
        |
        | transform every row in U0
        v
       U1
        |
        | rebase S0 from U0 to U1 by stable sample_id
        v
SelectionView(U1, rebase(S0, U1))
```

A later filter also evaluates all of `U1`. Its decisions are appended to the
returned boundary rather than used to reduce the next operation's input:

```text
filter(U1) -> D1 for every row in U1
return SelectionView(U1, rebase(S0, U1), select(D1))
```

Therefore a chain such as `filter -> synthesize -> filter -> tokenize` performs
both transforms and both decision runs over the complete universe of their
respective stages. Only the final returned dataset reflects the ordered
intersection of filter selections.

An unresolved decision is **unknown**, not reject. A live selection waits for
the decisions needed by an indexed access. Operations that require the complete
selected sequence, including `len()`, negative indexing, complete index listing,
and shuffle planning, wait for complete decision coverage.

## Sample lineage

`sample_id` is the stable sample-lineage key. Every one-to-one transform must
preserve it, and selection rebasing validates a one-to-one mapping by `sample_id`.
Missing or duplicate IDs are errors.

`sample_index` is only the dense ordinal inside one universe or one published
store. It is useful for local coverage, manifest lookup, and physical ordering,
but it is not a cross-stage lineage key and must not be used to prove alignment
after a transform or reorder.

Canonical stores persist both fields: `sample_id` carries lineage, while
`sample_index` remains dense in `0..N-1` for that store.

## Global operation identity

One materialized transform has global identities shared by all samples:

- `input_id` identifies the complete input-universe snapshot;
- `provider_id` identifies the AnyTrain backend plus the AnyDataset adapter and
  every semantic model/preprocessing choice that can change output;
- the output contract identifies the generated view and complete returned
  schema.

`provider_id` is store-level provenance. It is unrelated to an individual
sample, filter label, or selection and is never copied into sample metadata.
Model/checkpoint revision, tokenizer configuration, or semantic preprocessing
changes require a new `provider_id`. Batch size, device, process topology,
service endpoint, commit size, and writer concurrency are execution settings and
do not enter it.

Resume metadata fingerprints a dataset/provider factory only when its explicit
semantic ID is absent. Once `input_id` or `provider_id` is present, that ID is
the authoritative factory marker: execution-only fields such as device, cache
path, or endpoint may change without discarding compatible staged fragments.
The caller must version the ID for every semantic input or output change.

Transform identity and physical paths exclude selections. Applying a different
filter selection to the same universe must reuse the same compatible transform
materialization. Selection lineage is rebased onto the output universe only at
the returned dataset boundary.

## Public entrance

`ViewMaterializer.open()` resolves a ready store or creates the dynamic online
dataset through the same configured materializer:

```python
materializer = ViewMaterializer(
    store_root,
    dataset_id=logical_dataset_id,
    staging_dir=staging_dir,
    input_id=input_id,
    provider_id=provider_id,
    keep_schema=keep_schema,
    output=AudioView.LONGCAT,
    schema=output_schema,
)

dataset = materializer.open(
    dataset_factory=source_factory,
    provider_factory=provider_factory,
    device="cuda:0",
)

with dataset:
    sample = dataset[0]
```

If `dataset_factory()` returns a `SelectionView`, the materializer unwraps its
complete universe for provider execution and returns the rebased selection over
the output universe. Callers must not assume that `open()` always returns a bare
`MaterializingViewDataset`.

Resolution is strict:

1. A compatible ready schema-v3 store with explicit `input_id` is opened without
   constructing the source, provider, or staging sink. With `input_id_factory`,
   the input identity object is constructed once and its resolved ID must match
   canonical provenance; the provider and staging sink remain unconstructed. If
   `selection_factory` supplies that object, the opened selection is reused and
   owned by the returned rebased view.
2. If no canonical target exists, the source universe is constructed, its
   identity is resolved, the lifecycle lock is acquired, and online mode starts.
3. A non-empty canonical root without a ready marker is incomplete or corrupt
   and raises. An identity mismatch also raises; neither condition falls back
   online.

The selected mode is fixed for the opened dataset's lifetime.

`dataset_id` is the logical identity of the materialized view. Configure it
explicitly when physical roots may vary so online and ready `universe_id` values
do not depend on `output_dir` basename. Omitting it preserves the basename
fallback for compatibility.

## Online execution and full coverage

Foreground access has priority. Scalar access claims its universe index; batch
access validates, deduplicates, and claims requested universe indexes, invokes
the scalar or batch provider, then restores the caller's original order and
duplicates.

After the first foreground access, a background sweep claims every remaining
uncovered universe index. `wait()` and `close()` also start the sweep, so closing
an online dataset without reading a sample still covers the complete universe.
Sampler subsets, `drop_last`, early stopping, DDP tail truncation, and upstream
filter selections cannot reduce transform coverage.

Compatible staged indexes count as completed coverage and are skipped by the
background sweep. Staging itself is not readable, however, so a foreground
request that needs the value of an already staged index recomputes that value
online. The dataset-wide mode remains online until a canonical store is
published.

Provider results are completed for foreground waiters before persistence is
required to finish. Persistence uses a bounded asynchronous sink, so a slow
writer eventually applies backpressure without turning a completed provider
result into a cache lookup.

Provider, coverage, or writer failures are sticky. A foreground value that was
already completed may still be returned when a later asynchronous write fails;
`wait()`, `close()`, and subsequent operations surface the retained failure.

## Lifecycle

`MaterializingViewDataset` owns the source, provider, coverage coordinator,
staging sink, and materializer lock. It is lifecycle-controlled:

- `coverage_complete` reports whether every universe position is complete;
- `completed_count` reports complete coverage positions;
- `wait()` waits for the full sweep and staging flush;
- `close()` is idempotent, waits for complete coverage, closes the sink and
  resources, and releases the lock;
- the context manager calls `close()`.

`close()` does not publish the canonical store. A successful close guarantees
dense staging coverage for the opened operation. Canonical publication remains
an explicit `write()`/finalize step that validates coverage and atomically
publishes the ready marker.

The online dataset is intentionally single-process today. It rejects pickle and
forked access and therefore requires `DataLoader(num_workers=0)`. A future
multi-process implementation must centralize coverage claims and staging
ownership in one service; remote provider execution alone does not coordinate
writers.

## Staging and canonical publication

Staging fragments are private, immutable, and resumable. They are indexed by the
current universe's dense `sample_index`, but each output record inherits the
source `sample_id`. A coordinated writer persists each index at most once;
overlapping fragments, duplicate lineage IDs, incompatible identity, and corrupt
payloads are hard errors.

Canonical publication requires:

```text
completed sample_index set == {0, 1, ..., len(universe) - 1}
```

Publication also validates the complete schema, sample count, `input_id`,
`provider_id`, output contract, fragment integrity, unique `sample_id` lineage,
and payload references. Only then is the standalone store committed atomically.
An incomplete staging directory is never exposed through the public dataset
entrance.

After `close()`, normal materializer `write()` reuses compatible complete
fragments, computes only genuinely missing coverage, and publishes the canonical
store. A later `open()` then selects ready-store mode and never loads the provider.
An explicit `input_id` also avoids constructing the source; `input_id_factory`
still constructs its lightweight identity object to validate canonical provenance,
without reading source samples.

## Physical formats

The canonical store format remains deliberately hybrid:

- sample and view manifests are Parquet;
- large or heterogeneous payloads remain tar shards with validated manifest
  references and optional offset sidecars;
- online filter decision/selection fragments use Arrow IPC because they are
  compact, append-independent control-plane records;
- completed filter generations remain canonical Parquet partitions.

Arrow IPC decision fragments are not a new canonical payload format. Moving
waveform, token, or other store payloads from tar into Arrow is deferred until a
representative benchmark demonstrates better random access, sequential
throughput, space usage, and distributed-reader behavior. The benchmark gate is
tracked in [`experiments/todo.md`](experiments/todo.md).

## Failure contract

- Missing provider input is an error.
- Provider startup or execution failure is an error.
- A writer or coverage failure is sticky and remains visible at lifecycle
  boundaries.
- A ready-store identity mismatch never falls back online.
- A non-ready non-empty canonical root never falls back online.
- Selection identity never changes transform identity or physical output paths.
- Failure to preserve one-to-one `sample_id` lineage prevents selection rebasing
  and publication.
- A second online owner or concurrent finalize attempt is rejected while the
  lifecycle lock is held.

## Validation checklist

- Provider calls cover the complete universe even when the returned selection is
  small or no sample is read before `close()`.
- A filtered source and its unfiltered universe resolve the same transform
  identity when `input_id`, provider, and output contract match.
- One-to-one transforms preserve `sample_id`; output `sample_index` is dense.
- Scalar and batch access preserve requested order and duplicate results.
- Unknown filter decisions never behave as reject.
- `close()` waits for complete coverage and staging persistence but does not
  publish a ready store.
- Ready-store mode with explicit `input_id` constructs no source dataset,
  provider, or staging writer. With `input_id_factory`, it constructs only the
  input identity object and still constructs no provider or staging writer.
- Canonical manifests remain Parquet, payloads remain tar, and only online
  decision fragments use Arrow IPC.
