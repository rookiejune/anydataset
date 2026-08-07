# Online Materializing View

## Status

An online materialized view has two explicit roles:

- `ViewMaterializer.open()` is a read-only consumer entrance.
- `ViewMaterializer.produce()` is the only producer entrance.

The physical output root is their coordination address. Entrances configured
with the same root and compatible logical identity observe the same published
object. They do not coordinate through a provider instance, process-local
state, device policy, or a caller-visible update check.

A consumer opens the catalog once and receives the logical prefix published at
that instant. Its length remains fixed for the dataset lifetime. Training code
reopens the entrance at each epoch boundary and therefore sees the newest
prefix without first asking whether the root changed.

The catalog plus its immutable snapshot stores is the canonical dataset format.
No compact or merge step is required after full coverage; readers keep routing
the fixed prefix across its snapshot segments.

Snapshot segmentation is a dataset property, not a filter feature. Other
one-to-one operations may consume the same fixed segment sequence, while a
non-empty plain map-style dataset is adapted as one logical snapshot. An empty
dataset is already the complete empty prefix.

## Logical model

`catalog.json` is an atomic append-only manifest. Each entry names one
immutable canonical store below `snapshots/` and records its global
`[start, stop)` range. A segment contains only that delta and uses local dense
indexes `0..stop-start-1`; the consumer concatenates all catalog entries into
the logical dataset `0..latest`.

```text
physical root
├── catalog.json
└── snapshots/
    ├── 00000000-000000000000-000000050000/  # [0, 50k)
    ├── 00000001-000000050000-000000100000/  # [50k, 100k)
    └── 00000002-000000100000-000000127381/  # [100k, 127381)

consumer opened after revision 1 -> fixed logical dataset [0, 100k)
consumer opened after revision 2 -> fixed logical dataset [0, 127381)
```

An absent root or catalog is the valid empty prefix. Opening it does not create
directories, a catalog, a provider, a writer, or a lock. A legacy compatible
ready store remains readable as one sealed store.

## Public entrances

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

# Dedicated producer job.
materializer.produce(
    dataset_factory=source_factory,
    provider_factory=provider_factory,
    device="cuda:0",
    snapshot_samples=50_000,
)

# Repeat explicitly when the job intends to advance through more snapshots.

# Training process: reopen this at every epoch boundary.
with materializer.open(dataset_factory=source_factory) as dataset:
    train_one_epoch(dataset)
```

`open()` never accepts a provider or device. `status()` also inspects only the
published prefix and never constructs a provider. `produce()` owns device
selection, model construction, staging writes, and publication.

The selected catalog revision is fixed when `open()` returns. A concurrent
producer may append later revisions, but an already opened dataset does not
observe them and does not need to synchronize with them.

## Universe and selection

The producer materializes the complete map-style universe supplied by
`dataset_factory`; caller selection does not change transform identity,
coverage, or the physical root. If the public source is a `SelectionView`, its
ordered `sample_id` lineage is projected onto the published output prefix only
at the consumer boundary.

An unmaterialized source suffix is valid during projection. Unknown lineage,
duplicate lineage, or a selection row that is not in the source universe is an
error. This preserves chains such as `filter -> synthesize -> filter ->
tokenize`: each transform covers its complete stage universe, while the final
consumer sees the ordered intersection of selections that already falls within
the published prefix.

## Identity

One materialization is identified by:

- `dataset_id`, the stable logical dataset name;
- `split`;
- `input_id`, the semantic input identity;
- `provider_id`, the semantic transform implementation and model identity;
- the output view and schema contract.

Every catalog entry and delta store must match this identity. Reusing a
physical root with a different identity is a hard error. Physical paths,
device, batch size, process topology, cache location, service endpoint,
snapshot size, and writer concurrency are execution details and do not enter
the semantic identity.

When an explicit semantic ID is absent, resume compatibility fingerprints the
corresponding factory. With an explicit `input_id` or `provider_id`, that ID is
authoritative; callers must change it whenever source or provider semantics
change.

`sample_id` is the stable cross-stage lineage key. `sample_index` is only the
dense ordinal within a universe or one physical delta store. Publication
renumbers every segment locally while preserving each source `sample_id`.

## Producer execution

Each `produce()` call holds `<root>/.producer.lock` for that bounded invocation.
A second producer for the same root is rejected. The call publishes at most one
snapshot and then returns. If computation is needed, one provider is constructed
for the invocation and closed before return.

Provider output first enters private resumable fragments. Only durable dense
coverage can be published. One call computes at most `snapshot_samples` missing
rows and force-publishes that bounded prefix. If a failed earlier call already
left durable unpublished coverage, the next call may publish that coverage and
return without constructing a provider. Catalog writes are atomic, so consumers
see either the previous complete catalog or the next complete catalog, never a
partial entry.

A failed provider run leaves its unpublished fragments private. A later run
with compatible identity reuses durable coverage and computes only missing
indexes. It may reopen the same input identity with a monotonically larger
sample count and append the new suffix. Shrinking the input or changing its
identity is rejected.

If the source exposes `sealed=True` and complete coverage is published, the
catalog is sealed. Ordinary map-style sources without a `sealed` property are
also treated as fixed and sealed after complete publication. Only a source that
explicitly exposes `sealed=False` remains appendable across producer runs.

## Consumer execution

The consumer validates that catalog entries form one dense append-only prefix,
that paths stay inside the root, and that every referenced store has matching
identity, range length, schema, and provenance. It then opens the immutable
segments as one map-style dataset.

Scalar access, negative indexes, and `__getitems__` requests with arbitrary
order or duplicates are routed to segment-local indexes without changing the
request order. Closing the dataset only closes its segment readers; it never
waits for, signals, or releases a producer.

## Failure contract

- A missing catalog is an empty prefix, not a request to generate data.
- An unrecognized non-empty root is an error.
- Catalog gaps, overlaps, rewrites, path escapes, or corrupt delta stores are
  errors.
- Store, input, provider, or output-contract identity mismatch is an error.
- Provider startup or execution failure is returned only by the producer.
- Partial fragments are never exposed to consumers.
- A producer lease is exclusive per physical root.
- Selection projection must preserve one-to-one `sample_id` lineage.

## Offline boundary

The existing `write()` and `snapshot()` APIs remain offline materialization
tools with their existing multi-device and preview behavior. They do not define
the online consumer lifecycle. Online callers should use only `open()` for
consumption and `produce()` for generation.

## Validation checklist

- Opening an uninitialized root returns length zero and performs no writes.
- Opening the same compatible root returns the prefix named by its catalog.
- A dataset opened before publication keeps its original length.
- Reopening at the next epoch observes the latest published prefix.
- Every snapshot store is an immutable delta, while the logical dataset is the
  concatenation from index zero through the latest snapshot.
- Provider failure cannot expose unpublished coverage.
- Every producer call appends at most one snapshot; repeated calls advance the
  prefix explicitly.
- Producer reruns reuse durable fragments and support append-only input growth.
- A second producer and every identity mismatch fail explicitly.
- Selection projection permits an unpublished source suffix but rejects unknown
  or duplicate lineage.
