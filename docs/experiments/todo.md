# Pending Validation

## Versioned physical Parquet shards for distributed reads

- **Status:** design and benchmark pending.
- **Constraint:** preserve dense global modulo `sample_index` semantics within the
  physical dataset and stable `sample_id` lineage across one-to-one stages;
  filter, resume, and ordering behavior must remain unchanged.
- **Candidate design:** introduce a schema-versioned physical Parquet shard layout
  with an explicit global-index mapping, so ranks can decode disjoint row groups
  instead of redundantly decoding the same row group.
- **Design requirements:** define the index mapping, schema version, and an
  explicit migration path; do not add silent compatibility for an incompatible
  physical format.
- **Ready gate:** run a representative multi-rank benchmark that measures row-group
  decode count, I/O, CPU cost, and end-to-end throughput, and confirms repeated
  row-group decoding is a material bottleneck before implementing the redesign.
- **Acceptance:** reduce aggregate decode work or improve end-to-end throughput
  while preserving exact sample ownership and the existing filter/resume/index
  semantics.

## Arrow-backed canonical payload shards

- **Status:** benchmark and format design pending; do not implement yet.
- **Current contract:** canonical sample/view manifests are Parquet and
  heterogeneous payloads are tar shards. Online filter decision fragments may
  use Arrow IPC, but they are control-plane records rather than sample payloads.
- **Candidate design:** evaluate a versioned Arrow IPC payload layout for
  waveform, token, and structured tensor views without changing stable
  `sample_id` lineage or dense store-local `sample_index` semantics.
- **Ready gate:** benchmark representative Common Voice and two-waveform S2ST/TTS
  stores for scalar random access, reordered `__getitems__`, sequential scan,
  multi-rank reads, open-file pressure, storage size, write/finalize cost, and
  memory use. Compare against the existing Parquet-manifest/tar-payload store
  with offset sidecars.
- **Acceptance:** adopt Arrow payloads only if the benchmark shows a material
  end-to-end benefit without weakening payload type coverage, integrity checks,
  resumable publication, or explicit schema migration. Otherwise retain tar.

## Private staged-payload replay for online materializers

- **Status:** reader/index design and implementation pending.
- **Problem:** `MaterializingViewDataset` tracks completed coverage but does not
  retain completed payload values. A post-filter full scan followed by a
  tokenizer scan over the same online lifecycle can therefore invoke an
  expensive synthesis provider more than once per universe sample.
- **Constraint:** do not solve this with an unbounded in-memory waveform/token
  cache, and do not expose partial staging as a public dataset mode. Provider,
  writer, and replay failures must remain sticky.
- **Candidate design:** add a private immutable-fragment replay layer with a
  dedicated mapping from universe-global sparse `sample_index` to
  fragment-local dense row. Define writer-completion visibility before replay,
  validate fragment identity and lineage, and bound open fragment handles and
  decoded payload cache independently of universe size.
- **Ready gate:** verify one provider call per universe sample across sequential
  filter/synthesis/tokenizer reads in one online lifecycle, replay from resumed
  fragments, scalar/batch order and duplicates, async writer failure, close and
  handle cleanup, and FILE/tensor payload behavior. Re-run materializer, filter,
  lineage, store-reader, ruff, and basedpyright regression suites.
- **Acceptance:** reuse durable staged outputs without recomputation or unbounded
  payload memory, while keeping canonical publication explicit and partial
  staging private.
