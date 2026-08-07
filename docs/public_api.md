# Public API Boundaries

This document defines which import paths are stable for users and which paths
are internal implementation details. Python may allow importing an internal
module, but only the public paths below are covered by compatibility promises.

## Stability contract

- **Stable public API** follows semantic-versioning expectations.
- **Extension API** is public for advanced integrations, but changes may require
  migration notes when source/provider/runtime contracts evolve.
- **Internal API** is private to anydataset. Do not use it from application code;
  tests may import it to exercise implementation details.

As a rule, stable public names are exported from a package `__all__` and are
documented in README/docs examples. Incidental module attributes are not public
API unless they are listed here.

## Stable public API

Use these paths for application code:

- `anydataset`: top-level dataset construction, specs, presets, filters, and
  language helpers.
- `anydataset.types`: canonical sample, item, schema, role, modality, view,
  metadata, source, preset, language types, encoded `FileBytes` payloads, and
  the structured audio-token `SemanticAcousticView` and `SemanticGlobalView`
  mapping contracts.
- `anydataset.presets`: built-in dataset preset classes.
- `anydataset.dataset`: dataset base classes, index selection, generic collate
  helpers, and morphology collate/view contracts. Focused universe and selection
  extension contracts are listed below.
- `anydataset.filter`: cached filter rules, filtered datasets, scalar and batch
  filter predicate contracts, `FilterRun` / `FilterRunStatus` from the dynamic
  `FilterRule.open()` entrance, explicit blocking apply reports, online reject /
  replace filtering, filter decisions, and filter cleanup entry points.
- `anydataset.store`: canonical store writing; view, modality, and complete-sample
  materializers and provider protocols; scalar and batch transform types;
  materialization status and the lifecycle-controlled
  `MaterializingViewDataset` underlying online `ViewMaterializer.open()` results;
  explicit store migration; payload integrity checks; and retained-file leasing
  and cleanup. A selected input may return a `SelectionView` wrapping that online
  universe rather than the bare materializing dataset.
- `anydataset.runtime`: process/device runtime configuration.
- `anydataset.provider`: built-in model/provider classes.
- `anydataset.provider_service`: provider process server and remote provider /
  filter client factories.
- `anydataset.synthesis.s2st`: stable synthetic-S2ST source slots, growth plans,
  views, stage snapshots, and append-only final datasets. Concrete model and
  workspace bindings remain outside anydataset.
- `anydataset.quality`: quality rule-building utilities for text,
  translation, and speech filters.

Built-in presets are map-style and are constructed through
`AnyDataset.preset()`. `IterableAnyDataset` is constructed directly for custom
streaming sources; it does not expose a preset constructor.

`anydataset.provider` and `anydataset.provider_service` are intentionally
separate public surfaces. Provider modules define model/data transformation
objects; provider service defines process isolation, server lifecycle, and
remote client factories for executing those objects out of process. Wire
commands, request/response envelopes, connection loops, and serialization
helpers live under private implementation modules such as
`anydataset.provider._protocol` and are not public API.

Batch-aware integrations can type their filter predicates with
`anydataset.filter.BatchFilterPredicate`. Store providers can import
`BatchOutput`, `BatchViewTransform`, and `BatchModalityTransform` directly from
`anydataset.store`. Complete-sample integrations can type providers with
`SampleProvider` and `BatchSampleProvider`; callers do not need to depend on
the internal materializer modules that consume those contracts.

## Dynamic view composition

Dynamic operations distinguish complete execution from returned selection:

- transforms and filters execute over a complete `DatasetUniverse`;
- `SelectionView` applies the ordered intersection only at the returned dataset
  boundary;
- unknown live filter decisions are unresolved, not reject;
- one-to-one transforms preserve stable `sample_id` and rebase selections by that
  ID;
- `sample_index` is only a dense ordinal within one universe/store;
- transform and filter identities exclude selection state.

`FilterRule.open()` returns `FilterRun`. Use `run.dataset` for the live selection,
`run.wait()` to wait for complete decision coverage, and `run.close()` or the
context manager to wait and release resources. `FilterRule.apply()` remains the
blocking compatibility surface that returns `FilteredDataset`.

`ViewMaterializer.open()` keeps the ready path fully lazy when `input_id` is
explicit. With `input_id_factory`, it constructs the input identity object once
to validate canonical provenance, reuses that object when it also supplies the
returned selection, and still never constructs the provider. Otherwise
foreground access computes requested values and starts a background
full-universe sweep. `MaterializingViewDataset.close()` waits for complete
transform coverage and staging persistence; it does not publish the canonical
store. Publication remains an explicit `ViewMaterializer.write()`/finalize
operation. An optional logical `dataset_id` decouples canonical and universe
identity from the physical `output_dir` basename; omission preserves the basename
fallback.

`provider_id` is global operation provenance for the AnyTrain backend plus the
AnyDataset adapter and semantic model recipe. It is unrelated to a sample,
filter, or selection and is never sample metadata.

## Extension API

These paths are intended for users extending anydataset:

- `anydataset.dataset.universe` exports `DatasetUniverse`, `SampleIdentity`, and
  `IndexIdentity`. `DatasetUniverse` is the complete map-style sample space for
  operation coverage; stable lineage must come from `sample_id()`, not a dense
  position.
- `anydataset.dataset.view` exports the `Selection` protocol, `DecisionSet`,
  `StaticSelection`, `SelectionView`, and `UnknownDecisionError`. These types
  describe membership and ordered return-boundary intersection; they do not
  authorize transforms to scan only selected rows.

- `anydataset.dataset.source.DatasetSource` and
  `anydataset.dataset.source.ShardingSource` define source contracts.
- `anydataset.register_source(...)` / `anydataset.dataset.source.register_source(...)`
  register custom physical sources. Their optional `operational_load_options`
  names source-specific load options that do not participate in `Spec.id`.
- Source keys identify physical access categories such as Hugging Face datasets,
  Hugging Face Hub file trees, stores, and tabular files. Dataset-specific logic
  belongs in `anydataset.presets`, not in `anydataset.dataset.source`.
- Built-in source classes under `anydataset.dataset.source` represent generic
  physical source categories, not concrete datasets. New concrete built-in
  datasets should live under `anydataset.presets` and map onto these generic
  sources with a preset-local parser.
- Prefer `Spec`, presets, category source shorthands such as `hf://`,
  `hf-files://`, `hf-disk://`, and `store://`, and registration helpers in
  ordinary application code; source implementation modules remain outside the
  stable application API unless exported here.
- `anydataset.rowmap` contains helpers for mapping raw rows to canonical
  samples in presets and user-defined parsers.
- Provider protocols and wrapper classes exported from `anydataset.store`
  define view, modality, and complete-sample materialization contracts.
- Quality rule classes should be imported from `anydataset.quality` by default.
  Focused submodules such as `anydataset.quality.rules`,
  `anydataset.quality.text`, `anydataset.quality.translation`, and
  `anydataset.quality.speech` remain public extension paths.

## Internal API

Do not depend on these paths from user code:

- `anydataset._compat`, `anydataset._immutable`, `anydataset._io.*`,
  `anydataset._runtime.*`, and `anydataset._validation`.
- `anydataset.dataset._shuffle` and source-specific prepare helpers or
  prepared-row implementation details.
- `anydataset.dataset.source._registry`; this is source registry plumbing used by
  resolver/dataset internals, not an application extension point.
- Concrete dataset helpers embedded in presets, such as FSD50K parser/load
  helpers, unless they are explicitly exported from `anydataset.presets`.
- `anydataset.filter.cache.*`, `anydataset.filter.runtime.*`, and the concrete
  `anydataset.filter.live` implementation module. Import `FilterRun` and
  `FilterRunStatus` from `anydataset.filter`.
- `anydataset.store.config`, `anydataset.store.jsonio`,
  `anydataset.store.paths`, `anydataset.store.reader`,
  `anydataset.store.manifest.*`, `anydataset.store.materialize.*`,
  `anydataset.store.part.*`, and `anydataset.store.payload.*`.
- `anydataset.presets.registry` and preset-private parser/helper functions.
- `anydataset.provider._protocol` and provider service wire protocol /
  connection-loop helpers.
- Names or modules that start with `_`.

## Store migration policy

The canonical physical format remains Parquet manifests plus tar payload shards.
Online filter decision/selection fragments use Arrow IPC as private in-progress
control-plane records, while completed filter generations remain Parquet.
Arrow IPC fragments are not a public sample-payload format. Any migration of
waveform, token, or heterogeneous payloads to Arrow requires the benchmark gate
recorded in [`experiments/todo.md`](experiments/todo.md) and a new versioned store
contract.

Legacy store formats are not read through silent compatibility layers. Store
readers use an explicit `legacy_policy` when a legacy format is still readable:
the default is `reject`, while deliberate audits may opt in with `allow` or
`warn`. Strict publishing and cache-sensitive paths must continue to reject
legacy inputs. Upgrade a schema-v1 store with
`anydataset.store.migrate_store(source, output)` or
`anydataset-store migrate source output`. Schema-v2 stores lack provenance and
must be rematerialized to schema-v3; `migrate_store` does not upgrade them.
Store internals such as manifest readers, payload index writers, part commit
helpers, and JSON path helpers remain private implementation details.

## Runtime pickle policy

The store directory schema is the long-lived persisted data format. In
contrast, `StoreDataset`, `DatasetWriter`, and the retained-file lease carried
by store readers use independently versioned pickle states as runtime transport
contracts for process spawn and `DataLoader` workers. Do not retain those
pickles as durable dataset assets. Unversioned legacy runtime states follow the
explicit v0 migration path; unsupported versions, fields, and field types fail
instead of being guessed.

## Adding public names

When promoting a helper to public API:

1. Export it from the nearest stable package `__init__.py`.
2. Add it to that package's `__all__`.
3. Document the import path in README or focused docs.
4. Add a regression test that the public import path works.
5. Avoid exposing lower-level implementation modules when a package-level
   wrapper can preserve future refactoring freedom.
