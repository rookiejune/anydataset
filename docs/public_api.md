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
documented in README/docs examples. Some older implementation names may remain
importable for compatibility; new application code should use the stable paths
listed here instead of relying on incidental module attributes.

## Stable public API

Use these paths for application code:

- `anydataset`: top-level dataset construction, specs, presets, filters, and
  language helpers.
- `anydataset.types`: canonical sample, item, schema, role, modality, view,
  metadata, source, preset, and language types.
- `anydataset.presets`: built-in dataset preset classes.
- `anydataset.dataset`: dataset base classes, index selection, generic collate
  helpers, and morphology collate/view contracts.
- `anydataset.filter`: cached filter rules, filtered datasets, online reject /
  replace filtering, filter decisions, and filter cleanup entry points.
- `anydataset.store`: canonical store writing, view/materialization providers,
  explicit store migration, payload integrity checks, and retained-file cleanup.
- `anydataset.runtime`: process/device runtime configuration.
- `anydataset.provider`: built-in model/provider classes.
- `anydataset.provider_service`: provider process server and remote provider /
  filter client factories.

## Extension API

These paths are intended for users extending anydataset:

- `anydataset.dataset.source.DatasetSource` and
  `anydataset.dataset.source.ShardingSource` define source contracts.
- `anydataset.register_source(...)` / `anydataset.dataset.source.register_source(...)`
  register custom physical sources.
- Source keys identify physical access categories such as Hugging Face datasets,
  Hugging Face Hub file trees, stores, and tabular files. Dataset-specific logic
  belongs in `anydataset.presets`, not in `anydataset.dataset.source`.
- Built-in source classes under `anydataset.dataset.source` represent generic
  physical source categories, not concrete datasets. New concrete built-in
  datasets should live under `anydataset.presets` and map onto these generic
  sources with a preset-local parser.
- Older source implementation names may remain importable for compatibility;
  prefer `Spec`, presets, category source shorthands such as `hf://`,
  `hf-files://`, `hf-disk://`, and `store://`, and registration helpers in
  ordinary application code.
- `anydataset.rowmap` contains helpers for mapping raw rows to canonical
  samples in presets and user-defined parsers.
- Provider protocols and wrapper classes exported from `anydataset.store`
  define view and modality materialization contracts.
- Quality rule classes under `anydataset.quality.rules`,
  `anydataset.quality.text`, `anydataset.quality.translation`, and
  `anydataset.quality.speech` are public rule-building utilities.

## Internal API

Do not depend on these paths from user code:

- `anydataset._compat`, `anydataset._immutable`, `anydataset._io.*`,
  `anydataset._runtime.*`, and `anydataset._validation`.
- `anydataset.dataset._shuffle` and source-specific prepare helpers or
  prepared-row implementation details.
- `anydataset.dataset.source.registry._SourceFactory.create(...)` and
  `_SourceFactory.exist(...)`; these are registry plumbing used by
  resolver/dataset internals, not application extension points.
- Concrete dataset helpers embedded in presets, such as FSD50K parser/load
  helpers, unless they are explicitly exported from `anydataset.presets`.
- `anydataset.filter.cache.*` and `anydataset.filter.runtime.*`.
- `anydataset.store.config`, `anydataset.store.jsonio`,
  `anydataset.store.paths`, `anydataset.store.reader`,
  `anydataset.store.manifest.*`, `anydataset.store.materialize.*`,
  `anydataset.store.part.*`, and `anydataset.store.payload.*`.
- `anydataset.presets.registry` and preset-private parser/helper functions.
- Names that start with `_`, even if a compatibility module can import them.

## Store migration policy

Legacy store formats are not read through silent compatibility layers. Use
`anydataset.store.migrate_store(source, output)` or
`anydataset-store migrate source output` to create an explicit schema-current
store. Store internals such as manifest readers, payload index writers, part
commit helpers, and JSON path helpers remain private implementation details.

## Adding public names

When promoting a helper to public API:

1. Export it from the nearest stable package `__init__.py`.
2. Add it to that package's `__all__`.
3. Document the import path in README or focused docs.
4. Add a regression test that the public import path works.
5. Avoid exposing lower-level implementation modules when a package-level
   wrapper can preserve future refactoring freedom.
