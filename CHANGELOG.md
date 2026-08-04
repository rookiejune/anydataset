# Changelog

## Unreleased

- Add opt-in `cost_aggregation="padded_max"` dynamic batches with the hard
  bound `len(batch) * max(sample_cost) <= max_batch_memory`, while preserving
  additive `"sum"` aggregation by default.
- Add `RejectReplaceDataset` for low reject-rate online CPU safety nets that
  replace rejects via sequential look-ahead and a worker-local accept buffer.
- Add explicit filter apply reports via `FilterRule.apply_with_report(...)`,
  with segmented apply-call timing and hot-cache reports that avoid creating
  run log directories.
- Add `dataset.morphology` audio/speech/speech_grid batch contracts and
  `IndexSelection` for stable map-style index views.
- Extend speech quality filtering and filter collect/type contracts used by
  cached partitions.
- Optimize store and source hot paths with manifest/tar handle caches, persisted
  payload indexes and shard groups, lazy manifest reads, bounded CSV preparation,
  and explicit resource cleanup.
- Add integrity levels, filter rule identity/versioning, lazy filter index
  iterators, bounded writer queues, allocation-efficient collation, and
  large-dataset weighted sampling.
- Harden tar and cache sidecar validation, preserve sampling across extreme
  finite weight ranges, and retain row-map and manifest helper contracts while
  internals use the optimized paths.
- Change `SpeakerIdDataset` to accept explicit multi-reference
  `SpeakerAssignment` mappings, add generic grouped speaker-audio reads, and
  keep Qwen-specific synthesis outside the speaker dataset abstraction.
- Remove the legacy public `Task` enum and `anydataset.dataset.write`
  compatibility writer path; callers should use explicit schemas with
  `collate_fn(schema)` and the public `DatasetWriter` entry point.
- Keep schema-v2 store reads compatible, but emit a `RuntimeWarning` because
  v2 stores lack provenance and should be rematerialized or migrated before
  production publication or cache-sensitive derivation.

## 1.0.0 - 2026-07-03

- Stabilize the canonical `Sample = Mapping[tuple[Role, Modality], Item]`
  public data model.
- Stabilize map-style and iterable dataset entry points, built-in presets, and
  source registry shorthands.
- Stabilize cached filter partitions with resumable construction, metrics side
  output, and multi-device execution.
- Stabilize canonical store read/write APIs, logical store merge, and
  materialized view or modality delta stores.
- Document v1 release checks through `scripts/check_release.py`, which gates on
  version consistency, pytest, clean builds, `twine check`, and wheel-install
  smoke tests.
