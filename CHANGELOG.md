# Changelog

## Unreleased

- Optimize store and source hot paths with manifest/tar handle caches, persisted
  payload indexes and shard groups, lazy manifest reads, bounded CSV preparation,
  and explicit resource cleanup.
- Add integrity levels, filter rule identity/versioning, lazy filter index
  iterators, bounded writer queues, allocation-efficient collation, and
  large-dataset weighted sampling.
- Harden tar and cache sidecar validation, preserve sampling across extreme
  finite weight ranges, and retain legacy writer, row-map, and manifest helper
  contracts while internals use the optimized paths.
- Change `SpeakerIdDataset` to accept explicit multi-reference
  `SpeakerAssignment` mappings, add generic grouped speaker-audio reads, and
  keep Qwen-specific synthesis outside the speaker dataset abstraction.

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
