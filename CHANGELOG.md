# Changelog

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
