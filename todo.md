# TODO

## Filter follow-ups

- Consider an advanced metrics sink interface only if users need outputs beyond
  the built-in parquet side cache.
- Consider optional columnar metrics schemas after the JSON payload shape has
  stabilized in real use.
- Revisit lazy partition index loading only together with cache snapshot
  lifecycle; current filtered datasets materialize selected indices eagerly.

## Materializer follow-ups

- Revisit indexed loader performance before wiring LBA into `ViewMaterializer`.
  Store and most materializer inputs are map-style, so the default indexed path
  should likely be a map-style wrapper that returns `(sample_index, sample)` and
  uses a rank sampler over global indexes. Iterable/streaming inputs can keep
  the current runtime-sharded `IterableDataset` path behind a keyword-only escape
  hatch, with map-style errors telling users to pass that keyword.
- Separate the process-model discussion for materializer performance. Outer
  device/provider workers should stay spawn-friendly because providers often
  touch CUDA, but inner PyTorch DataLoader workers only load data and may not
  need the same start method. Compare always-spawn, torch default, and
  fork-on-Linux behavior for map-style indexed wrappers; if caching the dataset
  instance in the wrapper, make spawn serialization drop that cache so workers
  can lazily rebuild from `dataset_factory`.
- Recheck distributed LBA tail flush after the indexed loader decision. The
  current iterable wrapper preserves correctness by object gather, while a true
  map-style indexed loader should let LBA use metadata-only flush through stable
  global sample indexes.
