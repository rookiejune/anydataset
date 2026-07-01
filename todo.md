# TODO

## Filter follow-ups

- Consider an advanced metrics sink interface only if users need outputs beyond
  the built-in parquet side cache.
- Consider optional columnar metrics schemas after the JSON payload shape has
  stabilized in real use.
- Revisit lazy partition index loading only together with cache snapshot
  lifecycle; current filtered datasets materialize selected indices eagerly.

## Materializer follow-ups

- Validate server-mode fork reader/writer behavior on the target Linux machines
  with real store/materializer inputs. Local provider paths should keep spawn
  because those processes may own torch/CUDA/provider state directly.
- Recheck distributed LBA tail flush after the indexed loader decision. The
  map-style indexed loader should let LBA use metadata-only flush through stable
  global sample indexes.
