# TODO

## Filter follow-ups

- Consider an advanced metrics sink interface only if users need outputs beyond
  the built-in parquet side cache.
- Consider optional columnar metrics schemas after the JSON payload shape has
  stabilized in real use.
- Revisit lazy merging of multiple selected partition indexes only together with
  cache snapshot lifecycle. Single-label filtered datasets keep their shard-lazy
  file index; multi-label selections currently merge selected indexes eagerly.

## Materializer follow-ups

- Validate server-mode fork reader/writer behavior on the target Linux machines
  with real store/materializer inputs. Local provider paths should keep spawn
  because those processes may own torch/CUDA/provider state directly.
- Validate whether distributed LBA tail flush can use metadata-only flush with
  the stable global sample indexes now exposed by the map-style indexed loader.
