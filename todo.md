# TODO

## Store follow-ups

- Decide whether existing schema-v1 stores need an offline migration command.
  The reader now rejects them explicitly; re-materialization is the only built-in
  path to schema v2.
- Define an explicit cleanup or lease contract for
  `$ANYDATASET_HOME/cache/store-files`. Automatic eviction is unsafe while callers
  may retain returned `AudioView.FILE` paths beyond a dataset access.

## Filter follow-ups

- Replace same-path filter cache commits with immutable generations and an
  atomic current-generation pointer. Define a reader lease or cleanup contract
  before deleting old generations. Partition file fingerprints now fail fast
  when a live lazy view observes replacement, but they do not preserve the old
  snapshot or pin metrics iteration to it.
- Consider an advanced metrics sink interface only if users need outputs beyond
  the built-in parquet side cache.
- Consider optional columnar metrics schemas after the JSON payload shape has
  stabilized in real use.
- Revisit lazy merging of multiple selected partition indexes only together with
  cache snapshot lifecycle. Single-label filtered datasets keep their shard-lazy
  file index; multi-label selections currently merge selected indexes eagerly.

## Dataset follow-ups

- Define a source-level native indexed-sharding contract before using raw
  dataset `shard()` methods in `IterableAnyDataset.iter_indexed_shard()`. The
  current modulo fallback scans the full stream in every shard to preserve
  stable global sample indexes for filter and materializer caches; enumerating a
  native shard locally would reduce I/O but break that index alignment unless
  the source also propagates each row's original global index.

## Materializer follow-ups

- Benchmark per-request `ProviderServer` connection setup with real providers.
  Design a persistent-connection lifecycle only if IPC setup is a material part
  of batch latency; the current server intentionally isolates one request per
  connection.
- Validate server-mode fork reader/writer behavior on the target Linux machines
  with real store/materializer inputs. Local provider paths should keep spawn
  because those processes may own torch/CUDA/provider state directly.
- Validate whether distributed LBA tail flush can use metadata-only flush with
  the stable global sample indexes now exposed by the map-style indexed loader.
