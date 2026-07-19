# Verified conclusions

- Keep the one-request `ProviderServer` connection contract. Real LongCat
  requests showed a 0.135 ms connection/control upper bound, while most remote
  overhead came from batch serialization and transfer
  ([result, lines 16-29](results/001_runtime_followups.md#L16-L29)).
- Server-owned providers with forked materializer/filter readers work on the
  target Linux environment
  ([result, lines 31-36](results/001_runtime_followups.md#L31-L36)).
- Distributed LBA already selects metadata-only final flushes when stable sample
  indexes are available; no anydataset integration change is required
  ([result, lines 38-44](results/001_runtime_followups.md#L38-L44)).
