# Runtime follow-up validation

Date: 2026-07-19

## Goals

1. Measure per-request `ProviderServer` overhead with a real LongCat provider.
2. Separate connection/control round-trip latency from model and payload transfer.
3. Validate server-mode fork readers on Linux.
4. Verify whether distributed LBA already uses index metadata for final flushes.

## Method

- Compare direct and remote LongCat calls with the same 2-second waveform.
- Run batch sizes 1 and 8 after two warmup calls.
- Measure an empty `PING` request on the already-loaded server as an upper bound
  on the latency a persistent connection could remove.
- Run the remote provider/materializer/filter tests on Linux Python 3.9.
- Run the LBA distributed suite with two-rank Gloo smoke coverage.

The benchmark entry is `scripts/benchmark_provider_server.py`.
