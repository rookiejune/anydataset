# Runtime follow-up results

Date: 2026-07-19

## Environment

- LongCat: Fudan `145`, NVIDIA GeForce RTX 4090 D, Python 3.12, Torch 2.9,
  `LongCatAudioCodec_encoder.pt`, 2-second 16 kHz speech input.
- Linux fork validation: Fudan `144`, Python 3.9, Torch 2.8.
- Dynamic batching validation: local Python 3.9, Torch 2.8, two-rank Gloo
  smoke test.

## ProviderServer

| Batch | Repeats | Direct median | Remote median | Remote delta | Delta ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 23.235 ms | 24.937 ms | 1.702 ms | 7.32% |
| 8 | 20 | 23.631 ms | 26.801 ms | 3.170 ms | 13.42% |
| 8 | 10 | 24.851 ms | 29.704 ms | 4.853 ms | 19.53% |

On the final batch-8 run, an empty `PING` through the same Unix socket had a
0.135 ms median and 0.150 ms p95. That measurement still includes the
send/receive round trip a persistent client would retain, so 0.135 ms is an
upper bound rather than the expected saving.

The larger 3-5 ms remote delta is dominated by serializing and transferring the
batched waveform and outputs, not by opening the connection. A persistent
connection would add ownership, reconnect, fork, shutdown, and error-recovery
state while removing less than 0.135 ms per request in this workload. The
current one-request isolation contract is retained.

## Linux fork

- `test_provider_service.py`: 7 passed.
- Remote materializer fork-reader and remote filter fork-reader tests: 2 passed.
- Both paths completed with `ProviderServer(start_method="spawn")` and forked
  data readers on Linux.

## Distributed dynamic batching

The distributed dynamic batching validation confirmed that stable global sample
indexes are sufficient for metadata-only final flushes, while index-less records
need an object-gather fallback. The two-rank map-style and iterable smoke cases
passed, and no anydataset dataloader change was needed for this follow-up.
