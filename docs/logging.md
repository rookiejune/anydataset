# Production Logging

`anydataset` separates operational logs from dataset audit data:

- Runtime logs live under `$ANYDATASET_HOME/logs/<timestamp>-<pid>/`.
- `run.json` records the run id, process id, working directory, command argv,
  package version, and resolved `ANYDATASET_HOME`.
- `<source>.log` files remain human-readable text logs for warnings, resume
  summaries, cache decisions, lifecycle messages, and failures.
- `events.jsonl` mirrors those operational messages as structured JSON lines
  with `source`, `event`, `level`, `run_id`, `pid`, and event-specific
  `fields`.

The runtime logging helpers do not configure Python's root logger and do not
store sample-level audit metrics. Per-sample filter decisions and metrics remain
part of filter cache artifacts; store provenance remains part of store
manifests.

## Event Boundaries

- Source prepare events record cache hit/build state, source count, row count,
  converted/reused part counts, manifest path, cache directory, and worker
  settings.
- Filter events record cache miss reasons, resume coverage, throttled structured
  progress, per-worker performance summaries, predicate OOM batch splits,
  aggregate run summaries, and published generation paths.
- Materializer events record resume coverage, staged status, provider OOM batch
  splits, and final published store paths.
- Provider service events record server start, readiness, stop, forced
  termination, and startup failure.

`filter_predicate_oom_split` reports cumulative per-worker `oom_count` and
`predicate_calls`; `split_call_ratio` is `oom_count / predicate_calls` at the
time of that OOM.

`filter_progress` is emitted at most once every 30 seconds when scan or writer
progress changes, and once at run exit. It records scan/writer counts and rates,
resume/target counts, writer pending depth, maximum pending depth, accumulated
writer backpressure, and elapsed time. This event is intentionally throttled
rather than emitted per batch.

`filter_worker_summary` is emitted when a single-device scan or multi-device
worker exits normally or with an error. It records:

- requested batch size and reader worker/prefetch settings;
- processed, selected, and loader sample counts;
- loader batch-size min/mean/max and accumulated loader wait time;
- predicate setup time, call/sample counts, effective batch-size min/mean/max,
  and accumulated predicate time;
- final `oom_count`, `predicate_calls`, and `split_call_ratio`, including the
  zero-OOM case;
- multi-device output-queue blocked time, elapsed time, and sample throughput.

`filter_run_summary` aggregates the worker counters and adds scan/writer counts
and rates, writer job latency, maximum pending depth, writer backpressure,
fragment/replay time, execution settings, final status, and total elapsed time.
Failed and interrupted runs emit the summary as a warning when the process can
exit through the normal Python cleanup path. Forced process termination cannot
guarantee a final event; the latest `filter_progress` remains the recovery
signal in that case.

The timings are deliberately coarse, low-overhead wall-clock counters. They are
intended to distinguish reader starvation, predicate cost, inter-process queue
blocking, and writer backpressure. Predicate-specific internal stages remain
the predicate owner's responsibility, and GPU telemetry remains outside the
mandatory runtime rather than hard-coding an NVIDIA-specific collector.

Stdout progress remains a separate interactive surface. TTY sessions use tqdm;
non-interactive jobs print periodic one-line progress summaries so scheduler logs
show current throughput without opening the run log directory.

For resumed work, the primary count and percentage include previously completed
samples, while rate, ETA, and stage counters cover only work performed by the
current run. The display labels the historical `resumed` count and current-run
progress separately so cumulative coverage is not mistaken for throughput.
