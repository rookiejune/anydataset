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
- Filter events record cache miss reasons, resume coverage, predicate OOM batch
  splits with cumulative per-worker split/call counters, and published generation
  paths.
- Materializer events record resume coverage, staged status, provider OOM batch
  splits, and final published store paths.
- Provider service events record server start, readiness, stop, forced
  termination, and startup failure.

`filter_predicate_oom_split` reports cumulative per-worker `oom_count` and
`predicate_calls`; `split_call_ratio` is `oom_count / predicate_calls` at the
time of that OOM.

Stdout progress remains a separate interactive surface. TTY sessions use tqdm;
non-interactive jobs print periodic one-line progress summaries so scheduler logs
show current throughput without opening the run log directory.
