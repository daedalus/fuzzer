# Handover: skittercreek/tailslayer ports — remaining work

Original round 17 (last edit before deletion in `1c689e8a`, 2026-09-06).
Pruned 2026-09-26 to open items; full original: `git show 1c689e8a^:docs/handover/handover_skittercreek_tailslayer_port.md`.
Verified against `4c021daa`. Items 1–7 and 3 (scoped) shipped.

## Open

### G. Intermittent `shmat()` failure — watch item, no fix
Twice in ~50 runs the first `ShmCoverage` in a process read an empty edge table
after a child exited 0. Header not captured: unknown whether the child failed to
attach or the parent raced the read. Stale-view fix (`171c472f`, `cleanup()`
nulls `_map`/`_entries`/`_tail`) not established as cause: 0 in ~40 runs since
vs ~1/25 before. Segment exhaustion ruled out (`ipcs -m` clean).
Action only if `tests/test_ctx_and_map_size.py` drop-counter tests go
intermittently red: capture `read_edge_count()`, `read_dropped_edges()`,
`read_generation()` (`adapters/shm.py:ShmCoverage`) with the child's exit status.

### H. Item 13 — byte-level timing-anomaly attribution (deferred by design)
Join `ExecTimeCalibrator` anomaly timestamps
(`core/analyzers/analyzer_exec_time_anomaly.py`) to the mutation-event stream via
`core/temporal_join.py:join_streams`, attributing a slow exec to a byte/operator.
`services/report.py:_temporal_correlation` uses `join_streams` only for
coverage/discovery streams. Deferred until item 5 has real campaign output to
join against.

### Open question (Gabriel)
Real target with a non-CRC, non-Adler/Fletcher bitmask-style checksum/flags
field, to validate item 1 (`core/xor_map_solver.py`, GF(2) Gauss-Jordan)?
Absent one, item 1 stays synthetic-validated only.

### Caveat for any follow-up measurement
No coverage delta has been measured for any item here. Pre-C2-fix edges/s
figures are unreliable. Use `tools/benchmark.py paired` against a fixed seed and
exec budget; `ffmpeg_read` is the only real target that stresses the map cap,
`tools/gen_synthetic_target.py` gives known-answer targets.
