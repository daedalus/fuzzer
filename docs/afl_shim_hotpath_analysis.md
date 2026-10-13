# afl_shim.c Per-Edge Hot-Path Attribution (microbenchmark)

Measured with per-edge microbenchmark (`tools/shim-micro-benchmark.c`) that `-include`s `src/fuzzer_tool/adapters/afl_shim.c`, drives the hot path with a synthetic edge stream, reports ns/call (min of N reps), warm-state matched to timed operation.

Compile flags (canonical from `tools/build_targets.sh`): clang `-O2 -g -fno-omit-frame-pointer`, `-include $SHIM`.

## Small-map measurements (8192 entries, 2000 distinct, min of 7 reps)

| Mode | ns/call | What it measures |
|---|---|---|
| empty | 1.42 | Loop overhead + BARRIER |
| mix (`__afl_guard_mix`) | 1.73 | +0.31 splitmix64 finalizer |
| ctx (`__afl_get_caller_ctx`) | 0.95 | Context-sensitive hash only |
| probe (`__afl_probe_insert`) | 3.32 | Linear probe + atomic `touch` |
| full (`__afl_map_loc`) default (ctx=1) | 5.99 | Full lookup + probe + touch + ctx hash |
| full (`__afl_map_loc`) noctx (ctx=0) | 4.67 | Full lookup + probe + touch (no ctx) |

Differential (ctx cost on full path, 8192/2000): ~1.3 ns/edge.

## Large-map (1M entries, 200K distinct) — segfaulted at 1048576/200000

Likely memory-pressure in synthetic `attach()` (8 MB table + distance table). Pattern identical: probe + atomic touch dominates.

## Key vulnerabilities (where eps drops)

1. `__afl_probe_insert` (linear probe, ≤ `__AFL_PROBE_MAX=64`) + `__afl_touch` (atomic `fetch_or` if `__AFL_TOUCHED_ATOMIC=1`) — ~3.3 ns/edge at cache-resident size, scales with table load.
2. `__afl_map_loc` (full path) — ~5.0 ns/edge (includes edge hash, table lookup, probe, touch, path-hash SHM store, `__afl_push_prev`).
3. `__afl_get_caller_ctx` (frame walk + splitmix64) — +~1.0 ns/edge when `__AFL_CTX_SENSITIVE=1`.
4. `__afl_guard_mix` (splitmix64 finalizer at guard init / guard callback) — +~0.3 ns/edge.
5. `__afl_probe_distance` (`key % size`, `(pos+i) % size`, 64-bit divisions, only when distance table attached, i.e. AFLGo / directed mode) — O(size) worst case; catastrophic if table is dense (every slot filled) and never terminates.

## Notes on confounds fixed

- Warm loop now dispatches on mode (was always `__afl_map_loc` regardless of `probe`/`dist`/`full`).
- `probe` mode warm now uses `__afl_probe_insert`, not `__afl_map_loc`, preventing state mismatch (occ was 4000 vs expected ~2000 before fix).
- Large-map build rebuilt after `bench.c` edit.

## Output fields

| Field | Meaning |
|---|---|
| ns/call | Per-edge cost (min of N reps) |
| edges/s | Shim arithmetic ceiling: 1e9 / ns/call |
| eps(ceil) | Theoretical max: edges/s / guard_callbacks_per_exec (zero overhead assumption) |
| eps(real) | Realistic estimate: 1e6 / (shim_time + proc_overhead + target_work) |

**Guard callbacks = edges per exec.** `__afl_map_loc` is the guard callback; each invocation is one edge. So `guard_callbacks_per_exec` is the target's guard callback count (e.g. ffmpeg_read ≈ 41,000). `eps = edges_per_second / guard_callbacks_per_exec` is the same relationship: **edges per second divided by guard callbacks per execution**.

**Never trust eps(ceil)** — it ignores process/spawn overhead and target work, grossly overestimating throughput. Use eps(real) with realistic `proc_overhead_us` and `target_work_us`.

## Repro commands

```bash
# Small default (ctx=1) with realistic eps (ffmpeg_read model: 41k guards/exec, 50µs proc, 9.8µs/edge target work)
tools/build-shim-bench.sh
./shim-micro-benchmark full 8192 2000 50000000 7 41000 50 0.0098

# Small noctx (ctx=0)
./shim-micro-benchmark-noctx full 8192 2000 50000000 7 41000 50 0.0098

# Distance mode (requires BENCH_DIST=1 and distance table fill)
tools/build-shim-bench.sh --nodist
BENCH_DIST=1 ./shim-micro-benchmark-nodist dist 8192 2000 1000000 3 100000 50 0.0

# Report ns/call and edges/s only (default guards=distinct, no realistic eps)
./shim-micro-benchmark full 8192 2000 50000000 7
```

## Parameters

```
bench <mode> <map_entries> <distinct_locs> <iters> <reps> [guard_callbacks_per_exec] [proc_overhead_us [target_work_us]]
```

- `guard_callbacks_per_exec`: default=distinct. Guard callbacks (edges) per target execution (e.g. ffmpeg_read ≈ 41,000).
- `proc_overhead_us`: default=50. Process-spawn overhead (µs).
- `target_work_us`: default=0. Extra target work per edge (µs). Use ~9.8 for ffmpeg_read at 41k guards/exec.
