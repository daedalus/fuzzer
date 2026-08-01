# Architecture: Coverage, State, and Scheduling Internals

Deep details of subsystems that only matter when you are working inside them. The
entry-point AGENTS.md carries only the summary. Open this file when working on:
coverage/SHM internals, the AFL shim, `--no-shm`/`--deep-coverage` paths, the Elo
meta-scheduler, or state persistence (`state.json`, `edge_tracker.json`, `markov.json`).

## State Persistence

Fuzzer state is saved to `{corpus_dir}/state.json` on shutdown. Use `--resume` to continue:

- `state.json` — exec counts, crash sigs, op stats, seed metadata
- `edge_tracker.json` — per-seed edge coverage
- `markov.json` — Markov chain transitions

Reload paths must skip re-derivation (see "State & double-counting" in docs/refs/bug-classes.md).

## Coverage Modes

- `--no-shm` — forces ptrace for uninstrumented binaries
- `--deep-coverage` — x86-64 decoder for basic block discovery
- Default SHM — for AFL-instrumented targets

## Sparse Entry Coverage

The AFL shim (`src/fuzzer_tool/adapters/afl_shim.c`) uses an open-addressing hash
table of 8-byte entries instead of a fixed byte bitmap:

```c
struct __afl_entry { uint32_t edge_id; uint32_t count; };
```

- Edge ID = `prev_loc ^ cur_loc` (full 32-bit) — **no silent bucket collisions**
- Hash: `edge_id % map_size`, linear probing for matching or empty slot
- `AFL_MAP_SIZE` is in bytes (tradition); shim divides by 8 for entry count
- Default 64KB SHM → 8192 entries (same memory as old 64KB bitmap)
- Count is a 32-bit saturating counter (no Morris probability needed)
- Python API: `ShmCoverage.get_edge_ids()`, `.get_edge_counts()`, `.read_entries()`
- `EdgeTracker.record_edges()` accepts `set[int]` (sparse) or `bytes` (legacy byte-bitmap)

## Markov Persistence

- Markov chain saved to `markov.json` on exit
- Loaded on init; skip retrain if loaded to avoid double-counting
- Transitions accumulate across sessions

## Meta-Scheduler (Elo Arbitration)

- `--elo` alone enables Elo-based arbitration between operator strategies
  (bandit/MOpt/replicator) and seed strategies (ga/weighted/pareto/format); the
  separate `--meta-elo` flag was consolidated into `--elo` (see `_use_elo` in
  `src/fuzzer_tool/services/fuzzer.py`)
- Enable `--mc-bandit`/`--mopt`/`--replicator` alongside `--elo` to add those
  strategies to the arbitration pool
- All available strategies run in shadow; Elo picks which one to trust each iteration
- Strategy ratings tracked in `elo.json` under `strategy_ratings` / `strategy_match_count`
- Probabilistic selection via softmax over Elo gap (temperature=400)
