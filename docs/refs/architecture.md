# Architecture: Coverage, State, and Scheduling Internals

Deep details of subsystems that only matter when you are working inside them. The
entry-point AGENTS.md carries only the summary. Open this file when working on:
coverage/SHM internals, the AFL shim, `--no-shm`/`--deep-coverage` paths, the Elo
meta-scheduler, or state persistence (`state.pkl.gz` via `core/state_store.py`).

## State Persistence

Fuzzer state is saved to `{corpus_dir}/state.pkl.gz` on shutdown via `core/state_store.py:StateStore`. Use `--resume` to continue. Pass `--no-save-state` to skip writing the file entirely.

Sections: `corpus` (exec counts, crash sigs, op stats, seed metadata), `edge_tracker`, `markov`, `mi`, `elo`, `ga`, `qea`, `crash_mi`, `sensitivity`, `length_tracker`, `seed_quality`, plus opt-in learners via `Fuzzer._save_learned` / `_load_learned`: `op_credit`, `burn_front`, `pll`, `wfc_tables`. Legacy per-component JSON files are auto-migrated on first `--resume` and cleaned up via `cleanup_legacy()`.

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

- Edge ID = `caller_ctx ^ prev_loc ^ cur_loc` (full 32-bit) — **no silent bucket
  collisions**. `caller_ctx` is the call-stack-sensitive term, default-on in the
  shim (`__AFL_CTX_SENSITIVE=1`; `-D__AFL_CTX_SENSITIVE=0` restores plain
  `prev_loc ^ cur_loc`), masked to `__AFL_CTX_BITS` (default 8) and advertised
  via the `__afl_ctx_bits_N` symbol for map sizing. Every shim build carries
  `-fno-omit-frame-pointer` (applied centrally by `tools/build_targets.sh`)
  because the context walk reads the caller's saved frame pointer.
- `caller_ctx` hashes a return address, which moves between processes under
  ASLR — so ids would not be comparable across executions, which is why
  `adapters/process.disable_aslr()` runs before anything spawns. When ASLR
  survives that call, the shim hashes the address relative to the load base
  instead (`FUZZER_KEEP_ASLR=1`, read in the target), and
  `services/fuzzer._ensure_ctx_ids_are_exec_stable` sets that variable. A
  build that can do this advertises `__afl_ctx_relative_capable`; an older
  target has no such symbol, ignores the variable, and gets warned about at
  startup.
- The AFLGo distance channel is also default-on (`__AFL_DISTANCE_MODE=1`;
  `=0` opts out) — inert until directed mode uploads a distance table.
- Hash: `edge_id % map_size`, linear probing for matching or empty slot
- `AFL_MAP_SIZE` is in bytes (tradition); shim divides by 8 for entry count
- Default 64KB SHM → 8192 entries (same memory as old 64KB bitmap)
- Count is a 32-bit saturating counter (no Morris probability needed)
- Python API: `ShmCoverage.get_edge_ids()`, `.get_edge_counts()`, `.read_entries()`
- `EdgeTracker.record_edges()` accepts `set[int]` (sparse) or `bytes` (legacy byte-bitmap)

## Markov Persistence

- Markov chain saved to `markov` section in `state.pkl.gz` on exit
- Loaded on init; skip retrain if loaded to avoid double-counting
- Transitions accumulate across sessions

## Scheduling Architecture

Operator selection is arbitrated by seven schedulers in `core/schedulers/` plus
Elo meta-arbitration in `core/elo.py`. The schedulers are independent — they
never import each other — and each is a self-contained bandit/optimizer over
the operator space:

| File | Class | Mechanism |
|------|-------|-----------|
| `schedulers/op_monte_carlo.py` | `MonteCarloScheduler` | Thompson sampling over operators + CEM per-position byte distribution |
| `schedulers/op_mopt.py` | `MOptScheduler` | PSO over joint operator-probability space |
| `schedulers/op_replicator.py` | `ReplicatorScheduler` | Evolutionary replicator dynamics over the operator population |
| `schedulers/op_exp3.py` | `Exp3Scheduler` | EXP3 adversarial bandit (non-stationary rewards) |
| `schedulers/op_epsilon_greedy.py` | `EpsilonGreedyScheduler` | Epsilon-greedy with exponential annealing |
| `schedulers/op_hierarchical.py` | `HierarchicalBanditScheduler` | Two-level Thompson bandit: category → operator |
| `schedulers/op_gp_ucb.py` | `GPUCBScheduler` | GP-UCB with RBF kernel over operator-category features |
| `core/elo.py` | `BayesianEloTracker` | Meta-arbitration: Thompson-samples which scheduler's `select_op` to trust (`select_strategy`), ratings persisted to `elo.json` |

- `--elo` enables Elo arbitration between whichever schedulers are enabled
  (the separate `--meta-elo` flag was consolidated into `--elo`; see `_use_elo`
  in `src/fuzzer_tool/services/fuzzer.py`). Enable `--mc-bandit`/`--mopt`/
  `--replicator`/etc. alongside `--elo` to add those strategies to the pool.
- Probabilistic selection via Thompson sampling over the Gaussian posterior
  (softmax over Elo gap, temperature=400 for the operator-level ranking).

### Position arena (third Elo tournament)

`--position-arena` (needs `--elo`) arbitrates *where* a mutation lands under
`pos_<name>` keys, disjoint from operator and seed keys
(`strategy_arena()`). Pool = uniform + every enabled proposer; uniform is
first (Elo's cold-start pick) and is the floor. A declining arm is served by
uniform and charged as uniform. Matches: each arm that served a position in
the round plays each pool member that did not, with the round's
surprisal-weighted score (`PositionArena.settle`, called from
`Fuzzer._settle_positions`). `--burn-front` adds the burn-front arm and is
credited off-policy on every round, delocalised operators excluded.

### Recording (`.record()` fan-out, fuzzer.py:2418–2478)

Every enabled scheduler records shadow stats per run, with the same success +
surprisal weight regardless of which scheduler was consulted:

- Per-scheduler `record(op, success, weight=surprisal_weight)` for bandit
  (2418), mopt (2438, gated on `not _use_elo or _meta_strategy == "mopt"`),
  replicator (2445), exp3 (2452), eps_greedy (2459), hierarchical (2466),
  gp_ucb (2473).
- `elo.record_round` (2496) for operator-level matches when ≥2 ops were used.
- `_record_operator_strategy_matches` (2507/3037) and
  `_record_seed_strategy_matches` (2512/3020) for strategy-level matches.

### Selection (`select_op()` chain, operators.py:1556–1657)

1. Stall short-circuit (1560): `_stall_recovery_active` → `random_stall`.
2. Build `available` from enabled schedulers (1564–1580): replicator, bandit,
   mopt, cem (only if `mc.cem_fitted`), exp3, eps_greedy, hierarchical, gp_ucb.
3. Meta-strategy resolution (1582–1596): with Elo on, resolve once per exec
   (cached in `_meta_strategy_cached`, re-resolved if no longer available);
   with Elo on but <2 strategies, take `available[0]`.
4. Meta-gated dispatch (1601–1631): the Elo-chosen strategy's `select_op`.
   `cem` is reachable **only** via Elo.
5. **Fallback precedence when Elo is off (1632–1656):**
   `replicator → mopt → bandit → exp3 → eps_greedy → hierarchical → gp_ucb → random`
   (first enabled scheduler wins; `cem` is not in the fallback chain).

### Elo strategy keyspaces

Operator strategies use plain keys (`replicator`, `bandit`, …); seed strategies
use `seed_<name>`-prefixed keys (`seed_ga`, `seed_weighted`, …) — registered at
fuzzer.py:991–1017. The keyspaces are disjoint and never cross-compete; the
shared tracker dicts and the shared ranking table only *look* like one group.
Seed selection must select via the prefixed keys (see below).

### Seed-side arbitration

`SeedPicker._pick_seed_elo()` (seed_picker.py:32–79) builds the eligible seed
strategies (`ga`, `qea`, `weighted`, `pareto`, `format`, `bayesian`, `markov`,
`boltzmann`, `aflgo`), exposes the pool via `_seed_strategy_pool` (so shadow
matches are only recorded against strategies that were actually selectable),
and asks `_elo.select_strategy` for the winner using the `seed_*`-prefixed keys,
then strips the prefix for downstream use (`_seed_strategy`, `strategy_map`,
convergence report).
