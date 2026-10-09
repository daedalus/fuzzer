# Architecture: Coverage, State, and Scheduling Internals

Deep details of subsystems that only matter when you are working inside them. The
entry-point AGENTS.md carries only the summary. Open this file when working on:
coverage/SHM internals, the AFL shim, `--no-shm`/`--deep-coverage` paths, the Elo
meta-scheduler, or state persistence (`state.pkl.gz` via `core/state_store.py`).

## State Persistence

Fuzzer state is saved to `{corpus_dir}/state.pkl.gz` on shutdown via `core/state_store.py:StateStore`. Use `--resume` to continue. Pass `--no-save-state` to skip writing the file entirely.

Sections: `corpus` (exec counts, crash sigs, op stats, seed metadata), `edge_tracker`, `markov`, `mi`, `elo`, `ga`, `qea`, `crash_mi`, `sensitivity`, `length_tracker`, `seed_quality`, … (grep `_state_store.set`), plus opt-in learners via `Fuzzer._save_learned` / `_load_learned`: `op_credit`, `burn_front`, `pos_fractal`, `pos_context`, `pos_levy`, `pos_harmonic`, `pos_consolidated`, `pll`, `wfc_tables`, `dict_picker`, `gravity`. Legacy per-component JSON files are auto-migrated on first `--resume` and cleaned up via `cleanup_legacy()`.

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
- Home slot (`afl_shim.c` `__afl_home_slot`): Fibonacci hash
  `((u32)(edge_id * 0x9E3779B1) * size) >> 32`, no divide. Not `%`: ctx XORs
  the id's low 8 bits, so a modulo homed all ctx variants in one block.
  Mirrored by `adapters/shm.py:home_slot`.
- Linear probing, bounded to `__AFL_PROBE_MAX` (64) slots for insert and
  lookup; wrap is a subtraction.
- `AFL_MAP_SIZE` is the entry count, not bytes.
- Segment: 32-byte front region (`SHM_TABLE_OFFSET` = `shm.py`
  `SHM_METADATA_SIZE`), then 8192 entries × 8 B by default, then 16-byte
  distance tail and optional touched-slot bitmap. Layout version
  `__AFL_SHM_LAYOUT` = 3, advertised as `__afl_shm_layout_3`.
- `count` word: high 8 bits generation tag, low 24 bits saturating hit count
  (`__afl_map_reset` advances the tag; stale entries are reclaimed in place)
- Python API: `ShmCoverage.get_edge_ids()`, `.get_edge_counts()`
- `EdgeTracker.record_edges()` accepts `set[int]` (sparse) or `bytes` (legacy byte-bitmap)
- Write guard (`--shm-write-guard`, opt-in): every shim store to the segment goes
  through `__afl_wguard_open()`/`__afl_wguard_close()`. A new write site outside
  them faults under the guard; `tests/test_shim_write_guard.py` runs every path
  in `mprotect` mode to catch that.

## Markov Persistence

- Markov chain saved to `markov` section in `state.pkl.gz` on exit
- Loaded on init; skip retrain if loaded to avoid double-counting
- Transitions accumulate across sessions

## Scheduling Architecture

Operator selection is arbitrated by the `op_*.py` schedulers in
`core/schedulers/` (core seven below) plus Elo meta-arbitration in
`core/analyzers/analyzer_elo.py`. Each is a bandit/optimizer over the operator
space; only composites (`op_consolidated*`, `op_c2ucb`, `op_kuramoto`) import
other schedulers:

| File | Class | Mechanism |
|------|-------|-----------|
| `schedulers/op_monte_carlo.py` | `MonteCarloScheduler` | Thompson sampling over operators + CEM per-position byte distribution |
| `schedulers/op_mopt.py` | `MOptScheduler` | PSO over joint operator-probability space |
| `schedulers/op_replicator.py` | `ReplicatorScheduler` | Evolutionary replicator dynamics over the operator population |
| `schedulers/op_exp3.py` | `Exp3Scheduler` | EXP3 adversarial bandit (non-stationary rewards) |
| `schedulers/op_epsilon_greedy.py` | `EpsilonGreedyScheduler` | Epsilon-greedy with exponential annealing |
| `schedulers/op_hierarchical.py` | `HierarchicalBanditScheduler` | Two-level Thompson bandit: category → operator |
| `schedulers/op_gp_ucb.py` | `GPUCBScheduler` | GP-UCB with RBF kernel over operator-category features |
| `analyzers/analyzer_elo.py` | `BayesianEloTracker` | Meta-arbitration: Thompson-samples which scheduler's `select_op` to trust (`select_strategy`), ratings persisted to the `elo` section of `state.pkl.gz` |

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

### Target arena (fourth Elo tournament)

`--target-arena` (needs `--elo` and >1 target) arbitrates which binary runs
under `tgt_<name>` keys (`services/target_arena.py`). Pool = every
`TargetSchedule` policy (`core/target_schedule.py`, impls in
`core/schedulers/tgt_base.py`) + `gale_shapley` (`tgt_gale_shapley.py`, stable
seed->target matching via `core/stable_matching.py`) + `auction`
(`tgt_auction.py`, max-weight via `core/assignment.py`); `weighted` first.
`_select_next_target` -> `TargetArena.select`; `SeedPicker.pick_seed` takes a
matching arm's seed (`target_match`, unscored); `Fuzzer._settle_targets`
(from `FuzzRound._record_arenas`) feeds
every arm and plays served-vs-rest matches.

### Recording (`.record()` fan-out, `fuzz_round.py::FuzzRound._credit_ops`)

Every round yields `(op, success, surprisal_weight)` per used op:

- Off-policy schedulers (`_record_mc`, `_record_schedulers`) record every
  round regardless of who selected.
- On-policy ones record only rounds they selected (`f._op_selector`): mopt
  and op_firefly (crediting the drawing particle, `_record_particles`), exp3,
  exp4, cmaes, corral, tsallis, exp3_ix, regret_matching, automaton.
- `_record_elo`: `elo.record_round` for operator-level matches when ≥1 op
  was used (SLOPT rounds have one).
- `_record_arenas`: `_record_operator_strategy_matches`,
  `_record_seed_strategy_matches`, `_settle_positions`, `_settle_targets`.

### Selection (`OperatorEngine.select_op`, `services/operators.py`)

1. Stall short-circuit: `_stall_recovery_active` → `random_stall`.
2. `available = operator_strategy_pool(f)`: the single ballot shared with
   `_record_operator_strategy_matches`. `cem` only if `mc.cem_fitted`.
3. Elo on, ≥2 available: resolve once per exec (cached in
   `_meta_strategy_cached`, re-resolved if no longer available); Elo on, 1
   available: `available[0]`.
4. Elo off: first of `_FALLBACK_PRECEDENCE` that is available
   (`consolidated_v2 → consolidated_v1 → replicator → mopt → bandit → exp3 → …`).
   `cem`, `invasion` and exploratory arms (canary, katz, firefly, …) are
   absent, reachable **only** via Elo.
5. The chosen strategy is stored in `f._op_selector` and its `select_op`
   dispatched.

### Elo strategy keyspaces

Operator strategies use plain keys (`replicator`, `bandit`, …); seed strategies
use `seed_<name>`-prefixed keys (`seed_ga`, `seed_weighted`, …) — names in
`services/fuzzer.py` `_OPERATOR_STRATEGY_NAMES` / `_SEED_STRATEGY_NAMES`,
pre-registered by `core/analyzer_registry.py:_activate_elo`. The keyspaces are
disjoint and never cross-compete; the
shared tracker dicts and the shared ranking table only *look* like one group.
Seed selection must select via the prefixed keys (see below).

### Seed-side arbitration

`SeedPicker._pick_seed_elo()` (`services/seed_picker.py`) builds the eligible
seed strategies (`_elo_core_arms`, `_elo_flag_arms`, `_elo_entropy_arms`,
`_elo_gated_arms`: `ga`, `qea`, `weighted`, `mcts`, `pareto`, …), exposes the
pool via `_seed_strategy_pool` (so shadow
matches are only recorded against strategies that were actually selectable),
and asks `_elo.select_strategy` for the winner using the `seed_*`-prefixed keys,
then strips the prefix for downstream use (`_seed_strategy`, `strategy_map`,
convergence report).
