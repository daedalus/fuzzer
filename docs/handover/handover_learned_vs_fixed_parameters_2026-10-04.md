# Handover: Learned vs Fixed Parameters in Daedalus Fuzzer (2026-10-04)

## What

Analysis of which parameters in `daedalus/fuzzer` (fuzzer-tool) are **learned** (adapted online from execution feedback and persisted across `--resume`), which are **fixed** (set at campaign start or hard-coded constants), and which of the fixed ones are natural candidates to become learnable.

The fuzzer is deliberately information-dense: operator, seed, position and target selection are driven by independent bandits/optimizers under four disjoint Elo tournaments, plus value-level models (CEM, Markov, WFC, etc.). Almost everything that influences *which* action is chosen is already learned; the remaining fixed knobs are mostly meta-parameters ("how aggressively / how large / how often").

## Architecture summary (relevant bits)

Four disjoint Elo arenas (never play each other) live in `core/analyzers/analyzer_elo.py`:

| Arena    | Key prefix | Question answered                  |
|----------|------------|------------------------------------|
| Operator | (bare / `op_`) | Which mutator?                   |
| Seed     | `seed_`    | Which corpus entry?                |
| Position | `pos_`     | Where does the mutation land?      |
| Target   | `tgt_`     | Which binary (multi-target)?       |

Meta-arbitration among enabled schedulers is itself Elo-driven (`--elo`). State is stored in `{corpus}/state.pkl.gz` via `core/state_store.py` (legacy per-component JSON files are auto-migrated). Opt-in learners are explicitly saved/restored by `Fuzzer._save_learned` / `_load_learned` in `services/fuzzer.py`.

Key entry points:
- Operator schedulers: `core/schedulers/op_*.py` (common interface: `init_arm`, `select_op`, `record`, `bandit_stats`, `to_dict`/`from_dict`)
- Position schedulers: `core/schedulers/pos_*.py`
- Seed schedulers: `core/schedulers/seed_*.py`
- Elo: `core/analyzers/analyzer_elo.py` (`EloTracker`, default `k_factor=16`, temperature=400)
- Persistence: `core/state_store.py` + `Fuzzer._save_learned` / `_load_learned`

## Learned parameters (adapted online + persisted)

These change during a campaign from rewards (new edges, crashes, surprisal, timeouts, etc.) and are restored on `--resume`.

### Operator selection
- Beta posteriors (α, β) per operator — Thompson sampling (`MonteCarloScheduler` / `--mc-bandit`). Defaults to uniform prior `Beta(1,1)`; `arm_decay` (default 0.999) applied periodically.
- Elo ratings + pairwise match outcomes (`EloTracker.record_match` / `record_reward`). Separate crash-rating track. Default `k_factor=16`.
- MOpt particle swarm state (positions, velocities, induced distribution) — `--mopt`.
- Replicator dynamics population shares / fitness — `--replicator`.
- EXP3 / EXP3-IX / EXP4 weights — `--exp3` etc.
- CMA-ES mean, step-size σ, covariance over operator logits — `--cma-es` (pop-size=8, generation-size=200, step-size=0.3, elite-frac=0.5 are the *fixed* hyper-parameters of this learner).
- Hierarchical bandit (category → operator) posteriors.
- Many other bandit variants (Bayes-UCB, GP-UCB, BO-GP-UCB, KL-DUCB, SW-UCB, CUSUM-UCB, Corral, Firefly, FPL, etc.) keep their own arm statistics, sliding windows, and internal models.
- Second-order / pairwise operator-transition chains and blend coefficients (when non-zero).
- Ant-colony pheromone matrix + Laplace success-rate heuristic (`op_ant_colony.py`; defaults α=1.0, β=4.0).
- Credit / Shapley-style attribution values when those modules are active.
- Op-credit state (explicitly in `_save_learned`).

### Position selection
- All `pos_*` proposers maintain their own statistics or models (saliency/NEUZZ-style, burn-front, cmplog, boundary, fractal, lineage, Levy, consolidated, good-Turing, etc.).
- Position-arena Elo ratings (`pos_<name>` keys).
- Explicitly persisted via `_save_learned`: `pos_fractal`, `pos_context`, `pos_levy`, `pos_consolidated`, `burn_front`.

### Seed / corpus quality
- Seed-strategy Elo ratings (`seed_*` keys).
- Multiple independent scorers: entropy variants (KL, LOO, Shapley, z-score, residual, deviation, gradient), MCTS, MLFQ, CoDel, BFQ, SFQ, DRR, EEVDF, Kruskal-count, Good-Turing, residual, AIMD, etc.
- Length tracker, sensitivity maps, crash mutual information, occupation / transfer-entropy measures.
- Strata ledger + seed arm (restored via `_load_strata`).

### Value-level / generative models
- CEM per-position byte distributions (`--mc-cem`). Elite set, JS-divergence-driven adaptive refit interval, optional Dirichlet concentration. Defaults: `elite_frac=0.1`, `refit_interval=1000`.
- Markov-chain transition tables (`--markov` / `--markov-gen`). Transitions learned; order itself is a fixed CLI knob (default 1).
- Auto-dictionary / dict-picker Thompson state.
- WFC tables (`wfc_tables`), PLL state, gravity state — all in `_save_learned`.
- Format learner tables (when `--learn-format`).

### Execution / feedback adaptation
- Adaptive-havoc stage lengths (on by default; `--no-adaptive-havoc` disables).
- Adaptive timeout (opt-in `--adaptive-timeout`) — retunes per-execution timeout from observed runtimes.
- Temperature-control / ESO bandwidth, badness-floor, coverage-noise models, edge-matrix refits.
- Running moments (Welford/Pébay mean/var/skew/kurtosis) used for reward calibration, Brier scores, Sharpe/Kelly blending, etc.
- Dict-picker and gravity state (persisted).

### Explicitly saved by `_save_learned` (services/fuzzer.py ~4878)
```
op_credit, burn_front, pos_fractal, pos_context, pos_levy,
pos_consolidated, pll, wfc_tables, dict_picker, gravity
```
(Plus the bulk state sections: corpus, edge_tracker, markov, mi, elo, ga, qea, crash_mi, sensitivity, length_tracker, seed_quality, strata, …)

## Fixed parameters (set once or hard-coded)

### Common CLI defaults (from `cli/commands.py`)
| Parameter              | Default     | Notes |
|------------------------|-------------|-------|
| `--max-len`            | 4096        | |
| `--timeout`            | 1.0 s       | Adaptive path exists |
| `--mutations`          | 8           | Havoc intensity |
| Markov order           | 1           | |
| Dirichlet α mode       | FIXED       | Learned mode available |
| CMA-ES pop-size        | 8           | |
| CMA-ES generation-size | 200         | |
| CMA-ES step-size       | 0.3         | |
| CMA-ES elite-frac      | 0.5         | |
| pairwise-blend         | 0.0         | |
| second-order-blend     | 0.0         | |
| sharpe-kelly-blend     | 0.0         | |
| hierarchical-pooling   | 0.0         | MonteCarlo |
| Elo enabled            | False       | `--elo` / `--hail-mary` |
| Coverage               | True        | |

### Hard-coded algorithmic constants (examples)
- Elo: `k_factor=16.0`, temperature=400.0 (`analyzer_elo.py`)
- MonteCarloScheduler: `elite_frac=0.1`, `refit_interval=1000`, `arm_decay=0.999`, `MIN_BETA_PARAM=1e-6`, uniform prior `(1.0,1.0)`
- Ant colony: α=1.0, β=4.0
- Softmax floors, minimax discount/block (0.5 / 0.3), UCB min-samples base=20, edge-matrix refit=2000, campaign-ledger decay=0.05, covering-array candidate pool=50, etc.
- Coverage map size / context bits / SHM entry layout (shim)
- The operator registry itself (148 operators, 9 categories) and the set of enabled schedulers

## Fixed parameters that are learnable (candidates)

These are currently static but have clear online signals and fit the existing bandit / running-stats / Elo infrastructure:

| Fixed parameter | Signal already available | Existing partial support | Suggested direction |
|-----------------|--------------------------|---------------------------|---------------------|
| Timeout | Observed runtime distribution | `--adaptive-timeout` | Make default or tighten the retuning |
| Mutations / havoc intensity | Success rate of different stage lengths | Adaptive-havoc (default on) | Expose as continuous arm or meta-bandit |
| Max input length | Lengths that produce new coverage / crashes | Length tracker | Adaptive upper bound or length-biased sampling already partially present |
| CEM/MOpt/CMA-ES hyper-parameters (pop, σ, elite frac, refit interval, inertia) | JS divergence, generation quality, Brier | JS-adaptive CEM refit already exists | Promote more of them to adaptive |
| Elo `k_factor` & temperature | Reward variance, kurtosis, calibration | Temperature-control analyzer exists | Anneal or make reward-dependent |
| Bandit prior strength / Dirichlet concentration | Observed variance, Brier score | Running moments + Brier computed | Adaptive prior strength |
| Blend coefficients (pairwise, 2nd-order, Sharpe-Kelly, hierarchical pooling) | Marginal gain when non-zero | Currently static (often 0) | Treat as continuous arms or optimize by meta-bandit |
| Markov order | Predictive power / information criterion | Transitions already learned | Small discrete bandit over order |
| Power-schedule constants (AFL-style / AFLGo) | Classic online-tuning targets | Some schedules already implemented | Meta-bandit over schedule family + parameters |

## Files to read first
- `docs/ARCHITECTURE.md` — canonical layer model
- `docs/refs/architecture.md` — coverage, state, scheduling internals
- `core/analyzers/analyzer_elo.py` — EloTracker + four arenas
- `core/schedulers/op_monte_carlo.py` — Thompson + CEM reference implementation
- `core/schedulers/op_cmaes.py` — CMA-ES operator scheduler
- `services/fuzzer.py` — `_save_learned` / `_load_learned`, feature banners, settlement
- `core/state_store.py` — persistence contract
- `cli/commands.py` — all CLI defaults and flag wiring
- `docs/handover/handover_position_arena_2026-09-24.md` — recent position-arena work (example of the same style)

## Not done / open questions
- No quantitative A/B of making the “learnable fixed” set adaptive; the infrastructure is ready but effect sizes are unknown.
- Some learners (PLL restore path, certain pos_* state) are restored only when the corresponding analyzer is activated — check activation order on resume.
- Architecture diagram (`docs/architecture.dot`) and runtime banner can lag the code; treat `docs/ARCHITECTURE.md` as authoritative.

## How to continue
1. When adding a new scheduler, follow the existing `select_op`/`record`/`bandit_stats` + `to_dict`/`from_dict` interface and register it in the appropriate strategy name table.
2. Any new learned state that must survive `--resume` should either go into a main state section or be wired into `_save_learned` / `_load_learned`.
3. Prefer promoting an existing fixed hyper-parameter to adaptive (using the signals already computed) over inventing a parallel mechanism.