# Handover — Optimization Algorithm Integrations from Wikipedia Analysis

**Date:** 2026-09-15
**Base:** `1d5a402` (Finish missing and incomplete PoC mutators: ffconcat + jpeg2000 cdef + regression tests)
**Status:** ANALYSIS ONLY — integration not yet implemented. This document catalogs what to integrate from three Wikipedia articles and where each hook lands in the existing architecture.

Sources analysed:
- https://en.wikipedia.org/wiki/Bayesian_optimization
- https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
- https://en.wikipedia.org/wiki/Integer_relation_algorithm

---

## 1. Bayesian Optimization

### Current state

The fuzzer already has significant Bayesian infrastructure:

| Component | File | Mechanism |
|-----------|------|-----------|
| `BayesianEloTracker` | `core/elo.py` | Gaussian posteriors + Thompson sampling for scheduler selection |
| `GPUCBScheduler` | `core/schedulers/op_gp_ucb.py` | GP-UCB with RBF kernel over operator-category features |
| `MonteCarloScheduler` | `core/schedulers/op_monte_carlo.py` | Thompson sampling over Beta-Bernoulli + CEM byte distributions |
| `BayesianSeedQuality` | `core/seed_quality.py` | Beta-Bernoulli posterior per seed over `P(new_coverage)` |

### Gap analysis (from Wikipedia article)

The Bayesian optimization article describes a canonical loop: **probabilistic model → acquisition function → next evaluation point**. The fuzzer has the loop skeleton but uses UCB as the acquisition function rather than the more principled Expected Improvement (EI). The article also describes noisy BO, constrained BO, and batch/parallel evaluation — none of which are implemented.

### Integration points

| # | Concept | Integration | File | Disruption |
|---|---------|-------------|------|------------|
| BO-1 | **Expected Improvement (EI)** acquisition function | Replace UCB score in `GPUCBScheduler.select_op()` with EI: `EI(x) = (f_min - μ)Φ(z) + σφ(z)` where `z = (f_min - μ)/σ`. Uses existing GP posterior; only the scoring step changes. | `core/schedulers/op_gp_ucb.py` | Low |
| BO-2 | **Noisy Gaussian Process** | Add observation-noise term to GP predictions. Relevant because crash detection is uncertain (a crash may or may not fire on a given run). Add `noise_variance` parameter to `GPUCBScheduler.__init__`. | `core/schedulers/op_gp_ucb.py` | Low |
| BO-3 | **TPE Scheduler** (Tree-structured Parzen Estimator) | New scheduler per Wikipedia "related methods" section: models density of good vs bad outcomes rather than GP posterior. New file implementing `init_arm()`, `select_op()`, `record()`, `bandit_stats()`, `supports_priors = True`. | `core/schedulers/bayesian_optimization.py` (new) | Low |
| BO-4 | **Batch/Parallel BO** | Wikipedia describes batch methods (Gonzalez 2016) for concurrent evaluations. Add batch selection to GPUCBScheduler or new scheduler via `--bo-batch` flag. | `core/schedulers/op_gp_ucb.py` or new | Medium |
| BO-5 | **Constrained acquisition** | Wikipedia: "separate probabilistic models for objective and constraints, sampling criterion combines improvement with feasibility." Integrate with existing `field_constraints.py` — don't select operators that violate format constraints. | `core/schedulers/op_gp_ucb.py` + `core/field_constraints.py` | Medium |
| BO-6 | **Hyperparameter BO for Monte Carlo** | Use EI to auto-tune `MonteCarloScheduler` parameters (`elite_frac`, `refit_interval`, `arm_decay`) instead of heuristic `_adapt_interval()` at `op_monte_carlo.py:863`. | `core/schedulers/op_monte_carlo.py` | Medium |

---

## 2. BFGS Algorithm

### Current state

The fuzzer has first-order optimization but no quasi-Newton methods:

| Component | File | Mechanism |
|-----------|------|-----------|
| `gradient_descent` operator | `core/gradient_descent.py` | Angora-style descent with arithmetic ladder (±1, ±2, ... ±128). Uses Hamming distance to cmplog operands as objective. |
| `CMAESScheduler` | `core/schedulers/op_cmaes.py` | Covariance Matrix Adaptation — rank-μ update, no Hessian approximation. |
| `MOptScheduler` | `core/schedulers/op_mopt.py` | Particle Swarm — first-order, no gradient info. |

### Gap analysis

The BFGS article describes a quasi-Newton method that approximates the Hessian via rank-2 updates using only gradient evaluations: `B_{k+1} = B_k + (y_k y_k^T)/(y_k^T s_k) - (B_k s_k s_k^T B_k^T)/(s_k^T B_k s_k)`. Key advantages: O(n²) per update (vs O(n³) for Newton), no matrix inversion needed (Sherman-Morrison formula for inverse), and L-BFGS variant for high-dimensional problems. The fuzzer's `gradient_descent` operator uses coordinate-descent-style steps; BFGS would exploit curvature information across byte positions simultaneously.

### Integration points

| # | Concept | Integration | File | Disruption |
|---|---------|-------------|------|------------|
| BFGS-1 | **BFGS descent variant** | Add `_bfgs_descent()` method to `gradient_descent.py`. Use Hamming-distance gradient over byte positions, maintain dense Hessian approximation, line search via `scipy.optimize.line_search`. Gate behind a flag or make it the default (benchmark first). | `core/gradient_descent.py` | Medium |
| BFGS-2 | **Damped BFGS** | For non-convex objectives (the fuzzing landscape is non-convex), implement the Wolfe-condition curvature check from "Further developments" section. When `s^T y ≤ 0`, apply damped update modifying `s` or `y`. | `core/gradient_descent.py` | Medium |
| BFGS-3 | **L-BFGS for hyperparameter tuning** | Use limited-memory BFGS to optimize scheduler hyperparameters (length_scale in GPUCB, pop_size in CMA-ES, etc.) based on a rolling `edges/sec` metric. Finite-difference gradient from perturbing each hyperparameter. | `core/schedulers/bfgs_tuner.py` (new) | Medium |
| BFGS-4 | **CMA-ES → BFGS hybrid** | After CMA-ES converges (σ below threshold for N generations), switch to BFGS for local refinement of operator logit vector. Gradient estimate: `delta / sigma` from CMA-ES samples. | `core/schedulers/op_cmaes.py` | Low-Medium |
| BFGS-5 | **BFGS for seed energy optimization** | Use BFGS to optimize the `SeedScorer` energy function parameters (power schedule factors). The gradient of energy w.r.t. schedule parameters can be estimated from historical seed performance data. | `core/schedules.py` | Medium |

### Key implementation note

The BFGS inverse-Hessian update (Sherman-Morrison formula) avoids matrix inversion:
```
H_{k+1} = H_k + (s_k^T y_k + y_k^T H_k y_k)(s_k s_k^T)/(s_k^T y_k)^2 - (H_k y_k s_k^T + s_k y_k^T H_k)/(s_k^T y_k)
```
This maps to the `gradient_descent` operator where `s_k` is the mutation step vector and `y_k` is the gradient difference over the Hamming distance objective.

---

## 3. Integer Relation Algorithm (PSLQ / LLL)

### Current state

The fuzzer has related but distinct algebraic machinery:

| Component | File | What it does |
|-----------|------|-------------|
| Berlekamp-Massey | `core/berlekamp_massey.py` | Finds shortest LFSR over GF(2) generating a binary sequence. Handles XOR-linear relations. |
| ChecksumLearner | `core/checksum_learner.py` | Recovers CRC polynomials and integer modulus models from data/observations. |
| `int_checksum.py` / `int_checksum_solver.py` | | Recovers integer-modulus checksums (Adler-32, Fletcher, custom) via GCD over pairwise differences. |
| `field_constraints.py` | | Hardcoded field types (length, CRC32, sum, offset) with z3-based repair. |
| Grammar/TreeMutator | `core/grammar.py` | S-expression grammar rules, no integer relation detection. |

### Gap analysis

The Wikipedia article describes algorithms that find `a_1 x_1 + a_2 x_2 + ... + a_n x_n = 0` given real numbers `x_i`. Key algorithms: LLL (Lenstra-Lenstra-Lovász 1982), HJLS (1986), PSLQ (Ferguson-Bailey 1992). The article's primary applications are: (1) determining if a number is algebraic, (2) finding relations between mathematical constants, (3) experimental mathematics for discovering closed-form expressions.

In the fuzzer context, `x_i` are byte values at specific offsets, and integer relations reveal: checksum formulas, length field relationships, offset arithmetic, structural invariants. This complements the existing GF(2) Berlekamp-Massey (which handles XOR-linear relations) by covering **integer-linear relations over Z/nZ**.

### Integration points

| # | Concept | Integration | File | Disruption |
|---|---------|-------------|------|------------|
| IR-1 | **LLL/PSLQ implementation** | New pure-Python module (no new dependencies). Implement LLL lattice basis reduction and PSLQ integer relation detection. API: `find_integer_relations(values: list[float], max_coeff: int) -> list[list[int]]`. | `core/integer_relations.py` (new) | Medium |
| IR-2 | **Format signature discovery** | Replace/surpass `_FORMAT_SNIFFERS` byte-prefix matching. Scan candidate data for linear dependencies between byte groups using LLL. | `core/operator_registry.py` + new module | Medium |
| IR-3 | **Field type inference** | Auto-discover linear constraints between field groups (checksums, lengths, offsets) beyond the hardcoded types in `field_constraints.py`. Add `Field.Kind.INTEGER_RELATION` with computed relation coefficients. | `core/field_constraints.py` | Medium-High |
| IR-4 | **Format discovery operator** | New operator `format_discover` (adaptive or format band). Runs LLL on input bytes each call; if relation found, caches it and uses it to guide subsequent mutations (e.g., maintaining the discovered invariant during mutation). | `core/operator_registry.py` + `services/operators.py` | Medium |
| IR-5 | **Grammar rule induction** | `Grammar.from_integer_relations(data)` classmethod. Creates grammar rules encoding discovered constraints (e.g., "bytes at offsets [4,5,6,7] sum to value at offsets [0,3]"). | `core/grammar.py` | Medium |
| IR-6 | **Integer relation operator** | New operator in `adaptive` or `format` band that uses PSLQ to discover and maintain format invariants during mutation. Complements `crc_learn` by discovering non-CRC invariants. | `services/operators.py` + `core/operator_registry.py` | Medium |
| IR-7 | **Complement to Berlekamp-Massey** | BM handles GF(2) (XOR-linear); integer relations handle Z/nZ. Auto-route: if BM succeeds use `_poly`, otherwise try PSLQ for integer-modulus relations. This completes the checksum learning pipeline described in `DEEP_DIVE.md` §Checkin Learning. | `core/checksum_learner.py` | Low-Medium |

### Key implementation note

PSLQ is selected over LLL for the primary implementation because:
1. PSLQ is numerically stable and polynomial-time (Wikipedia: "selected as one of the Top Ten Algorithms of the Century")
2. It finds exact integer relations from approximate real-number inputs (relevant for noisy byte data)
3. It handles the `|a_i| ≤ N` constraint naturally, which maps to "coefficient magnitude bound" for format invariants

A pure-Python reference implementation can be based on the Ferguson-Bailey algorithm (Wikipedia references: `pslq-comp-alg.pdf`).

---

## Priority Matrix

| Priority | Item | Rationale |
|----------|------|-----------|
| **P0** | BO-1 (EI acquisition) | Single-file change, lowest disruption, most principled upgrade to existing GP-UCB. Directly replaces UCB per the Bayesian optimization article's "standard reference criterion." |
| **P0** | IR-1 (LLL/PSLQ module) | Prerequisite for all IR-* items. No new dependencies, pure-Python. |
| **P0** | IR-4 (format discovery operator) | Highest-value IR item — gives the fuzzer format-awareness without manual grammar rules. |
| **P1** | BFGS-1 (BFGS descent) | Replaces/augments arithmetic ladder in gradient_descent. Expected to improve cmplog operand solving rate. |
| **P1** | BO-3 (TPE scheduler) | New scheduler, fills a gap in the algorithm portfolio (density estimation vs GP). |
| **P1** | IR-7 (complement to BM) | Completes the checksum learning pipeline; minimal disruption (auto-route fallback). |
| **P2** | BFGS-4 (CMA-ES→BFGS hybrid) | Nice-to-have for CMA-ES convergence phase. |
| **P2** | BO-2 (noisy GP) | Needed for uncertain crash detection but adds complexity to GP code. |
| **P3** | BO-4 (batch BO) | Parallel evaluation; useful but not critical. |
| **P3** | IR-5/6 (grammar/operator) | Downstream of IR-1/4. |

---

## Dependencies & Prerequisites

| Prerequisite | Check | Command |
|---|---|---|
| `scipy` available for BFGS | `python -c "import scipy.optimize; print('ok')"` | |
| `sympy` or `fpylll` for LLL/PSLQ | Check `pyproject.toml`; prefer pure-Python fallback | |
| `scipy.optimize.line_search` | Used by BFGS-1; check availability | |

---

## Related Existing Documentation

| Document | Relevance |
|----------|-----------|
| `docs/DEEP_DIVE.md` §Scheduling Intelligence | Elo arbitration, operator registry — all new schedulers register here |
| `docs/DEEP_DIVE.md` §Checksum Learning | BFGS-4 and IR-7 extend the checksum/integrity learning pipeline |
| `docs/DEEP_DIVE.md` §Thermodynamic Scheduling | BO acquisition functions relate to the exploration/exploitation tradeoff managed by simulated annealing |
| `docs/refs/architecture.md` | Scheduler interface contract: `init_arm()`, `select_op()`, `record()`, `bandit_stats()`, `supports_priors` |
| `docs/refs/bug-classes.md` §Dispatch table | Hard Rule 12: register new operators in `REGISTRY`, not legacy lists |
| `AGENTS.md` Hard Rule 12 | Same — single source of truth for operator registration |

---

## Open Questions

1. **BFGS-1 benchmark**: Should BFGS descent be benchmarked against the arithmetic ladder before replacing it? The arithmetic ladder was specifically chosen (see `gradient_descent.py` lines 22-48) to fix a stalling problem. BFGS may reintroduce it on certain operand shapes.
2. **PSLQ precision**: PSLQ requires high-precision input. For byte-level data (0-255), precision is exact. But for derived quantities (ratios, frequencies), precision loss could produce false relations. What precision threshold should be required?
3. **BO-3 TPE vs BO-1 EI**: Both address the same acquisition function problem from different angles. Should EI be added to GPUCBScheduler (BO-1) or a separate TPE scheduler created (BO-3), or both?
