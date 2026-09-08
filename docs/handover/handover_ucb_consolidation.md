# Handover: UCB scheduler consolidation

Status: **PLAN ONLY — not implemented.** Save this file; implement when ready.

## Scope

Deduplicate six UCB-family schedulers in `src/fuzzer_tool/core/schedulers/`:
`ducb.py`, `kl_ducb.py`, `swucb.py`, `kl_swucb.py`, `cucb.py`, `gp_ucb.py`.

## Verified baseline (read-only checks done)

- `lizard --CCN=15` on all six: **0 warnings**. No function exceeds CCN 15.
- All six wired into `services/fuzzer.py`:
  - `_OPERATOR_STRATEGY_NAMES` at `fuzzer.py:86-105` includes `"gp_ucb"`, `"ducb"`, `"swucb"`, `"kl_ducb"`, `"kl_swucb"`, `"cucb"`.
  - Instantiation at `fuzzer.py:1768` (gp_ucb), `1776` (ducb), `1782` (swucb), `1791` (kl_ducb), `1799` (kl_swucb), `1809` (cucb). All share `rng=self._rand_pool` except GP-UCB.
  - `_register_arms` at `fuzzer.py:2233-2253`; calls at `2274, 2276, 2278, 2280, 2282, 2284`. All six have `supports_priors = False`.
  - `record()` loop at `fuzzer.py:4444-4462`; CUCB additionally calls `settle_round()` at `4467`.
  - `select_op()` dispatch in `services/operators.py:3651-3855` (Elo on) and `3825-3848` (Elo off).
- No scheduler imports another scheduler (`test_regression_scheduler_independence.py`).
- `bandit_stats()` is never called on any of the six by `fuzzer.py` or `operators.py`; only `self.mc.bandit_stats()` is used.
- External references to scheduler internals: only `tests/test_recency_and_combinatorial_schedulers.py` touches `_n_rel`, `_x_rel`, `_n_in_rel`, `_s_own_rel`, `_n_rounds_rel`, `_s_rounds_rel`, `_discount`, `_pending`. No production code touches them.

## Per-scheduler details (what makes each unique)

| Scheduler | `__init__` params | Width formula | Statistics | Diagnostics keys |
|---|---|---|---|---|
| DUCBScheduler | `gamma=0.9999, xi=0.6, b=1.0, exploration=0.25, rng=None` | `gaussian_scale / sqrt(n)` where `gaussian_scale = exploration * UCB_WIDTH_COEFF * b * sqrt(xi * log_n)` | `_n_rel`, `_x_rel`, `_discount` (relative to global discount) | `ducb_pulls`, `ducb_effective_n`, `ducb_arms` |
| KL_DUCBScheduler | `gamma=0.9999, xi=0.6, b=1.0, exploration=0.25, rng=None` | `kl_upper_bound(mean, xi*log_n/n) - mean` (Bernoulli KL bisection) | `_n_rel`, `_x_rel`, `_discount` | `kl_ducb_pulls`, `kl_ducb_effective_n`, `kl_ducb_arms` |
| SWUCBScheduler | `window=4000, xi=0.15, b=1.0, rng=None` | `width_scale / sqrt(n)` where `width_scale = b * sqrt(xi * log_n)` | `_history` (deque), `_counts`, `_sums`, `_known` | `swucb_pulls`, `swucb_window_fill`, `swucb_arms_in_window` |
| KL_SWUCBScheduler | `window=4000, xi=0.15, b=1.0, rng=None` | `kl_upper_bound(mean, xi*log_n/n) - mean` | `_history` (deque), `_counts`, `_sums`, `_known` | `kl_swucb_pulls`, `kl_swucb_window_fill`, `kl_swucb_arms_in_window` |
| CUCBScheduler | `gamma=0.9995, min_out_rounds=30.0, exploration=0.15, rng=None` | `radius_scale / sqrt(n_i)` where `radius_scale = exploration * sqrt(CUCB_RADIUS_COEFF * log_t)` | `_n_in_rel`, `_s_in_rel`, `_s_own_rel`, `_n_rounds_rel`, `_s_rounds_rel`, `_pending` | `cucb_rounds`, `cucb_arms`, `cucb_contrast_coverage` |
| GPUCBScheduler | `length_scale=1.0, beta=1.0, refit_interval=100, min_samples=3, kernel_floor=0.3` | `beta * sqrt(2 * log_t / n_eff)` | `_moments` (RunningMoments), `_features`, `_kernel_cache` | `gp_ucb_pulls`, `operators_tracked`, `kernel_entries` |

### Width formula detail

- DUCB/SWUCB: Gaussian `B * sqrt(xi * log(n) / n)` where B is the reward range (`self.b`). D-UCB prepends `exploration * UCB_WIDTH_COEFF` (UCB_WIDTH_COEFF = 2.0, the paper's leading coefficient). SW-UCB prepends just `b`.
- KL-DUCB/KL-SWUCB: Bernoulli KL bound via `_kl_ucb.kl_upper_bound(mean, xi*log_n/n) - mean`. Strict tightening of the Gaussian form for [0,1]-bounded rewards with mass at zero.
- CUCB: `exploration * sqrt(1.5 * log_t / n_i)` (CUCB_RADIUS_COEFF = 1.5 from Chen et al.).
- GP-UCB: `beta * sqrt(2 * log(t) / n_eff)` where `n_eff` is the arm's own count (neighbours' counts deliberately excluded from the width).

## What is common (extract to `ucb-common.py`)

Four of the six (DUCB, KL-DUCB, SWUCB, KL-SWUCB) share an identical skeleton:

1. `supports_priors = False` class attr.
2. `__init__` with validation + `self._rng = rng if rng is not None else RandPool()`.
3. `init_arm(name)` — register zero-count/reward state.
4. `select_op(ops)` — empty -> ""; single -> ops[0]; collect unpulled -> `self._rng.choice(unpulled)`; compute `log_n`; loop `mean + self._width(...)`; return best.
5. `record(name, success, weight=1.0)` — `_total_pulls += 1`, `reward = weight if success else 0.0`.
6. `bandit_stats()` returning a dict.

### Shared `__init__` structure (DUCB and KL-DUCB are identical)

```python
def __init__(self, gamma=0.9999, xi=0.6, b=1.0, exploration=0.25, rng=None):
    if not 0.0 < gamma <= 1.0: raise ValueError(...)
    if xi <= 0.0: raise ValueError(...)
    if exploration <= 0.0: raise ValueError(...)
    self.gamma = gamma; self.xi = xi; self.b = b; self.exploration = exploration
    self._rng = rng if rng is not None else RandPool()
    self._n_rel: dict[str, float] = {}
    self._x_rel: dict[str, float] = {}
    self._discount: float = 1.0
    self._total_pulls: int = 0
```

SWUCB/KL-SWUCB variant: `window` param instead of `gamma`; `_counts`/`_sums`/`_history`/`_known` instead of `_n_rel`/`_x_rel`/`_discount`.

## What is NOT common (do not touch)

- DUCB/KL-DUCB: discounted `_n_rel`/`_x_rel` + `_renormalise()` + `discounted_counts()`/`discounted_means()`.
- SWUCB/KL-SWUCB: windowed `_history`/`_counts`/`_sums` + `_evict()` + `windowed_counts()`/`windowed_means()`.
- CUCB: rounds, contrast, `settle_round()`, `_mu_hat()`, `estimated_rates()`, `contrast_coverage()`.
- GP-UCB: kernel, `RunningMoments`, `_predict()`, `kernel_matrix()`.

Each flavor keeps its own statistics, width formula, and diagnostics.

## Test coverage gaps

- **KL_DUCBScheduler and KL_SWUCBScheduler have zero test coverage** — no file imports them, no test asserts any of their behavior. Adding `UCBBase` must not break this (they remain importable under their own names).
- `test_regression_scheduler_rand_pool.py` covers DUCBScheduler, SWUCBScheduler, CUCBScheduler for Hard Rule 16 (RandPool usage). KL variants and GP-UCB are not in that test.
- `test_recency_and_combinatorial_schedulers.py::TestSharedContract` tests DUCBScheduler, SWUCBScheduler, CUCBScheduler for `supports_priors`, `select_op([]) == ""`, `select_op(["only"]) == "only"`, candidate subset, unknown-arm record, `bandit_stats()` JSON safety. KL variants and GP-UCB missing.

## Implementation plan

1. Create `src/fuzzer_tool/core/schedulers/ucb-common.py` with a `UCBBase` class holding the shared skeleton.
2. Refactor `ducb.py`, `kl_ducb.py`, `swucb.py`, `kl_swucb.py` to subclass `UCBBase` via `super()`.
3. Leave `cucb.py` and `gp_ucb.py` unchanged (their skeletons diverge too much).

## Verification

- `lizard --CCN=15` on all six: 0 warnings.
- `pytest` green.
