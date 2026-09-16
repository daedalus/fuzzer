# Handover — BO-GP-UCB Scheduler (Expected Improvement)

**Date:** 2026-09-15
**Files:** `src/fuzzer_tool/core/schedulers/bo_gp_ucb.py`, `tests/test_regression_bo_gp_ucb.py`
**CLI flags:** `--bo-gp-ucb`, `--bo-gp-length-scale`, `--bo-gp-noise`

## What was built

`BOGPUCBScheduler` — a bandit scheduler using Bayesian Optimization with
Expected Improvement (EI) acquisition and a noisy Gaussian Process posterior.
Scaffolded from `gp_ucb.py`; does not replace it.

### Acquisition function

```
EI(op) = (μ(op) - f_max) Φ(z) + σ(op) φ(z)
z      = (μ(op) - f_max) / σ(op)
```

- Cholesky decomposition for numerically stable GP posterior inference
- `noise` parameter adds σ² to the kernel diagonal — distinguishes observation
  noise from epistemic uncertainty; higher noise → more exploration
- `supports_priors = False` — GP posterior computed from observations, not Beta priors

## Wiring (all four layers complete)

| Layer | File | Change |
|---|---|---|
| Export | `core/schedulers/__init__.py` | `BOGPUCBScheduler` import + `__all__` |
| Core | `services/fuzzer.py` | `_OPERATOR_STRATEGY_NAMES`, constructor params (`bo_gp_ucb=False`, `bo_gp_length_scale=1.0`, `bo_gp_noise=0.01`), `self._bo_gp_ucb`, `_register_arms`, record fan-out, `_track_op_effect`, banner |
| Dispatch | `services/operators.py` | `_FALLBACK_PRECEDENCE`, `operator_strategy_pool()`, dispatch chain |
| CLI | `cli/commands.py` | `--bo-gp-ucb`, `--bo-gp-length-scale`, `--bo-gp-noise` args; `_HAIL_MARY_FLAGS`; `--elo all`; both Fuzzer() constructor calls |

## Tests

`tests/test_regression_bo_gp_ucb.py` — 14 tests, all passing:
- `TestBanditInterface` — `init_arm` / `select_op` / `record` / `bandit_stats`
- `TestExpectedImprovement` — EI formula fidelity, zero when no arm can beat f_max
- `TestNoisyGP` — posterior variance ordering with/without noise
- `TestRegressionEIFidelity` — EI against analytic values for known kernels

## Open / pending

- `tools/measure_bo_gp_ucb_signal.py` — signal measurement script (referenced in DEEP_DIVE, not yet created)
- Full pytest suite: one pre-existing failure in `test_coverage_regime.py::TestCoverageRegimeDetector::test_supercritical_when_healthy` (unrelated — `CoverageRegimeDetector.observe()` signature mismatch)
