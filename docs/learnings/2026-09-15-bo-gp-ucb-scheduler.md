# bo-gp-ucb-scheduler: Bayesian Optimization operator scheduler

**Date:** 2026-09-15
**Context:** fuzzer-new BO-GP-UCB scheduler implementation

## Problem
A new operator scheduler was needed that uses Bayesian Optimization with Expected Improvement acquisition and noisy Gaussian Process posterior. It had to coexist with the existing `op_gp_ucb.py` scheduler rather than replace it, and wire through the full fuzzer stack (export, fuzzer service, operator dispatch, CLI, tests, docs).

## Rejected
- **Rename `op_gp_ucb.py` or reuse it in place** — the user explicitly required keeping the existing GP-UCB scheduler intact.
- **UCB-style β parameter** — EI acquisition naturally balances exploration/exploitation without a separate β knob, so the new scheduler uses Expected Improvement instead.
- **Beta priors for the GP scheduler** — the GP posterior is computed from observations, not Beta priors, so `supports_priors = False` matches the existing GP-UCB convention.
- **Direct `random` usage** — project rule requires `RandPool` for production randomness.

## Approach
- Created `src/fuzzer_tool/core/schedulers/op_bo_gp_ucb.py` using `op_gp_ucb.py`'s feature/kernel infrastructure (one-hot category features, `init_arm`, `RunningMoments`, category tracking).
- Implemented EI acquisition: `EI(op) = (μ(op) - f_max)Φ(z) + σ(op)φ(z)` with `z = (μ(op) - f_max) / σ(op)`.
- Used Cholesky decomposition for stable GP posterior inference and added `noise` as observation σ² on the kernel diagonal.
- Wired all four layers: `__init__.py` export, `fuzzer.py` registration/record/banner, `operators.py` fallback precedence/dispatch, and `commands.py` CLI flags (`--bo-gp-ucb`, `--bo-gp-length-scale`, `--bo-gp-noise`).
- Added `bo_gp_ucb` to `--elo all`, `_HAIL_MARY_FLAGS`, scheduler reachability tests, and fallback precedence tests.
- Updated `docs/DEEP_DIVE.md` and created `docs/handover/handover_bo_gp_ucb_2026-09-15.md`.

## Key insight
The tricky part was not the EI math, but the wiring consistency: the CLI must pass `bo_gp_ucb` (not `bo_ucb`) to `Fuzzer`, and the banner must check `getattr(self, "_bo_gp_ucb", False)` because the constructor stores the scheduler instance under that attribute. ImpactGuard also needs an explicit suppression when a test's `@pytest.mark.parametrize` list is intentionally extended with a new scheduler case.

## Verification
- `tests/test_regression_bo_gp_ucb.py` — 14/14 passing.
- `tests/test_commands.py` — 6/6 passing.
- `tests/test_regression_scheduler_operator_reach.py::TestAllSchedulersReachAllOperators::test_every_exported_scheduler_is_covered` — passing.
- Focused combined run: 21/21 passing.
- Full suite was run before the final pull; one pre-existing failure remained in `test_coverage_regime.py::TestCoverageRegimeDetector::test_supercritical_when_healthy` (signature mismatch unrelated to this change).
- Pre-commit hooks passed after the ImpactGuard suppression: ruff-format, ruff, and ImpactGuard.
- Commit `275da32` pushed to `origin/master`.

## Generalizes to
When adding a new scheduler, the correctness risk is usually in the cross-layer wiring rather than the scheduler algorithm itself. Keep the parameter names, attribute names, fallback precedence, dispatch chain, CLI flags, and test parametrization perfectly aligned, and treat test decorator changes as intentional API-adjacent changes that may need ImpactGuard suppression.
