# Handover: seed-arena round-robin scheduler (2026-09-21)

## What
Added `SeedRoundRobinScheduler` (`core/schedulers/seed_round_robin.py`), the
seed-selection counterpart of the existing `RoundRobinScheduler`
(`core/schedulers/op_round_robin.py`): deterministic cycling through the
corpus in registration order, no exploitation signal.

## Why this design
`op_round_robin` already sits in an unusual spot among the operator
schedulers: unlike `op_katz`/`op_tang`/`op_canary`, it is trusted enough to
be reachable directly (`services/operators.py`'s no-`--elo` top-precedence
list), not gated to Elo-only arbitration. The seed arena had no equivalent
— every existing seed strategy is either Elo-only (`katz`, `tang`, `mcts`,
`alphabeta`, `canary`) or a real strategy that also runs standalone
(`kruskal_count`, the entropy arms, `residual`). This scheduler gives the
seed arena the same standalone-baseline treatment `round_robin` already has
on the operator side.

## Wiring (mirrors `seed_canary.py`'s wiring points, plus the non-elo
fallback chain `seed_kruskal_count`/the entropy arms/`seed_residual` use)
- `core/schedulers/seed_round_robin.py` — new module.
- `core/schedulers/__init__.py` — export.
- `services/fuzzer.py`:
  - `seed_round_robin_scheduler=False`, the new last constructor param
    (after `seed_canary_scheduler`, following this file's established
    "new flag appended last" convention — see `test_kruskal_count.py`'s
    and `test_novelty_confirm.py`'s constructor-shape tests, updated to
    pin the new last param).
  - `_use_seed_round_robin` / `_seed_round_robin` attrs, init block
    directly after `_seed_canary`'s.
  - `record()` wired into the same off-policy outcome-recording block as
    `_seed_canary` (Elo-compatibility bookkeeping only; the scheduler's
    own selection ignores it, same as `op_round_robin`'s `record()`).
  - `"round_robin"` added to `_SEED_STRATEGY_NAMES` (Elo pre-registration).
  - Added to both "enabled features" banners (the `seeds=...` strategy
    string and `_print_enabled_features`'s "Seed selection" group).
- `services/seed_picker.py`:
  - `_pick_seed_round_robin_seed()`, modeled on `_pick_seed_canary_seed()`.
  - Added to `_pick_seed_elo`'s available pool + `strategy_map`.
  - Added to `pick_seed()`'s no-`--elo` fallback chain (alongside
    `entropy_kl`/`entropy_zscore`/`entropy_deviation`/`entropy_gradient`/
    `residual`) since it needs no arbiter to run.
- `cli/commands.py`:
  - `--seed-round-robin-scheduler` flag (mirrors `--seed-canary-scheduler`'s
    naming).
  - Kwarg pass-through in `cmd_fuzz`.
  - Added to `_HAIL_MARY_FLAGS` (no `_EXCLUDED_OPT_IN` entry needed in
    `test_regression_hail_mary_gates.py` — that test derives everything
    from source, so adding the tuple entry alone satisfies it).

## Tests
- `tests/test_seed_round_robin.py` (new, 25 tests): scheduler unit tests
  (cycling order, JIT registration, registration-order persistence across
  a shrinking candidate set, record()/bandit_stats() Elo-compat
  bookkeeping, record() not affecting selection), `SeedPicker` wiring
  (Elo pool eligibility + dispatch, no-elo fallback dispatch and ordering,
  empty/disabled corpus handling), and `Fuzzer` constructor/CLI wiring.
- Updated `test_kruskal_count.py::test_constructor_flag_is_appended_last`
  and `test_novelty_confirm.py::test_constructor_flag_defaults_off_and_is_not_last`
  to point at the new last param (`seed_round_robin_scheduler`) instead of
  `seed_canary_scheduler`.
- Updated `test_regression_scheduler_independence.py`'s
  `TestSeedSchedulerExports` to cover the new export.
- Updated `test_regression_scheduler_operator_reach.py`'s
  `test_every_exported_scheduler_is_covered` exclusion set (it picks
  seeds, not operators — no `select_op(ops)` surface to drive).

## Verification
Targeted sweep (7 directly touched/modified files + broader seed-picker/
scheduler/regression files, 456 tests total across two runs) — all
passing, no regressions:
- `tests/test_seed_round_robin.py` + the 6 modified files: 179 passed.
- Broader seed-picker/scheduler sweep (`test_seed_picker*`,
  `test_seed_entropy_*`, `test_seed_residual.py`, `test_canary_scheduler.py`,
  `test_softmax_scheduler.py`, `test_topk_scheduler.py`,
  `test_tang_scheduler.py`, `test_regression_elo_all.py`): 277 passed.
- `tests/test_regressions.py`: 79 passed.
- `ruff check` clean on every touched/new file (the one repo-wide ruff
  finding, in `cli/commands.py::cmd_ppmd`'s unrelated import block, is
  pre-existing — confirmed via `git stash`).
- `mypy` strict clean on the new module (not in the exemption list, so it
  is held to the same strict bar as any new module in this repo). Run
  against numpy 1.26 in the sandbox — the sandbox's numpy 2.5 stubs hit an
  unrelated mypy/Python-version parse error repo-wide, not caused by this
  change and not present against an older numpy.

Not run: the full test suite (per the user's request to run only affected
tests) and the real-target integration/`--hail-mary` smoke run (no binary
target available in this sandbox); `test_apply_hail_mary_sets_every_listed_flag`
in `test_regression_hail_mary_gates.py` exercises the `--hail-mary` flag
end-to-end at the argparse level and passed.
