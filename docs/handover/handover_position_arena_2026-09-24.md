# Handover: position schedulers + Elo position arena (2026-09-24)

## What
Position selection (where a mutation lands) formalised as a third scheduling
axis and a third Elo tournament (`pos_<name>` keys). New `burn_front`
proposer. Flags `--position-arena` (needs `--elo`), `--burn-front`; both
off by default, excluded from `--hail-mary`.

## Files
- `core/schedulers/pos_base.py` -- `PositionScheduler` protocol, `Outcome`,
  `UniformPosition`, `CallablePosition` (adapter for existing trackers).
- `core/schedulers/pos_burn_front.py` -- `BurnFrontPositionScheduler`.
- `services/position_arena.py` -- `PositionArena`, `POSITION_STRATEGY_NAMES`.
- `core/analyzers/analyzer_elo.py` -- `Arena`, `strategy_arena()`,
  `POS_STRATEGY_PREFIX`; `strategies_below_canary` filters by arena.
- `services/operators.py::select_position` -- arena branch; burn-front as a
  legacy candidate.
- `services/fuzzer.py` -- ctor params (last two), `_settle_positions`,
  canary inspection, convergence rows, banners.
- `services/stats.py`, `services/report.py` -- three-way partition,
  `top_pos`, "Position strategies (Elo)".
- `core/analyzer_registry.py` -- `pos_` keys pre-registered.

## Decisions
- `uniform` is an arm and the floor (not a worst-in-class canary).
- Decline => uniform, charged as uniform.
- Matches: served arms vs unserved pool members, per round.
- burn_front credited off-policy; `_DELOCALISED_OPS` filtered.
- Existing trackers keep their own feedback; the arena only borrows proposals.
- `Fuzzer` imports the private `_DELOCALISED_OPS` from `operators.py`
  (no visibility change; tests already do this).

## Bug fixed on the way
`strategies_below_canary("canary")` treated every non-`seed_` key as an
operator, so `pos_` keys would have been flagged against the op canary.

## Not done
- No paired benchmark; effect unknown. Plan in DEEP_DIVE.
- Burn-front state is not persisted across resume.
- `docs/architecture.dot` not changed (no new node).
