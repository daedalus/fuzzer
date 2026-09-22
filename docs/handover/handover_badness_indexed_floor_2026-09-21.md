# Badness-indexed exploration floor for OpKatzScheduler

## Background

P1 of `docs/action_plan_compositional_stability.md`: express one
scheduler's exploration-floor guarantee as a parametric family indexed by
corpus/coverage "badness" instead of a single fixed constant — the
fuzzer's own analogue of the compositional-stability blog post's
badness-indexed retry-budget contract (`lambda(L)` retries for whatever
`L` turns out to be, rather than one fixed retry count).

`OpKatzScheduler.explore_floor` (and `OpKuramotoScheduler`'s copy of the
same draw, see `handover_op_katz_lockin_fix_2026-09-21.md` /
`handover_op_kuramoto_lockin_fix_2026-09-21.md`) was exactly that kind of
single-point guarantee: `floor = explore_floor / n` regardless of corpus
state. This patch makes `OpKatzScheduler`'s floor a function of a live
badness signal instead, as a proof of concept — `OpKuramotoScheduler`
shares the identical floor-mixing code and is a natural fast-follow, not
done here to keep this patch reviewable on its own.

## Design

`core/badness_floor.py` (new, no dependency on `op_katz.py`):

- `badness_from_regime(regime) -> float in [0, 1]`: maps
  `core/percolation.py`'s `CoverageRegime` (SUBCRITICAL/CRITICAL/
  SUPERCRITICAL — already computed by `CoverageRegimeDetector`, no new
  signal introduced) to a badness score. SUBCRITICAL (not percolating —
  stalled) = 1.0 (worst case); SUPERCRITICAL (percolating freely) = 0.0;
  CRITICAL = 0.5. `None` (detector not yet constructed) = 0.5, neutral —
  an absent signal must not silently pin the floor at its best case.
- `floor_for_badness(base_floor, badness, max_floor) -> float`: linear
  interpolation, `base_floor` at badness=0 through `max_floor` at
  badness=1, badness clamped to `[0, 1]` first. Validates
  `base_floor <= max_floor < 1.0` eagerly and raises `ValueError`
  otherwise — every member of the family must satisfy the same
  `0 <= floor < 1` invariant `OpKatzScheduler.__init__` already enforces
  for the single-point case, checked once at construction rather than
  left to surface inside a live `_select_probs` call.

`OpKatzScheduler` gets two new, both-optional constructor arguments:
`badness_fn` (a zero-arg callable sampled fresh on every `select_op` call)
and `max_explore_floor` (default `0.25`, unused unless `badness_fn` is
set). `badness_fn=None` (the default) preserves the exact original
behavior — every existing caller and test is unaffected. A `badness_fn`
that raises falls back to the static `explore_floor` for that call only
(`_current_explore_floor`'s `except Exception` — a missed badness
observation must not take `select_op` down, mirroring the same "a monitor
failing must not destabilize the thing it's watching" discipline
`handover_kuramoto_sync_monitor_2026-09-21.md`'s wiring follows).

Fuzzer-side wiring (`services/fuzzer.py`): a new
`Fuzzer._current_scheduling_badness()` method reads `self._regime.regime`
(always constructed — `coverage_regime` is in the analyzer registry's
always-on set) through `badness_from_regime`, and `OpKatzScheduler`'s
construction site passes `badness_fn=self._current_scheduling_badness`
when `--op-katz` is on. No new CLI flag: this rides the existing
`--op-katz` flag rather than adding a separate opt-in, since it changes
that scheduler's behavior only in degree (how high the floor climbs under
a stalled corpus), not in kind.

`_op_katz` is constructed earlier in `Fuzzer.__init__` than `_regime`
(analyzer-registry's main-phase `wire_all()` runs later) — safe because
`badness_fn` is a closure/bound method evaluated lazily on each
`select_op` call, by which point `__init__` has long finished;
`_current_scheduling_badness` defends against the ordering anyway via
`getattr(self, "_regime", None)`.

## Verification

`tests/test_badness_floor.py` (new, 14 tests): the regime→badness mapping
including the `None`-is-neutral case, and the interpolation/clamping/
validation contract of `floor_for_badness`.

`tests/test_op_katz.py`, new `TestBadnessIndexedFloor` class (9 tests):
`badness_fn=None` reproduces the exact static floor; badness 0.0/1.0 hit
the two family endpoints; fresh sampling per call; a raising `badness_fn`
falls back safely; invalid `max_explore_floor` rejected at construction;
and an end-to-end `_select_probs` comparison confirming pinned badness=1.0
yields strictly higher normalized selection probability for every
never-attempted arm than badness=0.0 does (a comparative, not absolute,
assertion — the first version asserted an absolute
`raw_floor * 0.9` threshold copied from the existing static-floor test and
it under-counted: a large `max_explore_floor` dilutes more under
post-floor renormalization than the existing test's default `0.06` floor
does, the same dilution dynamic `test_floor_bounds_probability_away_from_certainty`
already exercises, just more pronounced at a bigger floor).

`tests/test_regression_badness_indexed_floor_wiring.py` (new, 7 tests):
`_op_katz` off by default; `badness_fn` wired to the real
`_current_scheduling_badness` bound method (compared via `==`, not `is` —
bound-method access creates a fresh wrapper object each time); the
SUBCRITICAL/CRITICAL/SUPERCRITICAL/missing-detector badness readings; and
an end-to-end check that flipping `f._regime._regime` from SUPERCRITICAL
to SUBCRITICAL measurably raises `_op_katz`'s own live `_select_probs`
output.

Full affected suite (`test_badness_floor.py`, `test_op_katz.py`,
`test_katz*.py`, `test_kuramoto.py`, `test_op_kuramoto.py`,
`test_regression_badness_indexed_floor_wiring.py`,
`test_regression_analyzer_registry.py`,
`test_coverage_regime*.py`, `test_percolation.py`,
`test_regression_scheduler_fallback_precedence.py`,
`test_regression_enabled_features_*`): 263/263 pass. `ruff check` clean on
every touched/new file.

## Suggested next steps

1. Apply the same `badness_fn`/`max_explore_floor` plumbing to
   `OpKuramotoScheduler.select_op` (identical floor-mixing code, same
   `core.badness_floor` helper, no new module needed).
2. A real campaign comparison: does the badness-indexed floor actually
   recover from a stall faster than the static `0.06` default, or does the
   extra exploration tax during SUBCRITICAL regimes cost more than it
   buys? Nothing here measures that — this patch only makes the
   contract's shape parametric, the way P1 of
   `docs/action_plan_compositional_stability.md` asked for; whether the
   particular `_REGIME_BADNESS` mapping and `DEFAULT_MAX_EXPLORE_FLOOR` are
   good numbers is an empirical question for a fresh sweep, not asserted
   here.
3. P2 of the same action plan (empirical 2x2 linearization of
   operator-reward vs. corpus-diversity around stall/recovery events) is
   unrelated to this patch and still open.
