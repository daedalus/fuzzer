"""Badness-indexed exploration floor.

`core/schedulers/op_katz.py`'s `explore_floor` (and `op_kuramoto.py`'s
copy of the same draw) is a single fixed constant: whatever the corpus
state, the scheduler guarantees each arm at least `explore_floor / n`
probability. That is a single-point guarantee. This module makes it a
*parametric family* indexed by a runtime-observed "badness" score instead:
whatever badness turns out to be, the floor is at least ``lambda(badness)``
-- the same badness-indexed-contract move
`docs/action_plan_compositional_stability.md` (P1) describes, applied to
this scheduler's own exploration floor rather than a hypothetical example.

No new detector is introduced. Badness is read from
`CoverageRegimeDetector.regime` (`core/percolation.py`'s `CoverageRegime`
-- SUBCRITICAL/CRITICAL/SUPERCRITICAL), the fuzzer's existing
percolation-phase classification: a stalled, non-percolating corpus
(SUBCRITICAL) is worst-case badness and should push the floor up (force
more exploration); a freely percolating one (SUPERCRITICAL) is best-case
and leaves the floor at its normal minimum (exploit the working chain,
`op_katz`'s whole reason for existing). See
docs/handover/handover_badness_indexed_floor_2026-09-21.md.
"""

from __future__ import annotations

from fuzzer_tool.core.percolation import CoverageRegime

# SUBCRITICAL (not percolating -- stalled) is worst-case badness;
# SUPERCRITICAL (percolating freely) is best-case. CRITICAL (the
# transition point itself) sits at the midpoint. Ordinal, not measured --
# mirrors analyzer_coverage_regime.py's own _REGIME_RANK, which makes the
# same SUBCRITICAL < CRITICAL < SUPERCRITICAL association for the same
# reason (CoverageRegime is a plain, unordered enum.Enum).
_REGIME_BADNESS: dict[CoverageRegime, float] = {
    CoverageRegime.SUBCRITICAL: 1.0,
    CoverageRegime.CRITICAL: 0.5,
    CoverageRegime.SUPERCRITICAL: 0.0,
}

DEFAULT_MAX_EXPLORE_FLOOR = 0.25


def badness_from_regime(regime: CoverageRegime | None) -> float:
    """Map a percolation-phase label to a badness score in [0, 1].

    `None` (no regime detector active, or not constructed yet) is treated
    as neutral (0.5) rather than 0.0 -- an absent signal should not
    silently pin the floor at its minimum, the same "fail toward the
    middle, not toward the extreme that looks best" discipline
    `floor_for_badness`'s caller uses for a `badness_fn` that raises.
    """
    if regime is None:
        return 0.5
    return _REGIME_BADNESS.get(regime, 0.5)


def floor_for_badness(
    base_floor: float,
    badness: float,
    max_floor: float = DEFAULT_MAX_EXPLORE_FLOOR,
) -> float:
    """lambda(badness) = base_floor + (max_floor - base_floor) * badness.

    Linear interpolation between `base_floor` (badness=0.0 -- a healthy,
    percolating regime, exploit as before) and `max_floor` (badness=1.0 --
    a stalled regime, force more exploration). `badness` is clamped to
    [0, 1] first, so an out-of-range caller cannot push the floor outside
    [base_floor, max_floor].

    Raises:
        ValueError: if `max_floor` is not in `[base_floor, 1.0)`. Every
            member of this family must satisfy the same
            ``0.0 <= floor < 1.0`` invariant `OpKatzScheduler.__init__`
            already enforces for the single-point `explore_floor` --
            validated once here, at construction time, rather than left
            to surface as a `_select_probs` division/normalization bug
            under exactly the SUBCRITICAL conditions this exists to help.
    """
    if not (0.0 <= base_floor <= max_floor < 1.0):
        raise ValueError(
            f"max_floor must be in [base_floor={base_floor!r}, 1.0), got {max_floor!r}"
        )
    b = min(1.0, max(0.0, badness))
    return base_floor + (max_floor - base_floor) * b
