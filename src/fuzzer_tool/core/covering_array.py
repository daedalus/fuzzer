"""Generic t-way covering array construction (AETG-style greedy).

A covering array over parameters ``P1..Pk`` with value domains
``V1..Vk`` is a list of rows (one value per parameter) such that every
*t*-way combination of parameter values -- every ``(Pi=a, Pj=b, ...)``
for every size-``t`` subset of parameters -- appears in at least one
row. For ``t=2`` ("pairwise") this is the classic result that every
pair of field values gets exercised together at least once, in a row
count close to ``O(v^2 log k)`` rather than the ``O(v^k)`` of the full
cross product -- see ``docs/handover/handover_generators_2026-09-20.md``
(P3 candidates), which flagged this as the one genuinely missing
generator: small header fields (PNG IHDR has 7) are currently explored
by independent per-field mutation, which never guarantees a given pair
of boundary values -- e.g. ``color_type=3`` (indexed) with
``bit_depth=16`` (invalid for indexed color, per the PNG spec) -- is
ever tried together in the same input.

This module is deliberately just the combinatorics: it knows nothing
about PNG, bytes, or the fuzzer's operator model. See
``core/mutations/covering_array_mutate.py`` for the PNG IHDR operator
built on top of it.

Algorithm: standard AETG/greedy construction. Build the full set of
required ``(parameter subset, value combo)`` tuples up front; repeatedly
draw a pool of candidate rows (independent uniform-random per-field
draw from each domain), keep the candidate that covers the most
still-uncovered tuples, and remove what it covers. This is not
guaranteed minimal, but is simple, has no dependencies, and is the
textbook approach for small parameter counts -- the PNG IHDR case
(7 fields, domains of 3-10 values each) is well within where this
performs fine; nothing here has been tuned for hundreds of parameters.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from typing import Any

# One row: a value per parameter, in parameter order.
Row = tuple[Any, ...]
# A (parameter-index-subset, value-combo) pair -- one required interaction.
RequiredTuple = tuple[tuple[int, ...], tuple[Any, ...]]

# Random-candidate rows to sample per row-selection round. Higher finds
# a better (more tuples covered) row per round at proportional cost;
# 50 is the point where more candidates stopped shrinking the row count
# appreciably on the PNG IHDR domains this was built for (see
# test_covering_array.py's row-count regression bound).
_DEFAULT_CANDIDATE_POOL = 50


def _pick(rng: Any, values: Sequence[Any]) -> Any:
    """Draw one value from *values* using whatever *rng* offers.

    Accepts both ``RandPool`` (``choice`` takes ``list | tuple | bytes``)
    and stdlib ``random.Random`` (same ``choice`` signature, plus
    ``randint``) -- this module is used from both a live mutator (the
    former) and tests/tools (the latter).
    """
    values = list(values)
    if hasattr(rng, "choice"):
        return rng.choice(values)
    return values[rng.randint(0, len(values) - 1)]  # pragma: no cover - fallback


def _param_subsets(k: int, t: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(k), t))


def _required_tuples(
    value_sets: Sequence[Sequence[Any]], param_subsets: Sequence[tuple[int, ...]]
) -> set[RequiredTuple]:
    if not value_sets:
        # itertools.combinations(range(0), 0) yields one empty subset, and
        # itertools.product() over zero iterables yields one empty combo --
        # without this guard an empty domain would report one required
        # (vacuous) tuple instead of zero.
        return set()
    needed: set[RequiredTuple] = set()
    for subset in param_subsets:
        for combo in itertools.product(*(value_sets[i] for i in subset)):
            needed.add((subset, combo))
    return needed


def _row_tuples(row: Row, param_subsets: Sequence[tuple[int, ...]]) -> set[RequiredTuple]:
    return {(subset, tuple(row[i] for i in subset)) for subset in param_subsets}


def _effective_t(k: int, t: int) -> int:
    if k <= 0:
        return 0
    return max(1, min(t, k))


def required_tuple_count(value_sets: Sequence[Sequence[Any]], t: int = 2) -> int:
    """How many distinct ``(parameter subset, value combo)`` tuples *t*-way
    coverage of *value_sets* requires -- the size of the set a fully
    covering array must hit. Useful for sizing/logging, not needed to
    call ``generate``.
    """
    k = len(value_sets)
    if k == 0:
        return 0
    t = _effective_t(k, t)
    total = 0
    for subset in _param_subsets(k, t):
        prod = 1
        for i in subset:
            prod *= len(value_sets[i])
        total += prod
    return total


def generate(
    value_sets: Sequence[Sequence[Any]],
    t: int = 2,
    rng: Any = None,
    candidate_pool: int = _DEFAULT_CANDIDATE_POOL,
    max_rows: int | None = None,
) -> list[Row]:
    """Build a *t*-way covering array over *value_sets*.

    Args:
        value_sets: one non-empty sequence of candidate values per
            parameter, in parameter order. Row ``i`` of the result has
            ``row[i] in value_sets[i]``.
        t: interaction strength. ``2`` (pairwise) is the default and
            the only strength this has been validated against; values
            greater than ``len(value_sets)`` are clamped down to it.
        rng: anything with ``.choice(list)`` (``RandPool``) or
            ``.randint(a, b)`` (stdlib ``random.Random``). Defaults to
            a fresh ``random.Random()`` -- callers that need
            reproducibility across a fuzzing campaign must pass the
            fuzzer's own seeded rng.
        candidate_pool: random candidate rows considered per row
            picked; higher trades build cost for a smaller array.
        max_rows: stop early after this many rows even if some tuples
            remain uncovered. ``None`` (default) always finishes --
            for the small domains this module targets that is cheap,
            and a truncated array silently under-covers, which the
            operator built on top of this has no way to detect later.

    Returns:
        Rows covering every required tuple (unless ``max_rows`` cut
        the array short). Never raises for well-formed non-empty
        domains; a row that cannot improve coverage ends the loop
        early rather than spin, which should only be reachable via a
        broken domain (see ``value_sets[i]`` emptiness check below).
    """
    k = len(value_sets)
    if k == 0:
        return []
    for i, vs in enumerate(value_sets):
        if not vs:
            raise ValueError(f"value_sets[{i}] is empty -- no value to cover it with")

    t = _effective_t(k, t)
    rng = rng if rng is not None else random.Random()

    param_subsets = _param_subsets(k, t)
    needed = _required_tuples(value_sets, param_subsets)

    rows: list[Row] = []
    while needed:
        if max_rows is not None and len(rows) >= max_rows:
            break
        best_row: Row | None = None
        best_covered: set[RequiredTuple] | None = None
        best_gain = -1
        # As `needed` shrinks toward its last few tuples, a pool of
        # independent uniform-random rows has a real chance of missing
        # all of them (e.g. one specific pair among ~500, each row has
        # only a few-percent chance of hitting it) -- retry with fresh
        # candidates rather than settle for whatever this round drew.
        # Bounded generously; with non-empty domains this always
        # terminates in practice (each retry is an independent shot at
        # hitting `needed`, not a repeat of a failed one), the cap only
        # guards a genuine "stuck" case with a defined stopping point.
        attempts = 0
        max_attempts = 200
        while best_gain <= 0 and attempts < max_attempts:
            for _ in range(max(1, candidate_pool)):
                row = tuple(_pick(rng, vs) for vs in value_sets)
                covered = _row_tuples(row, param_subsets) & needed
                gain = len(covered)
                if gain > best_gain:
                    best_gain = gain
                    best_row = row
                    best_covered = covered
                    if best_gain == len(needed):
                        break
            attempts += 1
        if best_row is None or best_covered is None or best_gain <= 0:
            # Exhausted max_attempts without any candidate covering
            # anything still `needed`. Unreachable for well-formed
            # non-empty domains within 200*candidate_pool draws, but
            # stop rather than loop forever if it ever is.
            break
        rows.append(best_row)
        needed -= best_covered
    return rows


def verify_coverage(rows: Sequence[Row], value_sets: Sequence[Sequence[Any]], t: int = 2) -> bool:
    """True iff every required *t*-way tuple is hit by some row in *rows*."""
    return len(missing_tuples(rows, value_sets, t)) == 0


def missing_tuples(
    rows: Sequence[Row], value_sets: Sequence[Sequence[Any]], t: int = 2
) -> set[RequiredTuple]:
    """Required tuples *rows* fails to cover -- empty iff fully covering."""
    k = len(value_sets)
    t = _effective_t(k, t)
    param_subsets = _param_subsets(k, t)
    needed = _required_tuples(value_sets, param_subsets)
    covered: set[RequiredTuple] = set()
    for row in rows:
        covered |= _row_tuples(row, param_subsets)
    return needed - covered
