"""Stochastic L-system expansion: several weighted successors per symbol.

Each symbol rewrites to one of 2-3 successors, drawn per occurrence, so one
rule set yields many different derivations of the same window::

    0 (0.5) -> 1[0]0        0 (0.5) -> 0        1 -> 11

Same window, generation count, cap and overwrite as
:mod:`fuzzer_tool.core.mutations.lsystem_expand`; successors are slices of
the window, weights are small integers.

Draws are integers, not floats: cumulative weights are scaled to one fixed
range ``SCALE`` (divisible by every possible weight total), so ``r`` in
``[0, SCALE)`` picks each successor with exactly its weight's share and the
operator stays enumerable (``ExhaustivePool``)::

    1:3 -> thresholds (SCALE/4, SCALE)    1:1 -> (SCALE/2, SCALE)
"""

from __future__ import annotations

from bisect import bisect_right
from math import lcm

from fuzzer_tool.core.mutations.lsystem_expand import (
    _BYTE,
    MAX_EXPAND,
    MAX_GENS,
    SUCC_MAX,
    UNIT_MAX,
)
from fuzzer_tool.core.mutations.structured import _splice

#: Successors per symbol: 2..MAX_BRANCH.
MAX_BRANCH = 3
#: Largest integer weight of one successor.
MAX_WEIGHT = 4

#: Draw range: every weight total (2..MAX_BRANCH * MAX_WEIGHT) divides it.
SCALE = lcm(*range(2, MAX_BRANCH * MAX_WEIGHT + 1))

#: symbol -> (cumulative thresholds ending at SCALE, successors).
Rules = dict[int, tuple[tuple[int, ...], tuple[bytes, ...]]]


def derive_stoch(axiom: bytes, rules: Rules, gens: int, cap: int, rng) -> bytes:
    """Rewrite *axiom* in parallel, picking each successor at random.

    One integer draw per symbol per generation in ``[0, SCALE)`` (symbols
    without a rule ignore theirs); the draw ``r`` picks successor
    ``bisect_right(thresholds, r)``. A generation over *cap* is not applied.

    Args:
        axiom: Start string.
        rules: Symbol -> (cumulative thresholds, successors).
        gens: Generations to apply.
        cap: Longest string allowed.
        rng: Draw source with ``randrange_list``.

    Returns:
        The last generation that fits in *cap*.
    """
    get = rules.get
    cur = axiom
    for _ in range(gens):
        draws = rng.randrange_list(SCALE, len(cur))
        parts = [
            _BYTE[b] if (rule := get(b)) is None else rule[1][bisect_right(rule[0], u)]
            for b, u in zip(cur, draws, strict=True)
        ]
        if sum(map(len, parts)) > cap:
            break
        cur = b"".join(parts)
    return cur


def _rule(axiom: bytes, rng) -> tuple[tuple[float, ...], tuple[bytes, ...]]:
    """2..MAX_BRANCH weighted successors, each a slice of *axiom*."""
    n = len(axiom)
    succs, weights = [], []
    for _ in range(rng.randint(2, MAX_BRANCH)):
        start = rng.randint(0, n - 1)
        length = rng.randint(1, SUCC_MAX)
        succs.append(axiom[start : start + length])
        weights.append(rng.randint(1, MAX_WEIGHT))

    unit = SCALE // sum(weights)
    acc, cum = 0, []
    for w in weights:
        acc += w * unit
        cum.append(acc)
    return tuple(cum), tuple(succs)


def lsystem_stochastic(data: bytes, rng) -> bytes:
    """Overwrite *data* from a random window on with a stochastic expansion.

    Args:
        data: Input buffer.
        rng: ``RandPool`` (or ScriptedRng in tests).

    Returns:
        Buffer of the same length; empty input is returned unchanged.
    """
    n = len(data)
    if not n:
        return data

    unit = rng.randint(1, min(UNIT_MAX, n))
    offset = rng.randint(0, n - unit)
    gens = rng.randint(1, MAX_GENS)
    axiom = data[offset : offset + unit]

    rules = {sym: _rule(axiom, rng) for sym in dict.fromkeys(axiom)}
    return _splice(data, offset, derive_stoch(axiom, rules, gens, MAX_EXPAND, rng))
