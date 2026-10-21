from __future__ import annotations

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

#: symbol -> (weights, successors).
Rules = dict[int, tuple[tuple[int, ...], tuple[bytes, ...]]]


def derive_stoch(axiom: bytes, rules: Rules, gens: int, cap: int, rng) -> bytes:
    """Rewrite *axiom* in parallel, picking each successor at random.

    One integer draw per symbol per generation (symbols without a rule
    ignore theirs); the draw ``r`` picks successor by cumulating weights.
    A generation over *cap* is not applied.

    Args:
        axiom: Start string.
        rules: Symbol -> (weights, successors).
        gens: Generations to apply.
        cap: Longest string allowed.
        rng: Draw source with ``randint``.

    Returns:
        The last generation that fits in *cap*.
    """
    get = rules.get
    cur = axiom
    for _ in range(gens):
        parts = []
        for b in cur:
            rule = get(b)
            if rule is None:
                parts.append(_BYTE[b])
            else:
                weights, succs = rule
                total = sum(weights)
                r = rng.randint(1, total)
                acc = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        parts.append(succs[i])
                        break
        if sum(map(len, parts)) > cap:
            break
        cur = b"".join(parts)
    return cur


def _rule(axiom: bytes, rng) -> tuple[tuple[int, ...], tuple[bytes, ...]]:
    """2..MAX_BRANCH weighted successors, each a slice of *axiom*."""
    n = len(axiom)
    succs, weights = [], []
    for _ in range(rng.randint(2, MAX_BRANCH)):
        start = rng.randint(0, n - 1)
        length = rng.randint(1, SUCC_MAX)
        succs.append(axiom[start : start + length])
        weights.append(rng.randint(1, MAX_WEIGHT))

    return tuple(weights), tuple(succs)


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
