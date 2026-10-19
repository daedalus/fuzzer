"""Context-sensitive L-system expansion: ``left < a > right -> successor``.

A production fires only when the symbol sits between the given neighbours,
so the rewrite depends on where in the structure a byte is -- the shape of
field-boundary and delimiter-adjacent mutations::

    b < a > c -> aa        ...bac...  ->  ...baac...

Neighbours are read from the string of the previous generation (parallel
rewriting); a string end reads as ``EDGE``.  A symbol with no matching rule
is copied.  Same window, generation count, cap and overwrite as
:mod:`fuzzer_tool.core.mutations.lsystem_expand`; each rule is cut from the
window itself, so it fires at least once in generation 0.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.lsystem_expand import (
    _BYTE,
    MAX_EXPAND,
    MAX_GENS,
    SUCC_MAX,
    UNIT_MAX,
)
from fuzzer_tool.core.mutations.structured import _splice

#: Neighbour value at either end of the string.
EDGE = -1
#: Most context rules per call.
MAX_CTX_RULES = 3

#: (left, symbol, right) -> successor.
Rules = dict[tuple[int, int, int], bytes]


def derive_ctx(axiom: bytes, rules: Rules, gens: int, cap: int) -> bytes:
    """Apply context-sensitive *rules* in parallel for up to *gens* generations.

    Args:
        axiom: Start string.
        rules: (left, symbol, right) -> successor; ``EDGE`` at string ends.
        gens: Generations to apply.
        cap: Longest string allowed.

    Returns:
        The last generation that fits in *cap*.
    """
    get = rules.get
    cur = axiom
    for _ in range(gens):
        left = (EDGE, *cur[:-1])
        right = (*cur[1:], EDGE)
        parts = [get(t, _BYTE[t[1]]) for t in zip(left, cur, right, strict=True)]
        if sum(map(len, parts)) > cap:
            break
        cur = b"".join(parts)
    return cur


def _ctx_rules(axiom: bytes, rng) -> Rules:
    """1..MAX_CTX_RULES rules keyed on contexts present in *axiom*."""
    n = len(axiom)
    rules: Rules = {}
    for _ in range(rng.randint(1, MAX_CTX_RULES)):
        pos = rng.randint(0, n - 1)
        start = rng.randint(0, n - 1)
        length = rng.randint(1, SUCC_MAX)
        left = axiom[pos - 1] if pos else EDGE
        right = axiom[pos + 1] if pos + 1 < n else EDGE
        rules[(left, axiom[pos], right)] = axiom[start : start + length]
    return rules


def lsystem_ctx(data: bytes, rng) -> bytes:
    """Overwrite *data* from a random window on with a context-sensitive expansion.

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

    return _splice(data, offset, derive_ctx(axiom, _ctx_rules(axiom, rng), gens, MAX_EXPAND))
