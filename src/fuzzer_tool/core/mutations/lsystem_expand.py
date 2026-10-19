"""L-system expansion: parallel rewriting of a window of the input.

Parsers recurse and repeat; flat-byte mutation rarely builds the deep,
self-similar, correlated structure that reaches their depth counters,
stack limits and loop-count edges.  An L-system (Lindenmayer, 1968) does:
every symbol is rewritten at once, each generation::

    A -> AB, B -> A          A, AB, ABA, ABAAB, ABAABABA, ...  (|s| = Fibonacci)
    0 -> 1[0]0, 1 -> 11      0, 1[0]0, 11[1[0]0]1[0]0, ...     (nesting)

The operator takes a short window of the input as axiom and rewrites it for
1..MAX_GENS generations with one of two rule sets:

- D0L: each distinct byte of the window gets a successor, a short slice of
  the window itself, so the output reuses the seed's own tokens;
- Fibonacci: the window is split in two chunks A, B and grown by
  ``s[k+1] = s[k] + s[k-1]`` (the locally catenative form of A->AB, B->A).

The expansion overwrites the buffer from the window on, clipped at the end,
so length is preserved like every regularity operator.  Growth is
exponential, so the expansion is capped at MAX_EXPAND bytes.
"""

from __future__ import annotations

from enum import IntEnum

from fuzzer_tool.core.mutations.structured import _splice

#: Longest window used as axiom.
UNIT_MAX = 8
#: Longest successor in a D0L rule.
SUCC_MAX = 4
#: Most generations applied.
MAX_GENS = 6
#: Longest expansion computed, in bytes.
MAX_EXPAND = 2048
#: Smallest window the Fibonacci split needs (two non-empty chunks).
_FIB_MIN_UNIT = 2


#: ``_BYTE[b]`` is the one-byte string for *b*: the identity production.
_BYTE = tuple(bytes((b,)) for b in range(256))


class Mode(IntEnum):
    """Rule-set family."""

    D0L = 0
    FIBONACCI = 1


def derive(axiom: bytes, rules: dict[int, bytes], gens: int, cap: int) -> bytes:
    """Apply *rules* to *axiom* in parallel for up to *gens* generations.

    Symbols without a rule are copied (identity production).  A generation
    that would exceed *cap* bytes is not applied.

    Args:
        axiom: Start string.
        rules: Byte value -> successor.
        gens: Generations to apply.
        cap: Longest string allowed.

    Returns:
        The last generation that fits in *cap*.
    """
    get = rules.get

    cur = axiom
    for _ in range(gens):
        parts = [get(b, _BYTE[b]) for b in cur]
        if sum(map(len, parts)) > cap:
            break
        cur = b"".join(parts)
    return cur


def _fibonacci(axiom: bytes, gens: int, cap: int) -> bytes:
    """Grow ``s[k+1] = s[k] + s[k-1]`` from the two halves of *axiom*."""
    half = len(axiom) // 2
    prev, cur = axiom[half:], axiom[:half]
    for _ in range(gens):
        if len(cur) + len(prev) > cap:
            break
        prev, cur = cur, cur + prev
    return cur


def _d0l_rules(axiom: bytes, rng) -> dict[int, bytes]:
    """One successor per distinct byte: a slice of the axiom itself."""
    n = len(axiom)
    rules: dict[int, bytes] = {}
    for sym in dict.fromkeys(axiom):
        start = rng.randint(0, n - 1)
        length = rng.randint(1, SUCC_MAX)
        rules[sym] = axiom[start : start + length]
    return rules


def lsystem_expand(data: bytes, rng) -> bytes:
    """Overwrite *data* from a random window on with its L-system expansion.

    Args:
        data: Input buffer.
        rng: ``RandPool`` (or ScriptedRng in tests).

    Returns:
        Buffer of the same length; empty input is returned unchanged.
    """
    n = len(data)
    if not n:
        return data

    mode = Mode(rng.randint(0, 1))
    unit = rng.randint(1, min(UNIT_MAX, n))
    offset = rng.randint(0, n - unit)
    gens = rng.randint(1, MAX_GENS)
    axiom = data[offset : offset + unit]

    if mode is Mode.FIBONACCI and unit >= _FIB_MIN_UNIT:
        grown = _fibonacci(axiom, gens, MAX_EXPAND)
    else:
        grown = derive(axiom, _d0l_rules(axiom, rng), gens, MAX_EXPAND)
    return _splice(data, offset, grown)
