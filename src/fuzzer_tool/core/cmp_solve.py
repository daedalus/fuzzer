"""CmpSolve: write a cmp operand over the window that half-matches it.

Ported from honggfuzz mangle.c:mangle_CmpSolve.

    input   . . M A G X . .        operand  M A G I
                ^ 3/4 bytes match
    output  . . M A G I . .        (or the operand +/-1 as an LE int)

Redqueen needs the operand's exact bytes in the input; this fires when the
input is part of the way there (a field already holding some of the magic).
gradient_cmp tweaks one differing byte; this writes the whole operand, and
+/-1 covers ``<``/``<=`` boundaries.
"""

import heapq
from itertools import groupby

# honggfuzz caps feedback values at 32 bytes.
MAX_OPERAND_LEN = 32

# Widest operand treated as an integer for +/-1; wider is a string.
MAX_INT_LEN = 8

# Variant draw, randint(0, 3): honggfuzz's 50% exact / 25% +1 / 25% -1.
VARIANT_EXACT = 0
VARIANT_PLUS = 1
VARIANT_MINUS = 2
_VARIANT_MAX = 3


def cmp_solve(data: bytes, cmp_values: list[tuple[bytes, bytes]], rng) -> bytes | None:
    """Overwrite the first partial match of a random operand.

    Args:
        data: Input bytes.
        cmp_values: (operand_a, operand_b) pairs from cmplog.
        rng: RandPool-API generator.

    Returns:
        Mutated bytes (same length), or None when the drawn operand has no
        partial window.
    """
    if not data or not cmp_values:
        return None

    pair = cmp_values[rng.randint(0, len(cmp_values) - 1)]
    val = pair[rng.randint(0, 1)]
    if len(val) > MAX_OPERAND_LEN:
        return None

    off = partial_window(data, val)
    if off is None:
        return None

    val = _vary(val, rng.randint(0, _VARIANT_MAX))
    return data[:off] + val + data[off + len(val) :]


def partial_window(data: bytes, val: bytes) -> int | None:
    """Lowest offset where 0 < matching bytes < len(val), or None.

    Candidate offsets come lazily from each operand byte's positions in
    *data* (shifted by its index), merged ascending: every candidate has at
    least one match, so the first that is not a full match is the answer.
    Example: data b"MAGIxMAGX", val b"MAGI" -> 0 is full, 5 is partial.
    """
    n = len(val)
    last = len(data) - n
    if n < 2 or last < 0:
        return None

    starts = [_offsets(data, b, i, last) for i, b in enumerate(val)]
    for off, _ in groupby(heapq.merge(*starts)):
        if data[off : off + n] != val:
            return off
    return None


def _offsets(data: bytes, byte: int, shift: int, last: int):
    """Ascending window offsets placing *byte* at index *shift*."""
    pos = data.find(byte, shift)
    while pos != -1 and pos - shift <= last:
        yield pos - shift
        pos = data.find(byte, pos + 1)


def _vary(val: bytes, variant: int) -> bytes:
    """Operand +/-1 as a wrapping little-endian int; exact otherwise."""
    n = len(val)
    if n > MAX_INT_LEN or variant not in (VARIANT_PLUS, VARIANT_MINUS):
        return val

    delta = 1 if variant == VARIANT_PLUS else -1
    return ((int.from_bytes(val, "little") + delta) % (1 << (8 * n))).to_bytes(n, "little")
