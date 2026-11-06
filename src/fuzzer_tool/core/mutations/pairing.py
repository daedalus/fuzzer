"""Pairing-function operator: N x N <-> N (Hilbert's hotel).

Two adjacent ``w``-byte fields ``(x, y)`` and one ``2w``-byte field ``z``
are the same bits read two ways. Szudzik's pairing maps ``[0, 2^b)^2``
onto ``[0, 2^2b)`` exactly, so ``pack``/``unpack`` swap the readings with
no overflow. Steps walk the plane in an order that reaches every pair:
shells (``max(x, y)`` kept inside a shell) or Cantor diagonals (``x + y``
kept inside a diagonal). Packed coordinates (tile index, row/col, tag +
length) cross their boundary only at a shell or diagonal end.

    shell order (z)        diagonal order
    0  1  4  9             0  2  5  9
    2  3  5 10             1  4  8
    6  7  8 11             3  7
   12 13 14 15             6
    (row x, col y)         (row x, col y)
"""

from __future__ import annotations

from math import isqrt

from fuzzer_tool.core.mutations.structured import _splice

ENDIANS = ("little", "big")
PAIR_WIDTHS = (1, 2, 4)
PAIR_KINDS = ("pack", "unpack", "shell_next", "shell_prev", "diag_next", "diag_prev")


def pair(x: int, y: int) -> int:
    """Szudzik pairing: ``y^2 + x`` below the diagonal, else ``x^2 + x + y``."""
    return y * y + x if x < y else x * x + x + y


def unpair(z: int) -> tuple[int, int]:
    """Inverse of :func:`pair`."""
    s = isqrt(z)
    r = z - s * s
    return (r, s) if r < s else (s, r - s)


def cantor_next(x: int, y: int) -> tuple[int, int]:
    """Next pair in Cantor order: down the diagonal, then to the next one."""
    return (x - 1, y + 1) if x else (y + 1, 0)


def cantor_prev(x: int, y: int) -> tuple[int, int] | None:
    """Previous pair in Cantor order; ``None`` at the origin."""
    if y:
        return x + 1, y - 1
    return (0, x - 1) if x else None


def _step_shell(x, y, bits, delta):
    # Wraps mod 2^2b: (UMAX, UMAX) -> (0, 0) and back.
    return unpair((pair(x, y) + delta) & ((1 << (2 * bits)) - 1))


_STEPS = {
    "shell_next": lambda x, y, bits: _step_shell(x, y, bits, 1),
    "shell_prev": lambda x, y, bits: _step_shell(x, y, bits, -1),
    "diag_next": lambda x, y, _bits: cantor_next(x, y),
    "diag_prev": lambda x, y, _bits: cantor_prev(x, y),
}


def _pack(data, offset, width, endian):
    x = int.from_bytes(data[offset : offset + width], endian)
    y = int.from_bytes(data[offset + width : offset + 2 * width], endian)
    return _splice(data, offset, pair(x, y).to_bytes(2 * width, endian))


def _unpack(data, offset, width, endian):
    x, y = unpair(int.from_bytes(data[offset : offset + 2 * width], endian))
    return _splice(data, offset, x.to_bytes(width, endian) + y.to_bytes(width, endian))


def _step(data, offset, width, endian, kind):
    x = int.from_bytes(data[offset : offset + width], endian)
    y = int.from_bytes(data[offset + width : offset + 2 * width], endian)
    bits = 8 * width
    cell = _STEPS[kind](x, y, bits)
    if cell is None or max(cell) >> bits:
        return data

    x, y = cell
    return _splice(data, offset, x.to_bytes(width, endian) + y.to_bytes(width, endian))


def pairing_fn(data: bytes, rng) -> bytes:
    """Pack, unpack or step two adjacent fields as one pairing-function code.

    Kinds (``PAIR_KINDS``): ``(x, y) -> z``, ``z -> (x, y)``, next/previous
    Szudzik cell (wraps), next/previous Cantor cell (declines past ``UMAX``
    or before the origin).

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    widths = [w for w in PAIR_WIDTHS if 2 * w <= len(data)]
    if not widths:
        return data

    width = rng.choice(widths)
    offset = rng.randint(0, len(data) - 2 * width)
    endian = rng.choice(ENDIANS)
    kind = rng.choice(PAIR_KINDS)

    # e.g. width 1: 03 07 -> pack -> 00 34 (z = 7^2 + 3 = 52).
    if kind == PAIR_KINDS[0]:
        return _pack(data, offset, width, endian)
    if kind == PAIR_KINDS[1]:
        return _unpack(data, offset, width, endian)
    return _step(data, offset, width, endian, kind)
