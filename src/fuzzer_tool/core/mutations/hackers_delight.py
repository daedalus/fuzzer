"""Hacker's Delight rightmost-bit family and Gosper same-popcount operators.

Pure formulas and windowed mutators derived from Henry S. Warren Jr.,
Hacker's Delight (2002), Chapter 2 §2-1.

These are intended to be imported by structured.py / the operator registry.
"""

from __future__ import annotations

import random
from typing import Callable

# ---------------------------------------------------------------------------
# Pure formulas (branch-free, arbitrary-precision)
# ---------------------------------------------------------------------------

def clear_rightmost_1(x: int) -> int:
    """Turn off the rightmost 1-bit.  x & (x - 1)"""
    return x & (x - 1)


def isolate_rightmost_1(x: int) -> int:
    """Isolate the rightmost 1-bit.  x & -x"""
    return x & -x


def isolate_rightmost_0(x: int) -> int:
    """Isolate the rightmost 0-bit.  ~x & (x + 1)"""
    return (~x) & (x + 1)


def mask_trailing_zeros(x: int) -> int:
    """Mask identifying the trailing 0-bits.  ~x & (x - 1)"""
    return (~x) & (x - 1)


def right_propagate_1(x: int) -> int:
    """Right-propagate the rightmost 1-bit.  x | (x - 1)"""
    return x | (x - 1)


def clear_rightmost_run(x: int) -> int:
    """Turn off the rightmost contiguous string of 1-bits."""
    return ((x | (x - 1)) + 1) & x


def snoob(x: int) -> int:
    """Next higher number with the same population count (Gosper / snoob).

    Returns 0 when x == 0 (no successor in the natural numbers).
    """
    if x == 0:
        return 0
    smallest = x & -x
    ripple = x + smallest
    ones = x ^ ripple
    # smallest is a power of two; bit_length()-1 == trailing zeros
    shift = (smallest.bit_length() - 1) + 2
    return ripple | (ones >> shift)


def snoob_prev(x: int) -> int:
    """Previous number with the same population count.

    Linear search; a dual closed form can replace this later if needed.
    """
    if x == 0:
        return 0
    target = x.bit_count()
    y = x - 1
    while y > 0 and y.bit_count() != target:
        y -= 1
    return y


# ---------------------------------------------------------------------------
# Windowed mutators
# ---------------------------------------------------------------------------

_WORD_WIDTHS = (1, 2, 4, 8)


def _pick_window(data: bytes, rng: random.Random, min_len: int = 1) -> tuple[int, int]:
    if len(data) < min_len:
        return 0, 0
    candidates = [w for w in _WORD_WIDTHS if w <= len(data)]
    if not candidates:
        return 0, 0
    width = rng.choice(candidates)
    offset = rng.randint(0, len(data) - width)
    return offset, width


def _apply_word_op(
    data: bytes,
    rng: random.Random,
    op: Callable[[int], int],
    endian: str = "little",
) -> bytes:
    offset, width = _pick_window(data, rng)
    if width == 0:
        return data
    chunk = data[offset : offset + width]
    x = int.from_bytes(chunk, endian)
    y = op(x)
    y &= (1 << (width * 8)) - 1
    new_chunk = y.to_bytes(width, endian)
    return data[:offset] + new_chunk + data[offset + width :]


def rightmost_clear(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, clear_rightmost_1)


def rightmost_isolate(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, isolate_rightmost_1)


def rightmost_propagate(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, right_propagate_1)


def rightmost_run_clear(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, clear_rightmost_run)


def same_popcount_next(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, snoob)


def same_popcount_prev(data: bytes, rng: random.Random) -> bytes:
    return _apply_word_op(data, rng, snoob_prev)
