"""Shared helpers for frame-stream mutators (``ac3``, ``dca``, ``ivf``).

Bit fields are big-endian, MSB-first: bit 0 is the top bit of byte 0, the
order every A/52 / DTS header table uses.
"""

from __future__ import annotations

from enum import IntEnum

from fuzzer_tool.core.rand_pool import RandPool


class FrameOp(IntEnum):
    """Whole-frame edit that keeps every frame byte-identical."""

    DUP = 0
    DROP = 1
    SWAP = 2


def get_bits(buf: bytes | bytearray, pos: int, width: int) -> int:
    """Read *width* bits at bit offset *pos*.

    Example: ``get_bits(b"\\xb6", 1, 3) == 0b011``.
    """
    first, last = pos >> 3, (pos + width - 1) >> 3
    word = int.from_bytes(buf[first : last + 1], "big")
    shift = (last + 1) * 8 - pos - width
    return (word >> shift) & ((1 << width) - 1)


def set_bits(buf: bytearray, pos: int, width: int, value: int) -> None:
    """Write the low *width* bits of *value* at bit offset *pos*."""
    first, last = pos >> 3, (pos + width - 1) >> 3
    nbytes = last - first + 1
    shift = nbytes * 8 - (pos - first * 8) - width
    mask = ((1 << width) - 1) << shift

    word = int.from_bytes(buf[first : last + 1], "big")
    word = (word & ~mask) | ((value << shift) & mask)
    buf[first : last + 1] = word.to_bytes(nbytes, "big")


def reframe(frames: list[bytes], op: FrameOp, rng: RandPool) -> list[bytes] | None:
    """Duplicate, drop or swap frames; None when *op* needs more frames.

    Draw order: frame index, then (SWAP) a second, distinct index.
    """
    n = len(frames)
    if n == 0:
        return None
    i = rng.randint(0, n - 1)

    if op is FrameOp.DUP:
        return [*frames[: i + 1], frames[i], *frames[i + 1 :]]
    if n < 2:
        return None
    if op is FrameOp.DROP:
        return frames[:i] + frames[i + 1 :]

    # SWAP: draw from n - 1 slots and skip i, so j != i without retries.
    j = rng.randint(0, n - 2)
    j += j >= i
    out = list(frames)
    out[i], out[j] = out[j], out[i]
    return out
