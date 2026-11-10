"""Monotone bit mutations (port of zzuf's ``set`` and ``unset`` modes).

zzuf has three fuzzing modes: ``xor`` (flip), ``set`` (``byte |= mask``, bits
only go 0 -> 1) and ``unset`` (``byte &= ~mask``, bits only go 1 -> 0). The
XOR ops already here cover the first. These cover the other two:

    bit_set    set one bit that is currently 0     (zzuf set,   1 bit)
    bit_clear  clear one bit that is currently 1   (zzuf unset, 1 bit)
    or_mask    OR a 2-4 bit mask into a 1/2/4-byte window (zzuf set, a mask)

Why monotone helps: a flag or permission field is tested with ``&``, so the
useful edit is "turn this on" and never "toggle it", and 7-bit text wants bit 7
cleared, not flipped back on a byte that is already clean. A flip undoes
itself half the time on such a field; a monotone edit never does.

Unlike zzuf, which picks a bit blindly and wastes the exec when the bit is
already in the target state, every draw here is restricted to bits that will
change, so the output always differs from the input. When the byte at the
requested position has no such bit, ``bit_set``/``bit_clear`` look at the next
``SCAN_BYTES`` bytes (wrapping); ``None`` means nothing to change, and the
caller records a decline.

These are *not* XOR-linear (the effect depends on the current byte), so they
must stay out of ``operator_categories.XOR_LINEAR_OPS``.
"""

from __future__ import annotations

SCAN_BYTES = 32
MASK_WIDTHS = (1, 2, 4)
MASK_MIN_BITS = 2
MASK_MAX_BITS = 4
BITS_PER_BYTE = 8


def _bits_with_value(byte: int, value: int) -> list[int]:
    """Bit indices (0 = LSB) of *byte* currently equal to *value*."""
    return [i for i in range(BITS_PER_BYTE) if (byte >> i) & 1 == value]


def monotone_bit(data: bytes, byte_idx: int, rng, to_one: bool) -> bytes | None:
    """Set (``to_one``) or clear one bit, at or just after *byte_idx*.

    Picks uniformly among the bits that differ from the target state in the
    first byte (from *byte_idx*, wrapping, within ``SCAN_BYTES``) that has any.
    """
    n = len(data)
    if not n:
        return None
    src = 0 if to_one else 1  # the state a bit must be in to be worth editing
    start = byte_idx % n
    for step in range(min(n, SCAN_BYTES)):
        pos = (start + step) % n
        cands = _bits_with_value(data[pos], src)
        if not cands:
            continue
        bit = cands[rng.randint(0, len(cands) - 1)]
        out = bytearray(data)
        out[pos] ^= 1 << bit  # bit is known to be in `src`: XOR sets or clears it
        return bytes(out)
    return None


def bit_set(data: bytes, byte_idx: int, rng) -> bytes | None:
    """zzuf ``set``: one 0 bit becomes 1."""
    return monotone_bit(data, byte_idx, rng, True)


def bit_clear(data: bytes, byte_idx: int, rng) -> bytes | None:
    """zzuf ``unset``: one 1 bit becomes 0."""
    return monotone_bit(data, byte_idx, rng, False)


def or_mask(data: bytes, byte_idx: int, rng) -> bytes | None:
    """OR a ``MASK_MIN_BITS..MASK_MAX_BITS`` bit mask into a small window.

    The mask is drawn from the window's zero bits, so it is never a subset of
    what is already set. ``None`` when the window is all ones.
    """
    n = len(data)
    if not n:
        return None
    width = min(rng.choice(MASK_WIDTHS), n)
    start = max(0, min(byte_idx % n, n - width))

    zeros = [
        off * BITS_PER_BYTE + bit
        for off in range(width)
        for bit in _bits_with_value(data[start + off], 0)
    ]
    if not zeros:
        return None

    count = min(rng.randint(MASK_MIN_BITS, MASK_MAX_BITS), len(zeros))
    out = bytearray(data)
    for _ in range(count):
        off, bit = divmod(zeros.pop(rng.randint(0, len(zeros) - 1)), BITS_PER_BYTE)
        out[start + off] |= 1 << bit
    return bytes(out)
