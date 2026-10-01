"""Reversible-gate mutations: CNOT, CCNOT (Toffoli), CSWAP (Fredkin).

Classical reversible logic over the input's bit string. Bit ``b`` is bit
``b & 7`` of byte ``b >> 3``. Each gate is self-inverse, so a circuit is
undone by replaying it in reverse.

    CNOT  (c, t)     t ^= c                 1 bit changed at most
    CCNOT (a, b, t)  t ^= a & b             1 bit, nonlinear
    CSWAP (c, x, y)  if c: swap(x, y)       2 bits, keeps popcount

Why: a gate fires only when its control bits are set, so the edit depends
on input content (e.g. a flag bit in a tagged field). Toffoli circuits can
express any permutation of a window; a depth-k circuit still touches at
most 2k bits, sitting between ``bit_flip`` and ``feistel_scramble``.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

# Gates per circuit; keeps the edit local (<= 2 * depth bits).
_MAX_DEPTH = 4

# Same no-op retry budget as generic.bit_swap: a circuit whose controls all
# read 0 changes nothing, so redraw a few times before giving up.
_DEGENERATE_RETRIES = 4


class Gate(IntEnum):
    """Reversible gate kinds; value is the RNG draw index."""

    CNOT = 0
    CCNOT = 1
    CSWAP = 2


# Index lookup; ~9x cheaper than Gate(i) on the mutation hot path.
_GATES = tuple(Gate)


def _get(buf: bytearray, b: int) -> int:
    return (buf[b >> 3] >> (b & 7)) & 1


def _flip(buf: bytearray, b: int) -> None:
    buf[b >> 3] ^= 1 << (b & 7)


def apply_gate(buf: bytearray, gate: Gate, bits: tuple[int, ...]) -> None:
    """Apply one gate in place.

    Args:
        buf: Buffer mutated in place.
        gate: Gate kind.
        bits: ``(c, t)`` for CNOT, ``(a, b, t)`` for CCNOT, ``(c, x, y)``
            for CSWAP. Operands must be distinct.
    """
    if gate is Gate.CNOT:
        c, t = bits
        if _get(buf, c):
            _flip(buf, t)
        return

    if gate is Gate.CCNOT:
        a, b, t = bits
        if _get(buf, a) and _get(buf, b):
            _flip(buf, t)
        return

    # CSWAP: swapping two bits = flipping both when they differ.
    c, x, y = bits
    if _get(buf, c) and _get(buf, x) != _get(buf, y):
        _flip(buf, x)
        _flip(buf, y)


def _pick_ctrl(rng, nbits: int, ctrl_bytes: Sequence[int]) -> int:
    """Control bit: inside a tagged byte when given, else uniform."""
    if not ctrl_bytes:
        return rng.randint(0, nbits - 1)
    return rng.choice(ctrl_bytes) * 8 + rng.randint(0, 7)


def _pick_bits(rng, gate: Gate, nbits: int, ctrl_bytes: Sequence[int]):
    """Operands for *gate*, or None when they collide (gate skipped)."""
    ctrl = _pick_ctrl(rng, nbits, ctrl_bytes)
    if gate is Gate.CNOT:
        bits: tuple[int, ...] = (ctrl, rng.randint(0, nbits - 1))
    else:
        bits = (ctrl, rng.randint(0, nbits - 1), rng.randint(0, nbits - 1))

    if len(set(bits)) != len(bits):
        return None
    return bits


def rev_circuit(data: bytes, rng, ctrl_bytes: Sequence[int] = ()) -> bytes:
    """Apply a random depth-1..4 reversible circuit; length-preserving.

    Args:
        data: Input bytes.
        rng: RandPool-API RNG.
        ctrl_bytes: Byte offsets to draw control bits from (e.g. Weizz
            field spans). Empty means uniform over the whole input.

    Returns:
        Mutated bytes, or *data* unchanged when no gate fired.
    """
    if not data:
        return data
    nbits = len(data) * 8

    for _ in range(_DEGENERATE_RETRIES):
        out = bytearray(data)
        for _ in range(rng.randint(1, _MAX_DEPTH)):
            gate = _GATES[rng.randint(0, len(_GATES) - 1)]
            bits = _pick_bits(rng, gate, nbits, ctrl_bytes)
            if bits is None:
                continue
            apply_gate(out, gate, bits)

        if out != data:
            return bytes(out)

    return data
