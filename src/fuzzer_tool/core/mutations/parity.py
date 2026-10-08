"""Per-byte parity operators: ``parity_lock`` and ``parity_break``.

Formats that carry a parity bit in every byte drop a byte whose parity is
wrong: serial 7E1/7O1 frames (bit 7), DES keys (bit 0, odd), PS/2 and
MIDI-over-UART. A random bit flip breaks parity half the time, so half the
mutants die at the first check.

    7E1:  P d6 d5 d4 d3 d2 d1 d0      carrier bit 7
    DES:  d7 d6 d5 d4 d3 d2 d1 P      carrier bit 0

    parity_lock   fix P in every byte of a window   -> window valid
    parity_break  lock, then flip one byte's P      -> exactly one error

The lock reaches the code behind the check; the break reaches the parity
error handler, which random flips hit only mixed with other errors.
"""

from enum import Enum
from functools import lru_cache

from fuzzer_tool.core.mutations.structured import _region, _splice

CARRIERS = (0, 7)  # DES LSB, serial MSB
MIN_LEN = 2  # one byte is a plain bit flip


class Parity(Enum):
    EVEN = 0
    ODD = 1


@lru_cache(maxsize=len(CARRIERS) * len(Parity))
def _table(carrier: int, parity: Parity) -> bytes:
    """256-entry translate table: flip *carrier* where parity is wrong."""
    flip = 1 << carrier
    return bytes(b ^ flip if b.bit_count() & 1 != parity.value else b for b in range(256))


def _locked(data: bytes, rng) -> tuple[int, bytearray, int] | None:
    """(offset, locked window, carrier), or None when *data* is too short."""
    offset, length = _region(len(data), rng, min_len=MIN_LEN)
    if length < MIN_LEN:
        return None

    carrier = rng.choice(CARRIERS)
    parity = rng.choice(tuple(Parity))
    block = bytearray(data[offset : offset + length].translate(_table(carrier, parity)))
    return offset, block, carrier


def parity_lock(data: bytes, rng) -> bytes:
    """Give every byte of a random window one parity (even or odd).

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` or ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    locked = _locked(data, rng)
    if locked is None:
        return data
    offset, block, _ = locked
    return _splice(data, offset, bytes(block))


def parity_break(data: bytes, rng) -> bytes:
    """Lock a window's parity, then flip the parity bit of one byte in it.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` or ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    locked = _locked(data, rng)
    if locked is None:
        return data
    offset, block, carrier = locked

    # Single error: a SECDED/parity reader takes its detect path, not correct.
    block[rng.randint(0, len(block) - 1)] ^= 1 << carrier
    return _splice(data, offset, bytes(block))
