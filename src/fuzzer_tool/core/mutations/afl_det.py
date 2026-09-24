"""AFL's deterministic stages, index-addressable (T1-1).

AFL's ``fuzz_one`` sweeps every seed through a fixed sequence and keeps
the coverage-new outputs as queue seeds. Here the sequence is a pure map
``k -> variant``, so a caller can walk it one step per call from a cursor
and let a bandit decide how often that is worth it::

    flip1 | flip4 | flip8 | flip32 | arith8 | arith16 | arith32 | interest8 | interest16 | interest32
    8n      8n-3    n       n-3      n*2A     (n-1)*4A  (n-3)*4A  n*I8        (n-1)*2*I16  (n-3)*2*I32

``A = ARITH_MAX`` (+1, -1, +2, -2, ...), ``Ix`` = len of the repo's
``INTERESTING_x``; 16/32-bit stages do little-endian then big-endian.
Bit order is AFL's: bit 0 is the MSB of byte 0. The in-loop deterministic
stage (``services/operators._deterministic_mutation_stream``) only has
flip1, flip8, arith8 and interest8; this is the full-width sweep.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from fuzzer_tool.core.mutations.generic import (
    ARITH_MAX,
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
)

__all__ = ["ARITH_MAX", "STAGES", "det_total", "det_variant", "stage_sizes"]

# Two endiannesses per multi-byte stage.
_ENDIANS = ("little", "big")

# +delta / -delta per ARITH_MAX step.
_ARITH_SLOTS = 2 * ARITH_MAX


class Stage(NamedTuple):
    """One deterministic pass: ``sites(n)`` positions x ``per_site`` variants."""

    name: str
    sites: Callable[[int], int]
    per_site: int
    apply: Callable[[bytearray, int, int], None]


def _flip_bits(buf: bytearray, pos: int, nbits: int) -> None:
    for b in range(pos, pos + nbits):
        buf[b >> 3] ^= 0x80 >> (b & 7)


def _flip_bytes(buf: bytearray, pos: int, width: int) -> None:
    for i in range(pos, pos + width):
        buf[i] ^= 0xFF


def _add(buf: bytearray, pos: int, width: int, endian: str, delta: int) -> None:
    mask = (1 << (8 * width)) - 1
    val = int.from_bytes(buf[pos : pos + width], endian)
    buf[pos : pos + width] = ((val + delta) & mask).to_bytes(width, endian)


def _arith(width: int) -> Callable[[bytearray, int, int], None]:
    """j -> (endian, +/-(step)); little-endian block first for width > 1."""

    def apply(buf: bytearray, pos: int, j: int) -> None:
        endian = _ENDIANS[j // _ARITH_SLOTS]
        slot = j % _ARITH_SLOTS
        delta = (slot >> 1) + 1
        _add(buf, pos, width, endian, -delta if slot & 1 else delta)

    return apply


def _interest(width: int, values: list[int]) -> Callable[[bytearray, int, int], None]:
    mask = (1 << (8 * width)) - 1

    def apply(buf: bytearray, pos: int, j: int) -> None:
        endian = _ENDIANS[j // len(values)]
        buf[pos : pos + width] = (values[j % len(values)] & mask).to_bytes(width, endian)

    return apply


def _endians(width: int) -> int:
    return 1 if width == 1 else len(_ENDIANS)


# Site counts are in bits for the bit-walk stages, bytes otherwise.
STAGES: tuple[Stage, ...] = (
    Stage("flip1", lambda n: 8 * n, 1, lambda b, p, j: _flip_bits(b, p, 1)),
    Stage("flip4", lambda n: max(0, 8 * n - 3), 1, lambda b, p, j: _flip_bits(b, p, 4)),
    Stage("flip8", lambda n: n, 1, lambda b, p, j: _flip_bytes(b, p, 1)),
    Stage("flip32", lambda n: max(0, n - 3), 1, lambda b, p, j: _flip_bytes(b, p, 4)),
    Stage("arith8", lambda n: n, _ARITH_SLOTS, _arith(1)),
    Stage("arith16", lambda n: max(0, n - 1), _ARITH_SLOTS * _endians(2), _arith(2)),
    Stage("arith32", lambda n: max(0, n - 3), _ARITH_SLOTS * _endians(4), _arith(4)),
    Stage("interest8", lambda n: n, len(INTERESTING_8), _interest(1, INTERESTING_8)),
    Stage(
        "interest16",
        lambda n: max(0, n - 1),
        len(INTERESTING_16) * _endians(2),
        _interest(2, INTERESTING_16),
    ),
    Stage(
        "interest32",
        lambda n: max(0, n - 3),
        len(INTERESTING_32) * _endians(4),
        _interest(4, INTERESTING_32),
    ),
)


def stage_sizes(n: int) -> list[int]:
    """Variant count per stage for an *n*-byte input, in ``STAGES`` order."""
    return [s.sites(n) * s.per_site for s in STAGES]


def det_total(n: int) -> int:
    """Length of the full sweep for an *n*-byte input."""
    return sum(stage_sizes(n))


def det_variant(data: bytes, k: int) -> bytes | None:
    """The *k*-th output of the sweep over *data*, or None past its end.

    May equal *data* (e.g. an interesting value already present); callers
    that must not waste an execution skip those.
    """
    if k < 0:
        return None

    n = len(data)
    for stage in STAGES:
        size = stage.sites(n) * stage.per_site
        if k >= size:
            k -= size
            continue

        buf = bytearray(data)
        stage.apply(buf, k // stage.per_site, k % stage.per_site)
        return bytes(buf)
    return None
