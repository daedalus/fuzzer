"""Fast CRC folding for warm checksum models.

Port of crc-braid (github.com/marzooqy/crc-braid: zlib's braided CRC
generalised to widths 1-64) to Python + numpy. ``compute_checksum`` folds one
byte per interpreter step (~8 MiB/s); this engine folds a model it has seen
enough bytes of through one of two paths:

* **slice** (>= ``_SLICE_MIN_BYTES``): slicing-by-8, one 64-bit word and eight
  table lookups per step (~1.7x).
* **lanes** (>= ``_LANES_MIN_BYTES``): crc-braid's independent CRCs, widened to
  K numpy lanes, then merged pairwise (~10-20x)::

      data  | lane 0 | lane 1 | lane 2 | lane 3 |  each lane: L words
              crc0     crc1     crc2     crc3      (all K advance per step)
                 \\   /            \\   /
            skip(crc0)^crc1   skip(crc2)^crc3      skip = feed L*8 zero bytes
                      \\          /
                 skip2(left) ^ right  -> crc of the block

Every model is mapped onto crc-braid's unified domain: a right-shifting 64-bit
register (MSB-first registers are top-aligned and byte-swapped), so one
fold loop serves both bit orders. Byte-loop semantics stay in
``berlekamp_massey``; ``fold`` returns ``None`` whenever it declines.
"""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from collections.abc import Sequence
from enum import Enum, auto

import numpy as np


class BitOrder(Enum):
    """Shift direction of the CRC register."""

    MSB_FIRST = auto()  # non-reflected: normal-form poly, shifts left
    LSB_FIRST = auto()  # reflected: reversed-form poly, shifts right


_WORD = 8  # bytes per 64-bit word
_BYTE_MASK = 0xFF
_REG_BITS = 64
_MIN_WIDTH = 8  # byte-loop semantics below this differ (b & mask table)
_MAX_WIDTH = 64

_SLICE_MIN_BYTES = 64  # below: per-call setup outweighs the gain
_LANES_MIN_BYTES = 16384  # below: numpy per-op overhead dominates
_LANE_WORDS = 16  # words per lane per block (tuned: 64 KB-1 MB peak)
_MAX_LANES = 2048  # past this the (8, K) gathers spill the cache
_BUILD_BYTES = 8192  # cold-model bytes before tables (~1.4 ms) pay off
_MAX_ENGINES = 8  # ~0.3 MB each (slices + lane levels)
_MAX_TRACKED = 256  # cold models whose byte counts are remembered

_LITTLE_ENDIAN = sys.byteorder == "little"

# compute_checksum's reflect_in flag -> order, without a branch.
ORDER_BY_REFLECT = (BitOrder.MSB_FIRST, BitOrder.LSB_FIRST)

# Gather offsets: byte k of each lane indexes row k of a flat (8*256) table.
_SHIFTS = (np.arange(_WORD, dtype=np.uint64) * np.uint64(8))[:, None]
_OFFSETS = (np.arange(_WORD, dtype=np.uint64) * np.uint64(256))[:, None]
_NP_BYTE = np.uint64(_BYTE_MASK)
_NP_EIGHT = np.uint64(8)

# bit i of byte value b, as (256, 8) bool: expands a 64-entry basis to tables.
_BYTE_BITS = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(bool)


def _swap64(x: int) -> int:
    """Reverse the byte order of a 64-bit integer."""
    return int.from_bytes(x.to_bytes(_WORD, "little"), "big")


def _apply(flat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply a linear map stored as 8 byte tables to every lane of *v*."""
    return np.bitwise_xor.reduce(flat[((v >> _SHIFTS) & _NP_BYTE) + _OFFSETS], axis=0)


def _tables_from_basis(basis: np.ndarray) -> np.ndarray:
    """Expand basis[j] = op(1 << j) into flat 8x256 tables (XOR of set bits)."""
    cols = basis.reshape(_WORD, 1, 8)
    picked = np.where(_BYTE_BITS[None, :, :], cols, np.uint64(0))
    return np.bitwise_xor.reduce(picked, axis=2).ravel()


class CrcEngine:
    """Slice and lane tables for one ``(poly, width, order)`` model."""

    def __init__(self, table: Sequence[int], width: int, order: BitOrder) -> None:
        self._msb = order is BitOrder.MSB_FIRST
        self._shift = _REG_BITS - width

        # Byte table in the unified (right-shifting) domain.
        if self._msb:
            table = [_swap64(t << self._shift) for t in table]
        self._table = list(table)

        # slices[k][b]: byte b at word offset k, carried to the word's end.
        slices = [self._table]
        for _ in range(_WORD - 1):
            prev = slices[0]
            slices.insert(0, [(x >> 8) ^ self._table[x & _BYTE_MASK] for x in prev])
        self._slices = slices
        self._flat = np.array(slices, dtype=np.uint64).ravel()

        # Lane-merge tables: level j skips L*8*2**j zero bytes; built lazily.
        self._levels: list[np.ndarray] = []
        self._basis = self._skip_basis(_LANE_WORDS * _WORD)

    # -- public ------------------------------------------------------------

    def fold(self, reg: int, data) -> int:
        """Feed *data* into register *reg* (``compute_checksum`` convention)."""
        mv = memoryview(data).cast("B")
        n = len(mv)
        c = _swap64(reg << self._shift) if self._msb else reg

        done = 0
        if n >= _LANES_MIN_BYTES:
            c, done = self._fold_lanes(c, mv, n)
        c = self._fold_words(c, mv[done:])

        return _swap64(c) >> self._shift if self._msb else c

    # -- slice path --------------------------------------------------------

    def _fold_words(self, c: int, mv: memoryview) -> int:
        """Slicing-by-8 over whole words, then the byte tail."""
        n8 = len(mv) & ~(_WORD - 1)
        s0, s1, s2, s3, s4, s5, s6, s7 = self._slices
        for w in mv[:n8].cast("Q"):
            w ^= c
            c = (
                s0[w & 0xFF]
                ^ s1[(w >> 8) & 0xFF]
                ^ s2[(w >> 16) & 0xFF]
                ^ s3[(w >> 24) & 0xFF]
                ^ s4[(w >> 32) & 0xFF]
                ^ s5[(w >> 40) & 0xFF]
                ^ s6[(w >> 48) & 0xFF]
                ^ s7[w >> 56]
            )

        table = self._table
        for b in mv[n8:]:
            c = (c >> 8) ^ table[(c ^ b) & _BYTE_MASK]
        return c

    # -- lane path ---------------------------------------------------------

    def _fold_lanes(self, c: int, mv: memoryview, n: int) -> tuple[int, int]:
        """Fold whole K-lane blocks; return ``(register, bytes consumed)``."""
        lanes = 1 << (min(_MAX_LANES, n // (_WORD * _LANE_WORDS)).bit_length() - 1)
        words = lanes * _LANE_WORDS
        nblocks = n // (words * _WORD)
        levels = self._merge_levels(lanes.bit_length() - 1)
        blocks = np.frombuffer(mv, dtype="<u8", count=nblocks * words)
        blocks = blocks.reshape(nblocks, lanes, _LANE_WORDS)

        flat = self._flat
        for block in blocks:
            # Row j = word j of every lane, contiguous for the gather.
            rows = block.T.copy()
            r = np.zeros(lanes, dtype=np.uint64)
            r[0] = c
            for row in rows:
                r = _apply(flat, r ^ row)

            # Pairwise merge: left lane skips past the right one, then XOR.
            for tables in levels:
                r = _apply(tables, r[0::2]) ^ r[1::2]
            c = int(r[0])

        return c, nblocks * words * _WORD

    def _skip_basis(self, nbytes: int) -> np.ndarray:
        """basis[j] = register 1 << j after *nbytes* zero bytes."""
        table = np.array(self._table, dtype=np.uint64)
        v = np.uint64(1) << np.arange(_REG_BITS, dtype=np.uint64)
        for _ in range(nbytes):
            v = (v >> _NP_EIGHT) ^ table[v & _NP_BYTE]
        return v

    def _merge_levels(self, count: int) -> list[np.ndarray]:
        """First *count* merge tables; each level squares the previous skip."""
        while len(self._levels) < count:
            tables = _tables_from_basis(self._basis)
            self._levels.append(tables)
            self._basis = _apply(tables, self._basis)
        return self._levels[:count]


class _EngineCache:
    """FIFO of built engines plus byte counts of cold models (ski rental)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engines: OrderedDict[tuple, CrcEngine] = OrderedDict()
        self._seen: OrderedDict[tuple, int] = OrderedDict()

    def get(self, key: tuple, nbytes: int, table: Sequence[int]) -> CrcEngine | None:
        """Engine for *key*, built once the model has seen ``_BUILD_BYTES``."""
        # Hit path is lock-free (dict.get is atomic): it is the hot one.
        engine = self._engines.get(key)
        if engine is not None:
            return engine

        with self._lock:
            engine = self._engines.get(key)
            if engine is not None:
                return engine

            seen = self._seen.pop(key, 0) + nbytes
            if seen < _BUILD_BYTES:
                self._seen[key] = seen
                if len(self._seen) > _MAX_TRACKED:
                    self._seen.popitem(last=False)
                return None

            _poly, width, order = key
            engine = CrcEngine(table, width, order)
            self._engines[key] = engine
            if len(self._engines) > _MAX_ENGINES:
                self._engines.popitem(last=False)
            return engine

    def has(self, key: tuple) -> bool:
        with self._lock:
            return key in self._engines

    def clear(self) -> None:
        with self._lock:
            self._engines.clear()
            self._seen.clear()

    def sizes(self) -> tuple[int, int]:
        with self._lock:
            return len(self._engines), len(self._seen)


_CACHE = _EngineCache()


def fold(
    poly: int, width: int, order: BitOrder, reg: int, data, table: Sequence[int]
) -> int | None:
    """Fast-fold *data* into *reg*, or ``None`` to use the byte loop.

    Args:
        poly: Polynomial as passed to ``compute_checksum`` (cache key only).
        width: Register width in bits.
        order: Register shift direction.
        reg: Current register value, already masked to *width*.
        data: Bytes-like input.
        table: ``compute_checksum``'s 256-entry byte table for this model.
    """
    n = len(data)
    if n < _SLICE_MIN_BYTES or not _LITTLE_ENDIAN:
        return None
    if not _MIN_WIDTH <= width <= _MAX_WIDTH:
        return None

    engine = _CACHE.get((poly, width, order), n, table)
    if engine is None:
        return None
    return engine.fold(reg, data)


def cached(poly: int, width: int, order: BitOrder) -> bool:
    """True when an engine is built for this model."""
    return _CACHE.has((poly, width, order))


def clear() -> None:
    """Drop all engines and cold-model counts."""
    _CACHE.clear()


def sizes() -> tuple[int, int]:
    """``(built engines, tracked cold models)``."""
    return _CACHE.sizes()
