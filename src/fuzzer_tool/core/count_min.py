"""Count-min sketch over uint64 keys, and a buffered per-path exec counter.

Bounded-memory frequency estimates (Cormode & Muthukrishnan 2005)::

          key ──► h_0(key) ─► row 0  [ . . 3 . . . ]
              ──► h_1(key) ─► row 1  [ . 5 . . . . ]   estimate = min(3, 5, 4)
              ──► h_2(key) ─► row 2  [ . . . . 4 . ]

Estimates never undercount; plain update overcounts by at most e*N/width
with probability >= 1 - e^-depth. Conservative update (CU) raises a key's
cells only to its new estimate, tightening the overcount.

``PathFreq`` feeds it the shim's per-exec path hash for AFLFast-style power
schedules (``--path-freq cms``): hashes go to a list buffer and
reach the sketch in one vectorised ``add`` per ``flush_at`` execs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

_DEPTH = 4
_WIDTH = 1 << 14  # 4 x 16384 x 8 B = 512 KiB
_FLUSH_AT = 1024

# splitmix64 constants (Steele et al. 2014): salt stream + finaliser.
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_MASK64 = (1 << 64) - 1
_SH30, _SH27, _SH31 = np.uint64(30), np.uint64(27), np.uint64(31)
_ONE = np.uint64(1)


class PathCount(Enum):
    """What AFLFast's n_fuzz counts (``--path-freq``)."""

    FUZZ = "fuzz"  # the seed's own fuzz_count (legacy default)
    CMS = "cms"  # execs on the seed's path, count-min estimated


class Update(Enum):
    """Cell update rule."""

    PLAIN = "plain"
    CONSERVATIVE = "conservative"


def _salts(depth: int, seed: int) -> np.ndarray:
    """One splitmix64-stream salt per row: rows hash independently."""
    return np.array([(seed + (r + 1) * _GOLDEN) & _MASK64 for r in range(depth)], dtype=np.uint64)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, in place on a fresh uint64 array."""
    x ^= x >> _SH30
    x *= _MIX1
    x ^= x >> _SH27
    x *= _MIX2
    x ^= x >> _SH31
    return x


class CountMin:
    """``depth x width`` count-min sketch over uint64 keys.

    Args:
        depth: Rows (independent hashes); failure prob e^-depth.
        width: Cells per row, a power of two; overcount <= e*N/width.
        update: ``Update.PLAIN`` or ``Update.CONSERVATIVE``.
        seed: Row-salt seed; sketches merge only with equal seeds.
    """

    def __init__(
        self,
        depth: int = _DEPTH,
        width: int = _WIDTH,
        update: Update = Update.PLAIN,
        seed: int = 0,
    ) -> None:
        if depth < 1 or width < 1 or width & (width - 1):
            raise ValueError(f"need depth >= 1 and power-of-two width, got {depth}x{width}")
        self.update = update
        self.seed = seed
        self.total = 0
        self.table = np.zeros((depth, width), dtype=np.uint64)
        self._salts = _salts(depth, seed)[:, None]
        self._mask = np.uint64(width - 1)
        # Row r's cells start at r*width in the flat view.
        self._row_base = (np.arange(depth, dtype=np.int64) * width)[:, None]
        self._add = self._add_cu if update is Update.CONSERVATIVE else self._add_plain

    @property
    def nbytes(self) -> int:
        return int(self.table.nbytes)

    def _cells(self, keys: np.ndarray) -> np.ndarray:
        """Flat table index of every (row, key): shape (depth, n)."""
        h = _mix(keys[None, :] ^ self._salts)
        return (h & self._mask).astype(np.int64) + self._row_base

    def add(self, keys: np.ndarray) -> None:
        """Count each element of *keys* (uint64, repeats allowed) once."""
        if keys.size == 0:
            return
        self._add(np.asarray(keys, dtype=np.uint64))
        self.total += int(keys.size)

    def _add_plain(self, keys: np.ndarray) -> None:
        # Unbuffered add.at sums repeats. Measured per 1024 keys: 57 us vs
        # 122 us (unique + fancy +=) vs 337 us (bincount over all cells).
        np.add.at(self.table.reshape(-1), self._cells(keys).reshape(-1), _ONE)

    def _add_cu(self, keys: np.ndarray) -> None:
        # Batch CU: each distinct key's cells rise to (old estimate + its
        # batch count). Cells only grow and each key's cells reach at least
        # its true count, so estimates stay upper bounds.
        uniq, n = np.unique(keys, return_counts=True)
        cells = self._cells(uniq)
        flat = self.table.reshape(-1)
        target = flat[cells].min(axis=0) + n.astype(np.uint64)
        np.maximum.at(flat, cells.reshape(-1), np.broadcast_to(target, cells.shape).reshape(-1))

    def estimate(self, keys: np.ndarray) -> np.ndarray:
        """Upper-bound counts of *keys* (uint64 array)."""
        keys = np.asarray(keys, dtype=np.uint64)
        return self.table.reshape(-1)[self._cells(keys)].min(axis=0)

    def count(self, key: int) -> int:
        """Upper-bound count of one key."""
        return int(self.estimate(np.array([key], dtype=np.uint64))[0])

    def merge(self, other: CountMin) -> None:
        """Add *other*'s stream into this sketch (same shape and seed)."""
        if other.table.shape != self.table.shape or other.seed != self.seed:
            raise ValueError("count-min merge needs equal depth, width and seed")
        self.table += other.table
        self.total += other.total

    def to_dict(self) -> dict[str, Any]:
        """State-store payload: plain types only.

        The table goes as raw bytes: an ndarray under the store's pickle
        protocol 5 reduces to ``numpy._core.numeric._frombuffer``, which the
        safe loader refuses, and the whole section is dropped from the save.
        """
        return {
            "update": self.update.value,
            "seed": self.seed,
            "total": self.total,
            "shape": self.table.shape,
            "table": self.table.tobytes(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CountMin:
        depth, width = data["shape"]
        cms = cls(depth=depth, width=width, update=Update(data["update"]), seed=data["seed"])
        raw = data["table"]
        if len(raw) != cms.table.nbytes:
            raise ValueError(f"count-min table is {len(raw)} B, shape needs {cms.table.nbytes}")
        cms.table[:] = np.frombuffer(raw, dtype=np.uint64).reshape(depth, width)
        cms.total = int(data["total"])
        return cms


class PathFreq:
    """Per-path exec counter: buffered ``note`` per exec, CMS behind it.

    Args:
        sketch: Backing count-min sketch.
        flush_at: Buffered hashes per vectorised ``add``; bounds the buffer.
    """

    def __init__(self, sketch: CountMin, flush_at: int = _FLUSH_AT) -> None:
        self.sketch = sketch
        self._flush_at = flush_at
        # list beats array('Q') per append: 105 vs 119 ns/exec (min of 5).
        self._buf: list[int] = []

    @property
    def pending(self) -> int:
        return len(self._buf)

    def note(self, path_hash: int) -> None:
        """Count one exec of *path_hash* (hot path: one append)."""
        buf = self._buf
        buf.append(path_hash)
        if len(buf) >= self._flush_at:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        self.sketch.add(np.array(self._buf, dtype=np.uint64))
        self._buf = []

    def count(self, path_hash: int) -> int:
        """Execs seen on *path_hash* (upper bound)."""
        self.flush()
        return self.sketch.count(path_hash)

    def counts(self, path_hashes) -> np.ndarray:
        """Vectorised ``count`` over an iterable of hashes."""
        self.flush()
        return self.sketch.estimate(np.fromiter(path_hashes, dtype=np.uint64))

    def to_dict(self) -> dict[str, Any]:
        self.flush()
        return {"sketch": self.sketch.to_dict(), "flush_at": self._flush_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PathFreq:
        return cls(CountMin.from_dict(data["sketch"]), flush_at=int(data["flush_at"]))
