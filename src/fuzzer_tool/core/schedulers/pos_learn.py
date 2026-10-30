"""PositionLearnScheduler: LearnAFL's per-seed format mask.

Port of LearnAFL (Yue et al., "Greybox Fuzzing with Knowledge Enhancement",
IEEE Access 2019). A mutant that keeps the seed's path shows which bytes the
path does not depend on. Bytes never changed in such a mutant are *frozen*:
the format the path needs::

    seed     M A G I C 00 00 00 L E N ...
    kept     . . . . . x  x  x  . . . ...     (x: mutated, path unchanged)
    frozen   F F F F F .  .  .  F F F ...

Two modes per seed, as in LearnAFL's ``use_model`` 1 / 2:

    EXPLORE  mask still changing: propose frozen bytes, which tests them
             (a kept path unfreezes the byte; a moved one confirms it)
    EXPLOIT  STABLE_ROUNDS credited rounds without a new mutable byte:
             propose mutable bytes, and ``restore`` copies the frozen ones
             back after the round, so the mutant keeps the format

On the first switch to EXPLOIT the frozen runs go to the auto-dictionary
(``emit_tokens``, ``core/auto_dict.py::frozen_tokens``).

LearnAFL logged every mutant to disk and re-learned in a Python subprocess
every 1024 execs; here the mask updates in place from each settled round's
mutated offsets and ``moved(data)`` (``Fuzzer._path_moved``), at no extra
exec. Offsets are parent coordinates, so length-changing operators blur the
mask as in every arm. Only the first MAX_BYTES of a seed are learned.

Off-policy extra: credited every settled round whoever served. Declines
before a seed's first credited round and when no site lies inside the live
buffer. State: one bool per byte per seed, LRU-bounded (``MAX_SEEDS``), at
most 16 MiB. Not persisted.
"""

from __future__ import annotations

import enum
from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np

from fuzzer_tool.core.auto_dict import frozen_tokens
from fuzzer_tool.core.hashing import hash64_len
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

MAX_BYTES = 1 << 16  # learned prefix per seed
MAX_SEEDS = 256  # LRU bound on per-seed masks
STABLE_ROUNDS = 64  # credited rounds without a new mutable byte -> EXPLOIT
# restore(): fancy-index the frozen side up to this many offsets, else copy the
# whole seed and put the mutable bytes back (measured: 1.0 vs 1.2 us at 64 B,
# 195 vs 3.9 us at 64 KiB with 50 mutable bytes).
INDEX_RESTORE_MAX = 128


class Mode(enum.Enum):
    """Which side of the mask a seed's proposals land on."""

    EXPLORE = "explore"  # frozen bytes
    EXPLOIT = "exploit"  # mutable bytes, frozen ones restored


class _Mask:
    __slots__ = ("harvested", "mutable", "sites", "stable")

    def __init__(self, length: int) -> None:
        self.mutable = np.zeros(min(length, MAX_BYTES), bool)
        self.stable = 0
        self.harvested = False
        self.sites: dict[Mode, np.ndarray] = {}

    @property
    def mode(self) -> Mode:
        return Mode.EXPLOIT if self.stable >= STABLE_ROUNDS else Mode.EXPLORE

    def offsets(self, mode: Mode) -> np.ndarray:
        """Sorted offsets of *mode*'s side; cached until the mask changes."""
        got = self.sites.get(mode)
        if got is None:
            side = self.mutable if mode is Mode.EXPLOIT else ~self.mutable
            got = self.sites[mode] = np.flatnonzero(side)
        return got


class PositionLearnScheduler:
    """Propose on the frozen or mutable side of a learned path mask."""

    name = "learn"

    def __init__(
        self,
        rng: RandPool,
        moved: Callable[[bytes], bool | None],
        emit_tokens: Callable[[list[bytes]], None],
    ) -> None:
        self._rng = rng
        self._moved = moved
        self._emit = emit_tokens
        self._masks: OrderedDict[int, _Mask] = OrderedDict()

    def propose(self, data: bytes, buf_len: int) -> int | None:
        m = self._masks.get(hash64_len(data))
        if m is None or buf_len < 1:
            return None

        # Sorted offsets: the ones inside a shrunk buffer are a prefix.
        sites = m.offsets(m.mode)
        k = int(np.searchsorted(sites, buf_len))
        if k == 0:
            return None

        return int(sites[self._rng.randint(0, k - 1)])

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Kept path -> mutated bytes become mutable; a new one resets stability."""
        live = [o for o in offsets if 0 <= o < min(len(data), MAX_BYTES)]
        if not live:
            return

        moved = self._moved(data)
        if moved is None:
            return

        m = self._mask(data)
        idx = np.asarray(live, np.intp)
        if not moved and not m.mutable[idx].all():
            m.mutable[idx] = True
            m.sites.clear()
            m.stable = 0
            return

        m.stable += 1
        if m.stable >= STABLE_ROUNDS and not m.harvested:
            m.harvested = True
            self._emit(frozen_tokens(data, m.mutable))

    def restore(self, data: bytes, buf: bytearray) -> None:
        """EXPLOIT only: copy *data*'s frozen bytes into same-length *buf*."""
        m = self._masks.get(hash64_len(data))
        if m is None or m.mode is not Mode.EXPLOIT or len(buf) != len(data):
            return

        frozen = m.offsets(Mode.EXPLORE)
        if not len(frozen):
            return

        # View released before return: a live export would block buf resizes.
        view = np.frombuffer(buf, np.uint8)
        seed = np.frombuffer(data, np.uint8)
        if len(frozen) <= INDEX_RESTORE_MAX:
            view[frozen] = seed[frozen]
        else:
            # Learned prefix back from the seed, then the mutable bytes over it.
            n = len(m.mutable)
            mutable = m.offsets(Mode.EXPLOIT)
            keep = view[mutable]
            view[:n] = seed[:n]
            view[mutable] = keep
        del view

    # -- introspection ----------------------------------------------------------

    def mode(self, data: bytes) -> Mode | None:
        """*data*'s mode; None before its first credited round."""
        m = self._masks.get(hash64_len(data))
        return None if m is None else m.mode

    def mutable(self, data: bytes) -> np.ndarray | None:
        """*data*'s mutable-byte mask (tests, stats); None when unseen."""
        m = self._masks.get(hash64_len(data))
        return None if m is None else m.mutable

    def seed_count(self) -> int:
        return len(self._masks)

    # -- storage ----------------------------------------------------------------

    def _mask(self, data: bytes) -> _Mask:
        key = hash64_len(data)
        m = self._masks.get(key)
        if m is not None:
            self._masks.move_to_end(key)
            return m

        m = self._masks[key] = _Mask(len(data))
        while len(self._masks) > MAX_SEEDS:
            self._masks.popitem(last=False)
        return m
