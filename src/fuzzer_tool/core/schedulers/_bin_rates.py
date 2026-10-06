"""Per-seed Beta-style rates over offset bins, for a binary per-exec signal.

Shared by the ``changed`` (did the path move?) and ``rare_mask`` (was the
rare edge still hit?) position arms. Each bin of a seed holds two counts,
``n`` (credited trials) and ``s`` (credited successes), and weighs::

    w(bin) = (s + prior_a) / (n + prior_a + prior_b)

    e.g. prior 1/1:  untried 0.50   8 misses 0.10   8 hits 0.90

``propose`` draws a bin in proportion to ``w`` (one numpy cumsum, then a
uniform byte inside the bin), so no bin's weight reaches zero and untried
bins keep being explored. Binning follows ``pos_burn_front``: ``width =
ceil(len(data) / MAX_BINS)``, fixed from the parent seed; bins starting past a
shrunk live buffer are dropped.

State: two float32 arrays per seed, LRU-bounded (``MAX_SEEDS``), i.e. at most
``MAX_SEEDS * MAX_BINS * 8`` bytes (8 MiB).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import xxhash

from fuzzer_tool.core.rand_pool import RandPool

MAX_BINS = 4096  # offsets per seed are binned down to this, as pos_burn_front
MAX_SEEDS = 256  # LRU bound on per-seed tables


class _Table:
    __slots__ = ("n", "s", "width")

    def __init__(self, length: int) -> None:
        self.width = max(1, -(-length // MAX_BINS))
        bins = max(1, -(-length // self.width))
        self.n = np.zeros(bins, np.float32)
        self.s = np.zeros(bins, np.float32)


class BinRates:
    """Beta-style success rate per offset bin, per parent seed."""

    def __init__(self, rng: RandPool, prior_a: float, prior_b: float) -> None:
        self._rng = rng
        self._a = prior_a
        self._ab = prior_a + prior_b
        self._tables: OrderedDict[int, _Table] = OrderedDict()

    # -- evidence ---------------------------------------------------------------

    def credit(self, data: bytes, offsets: Sequence[int], share: float) -> None:
        """One trial per offset's bin, ``share`` success each."""
        live = [o for o in offsets if o >= 0]
        if not live or not data:
            return

        t = self._table(data, create=True)
        last = len(t.n) - 1
        for o in live:
            b = min(o // t.width, last)
            t.n[b] += 1.0
            t.s[b] += share

    def reset(self, data: bytes) -> None:
        """Forget *data*'s evidence (the signal it measured changed)."""
        self._tables.pop(xxhash.xxh3_64_intdigest(data), None)

    # -- proposal ---------------------------------------------------------------

    def weights(self, data: bytes, buf_len: int) -> np.ndarray | None:
        """Per-bin weights over bins starting inside *buf_len*; None if unseen."""
        t = self._table(data, create=False)
        if t is None:
            return None

        live = min(len(t.n), -(-buf_len // t.width))
        s = t.s[:live].astype(np.float64)
        n = t.n[:live].astype(np.float64)
        return (s + self._a) / (n + self._ab)

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """A byte in a rate-weighted bin; None when the seed has no evidence."""
        if buf_len < 1 or not data:
            return None

        w = self.weights(data, buf_len)
        if w is None or not len(w):
            return None

        cum = np.cumsum(w)
        b = int(np.searchsorted(cum, self._rng.random() * cum[-1], side="right"))
        b = min(b, len(cum) - 1)

        width = self._tables[xxhash.xxh3_64_intdigest(data)].width
        start = b * width
        span = min(width, buf_len - start)
        return start if span <= 1 else start + self._rng.randint(0, span - 1)

    # -- introspection ----------------------------------------------------------

    def counts(self, data: bytes) -> tuple[np.ndarray, np.ndarray] | None:
        t = self._tables.get(xxhash.xxh3_64_intdigest(data))
        return (t.n, t.s) if t is not None else None

    def width(self, data: bytes) -> int | None:
        """Bytes per bin for *data*'s table; None when the seed has no evidence."""
        t = self._tables.get(xxhash.xxh3_64_intdigest(data))
        return t.width if t is not None else None

    def seed_count(self) -> int:
        return len(self._tables)

    # -- storage ----------------------------------------------------------------

    def _table(self, data: bytes, *, create: bool) -> _Table | None:
        key = xxhash.xxh3_64_intdigest(data)
        t = self._tables.get(key)
        if t is not None:
            self._tables.move_to_end(key)
            return t
        if not create:
            return None

        t = self._tables[key] = _Table(len(data))
        while len(self._tables) > MAX_SEEDS:
            self._tables.popitem(last=False)
        return t
