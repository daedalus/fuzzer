"""PositionKadaneScheduler: mutate inside the seed's maximum-excess-gain run.

Every other bin-rate arm (``pos_changed``, ``pos_rare_mask``, ``pos_burn_front``)
draws a *bin* in proportion to its own rate. A block, splice or fill operator
does not touch one byte, it rewrites a contiguous window, and the best window
is not the union of the best bins: one dead bin in the middle of two hot ones
breaks the picture, and a lone lucky bin outweighs a wide, mildly hot stretch.
Picking the window is the maximum-sum-subarray problem (``core/kadane.py``).

Evidence is the same group test ``pos_changed`` uses, on the gain signal every
arm is credited with: a round that gained credits each offset's bin with
``1 / k`` of a success, a miss with 0 (``core/schedulers/_bin_rates.py``). The
score of a bin is its *excess* over what the seed's pooled rate predicts::

    p0   = sum(s) / sum(n)                      pooled gain rate of this seed
    x[b] = s[b] - p0 * n[b]                     excess successes in bin b

so a bin that gains at the pooled rate scores 0, an untried bin scores exactly 0
(no prior to strip out, and a gap of untried bins costs a window nothing), and
``sum(x) == 0`` over the whole seed, which makes the best window's total >= 0.
Kadane over ``x`` returns the contiguous run of bins with the most gain above
expectation; ``propose`` lands uniformly inside it.

Declines (``None``) while the seed has no gain yet (``p0 == 0``: every score is
0, there is no run to pick) and when no run beats ``MIN_EXCESS`` (noise: a single
early gain always makes a run of excess ``1 - p0``, which is evidence for
nothing). ``EXPLORE`` of proposals are uniform escapes so a window that stopped
paying cannot trap the arm.

Window API: ``window()`` returns the run as ``(offset, length)`` in bytes, the
shape ``core/mutations/structured._region`` returns. It is not wired into the
region operators (that is a 40-call-site change); the arm only proposes offsets.

Off-policy extra: credited every settled round whoever served. Per-seed state is
the ``BinRates`` table (LRU-bounded) plus a cached window, recomputed only after a
new credit. Not persisted, like ``changed`` and ``rare_mask``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import xxhash

from fuzzer_tool.core.kadane import max_subarray
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import MAX_SEEDS, BinRates
from fuzzer_tool.core.schedulers.pos_base import Outcome

PRIOR_A = 1.0  # BinRates prior; unused here (scores come from raw counts) but required
PRIOR_B = 1.0
MIN_EXCESS = 1.0  # a run must hold at least one whole gain above expectation
EXPLORE = 0.10  # uniform escapes regardless of the window
WINDOW_P = 0.5  # share of windowed-operator calls handed the window (rest stay random)


class PositionKadaneScheduler:
    """Offsets from the contiguous run of bins with the most gain above the pooled rate."""

    name = "kadane"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        self._rates = BinRates(rng, PRIOR_A, PRIOR_B)
        # seed hash -> [credits when computed, (live bins, run or None)]; run = (first, end) bins
        self._cache: OrderedDict[int, list] = OrderedDict()
        self._credits: OrderedDict[int, int] = OrderedDict()

    # -- protocol ---------------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None

        win = self.window(data, buf_len)
        if win is None:
            return None

        if self._rng.random() < EXPLORE:
            return self._rng.randint(0, buf_len - 1)

        offset, length = win
        return offset if length <= 1 else offset + self._rng.randint(0, length - 1)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Pooled test: miss -> 0 for all offsets; gain -> ``1/k`` each.

        ``weight`` is accepted for protocol parity and not used: the score is a
        count of gain rounds, and weighting them would make ``p0`` mean something
        other than a rate.
        """
        live = [o for o in offsets if o >= 0]
        if not live or not data:
            return

        self._rates.credit(data, live, 1.0 / len(live) if outcome is Outcome.GAIN else 0.0)
        key = xxhash.xxh3_64_intdigest(data)
        self._credits[key] = self._credits.get(key, 0) + 1
        self._credits.move_to_end(key)
        while len(self._credits) > MAX_SEEDS:
            self._credits.popitem(last=False)

    # -- window -----------------------------------------------------------------

    def window(self, data: bytes, buf_len: int) -> tuple[int, int] | None:
        """``(offset, length)`` in bytes of the best run inside *buf_len*; None if none."""
        if buf_len < 1 or not data:
            return None

        width = self._rates.width(data)
        if width is None:
            return None

        live = -(-buf_len // width)
        key = xxhash.xxh3_64_intdigest(data)
        credits = self._credits.get(key, 0)

        slot = self._cache.get(key)
        if slot is not None and slot[0] == credits and slot[1][0] == live:
            run = slot[1][1]
            self._cache.move_to_end(key)
        else:
            run = self._best_run(data, live)
            self._cache[key] = [credits, (live, run)]
            self._cache.move_to_end(key)
            while len(self._cache) > MAX_SEEDS:
                self._cache.popitem(last=False)

        if run is None:
            return None
        first, end = run
        start = first * width
        stop = min(end * width, buf_len)
        return start, stop - start

    def _scores(self, data: bytes, live: int) -> np.ndarray | None:
        """``s - p0 * n`` over the first *live* bins; None with no trials or no gain."""
        counts = self._rates.counts(data)
        if counts is None:
            return None

        n = counts[0][:live].astype(np.float64)
        s = counts[1][:live].astype(np.float64)
        total_n = float(n.sum())
        total_s = float(s.sum())
        if total_n <= 0.0 or total_s <= 0.0:
            return None
        return s - (total_s / total_n) * n

    def _best_run(self, data: bytes, live: int) -> tuple[int, int] | None:
        x = self._scores(data, live)
        if x is None:
            return None

        best = max_subarray(x)
        if best is None or best[2] < MIN_EXCESS:
            return None
        return best[0], best[1]

    def scores(self, data: bytes, buf_len: int) -> np.ndarray | None:
        """Per-bin excess scores (tests, stats); None before any gain."""
        width = self._rates.width(data)
        if width is None or buf_len < 1:
            return None
        return self._scores(data, -(-buf_len // width))
