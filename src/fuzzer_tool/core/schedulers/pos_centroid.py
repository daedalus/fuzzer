"""PositionCentroidScheduler: mutate where the seed reaches its cluster's missing members.

A cluster (core/edge_clusters.py) is one code path plus branches only some
seeds take. A seed holding the cluster's medoid (centroid) but not every
member sits on the path and misses siblings that other seeds reach from the
same centre: its gaps. This arm learns which offsets, when mutated, reach
one::

    gaps   = EdgeClusters.gaps(seed edges)   nearest the medoid first, <= 8
    round  = mutate offsets O, run
    hit    = any gap edge in the exec's edge set?
    rate   = per bin, (hits + PRIOR_A) / (trials + PRIOR_A + PRIOR_B)

The inverse of ``pos_rare_mask``: that arm learns where mutating KEEPS a
rare edge; this one where mutating REACHES a missing one. A round that
reaches no gap charges every offset it touched.

Gaps are recomputed every ``RETARGET_EVERY`` credits per seed, and at once
when the cluster model's ``version`` moves (a refit); a change resets the
seed's bins. No gaps means no
credit and ``propose`` declines (uniform, charged to the arm).

``gaps_of`` / ``hit_any`` are injected (``EdgeClusters.gaps`` over the
tracker's seed edges, the current exec's SHM table). ``hit_any`` is queried
only when gaps exist. Off-policy extra: credited every settled round
whoever served. Not persisted.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np

from fuzzer_tool.core.hashing import hash64_len
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import MAX_SEEDS, BinRates
from fuzzer_tool.core.schedulers.pos_base import Outcome

PRIOR_A = 1.0  # Laplace prior: an untried bin weighs 0.5
PRIOR_B = 1.0
RETARGET_EVERY = 256  # credits between gap recomputations, per seed


class PositionCentroidScheduler:
    """Rate-weighted bins, rate = how often mutating the bin reached a gap."""

    name = "centroid"

    def __init__(
        self,
        rng: RandPool,
        gaps_of: Callable[[bytes], Sequence[int]],
        hit_any: Callable[[Sequence[int]], bool | None],
        version: Callable[[], int] = lambda: 0,
    ) -> None:
        self._gaps_of = gaps_of
        self._hit_any = hit_any
        self._version = version
        self._rates = BinRates(rng, PRIOR_A, PRIOR_B)
        # seed hash -> [gap edges (tuple), credits since computed, model version]
        self._targets: OrderedDict[int, list] = OrderedDict()

    # -- target -----------------------------------------------------------------

    def _gaps(self, data: bytes, *, tick: bool) -> tuple[int, ...]:
        key = hash64_len(data)
        version = self._version()
        slot = self._targets.get(key)
        if slot is not None and slot[1] < RETARGET_EVERY and slot[2] == version:
            self._targets.move_to_end(key)
            slot[1] += tick
            return slot[0]

        gaps = tuple(self._gaps_of(data) or ())
        if slot is not None and slot[0] != gaps:
            self._rates.reset(data)

        self._targets[key] = [gaps, int(tick), version]
        self._targets.move_to_end(key)
        while len(self._targets) > MAX_SEEDS:
            self._targets.popitem(last=False)
        return gaps

    def gaps(self, data: bytes) -> tuple[int, ...]:
        """The seed's current gaps (tests, stats); computes them if unseen."""
        return self._gaps(data, tick=False)

    # -- protocol ---------------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if not self._gaps(data, tick=False):
            return None
        return self._rates.propose(data, buf_len)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Reached a gap -> every offset credited; reached none -> every offset charged."""
        live = [o for o in offsets if o >= 0]
        if not live:
            return

        gaps = self._gaps(data, tick=True)
        if not gaps:
            return

        hit = self._hit_any(gaps)
        if hit is None:
            return

        self._rates.credit(data, live, 1.0 if hit else 0.0)

    def weights(self, data: bytes, buf_len: int) -> np.ndarray | None:
        """Per-bin weights (tests, stats); None before any evidence."""
        return self._rates.weights(data, buf_len)
