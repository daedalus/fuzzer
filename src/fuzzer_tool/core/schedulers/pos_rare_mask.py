"""PositionRareMaskScheduler: mutate where the seed's rare branch survives.

FairFuzz (Lemieux & Sen, ASE 2018) targets a seed's *rarest* edge and builds a
*branch mask*: the offsets whose mutation keeps that edge hit. Mutating
inside the mask explores the neighbourhood of the rare branch instead of
falling off it::

    target  = argmin over the seed's edges of owner_count(edge)   (<= RARE_MAX)
    round   = mutate offsets O, run
    hit     = target still in the exec's edge set?
    mask    = per bin, (hits + PRIOR_A) / (trials + PRIOR_A + PRIOR_B)

FairFuzz computes the mask with one deterministic flip per byte; here it is
learned from every round's offsets (``core/schedulers/_bin_rates.py``). A
round that loses the edge charges every offset it touched: any one of them
may have broken it. The outcome and weight do not matter -- a gain that
dropped the rare edge still says the offset is outside the mask.

Target: lowest ``owner_count`` (ties: lowest edge id), recomputed every
``RETARGET_EVERY`` credits per seed; a change resets the seed's mask, whose
evidence was about the old edge. No edge at or under ``RARE_MAX`` owners
means no target: no credit, and ``propose`` declines.

``edges_of`` / ``owner_count`` / ``hit`` are injected (``EdgeTracker`` seed
edges and owner counts, the current exec's SHM edge set). ``hit`` is queried
only when a target exists, so an untargeted seed costs no SHM scan.

Off-policy extra: credited every settled round whoever served. Declines
(uniform, charged to the arm) with no target or no evidence yet. Not
persisted.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Collection, Sequence

import numpy as np
import xxhash

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import MAX_SEEDS, BinRates
from fuzzer_tool.core.schedulers.pos_base import Outcome

PRIOR_A = 1.0  # Laplace prior: an untried bin weighs 0.5
PRIOR_B = 1.0
RARE_MAX = 3  # rare = at most this many owners (EdgeTracker.rare_edge_count: < 4)
RETARGET_EVERY = 256  # credits between target recomputations, per seed


class PositionRareMaskScheduler:
    """Rate-weighted bins, rate = how often mutating the bin kept the rare edge."""

    name = "rare_mask"

    def __init__(
        self,
        rng: RandPool,
        edges_of: Callable[[bytes], Collection[int] | None],
        owner_count: Callable[[int], int],
        hit: Callable[[int], bool | None],
    ) -> None:
        self._edges_of = edges_of
        self._owner_count = owner_count
        self._hit = hit
        self._rates = BinRates(rng, PRIOR_A, PRIOR_B)
        # seed hash -> [target edge or None, credits since computed]
        self._targets: OrderedDict[int, list] = OrderedDict()

    # -- target -----------------------------------------------------------------

    def _rarest(self, data: bytes) -> int | None:
        edges = self._edges_of(data)
        if not edges:
            return None

        count = self._owner_count
        best = min(edges, key=lambda e: (count(e), e))
        return best if count(best) <= RARE_MAX else None

    def _target(self, data: bytes, *, tick: bool) -> int | None:
        key = xxhash.xxh3_64_intdigest(data)
        slot = self._targets.get(key)
        if slot is not None and slot[1] < RETARGET_EVERY:
            self._targets.move_to_end(key)
            slot[1] += tick
            return slot[0]

        edge = self._rarest(data)
        if slot is not None and slot[0] != edge:
            self._rates.reset(data)

        self._targets[key] = [edge, int(tick)]
        self._targets.move_to_end(key)
        while len(self._targets) > MAX_SEEDS:
            self._targets.popitem(last=False)
        return edge

    def target(self, data: bytes) -> int | None:
        """The seed's current rare edge (tests, stats); computes it if unseen."""
        return self._target(data, tick=False)

    # -- protocol ---------------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if self._target(data, tick=False) is None:
            return None
        return self._rates.propose(data, buf_len)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Hit -> every offset in the mask; lost -> every offset charged."""
        live = [o for o in offsets if o >= 0]
        if not live:
            return

        edge = self._target(data, tick=True)
        if edge is None:
            return

        hit = self._hit(edge)
        if hit is None:
            return

        self._rates.credit(data, live, 1.0 if hit else 0.0)

    def weights(self, data: bytes, buf_len: int) -> np.ndarray | None:
        """Per-bin weights (tests, stats); None before any evidence."""
        return self._rates.weights(data, buf_len)
