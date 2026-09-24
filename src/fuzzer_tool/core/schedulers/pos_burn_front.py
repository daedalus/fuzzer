"""BurnFrontPositionScheduler: a burn front over a seed's byte offsets.

Model, from a steel-wool fire (fuel, heat, conduction, oxidation, sparks)::

    offsets  = the wool          (binned to at most MAX_BINS)
    heat     = coverage gain credited to an offset, conducted to its
               neighbours -- adjacent bytes are usually one field
    fuel     = untried budget of a bin; each proposal burns part of it, so
               a hot region exhausts and the front moves on
    sparks   = SPARK_RATE uniform draws that jump off the front

There is no time step. Conduction is one Gaussian-kernel deposit per gain
(``heat += credit * K``), so nothing integrates and nothing can diverge --
the rule ``core/analyzers/analyzer_navier_stokes.py`` sets for field-like
models here.

``weight(bin) = heat * fuel``. A gain refuels its bin: a productive site is
not exhausted. Heat cools geometrically every COOL_EVERY proposals.

State is per parent seed, in memory only (LRU-bounded, MAX_SEEDS), and
sparse: at most MAX_HOT_BINS heated bins per seed. Not persisted; a fresh
run re-learns it from the first gains.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import xxhash

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

MAX_BINS = 4096  # offsets per seed are binned down to this
MAX_SEEDS = 256  # LRU bound on per-seed fronts
MAX_HOT_BINS = 256  # sparse heat cap per seed
KERNEL_SIGMA = 3.0  # conduction width, in bins
KERNEL_RADIUS = 9  # 3 sigma
FUEL_BURN = 0.15  # fuel fraction one proposal consumes
FUEL_FLOOR = 1e-3  # keeps every heated bin selectable
SPARK_RATE = 0.10  # uniform escapes from the front
COOL_EVERY = 32  # proposals between cooling steps
COOL_FACTOR = 0.9  # heat multiplier per cooling step
HEAT_FLOOR = 1e-4  # cooler bins are dropped

_KERNEL = tuple(
    math.exp(-(d * d) / (2.0 * KERNEL_SIGMA**2)) for d in range(-KERNEL_RADIUS, KERNEL_RADIUS + 1)
)


@dataclass
class _Front:
    width: int  # bytes per bin, fixed from the seed length
    heat: dict[int, float] = field(default_factory=dict)
    fuel: dict[int, float] = field(default_factory=dict)
    proposals: int = 0


class BurnFrontPositionScheduler:
    name = "burn_front"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        self._fronts: OrderedDict[int, _Front] = OrderedDict()

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """Offset from the seed's front; None when nothing burns or buf is empty."""
        front = self._fronts.get(self._key(data))
        if buf_len <= 0 or front is None or not front.heat:
            return None

        self._tick(front)
        if not front.heat:
            return None

        last = buf_len - 1
        if self._rng.random() < SPARK_RATE:
            return self._rng.randint(0, last)

        bins = list(front.heat)
        weights = [front.heat[b] * front.fuel.get(b, 1.0) for b in bins]
        hot = self._rng.weighted_choice(bins, weights)
        front.fuel[hot] = max(FUEL_FLOOR, front.fuel.get(hot, 1.0) * (1.0 - FUEL_BURN))

        lo = min(hot * front.width, last)
        hi = min(lo + front.width - 1, last)
        return self._rng.randint(lo, hi)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """On GAIN, light each offset's bin and its neighbours; refuel the bin."""
        offsets = [o for o in offsets if o >= 0]
        if outcome is not Outcome.GAIN or not offsets:
            return

        front = self._front_for(data)
        share = weight / len(offsets)
        for off in offsets:
            hot = off // front.width
            for i, k in enumerate(_KERNEL):
                b = hot + i - KERNEL_RADIUS
                if b >= 0:
                    front.heat[b] = front.heat.get(b, 0.0) + share * k
            front.fuel[hot] = 1.0
        self._trim(front)

    def hot_bins(self, data: bytes) -> dict[int, float]:
        """Heat per bin for a seed (a copy); empty when unknown."""
        front = self._fronts.get(self._key(data))
        return dict(front.heat) if front else {}

    def fuel_of(self, data: bytes, hot: int) -> float:
        front = self._fronts.get(self._key(data))
        return front.fuel.get(hot, 1.0) if front else 1.0

    def seed_count(self) -> int:
        return len(self._fronts)

    @staticmethod
    def _key(data: bytes) -> int:
        return xxhash.xxh3_64_intdigest(data)

    def _front_for(self, data: bytes) -> _Front:
        key = self._key(data)
        front = self._fronts.get(key)
        if front is None:
            width = max(1, -(-len(data) // MAX_BINS))
            front = self._fronts[key] = _Front(width=width)
            while len(self._fronts) > MAX_SEEDS:
                self._fronts.popitem(last=False)
        self._fronts.move_to_end(key)
        return front

    @staticmethod
    def _tick(front: _Front) -> None:
        """Cool every COOL_EVERY proposals; drop bins below HEAT_FLOOR."""
        front.proposals += 1
        if front.proposals % COOL_EVERY:
            return

        front.heat = {
            b: h * COOL_FACTOR for b, h in front.heat.items() if h * COOL_FACTOR >= HEAT_FLOOR
        }
        front.fuel = {b: f for b, f in front.fuel.items() if b in front.heat}

    @staticmethod
    def _trim(front: _Front) -> None:
        """Keep the MAX_HOT_BINS hottest bins."""
        if len(front.heat) <= MAX_HOT_BINS:
            return

        keep = sorted(front.heat, key=front.heat.__getitem__, reverse=True)[:MAX_HOT_BINS]
        front.heat = {b: front.heat[b] for b in keep}
        front.fuel = {b: f for b, f in front.fuel.items() if b in front.heat}
