"""PositionCanaryScheduler: a deliberately worst-possible position proposer.

The position-arena counterpart of ``core/schedulers/op_canary.py`` and
``core/schedulers/seed_canary.py``. ``services/position_arena.py``
arbitrates among every enabled position proposer (uniform, sensitivity,
te, phase, mi, crash_mi, region, burn_front, ...) with Elo, under
``pos_<n>`` keys. That tournament only produces *relative* standing -- it
has no independent floor of its own. Before this scheduler existed, the
arena substituted ``uniform`` for that role (see
``Fuzzer._check_canary_inspection``), but uniform is a real baseline
arm, not something built to lose: a proposer merely tying uniform is not
obviously broken the way a proposer at or below a deliberately-worst
floor is.

PositionCanaryScheduler closes that gap the same way op_canary/seed_canary
close it for their arenas. It has no candidate list to argmin over the way
those two do (an operator or seed key is a discrete, enumerable thing; a
byte offset is not), so it bins a seed's offsets the same way
``pos_burn_front.py`` does (``width = ceil(len(data) / MAX_BINS)``,
LRU-bounded over ``MAX_SEEDS`` seeds) and tracks a Beta(alpha, beta)
posterior per bin from the same ``record(data, offsets, outcome, weight)``
signal every other position scheduler's ``settle()`` calls feed
off-policy. It always proposes the FIRST byte of whichever bin has the
LOWEST posterior mean -- the opposite of the heat-weighted argmax every
real position proposer here performs. Ties (most commonly at the shared
Beta(1, 1) prior, before any bin has been recorded) go to the
lowest-indexed bin -- deterministic, no randomness anywhere in this path,
so canary never gets an accidental assist from luck either.

This is not a fuzzing strategy. It exists purely as an instrumented floor
for the position arena's tournament: see ``BayesianEloTracker.
strategies_below_canary`` and ``Fuzzer._check_canary_inspection``, which
log when a real proposer ranks at or below it -- not evidence canary is
doing well, evidence that proposer needs inspection.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import xxhash

from fuzzer_tool.core.schedulers.pos_base import Outcome

MAX_BINS = 4096  # offsets per seed are binned down to this, as pos_burn_front
MAX_SEEDS = 256  # LRU bound on per-seed posteriors


@dataclass
class _Posterior:
    width: int
    alpha: dict[int, float] = field(default_factory=dict)
    beta: dict[int, float] = field(default_factory=dict)


class PositionCanaryScheduler:
    """Deliberately worst-in-class position proposer; a floor for the pos Elo pool.

    Not a fuzzing strategy -- see the module docstring for why an
    intentionally-bad, signal-driven floor is more useful here than a
    signal-blind one (that role belongs to ``uniform``, already in the
    pool).
    """

    name = "canary"

    #: No meaningful priors: seeding it with a "good" prior would work
    #: against the one property that matters -- being worst. Mirrors
    #: CanaryScheduler.supports_priors / SeedCanaryScheduler.supports_priors.
    supports_priors = False

    def __init__(self) -> None:
        self._seeds: OrderedDict[int, _Posterior] = OrderedDict()

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """First byte of the lowest-posterior-mean bin; never declines on a live buffer."""
        if buf_len <= 0:
            return None
        post = self._posterior_for(data)
        num_bins = max(1, -(-len(data) // post.width)) if data else 1

        worst_bin = 0
        worst_mean = self._mean(post, 0)
        for b in range(1, num_bins):
            mean = self._mean(post, b)
            if mean < worst_mean:
                worst_bin, worst_mean = b, mean

        last = buf_len - 1
        return min(worst_bin * post.width, last)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """One fractional-Bernoulli observation per offset's bin -- the same signal
        every other position scheduler's outcome is judged by.
        """
        offsets = [o for o in offsets if o >= 0]
        if not offsets:
            return
        post = self._posterior_for(data)
        r = min(1.0, max(0.0, float(weight))) if outcome is Outcome.GAIN else 0.0
        for off in offsets:
            b = off // post.width
            post.alpha[b] = post.alpha.get(b, 1.0) + r
            post.beta[b] = post.beta.get(b, 1.0) + (1.0 - r)

    def bandit_stats(self, data: bytes) -> dict[int, tuple[float, float]]:
        """Return (alpha, beta) pseudocounts per bin for one seed, prior included."""
        post = self._seeds.get(self._key(data))
        if post is None:
            return {}
        bins = set(post.alpha) | set(post.beta)
        return {b: (post.alpha.get(b, 1.0), post.beta.get(b, 1.0)) for b in sorted(bins)}

    def seed_count(self) -> int:
        return len(self._seeds)

    @staticmethod
    def _mean(post: _Posterior, b: int) -> float:
        a = post.alpha.get(b, 1.0)
        beta = post.beta.get(b, 1.0)
        return a / (a + beta)

    @staticmethod
    def _key(data: bytes) -> int:
        return xxhash.xxh3_64_intdigest(data)

    def _posterior_for(self, data: bytes) -> _Posterior:
        key = self._key(data)
        post = self._seeds.get(key)
        if post is None:
            width = max(1, -(-len(data) // MAX_BINS)) if data else 1
            post = self._seeds[key] = _Posterior(width=width)
            while len(self._seeds) > MAX_SEEDS:
                self._seeds.popitem(last=False)
        self._seeds.move_to_end(key)
        return post
