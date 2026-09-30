"""Gale-Shapley target scheduler: stable seed -> target matching per epoch.

The other target arms pick a binary and leave the seed to the seed
scheduler. This one picks the pair. Each epoch samples ``batch`` seeds and
runs seed-proposing deferred acceptance (``core/stable_matching.py``):

    seeds    propose to targets by their yield there, (gains+1)/(tries+2)
    targets  keep the seeds they have run least (yield breaks ties)
    quotas   seats per target from the 1/edges weights, >= batch in total

The two tastes pull apart: seeds exploit where they paid off, targets
explore seeds they have not seen. The stable match is served one pair per
pick; ``take_hint`` hands the seed to ``SeedPicker``. Empty plan -> rematch.

Yield counts are learned off-policy from every settled round, per
(seed, target), LRU-bounded to ``max_seeds`` seeds. Not persisted.
"""

from __future__ import annotations

import math
from array import array
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable, Sequence

from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedulers.tgt_base import TargetRound, Weights
from fuzzer_tool.core.stable_matching import UNMATCHED, deferred_acceptance

DEFAULT_BATCH = 32
DEFAULT_MAX_SEEDS = 4096


def seat_quotas(k: int, weights: Sequence[float]) -> list[int]:
    """Seats per target, proportional to ``weights``, summing to at least ``k``.

    Non-finite or non-positive weights get no share; if none is usable every
    target gets an equal one. Ceil, so every seed can be seated.
    """
    if not weights:
        return []
    clean = [w if math.isfinite(w) and w > 0 else 0.0 for w in weights]
    total = sum(clean)
    if not (total > 0 and math.isfinite(total)):
        clean, total = [1.0] * len(weights), float(len(weights))

    seats = [math.ceil(k * w / total) for w in clean]

    # Float drift could shave a seat; hand it to the heaviest target.
    short = k - sum(seats)
    if short > 0:
        seats[clean.index(max(clean))] += short
    return seats


class GaleShapleyTarget:
    """Target arm that also chooses the seed (see module docstring)."""

    name = "gale_shapley"

    def __init__(
        self,
        n_targets: int,
        seeds: Callable[[int], list[bytes]],
        weights: Weights,
        key: Callable[[bytes], Hashable],
        batch: int = DEFAULT_BATCH,
        max_seeds: int = DEFAULT_MAX_SEEDS,
    ) -> None:
        self._n = n_targets
        self._seeds = seeds
        self._weights = weights
        self._key = key
        self._batch = batch
        self._max = max_seeds

        # key -> [tries per target..., gains per target...]
        self._rows: OrderedDict[Hashable, array] = OrderedDict()
        self._plan: deque[tuple[bytes, int]] = deque()
        self._hint: bytes | None = None
        self._turn = 0

    @property
    def tracked(self) -> int:
        """Seeds with yield rows."""
        return len(self._rows)

    def pick(self, n: int) -> int:
        if not self._plan:
            self._rematch(n)

        # No corpus yet: cycle, no hint.
        if not self._plan:
            self._hint = None
            idx = self._turn % n
            self._turn += 1
            return idx

        seed, idx = self._plan.popleft()
        self._hint = seed
        return idx

    def take_hint(self) -> bytes | None:
        """Seed matched to the last pick, once."""
        hint, self._hint = self._hint, None
        return hint

    def record(self, rnd: TargetRound) -> None:
        n = self._n
        if not 0 <= rnd.idx < n:
            return

        key = self._key(rnd.seed)
        row = self._rows.get(key)
        if row is None:
            row = array("l", [0] * (2 * n))
            self._rows[key] = row
            if len(self._rows) > self._max:
                self._rows.popitem(last=False)
        else:
            self._rows.move_to_end(key)

        row[rnd.idx] += 1
        if rnd.outcome is Outcome.GAIN:
            row[n + rnd.idx] += 1

    def _rates(self, seed: bytes, n: int) -> tuple[list[int], list[float]]:
        """(tries, Laplace yield) per target for ``seed``."""
        row = self._rows.get(self._key(seed))
        if row is None:
            return [0] * n, [0.5] * n
        tries = [row[t] for t in range(n)]
        rates = [(row[self._n + t] + 1) / (tries[t] + 2) for t in range(n)]
        return tries, rates

    def _rematch(self, n: int) -> None:
        seeds = list(dict.fromkeys(self._seeds(self._batch)))
        if not seeds:
            return

        stats = [self._rates(s, n) for s in seeds]

        # Seed taste: best yield first. Target taste: fewest tries, yield < 1 breaks ties.
        prefs = [sorted(range(n), key=lambda t, r=rates: (-r[t], t)) for _, rates in stats]
        scores = [[rates[t] - tries[t] for tries, rates in stats] for t in range(n)]
        quotas = seat_quotas(len(seeds), self._weights()[:n])

        match = deferred_acceptance(prefs, scores, quotas)
        self._plan.extend((s, t) for s, t in zip(seeds, match, strict=True) if t != UNMATCHED)
