"""Entropy-deviation seed strategy: byte entropy read against the corpus mean.

Plan: ``docs/handover/handover_entropy_seed_schedulers_2026-09-19.md`` §1.

The byte-content analogue of the edge-hit deviation bonus
``seed_picker._weight_entropy_and_distance`` already applies inside the
generic weighted picker. That method scores a seed by how far its *edge-hit*
Shannon entropy (``EdgeTracker.shannon_entropy_seed`` -- entropy over which
edges a seed hits and how often) deviates from the corpus's mean edge-hit
entropy. This is the same shape of bonus over an unrelated signal instead:
``byte_entropy_pct`` -- entropy over a seed's own *bytes*. A seed can be
entirely typical in its coverage profile while carrying byte content unlike
anything else in the corpus (or vice versa); the two signals are independent
and this module does not conflate them::

    deviation(s) = |byte_entropy_pct(s) - mean| / max(mean, 0.01)
    weight(s)    = 1.0 + min(deviation, 1.0) * 0.5

Same normalization ``_weight_entropy_and_distance`` uses for the edge-hit
version, applied to a different entropy.

Deliberately scores deviation from the *mean of per-seed entropies*, not
from the corpus-wide *pooled* distribution (what ``seed_entropy_kl.py``'s
KL-divergence arm scores against, and what
``core.byte_entropy.CumulativeByteEntropy`` tracks for the status line).
Pooling low- and high-entropy seeds together generally raises the pooled
figure above either individual value, let alone their average (Jensen's
inequality on the concave entropy function), so "mean of per-seed values"
and "pooled distribution's own entropy" are different numbers answering
different questions. The handover doc's §1 flagged measuring both against
a real corpus before assuming one is more useful than the other; this
module is deliberately the plain-mean version so the two can be A/B'd
independently rather than shipped conflated.

Below ``MIN_OBSERVATIONS`` distinct seeds a mean is too noisy to deviate
from, and :meth:`select` declines rather than picking off it -- same
warm-up gate ``EntropyZScoreSeedStrategy``/``CriticalSlowingDown`` use
before trusting their own variance/spread signals.
"""

from __future__ import annotations

from typing import Any

from fuzzer_tool.core.byte_entropy import byte_entropy_pct

#: Distinct seeds before the corpus mean is worth deviating against. Same
#: gate value as EntropyZScoreSeedStrategy.MIN_OBSERVATIONS.
MIN_OBSERVATIONS = 20

#: Deviation is capped here before scaling, and the max bonus is a 50%
#: weight increase -- not deviation-proportional without bound. Same
#: constants `_weight_entropy_and_distance` uses for the edge-hit version.
_MAX_DEVIATION = 1.0
_DEVIATION_GAIN = 0.5


def deviation_weight(entropy: float, mean_entropy: float) -> float:
    """Weight for one seed given its own and the corpus's mean byte entropy.

    Returns 1.0 (no bonus, no penalty) when ``mean_entropy <= 0`` -- a
    corpus of all-zero-entropy seeds has nothing to deviate from.
    """
    if mean_entropy <= 0:
        return 1.0
    deviation = abs(entropy - mean_entropy) / max(mean_entropy, 0.01)
    return 1.0 + min(deviation, _MAX_DEVIATION) * _DEVIATION_GAIN


class EntropyDeviationSeedStrategy:
    """Elo-arbitrated ``entropy_deviation`` seed arm (``--entropy-deviation``)."""

    def __init__(self, rng: Any, min_observations: int = MIN_OBSERVATIONS) -> None:
        self._rng = rng
        self._min_observations = min_observations
        self._entropy: dict[bytes, float] = {}
        self._sum = 0.0
        self._observed = 0
        self._selected = 0

    @property
    def warmed(self) -> bool:
        """True once enough distinct seeds have been seen to trust the mean."""
        return len(self._entropy) >= self._min_observations

    def _mean(self) -> float:
        n = len(self._entropy)
        return self._sum / n if n else 0.0

    def scores(self, seeds: list[bytes]) -> list[float]:
        """Deviation weight per seed, aligned 1:1 with ``seeds``."""
        self._sync(seeds)
        mean = self._mean()
        return [deviation_weight(self._entropy[s], mean) for s in seeds]

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Weighted draw, or None while the corpus mean is not trustworthy yet."""
        if not seeds:
            return None

        weights = self.scores(seeds)
        if not self.warmed:
            return None

        total = sum(weights)
        if total <= 0:
            return None

        self._selected += 1
        chosen: bytes = self._rng.weighted_choice(seeds, weights)
        return chosen

    def _sync(self, seeds: list[bytes]) -> None:
        """Observe seeds seen for the first time; drop what the corpus evicted.

        Mirrors ``EntropyZScoreSeedStrategy._sync``: the cache tracks the
        live corpus exactly, an O(1) running sum stands in for its
        ``RunningMoments`` (only the mean is needed here, not a windowed
        variance), and a seed pruned then re-admitted is observed again --
        harmless, it just nudges the mean by one seed's worth twice.
        """
        live = set(seeds)
        for seed in live.difference(self._entropy):
            entropy = byte_entropy_pct(seed)
            self._entropy[seed] = entropy
            self._sum += entropy
            self._observed += 1

        if len(self._entropy) > len(live):
            evicted = [k for k in self._entropy if k not in live]
            for k in evicted:
                self._sum -= self._entropy.pop(k)

    def stats(self) -> dict[str, Any]:
        return {
            "observed": self._observed,
            "selected": self._selected,
            "cached": len(self._entropy),
            "warmed": self.warmed,
            "mean_entropy": self._mean(),
        }
