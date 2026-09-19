"""Entropy z-score seed strategy: byte entropy read against the corpus's own spread.

Plan: ``docs/handover/handover_entropy_seed_schedulers_2026-09-19.md`` §3.

``SeedScorer._energy_factor`` reads byte entropy against fixed breakpoints
(``ENTROPY_SPARSE_PCT=25`` / ``ENTROPY_STRUCTURED_PCT=62`` /
``ENTROPY_RANDOM_PCT=93``). Those are target-agnostic: on an image, audio or
compressed-container target every well-formed seed legitimately sits above
93%, so the whole corpus lands in the one "probably random noise" bucket and
the signal is gone. This arm takes each seed's entropy as a z-score against
the corpus's own running mean and spread instead, which self-calibrates to
whatever regime the target actually has and leaves no constant to tune::

    z(s)      = (byte_entropy_pct(s) - mean) / stddev
    weight(s) = exp(-((z - target_z) / width)^2 / 2)

``target_z`` is the only knob and it points the arm: 0 favours seeds typical
for this corpus, a positive target chases the high-entropy tail, a negative
one the sparse tail. It does not replace ``SeedScorer`` -- that scales
*energy* once a seed is picked, this one decides *which* seed is picked --
and nothing here touches it.

Moments come from a windowed :class:`RunningMoments`. Windowed because the
unbounded variant retains every observation in a deque and serialises it, so
an all-history tracker would grow with the campaign; the window also lets the
calibration follow a corpus whose regime shifts as new formats are found.

Below ``MIN_OBSERVATIONS`` distinct seeds, or with no spread at all, the
variance estimate means nothing and :meth:`select` declines rather than
picking off it -- the same warm-up gate ``CriticalSlowingDown`` uses before
trusting its own variance signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fuzzer_tool.core.byte_entropy import byte_entropy_pct
from fuzzer_tool.core.running_stats import RunningMoments

#: Distinct seeds before the corpus's spread is worth calibrating against.
#: Same gate value as CriticalSlowingDown.min_observations.
MIN_OBSERVATIONS = 20

#: Observations the moments keep. 512 seeds of history is enough to pin a
#: target's entropy regime and bounds what a resume has to carry.
MOMENT_WINDOW = 512

#: Spread (in entropy percent) below which the corpus counts as having
#: none. Not "> 0": byte entropy is computed in floating point and a
#: single-symbol input lands 5.6e-15 above zero rather than on it, so a
#: corpus of constant seeds shows a 1.5e-15 stddev and z-scores that are
#: pure rounding noise amplified to +-3.
MIN_SPREAD_PCT = 1e-6

#: Floor under the gaussian so a far-tail seed stays reachable.
MIN_WEIGHT = 1e-6
STATE_VERSION = 1


def _valid_state(data: Any) -> bool:
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return False

    observed = data.get("observed", 0)
    if type(observed) is not int or observed < 0:
        return False

    return all(type(data.get(key, 0.0)) in (int, float) for key in ("target_z", "width"))


class EntropyZScoreSeedStrategy:
    """Elo-arbitrated ``entropy_zscore`` seed arm (``--entropy-zscore``)."""

    def __init__(
        self,
        rng: Any,
        target_z: float = 0.0,
        width: float = 1.0,
        window: int = MOMENT_WINDOW,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        self._rng = rng
        self._target = float(target_z)
        self._width = float(width) if width > 0 else 1.0
        self._min_observations = min_observations
        self._moments = RunningMoments(window=window)
        self._entropy: dict[bytes, float] = {}
        self._observed = 0
        self._selected = 0

    @property
    def warmed(self) -> bool:
        """True once enough distinct seeds have been seen to judge :attr:`ready`."""
        return self._moments.count >= self._min_observations

    @property
    def ready(self) -> bool:
        """True once the corpus has enough spread to calibrate against."""
        return self.warmed and self._moments.stddev > MIN_SPREAD_PCT

    def scores(self, seeds: list[bytes]) -> list[float]:
        """Gaussian weight per seed, peaked at ``target_z``."""
        self._sync(seeds)
        if not seeds:
            return []

        entropy = np.fromiter((self._entropy[s] for s in seeds), np.float64, len(seeds))
        stddev = self._moments.stddev
        if stddev <= MIN_SPREAD_PCT:
            return [1.0 + MIN_WEIGHT] * len(seeds)

        offset = ((entropy - self._moments.mean) / stddev - self._target) / self._width
        weights: list[float] = (np.exp(-0.5 * offset * offset) + MIN_WEIGHT).tolist()
        return weights

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Weighted draw, or None while the calibration is not trustworthy."""
        if not seeds:
            return None

        weights = self.scores(seeds)
        if not self.ready:
            return None

        self._selected += 1
        chosen: bytes = self._rng.weighted_choice(seeds, weights)
        return chosen

    def _sync(self, seeds: list[bytes]) -> None:
        """Observe seeds seen for the first time; drop the rest of the cache.

        The cache is bounded to the live corpus, but the moments deliberately
        keep what it evicted: they estimate the target's entropy regime, not
        the corpus's current census, and a minimization pass should not reset
        a calibration the whole run paid for. A seed pruned and later
        re-admitted is therefore observed twice, which moves the mean by
        O(1/window) and cannot flip a regime.
        """
        live = set(seeds)
        for seed in live.difference(self._entropy):
            entropy = byte_entropy_pct(seed)
            self._entropy[seed] = entropy
            self._moments.update(entropy)
            self._observed += 1

        if len(self._entropy) > len(live):
            self._entropy = {k: v for k, v in self._entropy.items() if k in live}

    def stats(self) -> dict[str, Any]:
        return {
            "observed": self._observed,
            "selected": self._selected,
            "cached": len(self._entropy),
            "ready": self.ready,
            "target_z": self._target,
            "width": self._width,
            "mean_entropy": self._moments.mean,
            "stddev_entropy": self._moments.stddev,
        }

    def to_dict(self) -> dict[str, Any]:
        """Counters and knobs only -- deliberately not the moments.

        A resume reloads the whole corpus and the first :meth:`scores` call
        observes all of it in one pass, so the calibration is rebuilt
        immediately and from the seeds that actually survived. Restoring a
        persisted window on top of that would count every one of them twice.
        """
        return {
            "version": STATE_VERSION,
            "observed": self._observed,
            "selected": self._selected,
            "target_z": self._target,
            "width": self._width,
        }

    @classmethod
    def from_dict(cls, data: Any, rng: Any, **kwargs: Any) -> EntropyZScoreSeedStrategy:
        """Restore the calibration; a malformed payload is ignored whole."""
        out = cls(rng, **kwargs)
        if not _valid_state(data):
            return out

        out._observed = data.get("observed", 0)
        out._selected = data.get("selected", 0)
        out._target = float(data.get("target_z", out._target))
        out._width = float(data.get("width", out._width)) or 1.0
        return out
