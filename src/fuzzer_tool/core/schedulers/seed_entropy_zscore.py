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

from fuzzer_tool.core.byte_entropy import (
    ENTROPY_SAMPLE_CAP,
    POOL_SMOOTHING,
    byte_entropy_pct,
    byte_histogram,
)
from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.core.schedulers.seed_entropy_kl import (
    NULL_MC_DRAWS,
    NULL_MC_SEED,
    NULL_MIN_SD,
    NULL_STALE_L1,
    _null_grid,
    null_kl_bits,
)

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


_PCT_PER_BIT = 100.0 / 8.0


class EntropyLengthNull:
    """Plug-in entropy of an n-byte sample drawn from the pool: bias and spread.

    For draws from ``q``, ``E[H(P_hat_n)] = H(q) - E[KL(P_hat_n || q)]``, so
    the plug-in entropy reads low by exactly the KL null mean
    (:func:`null_kl_bits`, exact). The spread has no cheap closed form in
    the sparse regime and is Monte-Carlo from a private fixed-seed generator
    (a calibration table, never a scheduling decision). Both are on the
    power-of-two grid up to the sample cap, log-log interpolated, in entropy
    percent. The bias is rebuilt on every :meth:`fit` (exact, 1-17 ms); the
    spread is reused until the pool has drifted :data:`NULL_STALE_L1`.
    """

    def __init__(self, draws: int = NULL_MC_DRAWS, seed: int = NULL_MC_SEED) -> None:
        self._draws = draws
        self._seed = seed
        self._grid = _null_grid(ENTROPY_SAMPLE_CAP)
        self._log_n = np.log(self._grid)
        self._log_bias = np.zeros_like(self._log_n)
        self._log_sd = np.zeros_like(self._log_n)
        self._ref: np.ndarray | None = None

    def fit(self, q: Any) -> None:
        probs = np.asarray(q, dtype=np.float64)
        probs = probs / probs.sum()
        bias = [null_kl_bits(probs, int(n)) * _PCT_PER_BIT for n in self._grid]
        self._log_bias = np.log(np.maximum(bias, 1e-12))
        ref = self._ref
        if ref is not None and float(np.abs(probs - ref).sum()) <= NULL_STALE_L1:
            return
        gen = np.random.Generator(np.random.PCG64(self._seed))
        sd: list[float] = []
        for n in self._grid.astype(np.int64):
            freq = gen.multinomial(int(n), probs, size=self._draws).astype(np.float64) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                bits = -np.where(freq > 0, freq * np.log2(freq), 0.0).sum(axis=1)
            sd.append(float(bits.std(ddof=1)) * _PCT_PER_BIT)
        self._log_sd = np.log(np.maximum(sd, NULL_MIN_SD))
        self._ref = probs

    def bias_pct(self, lengths: Any) -> np.ndarray:
        """Expected downward bias, entropy percent; 0 for an empty sample."""
        n = np.asarray(lengths, dtype=np.float64)
        out: np.ndarray = np.exp(np.interp(np.log(np.maximum(n, 1.0)), self._log_n, self._log_bias))
        out[n < 1] = 0.0
        return out

    def sd_pct(self, lengths: Any) -> np.ndarray:
        """Null standard deviation, entropy percent; floored above zero."""
        n = np.asarray(lengths, dtype=np.float64)
        out: np.ndarray = np.exp(np.interp(np.log(np.maximum(n, 1.0)), self._log_n, self._log_sd))
        return out


class EntropyZScoreSeedStrategy:
    """Elo-arbitrated ``entropy_zscore`` seed arm (``--entropy-zscore``)."""

    def __init__(
        self,
        rng: Any,
        target_z: float = 0.0,
        width: float = 1.0,
        window: int = MOMENT_WINDOW,
        min_observations: int = MIN_OBSERVATIONS,
        calibrate_length: bool = False,
    ) -> None:
        self._rng = rng
        self._calibrate = bool(calibrate_length)
        self._null = EntropyLengthNull()
        self._counts: dict[bytes, Any] = {}
        self._lengths: dict[bytes, int] = {}
        self._pool = np.zeros(256, dtype=np.int64)
        self._pool_version = 0
        self._fit_version = -1
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

        if self._calibrate:
            return self._calibrated_weights(seeds)

        entropy = np.fromiter((self._entropy[s] for s in seeds), np.float64, len(seeds))
        stddev = self._moments.stddev
        if stddev <= MIN_SPREAD_PCT:
            return [1.0 + MIN_WEIGHT] * len(seeds)

        offset = ((entropy - self._moments.mean) / stddev - self._target) / self._width
        weights: list[float] = (np.exp(-0.5 * offset * offset) + MIN_WEIGHT).tolist()
        return weights

    def _calibrated_weights(self, seeds: list[bytes]) -> list[float]:
        """Gaussian weight on the length-calibrated z-score.

        ``z_i = (H_i + bias(n_i) - mean) / sqrt(between^2 + sd0(n_i)^2)`` with
        ``between^2 = max(var - mean(sd0^2), 0)``: entropy is first put back
        on the pool's scale, then each seed is judged against its own null
        spread on top of the corpus's true between-seed spread. Adding the
        bias alone made short seeds scatter into the tails and did worse
        than nothing (Spearman +0.53).
        """
        if self._fit_version != self._pool_version:
            pool = self._pool + POOL_SMOOTHING
            self._null.fit(pool / pool.sum())
            self._fit_version = self._pool_version
        lengths = np.fromiter((self._lengths[s] for s in seeds), np.float64, len(seeds))
        entropy = np.fromiter((self._entropy[s] for s in seeds), np.float64, len(seeds))
        corrected = entropy + self._null.bias_pct(lengths)
        sd0 = np.maximum(self._null.sd_pct(lengths), NULL_MIN_SD)
        between = max(float(corrected.var()) - float(np.mean(sd0 * sd0)), 0.0)
        z = (
            (corrected - corrected.mean()) / np.sqrt(between + sd0 * sd0) - self._target
        ) / self._width
        weights: list[float] = (np.exp(-0.5 * z * z) + MIN_WEIGHT).tolist()
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
            counts, total = byte_histogram(seed)
            self._counts[seed] = np.asarray(counts, dtype=np.int64)
            self._lengths[seed] = int(total)
            self._pool += self._counts[seed]
            self._pool_version += 1

        if len(self._entropy) > len(live):
            for seed in set(self._entropy).difference(live):
                self._pool -= self._counts.pop(seed)
                del self._lengths[seed]
                self._pool_version += 1
            self._entropy = {k: v for k, v in self._entropy.items() if k in live}

    def stats(self) -> dict[str, Any]:
        return {
            "observed": self._observed,
            "selected": self._selected,
            "cached": len(self._entropy),
            "ready": self.ready,
            "target_z": self._target,
            "width": self._width,
            "calibrate_length": self._calibrate,
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
