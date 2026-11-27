"""ScaleBank: per-arm discounted reward estimators at several memories.

Time-frequency uncertainty (Gabor) applied to arm yields: an estimator with
memory M has noise variance ~sigma^2/(2M) but trails a drift delta by
~delta*(M-1). No single M is sharp in both. The bank keeps one estimator per
memory so a reader can pick the scale per arm (multi-resolution, like a
wavelet bank):

    memory 16    ####                     fast: tracks steps, noisy
    memory 64    ################
    memory 256   ################################  slow: precise, lags

Time is the arm's own pull count. Per scale j with lam_j = 1 - 1/M_j:

    n  <- lam*n  + 1        x <- lam*x + r        n2 <- lam^2*n2 + 1
    mean = x/n              n_eff = n^2/n2        (exact for iid rewards)

Noise variance uses successive differences (von Neumann), discounted at the
slowest memory: E[(r_t - r_{t-1})^2]/2 = sigma^2 + delta^2/2, so a drift
inflates it by delta^2/2 only, not by the ~M^2*delta^2 a plain variance pays.
"""

from __future__ import annotations

from array import array

#: Slots per scale: n, n2, x.
_PER_SCALE = 3
#: Trailing slots: discounted diff^2 sum, its weight, last reward, pull count.
_D, _ND, _LAST, _PULLS = range(4)
_TRAILER = 4


class ScaleBank:
    """Per-arm discounted mean / effective count at increasing memories."""

    def __init__(self, memories: tuple[float, ...]) -> None:
        if not memories:
            raise ValueError("memories must be non-empty")
        if any(m < 1.0 for m in memories):
            raise ValueError(f"memories must be >= 1, got {memories!r}")
        if any(b <= a for a, b in zip(memories, memories[1:], strict=False)):
            raise ValueError(f"memories must strictly increase, got {memories!r}")

        self.memories = tuple(memories)
        self._lam = tuple(1.0 - 1.0 / m for m in memories)
        self._lam2 = tuple(v * v for v in self._lam)
        self._tail = _PER_SCALE * len(memories)
        self._arms: dict[str, array[float]] = {}

    def _slots(self, arm: str) -> array[float]:
        slots = self._arms.get(arm)
        if slots is None:
            slots = array("d", bytes(8 * (self._tail + _TRAILER)))
            self._arms[arm] = slots
        return slots

    def update(self, arm: str, reward: float) -> None:
        """Fold one pull's reward into every scale of *arm*."""
        s = self._slots(arm)

        # Per-scale discounted sums.
        i = 0
        for lam, lam2 in zip(self._lam, self._lam2, strict=True):
            s[i] = lam * s[i] + 1.0
            s[i + 1] = lam2 * s[i + 1] + 1.0
            s[i + 2] = lam * s[i + 2] + reward
            i += _PER_SCALE

        # Successive-difference noise sums, slowest memory.
        t = self._tail
        if s[t + _PULLS] > 0.0:
            lam = self._lam[-1]
            diff = reward - s[t + _LAST]
            s[t + _D] = lam * s[t + _D] + diff * diff
            s[t + _ND] = lam * s[t + _ND] + 1.0
        s[t + _LAST] = reward
        s[t + _PULLS] += 1.0

    def mean(self, arm: str, scale: int) -> float:
        """Discounted mean reward of *arm* at *scale*; 0 with no evidence."""
        s = self._arms.get(arm)
        if s is None:
            return 0.0
        i = _PER_SCALE * scale
        return float(s[i + 2] / s[i])

    def n_eff(self, arm: str, scale: int) -> float:
        """Effective iid sample count n^2/n2 of *arm* at *scale*."""
        s = self._arms.get(arm)
        if s is None:
            return 0.0
        i = _PER_SCALE * scale
        return float(s[i] * s[i] / s[i + 1])

    def noise_var(self, arm: str) -> float:
        """Per-pull reward variance from successive differences; 0 below two pulls."""
        s = self._arms.get(arm)
        if s is None:
            return 0.0
        t = self._tail
        nd = s[t + _ND]
        if nd <= 0.0:
            return 0.0
        return float(s[t + _D] / (2.0 * nd))

    def pulls(self, arm: str) -> int:
        """Raw pull count of *arm*."""
        s = self._arms.get(arm)
        return 0 if s is None else int(s[self._tail + _PULLS])

    def arms(self) -> list[str]:
        """Arms with at least one pull."""
        return list(self._arms)
