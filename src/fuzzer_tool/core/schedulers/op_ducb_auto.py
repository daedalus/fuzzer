"""AutoDUCBScheduler: D-UCB with its memory set by the noise/lag tradeoff.

``DUCBScheduler`` forgets at a hand-picked rate (``per_arm`` pulls of memory
per arm). That memory W trades two errors that cannot both be small, the
time-frequency uncertainty of any windowed estimate:

    noise  ~ sigma^2 / (2W)      short memory: noisy mean
    lag    ~ delta * W           long memory: trails a drift delta per pull

    MSE(W) = sigma^2/(2W) + delta^2 W^2   ->   W* = (sigma^2 / (4 delta^2))^(1/3)

sigma^2 and delta^2 are measured per arm in its own pull time from a
``ScaleBank`` with a fast and a slow memory (M_f, M_s):

    sigma^2  successive-difference variance (drift-robust)
    g = mean_fast - mean_slow;  E[g] = delta*(M_s - M_f) under linear drift
    G = time-smoothed g^2;      E[G] = delta^2 (M_s-M_f)^2 + c*sigma^2
    delta^2 = max(0, G - c*sigma^2 - Z*se) / (M_s - M_f)^2

c = Var(g)/sigma^2 for iid rewards. g is ~N(0, c*sigma^2) with no drift, so
G over n ~ n_eff/M_s independent g's has se = c*sigma^2*sqrt(2/n): drift
smaller than Z*se is below the resolution limit and reads as zero.

Per arm, averaged over arms past warm-up, re-solved every ``retune_every`` records
(O(K) per retune, O(1) per record otherwise), clamped to
[per_arm_min, per_arm_max]. The result replaces ``per_arm``, so the global
discount stays ``gamma = 1 - 1/(per_arm * K)``.
"""

from __future__ import annotations

import math
from array import array

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._multiscale import ScaleBank
from fuzzer_tool.core.schedulers.op_ducb import DUCB_PER_ARM, DUCBScheduler

#: Memory bounds, pulls per arm. Below MIN the UCB width dominates every
#: index; above MAX a 147-arm campaign never forgets within a run.
PER_ARM_MIN = 8.0
PER_ARM_MAX = 1024.0

#: Drift probe memories, pulls of the arm itself.
FAST_MEMORY = 8.0
SLOW_MEMORY = 64.0

#: Records between re-solves of W*.
RETUNE_EVERY = 1024

#: Memory of the g^2 smoother, own pulls. Short enough that a step's echo
#: clears: DecayingBest late share (8 seeds, min) 0.880 at 256, 0.505 at 512.
SMOOTH_MEMORY = 256.0

#: Significance gate on drift, in standard errors of the null g^2 mean.
DRIFT_Z = 4.0

#: Slots of the per-arm g^2 smoother: weight sum, squared weight sum, sum.
_GN, _GN2, _GS = range(3)


def optimal_memory(sigma2: float, delta2: float, lo: float, hi: float) -> float:
    """W* = (sigma^2 / (4 delta^2))^(1/3), clamped to [lo, hi].

    Example: sigma^2=0.04, delta^2=1e-8 -> W* = (1e6)^(1/3) = 100.
    """
    if delta2 <= 0.0:
        return hi
    w = (sigma2 / (4.0 * delta2)) ** (1.0 / 3.0)
    return min(hi, max(lo, w))


def _diff_noise_coeff(fast: float, slow: float) -> float:
    """Var(mean_fast - mean_slow) / sigma^2 for iid rewards in steady state."""
    lf, ls = 1.0 - 1.0 / fast, 1.0 - 1.0 / slow
    var_f = (1.0 - lf) / (1.0 + lf)
    var_s = (1.0 - ls) / (1.0 + ls)
    cov = (1.0 - lf) * (1.0 - ls) / (1.0 - lf * ls)
    return var_f + var_s - 2.0 * cov


class AutoDUCBScheduler(DUCBScheduler):
    """D-UCB whose per-arm memory is re-solved from measured noise and drift.

    Args:
        fast: Fast drift-probe memory, own pulls.
        slow: Slow drift-probe memory; also the warm-up an arm needs before
            it counts toward the estimate.
        retune_every: Records between re-solves.
        per_arm_min: Shortest memory allowed.
        per_arm_max: Longest memory allowed.
        rng: Shared ``RandPool`` (Hard Rule 16).
        **kwargs: Forwarded to ``DUCBScheduler`` (xi, b, exploration).
    """

    supports_priors = False  # D-UCB keeps discounted sums, no Beta prior.

    def __init__(
        self,
        fast: float = FAST_MEMORY,
        slow: float = SLOW_MEMORY,
        retune_every: int = RETUNE_EVERY,
        per_arm_min: float = PER_ARM_MIN,
        per_arm_max: float = PER_ARM_MAX,
        rng: RandPool | None = None,
        **kwargs: float,
    ) -> None:
        if retune_every <= 0:
            raise ValueError(f"retune_every must be positive, got {retune_every!r}")
        if not 0.0 < per_arm_min <= per_arm_max:
            raise ValueError(
                f"need 0 < per_arm_min <= per_arm_max, got {per_arm_min}, {per_arm_max}"
            )
        # ScaleBank validates fast/slow (>= 1, increasing).
        self._bank = ScaleBank((fast, slow))
        self._lag_span = slow - fast
        self._noise_coeff = _diff_noise_coeff(fast, slow)
        self._warmup = int(slow)
        self._retune_every = retune_every
        self._until_retune = retune_every
        self._lo = per_arm_min
        self._hi = per_arm_max
        self._retunes = 0
        self._g_lam = 1.0 - 1.0 / SMOOTH_MEMORY
        self._g2: dict[str, array] = {}
        start = min(per_arm_max, max(per_arm_min, DUCB_PER_ARM))
        super().__init__(rng=rng, per_arm=start, **kwargs)

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Credit *name*; re-solve the memory every ``retune_every`` records."""
        super().record(name, success, weight)
        bank = self._bank
        bank.update(name, weight if success else 0.0)

        # Smooth g^2 once the slow probe is past warm-up.
        if bank.pulls(name) >= self._warmup:
            self._smooth_g2(name, bank.mean(name, 0) - bank.mean(name, 1))

        self._until_retune -= 1
        if self._until_retune > 0:
            return
        self._until_retune = self._retune_every
        self._retune()

    def _smooth_g2(self, name: str, g: float) -> None:
        """Discounted mean of g^2 with its weight sums (for n_eff)."""
        s = self._g2.get(name)
        if s is None:
            s = array("d", (0.0, 0.0, 0.0))
            self._g2[name] = s
        lam = self._g_lam
        s[_GN] = lam * s[_GN] + 1.0
        s[_GN2] = lam * lam * s[_GN2] + 1.0
        s[_GS] = lam * s[_GS] + g * g

    def drift_stats(self) -> tuple[float, float]:
        """Mean (sigma^2, delta^2) over arms past warm-up; (0, 0) if none."""
        bank = self._bank
        coeff = self._noise_coeff
        sigma_sum = 0.0
        excess_sum = 0.0
        for arm, s in self._g2.items():
            sigma2 = bank.noise_var(arm)
            sigma_sum += sigma2

            # Independent g's in the smoother: n_eff / slow memory, >= 1.
            n_indep = max(1.0, s[_GN] * s[_GN] / s[_GN2] / self._warmup)
            null = coeff * sigma2
            gate = DRIFT_Z * null * math.sqrt(2.0 / n_indep)
            excess_sum += max(0.0, s[_GS] / s[_GN] - null - gate)

        n = len(self._g2)
        if n == 0:
            return 0.0, 0.0
        return sigma_sum / n, excess_sum / n / (self._lag_span * self._lag_span)

    def _retune(self) -> None:
        """Set per_arm to W*; re-derive gamma once K is known."""
        sigma2, delta2 = self.drift_stats()
        if sigma2 == 0.0 and delta2 == 0.0 and not self._bank.arms():
            return
        self.per_arm = optimal_memory(sigma2, delta2, self._lo, self._hi)
        self._retunes += 1
        if self._k_seen > 0:
            self._grow_k(self._k_seen)

    def bandit_stats(self) -> dict:
        """D-UCB diagnostics plus the current solved memory."""
        stats = super().bandit_stats()
        stats["ducb_auto_per_arm"] = round(self.per_arm, 2)
        stats["ducb_auto_retunes"] = self._retunes
        return stats
