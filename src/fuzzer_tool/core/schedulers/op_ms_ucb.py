"""MultiScaleUCBScheduler: UCB reading each arm at its own resolution.

A fixed memory cannot be precise and responsive at once (time-frequency
uncertainty; see ``_multiscale.py``). This keeps a geometric bank of memories
per arm, ``base * ratio^j`` own pulls (default 16, 64, 256), and reads the
longest one that is still consistent with every shorter one -- Lepski's
adaptive-window rule:

    pick largest j with |m_j - m_i| <= z * sqrt(s2/ne_i + s2/ne_j)  for all i < j

    stationary arm   all scales agree      -> read 256 (precise)
    step at t0       256 lags, 16 agrees   -> read 16 until 64/256 catch up

s2 is the arm's drift-robust noise variance (successive differences), floored
so an all-zero history cannot make every gap significant. The read scale's
mean and effective count feed a Gaussian UCB index, as ``DUCBScheduler``:

    index = m_j + exploration * 2b * sqrt(xi * log(sum ne) / ne_j)

Memory is in the arm's own pull time, so ne_j saturates at ~2*M_j - 1: an
abandoned arm keeps a bounded count while log(sum ne) grows, so it is
retried. Cost: O(L^2) per record (L = 3), O(1) per arm per select.
"""

from __future__ import annotations

import math

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._multiscale import ScaleBank
from fuzzer_tool.core.schedulers.ucb_common import UCBBase

#: Shortest memory, own pulls.
MS_BASE = 16.0
#: Memory ratio between neighbouring scales.
MS_RATIO = 4.0
#: Scales in the bank.
MS_SCALES = 3
#: Lepski consistency threshold, in standard errors.
MS_Z = 3.0
#: Noise-variance floor: Bernoulli at p=0.001.
VAR_FLOOR = 1e-3
#: Confidence-width coefficient (Gaussian UCB, as DUCBScheduler).
UCB_WIDTH_COEFF = 2.0


class MultiScaleUCBScheduler(UCBBase):
    """Gaussian UCB over a per-arm multi-resolution (Lepski) mean.

    Args:
        base: Shortest memory, own pulls (>= 1).
        ratio: Memory ratio between scales (> 1).
        n_scales: Number of scales (>= 1).
        z: Lepski threshold in standard errors (> 0).
        xi: Exploration constant inside the width.
        b: Reward range.
        exploration: Multiplier on the width (DUCBScheduler's default).
        rng: Shared ``RandPool`` (Hard Rule 16); used to open unpulled arms.
    """

    supports_priors = False  # Discounted sums only, no Beta prior to seed.

    def __init__(
        self,
        base: float = MS_BASE,
        ratio: float = MS_RATIO,
        n_scales: int = MS_SCALES,
        z: float = MS_Z,
        xi: float = 0.6,
        b: float = 1.0,
        exploration: float = 0.25,
        rng: RandPool | None = None,
    ) -> None:
        if ratio <= 1.0:
            raise ValueError(f"ratio must be > 1, got {ratio!r}")
        if n_scales < 1:
            raise ValueError(f"n_scales must be >= 1, got {n_scales!r}")
        if z <= 0.0 or xi <= 0.0 or exploration <= 0.0:
            raise ValueError(f"z, xi, exploration must be positive: {z}, {xi}, {exploration}")

        # ScaleBank rejects base < 1.
        self.memories = tuple(base * ratio**j for j in range(n_scales))
        self._bank = ScaleBank(self.memories)
        self._slowest = n_scales - 1
        self.z = z
        self.xi = xi
        self.b = b
        self.exploration = exploration

        # Cached readout per arm, refreshed on record.
        self._scale: dict[str, int] = {}
        self._count: dict[str, float] = {}
        self._mean: dict[str, float] = {}
        self._fast_reads = 0
        super().__init__(rng=rng)

    def init_arm(self, name: str) -> None:
        """Arms are created lazily on first record; nothing to register."""

    # -- readout -------------------------------------------------------------

    def scale_of(self, arm: str) -> int:
        """Scale index currently read for *arm* (slowest when unseen)."""
        return self._scale.get(arm, self._slowest)

    def bank_mean(self, arm: str, scale: int) -> float:
        """Mean of *arm* at *scale*, straight from the bank."""
        return self._bank.mean(arm, scale)

    def bank_neff(self, arm: str, scale: int) -> float:
        """Effective count of *arm* at *scale*, straight from the bank."""
        return self._bank.n_eff(arm, scale)

    def _lepski(self, arm: str) -> int:
        """Largest scale consistent with every shorter one."""
        bank = self._bank
        s2 = max(bank.noise_var(arm), VAR_FLOOR)
        means = [bank.mean(arm, j) for j in range(self._slowest + 1)]
        vars_ = [s2 / bank.n_eff(arm, j) for j in range(self._slowest + 1)]
        z = self.z

        for j in range(self._slowest, 0, -1):
            m_j, v_j = means[j], vars_[j]
            if all(abs(m_j - means[i]) <= z * math.sqrt(v_j + vars_[i]) for i in range(j)):
                return j
        return 0

    # -- UCBBase hooks -------------------------------------------------------

    def _arm_count(self, op: str) -> float:
        return self._count.get(op, 0.0)

    def _arm_mean(self, op: str, n: float) -> float:  # noqa: ARG002
        return self._mean.get(op, 0.0)

    def _width(self, mean: float, n: float, log_n: float) -> float:  # noqa: ARG002
        return self._width_scale(log_n) / math.sqrt(n)

    def _width_scale(self, log_n: float) -> float:
        return self.exploration * UCB_WIDTH_COEFF * self.b * math.sqrt(self.xi * log_n)

    def _arm_update(self, name: str, reward: float) -> None:
        self._bank.update(name, reward)
        j = self._lepski(name)
        if j < self._slowest:
            self._fast_reads += 1
        self._scale[name] = j
        self._count[name] = self._bank.n_eff(name, j)
        self._mean[name] = self._bank.mean(name, j)

    def bandit_stats(self) -> dict:
        """Return multi-scale UCB diagnostics."""
        fast_arms = sum(1 for j in self._scale.values() if j < self._slowest)
        return {
            "ms_ucb_pulls": self._total_pulls,
            "ms_ucb_arms": len(self._scale),
            "ms_ucb_fast_arms": fast_arms,
            "ms_ucb_fast_reads": self._fast_reads,
        }
