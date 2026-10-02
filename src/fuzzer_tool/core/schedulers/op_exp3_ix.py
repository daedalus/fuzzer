"""EXP3IXScheduler: EXP3 with implicit exploration (Neu, NeurIPS 2015).

EXP3 (``op_exp3.py``) mixes ``gamma / K`` of uniform into the draw to bound
``1 / p`` in its estimator. IX instead adds ``gamma`` to the denominator::

    p_i      = softmax(-eta_t L_i)               no uniform mixing
    L_drawn += loss / (p_drawn + gamma_t)        implicit exploration
    eta_t    = scale * sqrt(2 ln K / (K t)),  gamma_t = ratio * eta_t

The estimate is biased low (optimistic) by ``p / (p + gamma)``, which is
what buys a high-probability regret bound -- EXP3's holds only in
expectation, and its estimate's variance is ``O(1 / p)``; IX's is
``O(1 / gamma)``. Losses are ``1 - reward``: an arm drawn and failing is
pushed down, which is the exploration.

Measured on the convergence harness (8 seeds, stationary best-arm tail
share min / median; DecayingBest late share median):

    form    scale  ratio   stationary      recovery
    loss    1      0.5     0.588 / 0.701   0.524     Neu's gamma = eta / 2
    loss    1      0.1     0.748 / 0.828   0.657     default
    loss    4      0.5     0.188 / 0.276   0.212
    gains   1      0.5     0.960 / 1.000   0.000     never recovers
    gains   16     0.5     0.000 / 1.000   0.000     wrong-arm lock-in

Loss form throughout: at fuzzing yields every arm's loss sits near 1, so
the optimism ``p / (p + gamma)`` dominates the signal at Neu's ratio, and
a larger ``eta`` only amplifies ``1 / p`` swings. The gains form
(``G += r / (p + gamma)``) is pessimistic for rarely drawn arms and locks.

Separate from ``op_exp3.py`` rather than a mode on it: that module's
Fenwick sampler is built around multiplicative weights with window decay;
IX is an FTRL softmax over cumulative losses with an anytime rate.

On-policy only, like ``exp3`` and ``corral``: the estimate divides by the
probability this scheduler drew the arm with.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers._reward import unit_reward

#: Cap on remembered unconsumed draws (see ``op_corral._PENDING_MAX``).
_PENDING_MAX = 256


class EXP3IXScheduler:
    """EXP3-IX over mutation operators.

    Args:
        eta_scale: Multiplier on Neu's anytime rate, finite and > 0.
        gamma_ratio: ``gamma_t / eta_t``, finite and >= 0; Neu uses 0.5.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: Exponential weights over losses have no Beta prior (Hard Rule 40).
    supports_priors = False

    def __init__(
        self, eta_scale: float = 1.0, gamma_ratio: float = 0.1, rng: RandPool | None = None
    ) -> None:
        if not (math.isfinite(eta_scale) and eta_scale > 0.0):
            raise ValueError(f"eta_scale must be finite and > 0, got {eta_scale!r}")
        if not (math.isfinite(gamma_ratio) and gamma_ratio >= 0.0):
            raise ValueError(f"gamma_ratio must be finite and >= 0, got {gamma_ratio!r}")
        self.eta_scale = float(eta_scale)
        self.gamma_ratio = float(gamma_ratio)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Array-backed cumulative loss estimates; _names[i] <-> _idx[name] == i.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._lossv = np.zeros(0, dtype=np.float64)

        # Operator -> (p drawn with, gamma at the draw), awaiting its reward.
        self._pending: dict[str, tuple[float, float]] = {}
        self._rounds = 0
        self._orphans = 0

    def init_arm(self, name: str) -> None:
        """Register an operator at zero cumulative loss (idempotent)."""
        if name in self._idx:
            return
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._lossv = np.append(self._lossv, 0.0)

    def rates(self, k: int) -> tuple[float, float]:
        """``(eta_t, gamma_t)`` for *k* offered arms at the next round."""
        k = max(k, 2)  # ln 1 = 0 would freeze a single-arm ballot's rate
        eta = self.eta_scale * math.sqrt(2.0 * math.log(k) / (k * (self._rounds + 1)))
        return eta, eta * self.gamma_ratio

    def _law(self, ops: list[str]) -> np.ndarray:
        """``softmax(-eta_t L)`` over *ops* as an array, registering anything new."""
        for op in ops:
            self.init_arm(op)
        losses = self._lossv[[self._idx[op] for op in ops]]
        eta, _ = self.rates(len(ops))

        # Shift by the minimum: the leader's weight is exp(0) = 1, no overflow.
        w = np.exp(-eta * (losses - losses.min()))
        return w / w.sum()

    def probabilities(self, ops: list[str]) -> dict[str, float]:
        """``softmax(-eta_t L)`` over *ops*."""
        if not ops:
            return {}
        return dict(zip(ops, self._law(ops).tolist(), strict=True))

    def select_op(self, ops: list[str]) -> str:
        """Draw one operator by inverse CDF; remember (p, gamma) for record."""
        if not ops:
            return ""
        p = self._law(ops)
        pos = int(np.searchsorted(np.cumsum(p), self._rng.random(), side="right"))
        pos = min(pos, len(ops) - 1)
        chosen = ops[pos]

        # Drop the oldest unconsumed draw rather than grow without bound.
        if len(self._pending) >= _PENDING_MAX and chosen not in self._pending:
            self._pending.pop(next(iter(self._pending)))
        self._pending[chosen] = (float(p[pos]), self.rates(len(ops))[1])
        return chosen

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Add the IX loss estimate to the arm this scheduler drew."""
        drawn = self._pending.pop(name, None)
        if drawn is None:
            self._orphans += 1
            return
        p, gamma = drawn
        loss = 1.0 - unit_reward(success, weight)
        self._lossv[self._idx[name]] += loss / (p + gamma)
        self._rounds += 1

    def bandit_stats(self) -> dict:
        """Return EXP3-IX diagnostics."""
        return {
            "exp3_ix_pulls": self._rounds,
            "exp3_ix_arms": len(self._names),
            "exp3_ix_orphans": self._orphans,
        }
