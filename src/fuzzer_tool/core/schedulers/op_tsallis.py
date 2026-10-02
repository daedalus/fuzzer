"""TsallisINFScheduler: 1/2-Tsallis-INF over mutation operators.

Tsallis-INF (Zimmert & Seldin, *An Optimal Algorithm for Stochastic and
Adversarial Bandits*, JMLR 2021) is follow-the-regularised-leader under the
1/2-Tsallis entropy ``psi(p) = -sum_i 4 sqrt(p_i)``. Its closed form is

    p_i = 4 / (eta_t (L_i - x))^2,   eta_t = eta / sqrt(t)

with ``L_i`` the cumulative importance-weighted loss and ``x < min L`` the
normaliser making ``sum p = 1``. It is order-optimal on both stochastic and
adversarial losses without knowing which it faces -- the fit for fuzzing,
where operator yield looks stationary early and turns adversarial as
coverage saturates.

Where it sits among the regularisers here::

    exp3     Shannon entropy   p ~ exp(-eta L)          exponential decay
    tsallis  1/2-Tsallis       p ~ 4 / (eta (L - x))^2  polynomial decay
    corral   log-barrier       1/p = 1/p + eta (l - x)  polynomial decay

Loss estimate
-------------
The paper's reduced-variance estimator: the drawn arm gets
``b + (loss - b) / p`` and every other arm ``b``, with ``b = 1/2`` when
``p >= eta_t^2`` and 0 otherwise. The common ``b`` cancels in the
normaliser (``p`` depends on ``L - x`` only), so only the drawn arm's
``(loss - b) / p`` is accumulated: O(1) per record.

Why not ``op_corral.py``'s running-mean baseline? Measured: with b ~ 0.95
(the mean loss at fuzzing yields) a failure costs the drawn arm +0.05/p and
one lucky success -0.95/p, so equal arms lock in. Under uniform 5% reward
it left 94 of 252 operators never drawn in 20k rounds; b = 1/2 reached all
252 at the same stationary convergence (best-arm tail share min 0.932 vs
0.938 over 8 seeds).

On-policy only, like ``corral``: ``record`` credits an arm only against the
probability this scheduler drew it with; anything else is an orphan.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers._reward import unit_reward

#: Newton steps for the normaliser. Started where the smallest term alone is
#: 1, the iteration descends monotonically (convex, increasing residual) and
#: grows the gap ~1.5x per step until quadratic convergence: ~10 at K = 260.
_NEWTON_ITERS = 64

#: Residual on ``sum(w) - 1`` that ends the solve; renormalised afterwards.
_SOLVE_TOL = 1e-13

#: Cap on remembered unconsumed draws (see ``op_corral._PENDING_MAX``).
_PENDING_MAX = 256

#: The reduced-variance estimator's loss shift (Zimmert & Seldin 2021, §4).
_HALF_SHIFT = 0.5


def tsallis_probs(losses: np.ndarray, eta: float) -> np.ndarray:
    """1/2-Tsallis-INF distribution for cumulative *losses* at rate *eta*.

    Newton on ``f(x) = sum 4 / (eta (L_i - x))^2 - 1``. ``f`` is convex and
    increasing in ``x`` below ``min L``; from ``x0 = min L - 2 / eta`` (where
    ``f >= 0``) each tangent step lands at or right of the root, so ``x``
    decreases monotonically and never crosses the pole. ``dw/dx = eta w^1.5``.
    """
    u0 = losses - losses.min()  # shift-invariant; keeps magnitudes small
    gap = 2.0 / eta  # L_i - x for the leader at x0
    for _ in range(_NEWTON_ITERS):
        w = 4.0 / (eta * (u0 + gap)) ** 2
        f = float(w.sum()) - 1.0
        if f < _SOLVE_TOL:
            break
        gap += f / (eta * float((w * np.sqrt(w)).sum()))
    w = 4.0 / (eta * (u0 + gap)) ** 2
    return w / w.sum()


class TsallisINFScheduler:
    """1/2-Tsallis-INF bandit over mutation operators.

    Args:
        eta: Learning-rate scale; round t uses ``eta / sqrt(t)``. 2.0 is the
            paper's rate and the measured optimum: stationary best-arm tail
            share min 0.817 at 1.0, 0.932 at 2.0, 0.000 at 4.0 (lock-in).
        mix: Uniform floor in [0, 0.5) applied at selection. Off by default:
            Tsallis-INF's polynomial tail already keeps arms reachable.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: FTRL over importance-weighted losses has no Beta prior (Hard Rule 40).
    supports_priors = False

    def __init__(self, eta: float = 2.0, mix: float = 0.0, rng: RandPool | None = None) -> None:
        if eta <= 0.0:
            raise ValueError(f"eta must be positive, got {eta!r}")
        if not 0.0 <= mix < 0.5:
            raise ValueError(f"mix must be in [0, 0.5), got {mix!r}")
        self.eta = float(eta)
        self.mix = float(mix)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Array-backed per-arm state; _names[i] <-> _idx[name] == i.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._lossv = np.zeros(0, dtype=np.float64)
        self._pullv = np.zeros(0, dtype=np.int64)
        self._winv = np.zeros(0, dtype=np.float64)

        # Operator -> probability it was drawn with, awaiting its reward.
        self._pending: dict[str, float] = {}
        self._ballot: list[str] = []

        self._rounds = 0
        self._orphans = 0
        self._expired_draws = 0

    def init_arm(self, name: str) -> None:
        """Register an operator at zero cumulative loss (idempotent)."""
        if name in self._idx:
            return
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._lossv = np.append(self._lossv, 0.0)
        self._pullv = np.append(self._pullv, 0)
        self._winv = np.append(self._winv, 0.0)

    def learning_rate(self) -> float:
        """``eta / sqrt(t)`` with t the 1-based round about to be played."""
        return self.eta / math.sqrt(self._rounds + 1)

    def probabilities(self, ops: list[str]) -> dict[str, float]:
        """Draw distribution restricted to *ops*, registering anything new."""
        if not ops:
            return {}
        for op in ops:
            self.init_arm(op)
        self._ballot = list(ops)
        idx = [self._idx[op] for op in ops]
        p = tsallis_probs(self._lossv[idx], self.learning_rate())
        if self.mix > 0.0:
            p = (1.0 - self.mix) * p + self.mix / len(ops)
        return dict(zip(ops, p.tolist(), strict=True))

    def select_op(self, ops: list[str]) -> str:
        """Draw one operator by inverse CDF and remember its probability."""
        probs = self.probabilities(ops)
        if not probs:
            return ""
        r = self._rng.random()
        cumulative = 0.0
        chosen = ops[-1]
        for op in ops:
            cumulative += probs[op]
            if r < cumulative:
                chosen = op
                break

        # Drop the oldest unconsumed draw rather than grow without bound.
        if len(self._pending) >= _PENDING_MAX and chosen not in self._pending:
            self._pending.pop(next(iter(self._pending)))
            self._expired_draws += 1
        self._pending[chosen] = probs[chosen]
        return chosen

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Accumulate the drawn arm's baseline-shifted importance-weighted loss."""
        p_drawn = self._pending.pop(name, None)
        if p_drawn is None or p_drawn <= 0.0:
            self._orphans += 1
            return
        j = self._idx[name]

        # Reduced-variance shift: 1/2 once p clears eta^2 (Zimmert & Seldin).
        reward = unit_reward(success, weight)
        eta = self.learning_rate()
        b = _HALF_SHIFT if p_drawn >= eta * eta else 0.0
        self._lossv[j] += (1.0 - reward - b) / p_drawn

        self._rounds += 1
        self._pullv[j] += 1
        self._winv[j] += reward

    def bandit_stats(self) -> dict[str, Any]:
        """Convergence stats over the arms the last round offered."""
        live = [a for a in (self._ballot or self._names) if a in self._idx]
        probs = self.probabilities(live) if live else {}
        n = len(probs)
        ent = -sum(q * math.log(q) for q in probs.values() if q > 0.0)
        return {
            "rounds": self._rounds,
            "arms": n,
            "registered": len(self._names),
            "probs": probs,
            "pulls": {a: int(self._pullv[self._idx[a]]) for a in live},
            "wins": {a: float(self._winv[self._idx[a]]) for a in live},
            "concentration": max(probs.values()) if probs else 0.0,
            "entropy": (ent / math.log(n)) if n > 1 else 0.0,
            "learning_rate": self.learning_rate(),
            "orphan_records": self._orphans,
            "expired_draws": self._expired_draws,
        }
