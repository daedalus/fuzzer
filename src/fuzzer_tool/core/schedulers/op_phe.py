"""PHEScheduler: Perturbed-History Exploration over mutation operators.

PHE (Kveton, Szepesvari, Ghavamzadeh & Boutilier, *Perturbed-History
Exploration in Stochastic Multi-Armed Bandits*, IJCAI 2019) explores by
adding fake history, not by a confidence width or a posterior::

    s_i pulls, V_i total reward
    m_i = ceil(a * s_i)                pseudo-pulls
    U_i ~ Binomial(m_i, 1/2)           pseudo-rewards
    index_i = (V_i + U_i) / (s_i + m_i)        ->  argmax

An arm never pulled is played first. With ``a > 1`` the paper proves
O(K log n / Delta) regret for rewards in [0, 1]; the noise scales with the
arm's own history, so it shrinks as the arm is pulled, like Thompson
sampling, but needs no posterior: one vectorised Binomial draw per round.

Why it is not ``monte_carlo``: Beta-Bernoulli Thompson is exact only for
0/1 rewards. ``record`` receives fractional surprisal weights, and PHE's
Bernoulli(1/2) pseudo-rewards are valid for any reward bounded in [0, 1].

Priors: Beta(alpha, beta) enters as pseudo-history, ``alpha - 1``
successes in ``alpha + beta - 2`` pulls; Beta(1, 1) adds nothing.

Off-policy safe: history is history whoever drew it, so it records every
round like ``monte_carlo``.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers._reward import unit_reward

#: Probability of each pseudo-reward: the midpoint of [0, 1] (the paper).
_PSEUDO_P = 0.5


class PHEScheduler:
    """Perturbed-history exploration bandit.

    Args:
        a: Perturbation scale, pseudo-pulls per real pull, finite and > 0.
            The paper's regret bound needs ``a > 1``; 1.1 is its experimental
            default and the best of the values swept with a > 1 (convergence
            harness, 12 seeds, stationary best-arm tail share min):
            0.966 at 1.0, 0.968 at 1.1, 0.942 at 2.0, 0.879 at 4.0.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: init_arm turns a Beta(alpha, beta) prior into pseudo-history
    #: (Hard Rule 40).
    supports_priors = True

    def __init__(self, a: float = 1.1, rng: RandPool | None = None) -> None:
        if not (math.isfinite(a) and a > 0.0):
            raise ValueError(f"a must be finite and > 0, got {a!r}")
        self.a = float(a)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Array-backed history; _names[i] <-> _idx[name] == i.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._pullv = np.zeros(0, dtype=np.float64)
        self._rewv = np.zeros(0, dtype=np.float64)
        self._records = 0

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an arm; a Beta prior becomes pseudo-history (idempotent)."""
        if name in self._idx:
            return
        wins = max(float(prior_alpha) - 1.0, 0.0)
        losses = max(float(prior_beta) - 1.0, 0.0)
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._pullv = np.append(self._pullv, wins + losses)
        self._rewv = np.append(self._rewv, wins)

    def history(self, name: str) -> tuple[float, float]:
        """(pulls, total reward) of *name*, prior included."""
        self.init_arm(name)
        j = self._idx[name]
        return float(self._pullv[j]), float(self._rewv[j])

    def select_op(self, ops: list[str]) -> str:
        """Argmax of the perturbed-history mean; unpulled arms first."""
        if not ops:
            return ""
        for op in ops:
            self.init_arm(op)
        idx = np.fromiter((self._idx[op] for op in ops), dtype=np.int64, count=len(ops))
        pulls = self._pullv[idx]

        # Unpulled arms have no history to perturb: play the first one.
        fresh = np.flatnonzero(pulls <= 0.0)
        if fresh.size:
            return ops[int(fresh[0])]

        # One Binomial draw per arm; m grows with the arm's own history.
        m = np.ceil(self.a * pulls)
        u = np.asarray(self._rng.binomial_array(m.astype(np.int64), _PSEUDO_P), dtype=np.float64)
        return ops[int(np.argmax((self._rewv[idx] + u) / (pulls + m)))]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Append one real pull with reward in [0, 1] to *name*'s history."""
        self.init_arm(name)
        j = self._idx[name]
        self._pullv[j] += 1.0
        self._rewv[j] += unit_reward(success, weight)
        self._records += 1

    def bandit_stats(self) -> dict:
        """Return PHE diagnostics."""
        return {
            "phe_pulls": self._records,
            "phe_arms": len(self._names),
            "phe_unpulled": int((self._pullv <= 0.0).sum()),
        }
