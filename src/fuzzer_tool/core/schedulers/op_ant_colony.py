"""AntColonyScheduler: MAX-MIN ant system over operator *sequences*.

Every other operator scheduler here scores operators one at a time. Ant
colony optimisation (Dorigo 1992; MAX-MIN variant, Stuetzle & Hoos 2000)
scores *transitions*: pheromone sits on the edge prev -> op, so "b pays
right after a" is learnable even when b alone is mediocre::

    P(op | prev) ~ tau(prev, op)^alpha * eta(op)^beta
    every record:  tau *= 1 - rho                    (evaporation)
    reward r > 0:  tau(prev, op) += r                (deposit)
    tau clamped to [tau_min, tau_max]; edges start at tau_max

``eta`` is the operator's Laplace success rate (Beta prior, default
Beta(1, 1)), the classic ACO heuristic. The MMAS bounds are what stop
stagnation: an edge never evaporates below ``tau_min``, so no transition
is ever unreachable, and no deposit run lifts one above ``tau_max``.

Chains: selection conditions on the operator this scheduler selected
last; deposits credit the edge from the operator recorded last. Both
streams follow the round's operator stack in order, so they name the
same edges. A virtual start node is the predecessor of the first
operator ever seen. Chains run across round boundaries, as in
``op_katz.py``'s transition graph.

Evaporation is lazy: one global scale multiplies the stored matrix (the
``op_exp3`` trick), so a record is O(1) and only the read clamps.

Not importance-weighted, so it records every round like ``op_katz``.
Memory is (K + 1)^2 floats: ~0.5 MB at the ~260 registered operators.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers._reward import unit_reward

#: Matrix row/column of the virtual start node; arm i lives at i + 1.
_START = 0

#: Global scale below which the stored matrix is folded back to scale 1.
_RENORM_SCALE = 1e-200


class AntColonyScheduler:
    """MAX-MIN ant system over operator transitions.

    Args:
        alpha: Pheromone exponent, finite and >= 0 (0 ignores pheromone).
        beta: Heuristic exponent, finite and >= 0. Above the textbook 2 on
            measurement (convergence harness, 8 seeds; stationary share
            min, regret slope max, DecayingBest late share min):
            beta 2 rho 0.02: 0.879 / 0.828 / 0.719; beta 3 rho 0.02: 0.956
            / 0.390 / 0.001; beta 4 rho 0.05: 0.962 / 0.343 / 0.704.
        rho: Evaporation per record, in (0, 1). Faster evaporation is what
            keeps beta 4 from locking in after a change (0.538 at 0.02).
        tau_min: Pheromone floor, > 0.
        tau_max: Pheromone cap and initial value, > ``tau_min``.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: init_arm seeds the eta heuristic with a Beta prior (Hard Rule 40).
    supports_priors = True

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 4.0,
        rho: float = 0.05,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        rng: RandPool | None = None,
    ) -> None:
        if not (math.isfinite(alpha) and alpha >= 0.0):
            raise ValueError(f"alpha must be finite and >= 0, got {alpha!r}")
        if not (math.isfinite(beta) and beta >= 0.0):
            raise ValueError(f"beta must be finite and >= 0, got {beta!r}")
        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must be in (0, 1), got {rho!r}")
        if not (0.0 < tau_min < tau_max):
            raise ValueError(f"need 0 < tau_min < tau_max, got {tau_min!r}, {tau_max!r}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Pheromone relative to _scale: actual tau = _scale * _tau[row, col].
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._tau = np.full((1, 1), self.tau_max)
        self._scale = 1.0

        # eta: Laplace success rate with a per-arm Beta prior.
        self._winv = np.zeros(0, dtype=np.float64)
        self._pullv = np.zeros(0, dtype=np.float64)
        self._prior_a = np.zeros(0, dtype=np.float64)
        self._prior_b = np.zeros(0, dtype=np.float64)

        # Chain positions (matrix indices): last selected and last recorded.
        self._sel_prev = _START
        self._rec_prev = _START
        self._records = 0

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an arm: fresh edges at tau_max, prior on eta (idempotent)."""
        if name in self._idx:
            return
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._tau = np.pad(self._tau, ((0, 1), (0, 1)), constant_values=self.tau_max / self._scale)
        self._winv = np.append(self._winv, 0.0)
        self._pullv = np.append(self._pullv, 0.0)
        self._prior_a = np.append(self._prior_a, max(float(prior_alpha), 1e-6))
        self._prior_b = np.append(self._prior_b, max(float(prior_beta), 1e-6))

    # -- reads ---------------------------------------------------------

    def _node(self, name: str | None) -> int:
        """Matrix index of *name*, or the start node for None."""
        if name is None:
            return _START
        self.init_arm(name)
        return self._idx[name] + 1

    def pheromone(self, prev: str | None, op: str) -> float:
        """Clamped tau on the edge *prev* -> *op* (None: start node)."""
        row, col = self._node(prev), self._node(op)
        tau = self._scale * float(self._tau[row, col])
        return min(max(tau, self.tau_min), self.tau_max)

    def heuristic(self, op: str) -> float:
        """eta(op): posterior-mean success rate."""
        j = self._node(op) - 1
        a, b = self._prior_a[j], self._prior_b[j]
        return float((self._winv[j] + a) / (self._pullv[j] + a + b))

    def _weights_at(self, row: int, idx: np.ndarray) -> np.ndarray:
        """tau^alpha * eta^beta from matrix row *row* to arm indices *idx*."""
        tau = np.clip(self._scale * self._tau[row, idx + 1], self.tau_min, self.tau_max)
        a, b = self._prior_a[idx], self._prior_b[idx]
        eta = (self._winv[idx] + a) / (self._pullv[idx] + a + b)
        return tau**self.alpha * eta**self.beta

    def weights(self, prev: str | None, ops: list[str]) -> list[float]:
        """Unnormalised transition weights from *prev* to each of *ops*."""
        row = self._node(prev)
        idx = np.fromiter((self._node(op) - 1 for op in ops), dtype=np.int64, count=len(ops))
        return self._weights_at(row, idx).tolist()

    # -- the scheduler interface ----------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Draw the next operator given the last one this colony selected."""
        if not ops:
            return ""
        idx = np.fromiter((self._node(op) - 1 for op in ops), dtype=np.int64, count=len(ops))
        cum = np.cumsum(self._weights_at(self._sel_prev, idx))
        pos = int(np.searchsorted(cum, self._rng.random() * cum[-1], side="right"))
        pos = min(pos, len(ops) - 1)
        self._sel_prev = int(idx[pos]) + 1
        return ops[pos]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Evaporate everywhere, then deposit r on the recorded edge."""
        col = self._node(name)
        j = col - 1
        r = unit_reward(success, weight)
        self._pullv[j] += 1.0
        self._winv[j] += r
        self._records += 1

        # Lazy evaporation: one multiply, folded back before underflow.
        self._scale *= 1.0 - self.rho
        if self._scale < _RENORM_SCALE:
            self._tau *= self._scale
            self._scale = 1.0

        # Deposit from the floor, capped at tau_max (MMAS bounds).
        if r > 0.0:
            row = self._rec_prev
            tau = max(self._scale * float(self._tau[row, col]), self.tau_min)
            self._tau[row, col] = min(tau + r, self.tau_max) / self._scale
        self._rec_prev = col

    def bandit_stats(self) -> dict:
        """Return ant-colony diagnostics."""
        n = len(self._names)
        live = np.clip(self._scale * self._tau[1:, 1:], self.tau_min, self.tau_max) if n else None
        return {
            "ant_colony_pulls": self._records,
            "ant_colony_arms": n,
            "ant_colony_edges_above_floor": int((live > self.tau_min).sum()) if n else 0,
        }
