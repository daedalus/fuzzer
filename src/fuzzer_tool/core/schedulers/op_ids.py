"""IDSScheduler: variance-based Information-Directed Sampling over operators.

IDS (Russo & Van Roy, *Learning to Optimize via Information-Directed
Sampling*, Operations Research 2018) picks the action distribution pi
minimising the information ratio

    Psi(pi) = (pi . Delta)^2 / (pi . g)

with ``Delta_k`` the expected regret of arm k and ``g_k`` the information
it carries about which arm is optimal. Thompson sampling explores in
proportion to P(arm is optimal); IDS explores in proportion to what a pull
*teaches*. With ~260 operators most of which are near-zero-yield, Thompson
keeps sampling every arm whose posterior still overlaps the leader's; IDS
skips an arm whose outcome cannot change the decision, however uncertain.

Estimation, from M joint posterior samples theta (M x K) of Beta arms::

    A*(m)   = argmax_k theta[m, k]                 optimal arm per sample
    rho*    = mean_m max_k theta[m, k]             expected optimal reward
    Delta_k = rho* - mean_m theta[m, k]
    g_k     = sum_a P(A* = a) (E[theta_k | A* = a] - E[theta_k])^2

``g`` is the variance-based gain (Russo & Van Roy §6), a lower bound on the
mutual information that needs no entropy estimate.

The optimum is supported on at most two arms. For a pair (i, j) with
mixing q on i, Psi is quadratic-over-linear hence convex in q, and its
stationary point is ``q* = Delta_j / (Delta_i - Delta_j) - 2 g_j / (g_i - g_j)``;
clipping to [0, 1] gives the pair's minimum. Only Pareto-frontier arms
(no other arm has both a smaller gap and a larger gain) can be in it: a
dominating arm lowers the numerator and raises the denominator at every q.
The frontier is O(K log K) and typically a handful of arms; its pairs are
scored in one vectorised pass (3.2 ms -> 0.08 ms at K = 260, identical
optimum over 300 random instances).

Amortised: sampling and the pair search cost O(M K + K^2), so the policy
per offered list is cached and recomputed only after ``refresh`` records.
Off-policy safe, like any Beta posterior: records every round.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool

#: Policies kept at once, one per distinct offered operator list (the same
#: bound as ``op_exp3._TREE_CACHE_MAX``); evicted first-in first-out.
_POLICY_CACHE_MAX = 16

#: Floor on Beta parameters (as ``op_monte_carlo.MIN_BETA_PARAM``).
_MIN_BETA_PARAM = 1e-6


def ids_gap_gain(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regret gap and variance-based information gain per arm.

    Args:
        theta: (M, K) posterior samples, one row per joint draw.

    Returns:
        ``(Delta, g)``, each of length K.
    """
    m, k = theta.shape
    best = theta.argmax(axis=1)
    mean = theta.mean(axis=0)
    gap = theta.max(axis=1).mean() - mean

    # Conditional means E[theta | A* = a] for the optimal arms that occur:
    # one-hot (M x U) against theta gives the per-class sums in one matmul.
    classes, inv, counts = np.unique(best, return_inverse=True, return_counts=True)
    onehot = np.zeros((m, len(classes)))
    onehot[np.arange(m), inv] = 1.0
    cond = (onehot.T @ theta) / counts[:, None]
    gain = (counts / m) @ ((cond - mean) ** 2)
    return gap, gain


def _ratio(gap_mix: np.ndarray, gain_mix: np.ndarray) -> np.ndarray:
    """``gap^2 / gain`` with 0/0 -> 0 (a certain best arm) and x/0 -> inf."""
    num = gap_mix * gap_mix
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / gain_mix
    return np.where(gain_mix > 0.0, out, np.where(num > 0.0, math.inf, 0.0))


def _frontier(gap: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Indices of arms not dominated in (smaller gap, larger gain).

    Sorted by gap (ties: larger gain first), an arm survives only if its
    gain beats every arm before it.
    """
    order = np.lexsort((-gain, gap))
    g = gain[order]
    keep = np.empty(len(g), dtype=bool)
    keep[0] = True
    keep[1:] = g[1:] > np.maximum.accumulate(g)[:-1]
    return order[keep]


def ids_pair(gap: np.ndarray, gain: np.ndarray) -> tuple[int, int, float]:
    """Information-ratio-minimising pair ``(i, j)`` and P(play i) = q."""
    f = _frontier(gap, gain)
    i, j, q = _pair_search(gap[f], gain[f])
    return int(f[i]), int(f[j]), q


def _pair_search(gap: np.ndarray, gain: np.ndarray) -> tuple[int, int, float]:
    """Exhaustive K^2 pair search; ``ids_pair`` runs it on the frontier."""
    gi, gj = gap[:, None], gap[None, :]
    vi, vj = gain[:, None], gain[None, :]

    # Stationary point of the convex pair ratio, clipped; NaN/inf -> 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        q_star = gj / (gi - gj) - 2.0 * vj / (vi - vj)
    q_star = np.clip(np.nan_to_num(q_star, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    # Candidates in tie-break order: i alone, j alone, the mixture.
    k = len(gap)
    psi = np.empty((3, k, k))
    psi[0] = np.broadcast_to(_ratio(gap, gain)[:, None], (k, k))
    psi[1] = np.broadcast_to(_ratio(gap, gain)[None, :], (k, k))
    psi[2] = _ratio(q_star * gi + (1.0 - q_star) * gj, q_star * vi + (1.0 - q_star) * vj)

    c, i, j = np.unravel_index(int(np.argmin(psi)), psi.shape)
    q = (1.0, 0.0, float(q_star[i, j]))[c]
    return int(i), int(j), q


def _reward(success: bool, weight: float) -> float:
    """Reward in [0, 1]; NaN counts as 0, infinities clamp to the bounds."""
    if not success or math.isnan(weight):
        return 0.0
    return min(1.0, max(0.0, weight))


class IDSScheduler:
    """Variance-based IDS over Beta-Bernoulli operator posteriors.

    Args:
        samples: Joint posterior draws M per policy solve, >= 2.
        refresh: Records between policy recomputations, >= 1.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: init_arm accepts a Beta(alpha, beta) prior (Hard Rule 40).
    supports_priors = True

    def __init__(self, samples: int = 128, refresh: int = 32, rng: RandPool | None = None) -> None:
        if samples < 2:
            raise ValueError(f"samples must be >= 2, got {samples!r}")
        if refresh < 1:
            raise ValueError(f"refresh must be >= 1, got {refresh!r}")
        self.samples = int(samples)
        self.refresh = int(refresh)
        self._rng = rng if rng is not None else get_default_rand_pool()

        # Array-backed posteriors; _names[i] <-> _idx[name] == i.
        self._names: list[str] = []
        self._idx: dict[str, int] = {}
        self._alphav = np.zeros(0, dtype=np.float64)
        self._betav = np.zeros(0, dtype=np.float64)

        # tuple(ops) -> (record stamp, arm i, arm j, P(i)).
        self._policies: dict[tuple, tuple[int, str, str, float]] = {}
        self._records = 0
        self._solves = 0

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an arm with a Beta prior; first registration wins."""
        if name in self._idx:
            return
        self._idx[name] = len(self._names)
        self._names.append(name)
        self._alphav = np.append(self._alphav, max(float(prior_alpha), _MIN_BETA_PARAM))
        self._betav = np.append(self._betav, max(float(prior_beta), _MIN_BETA_PARAM))

    def posterior(self, name: str) -> tuple[float, float]:
        """Current Beta (alpha, beta) of *name*."""
        self.init_arm(name)
        j = self._idx[name]
        return float(self._alphav[j]), float(self._betav[j])

    def select_op(self, ops: list[str]) -> str:
        """Play the cached IDS pair for *ops*, re-solving when stale."""
        if not ops:
            return ""
        key = tuple(ops)
        pol = self._policies.get(key)
        if pol is None or self._records - pol[0] >= self.refresh:
            pol = self._solve(ops)
            self._policies.pop(key, None)
            if len(self._policies) >= _POLICY_CACHE_MAX:
                del self._policies[next(iter(self._policies))]
            self._policies[key] = pol

        _stamp, a, b, q = pol
        if q >= 1.0 or a == b:
            return a
        if q <= 0.0:
            return b
        return a if self._rng.random() < q else b

    def _solve(self, ops: list[str]) -> tuple[int, str, str, float]:
        """Sample M joint posteriors over *ops* and minimise the ratio."""
        for op in ops:
            self.init_arm(op)
        idx = np.fromiter((self._idx[op] for op in ops), dtype=np.int64, count=len(ops))
        shape = (self.samples, len(ops))
        theta = self._rng.betavariate_array(
            np.broadcast_to(self._alphav[idx], shape), np.broadcast_to(self._betav[idx], shape)
        )
        gap, gain = ids_gap_gain(np.asarray(theta, dtype=np.float64))
        i, j, q = ids_pair(gap, gain)
        self._solves += 1
        return self._records, ops[i], ops[j], q

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Fractional Bernoulli update: reward r adds r hits and 1 - r misses."""
        self.init_arm(name)
        j = self._idx[name]
        r = _reward(success, weight)
        self._alphav[j] += r
        self._betav[j] += 1.0 - r
        self._records += 1

    def bandit_stats(self) -> dict:
        """Return IDS diagnostics."""
        return {
            "ids_pulls": self._records,
            "ids_arms": len(self._names),
            "ids_solves": self._solves,
            "ids_cached_policies": len(self._policies),
        }
