"""SuccessiveEliminationScheduler: racing / successive-elimination bandit.

Maintains an active set of operators. Arms are pulled round-robin among
the active set. After each observation, any arm whose upper confidence
bound falls strictly below the best lower confidence bound is eliminated.

Hoeffding radius (rewards in [0, 1]):

    rad_i = sqrt( (log(t K / δ)) / (2 n_i) )

Eliminate i when  mean_i + rad_i  <  max_j (mean_j - rad_j).

Deterministic given the observation stream (no sampling noise in the
selection rule beyond the initial ordering). Useful when many operators
are low-yield: dead arms are pruned rather than kept on life support by
a perpetual UCB bonus.

References:
- Even-Dar, Mannor, Mansour, "Action Elimination and Stopping Conditions
  for the Multi-Armed Bandit and Reinforcement Learning Problems"
  (JMLR 2006)
- Karnin, Tomer, Long, "Almost Optimal Exploration in Multi-Armed
  Bandits" (ICML 2013) — successive elimination analysis
"""

from __future__ import annotations

import math

from fuzzer_tool.core.rand_pool import RandPool


class SuccessiveEliminationScheduler:
    """Successive-elimination (racing) bandit for mutation operators.

    Args:
        delta: Failure probability for the Hoeffding bound. Smaller =
            wider radii = slower elimination.
        min_pulls: Minimum observations per arm before it is eligible
            for elimination. Prevents early false cuts on sparse signal.
        reopen_interval: If > 0, every this many total pulls re-admit
            every eliminated arm (non-stationary hedge). 0 = pure SE,
            never reopen.
        rng: Shared RandPool (Hard Rule 16). Used only for tie-breaks
            when the active set is empty or when reopening shuffles.
    """

    supports_priors = False

    def __init__(
        self,
        delta: float = 0.1,
        min_pulls: int = 3,
        reopen_interval: int = 0,
        rng: RandPool | None = None,
    ):
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta!r}")
        if min_pulls < 1:
            raise ValueError(f"min_pulls must be >= 1, got {min_pulls!r}")
        if reopen_interval < 0:
            raise ValueError(f"reopen_interval must be >= 0, got {reopen_interval!r}")

        self.delta = delta
        self.min_pulls = min_pulls
        self.reopen_interval = reopen_interval
        self._rng = rng if rng is not None else RandPool()

        self._mean: dict[str, float] = {}
        self._n: dict[str, int] = {}
        self._active: set[str] = set()
        self._eliminated: set[str] = set()
        self._order: list[str] = []  # registration / RR order
        self._rr_index: int = 0
        self._total_pulls: int = 0
        self._last_reopen_at: int = 0

    def init_arm(self, name: str) -> None:
        """Register an operator; starts active with zero counts."""
        if name in self._mean:
            return
        self._mean[name] = 0.0
        self._n[name] = 0
        self._active.add(name)
        self._order.append(name)

    def _radius(self, n: int) -> float:
        """Hoeffding radius for an arm with n pulls."""
        if n <= 0:
            return 1.0
        # log(t K / δ); use total pulls and known arm count
        k = max(len(self._mean), 1)
        t = max(self._total_pulls, 1)
        # clamp argument of log to avoid log(0)
        arg = max(t * k / self.delta, 1.0)
        return math.sqrt(math.log(arg) / (2.0 * n))

    def _maybe_reopen(self) -> None:
        if self.reopen_interval <= 0:
            return
        if self._total_pulls - self._last_reopen_at < self.reopen_interval:
            return
        if not self._eliminated:
            self._last_reopen_at = self._total_pulls
            return
        # Re-admit everyone; keep counts (warm start)
        self._active |= self._eliminated
        self._eliminated.clear()
        self._last_reopen_at = self._total_pulls

    def _eliminate(self) -> None:
        """Drop active arms whose UCB is below the best LCB."""
        if len(self._active) <= 1:
            return

        # Only arms with enough pulls participate in the bound comparison
        eligible = [a for a in self._active if self._n.get(a, 0) >= self.min_pulls]
        if len(eligible) < 2:
            return

        best_lcb = -math.inf
        for a in eligible:
            n = self._n[a]
            lcb = self._mean[a] - self._radius(n)
            if lcb > best_lcb:
                best_lcb = lcb

        to_drop = []
        for a in eligible:
            n = self._n[a]
            ucb = self._mean[a] + self._radius(n)
            if ucb < best_lcb:
                to_drop.append(a)

        for a in to_drop:
            self._active.discard(a)
            self._eliminated.add(a)

    def select_op(self, ops: list[str]) -> str:
        """Round-robin among active arms that appear in *ops*."""
        if not ops:
            return ""
        if len(ops) == 1:
            self.init_arm(ops[0])
            return ops[0]

        for op in ops:
            self.init_arm(op)

        self._maybe_reopen()

        # Restrict to arms offered this call
        offered = set(ops)
        active_offered = [a for a in self._order if a in self._active and a in offered]
        if not active_offered:
            # Everything offered is eliminated — reopen them for this call
            for op in ops:
                if op in self._eliminated:
                    self._eliminated.discard(op)
                    self._active.add(op)
            active_offered = [a for a in self._order if a in offered]
            if not active_offered:
                active_offered = list(ops)

        # Round-robin
        idx = self._rr_index % len(active_offered)
        chosen = active_offered[idx]
        self._rr_index += 1
        return chosen

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Update empirical mean and run elimination."""
        if name not in self._mean:
            self.init_arm(name)

        self._total_pulls += 1
        reward = weight if success else 0.0
        # Clamp to [0, 1] so Hoeffding assumptions hold
        reward = min(1.0, max(0.0, reward))

        n = self._n[name]
        mu = self._mean[name]
        self._mean[name] = mu + (reward - mu) / (n + 1)
        self._n[name] = n + 1

        self._eliminate()

    def bandit_stats(self) -> dict:
        """Diagnostics: active/eliminated sets and pull counts."""
        return {
            "se_pulls": self._total_pulls,
            "se_active": len(self._active),
            "se_eliminated": len(self._eliminated),
            "se_arms": len(self._mean),
            "active_ops": sorted(self._active),
            "eliminated_ops": sorted(self._eliminated),
        }
