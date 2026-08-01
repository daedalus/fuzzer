"""EpsilonGreedyScheduler: epsilon-greedy with annealing."""

import random


class EpsilonGreedyScheduler:
    """Epsilon-greedy bandit with exponential annealing.

    Classic multi-armed bandit: with probability epsilon explore uniformly,
    otherwise exploit the best-known arm. Epsilon decays exponentially:

        epsilon_t = max(min_epsilon, epsilon_0 * decay^t)

    Q-values use incremental sample-average updates:

        Q_i = Q_i + (reward - Q_i) / (n_i + 1)

    Args:
        epsilon_0: Initial exploration rate (1.0 = pure exploration).
        decay: Exponential decay factor per pull (0.9995 = ~9200 pulls to 0.01).
        min_epsilon: Floor on epsilon to maintain some exploration.
    """

    supports_priors = False

    def __init__(
        self,
        epsilon_0: float = 1.0,
        decay: float = 0.9995,
        min_epsilon: float = 0.01,
    ):
        self.epsilon_0 = epsilon_0
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.q_values: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._total_pulls: int = 0

    def init_arm(self, name: str) -> None:
        """Register an operator with zero initial Q and count."""
        self.q_values.setdefault(name, 0.0)
        self.counts.setdefault(name, 0)

    def select_op(self, ops: list[str]) -> str:
        """Select operator via epsilon-greedy with annealing.

        Returns a random operator with probability epsilon_t (explore),
        or the best-known operator otherwise (exploit).
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        epsilon = max(
            self.min_epsilon,
            self.epsilon_0 * (self.decay**self._total_pulls),
        )

        if epsilon > 0 and random.random() < epsilon:
            # Explore: uniform random
            return random.choice(ops)

        # Exploit: pick highest Q
        return max(ops, key=lambda o: self.q_values.get(o, 0.0))

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome and incrementally update Q-value.

        Uses the sample-average update:
            Q_new = Q_old + (reward - Q_old) / (n + 1)
        """
        self._total_pulls += 1
        reward = weight if success else 0.0

        n = self.counts.get(name, 0)
        q_current = self.q_values.get(name, 0.0)
        # Incremental update: Q = Q + (reward - Q) / (n + 1)
        self.q_values[name] = q_current + (reward - q_current) / (n + 1)
        self.counts[name] = n + 1

    def bandit_stats(self) -> dict:
        """Return epsilon-greedy diagnostics."""
        epsilon = max(
            self.min_epsilon,
            self.epsilon_0 * (self.decay**self._total_pulls),
        )
        return {
            "eps_greedy_pulls": self._total_pulls,
            "current_epsilon": epsilon,
            "best_op": (max(self.q_values, key=self.q_values.get) if self.q_values else None),
        }
