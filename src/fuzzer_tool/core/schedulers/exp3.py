"""Exp3Scheduler: adversarial bandit (EXP3)."""

import math
import random


class Exp3Scheduler:
    """EXP3 adversarial bandit for operator selection.

    The Exponential-weight algorithm for Exploration and Exploitation (Auer et
    al. 2002) handles non-stationary reward distributions that violate the
    i.i.d. assumption of Beta-Bernoulli Thompson sampling.

    At each round:
        p_i = (1 - gamma) * w_i / sum(w)  +  gamma / K    (mixture)
        sample i ~ p
        receive reward r in [0, 1]
        r̂_i = r / p_i   (importance-weighted estimator)
        w_i = w_i * exp(gamma * r̂_i / K)

    Args:
        gamma: Exploration rate in [0, 1]. Higher = more uniform exploration.
        window_decay: Exponential decay per update (1.0 = no decay).
            Values < 1.0 discount older observations.
    """

    # Declares that init_arm() does NOT accept informative priors (EXP3
    # uses uniform weight initialization, not Beta-Bernoulli).
    supports_priors = False

    def __init__(self, gamma: float = 0.1, window_decay: float = 0.999):
        self.gamma = gamma
        self.window_decay = window_decay
        self.weights: dict[str, float] = {}
        self._total_pulls: int = 0
        # Per-iteration selection probabilities — needed for importance-weighted
        # estimator in record().  select_op stores (op, p) here, record() reads it.
        self._last_probs: dict[str, float] = {}

    def init_arm(self, name: str) -> None:
        """Register an operator with initial weight 1.0."""
        if name not in self.weights:
            self.weights[name] = 1.0

    def select_op(self, ops: list[str]) -> str:
        """Select operator via EXP3 mixture distribution."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        K = len(ops)
        total_w = sum(self.weights.get(op, 1.0) for op in ops)
        if total_w <= 0:
            self._last_probs.clear()
            return random.choice(ops)

        # Build mixture: p = (1-γ) * w_i/Σw  +  γ/K
        probs: dict[str, float] = {}
        for op in ops:
            w = self.weights.get(op, 1.0)
            probs[op] = (1.0 - self.gamma) * (w / total_w) + self.gamma / K

        # Store probs for record() to use in the importance-weighted estimator
        self._last_probs = dict(probs)

        # Roulette-wheel selection
        r = random.random()
        cumulative = 0.0
        for op in ops:
            cumulative += probs[op]
            if r <= cumulative:
                return op
        return ops[-1]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome and update EXP3 weights.

        Uses the importance-weighted estimator: reward_estimate = r / p_i,
        where p_i is the probability this operator had when it was selected.
        """
        self._total_pulls += 1
        reward = weight if success else 0.0

        # Apply exponential decay to all weights (discounts old evidence)
        if self.window_decay < 1.0:
            for k in self.weights:
                self.weights[k] *= self.window_decay

        # EXP3 weight update: w_i *= exp(gamma * r̂_i / K)
        # r̂_i = reward / p_i  (importance-weighted)
        p = self._last_probs.get(name, 1.0 / max(len(self._last_probs), 1))
        K = max(len(self.weights), 1)
        estimated_reward = reward / max(p, 1e-9)
        self.weights[name] = self.weights.get(name, 1.0) * math.exp(
            self.gamma * estimated_reward / max(K, 1)
        )

        # Prevent floating-point blowup: renormalize if max weight is extreme
        max_w = max(self.weights.values())
        if max_w > 1e9:
            scale = 1.0 / max_w
            for k in self.weights:
                self.weights[k] *= scale

    def bandit_stats(self) -> dict:
        """Return EXP3 diagnostics."""
        return {
            "exp3_pulls": self._total_pulls,
            "exp3_max_weight": max(self.weights.values()) if self.weights else 0.0,
        }
