"""GradientBanditScheduler: softmax / Boltzmann preference bandit.

Sutton & Barto gradient bandit with a baseline. Preferences H_i are
updated by the REINFORCE-style rule:

    H_a ← H_a + α (R − R̄) (1 − π_a)     # chosen arm
    H_j ← H_j − α (R − R̄) π_j           # all other arms

where π = softmax(H / temperature). Temperature may be annealed so the
policy becomes greedier over time. No UCB term; pure preference gradient.

Complements EXP3 (adversarial weights) and ε-greedy (hard explore/exploit
switch) by keeping a soft distribution that can be temperature-scheduled.

References:
- Sutton & Barto, Reinforcement Learning: An Introduction, §2.8
  (Gradient Bandit Algorithms)
"""

from __future__ import annotations

import math

from fuzzer_tool.core.rand_pool import RandPool


class GradientBanditScheduler:
    """Softmax preference bandit (Boltzmann exploration) for operators.

    Args:
        alpha: Step size for preference updates. Typical range 0.05–0.3.
        temperature: Initial softmax temperature. Higher = more uniform.
        temp_decay: Multiplicative decay applied to temperature each pull.
            1.0 disables annealing. 0.9995 ≈ ε-greedy default schedule.
        min_temperature: Floor on temperature so the policy never becomes
            fully deterministic.
        baseline: If True (default), use an incremental average reward as
            the baseline R̄. If False, R̄ = 0 (pure REINFORCE).
        rng: Shared RandPool (Hard Rule 16).
    """

    supports_priors = False

    def __init__(
        self,
        alpha: float = 0.1,
        temperature: float = 1.0,
        temp_decay: float = 0.9995,
        min_temperature: float = 0.05,
        baseline: bool = True,
        rng: RandPool | None = None,
    ):
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha!r}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature!r}")
        if min_temperature <= 0.0:
            raise ValueError(
                f"min_temperature must be positive, got {min_temperature!r}"
            )

        self.alpha = alpha
        self._temperature0 = temperature
        self.temp_decay = temp_decay
        self.min_temperature = min_temperature
        self.use_baseline = baseline
        self._rng = rng if rng is not None else RandPool()

        # Preference weights H_i (unbounded, relative scale)
        self.preferences: dict[str, float] = {}
        self._total_pulls: int = 0

        # Running average reward for the baseline
        self._avg_reward: float = 0.0
        self._reward_count: int = 0

        # Last selection: needed so record() can apply the gradient only to
        # the arm that was actually drawn under this policy (and so the
        # π used in the update matches the π that produced the draw).
        self._last_op: str | None = None
        self._last_probs: dict[str, float] = {}

    def init_arm(self, name: str) -> None:
        """Register an operator with zero preference."""
        self.preferences.setdefault(name, 0.0)

    def _current_temperature(self) -> float:
        t = self._temperature0 * (self.temp_decay**self._total_pulls)
        return max(self.min_temperature, t)

    def _softmax(self, ops: list[str]) -> dict[str, float]:
        """Numerically stable softmax over preferences / temperature."""
        temp = self._current_temperature()
        # Shift by max for stability
        max_h = max(self.preferences.get(op, 0.0) for op in ops)
        exps: dict[str, float] = {}
        total = 0.0
        for op in ops:
            # Clamp the exponent to avoid overflow when temp is tiny
            z = (self.preferences.get(op, 0.0) - max_h) / temp
            if z < -50.0:
                e = 0.0
            elif z > 50.0:
                e = math.exp(50.0)
            else:
                e = math.exp(z)
            exps[op] = e
            total += e
        if total <= 0.0:
            # Degenerate: fall back to uniform
            u = 1.0 / len(ops)
            return {op: u for op in ops}
        return {op: e / total for op, e in exps.items()}

    def select_op(self, ops: list[str]) -> str:
        """Sample an operator from the current softmax policy."""
        if not ops:
            return ""
        if len(ops) == 1:
            self._last_op = ops[0]
            self._last_probs = {ops[0]: 1.0}
            return ops[0]

        for op in ops:
            self.init_arm(op)

        probs = self._softmax(ops)
        self._last_probs = dict(probs)

        r = self._rng.random()
        cumulative = 0.0
        chosen = ops[-1]
        for op in ops:
            cumulative += probs[op]
            if r <= cumulative:
                chosen = op
                break

        self._last_op = chosen
        return chosen

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Apply the gradient update for the arm that was last selected.

        Only the arm that this scheduler actually drew receives the
        preference gradient (and the complementary negative updates for
        the other arms in that draw). Shadow records for other
        schedulers' draws are ignored so the baseline and π stay
        consistent with the policy that produced the sample.
        """
        if name != self._last_op or not self._last_probs:
            return

        self._total_pulls += 1
        reward = weight if success else 0.0

        # Baseline is the average of *previous* rewards so the first
        # observation still produces a non-zero advantage.
        baseline = self._avg_reward if self.use_baseline and self._reward_count > 0 else 0.0
        advantage = reward - baseline

        if self.use_baseline:
            self._reward_count += 1
            self._avg_reward += (reward - self._avg_reward) / self._reward_count

        if advantage == 0.0:
            return

        # Classic gradient bandit update over the arms present at selection
        for op, pi in self._last_probs.items():
            h = self.preferences.get(op, 0.0)
            if op == name:
                self.preferences[op] = h + self.alpha * advantage * (1.0 - pi)
            else:
                self.preferences[op] = h - self.alpha * advantage * pi

    def last_selection_probs(self) -> dict[str, float]:
        """Mixture used for the most recent select_op (for diagnostics)."""
        return dict(self._last_probs)

    def bandit_stats(self) -> dict:
        """Return gradient-bandit diagnostics."""
        return {
            "gradient_pulls": self._total_pulls,
            "temperature": self._current_temperature(),
            "avg_reward": self._avg_reward,
            "n_arms": len(self.preferences),
            "best_op": (
                max(self.preferences, key=self.preferences.get)
                if self.preferences
                else None
            ),
        }
