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

**Two defects found post-hoc by testing against
``tests/support/bandit_env.py`` (the harness ``test_scheduler_convergence.py``
uses to gate every other scheduler in this tree) -- fixed here, documented
so they don't come back:**

1. The baseline was a plain sample average (``avg += (R - avg) / n``),
   which never forgets: after a long stable run it takes proportionally
   longer to reflect a regime change than it did to build up, since a 1/n
   average weights the newest sample identically to the oldest. Fixed to
   an exponentially-weighted update at the same step size as the
   preference update (``avg += alpha * (R - avg)``), consistent with a
   scheduler whose whole point is to track drift via a constant step size
   rather than a window.

2. Worse, and the reason the fix above was not sufficient by itself:
   ``min_temperature`` does not actually floor any arm's *selection
   probability*. It floors the temperature, but the exponent clamp at
   ``z > 50`` means a preference gap of only a few units at the minimum
   temperature (0.05 by default) already saturates the softmax to
   effectively one-hot -- probabilities on the order of 1e-20 for the
   losing arm, not the residual few-percent a "floor" name implies.
   Measured on this project's own ``StationaryBernoulli``/``DecayingBest``
   harness (seed 92) before this fix: the *stationary* case converged to
   ``tail_share == 1.0`` (zero residual exploration, not even the mild
   hedging every other scheduler in ``RELIABLE`` keeps), and the
   *decaying* case's tail share on the revived best arm was ``0.0`` --
   not slow recovery, no recovery, because a fully collapsed softmax has
   no gradient left to escape with (``pi*(1-pi) -> 0`` at both ends).
   Fixed with an explicit uniform-probability floor mixed in *after* the
   temperature-scaled softmax, the same fix ``Exp3Scheduler`` in this tree
   already applies for the identical reason: ``pi_final = (1-floor) *
   softmax + floor/K``. This bounds every arm's selection probability, and
   therefore its update magnitude in ``record()``, away from zero
   regardless of how confident the preferences have become or how low
   temperature has annealed.

   Even after both fixes, recovery within a realistic round budget is
   still not reliable in this project's actual ~12-arm ``DecayingBest``
   environment (tail share on the revived arm topped out around 0.04-0.05
   even at floor=0.5, which also wrecks stationary performance) -- this
   is a structural property of preference-gradient methods without an
   explicit forgetting term, not a tuning bug, and mirrors why
   ``EpsilonGreedy``/``MonteCarlo`` are also in that harness's ``STUCK``
   table rather than ``RECOVERS``. This scheduler is a legitimate,
   competitive *stationary* control arm (see ``RELIABLE`` in
   ``test_scheduler_convergence.py``) and is not a non-stationary/rotting-
   bandit solution; do not point campaigns expecting operator fatigue at
   it. It is Elo-only (excluded from ``_FALLBACK_PRECEDENCE``, see
   ``operators.py``) for that reason -- the same discipline
   ``core/schedulers/op_katz.py`` documents for an earlier unproven arm.
"""

from __future__ import annotations

import math

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool


class GradientBanditScheduler:
    """Softmax preference bandit (Boltzmann exploration) for operators.

    Args:
        alpha: Step size for preference updates. Typical range 0.05–0.3.
            Also used as the baseline's forgetting rate when ``baseline``
            is True -- see module docstring defect (1). Default lowered
            from an earlier 0.1: at 0.1, one seed in a 40-seed stationary
            sweep on this project's harness collapsed to tail_share 0.004
            on the true best arm (a 12-arm case with a 0.30 vs 0.18 gap,
            not a close call) while every neighboring alpha tried (0.05,
            0.15) converged normally on that same seed -- a narrow
            resonance between this exact step size and that draw
            sequence, not a close-margin problem. 0.05 passed all 40
            seeds with room (min share 0.938); kept as the default rather
            than chasing the specific seed, since the mechanism (REINFORCE
            preference swings compounding early, more likely at a larger
            fixed step) predicts other seeds or environments could hit
            the same resonance at 0.1.
        temperature: Initial softmax temperature. Higher = more uniform.
        temp_decay: Multiplicative decay applied to temperature each pull.
            1.0 disables annealing. 0.9995 ≈ ε-greedy default schedule.
        min_temperature: Floor on temperature. Despite the name this does
            NOT floor any arm's selection probability -- see ``floor``
            below and module docstring defect (2).
        floor: Uniform probability floor in [0, 1) mixed into the softmax
            after temperature scaling: ``pi = (1-floor)*softmax + floor/K``.
            Without this, a wide enough preference gap collapses every
            arm's update magnitude to zero together and the policy cannot
            recover from a regime switch in any practical number of
            rounds, regardless of temperature. 0.0 disables it and
            reproduces the original (broken, see module docstring)
            behavior.
        baseline: If True (default), use a recency-weighted average reward
            as the baseline R̄. If False, R̄ = 0 (pure REINFORCE).
        rng: Shared RandPool (Hard Rule 16).
    """

    supports_priors = False

    def __init__(
        self,
        alpha: float = 0.05,
        temperature: float = 1.0,
        temp_decay: float = 0.9995,
        min_temperature: float = 0.05,
        floor: float = 0.05,
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
        if not (0.0 <= floor < 1.0):
            raise ValueError(f"floor must be in [0, 1), got {floor!r}")

        self.alpha = alpha
        self._temperature0 = temperature
        self.temp_decay = temp_decay
        self.min_temperature = min_temperature
        self.floor = floor
        self.use_baseline = baseline
        self._rng = rng if rng is not None else get_default_rand_pool()

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
        probs = {op: e / total for op, e in exps.items()}
        if self.floor <= 0.0:
            return probs
        # Uniform-probability floor -- see module docstring defect (2) for
        # why min_temperature alone cannot keep this from collapsing to a
        # one-hot policy with no update gradient left to escape it.
        K = len(ops)
        share = self.floor / K
        return {op: (1.0 - self.floor) * p + share for op, p in probs.items()}

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

        # Baseline is a recency-weighted average of prior rewards (same
        # step size as the preference update, see module docstring defect
        # (1)) so the first observation still produces a non-zero
        # advantage and old evidence forgets at a fixed rate rather than
        # never.
        baseline = self._avg_reward if self.use_baseline and self._reward_count > 0 else 0.0
        advantage = reward - baseline

        if self.use_baseline:
            self._reward_count += 1
            if self._reward_count == 1:
                self._avg_reward = reward
            else:
                self._avg_reward = self._avg_reward + self.alpha * (reward - self._avg_reward)

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
