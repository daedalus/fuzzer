"""OpQLearnScheduler: tabular Q-learning over operator chains (Watkins 1989).

State = last recorded operator, action = next operator. One Bellman step per
record, off-policy, so every executed op teaches the table whoever picked it::

    Q[s, a] += alpha * (r + gamma * max_b Q[a, b] - Q[s, a])

    record(flip) record(havoc, ok)  ->  Q[flip, havoc] rises

Selection is epsilon-greedy over the current state's row; unseen pairs are 0
and ties keep the first candidate. Bandits score ops context-free; this one
learns which op pays *after* which. Table is ops x ops, bounded by the
registry. Experimental, Elo-only (absent from ``_FALLBACK_PRECEDENCE``).
"""

from __future__ import annotations

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts, reward

#: State before the first record.
START_STATE = ""


class OpQLearnScheduler(ArmCounts):
    """Epsilon-greedy over Q[last op, next op]."""

    #: Q starts at 0 for every pair; no prior override.
    supports_priors = False

    def __init__(self, rng, alpha: float = 0.1, gamma: float = 0.5, epsilon: float = 0.1) -> None:
        if rng is None:
            raise ValueError("OpQLearnScheduler requires a RandPool (Hard Rule 16)")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        super().__init__()
        self._rng = rng
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._q: dict[str, dict[str, float]] = {}  # state -> {action: Q}
        self._state = START_STATE

    def q_value(self, state: str, op: str) -> float:
        """Q[state, op]; 0 when unseen."""
        return self._q.get(state, {}).get(op, 0.0)

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        super().record(name, success, weight)

        # Bellman step on (state, name), bootstrapping from name's row.
        row = self._q.setdefault(self._state, {})
        old = row.get(name, 0.0)
        future = max(self._q.get(name, {}).values(), default=0.0)
        row[name] = old + self._alpha * (reward(success, weight) + self._gamma * future - old)
        self._state = name

    def select_op(self, ops: list[str]) -> str:
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        if self._epsilon > 0.0 and self._rng.random() < self._epsilon:
            return self._rng.choice(ops)

        row = self._q.get(self._state, {})
        return max(ops, key=lambda op: row.get(op, 0.0))
