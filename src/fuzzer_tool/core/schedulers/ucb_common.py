"""Shared skeleton for the discounted and windowed UCB schedulers.

Four schedulers (DUCB, KL-DUCB, SW-UCB, KL-SW-UCB) share the same algorithm
structure: init_arm, select_op with unpulled-arm priority, record with
total_pulls counting, and bandit_stats. They differ in state representation
(discounted vs windowed) and width formula (Gaussian vs KL), which the two
intermediate bases capture.
"""

from __future__ import annotations

import collections
import math
from abc import ABC, abstractmethod

from fuzzer_tool.core.rand_pool import RandPool

MIN_LOG_ARG = 1.0 + 1e-9


class UCBBase(ABC):
    """Abstract base: defines the shared algorithm skeleton.

    Subclasses provide the state-specific details via _arm_score() and
    bandit_stats().
    """

    supports_priors = False

    def __init__(self, rng: RandPool | None = None) -> None:
        self._rng = rng if rng is not None else RandPool()
        self._total_pulls: int = 0

    # -- arm bookkeeping --------------------------------------------------

    @abstractmethod
    def init_arm(self, name: str) -> None:
        """Register an operator with zero evidence."""
        ...

    # -- selection --------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Select the operator with the highest index.

        Arms with zero evidence are opened first (standard UCB init, also
        handles runtime-registered operators).
        """
        if not ops:
            return ""

        if len(ops) == 1:
            return ops[0]

        unpulled = [op for op in ops if self._arm_count(op) <= 0.0]
        if unpulled:
            return self._rng.choice(unpulled)

        n_total = sum(self._arm_count(op) for op in ops)
        log_n = math.log(max(n_total, MIN_LOG_ARG))

        best_op = ops[0]
        best_score = -math.inf
        for op in ops:
            n = self._arm_count(op)
            mean = self._arm_mean(op, n)
            score = mean + self._width(mean, n, log_n)
            if score > best_score:
                best_score = score
                best_op = op

        return best_op

    # -- update -----------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Credit *name* with this pull's reward."""
        self._total_pulls += 1
        reward = weight if success else 0.0
        self._arm_update(name, reward)

    # -- diagnostics ------------------------------------------------------

    @abstractmethod
    def bandit_stats(self) -> dict:
        """Return scheduler-specific diagnostics."""
        ...

    # -- hooks for subclasses ---------------------------------------------

    @abstractmethod
    def _arm_count(self, op: str) -> float:
        """Current evidence count for *op* (absolute units)."""
        ...

    @abstractmethod
    def _arm_mean(self, op: str, n: float) -> float:
        """Empirical mean reward for *op* given its count *n*."""
        ...

    @abstractmethod
    def _width(self, mean: float, n: float, log_n: float) -> float:
        """Confidence width added to one arm's index."""
        ...

    @abstractmethod
    def _arm_update(self, name: str, reward: float) -> None:
        """Incorporate one pull's reward into the state."""
        ...


class DiscountedUCBBase(UCBBase):
    """Base for schedulers that track exponentially discounted statistics."""

    def __init__(self, gamma: float, rng: RandPool | None = None) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
        self.gamma = gamma
        self._n_rel: dict[str, float] = {}
        self._x_rel: dict[str, float] = {}
        self._discount: float = 1.0
        super().__init__(rng=rng)

    # -- arm bookkeeping --------------------------------------------------

    def init_arm(self, name: str) -> None:
        self._n_rel.setdefault(name, 0.0)
        self._x_rel.setdefault(name, 0.0)

    def _renormalise(self) -> None:
        """Fold accumulated discount back into per-arm statistics."""
        d = self._discount
        for k in self._n_rel:
            self._n_rel[k] *= d
        for k in self._x_rel:
            self._x_rel[k] *= d
        self._discount = 1.0

    # -- hooks ------------------------------------------------------------

    def _arm_count(self, op: str) -> float:
        return self._n_rel.get(op, 0.0) * self._discount

    def _arm_mean(self, op: str, n: float) -> float:
        if n <= 0.0:
            return 0.0
        return (self._x_rel.get(op, 0.0) * self._discount) / n

    def _arm_update(self, name: str, reward: float) -> None:
        if self.gamma < 1.0:
            self._discount *= self.gamma
            if self._discount < 1e-12:
                self._renormalise()
        inv = 1.0 / self._discount
        self._n_rel[name] = self._n_rel.get(name, 0.0) + inv
        if reward:
            self._x_rel[name] = self._x_rel.get(name, 0.0) + reward * inv

    def discounted_counts(self) -> dict[str, float]:
        """Per-arm discounted pull count N_t(i), in absolute units."""
        d = self._discount
        return {k: v * d for k, v in self._n_rel.items()}

    def discounted_means(self) -> dict[str, float]:
        """Per-arm discounted empirical mean X_t(i)/N_t(i)."""
        d = self._discount
        out = {}
        for k, n_rel in self._n_rel.items():
            n = n_rel * d
            out[k] = (self._x_rel.get(k, 0.0) * d / n) if n > 0 else 0.0
        return out


class WindowedUCBBase(UCBBase):
    """Base for schedulers that track a fixed-size sliding window."""

    def __init__(self, window: int, rng: RandPool | None = None) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window!r}")
        self.window = window
        self._history: collections.deque = collections.deque()  # type: ignore[name-defined]
        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}
        self._known: set[str] = set()
        super().__init__(rng=rng)

    # -- arm bookkeeping --------------------------------------------------

    def init_arm(self, name: str) -> None:
        self._known.add(name)

    def _evict(self) -> None:
        """Drop the oldest pull, subtracting from that arm's statistics."""
        old_name, old_reward = self._history.popleft()
        remaining = self._counts.get(old_name, 0) - 1
        if remaining <= 0:
            self._counts.pop(old_name, None)
            self._sums.pop(old_name, None)
            return
        self._counts[old_name] = remaining
        if not old_reward:
            return
        new_sum = self._sums.get(old_name, 0.0) - old_reward
        self._sums[old_name] = new_sum if new_sum > 0.0 else 0.0

    # -- hooks ------------------------------------------------------------

    def _arm_count(self, op: str) -> float:
        return float(self._counts.get(op, 0))

    def _arm_mean(self, op: str, n: float) -> float:
        if n <= 0.0:
            return 0.0
        return self._sums.get(op, 0.0) / n

    def _arm_update(self, name: str, reward: float) -> None:
        self._history.append((name, reward))
        self._counts[name] = self._counts.get(name, 0) + 1
        if reward:
            self._sums[name] = self._sums.get(name, 0.0) + reward
        while len(self._history) > self.window:
            self._evict()

    def windowed_counts(self) -> dict[str, int]:
        """Per-arm pull count inside the current window."""
        return dict(self._counts)

    def windowed_means(self) -> dict[str, float]:
        """Per-arm empirical mean inside the current window."""
        return {k: self._sums.get(k, 0.0) / n for k, n in self._counts.items() if n > 0}
