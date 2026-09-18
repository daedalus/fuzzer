"""Softmax scheduler for operator selection.

Implementation mirrors the specification provided in the task description.
"""

from math import exp

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool


class SoftmaxScheduler:
    """SoftmaxScheduler: rank arms by softmax of their empirical mean reward.

    Args:
        tau: Temperature parameter (default 1.0).
        rng: Shared RandPool (Hard Rule 16).

    Supports the bandit interface used by the rest of the codebase.
    """

    supports_priors = False

    def __init__(self, tau: float = 1.0, rng: RandPool | None = None):
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        self.rng = rng if rng is not None else get_default_rand_pool()
        self._mean: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def init_arm(self, name: str) -> None:
        self._mean.setdefault(name, 0.0)
        self._counts.setdefault(name, 0)

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        self.init_arm(name)
        self._counts[name] += 1
        reward = weight if success else 0.0
        old = self._mean[name]
        # incremental mean update
        self._mean[name] = old + (reward - old) / self._counts[name]

    def select_op(self, ops: list[str]) -> str:
        """Select the candidate with the highest softmax probability.
        Unregistered candidates are registered on the fly (same
        just-in-time behavior as the other schedulers' select_op).
        Ties go to whichever candidate came first in ops.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        for op in ops:
            self.init_arm(op)
        # calculate prob-mass using softmax
        max_mean = max(self._mean[o] for o in ops)
        exps = [exp((self._mean[o] - max_mean) / self.tau) for o in ops]
        total = sum(exps)
        probs = [e / total for e in exps]
        if max_mean == min(self._mean[o] for o in ops):
            return ops[0]
        r = self.rng.random()
        accum = 0.0
        for op, p in zip(ops, probs, strict=True):
            accum += p
            if r <= accum:
                return op
        return ops[-1]

    def bandit_stats(self) -> dict[str, tuple[float, int]]:
        """Return (mean, count) for each registered arm."""
        return {
            op: (m, c)
            for op, m, c in zip(self._mean, self._mean.values(), self._counts.values(), strict=True)
        }


"""End of SoftmaxScheduler implementation."""
