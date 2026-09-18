"""Top-K scheduler for operator selection.

Implementation based on the provided specification.
"""

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool


class TopKScheduler:
    """TopKScheduler: keep a running weighted mean for each arm and at each round
    select uniformly among the top-k arms by that mean.

    Args:
        k: Number of top arms to select from (default 1).
        rng: Shared RandPool (Hard Rule 16).
    """

    supports_priors = False

    def __init__(self, k: int = 1, rng: RandPool | None = None):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
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
        """Select uniformly from the top-k arms by mean reward.
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
        # sort by mean reward (descending)
        sorted_ops = sorted(ops, key=lambda op: -self._mean[op])
        # select from top-k
        top_k = sorted_ops[: min(self.k, len(sorted_ops))]
        return self.rng.choice(top_k)

    def bandit_stats(self) -> dict[str, tuple[float, int]]:
        """Return (mean, count) for each registered arm."""
        return {
            op: (m, c)
            for op, m, c in zip(
                self._mean, self._mean.values(), self._counts.values(), strict=False
            )
        }


"""End of TopKScheduler implementation."""
