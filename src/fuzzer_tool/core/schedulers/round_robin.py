"""Round Robin scheduler: deterministic operator cycling."""


class RoundRobinScheduler:
    """Simple round-robin scheduler for operator selection.

    Cycles through available operators in fixed registration order.
    Provides deterministic baseline for Elo meta-scheduler competition.
    """

    supports_priors = False  # No meaningful priors for round-robin

    def __init__(self):
        self._index = 0
        self._operator_counts = {}  # name -> [successes, failures]
        self._operator_order = []  # maintained registration order

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an operator with zero initial counts."""
        if name not in self._operator_counts:
            self._operator_counts[name] = [0, 0]  # [successes, failures]
            self._operator_order.append(name)

    def select_op(self, ops: list[str]) -> str:
        """Select operator via round-robin cycling."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        # Filter to only registered operators in preferred order
        available = [op for op in self._operator_order if op in ops]
        if not available:
            # Fallback to registration order if none match available
            available = [op for op in ops if op in self._operator_counts]
        if not available:
            # Ultimate fallback to first op
            return ops[0]

        # Select current operator and advance index
        op = available[self._index % len(available)]
        self._index += 1
        return op

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome for Elo compatibility."""
        if name not in self._operator_counts:
            self._operator_counts[name] = [0, 0]
        if success:
            self._operator_counts[name][0] += weight
        else:
            self._operator_counts[name][1] += 1

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Return success/failure counts for each arm."""
        result = {}
        for name in sorted(self._operator_counts):
            successes, failures = self._operator_counts[name]
            result[name] = (float(successes), float(failures))
        return result
