"""KL_SWUCBScheduler: KL-UCB over sliding-window statistics.

Same windowed structure as ``SWUCBScheduler`` (Garivier & Moulines),
but the confidence width is the empirical-Bernoulli KL upper bound
instead of the Gaussian form.
"""

import collections
import math

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._kl_ucb import kl_upper_bound

MIN_LOG_ARG = 1.0 + 1e-9


class KL_SWUCBScheduler:
    """Sliding-Window UCB with KL-UCB confidence bound.

    Args:
        window: tau, the number of most recent pulls that count.
        xi: Exploration constant inside the KL budget ``xi * log(n) / N``.
        b: Reward range (unused by the KL width, kept for API parity).
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    supports_priors = False

    def __init__(
        self,
        window: int = 4000,
        xi: float = 0.15,
        b: float = 1.0,
        rng: RandPool | None = None,
    ):
        if window <= 0:
            raise ValueError(f"window must be positive, got {window!r}")
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi!r}")

        self.window = window
        self.xi = xi
        self.b = b
        self._rng = rng if rng is not None else RandPool()

        self._history: collections.deque = collections.deque()
        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}
        self._known: set[str] = set()
        self._total_pulls: int = 0

    # -- arm bookkeeping --------------------------------------------------

    def init_arm(self, name: str) -> None:
        """Register an operator. Windowed statistics start empty by design."""
        self._known.add(name)

    # -- selection --------------------------------------------------------

    def select_op(self, ops: list[str]) -> str:
        """Select the operator with the highest windowed-KL-UCB index."""
        if not ops:
            return ""

        if len(ops) == 1:
            return ops[0]

        n_total = 0
        unpulled = []
        for op in ops:
            n = self._counts.get(op, 0)
            if n <= 0:
                unpulled.append(op)
                continue
            n_total += n

        if unpulled:
            return self._rng.choice(unpulled)

        log_n = math.log(max(float(n_total), MIN_LOG_ARG))

        best_op = ops[0]
        best_score = -math.inf
        for op in ops:
            n = self._counts[op]
            mean = self._sums.get(op, 0.0) / n
            score = mean + self._width(mean, n, log_n)
            if score > best_score:
                best_score = score
                best_op = op

        return best_op

    def _width(self, mean: float, n: int, log_n: float) -> float:
        """KL-UCB confidence width: smallest q >= mean with KL(mean||q) >= xi*log(n)/n."""
        return kl_upper_bound(mean, self.xi * log_n / n) - mean

    # -- update -----------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Append this pull to the window, evicting anything older than tau."""
        self._total_pulls += 1
        reward = weight if success else 0.0

        self._history.append((name, reward))
        self._counts[name] = self._counts.get(name, 0) + 1
        if reward:
            self._sums[name] = self._sums.get(name, 0.0) + reward

        while len(self._history) > self.window:
            self._evict()

    def _evict(self) -> None:
        """Drop the oldest pull, subtracting it from that arm's statistics."""
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

    # -- diagnostics ------------------------------------------------------

    def windowed_counts(self) -> dict[str, int]:
        """Per-arm pull count inside the current window."""
        return dict(self._counts)

    def windowed_means(self) -> dict[str, float]:
        """Per-arm empirical mean inside the current window."""
        return {k: self._sums.get(k, 0.0) / n for k, n in self._counts.items() if n > 0}

    def bandit_stats(self) -> dict:
        """Return KL-SW-UCB diagnostics."""
        return {
            "kl_swucb_pulls": self._total_pulls,
            "kl_swucb_window_fill": len(self._history),
            "kl_swucb_arms_in_window": len(self._counts),
        }
