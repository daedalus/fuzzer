"""KL_SWUCBScheduler: KL-UCB over sliding-window statistics.

Same windowed structure as ``SWUCBScheduler`` (Garivier & Moulines),
but the confidence width is the empirical-Bernoulli KL upper bound
instead of the Gaussian form.
"""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._kl_ucb import kl_upper_bound
from fuzzer_tool.core.schedulers.ucb_common import WindowedUCBBase


class KL_SWUCBScheduler(WindowedUCBBase):
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
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi!r}")
        self.xi = xi
        self.b = b
        super().__init__(window=window, rng=rng)

    def _width(self, mean: float, n: float, log_n: float) -> float:
        """KL-UCB width: smallest q >= mean with KL(mean||q) >= xi*log(n)/n."""
        return kl_upper_bound(mean, self.xi * log_n / n) - mean

    def bandit_stats(self) -> dict:
        """Return KL-SW-UCB diagnostics."""
        return {
            "kl_swucb_pulls": self._total_pulls,
            "kl_swucb_window_fill": len(self._history),
            "kl_swucb_arms_in_window": len(self._counts),
        }
