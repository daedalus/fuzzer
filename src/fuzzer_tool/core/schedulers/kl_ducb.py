"""KL_DUCBScheduler: KL-UCB over discounted statistics.

Same discounted structure as ``DUCBScheduler`` (Garivier & Moulines),
but the confidence width is the empirical-Bernoulli KL upper bound
instead of the Gaussian form. For Bernoulli rewards the KL bound is
tighter, so it explores less and exploits the best arm more aggressively.
"""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._kl_ucb import kl_upper_bound
from fuzzer_tool.core.schedulers.ucb_common import DiscountedUCBBase


class KL_DUCBScheduler(DiscountedUCBBase):
    """Discounted UCB with KL-UCB confidence bound.

    Args:
        gamma: Discount factor per record, in (0, 1].
        xi: Exploration constant inside the KL budget
            ``xi * log(n) / N``.
        b: Reward range (unused by the KL width, kept for API parity).
        exploration: Multiplier on the whole confidence width.
            Applied to the KL bound as-is -- the KL theorem uses a
            leading constant of 1, so ``exploration`` is the tuning
            knob here.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    supports_priors = False

    def __init__(
        self,
        gamma: float = 0.9999,
        xi: float = 0.6,
        b: float = 1.0,
        exploration: float = 0.25,
        rng: RandPool | None = None,
    ):
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi!r}")
        if exploration <= 0.0:
            raise ValueError(f"exploration must be positive, got {exploration!r}")
        self.xi = xi
        self.b = b
        self.exploration = exploration
        super().__init__(gamma=gamma, rng=rng)

    def _width(self, mean: float, n: float, log_n: float) -> float:
        """KL-UCB width: smallest q >= mean with KL(mean||q) >= xi*log(n)/n."""
        return kl_upper_bound(mean, self.exploration * self.xi * log_n / n) - mean

    def bandit_stats(self) -> dict:
        """Return KL-D-UCB diagnostics."""
        counts = self.discounted_counts()
        return {
            "kl_ducb_pulls": self._total_pulls,
            "kl_ducb_effective_n": round(sum(counts.values()), 3),
            "kl_ducb_arms": len(self._n_rel),
        }
