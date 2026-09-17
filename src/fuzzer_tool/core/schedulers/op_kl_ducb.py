"""KL_DUCBScheduler: KL-UCB over discounted statistics.

Same discounted structure as ``DUCBScheduler`` (Garivier & Moulines),
but the confidence width is the empirical-Bernoulli KL upper bound
instead of the Gaussian form.

Defaults history
-----------------
Earlier versions defaulted to ``xi=0.6, exploration=0.25`` -- copied
verbatim from ``DUCBScheduler``, where that pair is an empirically
measured correction for the *Gaussian* bound's inflated ``2B`` leading
constant (see ``op_ducb.py``). That pair has no justification in the KL
paper: Garivier & Cappé (COLT 2011, arXiv:1102.2490) Remark 5 states
"in practice ... we rather suggest to choose c = 0", i.e. their own
recommended budget is the *unshrunk* ``log(t)``, since unlike the
Gaussian bound the KL bound needs no fudge factor in their (undiscounted)
setting.

Measuring both against ``tools/measure_klucb_signal.py`` and
``tests/support/bandit_env`` (stationary tail share + post-decay recovery,
5 seeds each) falsified the natural next guess: using the paper's own
unshrunk recommendation (``xi=1.0, exploration=1.0``) does not fix this
scheduler -- it makes it dramatically worse (stationary tail share
0.66 vs. DUCB's 0.98). A shrinkage sweep found a narrow band around
``xi=0.10`` that roughly matches DUCB on both stationary tail share
(0.98) and post-decay recovery (0.85); values above ~0.15 or below
~0.075 degrade sharply and become unstable across seeds. That the
optimum is neither the old borrowed-from-Gaussian value (0.6*0.25=0.15,
itself close by coincidence) nor the paper's own recommended value (1.0)
is further evidence for the gap this module's docstring above describes:
composing Garivier-Moulines' discounted self-normalized bound with
Garivier-Cappé's KL bound is not something either paper proves, and the
right tuning constant for that untested composition has to be found
empirically rather than read off of either source. See
``docs/handover/handover_kl_ducb_paper_fidelity_2026-09-14.md`` for the
full trace, including the (wrong) initial hypothesis that removing the
shrinkage entirely would fix it.
"""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._kl_ucb import kl_upper_bound
from fuzzer_tool.core.schedulers.ucb_common import DiscountedUCBBase


class KL_DUCBScheduler(DiscountedUCBBase):
    """Discounted UCB with KL-UCB confidence bound.

    Args:
        gamma: Discount factor per record, in (0, 1].
        xi: Exploration constant inside the KL budget
            ``xi * log(n) / N``. Default 0.10, found by sweeping tail
            share against ``DUCBScheduler`` on both a stationary and a
            decaying-best environment (5 seeds each) -- see the module
            docstring. Neither the old borrowed-from-Gaussian value
            (0.6) nor the paper's own unshrunk recommendation (1.0)
            performed well here.
        b: Reward range (unused by the KL width, kept for API parity).
        exploration: Multiplier on the whole confidence width, applied
            to the KL bound as-is. Left at 1.0 by default; ``xi`` alone
            carries the empirically-tuned shrinkage.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    supports_priors = False

    def __init__(
        self,
        gamma: float = 0.9999,
        xi: float = 0.10,
        b: float = 1.0,
        exploration: float = 1.0,
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
