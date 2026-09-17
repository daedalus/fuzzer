"""SWUCBScheduler: Sliding-Window UCB for abruptly-changing reward distributions.

Garivier & Moulines, *On Upper-Confidence Bound Policies for Switching Bandit
Problems* (arXiv:0805.3415), the second of the two policies in that paper.
Where D-UCB weights the past by gamma^age, SW-UCB uses a hard window: only the
last tau pulls count at all.

    N_t(i, tau) = sum_{s=t-tau+1..t} 1{I_s = i}
    X_t(i, tau) = sum_{s=t-tau+1..t} r_s * 1{I_s = i}

    index(i) = X_t(i,tau)/N_t(i,tau) + B*sqrt(xi * log(min(t, tau)) / N_t(i,tau))

Why keep both
------------
They share a regret bound and have genuinely different failure modes, so which
one wins is an empirical question per target:

* D-UCB never fully forgets. An arm pulled heavily then abandoned keeps a small
  residual weight forever, which stabilises its mean but tracks a *gradual*
  decay (coverage saturation, the usual fuzzing case) with a lag proportional
  to 1/(1-gamma).
* SW-UCB forgets completely at the window edge. Sharper on abrupt change
  points, noisier on stationary stretches, because its effective sample size is
  capped at tau no matter how long the campaign runs.

Cost
----
O(1) amortised per record and O(tau) memory: a deque of (arm, reward) pairs
with incrementally maintained per-arm sums, so eviction subtracts rather than
recomputing. Per-arm entries are dropped when an arm's windowed count reaches
zero, so an operator that leaves the candidate list stops occupying space.

Bundled rewards
---------------
As with D-UCB: ``_record_outcome`` calls ``record()`` once per operator per
round, so tau counts pulls. At ``mutations_per_input = 8`` a window of 4000
pulls spans ~500 mutation rounds.
"""

import math

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.ucb_common import WindowedUCBBase

MIN_LOG_ARG = 1.0 + 1e-9


class SWUCBScheduler(WindowedUCBBase):
    """Sliding-Window UCB (Garivier & Moulines) over mutation operators.

    Args:
        window: tau, the number of most recent pulls that count. The default
            4000 is ~500 mutation rounds at the default
            ``mutations_per_input = 8`` -- long enough that a 147-operator
            candidate list still gets ~27 pulls per arm inside the window,
            which is the floor below which the empirical means are noise.
        xi: Exploration constant inside the confidence width. The paper's
            analysis uses 1/2 and its experiments 0.6; both over-explore at
            this reward scale, the same finding ``GPUCBScheduler`` records for
            its ``beta`` and ``DUCBScheduler`` for its leading 2B. Measured
            tail share, 12 arms, 20k rounds, 3 seeds, as
            stationary | live-arm-after-decay | dead-arm-after-decay:

                window  xi=0.6          xi=0.3          xi=0.15         xi=0.05
                2000    0.670|0.42|.04  0.808|0.59|.03  0.898|0.72|.02  0.963|0.90|.01
                4000    0.806|0.59|.03  0.883|0.76|.02  0.948|0.86|.01  0.983|0.95|.00
                8000    0.805|0.73|.00  0.894|0.85|.00  0.945|0.92|.00  0.984|0.99|.00

            0.15 is the default rather than 0.05 because the window has to
            re-find the best arm among 147 candidates on a real campaign, not
            12, and the lower value buys stationary tail share with
            exploration the larger arm set still needs.
        b: Reward range; ``Fuzzer.fuzz_one`` clamps rewards to [0, 1].
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
        """Gaussian width: b * sqrt(xi * log_n) / sqrt(n)."""
        width_scale = self.b * math.sqrt(self.xi * log_n)
        return width_scale / math.sqrt(n)

    def bandit_stats(self) -> dict:
        """Return SW-UCB diagnostics."""
        return {
            "swucb_pulls": self._total_pulls,
            "swucb_window_fill": len(self._history),
            "swucb_arms_in_window": len(self._counts),
        }
