"""SWUCBScheduler: Sliding-Window UCB for abruptly-changing reward distributions.

Garivier & Moulines, *On Upper-Confidence Bound Policies for Switching Bandit
Problems* (arXiv:0805.3415), the second of the two policies in that paper.
Where D-UCB weights the past by gamma^age, SW-UCB uses a hard window: only the
last tau pulls count at all.

    N_t(i, tau) = sum_{s=t-tau+1..t} 1{I_s = i}
    X_t(i, tau) = sum_{s=t-tau+1..t} r_s * 1{I_s = i}

    index(i) = X_t(i,tau)/N_t(i,tau) + B*sqrt(xi * log(min(t, tau)) / N_t(i,tau))

Why keep both
-------------
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

import collections
import math

from fuzzer_tool.core.rand_pool import RandPool

MIN_LOG_ARG = 1.0 + 1e-9


class SWUCBScheduler:
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
        b: Reward range; rewards from ``_cost_adjusted_weight`` are in [0, 1].
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    # No (alpha, beta) pair to seed: the index is a windowed empirical mean.
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

        # Hard Rule 16: see ducb.py. Window eviction makes arms look unpulled
        # again, so this draw happens throughout the campaign, not just at
        # startup -- an unseeded stream here would cost reproducibility for
        # the whole run.
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
        """Select the operator with the highest windowed-UCB index.

        An arm with no pulls *inside the window* is treated as unpulled and
        gets priority. That is the mechanism which lets SW-UCB re-open an arm
        it abandoned: once the last of its pulls falls out the far end of the
        window it is indistinguishable from a fresh arm.
        """
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

        # log(min(t, tau)) in the paper; n_total is that quantity restricted
        # to the current candidate set.
        log_n = math.log(max(float(n_total), MIN_LOG_ARG))
        width_scale = self.b * math.sqrt(self.xi * log_n)

        best_op = ops[0]
        best_score = -math.inf
        for op in ops:
            n = self._counts[op]
            score = self._sums.get(op, 0.0) / n + width_scale / math.sqrt(n)
            if score > best_score:
                best_score = score
                best_op = op

        return best_op

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
            # Drop the arm's entries entirely rather than leaving a zero
            # behind: with 147 operators and a filtered candidate list, stale
            # zero-count keys would otherwise accumulate into a permanent
            # record of every arm ever pulled.
            self._counts.pop(old_name, None)
            self._sums.pop(old_name, None)
            return

        self._counts[old_name] = remaining
        if not old_reward:
            return

        # Clamp: repeated float subtraction can leave a residual negative
        # epsilon, which would make the empirical mean negative and rank a
        # live arm below an untried one.
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
        """Return SW-UCB diagnostics."""
        return {
            "swucb_pulls": self._total_pulls,
            "swucb_window_fill": len(self._history),
            "swucb_arms_in_window": len(self._counts),
        }
