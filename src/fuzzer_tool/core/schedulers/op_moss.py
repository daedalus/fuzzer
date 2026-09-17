"""MOSSScheduler: minimax-optimal UCB that stops exploring at the fair share.

Audibert & Bubeck, *Minimax Policies for Adversarial and Stochastic Bandits*
(COLT 2009); anytime form from Degenne & Perchet, *Anytime optimal
algorithms in stochastic multi-armed bandits* (ICML 2016):

    index(i) = mu_i + c * sqrt( (1+alpha)/2 * log+( t / (K * n_i) ) / n_i )

with t the pulls over the K candidates and log+ = max(log, 0). UCB1's width
is log(t)/n_i for every arm; MOSS's is zero once an arm holds its fair share
t/K. That removes UCB1's sqrt(log t) factor from the worst-case regret
(O(sqrt(Kt)) against O(sqrt(Kt log t))) and, in practice, stops the policy
re-opening every one of ~200 operators on each doubling of t.

Why it is here
--------------
The operator set is large and yields are low, which is the regime MOSS was
built for. Measured on a 150-arm environment (rates 0.002-0.08, 60k pulls,
3 seeds, mean successes):

    MOSS 3992 | Consolidated 3928 | MonteCarlo 3846 | DUCB 756 | SWUCB 659

(uniform selection expects ~560). The discounted/windowed UCBs keep 10k/4k
pulls of memory, ~30-70 per arm at this K -- too few for their width to
separate 0.08 from 0.01. On bandit_env's 12-arm stationary environment it is
in the leading group (1715 vs Consolidated 1696, KL-SW-UCB 1710).

Where it is weak
----------------
Undiscounted MOSS never forgets: on DecayingBest it recovers only 0.34 of
the tail onto the new best arm (Consolidated 0.98). ``gamma`` < 1 discounts
every statistic as D-UCB does; 0.99995 lifts recovery to 0.72 and costs 22%
on the 150-arm environment. The default keeps the large-K strength.

Rewards
-------
``record`` clamps ``weight`` to [0, 1] (the paper's bound assumes it; NaN
reads as 0) and counts one pull. The update is off-policy-safe -- a sample
mean does not care who pulled -- so it belongs in the shared record()
fan-out.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

#: Fold the global discount back into the arrays below this, before underflow.
_MIN_DISCOUNT = 1e-12


def _clamp_reward(success: bool, weight: float) -> float:
    """[0, 1]; NaN and failures give 0."""
    if not success:
        return 0.0
    return min(1.0, max(0.0, float(weight)))


class MOSSScheduler:
    """MOSS-anytime (Degenne & Perchet) over mutation operators.

    Args:
        gamma: Discount per record, in (0, 1]. 1.0 is MOSS; below 1 every
            count and sum decays by gamma per pull (see module docstring for
            the measured trade-off).
        alpha: The anytime paper's alpha >= 0; the width carries (1+alpha)/2.
            1.0 makes that factor 1, the original MOSS constant.
        exploration: Multiplier on the whole width. Measured on the 12-arm
            stationary environment over 100 seeds: 0.35 starved the best arm
            on one seed, 0.5 kept a minimum tail share of 0.948. On 150 arms
            smaller is better (1.0: 3123, 0.5: 3992, 0.35: 4363 successes),
            so 0.5 is the smallest value that did not starve.
        rng: Shared ``RandPool`` (Hard Rule 16). Drawn only to open an
            unpulled arm or break an exact tie between equally pulled arms.
    """

    #: No informative Beta prior to take: the index is a sample mean.
    supports_priors = False

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        exploration: float = 0.5,
        rng: RandPool | None = None,
    ) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        if exploration <= 0.0:
            raise ValueError(f"exploration must be positive, got {exploration!r}")
        self.gamma = gamma
        self.alpha = alpha
        self.exploration = exploration
        self._width_sq = exploration * exploration * (1.0 + alpha) / 2.0
        self._rng = rng if rng is not None else RandPool()

        # Counts and sums stored relative to one global discount factor, so a
        # record is O(1) (the D-UCB trick); true value = stored * _discount.
        self._index: dict[str, int] = {}
        self._names: list[str] = []
        self._n_rel = np.zeros(0)
        self._x_rel = np.zeros(0)
        self._discount = 1.0

        # Candidate lists repeat per seed; cache their index arrays by content.
        self._idx_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._total_pulls = 0

    # -- arm registry -----------------------------------------------------

    def _arm_id(self, name: str) -> int:
        aid = self._index.get(name)
        if aid is not None:
            return aid

        aid = len(self._names)
        self._index[name] = aid
        self._names.append(name)
        self._n_rel = np.append(self._n_rel, 0.0)
        self._x_rel = np.append(self._x_rel, 0.0)
        return aid

    def init_arm(self, name: str) -> None:
        """Register *name* with zero evidence."""
        self._arm_id(name)

    def _indices(self, ops: list[str]) -> np.ndarray:
        key = tuple(ops)
        idx = self._idx_cache.get(key)
        if idx is None:
            idx = np.fromiter((self._arm_id(op) for op in ops), dtype=np.int64, count=len(ops))
            self._idx_cache[key] = idx
        return idx

    # -- selection ----------------------------------------------------------

    def _scores(self, ops: list[str]) -> np.ndarray:
        """MOSS index per candidate; every candidate must have evidence."""
        idx = self._indices(ops)
        n = self._n_rel[idx] * self._discount
        x = self._x_rel[idx] * self._discount

        # log+(t / (K n)): zero at or above the fair share t/K.
        log_plus = np.log(n.sum() / (len(ops) * n))
        np.maximum(log_plus, 0.0, out=log_plus)
        return x / n + np.sqrt(self._width_sq * log_plus / n)

    def select_op(self, ops: list[str]) -> str:
        """Highest index; unpulled arms first, ties to the less-pulled arm."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        # Index first: registering an unseen op reallocates _n_rel, and
        # `self._n_rel[self._indices(ops)]` would read the old array.
        idx = self._indices(ops)
        n = self._n_rel[idx]

        # Unpulled arm: infinite index. Discounted counts never return to 0.
        unpulled = np.flatnonzero(n <= 0.0)
        if unpulled.size:
            return self._rng.choice([ops[i] for i in unpulled])

        scores = self._scores(ops)
        tied = np.flatnonzero(scores == scores.max())
        if tied.size == 1:
            return ops[int(tied[0])]

        # Exact ties are common: arms at fair share with equal means score
        # the same. List order would favour whatever the caller listed first.
        tied = tied[n[tied] == n[tied].min()]
        if tied.size == 1:
            return ops[int(tied[0])]
        return self._rng.choice([ops[i] for i in tied])

    # -- update -------------------------------------------------------------

    def _renormalise(self) -> None:
        self._n_rel *= self._discount
        self._x_rel *= self._discount
        self._discount = 1.0

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """One pull of *name* with reward ``weight`` clamped to [0, 1]."""
        reward = _clamp_reward(success, weight)
        aid = self._arm_id(name)
        self._total_pulls += 1

        if self.gamma < 1.0:
            self._discount *= self.gamma
            if self._discount < _MIN_DISCOUNT:
                self._renormalise()

        inv = 1.0 / self._discount
        self._n_rel[aid] += inv
        if reward:
            self._x_rel[aid] += reward * inv

    # -- diagnostics ----------------------------------------------------------

    def bandit_stats(self) -> dict:
        """Pull count, discounted evidence, and the top arms by mean."""
        n = self._n_rel * self._discount
        pulled = np.flatnonzero(n > 0.0)
        means = self._x_rel[pulled] * self._discount / n[pulled]
        order = pulled[np.argsort(-means, kind="stable")][:5]
        top = [(self._names[i], round(float(self._x_rel[i] / self._n_rel[i]), 4)) for i in order]
        return {
            "moss_pulls": self._total_pulls,
            "moss_effective_n": round(float(n.sum()), 6),
            "moss_arms": len(self._names),
            "moss_top": top,
        }
