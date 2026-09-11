"""ConsolidatedScheduler: one operator scheduler built from what measured best.

Why one scheduler and not a portfolio
-------------------------------------
``--elo`` arbitrates between the operator schedulers by playing the
selected one against every other with score = its own round outcome. At
fuzzing success rates that cannot concentrate: the selected strategy loses
~97% of its games whatever its quality, so Thompson over the ratings picks
whoever played least recently. Measured with BayesianEloTracker as
configured by ``_activate_elo``: a strategy three times as productive as
its rival (3% vs 1%) gets 51% of the picks, ten times (10% vs 1%) gets 54%.
Seventeen schedulers under Elo each got 5.6-6.4% of the picks, and the
portfolio found 13% fewer discoveries than its best member alone.

What the portfolio does contribute is data: every scheduler learning from
every round (the shared record() fan-out) beat each learning only from its
own. So the consolidation keeps the shared evidence and replaces the
arbitration with a single learner that has the ingredients the tournament
rewarded.

What it is made of
------------------
A tournament of the 17 operator schedulers on three synthetic environments
(``tests/support/bandit_env.py``'s stationary and decaying-best, and a
150-arm environment with rare heavy-tailed yields, fatigue on success and
periodic unlocks) put three ideas on top:

- **Thompson sampling on Beta evidence** (``MonteCarloScheduler``): best on
  the 150-arm environment.
- **A capped pseudocount** (``HierarchicalBanditScheduler``): rescaling
  alpha and beta together once their sum passes a ceiling keeps the mean
  and stops the variance shrinking -- sliding-window Thompson. Best on the
  decaying-best environment, where uncapped posteriors keep exploiting a
  dead arm.
- **Sharing strength across an operator category** (also Hierarchical's):
  with ~200 operators most arms are unsampled for most of a campaign, and
  the category's rate is the best available guess for them.

Hierarchical shares strength by *choosing* a category first, which starves
a good operator in a category whose average is poor. This scheduler shares
it through the *prior* instead: each operator's Beta prior is centred on
its category's (capped) rate with strength ``prior_strength``, and the
operator's own evidence overrides it as it accumulates. Selection is one
flat Thompson draw over every candidate.

Measured with this class (mean discoveries, 3 paired seeds; 60k rounds on
the 150-arm environment, 20k on decaying-best, 6k on stationary):

    ==================  =========  ============  ==========
    scheduler           150-arm    decaying      stationary
    ==================  =========  ============  ==========
    Consolidated        **2085**   **4639**      1682
    Bayes-UCB           2072       3814          1686
    MonteCarlo (TS)     2004       3189          1704
    Hierarchical        1985       4634          1691
    KL-SW-UCB           1939       4294          **1710**
    ==================  =========  ============  ==========

It is the only one in the leading group on all three: best on the 150-arm
environment, tied with Hierarchical on decaying-best (where Bayes-UCB and
MonteCarlo, which never forget, lose 18-31%), and within 1.6% of the best
on the stationary one, whose construction -- the best arm alone among
base-rate arms in its category -- is adversarial to category sharing on
purpose. Run-to-run spread from the sampler alone is about +/-3% (four RNG
seeds on one environment seed: 1706-1809), so the 150-arm lead over
Bayes-UCB is within noise and the lead over MonteCarlo and Hierarchical is
not. The prototype without the category prior scored ~5% below it there,
so the prior is doing the work, not the cap alone.

These are synthetic environments. The claim that matters -- more edges per
hour on a real target -- still needs the paired A/B (bench_paired) against
the default scheduler before this becomes a default.

Rewards
-------
``record(name, success, weight)`` adds ``weight`` (clamped to [0, 1]) to
the arm's successes and ``1 - weight`` to its failures: a fractional
Bernoulli observation, so a pull always carries exactly one unit of
evidence. ``Fuzzer.fuzz_one`` already bounds the weight to [0, 1].

The update is off-policy-safe -- a Beta posterior does not care who pulled
the arm -- so it belongs in the shared record() fan-out, unlike Exp3,
CMA-ES and MOpt.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.operator_categories import category_of
from fuzzer_tool.core.rand_pool import RandPool

#: Floor on Beta parameters handed to the sampler; numpy's beta rejects 0.
_MIN_PARAM = 1e-3


class ConsolidatedScheduler:
    """Flat Thompson sampling with a category-shrunk prior and capped evidence.

    Args:
        prior_strength: Pseudocount weight of the category prior. The prior
            for an operator in category c is
            Beta(m * mu_c, m * (1 - mu_c)), mu_c the category's smoothed
            success rate. 4 measured best of {2, 4, 8}.
        max_pseudocount: Ceiling on an operator's own alpha + beta. Beyond
            it both are rescaled together (mean kept, variance floored),
            which is how the scheduler forgets. 200 measured best of
            {100, 200, 400, 1000}; it is Hierarchical's value.
        category_max_pseudocount: The same ceiling for category evidence.
            Larger than the per-arm cap because a category pools many arms.
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    #: init_arm() accepts (prior_alpha, prior_beta) overrides from
    #: target_profiler.format_operator_priors(); see init_arm.
    supports_priors = True

    def __init__(
        self,
        prior_strength: float = 4.0,
        max_pseudocount: float = 200.0,
        category_max_pseudocount: float = 1000.0,
        rng: RandPool | None = None,
    ) -> None:
        if prior_strength < 0:
            raise ValueError(f"prior_strength must be >= 0, got {prior_strength!r}")
        if max_pseudocount <= 0 or category_max_pseudocount <= 0:
            raise ValueError("pseudocount ceilings must be > 0")
        self.prior_strength = float(prior_strength)
        self.max_pseudocount = float(max_pseudocount)
        self.category_max_pseudocount = float(category_max_pseudocount)
        self._rng = rng if rng is not None else RandPool()

        # Arm state lives in parallel numpy arrays indexed by arm id, so a
        # selection is one vectorized Beta draw over the candidates rather
        # than a Python loop over ~200 arms.
        self._index: dict[str, int] = {}
        self._names: list[str] = []
        self._alpha = np.zeros(0)
        self._beta = np.zeros(0)
        self._arm_cat = np.zeros(0, dtype=np.int64)

        self._cat_index: dict[str, int] = {}
        self._cat_alpha = np.zeros(0)
        self._cat_beta = np.zeros(0)

        # Candidate lists repeat (build_ops returns the same set for a seed),
        # so their index arrays are cached by content.
        self._idx_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._total_pulls = 0
        self._total_successes = 0.0

    # -- arm registry -----------------------------------------------------

    def _category_id(self, cat: str) -> int:
        cid = self._cat_index.get(cat)
        if cid is None:
            cid = len(self._cat_index)
            self._cat_index[cat] = cid
            self._cat_alpha = np.append(self._cat_alpha, 0.0)
            self._cat_beta = np.append(self._cat_beta, 0.0)
        return cid

    def _arm_id(self, name: str) -> int:
        aid = self._index.get(name)
        if aid is None:
            aid = len(self._names)
            self._index[name] = aid
            self._names.append(name)
            self._alpha = np.append(self._alpha, 0.0)
            self._beta = np.append(self._beta, 0.0)
            self._arm_cat = np.append(self._arm_cat, self._category_id(category_of(name)))
            self._idx_cache.clear()
        return aid

    def init_arm(
        self, name: str, prior_alpha: float | None = None, prior_beta: float | None = None
    ) -> None:
        """Register *name*; optionally bias it with a format prior.

        ``format_operator_priors`` expresses its hints as a Beta prior meant
        to replace the uniform Beta(1, 1). Here the prior comes from the
        category, so the hint is applied as what it adds over uniform:
        ``(prior_alpha - 1, prior_beta - 1)`` of initial evidence (floored at
        zero). The boosted (2, 1) becomes one early success -- a nudge the
        operator's own record overrides within a few dozen pulls, rather than
        a second prior competing with the category's. Re-registering an arm
        never resets it.
        """
        fresh = name not in self._index
        aid = self._arm_id(name)
        if fresh and prior_alpha is not None and prior_beta is not None:
            self._alpha[aid] += max(0.0, float(prior_alpha) - 1.0)
            self._beta[aid] += max(0.0, float(prior_beta) - 1.0)

    def _indices(self, ops: list[str]) -> np.ndarray:
        key = tuple(ops)
        idx = self._idx_cache.get(key)
        if idx is None:
            idx = np.fromiter((self._arm_id(op) for op in ops), dtype=np.int64, count=len(ops))
            self._idx_cache[key] = idx
        return idx

    # -- selection ----------------------------------------------------------

    def _prior(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cats = self._arm_cat[idx]
        ca = self._cat_alpha[cats]
        cb = self._cat_beta[cats]
        mu = (ca + 1.0) / (ca + cb + 2.0)
        return self.prior_strength * mu, self.prior_strength * (1.0 - mu)

    def select_op(self, ops: list[str]) -> str:
        """One Thompson draw per candidate; the largest wins."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        idx = self._indices(ops)
        pa, pb = self._prior(idx)
        draws = self._rng.betavariate_array(
            np.maximum(pa + self._alpha[idx], _MIN_PARAM),
            np.maximum(pb + self._beta[idx], _MIN_PARAM),
        )
        return ops[int(np.argmax(draws))]

    def posterior_mean(self, name: str) -> float:
        """The arm's current expected reward, prior included."""
        idx = np.array([self._arm_id(name)])
        pa, pb = self._prior(idx)
        a = float(pa[0] + self._alpha[idx[0]])
        b = float(pb[0] + self._beta[idx[0]])
        return a / (a + b) if a + b > 0 else 0.5

    # -- update -------------------------------------------------------------

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """One fractional-Bernoulli observation for *name* and its category."""
        r = min(1.0, max(0.0, float(weight))) if success else 0.0
        aid = self._arm_id(name)
        self._total_pulls += 1
        self._total_successes += r

        a = self._alpha[aid] + r
        b = self._beta[aid] + (1.0 - r)
        n = a + b
        if n > self.max_pseudocount:
            scale = self.max_pseudocount / n
            a *= scale
            b *= scale
        self._alpha[aid] = a
        self._beta[aid] = b

        cid = self._arm_cat[aid]
        ca = self._cat_alpha[cid] + r
        cb = self._cat_beta[cid] + (1.0 - r)
        n = ca + cb
        if n > self.category_max_pseudocount:
            scale = self.category_max_pseudocount / n
            ca *= scale
            cb *= scale
        self._cat_alpha[cid] = ca
        self._cat_beta[cid] = cb

    # -- diagnostics ----------------------------------------------------------

    def bandit_stats(self) -> dict:
        """Pull counts and the top arms by posterior mean."""
        top: list[tuple[str, float]] = []
        if self._names:
            idx = np.arange(len(self._names))
            pa, pb = self._prior(idx)
            a = pa + self._alpha
            b = pb + self._beta
            means = a / np.maximum(a + b, _MIN_PARAM)
            order = np.argsort(-means)[:5]
            top = [(self._names[i], round(float(means[i]), 4)) for i in order]
        return {
            "consolidated_pulls": self._total_pulls,
            "consolidated_successes": round(self._total_successes, 3),
            "consolidated_arms": len(self._names),
            "consolidated_top": top,
        }
