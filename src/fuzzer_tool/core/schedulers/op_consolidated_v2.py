"""ConsolidatedV2Scheduler: v1 with an optimistic, tempered Thompson draw.

Where it comes from
-------------------
Every operator scheduler in ``core/schedulers/`` (41 classes) was run on the
four ``tests/support/bandit_env.py`` environments, 3 seeds each (mean
successes; 6k rounds stationary, 20k decaying/rotting, 60k Fatigue150;
``--``: not finished):

    ==================  ==========  ========  =======  ==========
    scheduler           stationary  decaying  rotting  fatigue150
    ==================  ==========  ========  =======  ==========
    Consolidated v1     1696        4652      3553     1522
    Hierarchical        1693        **4658**  3573     1432
    KL-SW-UCB           1710        4294      3124     1442
    Bayes-UCB           1676        3700      3498     1457
    Corral              1616        3594      3497     **1536**
    MOSS                1715        3532      3494     1435
    FPL                 **1723**    3343      2715     1409
    TopK (greedy)       1058        3633      **3633** --
    ==================  ==========  ========  =======  ==========

v1 is the only one in the leading group on all four, so v2 keeps all of it:
the category-shrunk prior and the capped pseudocount (both from
Hierarchical), flat Thompson over Beta evidence (MonteCarlo), fractional
Bernoulli rewards. What the others beat it with is *less exploration*:
MOSS stops exploring at an arm's fair share, FPL's perturbation shrinks as
1/sqrt(t), and plain greedy wins the rotting environment, where exploiting
the current best is near-optimal. Thompson's symmetric draw spends half its
samples *below* an arm's mean -- pessimism that only ever demotes arms.

The change
----------
Score each candidate by ``mean + tau * max(0, draw - mean)``:

- **Optimistic** (May et al., *Optimistic Bayesian Sampling in Contextual-
  Bandit Problems*, JMLR 2012): a below-mean draw scores the mean, so an
  arm is never ranked below its own expectation.
- **Tempered**: an above-mean draw keeps ``tau`` of its excess -- the
  exploration bonus scaled down, as MOSS/FPL do with their widths.

``tau = 1`` is optimistic Thompson; ``tau -> 0`` is greedy on the
posterior mean. A sweep over tau in {0.5, 0.65, 0.75} (12 seeds) put all
three above v1 on every environment; 0.65 is the middle.

Measured
--------
Mean successes, 20 paired seeds (same environment stream per seed), real
classes. ``v1 control`` is v1 again under a different scheduler RNG seed:
the gap it shows is sampler noise alone.

    ==========  ====  ===========  ====  ===========  =========
    env         v1    v1 control   v2    v2 - v1      v2 > v1
    ==========  ====  ===========  ====  ===========  =========
    stationary  1729  1730         1759  +30 (+1.8%)  19/20
    decaying    4612  4617         4685  +73 (+1.6%)  20/20
    rotting     3502  3485         3558  +56 (+1.6%)  17/20
    fatigue150  1524  1520         1544  +20 (+1.3%)  16/20
    ==========  ====  ===========  ====  ===========  =========

Every v2 gain is at least 3 standard errors of the paired difference
(6-9); every control gap is within 2.

What did not help (12 seeds, v1 as the base): a global discount
(gamma 0.9999-0.99999: -2% to -9% on Fatigue150), per-arm staleness decay
(half-life 2k-20k: -4% to -17%), caps of 50/100/300, category caps of 300,
prior strengths of 2/8 -- all at or below v1. Re-opening stale arms to catch
Fatigue150's late unlocks costs more on its ~140 dead arms than it gains.

Synthetic environments again: the paired A/B (bench_paired) on a real
target is still the claim that matters.

Rewards and fan-out are v1's: off-policy-safe, one bounded observation per
record.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_consolidated_v1 import (
    _MIN_PARAM,
    ConsolidatedV1Scheduler,
    PriorMode,
)


class ConsolidatedV2Scheduler(ConsolidatedV1Scheduler):
    """v1's posterior, scored by an optimistic, tempered Thompson draw.

    Args:
        tau: Fraction of a draw's excess over the mean that counts, in
            (0, 1]. 0.65 measured; see the module docstring.
        prior_strength, max_pseudocount, category_max_pseudocount, rng,
        prior_mode: As ``ConsolidatedV1Scheduler``.
    """

    #: init_arm() is v1's: (prior_alpha, prior_beta) become initial evidence.
    supports_priors = True

    _STATS_PREFIX = "consolidated_v2"

    def __init__(
        self,
        tau: float = 0.65,
        prior_strength: float = 4.0,
        max_pseudocount: float = 200.0,
        category_max_pseudocount: float = 1000.0,
        rng: RandPool | None = None,
        prior_mode: PriorMode = PriorMode.FIXED,
    ) -> None:
        if math.isnan(tau) or not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau!r}")
        super().__init__(prior_strength, max_pseudocount, category_max_pseudocount, rng, prior_mode)
        self.tau = float(tau)

    def select_op(self, ops: list[str]) -> str:
        """Largest ``mean + tau * max(0, draw - mean)`` wins."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        idx = self._indices(ops)
        pa, pb = self._prior(idx)
        a = np.maximum(pa + self._alpha[idx], _MIN_PARAM)
        b = np.maximum(pb + self._beta[idx], _MIN_PARAM)
        mean = a / (a + b)

        # Below-mean draws score the mean; above-mean keep tau of the excess.
        # Computed in place: the Beta draw already dominates the cost.
        score = self._rng.betavariate_array(a, b) - mean
        np.maximum(score, 0.0, out=score)
        score *= self.tau
        score += mean
        return ops[int(np.argmax(score))]
