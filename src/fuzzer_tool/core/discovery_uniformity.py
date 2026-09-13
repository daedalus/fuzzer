"""Nonparametric Poisson-dispersion check on the coverage-discovery tick series.

Where this sits
----------------
``garch.py`` fits an ARCH(1,1) conditional-variance model.
``allan_variance.py`` classifies the noise type at several averaging times.
``critical_slowing.py`` watches for rising variance/autocorrelation before a
regime shift. All three assume a specific parametric shape for what
"normal" looks like, and score deviations from *that* shape.

This module makes a much weaker assumption. Under a homogeneous Poisson
discovery process, the per-tick edge-count deltas are i.i.d. Poisson(lambda)
draws for whatever lambda the campaign is currently running at -- and for a
Poisson variate, variance equals mean by construction. Fisher's index of
dispersion, ``D = sum((x_i - xbar)^2) / xbar``, is chi-square(n-1)
distributed under exactly that null, with no further shape assumption
needed. ``D`` far in the upper tail means clustered/bursty discovery
(over-dispersion); far in the lower tail means suspiciously metronomic
discovery (under-dispersion, e.g. a scheduler emitting a near-constant
count every tick). The two-sided fold below is the same construction
``birthday_spacings`` already uses for its own small-count statistic.

An earlier version of this module tried to feed a mid-p-corrected
per-count p-value into ``core.randomness.uniformity_report`` (the KS/
Kuiper/Fisher meta-architecture that module's own docstring advertises for
exactly this kind of validation job). That measured badly calibrated:
``birthday_spacings``'s own module comment already documents why -- a
small-Poisson-count mid-p value is discrete, "and cannot be uniform even
with the mid-p correction... usable as a single rejection signal but must
never be fed into ks_uniform/kuiper_uniform." A per-tick edge count is the
same shape of statistic, so it inherits the same restriction; measuring the
false-positive rate on simulated stationary Poisson series confirmed it
directly (~20-85% rejection at a nominal 1% alpha, depending on how the
rate was estimated) before this dispersion-test version replaced it, which
calibrates to within noise of the nominal rate on the same simulation.

Cost
----
:meth:`DiscoveryUniformityDetector.update` is O(1). :meth:`verdict` is
O(window) for the dispersion sum, called at most once per stats tick
alongside the other regime-detection readouts it sits next to.

Negative-binomial severity readout
-----------------------------------
The dispersion test above is deliberately a pure yes/no rejection of the
homogeneous-Poisson null -- it says "clustered" but not "how clustered."
When the null is rejected on the *over*-dispersed side (D high, variance
exceeds the mean), the counts are a candidate fit for a negative binomial:
NB is exactly the distribution obtained by letting a Poisson's rate itself
vary as a Gamma(r, theta) draw per tick (the gamma-Poisson mixture), so a
bursty discovery process -- ticks alternating between "nothing new" and "a
vein of coverage just opened up" -- is the textbook NB regime, not a
different failure mode that happens to also fail the dispersion test.
``fit_negative_binomial`` gives the method-of-moments estimate of NB's
aggregation parameter r from the same count window already collected here.
r is inversely related to burstiness: r -> infinity recovers the Poisson
limit (see the module's homogeneous case), while small r means a few ticks
account for most of the discovered coverage. This is a strictly cheaper
estimator than MLE (no digamma root-finding) and is reported only as a
severity readout alongside the existing p-value -- it is not itself a
second hypothesis test, and callers should gate on ``verdict()``'s
existing p-value/alpha before trusting it, since the moment estimator is
noisy at small sample counts and is undefined (returns ``None``) whenever
the sample variance does not exceed the sample mean.
"""

from __future__ import annotations

import collections

from fuzzer_tool.core.randomness import chisq_sf

_DEFAULT_WINDOW = 256
_DEFAULT_MIN_OBS = 32
_DEFAULT_ALPHA = 0.01


def dispersion_pvalue(counts) -> float:
    """Two-sided p-value of Fisher's index of dispersion for *counts*.

    ``D = sum((x_i - xbar)^2) / xbar ~ chi-square(n-1)`` under the
    homogeneous-Poisson null. Folded two-sided exactly as
    ``birthday_spacings`` folds its own tail: take twice the smaller of the
    upper- and lower-tail probabilities, clamped to ``[0, 1]``.

    Returns ``1.0`` (inconclusive) when there are fewer than two counts or
    the sample mean is zero -- the statistic is undefined either way.
    """
    n = len(counts)
    if n < 2:
        return 1.0
    xbar = sum(counts) / n
    if xbar <= 0:
        return 1.0
    d = sum((x - xbar) ** 2 for x in counts) / xbar
    dof = n - 1
    p_over = chisq_sf(d, dof)
    return max(0.0, min(1.0, 2.0 * min(p_over, 1.0 - p_over)))


def fit_negative_binomial(counts) -> dict | None:
    """Method-of-moments fit of a negative binomial to *counts*.

    Uses ``r = mean**2 / (var - mean)``, ``p = mean / var`` -- the
    standard NB method-of-moments pair, valid exactly when the sample is
    overdispersed relative to Poisson (``var > mean``). Returns ``None``
    when there are fewer than two counts, the mean is zero, or
    ``var <= mean`` (underdispersed or exactly Poisson: there is no
    over-dispersion for NB to explain, and the ``r`` formula would divide
    by zero or go negative).

    Returns a dict with ``r`` (aggregation parameter; smaller means
    burstier) and ``p`` (NB success-probability parameter in the "number
    of failures before r successes" parameterization) on success.
    """
    n = len(counts)
    if n < 2:
        return None
    mean = sum(counts) / n
    if mean <= 0:
        return None
    var = sum((x - mean) ** 2 for x in counts) / n
    if var <= mean:
        return None
    r = mean * mean / (var - mean)
    p = mean / var
    return {"r": r, "p": p}


class DiscoveryUniformityDetector:
    """Rolling Poisson-dispersion test of per-tick discovery counts.

    Args:
        window: Ring-buffer size of the counts the dispersion statistic is
            computed over.
        min_obs: Counts required before ``verdict()`` offers a real
            reading rather than the inconclusive default.
        alpha: Significance level below which the two-sided p-value is
            read as a rejection of the homogeneous-Poisson null.
    """

    def __init__(
        self,
        window: int = _DEFAULT_WINDOW,
        min_obs: int = _DEFAULT_MIN_OBS,
        alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        self.window = max(min_obs, window)
        self.min_obs = max(8, min_obs)
        self.alpha = alpha
        self._counts: collections.deque[int] = collections.deque(maxlen=self.window)

    def update(self, delta: int) -> None:
        """Record one tick's edge-discovery count.

        Negative deltas (a counter reset, a wrapped map) are clamped to
        zero rather than raising -- a zero-discovery tick is itself a
        legitimate, informative Poisson(lambda) observation.
        """
        self._counts.append(max(0, int(delta)))

    def verdict(self) -> dict:
        """Dispersion test on the collected count window.

        Returns ``{"homogeneous": True, "p": 1.0, "n": <count>}`` before
        *min_obs* observations, since there is not yet enough population
        for the statistic to say anything.
        """
        n = len(self._counts)
        if n < self.min_obs:
            return {"homogeneous": True, "p": 1.0, "n": n}
        p = dispersion_pvalue(self._counts)
        verdict = {"homogeneous": p > self.alpha, "p": p, "n": n}
        if not verdict["homogeneous"]:
            nb = fit_negative_binomial(self._counts)
            if nb is not None:
                verdict["nb_r"] = nb["r"]
                verdict["nb_p"] = nb["p"]
        return verdict

    def reset(self) -> None:
        """Clear all state, keeping the configured parameters."""
        self._counts.clear()
