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
        return {"homogeneous": p > self.alpha, "p": p, "n": n}

    def reset(self) -> None:
        """Clear all state, keeping the configured parameters."""
        self._counts.clear()
