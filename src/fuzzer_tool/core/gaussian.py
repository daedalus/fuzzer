"""Standard-normal CDF, exact to the precision of ``math.erf``.

``Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))`` is the closed form of the
Gaussian integral, so any quantity that is "the probability mass of a
normal below x" is arithmetic here rather than a sample average or a
surrogate squasher.

Why this module exists rather than a fourth private copy: three already
live in the tree (``edge_tracker._norm_cdf``, ``op_bo_gp_ucb._Phi``, and
``randomness._erfc``'s two-tailed p-values), each written for its own
caller.  New callers land here.  The existing three are deliberately left
alone -- they are correct, covered, and re-pointing them is churn that
does not belong in the commit that adds the module.

A note on when Phi is the right tool at all.  ``erf`` integrates a
*Gaussian*, so it earns its accuracy only where the quantity really is
approximately normal.  Several of this fuzzer's natural quantities are
not: seed sizes, execution times, and edge hit counts are all
right-skewed with heavy tails, and applying Phi on their raw axis is
barely better than the logistic it replaces.  Applied to ``log(x)`` --
where the multiplicative noise that generates them is additive, and the
CLT does apply -- it is exact for practical purposes.  Callers are
expected to have made that choice deliberately; see
``services/operators.py``'s corpus-size percentile feature for a worked
example of the difference.
"""

from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float, loc: float = 0.0, scale: float = 1.0) -> float:
    """Phi((x - loc) / scale) -- the normal CDF, via ``math.erf``.

    Args:
        x: Point at which to evaluate the CDF.
        loc: Distribution mean.
        scale: Distribution standard deviation.  A non-positive scale
            degenerates the distribution to a point mass at *loc*, which
            is returned as the step function rather than raising: callers
            reach this with an as-yet-unvaried running estimate (a corpus
            of identical seed sizes, a target with constant timing) and a
            hard error there would be a crash on a legitimate state.

    Returns:
        The cumulative probability, in [0, 1].
    """
    if scale <= 0.0:
        return 1.0 if x >= loc else 0.0
    return 0.5 * (1.0 + math.erf((x - loc) / (scale * _SQRT2)))


# Acklam's rational approximation to the inverse normal CDF.  Coefficients
# reproduced from the published algorithm; the raw approximation is good to
# ~1.15e-9 relative, and the Halley step in norm_ppf below (which costs one
# math.erfc) takes it to full double precision.
_PPF_LOW = 0.02425

_PPF_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_PPF_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_PPF_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_PPF_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def norm_ppf(p: float) -> float:
    """Inverse of :func:`norm_cdf` for the standard normal -- the probit.

    ``math`` ships ``erf`` and ``erfc`` but no ``erfinv``, so unlike
    ``norm_cdf`` this cannot be one stdlib call.  Acklam's rational
    approximation supplies the shape and a single Halley refinement
    against ``math.erfc`` removes the residual, which is the same
    pure-Python-replaces-scipy pattern ``edge_tracker._levenberg_marquardt``
    already follows.

    Args:
        p: Probability in (0, 1).  The open endpoints return -inf and
            +inf respectively, which is the mathematically correct limit;
            callers working near p = 1 (a Bayes-UCB quantile order of
            1 - 1/t, say) are expected to clamp first if they need a
            finite value.

    Returns:
        z such that ``norm_cdf(z) == p``.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    if p < _PPF_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        z = (
            ((((_PPF_C[0] * q + _PPF_C[1]) * q + _PPF_C[2]) * q + _PPF_C[3]) * q + _PPF_C[4]) * q
            + _PPF_C[5]
        ) / ((((_PPF_D[0] * q + _PPF_D[1]) * q + _PPF_D[2]) * q + _PPF_D[3]) * q + 1.0)
    elif p > 1.0 - _PPF_LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = -(
            ((((_PPF_C[0] * q + _PPF_C[1]) * q + _PPF_C[2]) * q + _PPF_C[3]) * q + _PPF_C[4]) * q
            + _PPF_C[5]
        ) / ((((_PPF_D[0] * q + _PPF_D[1]) * q + _PPF_D[2]) * q + _PPF_D[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        z = (
            (((((_PPF_A[0] * r + _PPF_A[1]) * r + _PPF_A[2]) * r + _PPF_A[3]) * r + _PPF_A[4]) * r)
            + _PPF_A[5]
        ) * q / (
            ((((_PPF_B[0] * r + _PPF_B[1]) * r + _PPF_B[2]) * r + _PPF_B[3]) * r + _PPF_B[4]) * r
            + 1.0
        )

    # One Halley step on f(z) = Phi(z) - p.  erfc is used rather than erf
    # because the interesting calls are deep in the upper tail, where
    # 1 - erf(z) cancels catastrophically and erfc does not.
    e = 0.5 * math.erfc(-z / _SQRT2) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(z * z / 2.0)
    return z - u / (1.0 + z * u / 2.0)
