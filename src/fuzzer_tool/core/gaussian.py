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
