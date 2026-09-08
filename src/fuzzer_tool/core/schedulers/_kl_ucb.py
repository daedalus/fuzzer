"""KL-UCB confidence bound for the D-UCB and SW-UCB indexes.

Both Garivier-Moulines policies use a Gaussian-style width
``B * sqrt(xi * log(n_t) / N_t(i))``.  That bound is valid for sub-Gaussian
rewards, but the cost-adjusted surprisal weights handed to ``record()`` are not
Gaussian: they are bounded in [0, 1] with a mass at zero (an operator that
found no new coverage gets exactly 0), so the true tail is heavier than the
Gaussian one and the Gaussian width under-covers.

KL-UCB (Cappé, Garivier, Maillard, Munos, Stoltz, 2013) replaces the Gaussian
width with the empirical-Bernoulli width: the smallest q in [p, 1] such that
the KL divergence between the empirical mean p and q is at least
log(n_t) / N_t(i).  For a Bernoulli that is

    d(p, q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))

and the bound is tight at q = 1, where d(p, 1) = inf.  We solve it by bisection
on q, which is monotone and takes ~20 iterations to float precision.

This module is a shared helper rather than a scheduler: the two indexes differ
in how they weight the past (discounted vs windowed) and in what n_t means,
but the per-arm width computation is identical, so duplicating it would mean
two places to keep the bisection correct.
"""

import math

#: KL(0.5 || q) = -log(2) - log(1 - q); solve for q when that equals the budget.
#: Closed form, used as the starting point of the bisection.
_KL_HALF_TO_ONE = -math.log(2.0)


def _kl_bernoulli(p: float, q: float) -> float:
    """KL divergence between two Bernoulli means, with the 0*log(0) convention."""
    if p <= 0.0:
        # KL(0 || q) = -log(1 - q): the second term is the only one that
        # survives, and it is 1 - q, not q.
        return -math.log(max(1.0 - q, 1e-300))
    if p >= 1.0:
        # KL(1 || q) = -log(q): the first term is the only one that survives.
        return -math.log(max(q, 1e-300))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_upper_bound(p: float, budget: float, tol: float = 1e-12) -> float:
    """Smallest q in [p, 1] with KL(p || q) >= budget (Bernoulli).

    Returns ``min(1.0, p + sqrt(2 * budget))`` when the budget is so small that
    the Gaussian approximation is within *tol* of the true bound -- that is the
    regime where the bisection would otherwise spend iterations on float noise,
    and the Gaussian form is the right answer there anyway.
    """
    if budget <= 0.0:
        return min(1.0, p)
    if p >= 1.0:
        # KL(1 || q) = -log(q), and the bound is the smallest q >= p, so the
        # only candidate is q = 1.0 -- where KL(1 || 1) = 0, below any positive
        # budget. No q in [p, 1] satisfies the constraint, so the bound is
        # saturated at the boundary. Returning exp(-budget) here would put the
        # "upper bound" *below* the empirical mean, which is what starved the
        # best arm in the DUCB measurement pass.
        return 1.0
    if p <= 0.0:
        # KL(0 || q) = -log(1 - q); the smallest q with -log(1 - q) >= budget
        # is 1 - exp(-budget), which is also the Gaussian bound at p=0.
        return min(1.0, 1.0 - math.exp(-budget))

    gaussian = p + math.sqrt(2.0 * budget)
    # The Gaussian approximation can overshoot 1.0 when p is near 1 and the
    # budget is large; KL(p || q) is only defined for q in (0, 1), so clamp
    # the trial point before evaluating it.
    if gaussian < 1.0 and _kl_bernoulli(p, gaussian) >= budget - tol:
        return min(1.0, gaussian)

    lo, hi = p, 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _kl_bernoulli(p, mid) < budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi
