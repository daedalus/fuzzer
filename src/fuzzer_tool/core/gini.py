"""Gini coefficient of inequality, for any per-item count distribution.

Why this exists
----------------
Several places in the fuzzer track a "how much of X does each item have"
distribution across a population that changes size over a campaign (seeds,
operators, edges, crash-signature clusters). Comparing skew across ticks --
or across campaigns/targets -- from raw counts alone is awkward: the
population size, the total, and the units all drift together. The Gini
coefficient collapses any such distribution to a single bounded number: 0.0
is perfectly even (every item holds the same share), and it climbs toward
1.0 as the total concentrates onto fewer and fewer items. Unlike Shannon
entropy (already used for the global edge-hit distribution, see
``EdgeTracker.shannon_entropy_global``), it does not need a population-size
correction to stay comparable across a changing item count, and unlike a
raw "top-1%-share" readout it is defined even when the population is small.

This is a read-only descriptive statistic, not a hypothesis test -- there
is no null distribution or p-value here, just a concentration measure. It
does not by itself say *why* a distribution is skewed (a legitimately hot
seed vs. a scheduler bug look identical to this function), only *how much*.

Formula
-------
For n items with non-negative values sorted ascending as v(1) <= ... <=
v(n) and total T = sum(v):

    G = 1 - 2 * sum(cumsum(v)) / (n * T) + 1/n

This is the same discrete estimator already used ad hoc in
``tools/edge_diagnostic.py``'s ``y_marginal()`` for post-hoc edge-hit-count
analysis; it is reproduced here as a shared, dependency-free implementation
so every live caller (seed energy, operator selection, edge hits, crash
clusters) computes it identically rather than each carrying its own copy.

Reference
---------
Gini, C. (1912). "Variabilita e mutabilita."
"""

from __future__ import annotations

from collections.abc import Iterable


def gini(values: Iterable[float]) -> float | None:
    """Gini coefficient of ``values``, in [0.0, 1.0].

    Returns ``None`` for an empty input -- there is no distribution to
    describe, not a measured value of zero inequality. A single item, or a
    population whose total is zero (nothing to distribute yet), both
    return ``0.0``: with nothing accumulated, or nowhere else to put it,
    there is no inequality to report. Negative values are not meaningful
    for the "share of a total" interpretation this function is built for
    and are rejected with a ``ValueError`` rather than silently producing
    a number outside [0, 1].
    """
    items = [float(v) for v in values]
    n = len(items)
    if n == 0:
        return None
    if any(v < 0 for v in items):
        raise ValueError("gini() requires non-negative values")

    total = sum(items)
    if total == 0.0:
        return 0.0

    asc = sorted(items)
    cum = 0.0
    cum_sum = 0.0
    for v in asc:
        cum += v
        cum_sum += cum

    return 1.0 - 2.0 * cum_sum / (n * total) + 1.0 / n
