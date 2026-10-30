"""Exact paired tests, no scipy (Hard Rule 51): McNemar, Fisher, Wilcoxon signed-rank.

Single home for the statistics ``tools/lib/bench_paired.py`` promotes on.
Exact rather than asymptotic because paired benchmark matrices are small:
the discordant count is routinely under 25, where chi-square is anticonservative.

Signed-rank null: W+ is the rank sum of a uniform random subset of the
ranks, so its distribution is a subset-sum count. Ranks are half-integers,
so doubled they are ints and the count is a polynomial product::

    ranks 1, 2.5, 2.5  ->  (1 + x^2)(1 + x^5)(1 + x^5)  ->  P(W+ = w) = [x^2w] / 2^n

One shifted add per rank instead of enumerating all 2^n sign vectors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from fuzzer_tool.core.rank_stats import average_ranks

# Largest n given the exact signed-rank null; above it, normal approximation.
EXACT_MAX_N = 20

# Size cut-overs, measured (min-of-7): numpy's fixed cost loses below them.
_NUMPY_RANK_MIN = 128  # deltas: pure-Python ranking below
_NUMPY_DP_MIN = 9  # nonzero deltas: pure-Python subset-sum DP below

# Fisher: tables within this relative slack of P(observed) count as as-extreme.
_FISHER_SLACK = 1e-9


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts *b* and *c*.

    Under the null the discordant pairs split Binomial(b + c, 0.5):
    p = 2 * P(X <= min(b, c)), capped at 1. ``mcnemar_exact(1, 9)`` = 22/1024.

    Raises:
        ValueError: on a negative count.
    """
    if b < 0 or c < 0:
        raise ValueError(f"negative discordant count ({b}, {c})")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]].

    Sums the hypergeometric probability of every table with the observed
    margins that is no more likely than the observed one. Treats the arms as
    independent samples, so on paired data it is a contrast to McNemar only.

    Raises:
        ValueError: on a negative cell.
    """
    if min(a, b, c, d) < 0:
        raise ValueError(f"negative cell in [[{a}, {b}], [{c}, {d}]]")
    n = a + b + c + d
    if n == 0:
        return 1.0

    row, col = a + b, a + c
    denom = math.comb(n, col)
    lo, hi = max(0, col - (c + d)), min(row, col)
    probs = [math.comb(row, x) * math.comb(c + d, col - x) / denom for x in range(lo, hi + 1)]
    cut = probs[a - lo] * (1 + _FISHER_SLACK)
    return min(1.0, sum(p for p in probs if p <= cut))


def wilcoxon_signed_rank(deltas: Sequence[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired deltas.

    Zeros carry no sign and are dropped; ties take their average rank.
    Exact null for n <= ``EXACT_MAX_N``, normal approximation with
    continuity correction above.

    Returns:
        ``(statistic, p_value)``; statistic is the smaller rank sum.
        ``[1, 2, 3]`` -> ``(0.0, 0.25)``.

    Raises:
        ValueError: on a NaN delta (it has no sign or rank).
    """
    rank = _np_signed_ranks if len(deltas) >= _NUMPY_RANK_MIN else _py_signed_ranks
    ranks, w_pos, w_neg = rank(deltas)
    n = len(ranks)
    if not n:
        return 0.0, 1.0

    stat = min(w_pos, w_neg)
    if n <= EXACT_MAX_N:
        return stat, _exact_signed_p([int(2.0 * r) for r in ranks], stat)
    return stat, _normal_signed_p(w_pos, n)


def _py_signed_ranks(deltas: Sequence[float]) -> tuple[list[float], float, float]:
    """1-based mid-ranks of the nonzero |deltas|, and the W+ / W- rank sums."""
    if any(d != d for d in deltas):
        raise ValueError("NaN delta")
    pairs = sorted((abs(d), d > 0) for d in deltas if d != 0.0)
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        # [i, j] is one run of tied magnitudes.
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    w_pos = sum((r for r, (_, up) in zip(ranks, pairs, strict=True) if up), 0.0)
    w_neg = sum((r for r, (_, up) in zip(ranks, pairs, strict=True) if not up), 0.0)
    return ranks, w_pos, w_neg


def _np_signed_ranks(deltas: Sequence[float]) -> tuple[np.ndarray, float, float]:
    """Vectorised ``_py_signed_ranks``; same half-integer ranks, so same sums."""
    d = np.asarray(deltas, dtype=np.float64)
    if np.isnan(d).any():
        raise ValueError("NaN delta")
    nz = d[d != 0.0]
    ranks = average_ranks(np.abs(nz)) + 1.0
    pos = nz > 0
    return ranks, float(ranks[pos].sum()), float(ranks[~pos].sum())


def _exact_signed_p(twice: list[int], stat: float) -> float:
    """Two-sided exact p from subset-sum counts of the doubled (integer) ranks."""
    counts = _dp_np(twice) if len(twice) >= _NUMPY_DP_MIN else _dp_py(twice)
    top = len(counts) - 1

    # W+ <= stat in the low tail, W+ >= total - stat in the high one.
    s2 = round(2.0 * stat)
    low = int(sum(counts[: s2 + 1]))
    high = int(sum(counts[top - s2 :]))
    return min(1.0, 2.0 * min(low, high) / (2 ** len(twice)))


def _dp_py(twice: list[int]) -> list[int]:
    """counts[w] = number of subsets of *twice* summing to w."""
    counts = [1]
    for r in twice:
        pad = [0] * r
        counts = [a + b for a, b in zip(counts + pad, pad + counts, strict=True)]
    return counts


def _dp_np(twice: list[int]) -> np.ndarray:
    """``_dp_py`` as shifted adds; counts <= 2^EXACT_MAX_N fit int64."""
    counts = np.zeros(sum(twice) + 1, dtype=np.int64)
    counts[0] = 1
    for r in twice:
        counts[r:] = counts[r:] + counts[:-r]
    return counts


def _normal_signed_p(w_pos: float, n: int) -> float:
    """Two-sided normal-approximation p with continuity correction."""
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (abs(w_pos - mean) - 0.5) / math.sqrt(var)
    if z < 0:
        return 1.0
    return min(1.0, math.erfc(z / math.sqrt(2.0)))
