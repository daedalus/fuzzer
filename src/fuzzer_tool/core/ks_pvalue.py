"""Shared Kolmogorov-Smirnov p-value helpers.

Unifies the asymptotic series previously private to ``edge_tracker`` with
the Marsaglia exact H-matrix path previously private to ``randomness``
(math-port plan P4). Callers keep their original behaviour: the two-sample
asymptotic used by the edge tracker, and the one-sample exact/asymptotic
switch used by ``ks_uniform``.
"""

from __future__ import annotations

import math

import numpy as np

# Series length for the asymptotic Kolmogorov tail (20 terms suffice in practice).
_ASYMPTOTIC_TERMS = 20


def kolmogorov_pvalue_two_sample(d: float, n: int, m: int) -> float:
    """P-value for two-sample KS statistic *d* with sample sizes *n*, *m*.

    Asymptotic series:
    ``P(D >= d) = 2 * sum_{k=1}^{∞} (-1)^{k-1} exp(-2 k^2 λ^2)``
    where ``λ = d * sqrt(n*m/(n+m))``.
    """
    if d <= 0:
        return 1.0
    if d >= 1.0:
        return 0.0
    nm = n * m / (n + m)
    lam = d * math.sqrt(nm)
    lam2 = lam * lam
    p = 0.0
    for k in range(1, _ASYMPTOTIC_TERMS + 1):
        term = ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam2)
        p += term
    return max(0.0, min(1.0, 2.0 * p))


def ks_exact_cdf(n: int, d: float) -> float:
    """Marsaglia-Tsang-Wang exact ``P(D_n < d)`` via the H-matrix power method.

    Expensive O(n · m³); preferred only for small n (≤140) in one-sample
    uniformity checks. See ``randomness.ks_uniform``.
    """
    if d <= 0.0:
        return 0.0
    if d >= 1.0:
        return 1.0
    k = int(n * d) + 1
    m = 2 * k - 1
    h = k - n * d
    hmat = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i - j + 1 >= 0:
                hmat[i][j] = 1.0
    for i in range(m):
        hmat[i][0] -= h ** (i + 1)
        hmat[m - 1][i] -= h ** (m - i)
    hmat[m - 1][0] += (2 * h - 1) ** m if (2 * h - 1) > 0 else 0.0
    for i in range(m):
        for j in range(m):
            if i - j + 1 > 0:
                for g in range(1, i - j + 2):
                    hmat[i][j] /= g
    q = np.linalg.matrix_power(hmat, n)
    s = q[k - 1][k - 1]
    for i in range(1, n + 1):
        s = s * i / n
        if s < 1e-140:
            s *= 1e140
    return float(s)


def kolmogorov_pvalue_one_sample(d: float, n: int) -> float:
    """Asymptotic one-sample KS p-value with Stephens correction."""
    if d <= 0 or n <= 0:
        return 1.0
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    s = sum(
        (-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
        for j in range(1, 101)
    )
    return max(0.0, min(1.0, 2.0 * s))
