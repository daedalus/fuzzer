"""Reference beta-binomial dispersion fit, derived independently of core/blup.py.

Solves E[S](rho) = S for rho term by term from the variance of a unit rate,
``Var(p_i) = v * (rho + (1 - rho) / n_i)``, and of the weighted mean, rather
than from the closed form the production code uses. E[S] is linear in rho,
so two evaluations solve it.
"""

from __future__ import annotations


def _expected_s(n: list[float], v: float, rho: float) -> float:
    total = sum(n)
    var = [v * (rho + (1.0 - rho) / ni) for ni in n]
    within = sum(ni * vi for ni, vi in zip(n, var, strict=True))
    mean_var = sum(ni * ni * vi for ni, vi in zip(n, var, strict=True)) / (total * total)
    return within - total * mean_var


def ref_strength(rho: float, total: float, m_max: float, m_min: float) -> float:
    """Strength m = 1 / rho - 1 under the policy bounds: rho clipped
    first, m capped at the evidence total and m_max."""
    ceiling = max(min(m_max, total), m_min)
    rho = min(max(rho, 1.0 / (1.0 + ceiling)), 1.0 / (1.0 + m_min))
    return min(max(1.0 / rho - 1.0, m_min), ceiling)


def ref_fit(succ: list[float], n: list[float]) -> tuple[float, float] | None:
    """(mu, rho) of the units with n > 0, or None when undefined."""
    pairs = [(s, k) for s, k in zip(succ, n, strict=True) if k > 0]
    if len(pairs) < 2:
        return None

    s = [p[0] for p in pairs]
    k = [p[1] for p in pairs]
    mu = sum(s) / sum(k)
    v = mu * (1.0 - mu)
    if v <= 0:
        return None

    observed = sum(ki * (si / ki - mu) ** 2 for si, ki in zip(s, k, strict=True))
    e0 = _expected_s(k, v, 0.0)
    e1 = _expected_s(k, v, 1.0)
    if e1 - e0 <= 1e-12:
        return None
    return mu, (observed - e0) / (e1 - e0)
