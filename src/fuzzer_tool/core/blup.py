"""Empirical-Bayes BLUP for grouped Beta-Bernoulli rates.

Each unit i (an operator, a seed) has a latent rate drawn from its group::

    p_i ~ Beta(m mu, m (1 - mu))        s_i | p_i ~ Bin(n_i, p_i)

The best linear unbiased predictor of p_i is the credibility blend::

    p_hat_i = k_i * s_i / n_i + (1 - k_i) * mu,    k_i = n_i / (n_i + m)

i.e. a Beta(m mu, m (1 - mu)) prior added to the unit's own evidence. The
prior strength m is not a tuning knob: it is set by how much the units of
a group actually differ. With the intra-class correlation rho = 1 / (1 + m),
a unit's rate has ``Var(s_i / n_i) = v (rho + (1 - rho) / n_i)``,
v = mu (1 - mu), so the weighted dispersion S = sum n_i (s_i/n_i - mu)^2 has

    E[S] = v * (rho * (N - sum n_i^2 / N) + (1 - rho) * (k - 1))

and solving E[S] = S gives rho (method of moments; Kleinman 1973).

    alike units   ->  S ~ v (k - 1)  ->  rho ~ 0  ->  m at its ceiling
    unlike units  ->  S large        ->  rho ~ 1  ->  m at MIN_STRENGTH

The strength is also capped at the group's total evidence N: a prior
fitted from N observations cannot carry more than N. Without that cap,
decayed evidence (n_i ~ 5) shows no dispersion beyond binomial noise and
would pin every unit under ``m_max`` pseudocounts.

The fit is undefined with fewer than two units with evidence, with mu at 0
or 1, or when every unit has n_i = 1 (E[S] does not depend on rho).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

#: Floor on the fitted prior strength: one pseudo-observation, so a unit
#: with no evidence still has a proper prior instead of Beta(eps, eps).
MIN_STRENGTH = 1.0

#: Default ceiling for population pools (seeds, operators): units judged
#: identical still let a unit's own evidence outweigh the prior after
#: about this many observations.
POOL_MAX_STRENGTH = 1000.0

#: Below this a variance or a denominator is treated as zero.
_EPS = 1e-12


class BlupFit(NamedTuple):
    """Per-group prior mean, prior strength, and whether the fit is defined."""

    mu: np.ndarray
    m: np.ndarray
    ok: np.ndarray


def fit_groups(
    succ: np.ndarray, n: np.ndarray, groups: np.ndarray, n_groups: int, m_max: float
) -> BlupFit:
    """Fit (mu, m) for every group in one vectorised pass.

    Args:
        succ: Per-unit (possibly fractional) successes.
        n: Per-unit evidence; units with n <= 0 are ignored.
        groups: Per-unit group id in ``[0, n_groups)``.
        n_groups: Number of groups.
        m_max: Ceiling on the prior strength.

    Returns:
        BlupFit with arrays of length ``n_groups``; ``m`` is clipped to
        ``[MIN_STRENGTH, min(m_max, N)]`` and only meaningful where ``ok``.
    """
    live = n > 0
    s = np.where(live, succ, 0.0)
    w = np.where(live, n, 0.0)
    safe_n = np.where(live, n, 1.0)

    # Per-group sums: N, total successes, unit count, sum n^2, sum s^2 / n.
    big_n = np.bincount(groups, weights=w, minlength=n_groups)
    total = np.bincount(groups, weights=s, minlength=n_groups)
    units = np.bincount(groups, weights=live.astype(float), minlength=n_groups)
    sq_n = np.bincount(groups, weights=w * w, minlength=n_groups)
    sq_s = np.bincount(groups, weights=s * s / safe_n, minlength=n_groups)

    safe_big = np.maximum(big_n, _EPS)
    mu = total / safe_big
    v = mu * (1.0 - mu)

    # S = sum n (p - mu)^2 = sum s^2 / n - N mu^2.
    disp = sq_s - big_n * mu * mu
    denom = big_n - sq_n / safe_big - (units - 1.0)
    ok = (units >= 2) & (v > _EPS) & (denom > _EPS)

    rho = (disp / np.maximum(v, _EPS) - (units - 1.0)) / np.where(ok, denom, 1.0)
    rho = np.clip(rho, 1.0 / (1.0 + m_max), 1.0 / (1.0 + MIN_STRENGTH))
    m = 1.0 / rho - 1.0
    ceiling = np.maximum(np.minimum(m_max, big_n), MIN_STRENGTH)
    return BlupFit(mu, np.clip(m, MIN_STRENGTH, ceiling), ok)


def fit_pool(succ: np.ndarray, n: np.ndarray, m_max: float) -> tuple[float, float] | None:
    """(mu, m) for one group of units, or None when undefined."""
    if len(n) == 0:
        return None

    fit = fit_groups(succ, n, np.zeros(len(n), dtype=np.int64), 1, m_max)
    if not fit.ok[0]:
        return None
    return float(fit.mu[0]), float(fit.m[0])


class PoolCache:
    """A population prior refitted every ``refit_every`` ticks.

    Callers ask for the prior once per posterior draw; refitting there is
    O(units) per draw, O(units^2) per selection. The fit moves slowly, so
    it is recomputed on a cadence of the caller's observation counter.
    """

    def __init__(self, m_max: float, refit_every: int) -> None:
        self._m_max = m_max
        self._refit_every = refit_every
        self._fit_at: int | None = None
        self._prior: tuple[float, float] | None = None

    def prior(
        self, tick: int, evidence: Callable[[], tuple[np.ndarray, np.ndarray]]
    ) -> tuple[float, float] | None:
        """Cached (mu, m); *evidence* returns per-unit (successes, n)."""
        # tick < _fit_at: the counter was restored from saved state.
        stale = self._fit_at is None or not 0 <= tick - self._fit_at < self._refit_every
        if stale:
            succ, n = evidence()
            self._prior = fit_pool(succ, n, self._m_max)
            self._fit_at = tick
        return self._prior
