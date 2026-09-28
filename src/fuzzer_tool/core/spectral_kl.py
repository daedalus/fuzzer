"""Bach's closed-form spectral KL estimator, for byte distributions.

Reference: F. Bach, "Exploding variance of means of exponentials:
least-squares to the rescue" (https://francisbach.com/spectral_log_density_estimation/,
paper arXiv:2605.10668). For features phi with moments mu, Sigma under p and
q, the divergence::

    F(p || q, phi) = 1/2 * int_0^1 d^T (rho Sp + (1-rho) Sq)^-1 d  dnu(rho),
    d = mu_p - mu_q,  dnu = 2 (1 - rho) drho   (the KL case)

is a *lower bound* on KL(p || q) built from a continuum of least-squares
problems, and one generalized eigendecomposition (Sp v = lambda Sq v,
v^T Sq v = I) collapses the integral to::

    F = sum_i (d^T v_i)^2 * f(lambda_i) / (lambda_i - 1)^2,
    f(t) = t ln t - t + 1.

Ridge regularisation replaces both Sigmas by Sigma + ridge * I, which is
exactly ridge regression at every rho. Cost is O(m^3) per row after the
moments, so a batch of rows against one pool is one stacked ``eigh``.

Two facts this module is tested on: one-hot features make F equal the
plug-in KL exactly (so the estimator only matters with *shared* features),
and F never exceeds the true KL with exact moments.
"""

from __future__ import annotations

import math

import numpy as np

#: Ridge added to both covariances. Sigma_q of a smoothed pool is near
#: singular (unseen bins carry ~1e-9 mass), and this is what keeps the
#: generalized eigenproblem well conditioned.
DEFAULT_RIDGE = 1e-3

_LN2 = math.log(2.0)


def nibble_features() -> np.ndarray:
    """256 x 32 feature map: one-hot of the high nibble, one-hot of the low.

    Shares statistical strength across the 256 byte values (16 + 16 weights
    instead of 256), which is the regime the spectral estimator exists for.
    """
    phi = np.zeros((256, 32), dtype=np.float64)
    values = np.arange(256)
    phi[values, values >> 4] = 1.0
    phi[values, 16 + (values & 15)] = 1.0
    return phi


def _kl_weight(lam: np.ndarray) -> np.ndarray:
    """f(t) / (t - 1)^2 for f(t) = t ln t - t + 1; 1/2 at t = 1, 1 at t = 0."""
    lam = np.maximum(lam, 0.0)
    delta = lam - 1.0
    out = np.empty_like(lam)
    near = np.abs(delta) < 1e-4
    far = ~near
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(lam > 0.0, lam * np.log(np.where(lam > 0.0, lam, 1.0)), 0.0) - lam + 1.0
        out[far] = (f / (delta * delta))[far]
    d = delta[near]
    out[near] = 0.5 - d / 6.0 + d * d / 12.0  # series of f(t)/(t-1)^2 at t = 1
    return out


def spectral_kl_nats(
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
    ridge: float = DEFAULT_RIDGE,
) -> float:
    """F(p || q, phi) in nats from one pair of moments (see module docs)."""
    m = len(mu_p)
    out = _spectral_stack(
        mu_p[None, :], sigma_p[None, :, :], mu_q, sigma_q + ridge * np.eye(m), ridge
    )
    return float(out[0])


def _spectral_stack(
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_q_ridged: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Batched F in nats: rows of moments against one (already ridged) q."""
    m = mu_p.shape[1]
    chol = np.linalg.cholesky(sigma_q_ridged)
    chol_inv = np.linalg.inv(chol)
    a = sigma_p + ridge * np.eye(m)[None, :, :]
    # C = L^-1 (Sp + ridge I) L^-T is symmetric with the pencil's eigenvalues.
    c = chol_inv[None, :, :] @ a @ chol_inv.T[None, :, :]
    c = 0.5 * (c + np.transpose(c, (0, 2, 1)))
    lam, vecs = np.linalg.eigh(c)
    d_tilde = (mu_p - mu_q[None, :]) @ chol_inv.T
    coef = np.einsum("nm,nmk->nk", d_tilde, vecs)
    total: np.ndarray = np.sum(coef * coef * _kl_weight(lam), axis=1)
    clipped: np.ndarray = np.maximum(total, 0.0)
    return clipped


def spectral_kl_rows_bits(
    rows: np.ndarray,
    q: np.ndarray,
    phi: np.ndarray | None = None,
    ridge: float = DEFAULT_RIDGE,
) -> np.ndarray:
    """F in bits for each row distribution (N x 256) against one pool ``q``.

    A row of zeros (an empty seed) scores 0. ``phi`` defaults to
    :func:`nibble_features`.
    """
    feats = nibble_features() if phi is None else phi
    m = feats.shape[1]
    rows = np.asarray(rows, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    mu_q = q @ feats
    sigma_q = (feats.T * q) @ feats + ridge * np.eye(m)
    mu_p = rows @ feats
    outer = (feats[:, :, None] * feats[:, None, :]).reshape(feats.shape[0], m * m)
    sigma_p = (rows @ outer).reshape(len(rows), m, m)

    out = _spectral_stack(mu_p, sigma_p, mu_q, sigma_q, ridge) / _LN2
    out[~rows.any(axis=1)] = 0.0
    return out


def null_spectral_curve(
    q: np.ndarray,
    grid: object,
    draws: int = 200,
    seed: int = 0,
    phi: np.ndarray | None = None,
    ridge: float = DEFAULT_RIDGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std-dev, in bits, of F for n draws from ``q``, for each n in ``grid``.

    The spectral score has no closed-form null (unlike plug-in KL), so both
    moments are Monte-Carlo over ``draws`` multinomial samples per n from a
    private fixed-seed generator: a calibration table, never a scheduling
    decision, and reproducible.
    """
    probs = np.asarray(q, dtype=np.float64)
    probs = probs / probs.sum()
    gen = np.random.Generator(np.random.PCG64(seed))
    means: list[float] = []
    sds: list[float] = []
    for n in np.asarray(grid, dtype=np.int64):
        rows = gen.multinomial(int(n), probs, size=draws).astype(np.float64) / n
        bits = spectral_kl_rows_bits(rows, probs, phi, ridge)
        means.append(float(bits.mean()))
        sds.append(float(bits.std(ddof=1)))
    return np.asarray(means), np.asarray(sds)
