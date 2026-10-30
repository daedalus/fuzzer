"""Synthetic series generators for calibrating detectors against known nulls.

Each generator draws only from a :class:`RandPool` (Hard Rule 16) and returns
a numpy array. They are the processes the periodicity, GARCH and regime
detectors are measured on (see ``detect_periodicity``'s false-positive table):

- :func:`ar1`   -- stationary Gaussian AR(1), lag-1 autocorrelation ``phi``.
- :func:`pink`  -- 1/f^beta noise, power spectrum slope exactly ``-beta``.
- :func:`cox`   -- Poisson counts whose log-rate drifts as an OU process.
- :func:`garch` -- GARCH(1,1), volatility clustering at constant mean.
"""

from __future__ import annotations

import math

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

COX_TAU = 40.0  # OU correlation time of the log-rate, in samples
COX_BASE = 3.0  # rate at the OU mean
COX_SIGMA = 0.6  # stationary std-dev of the log-rate
COX_MIN_RATE = 0.05  # rate floor: keeps every tick able to fire


def _walk(x0: float, phi: float, eps: list[float]) -> np.ndarray:
    """AR(1) recursion ``x[t] = phi * x[t-1] + eps[t-1]`` from ``x0``.

    Sequential by nature (no lfilter without scipy); a Python list loop is
    ~3x faster here than indexing numpy scalars.
    """
    out = [x0]
    x = x0
    for e in eps:
        x = phi * x + e
        out.append(x)
    return np.array(out)


def ar1(rng: RandPool, n: int, phi: float) -> np.ndarray:
    """Stationary Gaussian AR(1) with unit innovation variance.

    ``x[0]`` is drawn from the stationary law N(0, 1/(1-phi^2)), so there is
    no burn-in transient.

    Raises:
        ValueError: ``|phi| >= 1`` (no stationary law).
    """
    if abs(phi) >= 1.0:
        raise ValueError(f"ar1 needs |phi| < 1, got {phi}")
    if n <= 0:
        return np.empty(0)

    e = rng.normal_array(n)
    return _walk(float(e[0]) / math.sqrt(1.0 - phi * phi), phi, e[1:].tolist())


def pink(rng: RandPool, n: int, beta: float) -> np.ndarray:
    """1/f^beta noise via inverse rFFT: exact amplitudes, uniform phases.

    The power at frequency f is exactly ``c * f^-beta`` (Nyquist excepted,
    whose phase is discarded by ``irfft``); ``beta=0`` is white. Zero mean.
    """
    if n <= 0:
        return np.empty(0)

    f = np.fft.rfftfreq(n)[1:]
    phase = rng.random_array(f.size) * (2.0 * math.pi)
    spec = np.concatenate(([0.0], f ** (-beta / 2.0) * np.exp(1j * phase)))
    return np.fft.irfft(spec, n=n)


def cox(
    rng: RandPool,
    n: int,
    tau: float = COX_TAU,
    base: float = COX_BASE,
    sigma: float = COX_SIGMA,
) -> np.ndarray:
    """Poisson counts whose rate ``base * exp(sigma * z)`` drifts; z is OU.

    A doubly stochastic (Cox) process: counts are overdispersed
    (var > mean) and autocorrelated over ~``tau`` samples. ``sigma=0``
    freezes the rate and reduces it to Poisson(base).

    Raises:
        ValueError: ``tau <= 0`` or ``base < 0``.
    """
    if tau <= 0.0 or base < 0.0:
        raise ValueError(f"cox needs tau > 0 and base >= 0, got tau={tau}, base={base}")
    if n <= 0:
        return np.empty(0, dtype=np.int64)

    # Unit-variance OU sampled at step 1: z[t] = a z[t-1] + sqrt(1-a^2) e[t].
    a = math.exp(-1.0 / tau)
    e = rng.normal_array(n) * math.sqrt(1.0 - a * a)
    z = _walk(float(e[0]), a, e[1:].tolist())

    rates = np.maximum(COX_MIN_RATE, base * np.exp(sigma * z))
    return rng.poisson_array(rates)


def garch(
    rng: RandPool,
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    mean: float = 0.0,
) -> np.ndarray:
    """GARCH(1,1): ``x[t] = mean + eps[t]``, ``eps[t] ~ N(0, s2[t])``.

    ``s2[t+1] = omega + alpha * eps[t]^2 + beta * s2[t]``, started at the
    unconditional variance ``omega / (1 - alpha - beta)``.

    Raises:
        ValueError: ``omega <= 0``, a negative coefficient, or
            ``alpha + beta >= 1`` (no finite unconditional variance).
    """
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError(
            f"garch needs omega > 0, alpha, beta >= 0, alpha + beta < 1; "
            f"got {omega}, {alpha}, {beta}"
        )
    if n <= 0:
        return np.empty(0)

    s2 = omega / (1.0 - alpha - beta)
    out = []
    for z in rng.normal_array(n).tolist():
        eps = math.sqrt(s2) * z
        out.append(eps)
        s2 = omega + alpha * eps * eps + beta * s2
    return np.array(out) + mean
