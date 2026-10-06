"""Universal-pattern field and its phase-coherence diagnostic.

Standalone diagnostic, in the same spirit as ``kuramoto.py``: not wired into
any scheduler or the CLI. The field is a sum of angular harmonics with Bessel
radial envelopes::

    Psi(r, th, t) = sum_n A_n * J_n(k_n r + beta n) * cos(n th + phi_n + w_n t)

with optional pairwise coupling (``nonlinear_lambda``) and ``recursion_depth``
phi^-1-scaled radial copies. ``mode_coherence`` reports the Kuramoto order
parameter of the mode phases ``n * th0 + phi_n + w_n t`` -- r = 1 means every
mode peaks at the same angle -- reusing ``circular_stats.order_parameter``.

Inputs are validated rather than coerced silently: integer modes only,
``len(amplitudes) == len(modes)``, and ``r`` / ``theta`` broadcast together.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from fuzzer_tool.core.bessel import jn
from fuzzer_tool.core.circular_stats import order_parameter

PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI


def _check_modes(modes: Sequence[int], amplitudes: Sequence[float] | None):
    ms = []
    for m in modes:
        if int(m) != m or m < 0:
            raise ValueError(f"modes must be non-negative integers, got {m!r}")
        ms.append(int(m))
    if not ms:
        raise ValueError("modes must be non-empty")
    if amplitudes is None:
        amps = np.full(len(ms), 1.0 / len(ms))
    else:
        amps = np.asarray(amplitudes, dtype=np.float64)
        if amps.shape != (len(ms),):
            raise ValueError(f"amplitudes length {amps.size} != modes length {len(ms)}")
    return ms, amps


def universal_pattern(
    r,
    theta,
    t: float = 0.0,
    modes: Sequence[int] = (5, 6, 12),
    amplitudes: Sequence[float] | None = None,
    k0: float = 3.0,
    alpha_k: float = 0.0,
    omega0: float = 0.4,
    alpha_omega: float = 1.0,
    phase_spread: float = 0.0,
    radial_phase: float = 0.0,
    recursion_depth: int = 1,
    recursion_decay: float = 0.5,
    nonlinear_lambda: float = 0.0,
) -> np.ndarray:
    """Evaluate the field on broadcast ``(r, theta)``; never mutates inputs."""
    ms, amps = _check_modes(modes, amplitudes)
    rr = np.asarray(r, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    rr, th = np.broadcast_arrays(rr, th)
    if recursion_depth < 0:
        raise ValueError("recursion_depth must be >= 0")

    psi = np.zeros(rr.shape, dtype=np.float64)
    weight = 1.0
    cur = rr
    for _ in range(recursion_depth):
        fields = []
        for i, n in enumerate(ms):
            k_n = k0 * (n**alpha_k if n > 0 else (1.0 if alpha_k == 0 else 0.0))
            w_n = omega0 * (n**alpha_omega if n > 0 else (1.0 if alpha_omega == 0 else 0.0))
            fields.append(
                amps[i]
                * jn(n, k_n * cur + radial_phase * n)
                * np.cos(n * th + i * phase_spread + w_n * t)
            )
        level = np.sum(fields, axis=0)
        if nonlinear_lambda != 0.0:
            # sum_{i<j} f_i f_j = ((sum f)^2 - sum f^2) / 2, O(M) not O(M^2)
            sq = np.sum([f * f for f in fields], axis=0)
            level = level + nonlinear_lambda * 0.5 * (level * level - sq)
        psi += weight * level
        cur = cur * PHI_INV
        weight *= recursion_decay
    return psi


def common_symmetry(modes: Sequence[int]) -> int:
    """GCD of the non-zero modes: the field has exactly this rotational order."""
    g = 0
    for m in modes:
        g = math.gcd(g, int(m))
    return g


def mode_coherence(
    theta0: float,
    modes: Sequence[int],
    t: float = 0.0,
    omega0: float = 0.4,
    alpha_omega: float = 1.0,
    phase_spread: float = 0.0,
    amplitudes: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Kuramoto ``(r, psi)`` of the mode phases at angle *theta0*."""
    ms, amps = _check_modes(modes, amplitudes)
    phases = [
        n * theta0 + i * phase_spread + omega0 * (n**alpha_omega if n > 0 else 0.0) * t
        for i, n in enumerate(ms)
    ]
    return order_parameter(phases, np.abs(amps))
