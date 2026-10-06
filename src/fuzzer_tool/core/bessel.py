"""Integer-order Bessel function of the first kind, numpy only.

Hard Rule 51: no scipy. ``jn`` evaluates Bessel's integral

    J_n(x) = (1/pi) * integral_0^pi cos(n*t - x*sin(t)) dt

with the trapezoid rule. The integrand is the restriction of a 2*pi-periodic
analytic function, so the rule converges geometrically once the node count
exceeds roughly ``|x| + n``; ``_nodes`` adds a fixed margin on top. Accuracy
is ~1e-14 absolute for the arguments this repo uses (|x| < a few hundred).

Cost is O(len(x) * nodes), which is fine for diagnostics and for scheduler
tables built once per seed, not for per-exec use.
"""

from __future__ import annotations

import numpy as np

_MARGIN = 40  # nodes beyond |x| + n; geometric convergence makes this ample
_MAX_NODES = 1 << 16  # guards against a huge |x| allocating an enormous grid


def _nodes(max_abs_x: float, n: int) -> int:
    m = int(np.ceil(max_abs_x)) + abs(n) + _MARGIN
    return min(max(m, 64), _MAX_NODES)


def jn(n: int, x) -> np.ndarray:
    """``J_n(x)`` for integer *n* and array-like *x* (always float64 out).

    Non-finite entries of *x* give NaN. ``n`` must be an integer; negative
    orders use ``J_{-n} = (-1)^n J_n``.
    """
    if int(n) != n:
        raise ValueError(f"order must be an integer, got {n!r}")
    n = int(n)

    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return out

    xs = arr[finite]
    big = np.abs(xs) * 1.0
    m = _nodes(float(big.max()), n)
    # Midpoint nodes on (0, pi): trapezoid on the periodic extension.
    tau = (np.arange(m, dtype=np.float64) + 0.5) * (np.pi / m)

    # Chunk over x so memory stays bounded for big grids.
    res = np.empty(xs.shape, dtype=np.float64)
    step = max(1, (1 << 22) // m)
    for lo in range(0, xs.size, step):
        chunk = xs[lo : lo + step, None]
        res[lo : lo + step] = np.cos(n * tau[None, :] - chunk * np.sin(tau)[None, :]).mean(axis=1)

    res[xs == 0.0] = 1.0 if n == 0 else 0.0  # exact, not 1e-17 quadrature noise
    out[finite] = res
    return out
