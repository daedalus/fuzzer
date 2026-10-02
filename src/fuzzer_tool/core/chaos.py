"""Chaotic inertia schedule for the swarm schedulers (MOpt, firefly).

The r = 4 logistic map ``z <- 4 z (1 - z)`` replaces a constant inertia
weight with a deterministic, aperiodic sequence on (0, 1)::

    z:  0.20 -> 0.64 -> 0.92 -> 0.29 -> 0.82 -> ...
    w = W_MIN + (W_MAX - W_MIN) * z     (MOpt velocity)
    a = alpha * 2 * z                   (firefly random step, mean alpha)

Its invariant density is arcsine, piling up near 0 and 1, so the swarm
alternates between short-memory exploration and long-memory exploitation
windows instead of holding one compromise.

Float hazard: in binary64 the orbit can land on 0.5 (-> 1 -> 0 forever) or
the fixed point 0.75. Any such landing is re-drawn from the shared RandPool
(Hard Rule 16), so seeded runs stay reproducible.
"""

from __future__ import annotations

import enum

from fuzzer_tool.core.rand_pool import RandPool

# Logistic parameter: r = 4 is the fully chaotic case on [0, 1].
_LOGISTIC_R = 4.0

# Orbit points closer than this to 0 or 1 are treated as collapsed.
LOGISTIC_EPS = 1e-9

# Non-zero fixed point of the r = 4 map: 1 - 1/r.
_FIXED_POINT = 1.0 - 1.0 / _LOGISTIC_R

# MOpt chaotic inertia band. Arcsine mean 0.5 maps to 0.65, near the
# constant default 0.7.
W_MIN = 0.4
W_MAX = 0.9

# Firefly step multiplier: 2 * E[z] = 1 keeps the mean step at alpha.
_ALPHA_SCALE = 2.0


class InertiaMode(enum.Enum):
    """How a swarm scheduler sets its per-window inertia."""

    CONSTANT = "constant"  # fixed w / alpha (default)
    CHAOTIC = "chaotic"  # logistic-map driven


class LogisticMap:
    """r = 4 logistic map with float-collapse reseeding.

    Args:
        rng: Shared RandPool; draws the start and every reseed.
        z0: Explicit start (tests). ``None`` draws one from ``rng``.
    """

    def __init__(self, rng: RandPool, z0: float | None = None) -> None:
        self._rng = rng
        self.z = self._draw() if z0 is None else z0

    def _draw(self) -> float:
        """Uniform draw mapped into the open interval (EPS, 1 - EPS)."""
        return LOGISTIC_EPS + (1.0 - 2.0 * LOGISTIC_EPS) * self._rng.random()

    def step(self) -> float:
        """Advance one iterate; reseed if the orbit collapsed."""
        z = _LOGISTIC_R * self.z * (1.0 - self.z)

        # Collapse: 0.5 -> 1 -> 0, or the 0.75 fixed point.
        if z <= LOGISTIC_EPS or z >= 1.0 - LOGISTIC_EPS or z == _FIXED_POINT:
            z = self._draw()

        self.z = z
        return z

    def inertia(self) -> float:
        """Next MOpt inertia weight in [W_MIN, W_MAX]."""
        return W_MIN + (W_MAX - W_MIN) * self.step()

    def alpha_factor(self) -> float:
        """Next firefly step multiplier, mean 1."""
        return _ALPHA_SCALE * self.step()


def make_chaos(mode: InertiaMode, rng: RandPool) -> LogisticMap | None:
    """Map for CHAOTIC mode; None for CONSTANT so its RNG stream is untouched."""
    if mode is InertiaMode.CHAOTIC:
        return LogisticMap(rng)
    return None
