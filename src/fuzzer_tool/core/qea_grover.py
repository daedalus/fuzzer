"""Grover-style M/N-adaptive rotation angle for QEA (prototype, NOT wired in).

MEASURED RESULT (tools/sweep_qea_grover_angle.py, n=256 bits): the step-size
formula is exact, but the M estimate it needs is not obtainable this way.
OR-accumulating ``best ^ collapsed`` on strict improvements saturates toward N
(true M=32 of 256: 122 bits marked after ONE improvement, 241 after ~30),
because dead bits also differ between two random collapses. The angle then pins
at ``max_angle`` and the adaptive mode behaves like constant 0.20 (median 3584
evals, 3/10 failures at M/N=0.125, versus 168 for constant 0.05). Restricting
evidence to improvements with Hamming diff <= 8 never yields a sample. So the
wiring into ``QEALifecycle`` was dropped; what remains is the math and the
harness. A usable M estimator needs single-bit attribution (flip one bit, see
whether fitness moves), not diffs between random collapses.


Grover's algorithm on N items with M marked ones rotates the state by
``2*theta`` per iteration in the span of |good>, |bad>, where
``sin(theta) = sqrt(M/N)``; the success probability after k iterations is
``sin^2((2k+1)*theta)``, so the optimum is ``k* = pi/(4*theta) - 1/2`` and
running past it *loses* amplitude (overshoot).

QEA's ``rotation_gate`` uses a constant per-bit step ``delta``. This module
supplies the M/N-dependent step Grover would use, with

* N = number of input bits an individual represents, and
* M = number of bits with *evidence of mattering*: bits where a collapse that
  strictly beat the lineage's best differed from that best. The evidence is
  accumulated by :class:`fuzzer_tool.core.live_bit_mask.LiveBitMaskEstimator`
  (the OR-accumulator), i.e. a monotone lower bound on M -- so the resulting
  angle is a lower bound on the "true" Grover angle, never an overestimate.

Honest limits (why this is a prototype, off by default):

* Grover's speedup comes from *coherent* amplitude on all marked states at
  once. QEA's amplitudes are a product distribution over bits; only the
  *step-size formula* transfers, not the quadratic speedup.
* The OR-mask is a lower bound on M and can only grow. Early on, M is
  underestimated, so the angle is conservative.
* ``edge_count > parent.edge_count`` (strict) is used as evidence, not the
  ``>=`` used for promotion in ``QEALifecycle``: ties are the common case and
  would mark every differing bit as "live".
"""

from __future__ import annotations

import math

from fuzzer_tool.core.live_bit_mask import LiveBitMaskEstimator

GROVER_GAIN_DEFAULT = 1.0
GROVER_MIN_ANGLE_DEFAULT = 0.005
GROVER_MAX_ANGLE_DEFAULT = 0.2


def grover_theta(m: int, n: int) -> float:
    """theta = asin(sqrt(m/n)). ``m`` is clamped into [0, n]."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    m = min(max(m, 0), n)
    return math.asin(math.sqrt(m / n))


def grover_k_opt(m: int, n: int) -> float:
    """Real-valued optimal Grover iteration count, ``pi/(4*theta) - 1/2``.

    Returns ``inf`` for m == 0 (no marked item: no iteration count helps).
    """
    theta = grover_theta(m, n)
    if theta == 0.0:
        return math.inf
    return math.pi / (4.0 * theta) - 0.5


def grover_success_probability(k: int, m: int, n: int) -> float:
    """sin^2((2k+1)*theta): exact Grover success probability after k steps."""
    return math.sin((2 * k + 1) * grover_theta(m, n)) ** 2


def grover_angle(
    m: int,
    n: int,
    *,
    gain: float = GROVER_GAIN_DEFAULT,
    min_angle: float = GROVER_MIN_ANGLE_DEFAULT,
    max_angle: float = GROVER_MAX_ANGLE_DEFAULT,
) -> float:
    """Per-step rotation ``gain * 2*theta(m, n)``, clamped to [min, max]."""
    return min(max_angle, max(min_angle, gain * 2.0 * grover_theta(m, n)))


class LiveFractionTracker:
    """Per-lineage M/N estimate built on :class:`LiveBitMaskEstimator`."""

    def __init__(self, n_bits: int) -> None:
        self.n_bits = n_bits
        self._est = LiveBitMaskEstimator(n_bits)

    @property
    def m(self) -> int:
        return self._est.mask.bit_count()

    def observe_improvement(self, best_bits: int, collapsed_bits: int) -> None:
        """Record a strict improvement: differing bits are evidence of liveness."""
        self._est.observe(best_bits, collapsed_bits)

    def angle(self, fallback: float, **kw: float) -> float:
        """Grover angle once any evidence exists, else ``fallback``."""
        m = self.m
        return fallback if m == 0 else grover_angle(m, self.n_bits, **kw)
