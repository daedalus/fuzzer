"""Circular statistics over byte offsets folded into record phase.

``periodicity.estimate_record_size`` infers a record stride ``L`` for a seed
(a capture file is packets, a bitmap is scanlines, a table file is rows).
Every consumer of that stride so far treats records as opaque: they get
shuffled, or a field hypothesis is accepted only when it lands on a record
boundary.  Nothing asks the question the stride actually enables — *which
offset inside the record is where coverage comes from*.

That question is circular, not linear.  Fold an absolute offset ``o`` to a
phase on the record::

    theta = 2*pi * (o mod L) / L

and a set of productive offsets becomes a set of points on a circle.  Their
Kuramoto order parameter is the standard summary::

    r * exp(i*psi) = sum(w_k * exp(i*theta_k)) / sum(w_k)

``r`` in [0, 1] measures how phase-locked the offsets are — 0 when they are
spread around the record, 1 when they all sit at the same field — and
``psi`` names the field.  Under the null "offsets are uniform around the
record" the same ``r`` is the Rayleigh test statistic, which has a closed
form, so the concentration can be accepted or rejected at a stated alpha
rather than by a tuned ratio.

Why this is worth having: coverage feedback about byte positions is
expensive, so it is collected for low absolute offsets only (see
``te_position.update_te_causal_map``, capped at 64 bytes).  A *confirmed*
phase lock extrapolates that cheap local evidence across the whole buffer —
offset 5 being hot at stride 16 predicts 21, 37, 53, ... — which is exactly
the inference the linear position maps cannot make.  A rejected one says
the extrapolation is not warranted, which is the more important half.

ASCII, stride 8, offsets {3, 11, 19, 27}::

      record 0   record 1   record 2   record 3
      |..X.....| |..X.....| |..X.....| |..X.....|     folded ->   . X . . . . . .
                                                                    ^ r = 1, psi -> offset 3

Weighting: callers pass evidence weights (edge counts, hit counts), which
are wildly unequal.  Significance therefore uses Kish's effective sample
size ``(sum w)^2 / sum(w^2)`` rather than the number of observations, so a
single dominant position — which is phase-locked with itself by
construction — cannot manufacture a result.

Reference
---------
Kuramoto, "Chemical Oscillations, Waves, and Turbulence", §5.2 (order
parameter).  Mardia & Jupp, "Directional Statistics", §6.3 (Rayleigh test).
Wilkie, "Rayleigh Test for Randomness of Circular Data", Appl. Stat. 32
(1983) — the closed-form p-value used here.
Kish, "Survey Sampling" (1965), §8.2 (effective sample size).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

TAU = 2.0 * math.pi

# Phase is undefined for a stride of 1: every offset folds to phase 0, so r
# is 1 by construction and carries no information about the buffer.
MIN_STRIDE = 2

# Default significance for accepting a phase lock. Deliberately tighter than
# a conventional 0.05: the result is used to redirect mutation effort across
# an entire buffer, and the test is evaluated on every consulted seed.
DEFAULT_ALPHA = 0.01


@dataclass(frozen=True)
class PhaseConcentration:
    """Result of folding offsets onto a record and testing for a lock.

    Attributes:
        r: Kuramoto order parameter in [0, 1].
        psi: Mean phase in radians, [0, TAU).
        p_value: Rayleigh p-value against the uniform-phase null.
        offset: ``psi`` expressed as a byte offset in [0, stride).
        stride: The record stride the offsets were folded onto.
        effective_n: Kish effective sample size behind the test.
    """

    r: float
    psi: float
    p_value: float
    offset: int
    stride: int
    effective_n: float

    def is_locked(self, alpha: float = DEFAULT_ALPHA) -> bool:
        """True when the uniform-phase null is rejected at *alpha*."""
        return self.p_value < alpha


def order_parameter(
    phases: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float]:
    """Kuramoto order parameter ``(r, psi)`` of *phases*.

    Returns ``(0.0, 0.0)`` for empty input. ``psi`` is meaningless when
    ``r`` is 0 and is reported as the ``atan2`` branch value regardless.
    """
    theta = np.asarray(phases, dtype=np.float64)
    if theta.size == 0:
        return 0.0, 0.0

    if weights is None:
        c = float(np.cos(theta).mean())
        s = float(np.sin(theta).mean())
    else:
        w = np.asarray(weights, dtype=np.float64)
        total = float(w.sum())
        if total <= 0.0:
            return 0.0, 0.0
        c = float(np.cos(theta) @ w) / total
        s = float(np.sin(theta) @ w) / total

    return math.hypot(c, s), math.atan2(s, c) % TAU


def rayleigh_pvalue(n: float, r: float) -> float:
    """P-value of the Rayleigh test for circular uniformity.

    Wilkie's closed form, which is accurate to well under the Monte-Carlo
    error of any calibration we would run against it, and needs no series
    or table::

        p = exp(sqrt(1 + 4n + 4n^2 (1 - r^2)) - (1 + 2n))

    Args:
        n: Sample size — the *effective* size when observations are weighted.
        r: Order parameter in [0, 1].
    """
    if n <= 0.0:
        return 1.0

    r = min(max(r, 0.0), 1.0)
    k = 1.0 + 2.0 * n

    return min(1.0, math.exp(math.sqrt(k * k - 4.0 * n * n * r * r) - k))


def fold_offsets(offsets: Sequence[int] | np.ndarray, stride: int) -> np.ndarray:
    """Map absolute byte offsets to phases on a *stride*-byte record."""
    o = np.asarray(offsets, dtype=np.int64)

    return TAU * (o % stride) / stride


def concentration(
    offsets: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None,
    stride: int,
) -> PhaseConcentration | None:
    """Test whether *offsets* concentrate at one phase of a *stride* record.

    Returns ``None`` when the question is not askable: a stride below
    ``MIN_STRIDE``, no offsets, or weights that are not a usable positive
    measure.  ``None`` means "no answer", never "no lock" — callers that
    need the distinction read :attr:`PhaseConcentration.is_locked`.
    """
    if stride < MIN_STRIDE:
        return None

    o = np.asarray(offsets, dtype=np.int64)
    if o.size == 0:
        return None

    if weights is None:
        n_eff = float(o.size)
        w = None
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.size != o.size or np.any(w < 0.0):
            return None
        total = float(w.sum())
        sq = float(w @ w)
        if total <= 0.0 or sq <= 0.0:
            return None
        # Kish: unequal weights buy less evidence than their count suggests.
        n_eff = total * total / sq

    r, psi = order_parameter(fold_offsets(o, stride), w)
    offset = int(round(psi / TAU * stride)) % stride

    return PhaseConcentration(
        r=r,
        psi=psi,
        p_value=rayleigh_pvalue(n_eff, r),
        offset=offset,
        stride=stride,
        effective_n=n_eff,
    )
