"""Overlapping Allan deviation for classifying fuzzer time-series noise.

Standalone diagnostic, same status as ``core/kuramoto.py`` and
``core/pll.py``: not wired into any scheduler, analyzer, or the CLI.
Built to close the open question left in both of those modules'
docstrings and in ``docs/handover/handover_phase_noise_analysis_2026-09-26.md``:
is a given series' drift (or a PLL's frequency-tracking residual) white
frequency noise, flicker (1/f) noise, or a sustained random-walk / drift
component (e.g. thermal throttling)? See that handover for the
phase-noise-theory background this implements.

Held to the same synthetic-only validation bar ``kuramoto.py`` and
``pll.py`` were held to before any real-campaign data was available:
this has only been checked against synthetic white-frequency-noise and
random-walk-frequency-noise series in its own test suite, not against a
real campaign. Whether the fuzzer's own exec-time or discovery-edge
series match any of these idealized regimes is the same open empirical
question those two modules' docstrings already leave unanswered.

Model
-----
Given a sequence of fractional-frequency samples ``y_0 .. y_{N-1}`` at a
fixed sample spacing ``tau0``, the fully-overlapping Allan variance at
averaging time ``tau = m * tau0`` is

    sigma_y^2(tau) = 1 / (2*(N - 2m)) * sum_{i=0}^{N-2m-1}
                     (ybar_{i+m} - ybar_i)^2

where ``ybar_i`` is the mean of the ``m`` samples starting at index
``i``. This is the standard overlapping-Allan-variance estimator from
the phase/frequency-stability literature (Rubiola's textbook, cited by
the Wikipedia phase-noise article that prompted this module) -- nothing
here is a novel estimator, only its application to fuzzer telemetry is.

The Allan deviation ``sigma_y(tau) = sqrt(sigma_y^2(tau))`` follows a
power law ``sigma_y(tau) ~ tau^mu`` whose exponent identifies the
dominant noise type over that range of ``tau``:

    mu = -1    white or flicker phase noise (steep drop, short tau)
    mu = -1/2  white frequency noise (uncorrelated frequency samples)
    mu =  0    flicker frequency noise (the "flicker floor")
    mu = +1/2  random-walk frequency noise (e.g. slow thermal drift)
    mu = +1    deterministic linear frequency drift

Real series typically show more than one regime across different
``tau`` ranges -- this is the entire point of plotting Allan deviation
instead of reporting a single variance number -- so :func:`allan_deviation`
returns the full per-``tau`` curve and :func:`classify_segments`
classifies the *local* slope between each adjacent pair of points
rather than collapsing the whole curve to one label.

Estimator variance grows with ``m`` (fewer independent overlapping
windows: ``N - 2m`` of them), so the local-slope estimate between two
adjacent points gets noisier as ``m`` approaches ``N/2`` -- expected,
not a bug; :class:`Segment` carries the ``n_pairs`` of both endpoints
so a caller can discount low-count segments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class NoiseRegime(Enum):
    """Power-law noise type identified by an Allan-deviation slope."""

    PHASE_NOISE = "phase_noise"  # mu ~ -1
    WHITE_FM = "white_fm"  # mu ~ -1/2
    FLICKER_FM = "flicker_fm"  # mu ~ 0
    RANDOM_WALK_FM = "random_walk_fm"  # mu ~ +1/2
    DRIFT = "drift"  # mu ~ +1
    UNKNOWN = "unknown"  # doesn't land near any canonical exponent


#: Canonical Allan-deviation slope for each noise regime (see module
#: docstring). classify_segments matches an observed slope to whichever
#: of these it falls within _SLOPE_HALF_WIDTH of.
_CANONICAL_SLOPES: dict[NoiseRegime, float] = {
    NoiseRegime.PHASE_NOISE: -1.0,
    NoiseRegime.WHITE_FM: -0.5,
    NoiseRegime.FLICKER_FM: 0.0,
    NoiseRegime.RANDOM_WALK_FM: 0.5,
    NoiseRegime.DRIFT: 1.0,
}
#: Half-width of the matching band around each canonical slope. The
#: canonical slopes are spaced 0.5 apart, so 0.25 tiles them with no gaps
#: and no overlap; a slope exactly halfway between two regimes (e.g.
#: -0.25) matches whichever ``min`` picks first, which is fine since it
#: is genuinely ambiguous between the two at that point.
_SLOPE_HALF_WIDTH = 0.25


@dataclass(frozen=True)
class AllanPoint:
    """One point of the Allan deviation curve.

    Args:
        tau: Averaging time (``m * tau0``), in ``tau0``'s units.
        m: Averaging factor.
        adev: Allan deviation at this tau.
        n_pairs: Number of overlapping windows the estimate was formed
            from (``N - 2m``). Larger is a more reliable estimate.
    """

    tau: float
    m: int
    adev: float
    n_pairs: int


@dataclass(frozen=True)
class Segment:
    """Local log-log Allan-deviation slope between two adjacent points."""

    tau_lo: float
    tau_hi: float
    slope: float
    regime: NoiseRegime


def _classify_slope(mu: float) -> NoiseRegime:
    best = min(_CANONICAL_SLOPES, key=lambda r: abs(_CANONICAL_SLOPES[r] - mu))
    if abs(_CANONICAL_SLOPES[best] - mu) <= _SLOPE_HALF_WIDTH:
        return best
    return NoiseRegime.UNKNOWN


def _default_m_values(n: int) -> list[int]:
    max_m = max(1, n // 4)
    out = []
    m = 1
    while m <= max_m:
        out.append(m)
        m *= 2
    return out


def allan_deviation(
    y: Sequence[float],
    tau0: float = 1.0,
    m_values: Sequence[int] | None = None,
) -> list[AllanPoint]:
    """Overlapping Allan deviation of frequency-like samples ``y``.

    Args:
        y: Samples evenly spaced by ``tau0`` -- a frequency estimate, a
           rate, or any series whose *level* (not phase) is the
           quantity of interest. Must have at least 4 finite samples.
        tau0: Sample spacing (seconds, ticks, execs -- caller's units).
        m_values: Averaging factors to evaluate (``tau = m * tau0``).
           Defaults to powers of two from 1 up to ``len(y) // 4``.

    Returns:
        One :class:`AllanPoint` per usable ``m``, ascending by ``tau``.
        An ``m`` that would leave fewer than 1 overlapping window
        (``2*m >= len(y)``) is silently skipped rather than raising --
        the same "degrade rather than fail" pattern already used by
        ``services/stats.py`` for sparse data -- so explicit
        ``m_values`` may return fewer points than given.

    Raises:
        ValueError: fewer than 4 finite samples in ``y``, ``tau0 <= 0``,
            or an explicit ``m_values`` entry is not a positive int.
    """
    ys = [float(v) for v in y]
    if len(ys) < 4:
        raise ValueError(f"need at least 4 samples, got {len(ys)}")
    if not all(math.isfinite(v) for v in ys):
        raise ValueError("y must contain only finite values")
    if not tau0 > 0:
        raise ValueError(f"tau0 {tau0!r} must be positive")

    n = len(ys)
    if m_values is None:
        m_values = _default_m_values(n)
    else:
        for m in m_values:
            if not (isinstance(m, int) and m > 0):
                raise ValueError(f"m_values entries must be positive ints, got {m!r}")

    # Prefix sums for O(1) windowed means; O(N) total instead of the
    # naive O(N*m) per m.
    prefix = [0.0]
    for v in ys:
        prefix.append(prefix[-1] + v)

    def window_mean(start: int, m: int) -> float:
        return (prefix[start + m] - prefix[start]) / m

    points: list[AllanPoint] = []
    for m in sorted(set(m_values)):
        n_pairs = n - 2 * m
        if n_pairs < 1:
            continue
        total = 0.0
        for i in range(n_pairs):
            diff = window_mean(i + m, m) - window_mean(i, m)
            total += diff * diff
        variance = total / (2.0 * n_pairs)
        points.append(AllanPoint(tau=m * tau0, m=m, adev=math.sqrt(variance), n_pairs=n_pairs))
    return points


def classify_segments(points: Sequence[AllanPoint]) -> list[Segment]:
    """Local noise-regime classification between each adjacent pair of points.

    A single global slope over the whole curve would conflate regimes
    that show up at different averaging times -- the entire reason to
    plot Allan deviation instead of reporting one number -- so this
    classifies each adjacent pair independently instead of fitting one
    line through all of them.

    Points with ``adev == 0`` (a perfectly flat window-to-window mean,
    e.g. a constant series) are skipped since ``log(0)`` is undefined;
    a genuinely flat series produces no segments at all rather than
    raising.
    """
    usable = [p for p in points if p.adev > 0.0]
    segments: list[Segment] = []
    for a, b in zip(usable, usable[1:], strict=False):
        if a.tau == b.tau:
            continue
        slope = (math.log(b.adev) - math.log(a.adev)) / (math.log(b.tau) - math.log(a.tau))
        segments.append(Segment(a.tau, b.tau, slope, _classify_slope(slope)))
    return segments
