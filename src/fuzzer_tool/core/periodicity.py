"""FFT-based periodicity detection for structural analysis.

Two capabilities, both using only real-input FFTs (``numpy.fft.rfft`` /
``numpy.fft.irfft`` — the DFT of real data is Hermitian-symmetric, so the
non-redundant half is all that is ever computed):

1. ``estimate_record_size``: byte-level record-stride inference. Many binary
   formats are arrays of fixed-size records (packet headers in a capture
   file, pixel scanlines, table entries, TLV streams). Via the
   Wiener-Khinchin theorem the full autocorrelation function is the inverse
   FFT of the power spectral density — O(N log N) instead of O(N^2) — and a
   sharp autocorrelation peak at lag L says "this buffer is N repetitions of
   an L-byte record", discoverable from the raw bytes alone with zero
   execution and zero mutation history. The smallest locally-dominant lag is
   returned (not the global max) so a strong peak at the fundamental period
   does not get confused with its own harmonics (period 8 also peaks at
   lags 16, 24, ...).

2. ``detect_periodicity``: spectral analysis of a real-valued time series.
   A plain rfft power spectrum (DC excluded) answers "is there a
   dominant non-DC frequency, and what period does it correspond to?" —
   e.g. attributing a periodic overhead in per-execution timings or a
   periodic component in the coverage discovery-rate series to a specific
   cadence, something Allan-variance (noise-type) and dispersion-index
   (burstiness) diagnostics are not designed to catch. Significance is
   gated by Fisher's g-test: the largest periodogram ordinate divided by
   the total power is scored against its exact closed-form null
   distribution, so noise is rejected at the nominal alpha rate rather
   than by a hand-tuned ratio against the median.

No scipy is used anywhere in this project; the windowing (Hanning) and all
FFT math stay within numpy.

A second, cheaper mode is available when a prior ``expected_period`` is
known: harmonic binning and a fine peak histogram.  This is *not* a
replacement for :func:`detect_periodicity` when no prior exists — the two
are complementary tools for different situations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Minimum buffer length for a meaningful record-size estimate: 64 bytes is
# >= 8 repetitions of an 8-byte record, giving the peak scan enough signal.
DEFAULT_MIN_LEN = 64
# Upper bound on the searchable period, regardless of buffer size.
DEFAULT_MAX_LAG = 256
# Analysis window cap: the lag scan never exceeds DEFAULT_MAX_LAG (256), so a
# 16-period window over the largest searchable record keeps the FFT cost O(1)
# for arbitrarily large buffers (a 2.8 MB seed cost ~2.4 s of FFT per call
# uncapped) while retaining full detection power. Buffers shorter than the
# cap are analyzed in full — byte-identical to the uncapped path.
DEFAULT_MAX_WINDOW = 16 * DEFAULT_MAX_LAG
# Significance constant: the normalized autocorrelation of white noise at a
# given lag is approximately N(0, 1/n), so a candidate peak must exceed
# SIGMA_CUTOFF / sqrt(n) — a multiple-comparisons-aware ~4-sigma bound over
# the ~len(data)//3 lags scanned. Keeps random bytes from producing spurious
# record strides (a plain fixed threshold like 0.10 lets 1-in-10 lags through
# on a 256-byte buffer).
SIGMA_CUTOFF = 4.0

# Thresholds for the harmonic-binned periodicity classifier.
HARMONIC_PERIODIC_THRESH: float = 0.30
HARMONIC_WEAK_THRESH: float = 0.15

# Defaults for harmonic_fraction / locate_peak_period.
DEFAULT_N_HARMONICS: int = 3
DEFAULT_HARMONIC_TOLERANCE: float = 0.15
DEFAULT_PEAK_BINS: int = 200
DEFAULT_SEARCH_WIDTH: float = 0.5


def estimate_record_size(
    data: bytes,
    min_len: int = DEFAULT_MIN_LEN,
    max_lag: int | None = None,
    min_rel_peak: float = 0.10,
    peak_to_median: float = 2.0,
) -> int | None:
    """Infer the fixed record stride of a byte buffer via FFT autocorrelation.

    Applies the Wiener-Khinchin autocorrelation trick: mean-subtract and
    Hanning-window the byte values (suppressing the edge-discontinuity
    spectral leakage any finite window introduces), take the power spectral
    density via ``rfft``, then return to the lag domain with ``irfft``. The
    normalized autocorrelation peaks at lags that are multiples of the
    record stride; the *smallest* locally-dominant peak is the fundamental
    period.

    Args:
        data: Raw seed bytes to analyze.
        min_len: Buffers shorter than this return ``None`` (not enough
            repetitions of any plausible record to detect).
        max_lag: Upper bound on the searchable period in bytes. Defaults to
            ``min(len(data) // 3, 256)`` so at least three full periods must
            be observable.
        min_rel_peak: A candidate peak must reach at least this fraction of
            the lag-0 autocorrelation (total variance). The effective
            threshold is ``max(min_rel_peak, SIGMA_CUTOFF / sqrt(n))`` — the
            sigma bound rejects the spurious peaks white noise produces at
            ~1/sqrt(n) magnitude.
        peak_to_median: A candidate peak must beat the median of the scanned
            lags by at least this factor (noise-floor rejection).

    Returns:
        The inferred record stride in bytes, or ``None`` when the buffer is
        too short, constant, or has no locally-dominant periodic structure.

    Note:
        Only the first ``DEFAULT_MAX_WINDOW`` (4096) bytes are analyzed: the
        lag scan is capped at ``DEFAULT_MAX_LAG`` (256), so the remaining
        buffer would add FFT cost without adding searchable lags. Buffers of
        at most 4096 bytes are analyzed in full, byte-identical to the
        uncapped path.
    """
    if not data or len(data) < min_len:
        return None
    n = len(data)
    limit = min(max_lag if max_lag is not None else n // 3, DEFAULT_MAX_LAG)
    if limit < 2:
        return None

    # Only the first DEFAULT_MAX_WINDOW bytes are ever analyzed: the lag scan
    # is capped at DEFAULT_MAX_LAG (>= 16 periods observable in the window),
    # so the full buffer would only add FFT cost, not searchable lags. The
    # sigma bound uses the window length too — it calibrates the noise floor
    # of this specific autocorrelation estimate (ac[k] ~ N(0, 1/w)).
    w = min(n, DEFAULT_MAX_WINDOW)
    x = np.frombuffer(data[:w], dtype=np.uint8).astype(np.float64)
    x = (x - x.mean()) * np.hanning(w)
    power = np.fft.rfft(x)
    power = power * power.conj()
    ac = np.fft.irfft(power, n=w)
    total = ac[0]
    if total <= 0.0:
        return None
    ac = ac / total

    scanned = ac[1 : limit + 1]
    if scanned.size == 0:
        return None
    # Noise-floor gate: a fixed fraction of the lag-0 autocorrelation plus a
    # multiple-comparisons-aware sigma bound (white-noise ac[k] ~ N(0, 1/w)).
    threshold = max(min_rel_peak, SIGMA_CUTOFF / math.sqrt(w))
    floor = peak_to_median * float(np.median(scanned))
    for k in range(1, limit + 1):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] >= threshold and ac[k] >= floor:
            return k
    return None


@dataclass(frozen=True)
class SpectralPeriodicity:
    """Result of a spectral scan over a real-valued time series."""

    dominant_period: float | None  # series samples per cycle; None if not significant
    peak_strength: float  # Fisher g statistic: peak non-DC ordinate / total non-DC power
    peak_bin: int  # index of the dominant non-DC frequency bin (0 if none)
    period_seconds: float | None  # dominant_period * sample_interval
    n_samples: int
    significant: bool
    p_value: float = 1.0  # exact upper-tail P(G > g) under the white-noise null
    ar_order: int = 0  # order of the AR filter applied before the periodogram
    ar_coeffs: tuple[float, ...] = ()  # its coefficients, for the display


def fisher_g_pvalue(g: float, m: int, alpha: float = 0.05) -> float:
    """Exact upper-tail P(G > g) for Fisher's g over ``m`` iid exponential ordinates.

    For Gaussian white noise the periodogram ordinates of the ``m`` full
    frequency bins are i.i.d. exponential, so the g statistic (largest
    ordinate / total power) has the closed-form survival function

    ``P(G > g) = sum_{k=1}^{floor(1/g)} (-1)^(k-1) * C(m, k) * (1 - k*g)^(m-1)``.

    Terms are computed in log space (lgamma for the binomial coefficient,
    log1p for the power) so large ``m`` cannot overflow; when the first
    term is below ``alpha`` it is returned directly (an upper bound on the
    true p-value, so the ``p < alpha`` decision is exact), and when terms
    start growing the true p-value is ~1 and 1.0 is returned to avoid
    catastrophic cancellation.

    Args:
        g: Observed g statistic, in ``(0, 1]``.
        m: Number of independent ordinates (full non-DC, non-Nyquist bins).
        alpha: Significance level; only used for the early-return bounds.

    Returns:
        The exact (or conservatively bounded) p-value in ``[0, 1]``.
    """
    if m <= 0 or not 0.0 < g <= 1.0:
        return 1.0
    if g >= 1.0:
        # All spectral power in a single bin (float64 rounding of a clean
        # integer-period signal); p = 0 unless m == 1, where the formula
        # gives P(G > 1) = 0^0 = 1 by convention.
        return 0.0 if m > 1 else 1.0
    p1 = m * (1 - g) ** (m - 1)
    if p1 < alpha:
        return p1
    if p1 > 1.0:
        return 1.0
    s = 0.0
    for k in range(1, min(m, math.floor(1.0 / g)) + 1):
        if k * g >= 1.0:
            term = 0.0
        else:
            log_c = math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
            log_t = log_c + (m - 1) * math.log1p(-k * g)
            term = math.exp(log_t) if log_t > -745.0 else 0.0
        s += term if k % 2 == 1 else -term
        if k >= 2 and abs(term) > 1.0:
            return 1.0
    return s



_LN2 = math.log(2.0)

PREWHITEN_MAX_ORDER = 8
"""Largest AR order tried when estimating the spectral background.

Eight, with :data:`PREWHITEN_PEAK_CLIP` at 4.0, was the best point on the
measured trade: white-noise false positives stay at the nominal 0.046 and a
drifting-rate null drops from 0.526 to 0.102 at n=512. Order 12 buys nothing
on the null (0.104) and pushes white noise to 0.068, i.e. it starts fitting
the noise. Order 5 leaves the null at 0.132.
"""

PREWHITEN_PEAK_CLIP = 4.0
"""Ordinates above this multiple of the *local* background are clipped out of
the background fit.

This is the part that is easy to get wrong, and getting it wrong is silent.
A periodic component is itself strongly autocorrelated, so an AR model fitted
to the raw series *models the tone* and the filter then cancels the very
signal the test is looking for. Measured: a bin-64 sinusoide at amplitude 2.0
over unit white noise was reported at bin 27 — detected, wrong answer.
Clipping against a *global* median instead fails the other way: a red
background legitimately sits far above the global median, so clipping flattens
the structure that needs modelling and the null barely improves (0.560 ->
0.532). Clipping against a local running median does both jobs.
"""

PREWHITEN_CLIP_ITERS = 2
"""Clipping passes. The local median is itself computed from clipped data, so
one extra pass sharpens the background estimate under a broad peak."""


def _local_median(values: np.ndarray, width: int) -> np.ndarray:
    """Running median of *values*, window *width*, reflected at the edges."""
    m = values.size
    if width >= m:
        width = m if m % 2 else m - 1
    if width % 2 == 0:
        width -= 1
    if width < 3:
        return np.full(m, float(np.median(values)))
    half = width // 2
    padded = np.pad(values, (half, half), mode="reflect")
    return np.array([np.median(padded[i : i + width]) for i in range(m)])


def fit_ar_yule_walker(acov: np.ndarray, order: int) -> tuple[np.ndarray, float] | None:
    """Yule-Walker AR(*order*) fit from an autocovariance sequence.

    Returns ``(coefficients, residual variance)``, or None if the Toeplitz
    system is singular. Yule-Walker rather than least squares because it is a
    closed form and, more importantly here, yields a *smooth* fitted
    spectrum. A noisy background estimate is what breaks the obvious
    alternative: normalising each ordinate by a median-filtered local
    background fixes the red-noise nulls but pushes the white-noise
    false-positive rate from 0.05 to 0.16, because dividing by a noisy
    estimate inflates the tail of the maximum. Here the median filter only
    *identifies* outliers; the smooth AR fit is what does the whitening.
    """
    if order < 1 or acov.size < order + 1:
        return None
    R = np.empty((order, order), dtype=np.float64)
    for i in range(order):
        for j in range(order):
            R[i, j] = acov[abs(i - j)]
    r = acov[1 : order + 1]
    try:
        coeffs = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return None
    return coeffs, max(float(acov[0] - coeffs @ r), 1e-12)


def background_autocovariance(
    x: np.ndarray,
    clip: float = PREWHITEN_PEAK_CLIP,
    iters: int = PREWHITEN_CLIP_ITERS,
) -> np.ndarray:
    """Autocovariance of the *background* of a mean-centred series.

    Takes the periodogram, clips every ordinate down to ``clip`` times its
    local background (estimated by a running median, converted from median to
    mean by the ``/ln 2`` factor for an exponential), and inverts the clipped
    periodogram. What comes back describes the smooth part of the spectrum
    with narrowband peaks removed — so an AR model fitted to it whitens the
    background without cancelling a tone. See :data:`PREWHITEN_PEAK_CLIP`.
    """
    n = x.size
    periodogram = np.abs(np.fft.rfft(x)) ** 2 / n
    clipped = periodogram.copy()
    width = max(9, 2 * int(round(periodogram.size**0.5)) + 1)
    for _ in range(max(iters, 1)):
        background = np.maximum(_local_median(clipped, width) / _LN2, 1e-300)
        clipped = np.minimum(clipped, clip * background)
    return np.fft.irfft(clipped, n=n)


def prewhiten(
    series: Sequence[float], max_order: int = PREWHITEN_MAX_ORDER
) -> tuple[np.ndarray, int, tuple[float, ...]]:
    """Remove autocorrelated background so Fisher's g-test null applies.

    Estimates the background autocovariance with
    :func:`background_autocovariance`, fits AR(p) for p in ``0..max_order`` by
    Yule-Walker, picks p by AIC, and returns the residual
    ``x[t] - sum_k phi_k x[t-k]``. Order 0 returns the centred series
    unchanged, which is what white noise gets.

    The filter multiplies the spectrum by ``|1 - sum_k phi_k e^{-ikw}|^2``, so
    it flattens a smooth background without moving a peak: a genuine
    oscillation stays at the same frequency. What it costs is *very* low
    frequency detectability, because that is what the filter attenuates — see
    :func:`detect_periodicity` for the measured trade.

    Returns:
        ``(residual, order, coefficients)``. The residual is shorter than
        ``series`` by ``order`` samples.
    """
    x = np.asarray(series, dtype=np.float64)
    # Size check before the mean: x.mean() on an empty array warns and
    # returns nan, which would propagate silently through the fit.
    if x.size < 16 or max_order < 1:
        return (x - x.mean() if x.size else x), 0, ()
    x = x - x.mean()
    n = x.size
    if float(np.dot(x, x)) <= 0.0:
        return x, 0, ()
    acov = background_autocovariance(x)
    if acov[0] <= 0.0:
        return x, 0, ()
    best_order = 0
    best_coeffs: np.ndarray | None = None
    best_aic = n * math.log(max(float(acov[0]), 1e-300))
    for order in range(1, min(max_order, n // 4) + 1):
        fit = fit_ar_yule_walker(acov, order)
        if fit is None:
            continue
        coeffs, resid_var = fit
        aic = n * math.log(resid_var) + 2 * order
        if aic < best_aic:
            best_order, best_coeffs, best_aic = order, coeffs, aic
    if best_order == 0 or best_coeffs is None:
        return x, 0, ()
    resid = x[best_order:].copy()
    for k in range(1, best_order + 1):
        resid -= best_coeffs[k - 1] * x[best_order - k : n - k]
    return resid, best_order, tuple(float(c) for c in best_coeffs)


def detect_periodicity(
    series: Sequence[float],
    sample_interval: float = 1.0,
    min_samples: int = 64,
    alpha: float = 0.05,
    prewhiten_series: bool = True,
) -> SpectralPeriodicity:
    """Detect a dominant non-DC periodic component in a real-valued series.

    Takes the rfft power spectrum of the mean-subtracted series and
    searches for the strongest non-DC bin, scoring it with Fisher's g-test
    for hidden periodicity: the ratio of the largest periodogram ordinate
    to the total power, compared against its exact closed-form null
    distribution. The DC (mean) bin is excluded by construction — the
    question is purely "is there an oscillation at a specific frequency".
    The Nyquist bin (n even) is excluded from the peak search because its
    ordinate has one degree of freedom, not two — a period-2 alternation
    is therefore not detectable. The ``peak_bin >= 2`` gate rejects a
    "peak" at the lowest non-DC bin, which is indistinguishable from
    linear drift.

    **The null is white noise, so the series has to be whitened first.**
    Fisher's g compares the largest ordinate to the *total* power under the
    assumption that the expected spectrum is flat. It is not flat for any
    series whose rate drifts, and the discovery-rate series this is applied
    to is exactly that: ``coverage_regime.py``, ``critical_slowing.py`` and
    ``garch.py`` all exist on the premise that the rate is non-stationary. So
    ``prewhiten_series`` defaults True and an AR(p) background fit (see
    :func:`prewhiten`) flattens the spectrum before the periodogram is
    scored. Measured false-positive rates at nominal alpha=0.05, 500-1000
    replicates, raw versus pre-whitened:

    ===========================  =====  ==========  =============
    null                         n      raw         pre-whitened
    ===========================  =====  ==========  =============
    Gaussian white               256    0.049       0.048
    Gaussian white               512    0.046       0.046
    Poisson, drifting OU rate    256    0.367       0.062
    Poisson, drifting OU rate    512    0.526       0.102
    AR(1) phi=0.7                512    0.936       0.069
    AR(1) phi=0.9                512    0.848       0.055
    AR(1) phi=-0.6               512    0.980       0.046
    1/f and 1/f^2                512    0.000       0.000
    ===========================  =====  ==========  =============

    **The drifting-rate null is improved 5x but is not nominal**: 0.102
    against 0.05 at n=512. Said plainly rather than rounded away. A Poisson
    count series with a drifting rate has a Lorentzian-plus-flat-floor
    spectrum, and an order-8 AR fit cannot flatten both halves of it
    completely; raising the order to 12 does not help the null (0.104) and
    inflates the white-noise rate to 0.068, which is the fit starting to
    model the noise. Treat a PERIODIC verdict on a drifting series as worth a
    look, not as established.

    Two things worth knowing before changing this. First, pure 1/f was
    already handled, and not by the null: its peak collapses into bin 1,
    which the ``peak_bin >= 2`` gate rejects. Second, volatility clustering
    is *not* the problem -- GARCH(1,1) work with no mean-level
    autocorrelation gives 0.044 against a 0.049 control, because variance
    clustering leaves the ordinates exchangeable in expectation. It is
    mean-level rate drift only.

    **The cost.** The filter attenuates what it flattens, so periodicity at
    very low frequency -- a handful of cycles across the whole window --
    gets harder to see. That is the same confound the ``peak_bin >= 2`` gate
    exists for, one bin further out. At moderate and high frequencies
    pre-whitening *gains* power, because removing the background is what lets
    a modest peak stand out. A corpus-sync artifact, the motivating
    hypothesis for the discovery-rate scan, has a period of order the sync
    interval and therefore a high bin, so it sits where this is strictly
    better. A tone is not cancelled at any amplitude tested (1.0 to 8.0 over
    unit white noise, all reported at the correct bin) -- see
    :data:`PREWHITEN_PEAK_CLIP` for why that needed guarding.

    Args:
        series: Uniformly-sampled observations (per-execution timings,
            per-interval discovery deltas, ...).
        sample_interval: Seconds between samples, used only to derive
            ``period_seconds``. Defaults to 1.0 (period reported in samples).
        min_samples: Shorter series are reported as not significant.
        alpha: Significance level for the Fisher g-test. With alpha=0.05,
            pure white noise is flagged at the nominal ~5% rate by design.
        prewhiten_series: Fit and divide out an AR(p) background before
            taking the periodogram, so the g-test's white-noise null holds.
            Defaults True; pass False only to reproduce a raw periodogram.

    Returns:
        A :class:`SpectralPeriodicity` with ``significant`` False for
        constant, too-short, or noise-dominated series.
    """
    n = len(series)
    if n < 2 or n < min_samples:
        return SpectralPeriodicity(None, 0.0, 0, None, n, False)
    if prewhiten_series:
        x, ar_order, ar_coeffs = prewhiten(series)
    else:
        x = np.asarray(series, dtype=np.float64)
        x = x - x.mean()
        ar_order, ar_coeffs = 0, ()
    # The filter drops `ar_order` samples, so the periodogram is over `m`
    # points and a peak at bin k is a period of m/k *samples*. The sample
    # spacing is unchanged, so the reported period stays in the caller's
    # units; n_samples keeps reporting what the caller passed.
    m = x.size
    if m < 2:
        return SpectralPeriodicity(None, 0.0, 0, None, n, False, 1.0, ar_order, ar_coeffs)
    power = np.abs(np.fft.rfft(x)) ** 2
    full = power[1 : (m + 1) // 2]
    if full.size == 0:
        return SpectralPeriodicity(None, 0.0, 0, None, n, False, 1.0, ar_order, ar_coeffs)
    peak_bin = int(np.argmax(full)) + 1
    peak_ord = float(full[peak_bin - 1])
    if peak_ord <= 0.0:
        return SpectralPeriodicity(None, 0.0, peak_bin, None, n, False, 1.0, ar_order, ar_coeffs)
    total_ord = float(power[1:].sum())
    g = peak_ord / total_ord
    p_value = fisher_g_pvalue(g, full.size, alpha)
    significant = p_value < alpha and peak_bin >= 2
    dominant_period = m / peak_bin if significant else None
    period_seconds = dominant_period * sample_interval if dominant_period is not None else None
    return SpectralPeriodicity(
        dominant_period,
        g,
        peak_bin,
        period_seconds,
        n,
        significant,
        p_value,
        ar_order,
        ar_coeffs,
    )


def harmonic_fraction(
    intervals: Sequence[float],
    expected_period: float,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    tolerance: float = DEFAULT_HARMONIC_TOLERANCE,
) -> dict[str, float]:
    """Bin intervals by proximity to the first ``n_harmonics`` of ``expected_period``.

    For each interval, the nearest harmonic ``h * expected_period`` is
    considered a candidate if the interval lies within ``±tolerance`` of it.
    Fractions are computed as the share of intervals assigned to each
    harmonic, plus a ``"total"`` entry for all intervals that matched any
    harmonic in range.

    Args:
        intervals: Observed inter-event intervals in the same units as
            ``expected_period``.
        expected_period: Candidate period from domain knowledge.
        n_harmonics: How many harmonics to test.  Defaults to 3.
        tolerance: Relative tolerance window around each harmonic
            (e.g. ``0.15`` accepts anything within 15% of the target).

    Returns:
        A mapping ``{ "1": fraction, ..., "total": total_fraction }``.
        Keys are stringified harmonic indices; values are in ``[0, 1]``.
    """
    if expected_period <= 0.0 or not intervals:
        return {str(i): 0.0 for i in range(1, n_harmonics + 1)} | {"total": 0.0}
    counts: dict[str, int] = {str(i): 0 for i in range(1, n_harmonics + 1)}
    matched = 0
    for dt in intervals:
        best = None
        best_rel = float("inf")
        for h in range(1, n_harmonics + 1):
            target = h * expected_period
            rel = abs(dt - target) / expected_period
            if rel < best_rel:
                best_rel = rel
                best = h
        if best is not None and best_rel <= tolerance:
            counts[str(best)] += 1
            matched += 1
    total = matched / len(intervals)
    return {str(i): counts[str(i)] / len(intervals) for i in range(1, n_harmonics + 1)} | {
        "total": total
    }


def locate_peak_period(
    intervals: Sequence[float],
    expected_period: float,
    bins: int = DEFAULT_PEAK_BINS,
    search_width: float = DEFAULT_SEARCH_WIDTH,
) -> tuple[float, float]:
    """Find the dominant period near ``expected_period`` via a fine histogram.

    Intervals are histogrammed into ``bins`` equal-width bins spanning
    ``[expected_period * (1 - search_width), expected_period * (1 + search_width)]``
    and the bin with the highest count is returned as the peak period.

    Args:
        intervals: Observed inter-event intervals.
        expected_period: Center of the search window.
        bins: Number of histogram bins.  Defaults to 200.
        search_width: Relative half-width of the search window.  Defaults to
            0.5, so the window spans 50% below to 50% above the prior.

    Returns:
        ``(peak_period, deviation_fraction)`` where ``deviation_fraction``
        is ``abs(peak_period - expected_period) / expected_period``.
    """
    if not intervals or expected_period <= 0.0 or bins <= 0:
        return expected_period, 0.0
    low = expected_period * (1.0 - search_width)
    high = expected_period * (1.0 + search_width)
    if high <= low:
        return expected_period, 0.0
    counts, edges = np.histogram(list(intervals), bins=bins, range=(low, high))
    peak_idx = int(np.argmax(counts))
    peak_period = float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)
    deviation = abs(peak_period - expected_period) / expected_period
    return peak_period, deviation


def classify_periodicity(harmonic_total_fraction: float) -> Literal["periodic", "weak", "none"]:
    """Classify a harmonic-binned periodicity signal.

    Args:
        harmonic_total_fraction: Share of intervals that fell on any tested
            harmonic, as returned by :func:`harmonic_fraction`.

    Returns:
        ``"periodic"`` when the fraction is at least
        :data:`HARMONIC_PERIODIC_THRESH`, ``"weak"`` when it is at least
        :data:`HARMONIC_WEAK_THRESH`, or ``"none"`` otherwise.
    """
    if harmonic_total_fraction >= HARMONIC_PERIODIC_THRESH:
        return "periodic"
    if harmonic_total_fraction >= HARMONIC_WEAK_THRESH:
        return "weak"
    return "none"
