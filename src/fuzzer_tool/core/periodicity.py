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
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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


def detect_periodicity(
    series: Sequence[float],
    sample_interval: float = 1.0,
    min_samples: int = 64,
    alpha: float = 0.05,
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

    Args:
        series: Uniformly-sampled observations (per-execution timings,
            per-interval discovery deltas, ...).
        sample_interval: Seconds between samples, used only to derive
            ``period_seconds``. Defaults to 1.0 (period reported in samples).
        min_samples: Shorter series are reported as not significant.
        alpha: Significance level for the Fisher g-test. With alpha=0.05,
            pure white noise is flagged at the nominal ~5% rate by design.

    Returns:
        A :class:`SpectralPeriodicity` with ``significant`` False for
        constant, too-short, or noise-dominated series.
    """
    n = len(series)
    if n < 2 or n < min_samples:
        return SpectralPeriodicity(None, 0.0, 0, None, n, False)
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    full = power[1 : (n + 1) // 2]
    if full.size == 0:
        return SpectralPeriodicity(None, 0.0, 0, None, n, False)
    peak_bin = int(np.argmax(full)) + 1
    peak_ord = float(full[peak_bin - 1])
    if peak_ord <= 0.0:
        return SpectralPeriodicity(None, 0.0, peak_bin, None, n, False)
    total_ord = float(power[1:].sum())
    g = peak_ord / total_ord
    p_value = fisher_g_pvalue(g, full.size, alpha)
    significant = p_value < alpha and peak_bin >= 2
    dominant_period = n / peak_bin if significant else None
    period_seconds = dominant_period * sample_interval if dominant_period is not None else None
    return SpectralPeriodicity(
        dominant_period, g, peak_bin, period_seconds, n, significant, p_value
    )
