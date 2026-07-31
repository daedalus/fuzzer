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
   A plain rfft magnitude spectrum (DC excluded) answers "is there a
   dominant non-DC frequency, and what period does it correspond to?" —
   e.g. attributing a periodic overhead in per-execution timings or a
   periodic component in the coverage discovery-rate series to a specific
   cadence, something Allan-variance (noise-type) and dispersion-index
   (burstiness) diagnostics are not designed to catch.

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
    """
    if not data or len(data) < min_len:
        return None
    n = len(data)
    limit = min(max_lag if max_lag is not None else n // 3, DEFAULT_MAX_LAG)
    if limit < 2:
        return None

    x = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    x = (x - x.mean()) * np.hanning(n)
    power = np.fft.rfft(x)
    power = power * power.conj()
    ac = np.fft.irfft(power, n=n)
    total = ac[0]
    if total <= 0.0:
        return None
    ac = ac / total

    scanned = ac[1 : limit + 1]
    if scanned.size == 0:
        return None
    # Noise-floor gate: a fixed fraction of the lag-0 autocorrelation plus a
    # multiple-comparisons-aware sigma bound (white-noise ac[k] ~ N(0, 1/n)).
    threshold = max(min_rel_peak, SIGMA_CUTOFF / math.sqrt(n))
    floor = peak_to_median * float(np.median(scanned))
    for k in range(1, limit + 1):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] >= threshold and ac[k] >= floor:
            return k
    return None


@dataclass(frozen=True)
class SpectralPeriodicity:
    """Result of a spectral scan over a real-valued time series."""

    dominant_period: float | None  # series samples per cycle; None if not significant
    peak_strength: float  # peak bin magnitude / median of non-DC bins
    peak_bin: int  # index of the dominant non-DC frequency bin (0 if none)
    period_seconds: float | None  # dominant_period * sample_interval
    n_samples: int
    significant: bool


def detect_periodicity(
    series: Sequence[float],
    sample_interval: float = 1.0,
    min_samples: int = 64,
    peak_to_median: float = 2.0,
) -> SpectralPeriodicity:
    """Detect a dominant non-DC periodic component in a real-valued series.

    Takes the rfft magnitude spectrum of the mean-subtracted series and
    searches for the strongest non-DC bin, scoring it against the median of
    the remaining bins. The DC (mean) bin is excluded by construction —
    the question is purely "is there an oscillation at a specific frequency".

    Args:
        series: Uniformly-sampled observations (per-execution timings,
            per-interval discovery deltas, ...).
        sample_interval: Seconds between samples, used only to derive
            ``period_seconds``. Defaults to 1.0 (period reported in samples).
        min_samples: Shorter series are reported as not significant.
        peak_to_median: The dominant bin must exceed the median of the
            non-DC bins by at least this factor to be significant.

    Returns:
        A :class:`SpectralPeriodicity` with ``significant`` False for
        constant, too-short, or noise-dominated series.
    """
    n = len(series)
    if n < 2 or n < min_samples:
        return SpectralPeriodicity(None, 0.0, 0, None, n, False)
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    mag = np.abs(np.fft.rfft(x))
    bins = mag[1:]
    if bins.size == 0:
        return SpectralPeriodicity(None, 0.0, 0, None, n, False)
    peak_bin = int(np.argmax(bins)) + 1
    peak_mag = float(bins[peak_bin - 1])
    if peak_mag <= 0.0:
        return SpectralPeriodicity(None, 0.0, peak_bin, None, n, False)
    median_mag = float(np.median(bins))
    peak_strength = peak_mag / median_mag if median_mag > 0.0 else math.inf
    significant = peak_bin >= 2 and peak_strength >= peak_to_median
    dominant_period = n / peak_bin if significant else None
    period_seconds = dominant_period * sample_interval if dominant_period is not None else None
    return SpectralPeriodicity(
        dominant_period, peak_strength, peak_bin, period_seconds, n, significant
    )
