"""Regression tests for Fisher's g-test null in ``core/periodicity.py`` (P0-T3).

``detect_periodicity`` scores the largest periodogram ordinate against a null
of i.i.d. exponential ordinates -- Gaussian white noise, i.e. a *constant*
rate. ``services/report.py`` applies it to the discovery-rate series, which is
the one ``coverage_regime.py``, ``critical_slowing.py`` and ``garch.py`` all
exist on the premise is non-stationary. Measured false-positive rate on a
Poisson series with a slowly drifting rate: 0.176 at n=128, 0.378 at n=256 and
0.540 at n=512, against a nominal 0.05 -- and the report's wording sends the
reader after a corpus-sync artifact that is not there.

Pinned here:

1. The drifting-rate null no longer fires at anything like the old rate.
2. White noise stays at the nominal rate, so the fix did not buy its
   calibration by making the test blind.
3. **A genuine tone is not cancelled.** This is the trap: a periodic
   component is strongly autocorrelated, so an AR model fitted to the raw
   series models the tone and the filter removes it. The first version of
   this fix reported a bin-64 sinusoid at bin 27.
4. Pre-whitening does not move a peak's frequency, only its amplitude.
5. The two falsified suspicions, so they are not re-proposed: GARCH-style
   volatility clustering never inflated this test, and pure 1/f was already
   handled by the ``peak_bin >= 2`` gate.

Replicate counts are kept low enough for a fast suite, so the thresholds are
loose; the tight numbers live in the ``detect_periodicity`` docstring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.periodicity import (
    PREWHITEN_MAX_ORDER,
    background_autocovariance,
    detect_periodicity,
    fit_ar_yule_walker,
    prewhiten,
)


def _cox(rng: np.random.Generator, n: int, tau: float = 40.0, base: float = 3.0):
    """Poisson counts whose rate is itself a slowly drifting OU process."""
    a = math.exp(-1.0 / tau)
    z = 0.0
    out = np.empty(n)
    for i in range(n):
        z = a * z + math.sqrt(1.0 - a * a) * rng.standard_normal()
        out[i] = rng.poisson(max(0.05, base * math.exp(0.6 * z)))
    return out


def _ar1(rng: np.random.Generator, n: int, phi: float):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _pink(rng: np.random.Generator, n: int, beta: float):
    f = np.fft.rfftfreq(n)[1:]
    phase = rng.uniform(0.0, 2.0 * math.pi, f.size)
    spec = np.concatenate(([0.0], f ** (-beta / 2.0) * np.exp(1j * phase)))
    return np.fft.irfft(spec, n=n)


def _rate(series_fn, reps: int, **kw) -> float:
    hits = sum(
        1 for _ in range(reps) if detect_periodicity(series_fn().tolist(), min_samples=50, **kw).significant
    )
    return hits / reps


# --- 1. the defect ---------------------------------------------------------


@pytest.mark.parametrize("n", [256, 512])
def test_drifting_rate_null_is_not_flagged_wholesale(n: int) -> None:
    """Was 0.378 (n=256) and 0.540 (n=512) against a nominal 0.05."""
    rng = np.random.default_rng(20260913 + n)
    rate = _rate(lambda: _cox(rng, n), 150)
    assert rate < 0.20, f"drifting-rate false-positive rate {rate:.3f} at n={n}"


@pytest.mark.parametrize("phi", [0.7, 0.9, -0.6])
def test_autocorrelated_nulls_are_not_flagged_wholesale(phi: float) -> None:
    """AR(1) nulls ran 0.85-0.98 before the fix, in both signs of phi."""
    rng = np.random.default_rng(hash(("ar", phi)) % (2**32))
    rate = _rate(lambda: _ar1(rng, 512, phi), 150)
    assert rate < 0.20, f"AR(1) phi={phi} false-positive rate {rate:.3f}"


def test_raw_periodogram_still_reproduces_the_defect() -> None:
    """prewhiten_series=False must still be the old behaviour.

    Without this the fix is unfalsifiable from inside the suite: nothing else
    here can tell 'the null was repaired' from 'the test went blind'.
    """
    rng = np.random.default_rng(4242)
    raw = _rate(lambda: _ar1(rng, 512, 0.7), 120, prewhiten_series=False)
    assert raw > 0.5, f"expected the documented inflation, saw {raw:.3f}"


# --- 2. calibration preserved ---------------------------------------------


@pytest.mark.parametrize("n", [128, 256, 512])
def test_white_noise_stays_at_the_nominal_rate(n: int) -> None:
    rng = np.random.default_rng(777 + n)
    rate = _rate(lambda: rng.standard_normal(n), 250)
    assert 0.0 <= rate < 0.13, f"white-noise rate {rate:.3f} at n={n}"


# --- 3. the trap: a real tone must survive --------------------------------


@pytest.mark.parametrize("amp", [1.0, 2.0, 4.0, 8.0])
def test_a_real_tone_is_not_cancelled_by_its_own_background_fit(amp: float) -> None:
    """The first version of this fix reported a bin-64 tone at bin 27.

    A periodic component is strongly autocorrelated, so an AR fit to the raw
    series models the tone. The background fit therefore has to be peak-robust
    -- see PREWHITEN_PEAK_CLIP.
    """
    rng = np.random.default_rng(int(amp * 1000) + 5)
    n, bin_k = 512, 64
    t = np.arange(n)
    detected = 0
    for _ in range(25):
        x = rng.standard_normal(n) + amp * np.sin(2.0 * math.pi * bin_k * t / n)
        res = detect_periodicity(x.tolist(), min_samples=50)
        if not res.significant:
            continue
        # Map the bin back through the shortened series: the filter drops
        # ar_order samples, so bin k of m points is frequency k/m.
        m = res.n_samples - res.ar_order
        if abs(res.peak_bin * n / m - bin_k) < 2.0:
            detected += 1
    assert detected >= 23, f"amp={amp}: recovered the tone {detected}/25 times"


def test_prewhitening_does_not_shift_a_peaks_frequency() -> None:
    """The filter changes amplitudes, not frequencies."""
    rng = np.random.default_rng(99)
    n = 512
    t = np.arange(n)
    for bin_k in (16, 64, 128):
        x = _ar1(rng, n, 0.7) + 6.0 * np.sin(2.0 * math.pi * bin_k * t / n)
        res = detect_periodicity(x.tolist(), min_samples=50)
        assert res.significant
        m = res.n_samples - res.ar_order
        assert abs(res.peak_bin * n / m - bin_k) < 2.0, (
            f"bin {bin_k} reported at {res.peak_bin * n / m:.1f}"
        )


# --- 4. the two falsified suspicions --------------------------------------


def test_volatility_clustering_never_inflated_this_test() -> None:
    """GARCH-style clustering leaves the ordinates exchangeable.

    Recorded as a test because it is the plausible-and-wrong hypothesis: the
    tree has a GARCH module, so 'the discovery series is volatility-clustered'
    looks like it should explain the false positives. It does not -- measured
    0.044 against a 0.049 control. Only mean-level rate drift does.
    """
    rng = np.random.default_rng(31337)

    def garch(n: int, omega=0.05, alpha=0.15, beta=0.80):
        s2 = omega / (1.0 - alpha - beta)
        x = np.empty(n)
        e = rng.standard_normal(n)
        for i in range(n):
            x[i] = math.sqrt(s2) * e[i]
            s2 = omega + alpha * x[i] ** 2 + beta * s2
        return x

    raw = _rate(lambda: garch(512), 150, prewhiten_series=False)
    assert raw < 0.13, f"raw g-test on clustered variance: {raw:.3f}"


@pytest.mark.parametrize("beta", [1.0, 2.0])
def test_pure_power_law_was_already_handled_by_the_bin_gate(beta: float) -> None:
    """1/f peaks collapse into bin 1, which ``peak_bin >= 2`` rejects."""
    rng = np.random.default_rng(int(beta * 10) + 88)
    raw = _rate(lambda: _pink(rng, 512, beta), 120, prewhiten_series=False)
    assert raw < 0.05, f"raw rate on 1/f^{beta}: {raw:.3f}"


# --- 5. the pieces --------------------------------------------------------


def test_white_noise_mostly_selects_order_zero() -> None:
    """AIC on a finite white sample picks a spurious order sometimes.

    Asserted distributionally rather than per-sample: a strict ``order == 0``
    fails a few percent of the time, and the property that matters -- the
    false-positive rate staying nominal -- is covered above. Order 0 must
    return the centred series untouched, which is asserted where it occurs.
    """
    rng = np.random.default_rng(5)
    orders = []
    for _ in range(200):
        x = rng.standard_normal(512)
        res, order, coeffs = prewhiten(x.tolist())
        orders.append(order)
        assert len(coeffs) == order
        assert res.size == x.size - order
        if order == 0:
            np.testing.assert_allclose(res, x - x.mean(), atol=1e-12)
    zero_frac = orders.count(0) / len(orders)
    assert zero_frac > 0.5, f"order 0 chosen only {zero_frac:.2f} of the time"
    assert float(np.mean(orders)) < 1.5, f"mean order {np.mean(orders):.2f} on white noise"


def test_strongly_autocorrelated_input_selects_a_positive_order() -> None:
    rng = np.random.default_rng(6)
    res, order, coeffs = prewhiten(np.cumsum(rng.standard_normal(512)).tolist())
    assert 1 <= order <= PREWHITEN_MAX_ORDER
    assert len(coeffs) == order
    assert res.size == 512 - order


def test_short_and_degenerate_inputs_are_returned_unchanged() -> None:
    for series in ([0.0] * 64, [1.0] * 8, [], [3.0]):
        res, order, coeffs = prewhiten(series)
        assert order == 0 and coeffs == ()
        assert res.size == len(series)


def test_background_autocovariance_clips_a_tone_out_of_the_fit() -> None:
    """The whole point of the clip: a tone must not reach the background."""
    rng = np.random.default_rng(11)
    n = 512
    t = np.arange(n)
    noise = rng.standard_normal(n)
    tone = noise + 8.0 * np.sin(2.0 * math.pi * 64.0 * t / n)
    c_noise = background_autocovariance(noise - noise.mean())
    c_tone = background_autocovariance(tone - tone.mean())
    # Lag-0 background power should be close despite an 8x-amplitude tone.
    assert c_tone[0] < 3.0 * c_noise[0], (
        f"tone leaked into the background fit: {c_tone[0]:.3f} vs {c_noise[0]:.3f}"
    )


def test_yule_walker_rejects_a_singular_system() -> None:
    assert fit_ar_yule_walker(np.zeros(4), 2) is None
    assert fit_ar_yule_walker(np.array([1.0]), 3) is None
    assert fit_ar_yule_walker(np.array([1.0, 0.5, 0.2]), 0) is None


def test_result_reports_the_background_it_removed() -> None:
    rng = np.random.default_rng(12)
    res = detect_periodicity(_ar1(rng, 512, 0.8).tolist(), min_samples=50)
    assert res.ar_order >= 1
    assert len(res.ar_coeffs) == res.ar_order
    assert res.n_samples == 512  # reports what the caller passed, not the residual
