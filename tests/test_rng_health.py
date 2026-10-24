"""Tests for core/rng_health.py — the quick startup PRNG sanity check."""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.rng_health import quick_health_check


class _ConstRng:
    """Degenerate RNG: every draw is the same byte."""

    def randint_list(self, a, b, count):
        return [a] * count


class _StickyRng:
    """Marginally-uniform but highly autocorrelated stream.

    Alternates two extreme values, so monobit/byte_chisq alone would not
    catch it -- exercises why repeat_test is included.
    """

    def randint_list(self, a, b, count):
        return [0 if i % 2 == 0 else 255 for i in range(count)]


class _LowEntropyRng:
    """Only ever emits 2 distinct values, unevenly split -- fails byte_chisq."""

    def randint_list(self, a, b, count):
        return [1] * (count * 9 // 10) + [2] * (count - count * 9 // 10)


def test_real_randpool_passes():
    pool = RandPool(seed=12345)
    result = quick_health_check(pool)
    assert result.ok
    assert not result.warnings
    assert result.combined_p >= 0.01


def test_real_randpool_passes_across_several_seeds():
    # A single seed passing could be luck; check a handful stay well-behaved.
    for seed in range(10):
        result = quick_health_check(RandPool(seed=seed))
        assert result.ok, f"seed={seed} flagged: {result.summary()}"


def test_constant_stream_flagged():
    result = quick_health_check(_ConstRng())
    assert not result.ok
    assert any("constant" in w for w in result.warnings)


def test_sticky_stream_flagged_by_repeat_test():
    result = quick_health_check(_StickyRng())
    assert not result.ok
    # monobit alone would not see this -- confirm repeat/runs did the work.
    assert result.pvalues["repeat"] < 0.01 or result.pvalues["runs"] < 0.01


def test_low_entropy_stream_flagged_by_byte_chisq():
    result = quick_health_check(_LowEntropyRng())
    assert not result.ok
    assert result.pvalues["byte_chisq"] < 0.01


def test_low_entropy_stream_flagged_by_entropy_test():
    result = quick_health_check(_LowEntropyRng())
    assert not result.ok
    assert result.pvalues["entropy"] < 0.01


def test_sample_size_is_small_and_configurable():
    pool = RandPool(seed=1)
    result = quick_health_check(pool, n_bytes=512)
    assert result.n_bytes == 512


def test_never_raises_on_broken_rng():
    class _BrokenRng:
        def randint_list(self, a, b, count):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        # quick_health_check itself does not swallow errors -- that is the
        # caller's job (Fuzzer._report_rng_health wraps it); confirm the
        # exception is not silently turned into a bogus "pass" result here.
        quick_health_check(_BrokenRng())


def test_summary_format_ok_and_suspect():
    ok_result = quick_health_check(RandPool(seed=7))
    assert ok_result.summary().startswith("OK")
    assert "entropy=" in ok_result.summary()

    bad_result = quick_health_check(_ConstRng())
    assert bad_result.summary().startswith("SUSPECT")
    assert "entropy=" in bad_result.summary()


def test_reports_rng_health_prints_banner_line(capsys):
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = object.__new__(Fuzzer)
    f._rng = RandPool(seed=99)
    f._report_rng_health()
    out = capsys.readouterr().out
    assert "[*] RNG health:" in out


def test_reports_rng_health_warns_on_bad_rng(capsys):
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = object.__new__(Fuzzer)
    f._rng = _ConstRng()
    f._report_rng_health()
    out = capsys.readouterr().out
    assert "[!] WARNING:" in out
    assert "PRNG health check" in out


def test_reports_rng_health_never_raises_on_check_error(capsys):
    from fuzzer_tool.services.fuzzer import Fuzzer

    class _BrokenRng:
        def randint_list(self, a, b, count):
            raise RuntimeError("boom")

    f = object.__new__(Fuzzer)
    f._rng = _BrokenRng()
    f._report_rng_health()  # must not raise


# ── calibration: healthy streams must fail at the nominal rate ─────────

_NULL_STREAMS = 1000
_ALPHA = 0.01
# Binomial(1000, 0.01) exceeds 20 with p ~ 0.001; the double-counted
# entropy p-value gave ~37.
_MAX_FALSE_ALARMS = 20


def test_regression_health_false_alarm_rate():
    fails = sum(not quick_health_check(RandPool(seed=s)).ok for s in range(_NULL_STREAMS))
    assert fails <= _MAX_FALSE_ALARMS, f"{fails}/{_NULL_STREAMS} healthy streams flagged"


def test_regression_entropy_not_combined():
    # entropy is a G-test on the same histogram as byte_chisq (corr ~0.99):
    # Fisher must see it once, not twice.
    from fuzzer_tool.core.randomness import fishers_method

    r = quick_health_check(RandPool(seed=3))
    independent = [p for name, p in r.pvalues.items() if name != "entropy"]
    assert r.combined_p == pytest.approx(fishers_method(independent))
    assert "entropy" in r.pvalues  # still reported


class _SkewedRng:
    """Uniform over 0..239 only: no constant, no stickiness, just 16 missing values."""

    def __init__(self):
        self._rng = np.random.default_rng(5)

    def randint_list(self, a, b, count):
        return self._rng.integers(0, 240, count).tolist()


def test_skewed_stream_flagged_without_entropy_in_combination():
    r = quick_health_check(_SkewedRng())
    assert not r.ok
    assert r.pvalues["byte_chisq"] < _ALPHA


def test_regression_entropy_test_calibrated():
    # Raw G overstates chi2 at 16 counts/cell; Williams' q fixes it.
    from fuzzer_tool.core.randomness import byte_chisq, ks_uniform, shannon_entropy_test

    rng = np.random.default_rng(11)
    streams = [rng.integers(0, 256, 4096, dtype=np.uint8).tobytes() for _ in range(2000)]

    # Control (Rule 46): the oracle must pass a known-calibrated test first.
    assert ks_uniform([byte_chisq(d) for d in streams]) > 0.001
    assert ks_uniform([shannon_entropy_test(d) for d in streams]) > 0.001
