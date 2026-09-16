"""Tests for core/rng_health.py — the quick startup PRNG sanity check."""

from __future__ import annotations

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
