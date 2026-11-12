"""Tests for core/rng_health.py — the quick startup PRNG sanity check."""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core import rand_pool as rp
from fuzzer_tool.core.rand_pool import _POOL_ENTRIES, RandPool
from fuzzer_tool.core.rng_health import quick_health_check, structure_check


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
    f.seed = None
    f._report_rng_health()
    out = capsys.readouterr().out
    assert "[*] RNG health:" in out


def test_reports_rng_health_warns_on_bad_rng(capsys):
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = object.__new__(Fuzzer)
    f._rng = _ConstRng()
    f.seed = None
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


# ── structure_check: exact checks the byte battery cannot see ──────────

_STRUCT_SEED = 42
_LAG = 3


class _Replay(RandPool):
    """Every refill replays the same batch (generator reset per refill)."""

    __slots__ = ()

    def _generate_pool(self):
        self._rng = np.random.default_rng(7)
        super()._generate_pool()


class _HighStuck(RandPool):
    """Upper 24 bits of every word are zero; the low byte stays uniform."""

    __slots__ = ()

    def _generate_pool(self):
        super()._generate_pool()
        self._pool &= 0xFF


class _LagCopy(RandPool):
    """Every third word copies the word _LAG draws back."""

    __slots__ = ()

    def _generate_pool(self):
        super()._generate_pool()
        dst = self._pool[_LAG::_LAG]
        dst[:] = self._pool[0:-_LAG:_LAG][: len(dst)]
        np.copyto(self._m256, self._pool, casting="unsafe")


def _tagged_default(seed):
    """The default pool as Fuzzer.__init__ seeds it."""
    rp.reset_default_rand_pool(seed)
    return rp.get_default_rand_pool()


@pytest.fixture
def _restore_default():
    yield
    rp.reset_default_rand_pool()


@pytest.mark.usefixtures("_restore_default")
@pytest.mark.parametrize("seed", [None, 0, _STRUCT_SEED, -5])
def test_structure_healthy_pool_clean(seed):
    pool = RandPool(seed=seed)
    assert structure_check(pool, seed=seed, default_pool=_tagged_default(seed)) == []


def test_structure_flags_refill_replay():
    """The byte battery passes this pool (p=0.84 measured)."""
    assert quick_health_check(_Replay(1)).ok
    warnings = structure_check(_Replay(1))
    assert any("replays" in w for w in warnings)


def test_structure_flags_stuck_high_bits():
    """Low byte uniform, upper 24 bits zero: only a byte-lane test sees it."""
    assert quick_health_check(_HighStuck(1)).ok
    warnings = structure_check(_HighStuck(1))
    assert any("byte lane" in w for w in warnings)


def test_structure_flags_lag_repeats():
    assert quick_health_check(_LagCopy(1)).ok
    warnings = structure_check(_LagCopy(1))
    assert any(f"lag {_LAG}" in w for w in warnings)


def test_structure_flags_seed_wired_to_constant():
    """--seed given, but the pool runs the unseeded fallback stream."""
    warnings = structure_check(RandPool(seed=None), seed=_STRUCT_SEED)
    assert any("seed" in w for w in warnings)


def test_structure_seed_check_survives_consumed_draws():
    """Adversarial: the live pool is already 10k draws into the stream."""
    pool = RandPool(seed=None)
    pool.random_list(10_000)
    assert any("seed" in w for w in structure_check(pool, seed=_STRUCT_SEED))


def test_structure_no_seed_no_seed_warning():
    """Falsification: unseeded runs use the fallback stream by design."""
    assert structure_check(RandPool(seed=None), seed=None) == []


def test_structure_flags_default_mirroring_main():
    """The pre-2026-11-08 bug: default pool = RandPool(seed) at an offset."""
    main = RandPool(seed=_STRUCT_SEED)
    mirror = RandPool(seed=_STRUCT_SEED)
    mirror.random_list(5_000)
    warnings = structure_check(main, seed=_STRUCT_SEED, default_pool=mirror)
    assert any("default pool" in w for w in warnings)


def test_structure_control_independent_default_clean():
    """Control (rule 46): an unrelated default pool must not match."""
    main = RandPool(seed=_STRUCT_SEED)
    other = RandPool(seed=_STRUCT_SEED + 1)
    assert structure_check(main, seed=_STRUCT_SEED, default_pool=other) == []


def test_structure_does_not_move_live_streams():
    """Checks draw from clones: the campaign streams are untouched."""
    main, default = RandPool(seed=3), RandPool(seed=4)
    main.random()
    want = (RandPool(seed=3).random_list(2)[1], RandPool(seed=4).random())
    structure_check(main, seed=3, default_pool=default)
    assert (main.random(), default.random()) == want


def test_structure_false_alarm_rate():
    """Healthy pools: exact checks ~never fire, lane chisq at 1e-4 each."""
    fired = [s for s in range(200) if structure_check(RandPool(seed=s), seed=s)]
    assert fired == []


def test_reports_structure_warning_on_banner(capsys):
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = object.__new__(Fuzzer)
    f._rng = RandPool(seed=None)
    f.seed = _STRUCT_SEED
    f._report_rng_health()
    out = capsys.readouterr().out
    assert "PRNG structure" in out


@pytest.mark.usefixtures("_restore_default")
def test_stall_reseed_runs_structure_check(monkeypatch):
    """Wiring: every stall reseed rechecks the new streams."""
    from fuzzer_tool.services import fuzzer as fz

    seen = []
    monkeypatch.setattr(fz, "structure_check", lambda pool, **kw: seen.append(kw["seed"]) or [])
    f = object.__new__(fz.Fuzzer)
    f.seed = _STRUCT_SEED
    f._stall_reseed_count = 0
    f._rng = RandPool(seed=_STRUCT_SEED)
    new_seed = f._reseed_after_stall()
    assert seen == [new_seed]


def test_report_line_suspect_on_structure_warning():
    from unittest.mock import MagicMock

    from fuzzer_tool.services.report import _rng_health_line

    f = MagicMock()
    f._rng = RandPool(seed=None)
    f.seed = _STRUCT_SEED
    assert "SUSPECT" in _rng_health_line(f)


def test_structure_replay_period_is_pool_size():
    """The replay probe's period tracks the pool's refill size."""
    from fuzzer_tool.core import rng_health

    assert rng_health._REPLAY_PERIOD == _POOL_ENTRIES
