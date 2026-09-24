"""Tests for PLLMonitor (core/analyzers/analyzer_pll.py)."""

import math

import pytest

from fuzzer_tool.core.analyzers.analyzer_pll import (
    PENDING_CAP,
    WARMUP,
    PLLMonitor,
    Series,
    Stall,
)
from fuzzer_tool.core.rand_pool import RandPool

PERIOD = 20.0


def _sine(n: int, period: float = PERIOD, start: int = 0) -> list[float]:
    """Clean sinusoid on a large DC offset (exec times are never zero-mean)."""
    return [100.0 + 5.0 * math.sin(2.0 * math.pi * (start + t) / period) for t in range(n)]


def _noise(n: int, seed: int) -> list[float]:
    return RandPool(seed=seed).gauss_list(100.0, 5.0, n)


def _feed(m: PLLMonitor, xs, stall=Stall.NO, series=Series.EXEC_TIME, exec_count=0):
    for x in xs:
        m.push(series, x)
    return m.flush(exec_count, stall)


class TestBootstrap:
    def test_waits_for_warmup(self):
        m = PLLMonitor()
        _feed(m, _sine(WARMUP - 1))
        assert m.state(Series.EXEC_TIME) is None
        assert m.bootstrap_period(Series.EXEC_TIME) is None

    def test_bootstraps_from_detected_period(self):
        m = PLLMonitor()
        _feed(m, _sine(WARMUP))
        period = m.bootstrap_period(Series.EXEC_TIME)
        assert period is not None
        assert period == pytest.approx(PERIOD, rel=0.1)
        assert m.state(Series.EXEC_TIME) is not None

    def test_noise_never_bootstraps(self):
        # Falsification: white noise has no period, so no loop is built and
        # the miss counter proves the bootstrap was attempted.
        m = PLLMonitor()
        _feed(m, _noise(WARMUP * 3, seed=7))
        assert m.bootstrap_period(Series.EXEC_TIME) is None
        assert m.summary(Series.EXEC_TIME)["misses"] >= 3

    def test_series_are_independent(self):
        m = PLLMonitor()
        _feed(m, _sine(WARMUP), series=Series.DISCOVERY)
        assert m.bootstrap_period(Series.DISCOVERY) is not None
        assert m.bootstrap_period(Series.EXEC_TIME) is None


class TestTransitions:
    def test_sine_locks_and_records_transition(self):
        m = PLLMonitor()
        out = _feed(m, _sine(WARMUP * 4), exec_count=123)
        assert m.state(Series.EXEC_TIME).locked
        locks = [t for t in out if t.locked]
        assert locks
        assert locks[0].series is Series.EXEC_TIME
        assert locks[0].exec_count == 123
        assert m.state(Series.EXEC_TIME).period == pytest.approx(PERIOD, rel=0.05)

    def test_switch_to_noise_unlocks(self):
        m = PLLMonitor()
        _feed(m, _sine(WARMUP * 4))
        out = _feed(m, [100.0] * (WARMUP * 2))
        assert not m.state(Series.EXEC_TIME).locked
        assert any(not t.locked for t in out)

    def test_transitions_bounded(self):
        m = PLLMonitor(max_transitions=2)
        for _ in range(5):
            _feed(m, _sine(WARMUP * 4))
            _feed(m, [100.0] * (WARMUP * 2))
        assert len(m.transitions) == 2

    def test_flush_is_idempotent_when_empty(self):
        m = PLLMonitor()
        _feed(m, _sine(WARMUP * 4))
        assert m.flush(0, Stall.NO) == []


class TestStallLift:
    def test_none_without_stalled_ticks(self):
        m = PLLMonitor()
        _feed(m, _sine(WARMUP * 4))
        assert m.stall_lift(Series.EXEC_TIME) is None

    def test_none_without_transitions(self):
        m = PLLMonitor()
        _feed(m, _noise(WARMUP * 2, seed=3), stall=Stall.YES)
        assert m.stall_lift(Series.EXEC_TIME) is None

    def test_lift_matches_hand_count(self):
        # Lock happens during the stalled batch; the quiet flat tail is not
        # stalled. lift = P(stall | transition) / P(stall), derived from the
        # summary counters rather than echoed.
        m = PLLMonitor()
        _feed(m, _sine(WARMUP * 4), stall=Stall.YES)
        _feed(m, _sine(WARMUP * 4, start=WARMUP * 4), stall=Stall.NO)
        s = m.summary(Series.EXEC_TIME)
        assert s["transitions"] >= 1
        expected = (s["transitions_stalled"] / s["transitions"]) / (s["ticks_stalled"] / s["ticks"])
        assert m.stall_lift(Series.EXEC_TIME) == pytest.approx(expected)
        assert m.stall_lift(Series.EXEC_TIME) > 1.0


class TestControl:
    def test_same_input_same_trace(self):
        # Rule 46 control: the monitor against itself.
        a, b = PLLMonitor(), PLLMonitor()
        xs = _sine(WARMUP * 3) + _noise(WARMUP, seed=11)
        ta, tb = _feed(a, xs), _feed(b, xs)
        assert ta == tb
        assert a.summary(Series.EXEC_TIME) == b.summary(Series.EXEC_TIME)

    def test_batching_does_not_change_trace(self):
        xs = _sine(WARMUP * 3)
        one = PLLMonitor()
        _feed(one, xs)
        many = PLLMonitor()
        for i in range(0, len(xs), 37):
            _feed(many, xs[i : i + 37])
        assert one.summary(Series.EXEC_TIME) == many.summary(Series.EXEC_TIME)


class TestAdversarial:
    def test_non_finite_samples_dropped(self):
        m = PLLMonitor()
        xs = _sine(WARMUP)
        xs[5] = float("nan")
        xs[9] = float("inf")
        _feed(m, xs)
        assert m.summary(Series.EXEC_TIME)["dropped"] == 2
        assert m.bootstrap_period(Series.EXEC_TIME) is None  # 2 short of warmup
        _feed(m, _sine(2, start=WARMUP))
        assert m.bootstrap_period(Series.EXEC_TIME) is not None

    def test_flat_series_never_bootstraps(self):
        m = PLLMonitor()
        _feed(m, [5.0] * (WARMUP * 2))
        assert m.bootstrap_period(Series.EXEC_TIME) is None

    def test_pending_cap_flushes_inline(self):
        # A campaign that never reaches a stats tick must not grow the
        # pending buffer without bound.
        m = PLLMonitor()
        for x in _sine(PENDING_CAP + 1):
            m.push(Series.EXEC_TIME, x)
        assert m.pending(Series.EXEC_TIME) <= 1
        assert m.bootstrap_period(Series.EXEC_TIME) is not None

    def test_period_two_alternation_rejected(self):
        # Nyquist: detect_periodicity cannot see period 2 and from_period
        # refuses it; the monitor must stay unbootstrapped, not raise.
        m = PLLMonitor()
        _feed(m, [100.0 + (5.0 if t % 2 else -5.0) for t in range(WARMUP * 2)])
        assert m.bootstrap_period(Series.EXEC_TIME) is None

    def test_invalid_warmup_rejected(self):
        with pytest.raises(ValueError):
            PLLMonitor(warmup=8)


class _FakeFuzzer:
    """Minimal stand-in carrying what the stats/report consumers read."""

    def __init__(self, pll):
        from array import array

        self._pll = pll
        self._pll_disc_idx = 0
        self._discovery_edges = array("Q")
        self._stall_recovery_active = False
        self.exec_count = 0


class TestConsumers:
    def test_stats_drains_discovery_deltas_once(self):
        from fuzzer_tool.services.stats import _pll_str

        f = _FakeFuzzer(PLLMonitor())
        cum = 0
        for x in _sine(WARMUP * 2):
            cum += int(x)
            f._discovery_edges.append(cum)
        _pll_str(f)
        first = f._pll.summary(Series.DISCOVERY)["ticks"]
        assert f._pll.bootstrap_period(Series.DISCOVERY) is not None
        _pll_str(f)  # no new snapshots: nothing re-fed
        assert f._pll.summary(Series.DISCOVERY)["ticks"] == first

    def test_stats_reports_tracked_period(self):
        from fuzzer_tool.services.stats import _pll_str

        f = _FakeFuzzer(PLLMonitor())
        for x in _sine(WARMUP * 4):
            f._pll.push(Series.EXEC_TIME, x)
        out = _pll_str(f)
        assert out.startswith(" | pll: t:")
        assert "L" in out

    def test_stats_stall_flag_reaches_monitor(self):
        from fuzzer_tool.services.stats import _pll_str

        f = _FakeFuzzer(PLLMonitor())
        f._stall_recovery_active = True
        for x in _sine(WARMUP * 2):
            f._pll.push(Series.EXEC_TIME, x)
        _pll_str(f)
        s = f._pll.summary(Series.EXEC_TIME)
        assert s["ticks_stalled"] == s["ticks"] > 0

    def test_stats_empty_when_off(self):
        from unittest.mock import MagicMock

        from fuzzer_tool.services.stats import _pll_str

        assert _pll_str(MagicMock()) == ""

    def test_report_lines(self):
        from fuzzer_tool.services.report import _pll_lines

        f = _FakeFuzzer(PLLMonitor())
        for x in _sine(WARMUP * 4):
            f._pll.push(Series.EXEC_TIME, x)
        f._pll.flush(0, Stall.NO)
        lines = _pll_lines(f)
        assert any("Exec time PLL" in ln and "locked" in ln for ln in lines)
        assert any("Discovery PLL" in ln and "no period" in ln for ln in lines)

    def test_report_empty_when_off(self):
        from unittest.mock import MagicMock

        from fuzzer_tool.services.report import _pll_lines

        assert _pll_lines(MagicMock()) == []
