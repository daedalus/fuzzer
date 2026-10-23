"""Per-exec target timer and per-cycle fuzzer timer, averaged on the stats line."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fuzzer_tool.core.exec_timer import NS_PER_MS, ExecTimer
from fuzzer_tool.services.stats import StatsReporter
from tests import test_regression_end_of_run_persistence as end_of_run


class TestExecTimer:
    def test_window_averages_then_resets(self):
        t = ExecTimer()
        for ns in (1 * NS_PER_MS, 3 * NS_PER_MS):
            t.add_target(ns)
        t.add_cycle(10 * NS_PER_MS)

        assert t.drain() == (2.0, 10.0)
        assert t.drain() == (None, None)

    def test_empty_window_has_no_average(self):
        """Adversarial: no samples must not divide by zero."""
        t = ExecTimer()
        t.add_cycle(5 * NS_PER_MS)
        assert t.drain() == (None, 5.0)

    def test_huge_samples_stay_exact(self):
        """Adversarial: hour-long execs (ints, no float drift in the sum)."""
        t = ExecTimer()
        hour_ns = 3600 * 1000 * NS_PER_MS
        for _ in range(3):
            t.add_target(hour_ns)
        assert t.drain()[0] == hour_ns / NS_PER_MS

    def test_lifetime_survives_drain(self):
        t = ExecTimer()
        t.add_target(NS_PER_MS)
        t.add_cycle(4 * NS_PER_MS)
        t.drain()
        t.add_target(3 * NS_PER_MS)
        t.drain()
        assert t.lifetime() == (2.0, 4.0)

    def test_lifetime_empty(self):
        """Adversarial: no samples ever -> no average."""
        assert ExecTimer().lifetime() == (None, None)


class TestStatsLine:
    def _str(self, timer):
        return StatsReporter._print_stats_timer_str(None, SimpleNamespace(_exec_timer=timer))

    def test_reports_both_averages(self):
        t = ExecTimer()
        t.add_target(NS_PER_MS // 2)
        t.add_cycle(2 * NS_PER_MS)
        assert self._str(t) == " | tgt: 0.50ms cyc: 2.00ms diff: 1.50ms avg_diff: 1.50ms"

    def test_missing_side_prints_dash(self):
        assert self._str(ExecTimer()) == " | tgt: - cyc: - diff: - avg_diff: -"

    def test_regression_avg_diff_not_rounded(self):
        """Falsification: sub-ms overhead must not round to 0.00ms."""
        t = ExecTimer()
        t.add_target(NS_PER_MS)
        t.add_cycle(NS_PER_MS * 13 // 10)
        assert self._str(t).endswith("avg_diff: 0.30ms")

    def test_regression_avg_diff_weighted(self):
        """Adversarial: a 1-cycle window must not weigh as much as a 9-cycle one."""
        t = ExecTimer()
        t.add_target(NS_PER_MS)
        t.add_cycle(11 * NS_PER_MS)
        self._str(t)
        for _ in range(9):
            t.add_target(NS_PER_MS)
            t.add_cycle(NS_PER_MS)

        # Lifetime: tgt 10/10 = 1ms, cyc 20/10 = 2ms -> 1ms (unweighted: 5ms).
        assert self._str(t).endswith("diff: 0.00ms avg_diff: 1.00ms")

    def test_regression_empty_window_keeps_avg(self):
        """Adversarial: an idle window prints '-' for diff, keeps lifetime avg."""
        t = ExecTimer()
        t.add_target(NS_PER_MS)
        t.add_cycle(3 * NS_PER_MS)
        self._str(t)
        assert self._str(t) == " | tgt: - cyc: - diff: - avg_diff: 2.00ms"


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="fuzz_timer_") as d:
        root = Path(d)
        (root / "corpus").mkdir()
        (root / "corpus" / "seed_a").write_bytes(b"AAAA")
        yield end_of_run._make_fuzzer(root, corpus_dir=str(root / "corpus"), quiet_stats=True)


class TestWiring:
    def test_run_target_feeds_target_timer(self, fuzzer):
        fuzzer._exec_timer.drain()
        with patch.object(fuzzer._runner, "run_target", return_value=(0, "")):
            fuzzer._run_target(b"x")
            fuzzer._run_target(b"y")
        assert fuzzer._exec_timer._target_n == 2

    def test_fuzz_one_feeds_cycle_timer(self, fuzzer):
        fuzzer._exec_timer.drain()
        with patch("fuzzer_tool.services.fuzzer.FuzzRound") as rnd:
            rnd.return_value.run.return_value = False
            fuzzer.fuzz_one(b"x")
        assert fuzzer._exec_timer._cycle_n == 1
