"""Regression: the first `[*] execs` stats line must not wait for EPS to stabilize.

The early stats ticks can show inflated EPS (bursty warm-up, startup time in
the denominator).  The old stats-interval calculation fed that single raw
reading straight back into `10 * last_eps` for every tick.  Fix: the first
tick prints promptly at 1x EPS, and subsequent ticks are spaced at
`10 * last_avg_eps` (mean of the last 10 avg-eps samples), falling back to the
fixed stats interval while the window is still filling.
"""

from array import array
from time import time
from unittest.mock import patch

from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.stats import StatsReporter
from tests.test_stats_reporter import _mock_fuzzer


def _stub_fuzzer(history: array, history_max: int = 10) -> Fuzzer:
    """Minimal Fuzzer without __init__ side effects, for interval helpers."""
    f = Fuzzer.__new__(Fuzzer)
    f._eps_history = history
    f._eps_history_max = history_max
    f.stats_interval = 1000
    f.exec_count = 1000
    f.start_time = time() - 1.0  # 1 second of elapsed run time
    f._resume_baseline_exec = 0  # fresh run, no resumed baseline
    return f


def test_last_avg_eps_returns_zero_until_window_full():
    """Partial windows must not feed the interval calc (unstable EPS)."""
    f = _stub_fuzzer(array("d", [100.0, 200.0, 300.0]))
    assert f._last_avg_eps() == 0.0


def test_last_avg_eps_means_full_window():
    """Once the window fills, last_avg_eps is the mean of the samples."""
    f = _stub_fuzzer(array("d", [100.0] * 9 + [200.0]))
    assert f._last_avg_eps() == 110.0


def test_first_interval_uses_one_x_eps():
    """The very first tick spaces at 1x EPS so the line appears promptly."""
    f = _stub_fuzzer(array("d"))
    # 1000 execs over ~1s of elapsed run time -> ~1x EPS spacing.
    assert 900 <= f._stats_effective_interval() <= 1000


def test_warmup_intervals_use_fixed_stats_interval():
    """While the window fills, the fixed stats interval is used, not 10x."""
    f = _stub_fuzzer(array("d", [1_000_000.0]))
    assert f._stats_effective_interval() == 1000


def test_effective_interval_scales_with_window_mean():
    """Once stabilized, the interval is 10x the mean of the window."""
    f = _stub_fuzzer(array("d", [100.0] * 10))
    assert f._stats_effective_interval() == 1000
    f._eps_history = array("d", [500.0] * 10)
    assert f._stats_effective_interval() == 5000


def test_stats_line_prints_on_first_tick():
    """print_stats() emits the line from the first tick (no hold)."""
    fuzzer = _mock_fuzzer(_eps_history=array("d"), _eps_history_max=10)
    reporter = StatsReporter(fuzzer)
    with patch("builtins.print") as mock_print:
        reporter.print_stats()
        assert mock_print.call_count == 1
        assert "[*] execs:" in mock_print.call_args[0][0]


def test_stats_line_prints_with_full_window():
    """print_stats() emits the line once the window is full."""
    fuzzer = _mock_fuzzer(_eps_history=array("d", [100.0] * 10), _eps_history_max=10)
    reporter = StatsReporter(fuzzer)
    with patch("builtins.print") as mock_print:
        reporter.print_stats()
        assert mock_print.call_count == 1
        assert "[*] execs:" in mock_print.call_args[0][0]
