"""Branch miss rate (branch_misses / branches) in the perf adapter and the perf: stats field."""

from types import SimpleNamespace

from fuzzer_tool.adapters.perf_event import PerfCounters
from fuzzer_tool.services.stats import StatsReporter


def test_branch_miss_rate_zero_without_branches():
    assert PerfCounters().stats["branch_miss_rate"] == 0


def test_read_and_reset_accumulates_branch_miss_rate():
    pc = PerfCounters()
    seq = iter(
        [
            {"branches": 1000, "branch_misses": 20},
            {"branches": 4000, "branch_misses": 50},
        ]
    )
    pc.read_values = lambda: next(seq)
    pc.read_and_reset()
    pc.read_and_reset()
    s = pc.stats
    assert s["total_branches"] == 4000
    assert s["total_branch_misses"] == 50
    assert abs(s["branch_miss_rate"] - 50 / 4000) < 1e-12


def test_perf_field_shows_branch_miss_rate():
    pc = PerfCounters()
    pc._read_count = 1
    pc._total_instructions = 10_000
    pc._total_branches = 2_000
    pc._total_branch_misses = 50
    line = StatsReporter._print_stats_perf_str(None, SimpleNamespace(_perf_counters=pc))
    assert "50bm 2.50% " in line
