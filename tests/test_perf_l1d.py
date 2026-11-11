"""L1D read access / read miss counters in the perf adapter and the d1: stats field."""

from types import SimpleNamespace

from fuzzer_tool.adapters import perf_event
from fuzzer_tool.adapters.perf_event import PerfCounters
from fuzzer_tool.services.stats import StatsReporter


def test_l1d_config_encoding():
    # include/uapi/linux/perf_event.h: id | (op << 8) | (result << 16)
    # L1D=0, OP_READ=0, RESULT_ACCESS=0, RESULT_MISS=1
    t, cfg, _ = perf_event.COUNTER_DEFS["l1d_read_access"]
    assert (t, cfg) == (3, 0x0)
    t, cfg, _ = perf_event.COUNTER_DEFS["l1d_read_miss"]
    assert (t, cfg) == (3, 0x10000)


def test_default_counters_include_l1d():
    pc = PerfCounters()
    assert "l1d_read_access" in pc.counter_names
    assert "l1d_read_miss" in pc.counter_names
    assert pc.stats["total_l1d_read_access"] == 0
    assert pc.stats["l1d_miss_rate"] == 0


def test_read_and_reset_accumulates_l1d():
    pc = PerfCounters()
    seq = iter(
        [
            {"l1d_read_access": 100, "l1d_read_miss": 7},
            {"l1d_read_access": 250, "l1d_read_miss": 12},
        ]
    )
    pc.read_values = lambda: next(seq)
    pc.read_and_reset()
    pc.read_and_reset()
    s = pc.stats
    assert s["total_l1d_read_access"] == 250
    assert s["total_l1d_read_miss"] == 12
    assert abs(s["l1d_miss_rate"] - 12 / 250) < 1e-12


def _line(perf):
    return StatsReporter._print_stats_d1_str(None, SimpleNamespace(_perf_counters=perf))


def test_d1_field_absent_without_perf_or_reads():
    assert _line(None) == ""
    assert _line(PerfCounters()) == ""


def test_d1_field_absent_when_l1d_not_counted():
    pc = PerfCounters()
    pc._read_count = 3  # reads happened, but the L1D events never opened
    assert _line(pc) == ""


def test_d1_field_format():
    pc = PerfCounters()
    pc._read_count = 2
    pc._total_l1d_read_access = 1_000_000
    pc._total_l1d_read_miss = 25_000
    assert _line(pc) == " | d1: 1,000,000r 25,000m 2.50%"
