"""Extra cache/bus events: ids match enum perf_hw_cache_id / perf_hw_id encoding."""

from fuzzer_tool.adapters import perf_event
from fuzzer_tool.adapters.perf_event import COUNTER_DEFS, DEFAULT_COUNTERS, PERF_TYPE_HW_CACHE


def _cfg(name):
    typ, config, _ = COUNTER_DEFS[name]
    return typ, config & 0xFF, (config >> 8) & 0xFF, (config >> 16) & 0xFF


def test_cache_event_encoding():
    # (cache_id, op, result); L1D=0 L1I=1 LL=2 DTLB=3 ITLB=4 BPU=5, READ=0 WRITE=1, ACCESS=0 MISS=1
    expect = {
        "l1d_write_access": (0, 1, 0),
        "l1d_write_miss": (0, 1, 1),
        "l1i_read_miss": (1, 0, 1),
        "bpu_read_access": (5, 0, 0),
        "bpu_read_miss": (5, 0, 1),
    }
    for name, want in expect.items():
        typ, *got = _cfg(name)
        assert typ == PERF_TYPE_HW_CACHE
        assert tuple(got) == want, name


def test_bus_cycles_id():
    assert COUNTER_DEFS["bus_cycles"][1] == 6


def test_new_events_opt_in_only():
    for name in (
        "l1d_write_access",
        "l1d_write_miss",
        "l1i_read_miss",
        "bpu_read_access",
        "bpu_read_miss",
        "bus_cycles",
    ):
        assert name not in DEFAULT_COUNTERS


def test_stats_has_totals_for_new_events():
    pc = perf_event.PerfCounters()
    s = pc.stats
    for name in ("l1i_read_miss", "bpu_read_miss", "l1d_write_miss", "bus_cycles"):
        assert s[f"total_{name}"] == 0
