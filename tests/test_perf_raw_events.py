"""Raw PMU events (rXXXX) and Skylake-X STLB presets in the perf adapter."""

import argparse

import pytest

from fuzzer_tool.adapters import perf_event
from fuzzer_tool.adapters.perf_event import (
    COUNTER_DEFS,
    PERF_TYPE_RAW,
    PerfCounters,
    counter_supported,
    cpu_signature,
    parse_raw_name,
    resolve_counter,
)
from fuzzer_tool.cli.commands import _hw_perf_counter_list

SKX = ("GenuineIntel", 6, 0x55)
OTHER = ("AuthenticAMD", 25, 1)


def test_preset_encodings():
    # umask << 8 | event select, from Intel's SKX perfmon tables.
    assert COUNTER_DEFS["skx_dtlb_load_stlb_hit"][:2] == (PERF_TYPE_RAW, 0x2008)
    assert COUNTER_DEFS["skx_stlb_miss_loads"][:2] == (PERF_TYPE_RAW, 0x11D0)
    assert COUNTER_DEFS["skx_stlb_miss_stores"][:2] == (PERF_TYPE_RAW, 0x12D0)


@pytest.mark.parametrize(
    ("name", "cfg"),
    [("r2008", 0x2008), ("r11D0", 0x11D0), ("rc0", 0xC0), ("r" + "f" * 16, (1 << 64) - 1)],
)
def test_parse_raw_name(name, cfg):
    assert parse_raw_name(name) == cfg
    assert resolve_counter(name) == (PERF_TYPE_RAW, cfg, True)


@pytest.mark.parametrize("name", ["r", "rxyz", "ref_cycles", "r" + "f" * 17, "R2008", "2008", ""])
def test_not_raw_names(name):
    assert parse_raw_name(name) is None


def test_resolve_known_and_unknown():
    assert resolve_counter("instructions") == COUNTER_DEFS["instructions"]
    assert resolve_counter("nope") is None


def test_counter_supported_by_model():
    assert counter_supported("instructions", OTHER)
    assert counter_supported("r2008", OTHER)
    assert counter_supported("skx_stlb_miss_loads", SKX)
    assert not counter_supported("skx_stlb_miss_loads", OTHER)
    assert not counter_supported("skx_stlb_miss_loads", None)


def test_cpu_signature_parses(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text("processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 85\n")
    assert cpu_signature(str(f)) == SKX
    assert cpu_signature(str(tmp_path / "missing")) is None
    f.write_text("processor\t: 0\n")
    assert cpu_signature(str(f)) is None


def test_perfcounters_keeps_raw_and_filters_presets(monkeypatch):
    monkeypatch.setattr(perf_event, "cpu_signature", lambda *a: OTHER)
    pc = PerfCounters(counter_names=["instructions", "r2008", "skx_stlb_miss_loads", "bogus"])
    assert pc.counter_names == ["instructions", "r2008"]
    assert pc.stats["total_r2008"] == 0
    monkeypatch.setattr(perf_event, "cpu_signature", lambda *a: SKX)
    pc = PerfCounters(counter_names=["skx_stlb_miss_loads"])
    assert pc.counter_names == ["skx_stlb_miss_loads"]


def test_raw_totals_accumulate():
    pc = PerfCounters(counter_names=["r2008"])
    seq = iter([{"r2008": 70}, {"r2008": 75}])
    pc.read_values = lambda: next(seq)
    pc.read_and_reset()
    pc.read_and_reset()
    assert pc.stats["total_r2008"] == 75


def test_attr_uses_raw_type():
    import ctypes

    pc = PerfCounters(counter_names=["r11d0"])
    attr = pc._attr("r11d0")
    assert len(attr) == ctypes.sizeof(perf_event.perf_event_attr)
    pe = perf_event.perf_event_attr.from_buffer_copy(attr)
    assert (pe.type, pe.config) == (PERF_TYPE_RAW, 0x11D0)


def test_group_layout_beyond_table():
    assert perf_event._group_layout(len(COUNTER_DEFS) + 5).size == 8 * (
        perf_event._GROUP_HDR + len(COUNTER_DEFS) + 5
    )


def test_cli_accepts_raw_and_rejects_unknown():
    assert _hw_perf_counter_list("instructions,r2008") == ["instructions", "r2008"]
    with pytest.raises(argparse.ArgumentTypeError, match="rXXXX"):
        _hw_perf_counter_list("bogus")


def test_cli_rejects_preset_on_wrong_cpu(monkeypatch):
    monkeypatch.setattr(perf_event, "cpu_signature", lambda *a: OTHER)
    with pytest.raises(argparse.ArgumentTypeError, match="not valid on this CPU"):
        _hw_perf_counter_list("skx_dtlb_load_stlb_hit")
    monkeypatch.setattr(perf_event, "cpu_signature", lambda *a: SKX)
    assert _hw_perf_counter_list("skx_dtlb_load_stlb_hit") == ["skx_dtlb_load_stlb_hit"]


def test_stats_stlb_field():
    from types import SimpleNamespace

    from fuzzer_tool.services.stats import StatsReporter

    pc = PerfCounters(counter_names=["instructions"])
    pc._read_count = 1
    pc._totals["skx_dtlb_load_stlb_hit"] = 70
    pc._totals["skx_stlb_miss_loads"] = 1
    line = StatsReporter._print_stats_cache_str(None, SimpleNamespace(_perf_counters=pc))
    assert "stlb: 70h 1m" in line
