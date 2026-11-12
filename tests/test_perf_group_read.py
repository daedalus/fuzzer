"""Grouped perf reads, multiplex scaling, persistent fds, software fallback.

The syscall is faked through ``PerfCounters._libc``: each open records the
attr it was handed, so the tests assert on the exact layout the kernel sees.
"""

import argparse
import os
import struct

import pytest

from fuzzer_tool.adapters import perf_event
from fuzzer_tool.adapters.perf_event import (
    COUNTER_DEFS,
    PERF_FORMAT_GROUP,
    PERF_FORMAT_TOTAL_TIME_ENABLED,
    PERF_FORMAT_TOTAL_TIME_RUNNING,
    PerfCounters,
)

_FMT = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING


class _FakeSys:
    """perf_event_open stand-in: real /dev/null fds, attrs recorded."""

    def __init__(self, refuse=lambda attr, group_fd: False):
        self.opens: list[tuple[perf_event.perf_event_attr, int, int]] = []

        # A plain function: PerfCounters sets restype/argtypes on it.
        def syscall(nr, attr_buf, pid, cpu, group_fd, flags):
            attr = perf_event.perf_event_attr.from_buffer_copy(bytes(attr_buf))
            if refuse(attr, group_fd):
                return -1
            fd = os.open(os.devnull, os.O_RDONLY)
            self.opens.append((attr, pid, group_fd))
            return fd

        self.syscall = syscall


def _pc(names, fake=None, monkeypatch=None) -> PerfCounters:
    pc = PerfCounters(counter_names=names)
    pc._available = True
    pc._libc = fake or _FakeSys()
    monkeypatch.setattr(perf_event.fcntl, "ioctl", lambda *a: 0)
    return pc


def _group_buf(enabled, running, *vals) -> bytes:
    return struct.pack(f"<{3 + len(vals)}Q", len(vals), enabled, running, *vals)


def test_new_counter_encodings():
    cache = perf_event._hw_cache_config
    assert COUNTER_DEFS["llc_read_access"][:2] == (perf_event.PERF_TYPE_HW_CACHE, cache(2, 0, 0))
    assert COUNTER_DEFS["llc_read_miss"][:2] == (perf_event.PERF_TYPE_HW_CACHE, cache(2, 0, 1))
    assert COUNTER_DEFS["dtlb_read_miss"][:2] == (perf_event.PERF_TYPE_HW_CACHE, cache(3, 0, 1))
    assert COUNTER_DEFS["itlb_read_miss"][:2] == (perf_event.PERF_TYPE_HW_CACHE, cache(4, 0, 1))
    assert COUNTER_DEFS["task_clock"][:2] == (perf_event.PERF_TYPE_SOFTWARE, 1)
    assert COUNTER_DEFS["alignment_faults"][:2] == (perf_event.PERF_TYPE_SOFTWARE, 7)


def test_members_join_the_leader_group(monkeypatch):
    fake = _FakeSys()
    pc = _pc(["instructions", "branches", "page_faults"], fake, monkeypatch)
    assert pc.open_for_pid(1234)

    leader_fd = pc._fds["instructions"]
    assert [g for _a, _p, g in fake.opens] == [-1, leader_fd, leader_fd]
    assert all(a.read_format == _FMT for a, _p, _g in fake.opens)
    assert pc._groups == [(leader_fd, ("instructions", "branches", "page_faults"))]
    pc.close()


def test_refused_member_opens_standalone(monkeypatch):
    """Adversarial: a member the group refuses still gets counted, alone."""
    fake = _FakeSys(refuse=lambda a, g: g != -1 and a.config == 5)
    pc = _pc(["instructions", "branch_misses", "branches"], fake, monkeypatch)
    pc.open_for_pid(1)

    lead, miss = pc._fds["instructions"], pc._fds["branch_misses"]
    assert pc._groups == [(lead, ("instructions", "branches")), (miss, ("branch_misses",))]
    pc.close()


def test_group_read_scales_by_enabled_over_running(monkeypatch):
    pc = _pc(["instructions", "branches"], monkeypatch=monkeypatch)
    pc._groups = [(99, ("instructions", "branches"))]
    monkeypatch.setattr(perf_event.os, "read", lambda fd, n: _group_buf(1000, 250, 10, 30))

    # 1/4 of the time on the PMU: every value is an estimate x4.
    assert pc.read_values() == {"instructions": 40, "branches": 120}


def test_unschedulable_group_is_split(monkeypatch):
    """Falsification: a group that never ran must not read as zero work."""
    pc = _pc(["instructions", "branches"], monkeypatch=monkeypatch)
    pc._groups = [(99, ("instructions", "branches"))]
    monkeypatch.setattr(perf_event.os, "read", lambda fd, n: _group_buf(1000, 0, 0, 0))

    assert pc.read_values() == {"instructions": 0, "branches": 0}
    assert pc._grouped is False

    fake = _FakeSys()
    pc._libc = fake
    pc.open_for_pid(7)
    assert [g for _a, _p, g in fake.opens] == [-1, -1]
    pc.close()


def test_ensure_open_reuses_fds_for_same_pid(monkeypatch):
    fake = _FakeSys()
    pc = _pc(["instructions", "branches"], fake, monkeypatch)
    pc.ensure_open(42)
    pc.ensure_open(42)
    assert len(fake.opens) == 2

    pc.ensure_open(43)
    assert len(fake.opens) == 4
    assert {p for _a, p, _g in fake.opens} == {42, 43}
    pc.close()


def test_open_for_pid_always_reopens(monkeypatch):
    """Spawn path: a recycled pid number is a different process."""
    fake = _FakeSys()
    pc = _pc(["instructions"], fake, monkeypatch)
    pc.open_for_pid(42)
    pc.open_for_pid(42)
    assert len(fake.opens) == 2
    pc.close()


def test_scaled_delta_never_negative():
    """Adversarial: a shrinking multiplex estimate is no work, not 2^64."""
    pc = PerfCounters(counter_names=["instructions"])
    seq = iter([{"instructions": 1000}, {"instructions": 990}])
    pc.read_values = lambda: next(seq)
    assert pc.read_and_reset()["instructions"] == 1000
    assert pc.read_and_reset()["instructions"] == 0


def test_totals_cover_every_counter():
    pc = PerfCounters(counter_names=["page_faults", "context_switches", "llc_read_miss"])
    seq = iter([{"page_faults": 7, "context_switches": 2, "llc_read_miss": 5}])
    pc.read_values = lambda: next(seq)
    pc.read_and_reset()

    s = pc.stats
    assert s["total_page_faults"] == 7
    assert s["total_context_switches"] == 2
    assert s["total_llc_read_miss"] == 5


def test_software_fallback_when_no_pmu(monkeypatch):
    """No PMU (most VMs): software events still make the counters usable."""
    fake = _FakeSys(refuse=lambda a, g: a.type != perf_event.PERF_TYPE_SOFTWARE)
    monkeypatch.setattr(perf_event.ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(perf_event.os, "geteuid", lambda: 0)
    pc = PerfCounters()

    assert pc.available
    assert not pc.has_pmu


def test_nothing_opens_means_unavailable(monkeypatch):
    fake = _FakeSys(refuse=lambda a, g: True)
    monkeypatch.setattr(perf_event.ctypes, "CDLL", lambda *a, **k: fake)
    monkeypatch.setattr(perf_event.os, "geteuid", lambda: 0)
    assert not PerfCounters().available


def test_defaults_include_free_software_events():
    for name in ("task_clock", "page_faults", "context_switches"):
        assert name in perf_event.DEFAULT_COUNTERS


def _sw_ok() -> bool:
    pc = PerfCounters(counter_names=["task_clock"])
    return pc.available


@pytest.mark.skipif(not _sw_ok(), reason="perf_event_open unavailable")
def test_real_software_counters_count_own_work():
    pc = PerfCounters(counter_names=["task_clock", "page_faults"])
    assert pc.ensure_open(0)
    pc.read_and_reset()

    # 64 MiB, touched page by page: minor faults the kernel must count.
    buf = bytearray(64 << 20)
    for i in range(0, len(buf), 4096):
        buf[i] = 1
    d = pc.read_and_reset()
    pc.close()

    # At least one fault per 2 MiB: THP may back the buffer with huge pages.
    assert d["page_faults"] >= len(buf) >> 21
    assert d["task_clock"] > 0


def test_cli_counter_list(monkeypatch):
    from fuzzer_tool.cli import commands

    names = commands._hw_perf_counter_list("instructions,page_faults")
    assert names == ["instructions", "page_faults"]
    with pytest.raises(argparse.ArgumentTypeError):
        commands._hw_perf_counter_list("instructions,bogus")


def test_cli_flag_reaches_fuzzer(monkeypatch):
    from fuzzer_tool.cli import commands

    seen = {}
    monkeypatch.setattr(commands, "cmd_fuzz", lambda args: seen.setdefault("a", args) and 0)
    argv = ["fuzzer-tool", "fuzz", "/bin/true", "--hw-perf-counters", "task_clock"]
    monkeypatch.setattr(commands.sys, "argv", argv)
    commands.main()
    assert seen["a"].hw_perf_counters == ["task_clock"]


def test_fuzzer_uses_requested_counters(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    if not _sw_ok():
        pytest.skip("perf_event_open unavailable")
    corpus = tmp_path / "c"
    corpus.mkdir()
    f = Fuzzer(
        target="/bin/true",
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "k"),
        hw_perf_counters=["task_clock"],
        use_coverage=False,
    )
    assert f.hw_perf
    assert f._perf_counters.counter_names == ["task_clock"]
    f._perf_counters.close()
