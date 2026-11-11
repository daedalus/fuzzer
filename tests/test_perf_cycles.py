"""Cycle counters: real IPC, stall rates, and the uapi software event ids."""

from types import SimpleNamespace

from fuzzer_tool.adapters import perf_event
from fuzzer_tool.adapters.perf_event import PerfCounters
from fuzzer_tool.services.stats import StatsReporter


def test_sw_event_ids_match_uapi():
    # enum perf_sw_ids: 1 is TASK_CLOCK, 2 is PAGE_FAULTS; MIN/MAJ are 5/6.
    assert perf_event.COUNTER_DEFS["page_faults"][1] == 5
    assert perf_event.COUNTER_DEFS["page_faults_major"][1] == 6
    assert perf_event.COUNTER_DEFS["context_switches"][1] == 3


def test_hw_event_ids_match_uapi():
    assert perf_event.COUNTER_DEFS["stalled_cycles_frontend"][1] == 7
    assert perf_event.COUNTER_DEFS["stalled_cycles_backend"][1] == 8
    assert perf_event.COUNTER_DEFS["ref_cycles"][1] == 9


def test_cycles_in_defaults_stalls_not():
    pc = PerfCounters()
    assert "cpu_cycles" in pc.counter_names
    assert "stalled_cycles_frontend" not in pc.counter_names


def test_ipc_zero_without_cycles_ipb_kept():
    pc = PerfCounters()
    pc._total_instructions = 900
    pc._total_branches = 100
    s = pc.stats
    assert s["ipc"] == 0
    assert s["ipb"] == 9


def test_ipc_and_stall_rates():
    pc = PerfCounters()
    seq = iter(
        [
            {
                "instructions": 3000,
                "cpu_cycles": 2000,
                "stalled_cycles_frontend": 200,
                "stalled_cycles_backend": 600,
            }
        ]
    )
    pc.read_values = lambda: next(seq)
    pc.read_and_reset()
    s = pc.stats
    assert abs(s["ipc"] - 1.5) < 1e-12
    assert abs(s["stall_frontend_rate"] - 0.1) < 1e-12
    assert abs(s["stall_backend_rate"] - 0.3) < 1e-12


def _line(pc):
    return StatsReporter._print_stats_perf_str(None, SimpleNamespace(_perf_counters=pc))


def test_perf_field_ipc_and_stall():
    pc = PerfCounters()
    pc._read_count = 1
    pc._total_instructions = 3000
    pc._total_branches = 500
    pc._total_cpu_cycles = 2000
    pc._total_stalled_frontend = 200
    pc._total_stalled_backend = 600
    line = _line(pc)
    assert "ipb:6.00 ipc:1.50 stall:10%f/30%b" in line
