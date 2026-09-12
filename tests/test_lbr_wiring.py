"""The branch-record backend is reachable from the CLI and drained per exec.

Structured like tests/test_pt_wiring.py, for the same reason: tests that
drive ``LbrSession`` directly pass with every call site deleted, so the call
sites are read from source.

The last class is specific to this backend.  The map is *sampled*, so absence
of an edge is a property of the sampling period rather than of the input.  Two
consumers read absence as a fact -- stability calibration and the trim's
subset test -- and both must keep reading the SHM map only.
"""

import inspect
import subprocess
import sys

import pytest

from fuzzer_tool.adapters import lbr_trace, process
from fuzzer_tool.cli import commands
from fuzzer_tool.core import branch_record
from fuzzer_tool.core.branch_record import BranchCoverage
from fuzzer_tool.services import corpus_manager
from fuzzer_tool.services import fuzzer as fuzzer_mod
from fuzzer_tool.services import parallel, runner, stats

FLAGS = ("lbr", "lbr_period")


def _fuzz_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", *argv],
        capture_output=True,
        text=True,
    )


class TestFlagReachesEveryLayer:
    def test_argparse_defines_both_flags(self):
        out = _fuzz_cli("--help").stdout
        assert "--lbr" in out
        assert "--lbr-period" in out

    def test_help_says_the_signal_is_sampled(self):
        """A reader who takes this for edge coverage will misread a plateau."""
        assert "ampled" in _fuzz_cli("--help").stdout

    def test_period_rejects_a_non_integer(self):
        result = _fuzz_cli("target", "--lbr-period", "often")
        assert result.returncode != 0

    def test_fuzzer_accepts_both_kwargs(self):
        params = inspect.signature(fuzzer_mod.Fuzzer.__init__).parameters
        assert set(FLAGS) <= set(params)

    @pytest.mark.parametrize("fn", [parallel.run_parallel, parallel._worker_main])
    def test_parallel_signatures_carry_the_flags(self, fn):
        assert set(FLAGS) <= set(inspect.signature(fn).parameters)

    def test_cmd_fuzz_passes_the_flags_to_both_entry_points(self):
        src = inspect.getsource(commands)
        for flag in FLAGS:
            assert src.count(f'{flag}=getattr(args, "{flag}"') == 2

    def test_worker_forwards_the_flags_to_fuzzer(self):
        src = inspect.getsource(parallel)
        for flag in FLAGS:
            assert src.count(f"{flag}={flag},") == 2


class TestExecPathCallSites:
    def test_all_three_runners_take_a_session(self):
        for fn in (process.run_target_fast, process.run_target_stdin, process.run_target_file):
            assert "lbr_session" in inspect.signature(fn).parameters

    def test_all_three_runners_attach_it(self):
        assert inspect.getsource(process).count("lbr_session.attach(") == 3

    def test_runner_passes_and_drains_at_every_exec_path(self):
        src = inspect.getsource(runner)
        assert src.count("lbr_session=f._lbr_session") == 3
        assert src.count("f._lbr_session.drain()") == 3

    def test_runner_resets_the_map_per_execution(self):
        assert inspect.getsource(runner).count("f.branch_cov.reset_edge_map()") == 2

    def test_novelty_disjunction_consults_the_branch_map(self):
        src = inspect.getsource(fuzzer_mod)
        assert src.count("self.branch_cov and self.branch_cov.is_new_coverage()") == 2

    def test_forkserver_is_skipped_when_lbr_is_active(self):
        assert "or self._lbr_session" in inspect.getsource(fuzzer_mod)

    def test_stats_emits_branch_keys(self):
        src = inspect.getsource(stats)
        assert "branch_cov.stats" in src
        assert "br_lost_records" in src


class TestSampledMapStaysOutOfAbsenceConsumers:
    def test_module_declares_itself_sampled(self):
        assert branch_record.SAMPLED is True

    def test_stability_calibration_reads_only_the_shm_map(self):
        """It masks edges that fail to reproduce. Fed a sampled map it would
        mask every branch the period happened to miss -- all of them
        perfectly reproducible -- and never unmask them."""
        src = inspect.getsource(fuzzer_mod.Fuzzer._calibrate_seed_stability)
        assert "branch_cov" not in src
        assert "shm_cov" in src

    def test_trim_subset_test_reads_only_shm_or_ptrace(self):
        """The trim keeps a shorter input when its edges are a superset. A
        sampled map makes the subset test a coin flip on the period."""
        src = inspect.getsource(corpus_manager.CorpusManager.trim_new_coverage)
        assert "branch_cov" not in src

    def test_coverage_is_a_lower_bound_not_an_equality(self):
        """Two ingests of the same bytes cannot subtract entries: sampling
        yields false negatives, never false positives, and that asymmetry is
        what makes discovery sound while absence is not."""
        cov = BranchCoverage()
        cov.record_edge(0x1000, 0x2000)
        before = cov.cumulative_edges
        cov.ingest(b"")
        assert cov.cumulative_edges == before


class _Sink:
    def __init__(self):
        self.chunks = []

    def ingest(self, raw: bytes) -> int:
        self.chunks.append(raw)
        return len(raw)


class TestDrain:
    def test_drain_without_an_open_event_is_zero(self):
        assert lbr_trace.LbrSession(sink=_Sink()).drain() == 0

    def test_attach_failure_is_counted_not_raised(self):
        session = lbr_trace.LbrSession(sink=_Sink())
        assert session.attach(1) is False
        assert session.stats["br_trace_attach_failures"] == 1

    def test_drain_stops_the_trace_before_reading_it(self, monkeypatch):
        session = lbr_trace.LbrSession(sink=_Sink())
        order = []
        session._fd = 99
        monkeypatch.setattr(session, "disable", lambda: order.append("disable") or True)
        monkeypatch.setattr(session, "read_records", lambda: order.append("read") or b"x")
        monkeypatch.setattr(session, "close", lambda: order.append("close"))
        session.drain()
        assert order == ["disable", "read", "close"]

    def test_real_coverage_map_works_as_the_sink(self):
        cov = BranchCoverage()
        assert lbr_trace.LbrSession(sink=cov).sink is cov
