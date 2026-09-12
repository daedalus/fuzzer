"""The Intel PT backend is reachable from the CLI and drained per execution.

Behavioural tests that drive ``PtTraceSession`` directly pass with every call
site deleted, so the ones here read the source of the call sites themselves —
the same gap that let a drop-triggered resize ship unreferenced.  Six places
have to agree for ``--intel-pt`` to do anything: argparse, the direct
``Fuzzer(...)`` call, ``run_parallel``, ``_worker_main``, the attach in
``adapters/process.py`` and the drain in ``services/runner.py``.
"""

import inspect
import subprocess
import sys

import pytest

from fuzzer_tool.adapters import process, pt_trace
from fuzzer_tool.cli import commands
from fuzzer_tool.core.intel_pt import PtCoverage, PtMapMode
from fuzzer_tool.services import fuzzer as fuzzer_mod
from fuzzer_tool.services import parallel, runner, stats

FLAGS = ("intel_pt", "intel_pt_mode")


def _fuzz_cli(*argv: str) -> subprocess.CompletedProcess:
    """The parser is built inline in ``main()``, so there is no factory to
    call; the CLI itself is the only way to ask argparse what it accepts."""
    return subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", *argv],
        capture_output=True,
        text=True,
    )


class TestFlagReachesEveryLayer:
    def test_argparse_defines_both_flags(self):
        out = _fuzz_cli("--help").stdout
        assert "--intel-pt" in out
        assert "--intel-pt-mode" in out

    def test_argparse_advertises_both_modes(self):
        out = _fuzz_cli("--help").stdout
        assert "block" in out and "edge" in out

    def test_argparse_rejects_an_unknown_mode(self):
        """choices= is what keeps a typo from silently selecting block."""
        result = _fuzz_cli("target", "--intel-pt-mode", "sideways")
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    def test_fuzzer_accepts_both_kwargs(self):
        params = inspect.signature(fuzzer_mod.Fuzzer.__init__).parameters
        assert set(FLAGS) <= set(params)

    @pytest.mark.parametrize("fn", [parallel.run_parallel, parallel._worker_main])
    def test_parallel_signatures_carry_the_flags(self, fn):
        """A flag missing here is dropped silently for every ``-j N`` run --
        the failure mode that hid --kl-ducb and --markov-blend."""
        assert set(FLAGS) <= set(inspect.signature(fn).parameters)

    def test_cmd_fuzz_passes_the_flags_to_both_entry_points(self):
        src = inspect.getsource(commands)
        for flag in FLAGS:
            # once into Fuzzer(...), once into run_parallel(...)
            assert src.count(f'{flag}=getattr(args, "{flag}"') == 2

    def test_worker_forwards_the_flags_to_fuzzer(self):
        src = inspect.getsource(parallel)
        for flag in FLAGS:
            assert src.count(f"{flag}={flag},") == 2


class TestExecPathCallSites:
    def test_all_three_runners_take_a_session(self):
        for fn in (process.run_target_fast, process.run_target_stdin, process.run_target_file):
            assert "pt_session" in inspect.signature(fn).parameters

    def test_all_three_runners_attach_it(self):
        src = inspect.getsource(process)
        assert src.count("pt_session.attach(") == 3

    def test_runner_passes_and_drains_at_every_exec_path(self):
        src = inspect.getsource(runner)
        assert src.count("pt_session=f._pt_session") == 3
        assert src.count("f._pt_session.drain()") == 3

    def test_runner_resets_the_decoder_per_execution(self):
        """Without this the previous run's reference IP completes a
        compressed IP in the next one, fabricating a block."""
        assert inspect.getsource(runner).count("f.pt_cov.reset_edge_map()") == 2

    def test_novelty_disjunction_consults_the_pt_map(self):
        src = inspect.getsource(fuzzer_mod)
        assert src.count("self.pt_cov and self.pt_cov.is_new_coverage()") == 2

    def test_forkserver_is_skipped_when_pt_is_active(self):
        """The forkserver path never reaches the attach sites, so a PT run
        that also took the forkserver would trace nothing."""
        assert "or self._pt_session" in inspect.getsource(fuzzer_mod)

    def test_stats_emits_pt_keys(self):
        src = inspect.getsource(stats)
        assert "pt_cov.stats" in src
        assert "pt_trace_lost_bytes" in src


class _Sink:
    def __init__(self):
        self.chunks = []

    def ingest(self, raw: bytes) -> int:
        self.chunks.append(raw)
        return len(raw)


class TestDrain:
    def _session(self, tmp_path, sink=None):
        return pt_trace.PtTraceSession(root=str(tmp_path / "absent"), sink=sink)

    def test_drain_without_an_open_event_is_zero(self, tmp_path):
        assert self._session(tmp_path, _Sink()).drain() == 0

    def test_drain_does_not_call_the_sink_on_an_empty_trace(self, tmp_path):
        sink = _Sink()
        assert self._session(tmp_path, sink).drain() == 0
        assert sink.chunks == []

    def test_drain_feeds_the_sink_and_reports_new_entries(self, tmp_path, monkeypatch):
        sink = _Sink()
        session = self._session(tmp_path, sink)
        order = []
        session._fd = 99  # past the early return; close() below is stubbed
        monkeypatch.setattr(session, "disable", lambda: order.append("disable") or True)
        monkeypatch.setattr(session, "read_trace", lambda: order.append("read") or b"\x02\x82")
        monkeypatch.setattr(session, "close", lambda: order.append("close"))
        assert session.drain() == 2
        assert sink.chunks == [b"\x02\x82"]

    def test_drain_stops_the_trace_before_reading_it(self, tmp_path, monkeypatch):
        """Reading a live ring races the producer: the head can pass the tail
        mid-copy and hand back bytes that were already recycled."""
        session = self._session(tmp_path, _Sink())
        order = []
        session._fd = 99
        monkeypatch.setattr(session, "disable", lambda: order.append("disable") or True)
        monkeypatch.setattr(session, "read_trace", lambda: order.append("read") or b"x")
        monkeypatch.setattr(session, "close", lambda: order.append("close"))
        session.drain()
        assert order == ["disable", "read", "close"]

    def test_drain_releases_the_event_even_with_no_sink(self, tmp_path, monkeypatch):
        """One fd plus two mmaps per execution leak otherwise."""
        session = self._session(tmp_path, None)
        closed = []
        session._fd = 99
        monkeypatch.setattr(session, "disable", lambda: True)
        monkeypatch.setattr(session, "read_trace", lambda: b"x")
        monkeypatch.setattr(session, "close", lambda: closed.append(True))
        session.drain()
        assert closed == [True]

    def test_attach_failure_is_counted_not_raised(self, tmp_path):
        session = self._session(tmp_path, _Sink())
        assert session.attach(1) is False
        assert session.attach_failures == 1
        assert session.stats["pt_trace_attach_failures"] == 1

    def test_sink_defaults_to_none(self, tmp_path):
        assert self._session(tmp_path).sink is None

    def test_real_coverage_map_works_as_the_sink(self, tmp_path):
        """The duck type the session expects is what PtCoverage provides."""
        cov = PtCoverage(mode=PtMapMode.BLOCK)
        session = self._session(tmp_path, cov)
        assert session.sink is cov
        assert cov.ingest(b"") == 0
