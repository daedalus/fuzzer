"""Igor crash clustering service: coverage reduction, probe driver, CLI (E4)."""

import stat
import sys

from fuzzer_tool.core.crash_graph import graph_from_edges, graph_from_fires
from fuzzer_tool.services.crash_cluster import (
    CoverageReducer,
    Trace,
    TraceProbe,
    cluster_inputs,
)
from tests.support.scripted_rng import ScriptedRng

# Mutation table order in CoverageReducer.
_CUT, _BYTE, _TRUNC = 0, 1, 2


def _probe(data: bytes) -> Trace:
    """Crashes ("S") iff b"BUG" is present; one edge per input byte."""
    sig = "S" if b"BUG" in data else None
    return Trace(sig, graph_from_edges(range(len(data))))


class TestReducer:
    def test_cut_with_fewer_edges_is_kept(self):
        data = b"BUG" + b"x" * 5
        start, ln = 3, 5
        rng = ScriptedRng(randints=[_CUT, start, ln])

        out, trace = CoverageReducer(_probe, rng, max_execs=1).reduce(data, _probe(data))

        assert out == data[:start] + data[start + ln :]
        assert trace.graph.size == len(out)

    def test_falsification_mutant_losing_crash_is_rejected(self):
        data = b"BUG" + b"x" * 5
        rng = ScriptedRng(randints=[_TRUNC, 1])  # -> b"B", no crash

        out, _ = CoverageReducer(_probe, rng, max_execs=1).reduce(data, _probe(data))

        assert out == data

    def test_adversarial_other_signature_is_rejected(self):
        # A smaller mutant that crashes differently is another bug, not a reduction.
        def probe(d):
            return Trace("S" if d == b"BUGxx" else "T", graph_from_edges(range(len(d))))

        rng = ScriptedRng(randints=[_TRUNC, 3])

        out, _ = CoverageReducer(probe, rng, max_execs=1).reduce(b"BUGxx", probe(b"BUGxx"))

        assert out == b"BUGxx"

    def test_equal_coverage_same_length_is_rejected(self):
        data = b"BUGxx"
        rng = ScriptedRng(randints=[_BYTE, 4, ord("y")])

        out, _ = CoverageReducer(_probe, rng, max_execs=1).reduce(data, _probe(data))

        assert out == data

    def test_budget_bounds_execs(self):
        calls = []

        def probe(d):
            calls.append(d)
            return _probe(d)

        rng = ScriptedRng(randints=[_TRUNC, 4, _TRUNC, 3])
        CoverageReducer(probe, rng, max_execs=2).reduce(b"BUGxx", _probe(b"BUGxx"))

        assert len(calls) == 2

    def test_adversarial_empty_input_runs_nothing(self):
        rng = ScriptedRng()  # any draw raises StopIteration

        out, _ = CoverageReducer(_probe, rng, max_execs=5).reduce(
            b"", Trace("S", graph_from_edges([]))
        )

        assert out == b""


class TestClusterInputs:
    def test_groups_by_trace_and_drops_non_crashing(self, tmp_path):
        traces = {
            b"a1": Trace("S1", graph_from_fires([1, 2, 3])),
            b"a2": Trace("S2", graph_from_fires([1, 2, 3])),  # other stack, same path
            b"b": Trace("S3", graph_from_fires([7, 8, 9])),
            b"ok": Trace(None, graph_from_fires([1, 2, 3])),
        }
        paths = []
        for i, data in enumerate(traces):
            p = tmp_path / f"crash_{i}.bin"
            p.write_bytes(data)
            paths.append(p)

        clusters = cluster_inputs(
            paths, lambda d: traces[d], ScriptedRng(), max_execs=0, iters=2, threshold=0.7
        )

        assert clusters == [[0, 1], [2]]


class TestTraceProbe:
    def test_reads_fire_log_and_signal(self, tmp_path):
        # Stand-in for a -D__AFL_TRACE_FIRES=1 build that segfaults.
        target = tmp_path / "t.sh"
        target.write_text(
            '#!/bin/sh\nprintf "1\\n2\\n1\\n3\\n" >> "$__AFL_FIRES_OUT"\nkill -SEGV $$\n'
        )
        target.chmod(target.stat().st_mode | stat.S_IXUSR)

        probe = TraceProbe(str(target), timeout=5.0)
        try:
            trace = probe(b"x")
            again = probe(b"x")
        finally:
            probe.close()

        assert trace.sig == "signal:11"
        assert trace.graph.labels == [1, 2, 3]
        assert again.graph.labels == trace.graph.labels  # log reset per exec
        assert not list(tmp_path.glob("*fires*"))


class TestCli:
    def test_main_wires_arguments(self, tmp_path, monkeypatch):
        from fuzzer_tool.cli.commands import main

        seen = {}
        monkeypatch.setattr(
            "fuzzer_tool.services.crash_cluster.crash_cluster",
            lambda **kw: seen.update(kw) or [],
        )
        argv = ["fuzzer-tool", "crash-cluster", "/bin/true", str(tmp_path)]
        argv += ["--threshold", "0.6", "--wl-iters", "2", "--reduce-execs", "7"]
        monkeypatch.setattr(sys, "argv", argv)

        assert main() == 0
        assert seen["crashes_dir"] == str(tmp_path)
        assert (seen["threshold"], seen["iters"], seen["max_execs"]) == (0.6, 2, 7)
