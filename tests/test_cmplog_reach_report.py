"""Does the comparison instrumentation actually reach the target?

Every way of getting that wrong is silent. An -O2 build without
``-fno-builtin`` inlines the comparisons and the libc layer sees nothing
(measured on ``cmplog_exercise.c``: 20 call sites, 4 records). A preload
loses the symbol-lookup race to the executable's own weak sancov stubs and
the shim is never entered. A target never reaches its parser on the seeds
it was handed. From the outside all three look the same -- the campaign
runs and the token pool stays empty -- which reads as "this target has no
interesting comparisons".

The counters answer it directly. Calibration is where it is cheap: every
seed has just been executed exactly once, so the totals are over a known
number of unmutated executions.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector
from fuzzer_tool.services.fuzzer import Fuzzer


def _collector(fired: dict[str, int], asserted: dict[str, int] | None = None):
    c = CmplogCollector()
    c.cmp_fired = dict(fired)
    c.cmp_asserted = dict(asserted or {})
    return c


class TestLayerTotals:
    def test_split_is_by_layer_not_by_name_list(self):
        c = _collector(
            {"memcmp": 10, "strstr": 4, "trace_cmp4": 7, "trace_switch": 1},
            {"memcmp": 2, "trace_cmp4": 3},
        )
        assert c.layer_totals() == ((14, 2), (8, 3))

    def test_an_unknown_trace_callback_lands_on_layer_two(self):
        """The prefix rule, so a new callback needs no edit here."""
        assert _collector({"trace_cmp16": 5}).layer_totals() == ((0, 0), (5, 0))

    def test_no_counters_is_zero_not_an_error(self):
        assert CmplogCollector().layer_totals() == ((0, 0), (0, 0))


class TestReachReport:
    """``_report_comparison_reach`` reads two attributes and prints."""

    @pytest.fixture
    def f(self):
        obj = Fuzzer.__new__(Fuzzer)
        obj._cmplog = None
        obj._reset_cmplog = lambda: None
        return obj

    def _run(self, f, capsys, fired, execs=11):
        f._cmplog = _collector(fired)
        f._cmplog.collect_counts = lambda: ({}, {})
        f._report_comparison_reach(execs)
        return capsys.readouterr().out

    def test_silence_is_the_case_that_warns(self, f, capsys):
        out = self._run(f, capsys, {})
        assert out.startswith("[!]")
        assert "no comparisons observed" in out
        assert "11 seed executions" in out

    def test_both_layers_live_reports_the_split(self, f, capsys):
        out = self._run(f, capsys, {"memcmp": 40, "trace_cmp4": 2})
        assert out.startswith("[*]")
        assert "42 comparisons" in out
        assert "libc 40" in out and "trace-cmp 2" in out
        assert "inlined" not in out

    def test_trace_cmp_only_flags_the_inlined_regime(self, f, capsys):
        """The regime where the record stream is large and nearly worthless.

        trace-cmp instruments IR ``icmp``; clang's ExpandMemCmp runs after
        it, so what it logs for a memcmp is ``result == 0`` -- the
        degenerate pair, not the operands.
        """
        out = self._run(f, capsys, {"trace_cmp4": 900})
        assert "libc 0" in out
        assert "inlined" in out
        assert "-fno-builtin-memcmp" in out

    def test_libc_only_is_the_good_case_and_stays_quiet(self, f, capsys):
        out = self._run(f, capsys, {"memcmp": 12})
        assert "inlined" not in out

    def test_no_cmplog_prints_nothing(self, capsys):
        f = Fuzzer.__new__(Fuzzer)
        f._cmplog = None
        f._report_comparison_reach(11)
        assert capsys.readouterr().out == ""

    def test_no_executions_prints_nothing(self, f, capsys):
        """An empty corpus makes the denominator meaningless, not alarming."""
        assert self._run(f, capsys, {}, execs=0) == ""
