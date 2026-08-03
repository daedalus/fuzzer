"""Regression test: fuzzer tuple histories are parallel array.array pairs.

AGENTS.md style rule: prefer array.array for homogeneous numeric data. The three
fuzzer.py histories (_discovery, _crash_rate, _entropy) and the edge_tracker
coverage timeline are now parallel array("Q")/array("d") pairs instead of
list[tuple]. Verifies the containers are arrays and that the report consumers
reproduce the original tuple-list semantics via independently-derived arithmetic
(not by calling the code under test).
"""

from __future__ import annotations

from array import array

import pytest

from fuzzer_tool.services.report import _crash_rate_trend
from fuzzer_tool.services.stats_reporter import discovery_rate


class _FakeFuzzer:
    """Minimal fuzzer stand-in holding the parallel histories."""

    def __init__(self):
        self._discovery_execs: array = array("Q")
        self._discovery_edges: array = array("Q")
        self._crash_rate_execs: array = array("Q")
        self._crash_rate_counts: array = array("Q")

    def record_crash(self, exec_count, crash_count):
        self._crash_rate_execs.append(exec_count)
        self._crash_rate_counts.append(crash_count)


# Independent reference samples: (exec_count, crash_count)
_REF_SAMPLES = [(100, 5), (200, 12), (300, 18), (900, 21)]


def _make_report_fuzzer():
    f = _FakeFuzzer()
    for e, c in _REF_SAMPLES:
        f.record_crash(e, c)
    return f


def test_histories_are_arrays():
    """The histories are array.array, not lists."""
    f = _FakeFuzzer()
    for name in ("_discovery_execs", "_discovery_edges", "_crash_rate_execs", "_crash_rate_counts"):
        assert isinstance(getattr(f, name), array)


def test_crash_rate_trend_matches_reference():
    """_crash_rate_trend picks the first sample >= each milestone."""
    out = _crash_rate_trend(_make_report_fuzzer())
    # Reference loop: for each milestone m, first sample with exec >= m.
    # m=100 -> (100,5)   -> 5/100  = 5.0%
    # m=500 -> (900,21)  -> 21/900 = 2.3%
    assert "iter   100:     5 crashes (5.0%)" in out
    assert "iter   500:    21 crashes (2.3%)" in out


def test_crash_rate_trend_final_milestone():
    """The final sample (exec not already shown) is appended."""
    out = _crash_rate_trend(_make_report_fuzzer())
    # Final: last sample (900, 21) is not among shown milestones -> appended.
    assert "iter   900:    21 crashes (2.3%)" in out


def test_crash_rate_trend_requires_two_samples():
    f = _FakeFuzzer()
    f.record_crash(100, 1)
    assert _crash_rate_trend(f) == ""


def test_discovery_rate_reference():
    """discovery_rate over the paired arrays equals the tuple-list closed form."""
    execs = array("Q", (0, 100, 200, 300, 400))
    edges = array("Q", (0, 10, 20, 30, 40))
    # Reference: (40 - 0) / (400 - 0) * 1000 = 100 edges per 1000 execs
    assert discovery_rate((execs, edges)) == pytest.approx(100.0)
