"""Edges per bit over time: timeline column, trend verdict, summary line."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.core.edge_density import (
    TREND_DENSITY_SHIFT,
    TREND_EDGE_GROWTH,
    Trend,
    density_trend,
    edges_per_bit,
)
from fuzzer_tool.core.edge_tracker import COVERAGE_TIMELINE_MAX, EdgeTracker
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.stats import StatsReporter


def test_bloat_when_density_falls_and_edges_flat():
    """Edges +2%, bytes x4: density -75% -> BLOAT."""
    got = density_trend([100, 102], [1000, 4000])
    assert got.verdict is Trend.BLOAT
    assert got.first == pytest.approx(edges_per_bit(100, 1000))
    assert got.last == pytest.approx(edges_per_bit(102, 4000))


def test_density_fall_with_edge_growth_is_steady():
    """Falsification: density fell but edges grew past the bound -> not bloat."""
    edges = [100, int(100 * (1 + 2 * TREND_EDGE_GROWTH))]
    assert density_trend(edges, [1000, 4000]).verdict is Trend.STEADY


def test_compacting_when_density_rises():
    """Same edges, a third of the bytes: COMPACT."""
    assert density_trend([100, 100], [3000, 1000]).verdict is Trend.COMPACT


def test_small_move_is_steady():
    shift = 1 - TREND_DENSITY_SHIFT / 2
    assert density_trend([100, 100], [1000, int(1000 / shift)]).verdict is Trend.STEADY


@pytest.mark.parametrize(
    ("edges", "nbytes"),
    [([], []), ([5], [10]), ([5, 6], [0, 0]), ([0, 0], [10, 10]), ([5, 6], [10, 0])],
)
def test_too_few_known_points(edges, nbytes):
    """Adversarial: under two points with edges and bytes -> None."""
    assert density_trend(edges, nbytes) is None


def test_unknown_points_skipped():
    """Legacy (0-byte) and pre-coverage (0-edge) points do not anchor the trend."""
    got = density_trend([0, 100, 100, 101], [500, 0, 1000, 4000])
    assert got.first == pytest.approx(edges_per_bit(100, 1000))
    assert got.verdict is Trend.BLOAT


def test_tracker_records_bytes_in_step():
    et = EdgeTracker()
    for i in range(COVERAGE_TIMELINE_MAX + 1):
        et.record_coverage_snapshot(i, corpus_bytes=i + 1)
    assert len(et._coverage_bytes) == len(et._coverage_execs)
    assert et._coverage_bytes[-1] == COVERAGE_TIMELINE_MAX + 1


def test_tracker_roundtrip_and_legacy():
    et = EdgeTracker()
    et.record_coverage_snapshot(1, corpus_bytes=10)
    et.record_coverage_snapshot(2, corpus_bytes=20)

    back = EdgeTracker()
    back.from_dict(et.to_dict())
    assert list(back._coverage_bytes) == [10, 20]

    legacy = et.to_dict()
    del legacy["coverage_bytes"]
    old = EdgeTracker()
    old.from_dict(legacy)
    assert list(old._coverage_bytes) == [0, 0]


def test_tracker_misaligned_saved_bytes_dropped():
    """Adversarial: a bytes column that does not match the timeline reads as unknown."""
    et = EdgeTracker()
    et.record_coverage_snapshot(1, corpus_bytes=10)
    state = et.to_dict()
    state["coverage_bytes"] = [10, 20, 30]
    back = EdgeTracker()
    back.from_dict(state)
    assert list(back._coverage_bytes) == [0]


def test_fuzzer_snapshot_passes_corpus_bytes():
    calls = []
    f = Fuzzer.__new__(Fuzzer)
    f.exec_count = 7
    f.corpus = [b"ab", b"cde"]
    f._edge_tracker = SimpleNamespace(
        record_coverage_snapshot=lambda n, corpus_bytes: calls.append((n, corpus_bytes))
    )
    f._snapshot_coverage()
    assert calls == [(7, 5)]


def test_summary_prints_trend(capsys):
    f = MagicMock()
    f.corpus = [b"x" * 10]
    f.ptrace_cov = None
    f.shm_cov._peak_cumulative_edges = 4
    et = f._edge_tracker
    et.get_cumulative_edge_count.return_value = 4
    et.bitmap_density.return_value = 0.0
    et.birthday_collision_risk.return_value = 0.0
    et.recommended_map_size.return_value = None
    et.good_turing_estimate.return_value = {"n": 0}
    et.coverage_growth_model.return_value = {"confidence": 0.0}
    et.bayesian_coverage_growth_model.return_value = {}
    et.density_trend.return_value = density_trend([100, 101], [1000, 4000])
    with (
        patch.object(StatsReporter, "_print_summary_zipf"),
        patch.object(StatsReporter, "_print_summary_eta"),
        patch.object(StatsReporter, "_print_summary_edge_stats"),
        patch.object(StatsReporter, "_print_summary_confirm"),
    ):
        StatsReporter(f)._print_summary_coverage(f)

    line = next(ln for ln in capsys.readouterr().out.splitlines() if "Edges/bit trend" in ln)
    assert Trend.BLOAT.value in line
    assert f"{edges_per_bit(100, 1000):.4f}" in line
