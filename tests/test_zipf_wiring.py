"""Wiring of core/zipf.py: EdgeTracker API, saturation-gate veto, stats, report."""

import io
import types
from contextlib import redirect_stdout

from fuzzer_tool.core import edge_tracker as edge_tracker_mod
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.zipf import HeapsFit, TailLaw, ZipfFit
from fuzzer_tool.services.report import _zipf_tail
from fuzzer_tool.services.seed_picker import (
    SATURATION_REFRESH_EXECS,
    ZIPF_GROWTH_BETA,
    SeedPicker,
)
from fuzzer_tool.services.stats import _zipf_stats

SEEDS = 100  # corpus size m: owner counts are capped here
ALPHA = 2.0
EDGES = 4000


def _owner_counts() -> list[int]:
    """Per-edge owner counts following a truncated pmf k^-ALPHA on 1..SEEDS."""
    norm = sum(k**-ALPHA for k in range(1, SEEDS + 1))
    counts: list[int] = []
    for k in range(1, SEEDS + 1):
        counts.extend([k] * round(EDGES * k**-ALPHA / norm))
    return counts


def _zipf_tracker() -> EdgeTracker:
    """Tracker whose edge e is covered by exactly counts[e] seeds."""
    counts = _owner_counts()
    seeds: list[set[int]] = [set() for _ in range(SEEDS)]
    for edge, c in enumerate(counts):
        for s in range(c):
            seeds[s].add(edge)
    t = EdgeTracker()
    for s, edges in enumerate(seeds):
        t.record_edges(f"s{s}", edges)
    return t


def _heaps_tracker(beta: float, points: int = 60) -> EdgeTracker:
    """Tracker whose coverage timeline follows D(N) = 2 * N^beta."""
    t = EdgeTracker()
    edge = 0
    for i in range(points):
        n = int(1000 * 1.1**i)
        target = max(1, round(2.0 * n**beta))
        if target > edge:
            t.record_edges(f"h{i}", set(range(edge, target)))
            edge = target
        t.record_coverage_snapshot(n)
    return t


class TestEdgeTrackerZipf:
    def test_fit_reads_owner_counts(self):
        fit = _zipf_tracker().zipf_estimate()
        assert isinstance(fit, ZipfFit)
        assert fit.law is TailLaw.POWER_LAW
        assert abs(fit.alpha - ALPHA) < 0.05

    def test_empty_tracker_is_insufficient(self):
        assert EdgeTracker().zipf_estimate().law is TailLaw.INSUFFICIENT

    def test_cached_until_coverage_changes(self):
        t = _zipf_tracker()
        first = t.zipf_estimate()
        assert t.zipf_estimate() is first

        # A known seed gaining an already-known edge changes one owner count
        # without changing m or the edge total; the cache must still notice.
        t.record_edges("s0", {EDGES - 1})
        assert t.zipf_estimate() is not first

    def test_prune_and_add_with_equal_totals_refits(self, monkeypatch):
        # Adversarial: a prune plus a new seed can leave (seeds, incidences,
        # edges) unchanged while the owner spectrum moves; the memo must not
        # key on those totals.
        calls: list[int] = []
        real = edge_tracker_mod.fit_zipf
        monkeypatch.setattr(
            edge_tracker_mod, "fit_zipf", lambda c, xmax=0: calls.append(xmax) or real(c, xmax)
        )
        t = EdgeTracker(max_tracked_seeds=2)
        t.record_edges("a", {1})
        t.record_edges("b", {2})
        t.zipf_estimate()
        t.record_edges("c", {2})  # third seed: prunes back to the ceiling
        assert len(t.seed_edges) == 2
        t.zipf_estimate()
        assert len(calls) == 2

    def test_restore_drops_cache(self):
        t = _zipf_tracker()
        t.zipf_estimate()
        restored = EdgeTracker()
        restored.from_dict(t.to_dict())
        assert restored.zipf_estimate() == t.zipf_estimate()

    def test_heaps_reads_timeline(self):
        fit = _heaps_tracker(0.5).heaps_estimate()
        assert isinstance(fit, HeapsFit)
        assert abs(fit.beta - 0.5) < 0.03

    def test_heaps_none_without_timeline(self):
        assert EdgeTracker().heaps_estimate() is None


class _Fuzzer:
    def __init__(self, tracker):
        self._edge_tracker = tracker
        self.exec_count = 0
        self._last_new_edge_exec = 0
        self._cached_weights: dict = {}


def _gate(saturation: float, law: TailLaw, beta: float):
    t = EdgeTracker()
    t.good_turing_estimate = lambda: {"saturation": saturation}  # type: ignore[method-assign]
    fit = ZipfFit(ALPHA, 1, 500, 1.0, 0.01, 5.0, law)
    t.zipf_estimate = lambda: fit  # type: ignore[method-assign]
    t.heaps_estimate = lambda: HeapsFit(beta, 1.0, 0.999)  # type: ignore[method-assign]
    f = _Fuzzer(t)
    return f, SeedPicker(f)


class TestSaturationGateVeto:
    def test_power_law_growth_vetoes_the_gate(self):
        # Chao2 reads saturated, but the tail is Zipf and still growing.
        f, p = _gate(1.0, TailLaw.POWER_LAW, ZIPF_GROWTH_BETA * 2)
        assert p._saturation_gate() is False

    def test_plateau_keeps_the_gate(self):
        f, p = _gate(1.0, TailLaw.POWER_LAW, 0.0)
        assert p._saturation_gate() is True

    def test_non_power_law_keeps_the_gate(self):
        # Falsification: growth alone is not enough without a Zipf tail.
        f, p = _gate(1.0, TailLaw.NOT_POWER_LAW, ZIPF_GROWTH_BETA * 2)
        assert p._saturation_gate() is True

    def test_insufficient_keeps_the_gate(self):
        f, p = _gate(1.0, TailLaw.INSUFFICIENT, ZIPF_GROWTH_BETA * 2)
        assert p._saturation_gate() is True

    def test_unsaturated_skips_the_fit(self):
        # Adversarial cost check: below the gate the fit must not run at all.
        t = EdgeTracker()
        t.good_turing_estimate = lambda: {"saturation": 0.1}  # type: ignore[method-assign]

        def _boom():
            raise AssertionError("zipf fit ran below the gate")

        t.zipf_estimate = _boom  # type: ignore[method-assign]
        f = _Fuzzer(t)
        assert SeedPicker(f)._saturation_gate() is False

    def test_veto_refreshes_with_the_estimate(self):
        f, p = _gate(1.0, TailLaw.POWER_LAW, ZIPF_GROWTH_BETA * 2)
        assert p._saturation_gate() is False

        # Growth stops; the veto clears on the next refresh, not before.
        f._edge_tracker.heaps_estimate = lambda: HeapsFit(0.0, 1.0, 1.0)
        f.exec_count = SATURATION_REFRESH_EXECS - 1
        f._last_new_edge_exec = f.exec_count
        assert p._saturation_gate() is False
        f.exec_count = SATURATION_REFRESH_EXECS
        f._last_new_edge_exec = f.exec_count
        assert p._saturation_gate() is True


def _f_with(tracker):
    return types.SimpleNamespace(_edge_tracker=tracker)


class TestReportSection:
    def test_section_present(self):
        text = _zipf_tail(_f_with(_zipf_tracker()))
        assert "Zipf" in text
        assert "power_law" in text

    def test_section_absent_when_insufficient(self):
        assert _zipf_tail(_f_with(EdgeTracker())) == ""

    def test_section_absent_without_tracker(self):
        assert _zipf_tail(types.SimpleNamespace()) == ""


class TestStats:
    def test_dump_entry(self):
        stats: dict = {}
        _zipf_stats(_f_with(_zipf_tracker()), stats)
        assert stats["zipf"]["law"] == "power_law"
        assert abs(stats["zipf"]["alpha"] - ALPHA) < 0.05

    def test_dump_entry_absent_when_insufficient(self):
        stats: dict = {}
        _zipf_stats(_f_with(EdgeTracker()), stats)
        assert "zipf" not in stats

    def test_summary_lines(self):
        from fuzzer_tool.services.stats import StatsReporter

        reporter = StatsReporter.__new__(StatsReporter)
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter._print_summary_zipf(_f_with(_heaps_tracker(0.5)))
            reporter._print_summary_zipf(_f_with(_zipf_tracker()))
        out = buf.getvalue()
        assert "Heaps" in out
        assert "Zipf tail" in out
