"""Wiring of core/edge_clusters.py: tracker, seed picker, CLI, summary, report."""

import ast
import inspect
import io
import math
import tempfile
import types
from contextlib import redirect_stdout
from unittest.mock import patch

from fuzzer_tool.cli import commands
from fuzzer_tool.core.edge_clusters import EdgeClusters
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.report import _edge_clusters
from fuzzer_tool.services.seed_picker import (
    RARE_EDGE_GAIN,
    RARE_EDGE_OWNERS,
    SeedPicker,
    _EdgeVectors,
)
from fuzzer_tool.services.stats import StatsReporter

# Same layout as test_edge_clusters: A={1,2,3} x4 seeds, B={5} x3, C={9,10,11} on s4,
# plus s5 = C's twin block {20,21,22} and shared edge 1 (per edge 3/4 unique,
# per class 1/2).
SEEDS = {
    "s0": {1, 2, 3, 5},
    "s1": {1, 2, 3, 5},
    "s2": {1, 2, 3, 5},
    "s3": {1, 2, 3},
    "s4": {9, 10, 11},
}
HONGG_RARE = 4  # rare_edge_count's default threshold (< 4 owners)


def _make_fuzzer(**kw) -> Fuzzer:
    tmpdir = tempfile.mkdtemp(prefix="fuzz_clusters_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            **kw,
        )


def _fill(et: EdgeTracker, seeds=SEEDS) -> None:
    for k, edges in seeds.items():
        et.record_edges(k, set(edges))


def _attached(seeds=SEEDS) -> tuple[EdgeTracker, EdgeClusters]:
    et = EdgeTracker()
    _fill(et, seeds)
    clusters = EdgeClusters()
    clusters.refit(et.seed_edges)
    et.attach_clusters(clusters)
    return et, clusters


# ── EdgeTracker ─────────────────────────────────────────────────────────────


def test_rare_edge_count_counts_classes():
    plain = EdgeTracker()
    _fill(plain)
    et, _ = _attached()
    assert plain.rare_edge_count("s4") == 3
    assert et.rare_edge_count("s4") == 1
    assert et.rare_edge_count("s0", HONGG_RARE) == plain.rare_edge_count("s0", HONGG_RARE) == 1


def test_subsumption_weight_by_class():
    seeds = dict(SEEDS, s5={1, 20, 21, 22})
    plain = EdgeTracker()
    _fill(plain, seeds)
    et, _ = _attached(seeds)
    assert plain.compute_subsumption_weight("s5") == 3 / 4
    assert et.compute_subsumption_weight("s5") == 1 / 2
    assert et.compute_subsumption_weight("missing") == 1.0


def test_edges_after_refit_count_per_edge():
    # Adversarial: edges discovered after the last refit have no class yet;
    # they must still count (one each), not vanish.
    et, _ = _attached()
    et.record_edges("s6", {40, 41})
    assert et.rare_edge_count("s6") == 2


# ── SeedPicker ──────────────────────────────────────────────────────────────


def _picker_fuzzer():
    f = _make_fuzzer(edge_clusters=True)
    _fill(f._edge_tracker)
    f._edge_clusters.refit(f._edge_tracker.seed_edges)
    f._recent_seed_edges = []
    return f


def test_rare_bonus_counts_classes_and_near_misses():
    f = _picker_fuzzer()
    picker = SeedPicker(f)
    et = f._edge_tracker
    for key, (rare, near) in {"s4": (1, 0), "s3": (0, 1)}.items():
        edges = et.seed_edges[key]
        total = sum(et.edge_owner_count(e) for e in edges)
        want = SeedPicker._apply_edge_penalties(1.0, 1, len(edges), rare, total, 0)
        want *= 1.0 + RARE_EDGE_GAIN * math.log2(1.0 + near)
        assert picker._weight_edge_penalties(key, 1.0, 1, f, {}) == want


def test_vector_path_matches_scalar_with_clusters():
    # Control: the dense pass and the scalar fallback agree under clusters.
    f = _picker_fuzzer()
    picker = SeedPicker(f)
    et = f._edge_tracker
    scalar = {k: picker._weight_edge_penalties(k, 1.0, 1, f, {}) for k in SEEDS}
    vectors = _EdgeVectors()
    vectors.prepare([et.seed_edges[k] for k in SEEDS], et._edge_owner_count, [], f._edge_clusters)
    picker._pass_vectors = vectors
    try:
        vec = {k: picker._weight_edge_penalties(k, 1.0, 1, f, {}) for k in SEEDS}
    finally:
        picker._pass_vectors = None
    assert vec == scalar
    assert vectors.stats(et.seed_edges["s4"])[0] == 1  # one rare class, three edges


def test_off_by_default_keeps_per_edge_rarity():
    f = _make_fuzzer()
    _fill(f._edge_tracker)
    assert f._edge_clusters is None
    assert f._edge_tracker.rare_edge_count("s4") == 3
    assert RARE_EDGE_OWNERS == 3


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_flag_wired():
    # Same check as test_afl_det.test_cli_flag_wired.
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    assert calls
    assert all("edge_clusters" in {k.arg for k in c.keywords} for c in calls)
    assert "edge_clusters" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
    assert "edge_clusters" in commands._HAIL_MARY_FLAGS


def test_fuzzer_attaches_tracker_when_on():
    f = _make_fuzzer(edge_clusters=True)
    _fill(f._edge_tracker)
    f._edge_clusters.refit(f._edge_tracker.seed_edges)
    assert f._edge_tracker.rare_edge_count("s4") == 1


# ── summary / report ────────────────────────────────────────────────────────


def _ns(et, clusters=None):
    return types.SimpleNamespace(_edge_tracker=et, _edge_clusters=clusters)


def test_summary_line_reports_class_collapse():
    et = EdgeTracker()
    _fill(et)
    buf = io.StringIO()
    with redirect_stdout(buf):
        StatsReporter.__new__(StatsReporter)._print_summary_edge_clusters(_ns(et))
    out = buf.getvalue()
    assert "Edge clusters:     3 classes, 2 clusters for 7 edges" in out
    assert "3 singleton edges in 1 classes" in out


def test_summary_silent_without_seeds():
    buf = io.StringIO()
    with redirect_stdout(buf):
        StatsReporter.__new__(StatsReporter)._print_summary_edge_clusters(_ns(EdgeTracker()))
    assert buf.getvalue() == ""


def test_report_section():
    et, clusters = _attached()
    text = _edge_clusters(_ns(et, clusters))
    assert "--- Edge Clusters ---" in text
    assert "Near-miss seeds:  1" in text
    assert _edge_clusters(_ns(EdgeTracker())) == ""
