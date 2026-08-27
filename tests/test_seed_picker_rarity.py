"""Regression tests for the rare-edge energy bonus in SeedPicker.

The bonus is defined over edge *incidence* -- how many distinct corpus seeds
reach an edge -- not over bucketed execution hit volume. See
services/seed_picker.py::_weight_edge_penalties.
"""

import tempfile
from unittest.mock import patch

from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.seed_picker import (
    CROWDED_EDGE_OWNERS,
    RARE_EDGE_OWNERS,
    SeedPicker,
)


def _make_fuzzer():
    tmpdir = tempfile.mkdtemp(prefix="fuzz_rarity_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )


def _weight(f, seed_key, fuzz_count=1):
    return SeedPicker(f)._weight_edge_penalties(seed_key, 1.0, fuzz_count, f)


def test_hot_loop_edges_do_not_hide_rarity():
    """A rare edge stays rare however many times a single seed re-runs it.

    The old test read _global_edge_hits and required <= 2, so an edge inside a
    loop -- hundreds of bucketed counter units from one execution -- was never
    counted as rare even when exactly one seed reached it.
    """
    f = _make_fuzzer()
    et = f._edge_tracker
    # One seed, four edges, each hit hard. Owner count is 1 for all of them.
    et.record_edges("solo", {1, 2, 3, 4}, hit_counts={1: 500, 2: 400, 3: 300, 4: 200})
    assert all(et.edge_owner_count(e) == 1 for e in (1, 2, 3, 4))
    assert all(et._global_edge_hits[e] > 2 for e in (1, 2, 3, 4))
    assert _weight(f, "solo") > 1.0


def test_widely_covered_edges_are_not_rare():
    """Edges reached by many seeds earn no rare bonus."""
    f = _make_fuzzer()
    et = f._edge_tracker
    shared = {10, 11, 12}
    for i in range(RARE_EDGE_OWNERS + 5):
        et.record_edges(f"s{i}", shared)
    assert et.edge_owner_count(10) > RARE_EDGE_OWNERS
    assert _weight(f, "s0") <= 1.0


def test_rare_bonus_is_sublinear_and_applied_once():
    """The bonus grows logarithmically, not as a product of two linear terms.

    rare_count and gap_score were the same quantity multiplied in separately,
    giving (1 + 0.5r)(1 + 0.3r) -- about 77x at r=20, unbounded above.
    """
    f = _make_fuzzer()
    f._edge_tracker.record_edges("wide", set(range(1, 21)))
    w = _weight(f, "wide")
    assert w < 5.0, f"rare bonus blew up: {w}"
    # Still meaningfully above a seed with a single rare edge.
    g = _make_fuzzer()
    g._edge_tracker.record_edges("narrow", {1})
    assert w > _weight(g, "narrow")


def test_crowded_coverage_is_penalised_not_rewarded():
    """A seed whose edges everyone else reaches loses energy.

    The old mean-hit term boosted seeds with high hit volume, i.e. hot loops,
    which inverts the rarity principle the schedule is built on.
    """
    f = _make_fuzzer()
    et = f._edge_tracker
    shared = {20, 21, 22}
    for i in range(int(CROWDED_EDGE_OWNERS) * 3):
        et.record_edges(f"c{i}", shared)
    assert et.edge_owner_count(20) > CROWDED_EDGE_OWNERS
    assert _weight(f, "c0") < 1.0


def test_empty_seed_returns_weight_unchanged():
    f = _make_fuzzer()
    assert _weight(f, "missing") == 1.0
