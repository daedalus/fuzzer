"""Regression tests for incidence-based richness estimation.

good_turing_estimate() is a Chao2 estimator over the *incidence* spectrum:
Q_k counts edges reached by exactly k distinct seeds. It previously read the
bucketed execution hit counters in _global_edge_hits, which describe hit
volume, not how many corpus entries reach an edge.
"""

import math

from fuzzer_tool.core.edge_tracker import EdgeTracker


def _tracker(seed_edge_sets, hit_counts=None):
    et = EdgeTracker(map_size=4096)
    for key, edges in seed_edge_sets.items():
        et.record_edges(key, set(edges), hit_counts=hit_counts)
    return et


def test_hit_volume_does_not_change_the_spectrum():
    """Re-running the same seeds harder must not move Q1/Q2 or the estimate.

    Under the old abundance spectrum it did: the counters summed across
    executions, so replaying a seed migrated its edges out of the singleton
    bucket and the estimate collapsed for reasons unrelated to coverage.
    """
    sets = {"a": {1, 2, 3}, "b": {3, 4, 5}, "c": {5, 6}}
    quiet = _tracker(sets)
    loud = _tracker(sets, hit_counts=dict.fromkeys(range(1, 7), 900))
    for _ in range(20):  # replay every seed many times over
        for key, edges in sets.items():
            loud.record_edges(key, set(edges), hit_counts=dict.fromkeys(range(1, 7), 900))

    q = quiet.good_turing_estimate()
    hot = loud.good_turing_estimate()
    assert (q["n1"], q["n2"]) == (hot["n1"], hot["n2"])
    assert q["estimated_undiscovered"] == hot["estimated_undiscovered"]


def test_spectrum_counts_seeds_per_edge():
    et = _tracker({"a": {1, 2, 3}, "b": {2, 3}, "c": {3}})
    gt = et.good_turing_estimate()
    assert gt["m"] == 3
    assert gt["n1"] == 1  # edge 1
    assert gt["n2"] == 1  # edge 2
    assert gt["n"] == 3


def test_chao2_matches_the_closed_form():
    """Q2 >= 10 selects the classic estimator; check it against the formula."""
    sets = {}
    # 30 edges reached by exactly 2 seeds, 12 reached by exactly 1.
    for i in range(30):
        sets[f"p{i}"] = {i}
        sets[f"q{i}"] = {i}
    for i in range(12):
        sets[f"solo{i}"] = {1000 + i}
    et = _tracker(sets)
    gt = et.good_turing_estimate()
    assert (gt["n1"], gt["n2"]) == (12, 30)
    m, s_obs = gt["m"], 42
    expected = ((m - 1) / m) * (12 * 12) / (2 * 30)
    assert gt["estimated_undiscovered"] == int(expected)
    assert math.isclose(gt["saturation"], s_obs / (s_obs + expected), rel_tol=1e-6)


def test_bias_corrected_form_is_finite_without_doubletons():
    """Q2 = 0 used to force a raw-N1 fallback plus an arbitrary 0.5 damping."""
    et = _tracker({f"s{i}": {i} for i in range(8)})
    gt = et.good_turing_estimate()
    assert gt["n2"] == 0
    assert gt["estimated_undiscovered"] > 0
    assert math.isfinite(gt["ci_high"])


def test_confidence_interval_brackets_the_estimate():
    et = _tracker({"a": {1, 2, 3}, "b": {2, 3, 4}, "c": {3, 4, 5}, "d": {5, 6}})
    gt = et.good_turing_estimate()
    assert gt["ci_low"] <= gt["chao2"] <= gt["ci_high"]
    assert gt["ci_low"] >= gt["n"] - 1  # never below what we already saw
    assert gt["confidence"] in ("low", "medium", "high")


def test_discovery_probability_falls_as_coverage_consolidates():
    """1 - C is the chance the next seed reaches code nothing else reaches."""
    fresh = _tracker({f"s{i}": {i} for i in range(10)})  # all singletons
    settled = _tracker({f"s{i}": set(range(10)) for i in range(10)})  # all shared
    assert fresh.good_turing_estimate()["discovery_probability"] > 0.5
    assert settled.good_turing_estimate()["discovery_probability"] == 0.0
    assert settled.good_turing_estimate()["saturation"] == 1.0


def test_single_seed_yields_no_estimate():
    """One sampling unit makes every edge a singleton by construction."""
    gt = _tracker({"only": {1, 2, 3, 4, 5}}).good_turing_estimate()
    assert gt["m"] == 1
    assert gt["estimated_undiscovered"] == 0
    assert gt["confidence"] == "low"


def test_empty_tracker():
    gt = EdgeTracker(map_size=256).good_turing_estimate()
    assert gt["n"] == 0
    assert gt["estimated_undiscovered"] == 0
    assert gt["saturation"] == 0.0
