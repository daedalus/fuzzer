"""Regression: EdgeTracker.shannon_entropy_seed is cached per seed.

ffmpeg campaign, 6,000 rounds: the seed-weight refresh recomputed every
seed's hit-count entropy each time (325k calls, 14.6 s) although a seed's
counts only change when record_edges() touches it.
"""

from __future__ import annotations

import math
import time

from fuzzer_tool.core.edge_tracker import EdgeTracker


def _entropy(hc: dict[int, int]) -> float:
    """Shannon entropy from the probability form, independent of the code."""
    total = sum(hc.values())
    return -sum((c / total) * math.log2(c / total) for c in hc.values() if c)


def test_regression_entropy_seed_cache_tracks_record_edges():
    et = EdgeTracker()
    et.record_edges("a", {1, 2, 3}, hit_counts={1: 1, 2: 2, 3: 8})
    first = et.shannon_entropy_seed("a")
    assert math.isclose(first, _entropy({1: 1, 2: 2, 3: 8}), abs_tol=1e-9)

    et.record_edges("a", {1, 2, 3}, hit_counts={1: 4, 2: 4, 3: 4})
    assert math.isclose(et.shannon_entropy_seed("a"), math.log2(3), abs_tol=1e-9)


def test_regression_entropy_seed_cache_sees_replaced_dict():
    """Falsification: a caller swapping the dict must not read a stale value."""
    et = EdgeTracker()
    et.seed_hit_counts["a"] = {10: 5, 20: 5}
    assert math.isclose(et.shannon_entropy_seed("a"), 1.0, abs_tol=1e-9)
    et.seed_hit_counts["a"] = {10: 5}
    assert et.shannon_entropy_seed("a") == 0.0


def test_regression_entropy_seed_cache_adversarial():
    """Missing key, emptied dict, grown dict, pruned seed."""
    et = EdgeTracker(max_tracked_seeds=1000)
    assert et.shannon_entropy_seed("nope") == 0.0

    et.seed_hit_counts["a"] = {1: 3, 2: 3}
    assert math.isclose(et.shannon_entropy_seed("a"), 1.0, abs_tol=1e-9)
    et.seed_hit_counts["a"][3] = 3
    assert math.isclose(et.shannon_entropy_seed("a"), math.log2(3), abs_tol=1e-9)
    et.seed_hit_counts["a"].clear()
    assert et.shannon_entropy_seed("a") == 0.0

    for i in range(30):
        et.record_edges(f"s{i}", {i, 999}, hit_counts={i: 2, 999: 2})
        et.shannon_entropy_seed(f"s{i}")
    et.max_tracked_seeds = 10
    et._maybe_prune()
    assert set(et._entropy_cache) <= set(et.seed_hit_counts)


def test_regression_entropy_seed_cache_speed():
    et = EdgeTracker()
    for s in range(300):
        edges = set(range(s, s + 2400))
        et.record_edges(f"s{s}", edges, hit_counts={e: 1 + (e * 7) % 13 for e in edges})
    keys = list(et.seed_hit_counts)

    t0 = time.perf_counter()
    for k in keys:
        et.shannon_entropy_seed(k)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10):
        for k in keys:
            et.shannon_entropy_seed(k)
    warm = (time.perf_counter() - t0) / 10

    assert warm * 5 < cold, f"warm {warm:.4f}s vs cold {cold:.4f}s"
