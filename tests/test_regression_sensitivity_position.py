"""Regression test: ByteSensitivityTracker.get_weighted_position bisect path.

The O(buf_len) cumulative walk was replaced with a cached array("d")
cumulative-sums + bisect_left (the mi.weighted_position pattern). The bisect
path must be EXACTLY equivalent to the legacy walk — same rng draw position,
same "first i with cumulative >= r" semantics — including edge cases (zero
weights, r == 0, r == total, len(scores) < buf_len) and the negative-score
fallback (JSON load path can hold garbage). Cache invalidation is pinned too.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from fuzzer_tool.core.sensitivity import ByteSensitivityTracker


def _legacy_walk(scores, buf_len, r):
    """Verbatim legacy get_weighted_position walk (independent reference)."""
    if len(scores) < buf_len:
        return None
    total = sum(scores[:buf_len])
    if total <= 0:
        return None
    cumulative = 0.0
    for i in range(buf_len):
        cumulative += scores[i]
        if r <= cumulative:
            return i
    return buf_len - 1


def _draw(tracker, seed_key, buf_len, r_val):
    """Run get_weighted_position with a fixed random.random() return."""
    with patch("fuzzer_tool.core.sensitivity.random.random", return_value=r_val):
        return tracker.get_weighted_position(seed_key, buf_len)


def _make_tracker(scores):
    t = ByteSensitivityTracker()
    t._sensitivity[b"seed"] = scores
    return t


SCORE_LISTS = [
    [0.0, 0.0, 0.0, 0.0, 0.0],  # all zero -> total <= 0 -> None
    [1.0],  # single element
    [0.2, 0.8],  # len == buf_len
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # uniform
    [1.0, 0.0, 0.0, 1.0, 0.0],  # zeros inside
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],  # non-decreasing, ties
    [0.5, -0.2, 0.8],  # negative (garbage JSON) -> fallback to the walk
]


def test_fixed_random_matches_legacy():
    rng_values = [0.0, 1e-300, 0.499, 1.0 - 1e-16]
    for scores in SCORE_LISTS:
        t = _make_tracker(scores)
        for buf_len in (1, len(scores), len(scores) + 3):
            total = sum(scores[:buf_len])
            for rv in rng_values:
                got = _draw(t, b"seed", buf_len, rv)
                exp = _legacy_walk(scores, buf_len, rv * total) if total > 0 else None
                assert got == exp, (scores, buf_len, rv, got, exp)
                assert got is None or 0 <= got < buf_len


def test_randomized_matches_manual_oracle():
    """500 randomized trials; oracle = manual accumulation (independent)."""
    rng = random.Random(20260803)
    for _ in range(500):
        n = rng.randint(1, 60)
        scores = [rng.random() * 0.5 for _ in range(n)]
        if rng.random() < 0.3:
            scores[rng.randrange(n)] = 0.0  # inject zeros/ties
        buf_len = rng.randint(1, n)
        total = sum(scores[:buf_len])
        if total <= 0:
            continue
        r_val = rng.random()
        t = _make_tracker(scores)
        got = _draw(t, b"seed", buf_len, r_val)
        # Oracle: first i < buf_len with manual cumulative >= r.
        r = r_val * total
        cum = 0.0
        exp = buf_len - 1
        for i in range(buf_len):
            cum += scores[i]
            if r <= cum:
                exp = i
                break
        assert got == exp, (scores, buf_len, r_val, got, exp)


def test_cache_built_and_reused():
    t = _make_tracker([0.2, 0.3, 0.5])
    _draw(t, b"seed", 3, 0.5)
    assert b"seed" in t._cum_cache
    assert t._cum_cache[b"seed"].tolist() == pytest.approx([0.2, 0.5, 1.0])
    # A second call reuses the cache (no rebuild) and matches the walk.
    exp = _legacy_walk([0.2, 0.3, 0.5], 3, 0.5)
    assert _draw(t, b"seed", 3, 0.5) == exp


def test_cache_survives_idempotent_reanalysis():
    """analyze_seed is idempotent (early return) so the cache stays valid."""
    t = ByteSensitivityTracker(sample_rate=1.0)
    t.analyze_seed(b"abcd", {1}, lambda data: {1} if data == b"abcd" else {2})
    assert _draw(t, b"abcd", 4, 0.5) is not None  # cache built
    assert b"abcd" in t._cum_cache
    t.analyze_seed(b"abcd", {1}, lambda data: {1} if data == b"abcd" else {2})
    assert b"abcd" in t._cum_cache  # scores unchanged, cache not dropped


def test_cache_invalidated_on_eviction():
    t = ByteSensitivityTracker(max_seeds=1, sample_rate=1.0)
    t.analyze_seed(b"a", {1}, lambda data: {1})
    t.analyze_seed(b"b", {1}, lambda data: {1})  # evicts "a"
    assert b"a" not in t._cum_cache


def test_cache_cleared_on_load():
    t = _make_tracker([0.2, 0.3, 0.5])
    _draw(t, b"seed", 3, 0.5)
    assert t._cum_cache
    t.load({"sensitivity": {b"other".hex(): [0.1, 0.2]}})
    assert not t._cum_cache
    assert t._sensitivity == {b"other": [0.1, 0.2]}
