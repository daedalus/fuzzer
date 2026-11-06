"""Seed-pick hot path: fallback arms bound once, edge stats batched per pass.

1. ``_pick_unarbitrated`` rebuilt a 20-arm tuple (9 ``functools.partial``)
   and called every arm on each pick; disabled arms only read one attribute
   and returned None (1.9% of the fuzzgoat main thread, sampled). Only armed
   ones are called now, rebound when an arm attribute changes.
2. ``_EdgeVectors.stats`` ran three numpy gathers per seed per weight pass
   (1.3%). ``prepare`` now reduces every pass set at once.
"""

import random

import numpy as np
import pytest

from fuzzer_tool.services.seed_picker import RARE_EDGE_OWNERS, SeedPicker, _EdgeVectors
from tests.test_seed_picker_edge_vectors import _make_fuzzer, _random_campaign

_ARM_METHODS = (
    "_pick_entropy_kl_seed",
    "_pick_entropy_zscore_seed",
    "_pick_residual_seed",
    "_pick_strata_seed",
    "_pick_os_seed",
    "_pick_seed_drr_seed",
    "_pick_seed_round_robin_seed",
)


class _Arm:
    """Strategy stub: picks a fixed seed, counts calls."""

    def __init__(self, seed):
        self.seed, self.calls = seed, 0

    def select(self, corpus):
        self.calls += 1
        return self.seed


@pytest.fixture
def fuzzer():
    f = _make_fuzzer()
    f.corpus = [b"a", b"b", b"c"]
    for s in f.corpus:
        f.seed_meta[s] = {"fuzz_count": 1, "coverage_edges": 1, "added_at": 0.0, "momentum": 0.0}
    return f


def _count_arm_calls(monkeypatch) -> dict:
    calls = {"n": 0}
    for name in _ARM_METHODS:
        real = getattr(SeedPicker, name)

        def counting(self, *a, _real=real):
            calls["n"] += 1
            return _real(self, *a)

        monkeypatch.setattr(SeedPicker, name, counting)
    return calls


def test_disabled_arms_not_called(fuzzer, monkeypatch):
    """Falsification: with every arm off, no arm method runs per pick."""
    picker = SeedPicker(fuzzer)
    calls = _count_arm_calls(monkeypatch)
    for _ in range(5):
        assert picker._pick_unarbitrated() in fuzzer.corpus
    assert calls["n"] == 0


def test_arm_enabled_later_is_used(fuzzer):
    """Adversarial: an arm attached after the first pick takes the next one."""
    picker = SeedPicker(fuzzer)
    picker._pick_unarbitrated()
    fuzzer._entropy_kl = _Arm(b"c")
    assert picker._pick_unarbitrated() == b"c"

    fuzzer._entropy_kl = None  # detached again: back to the weighted picker
    fuzzer._entropy_zscore = _Arm(b"b")
    assert picker._pick_unarbitrated() == b"b"


def test_arm_priority_order_kept(fuzzer):
    """Adversarial: with two arms on, the earlier one in the chain wins."""
    picker = SeedPicker(fuzzer)
    late, early = _Arm(b"a"), _Arm(b"b")
    fuzzer._entropy_loo = late
    fuzzer._entropy_kl = early
    assert picker._pick_unarbitrated() == b"b"
    assert (early.calls, late.calls) == (1, 0)


def test_declining_arm_falls_through(fuzzer):
    """Adversarial: an armed arm returning None hands the pick on."""

    class _Decline(_Arm):
        def select(self, corpus):
            self.calls += 1
            return None

    picker = SeedPicker(fuzzer)
    first, second = _Decline(None), _Arm(b"a")
    fuzzer._entropy_kl, fuzzer._entropy_zscore = first, second
    assert picker._pick_unarbitrated() == b"a"
    assert (first.calls, second.calls) == (1, 1)


# ── batched edge stats ──────────────────────────────────────────────────


def _per_set(vectors: _EdgeVectors, edges: set) -> tuple[int, int, int]:
    """Oracle: the unbatched three-gather reduction for one set."""
    idx = vectors.indices(edges)
    n = vectors.owners[idx]
    overlap = int(vectors.recent[idx].sum()) if vectors.recent is not None else 0
    return vectors._rare(edges, idx, n), int(n.sum()), overlap


@pytest.mark.parametrize("trial", range(12))
def test_batched_stats_match_per_set(trial):
    rng = random.Random(trial)
    f, keys = _random_campaign(rng, trial)
    et = f._edge_tracker
    sets = [et.seed_edges[k] for k in keys]
    vectors = _EdgeVectors()
    vectors.prepare(sets, et._edge_owner_count, f._recent_seed_edges)

    want = [_per_set(vectors, s) for s in sets]
    assert [_per_set(vectors, s) for s in sets] == want  # control (Rule 46)
    got = [vectors.stats(s) for s in sets]
    assert got == want
    assert all(type(x) is int for row in got for x in row)


def test_set_grown_after_prepare_recomputes():
    """Adversarial: a set that changed since prepare is not served stale."""
    rng = random.Random(3)
    f, keys = _random_campaign(rng, 3)
    et = f._edge_tracker
    sets = [et.seed_edges[k] for k in keys]
    vectors = _EdgeVectors()
    vectors.prepare(sets, et._edge_owner_count, f._recent_seed_edges)

    s = sets[0]
    s.add(next(iter(sets[-1])))  # a known edge, so owners covers it
    assert vectors.stats(s) == _per_set(vectors, s)


def test_set_outside_pass_falls_back():
    """Adversarial: a set prepare never saw still gets exact stats."""
    rng = random.Random(4)
    f, keys = _random_campaign(rng, 4)
    et = f._edge_tracker
    sets = [et.seed_edges[k] for k in keys]
    vectors = _EdgeVectors()
    vectors.prepare(sets[1:], et._edge_owner_count, f._recent_seed_edges)
    assert vectors.stats(sets[0]) == _per_set(vectors, sets[0])


def test_batched_counts_are_exact_sums():
    """Falsification: hand-built owners give the expected integers."""
    a, b = {1, 2, 3}, {3, 4}
    owners = {1: 1, 2: 5, 3: 2, 4: 9}
    vectors = _EdgeVectors()
    vectors.prepare([a, b], owners, [b])
    rare = [sum(owners[e] <= RARE_EDGE_OWNERS for e in s) for s in (a, b)]
    recent = [sum(1 for e in s if e in b) for s in (a, b)]
    assert vectors.stats(a) == (rare[0], 8, recent[0])
    assert vectors.stats(b) == (rare[1], 11, recent[1])
    assert np.array_equal(vectors.owners[vectors.indices(a)], [owners[e] for e in a])


def test_pass_sets_need_no_per_set_gather(monkeypatch):
    """Falsification: after prepare, stats on a pass set does no indexing."""
    rng = random.Random(5)
    f, keys = _random_campaign(rng, 5)
    et = f._edge_tracker
    sets = [et.seed_edges[k] for k in keys]
    vectors = _EdgeVectors()
    vectors.prepare(sets, et._edge_owner_count, f._recent_seed_edges)

    def boom(_edges):
        raise AssertionError("per-set gather on a batched set")

    monkeypatch.setattr(vectors, "indices", boom)
    for s in sets:
        vectors.stats(s)
