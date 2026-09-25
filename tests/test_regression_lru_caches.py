"""Regression: bounded caches wiped every entry on overflow (K3).

Five caches hit their cap and called ``clear()``: one insert past the cap
threw away the whole working set, including the entry just read. Each now
evicts only its least-recently-used entry.

Every site test: fill to capacity, re-read the oldest key, insert one more.
LRU keeps the re-read key and drops the next-oldest; clear-on-full keeps
neither.
"""

import os
import pickle

import pytest

from fuzzer_tool.core.lru import LRUCache
from fuzzer_tool.core.mutations.fractal_voronoi import FractalVoronoiMutator
from fuzzer_tool.core.path_constraints import BranchRecord, PathConstraintSolver

# ── LRUCache ──────────────────────────────────────────────────────────


def test_lru_evicts_least_recent_only():
    cache = LRUCache(3)
    for k in "abc":
        cache[k] = k.upper()

    assert cache.get("a") == "A"
    cache["d"] = "D"

    assert list(cache) == ["c", "a", "d"]


def test_lru_overwrite_refreshes_without_growth():
    cache = LRUCache(2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"] = 3
    cache["c"] = 4

    assert dict(cache) == {"a": 3, "c": 4}


def test_lru_get_miss_returns_default_and_keeps_order():
    cache = LRUCache(2)
    cache["a"] = 1

    assert cache.get("zz") is None
    assert cache.get("zz", 7) == 7
    assert list(cache) == ["a"]


def test_lru_on_evict_sees_each_dropped_key():
    dropped = []
    cache = LRUCache(2, on_evict=dropped.append)
    for k in range(5):
        cache[k] = k

    assert dropped == [0, 1, 2]


def test_lru_rejects_non_positive_capacity():
    """Adversarial: capacity 0 would evict every insert — a cache that
    never holds anything and silently costs an OrderedDict per call."""
    with pytest.raises(ValueError):
        LRUCache(0)
    with pytest.raises(ValueError):
        LRUCache(-1)


def test_lru_pickle_round_trip_keeps_capacity_and_order():
    """Adversarial: OrderedDict's default reduce calls cls() with no
    capacity, which would fail or unbound the cache on unpickle."""
    cache = LRUCache(2)
    cache["a"] = 1
    cache["b"] = 2

    clone = pickle.loads(pickle.dumps(cache))
    clone["c"] = 3

    assert list(clone) == ["b", "c"]


def test_lru_empty_equals_empty_dict():
    assert LRUCache(4) == {}


# ── sites ─────────────────────────────────────────────────────────────


def test_regression_region_cache_keeps_recent_seed():
    from fuzzer_tool.services.operators import _REGION_CACHE_MAX, OperatorEngine
    from tests.test_regression_region_profile import _MockFuzzer

    engine = OperatorEngine(_MockFuzzer())
    seeds = [os.urandom(8192) for _ in range(_REGION_CACHE_MAX + 1)]
    for s in seeds[:-1]:
        engine.region_weights(s)
    first = engine.region_weights(seeds[0])

    engine.region_weights(seeds[-1])

    assert len(engine._region_cache) == _REGION_CACHE_MAX
    assert engine.region_weights(seeds[0]) is first


def test_regression_region_liveness_follows_eviction():
    """Liveness must drop exactly the evicted seed, not every seed."""
    from fuzzer_tool.services.operators import _REGION_CACHE_MAX, OperatorEngine
    from tests.test_regression_region_profile import _MockFuzzer

    engine = OperatorEngine(_MockFuzzer())
    seeds = [os.urandom(8192) for _ in range(_REGION_CACHE_MAX + 1)]
    for s in seeds[:-1]:
        engine.region_weights(s)
    keys = list(engine._region_cache)
    for k in keys[:2]:
        engine._region_liveness[k] = [None]

    engine.region_weights(seeds[-1])

    assert keys[0] not in engine._region_liveness
    assert keys[1] in engine._region_liveness


def test_regression_rq_cache_keeps_recent_pair(monkeypatch):
    import fuzzer_tool.core.rq_encodings as rq

    cap = 4
    monkeypatch.setattr(rq, "_rq_mutations_cache", LRUCache(cap))
    data = b"\x01\x00\x00\x00" * 4

    def run(i):
        rq.generate_mutations(i.to_bytes(4, "little"), b"ZZZZ", 32, "CMP", data)

    for i in range(cap):
        run(i)
    run(0)
    run(cap)

    keys = [k[2] for k in rq._rq_mutations_cache]
    assert len(keys) == cap
    assert (0).to_bytes(4, "little") in keys
    assert (1).to_bytes(4, "little") not in keys


def test_regression_rq_cache_is_lru():
    """Module-level cache itself must be the bounded LRU, not a dict."""
    import fuzzer_tool.core.rq_encodings as rq

    assert isinstance(rq._rq_mutations_cache, LRUCache)


def test_regression_attempted_keeps_recent_branch(monkeypatch):
    import fuzzer_tool.core.path_constraints as pc

    cap = 4
    monkeypatch.setattr(pc, "MAX_ATTEMPTED", cap)
    solver = PathConstraintSolver()
    data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"

    def rec(i):
        return BranchRecord(
            (0x10).to_bytes(4, "little"), (i + 0x100).to_bytes(4, "little"), -1, 4, i
        )

    for i in range(cap + 1):
        solver.negate(rec(i), data)

    assert len(solver._attempted) == cap
    assert [r.key for r in solver.frontier([rec(0)], data)] == [rec(0).key]
    assert solver.frontier([rec(1)], data) == []


def test_regression_ppmd_cache_keeps_recent_seed(monkeypatch):
    import fuzzer_tool.core.analyzers.analyzer_corpus_compression as cc_mod

    cc = cc_mod.CorpusCompressor()
    if not cc.enabled:
        pytest.skip("pyppmd not installed")
    cap = 4
    monkeypatch.setattr(cc_mod, "PPMD_CACHE_MAX", cap)
    cc = cc_mod.CorpusCompressor()
    seeds = [os.urandom(512) for _ in range(cap + 1)]
    for s in seeds[:-1]:
        cc.compute_seed_ratio(s)
    cc.compute_seed_ratio(seeds[0])

    cc.compute_seed_ratio(seeds[-1])

    assert len(cc._seed_ratios) == cap
    assert cc_mod._ppmd_cache_key(seeds[0]) in cc._seed_ratios
    assert cc_mod._ppmd_cache_key(seeds[1]) not in cc._seed_ratios


def test_regression_plan_cache_keeps_recent_length():
    m = FractalVoronoiMutator()
    cap = m._PLAN_CACHE_MAX
    lengths = list(range(64, 64 + cap + 1))
    for n in lengths[:-1]:
        m._plan(16, n)
    first = m._plan(16, lengths[0])

    m._plan(16, lengths[-1])

    assert len(m._plan_cache) == cap
    assert m._plan(16, lengths[0]) is first
