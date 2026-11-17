"""``native_scan.SortedIndex``: S+ tree lower_bound matches ``bisect_left``.

The oracle is ``bisect.bisect_left`` per query, independent of both the
kernel and ``np.searchsorted``; it is first run against a second copy of
itself (Hard Rule 46). Sizes straddle the tree's node (8 keys) and level
(9 children) boundaries; batch sizes straddle the numpy/kernel cutoff, the
16-query lockstep and the staging-buffer size.
"""

from __future__ import annotations

import bisect
import pickle

import numpy as np
import pytest

from fuzzer_tool.adapters import native_scan

I64 = np.iinfo(np.int64)

# Node = 8 keys, 9 children: 8 / 72 / 648 are full levels, +-1 forces padding.
SIZES = (1, 2, 7, 8, 9, 71, 72, 73, 647, 648, 649, 5000)
MIN = native_scan._SP_MIN_BATCH
STAGED = native_scan._SP_STAGED
BATCHES = (0, 1, 17, MIN - 1, MIN, MIN + 1, MIN + 15, MIN + 16, 1000, STAGED, STAGED + 1)

needs_kernel = pytest.mark.skipif(
    not native_scan.sorted_index(np.arange(1, dtype=np.int64)).uses_kernel,
    reason="no C compiler or no AVX2 for the S+ tree kernel",
)


def _oracle(keys: np.ndarray, q: np.ndarray) -> list[int]:
    """Per-query bisect_left over a Python list: no numpy, no kernel."""
    ref = keys.tolist()
    return [bisect.bisect_left(ref, x) for x in q.tolist()]


def _keys(rng, n: int) -> np.ndarray:
    return np.unique(rng.integers(-(1 << 40), 1 << 40, n * 3))[:n].astype(np.int64)


def _queries(rng, keys: np.ndarray, m: int) -> np.ndarray:
    """Half exact hits, half misses, including below-min and above-max."""
    hits = rng.choice(keys, m) if keys.size else np.zeros(m, np.int64)
    miss = rng.integers(-(1 << 41), 1 << 41, m)
    return np.where(rng.random(m) < 0.5, hits, miss).astype(np.int64)


def test_oracle_against_itself():
    rng = np.random.default_rng(0)
    keys = _keys(rng, 500)
    q = _queries(rng, keys, 300)
    assert _oracle(keys, q) == _oracle(keys.copy(), q.copy())


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("m", BATCHES)
def test_matches_bisect(n, m):
    rng = np.random.default_rng(n * 1009 + m)
    keys = _keys(rng, n)
    q = _queries(rng, keys, m)
    assert native_scan.sorted_index(keys).lower_bound(q).tolist() == _oracle(keys, q)


def test_fallback_matches_bisect(monkeypatch):
    """No kernel (no compiler, non-x86, no AVX2): numpy path, same answers."""
    monkeypatch.setattr(native_scan, "load", lambda: None)
    rng = np.random.default_rng(1)
    keys = _keys(rng, 649)
    q = _queries(rng, keys, 100)
    index = native_scan.sorted_index(keys)
    assert not index.uses_kernel
    assert index.lower_bound(q).tolist() == _oracle(keys, q)


def test_empty_keys():
    index = native_scan.sorted_index(np.empty(0, dtype=np.int64))
    q = np.array([I64.min, 0, I64.max], dtype=np.int64)
    assert index.lower_bound(q).tolist() == [0, 0, 0]


def test_adversarial_int64_extremes():
    """Keys at INT64_MIN/MAX collide with the tree's INT64_MAX padding."""
    keys = np.array([I64.min, I64.min + 1, -1, 0, 1, I64.max - 1, I64.max], dtype=np.int64)
    edge = [I64.min, I64.min + 1, -2, 0, 2, I64.max - 1, I64.max]
    q = np.array(edge * (MIN + 1), dtype=np.int64)  # 1351 = 84 x 16 + 7: extremes in the tail
    assert native_scan.sorted_index(keys).lower_bound(q).tolist() == _oracle(keys, q)


def test_adversarial_query_layouts():
    """int32, uint32 and strided queries are converted, not misread."""
    rng = np.random.default_rng(2)
    keys = _keys(rng, 300)
    base = _queries(rng, keys, 2 * MIN)
    index = native_scan.sorted_index(keys)
    strided = np.repeat(base, 2)[::2]
    assert index.lower_bound(strided).tolist() == _oracle(keys, base)

    small = np.arange(-MIN, MIN, dtype=np.int32)
    assert index.lower_bound(small).tolist() == _oracle(keys, small.astype(np.int64))

    unsigned = np.arange(0, 2 * MIN, dtype=np.uint32)
    assert index.lower_bound(unsigned).tolist() == _oracle(keys, unsigned.astype(np.int64))


def test_keys_are_copied():
    """Mutating the caller's array after building must not change answers."""
    keys = np.arange(0, 100, 2, dtype=np.int64)
    index = native_scan.sorted_index(keys)
    keys[:] = 10**9
    q = np.arange(-5, 2 * MIN, dtype=np.int64)
    assert index.lower_bound(q).tolist() == _oracle(np.arange(0, 100, 2), q)


@needs_kernel
def test_falsify_kernel_reads_the_tree():
    """Corrupting the tree changes the answers: the tests exercise the kernel."""
    keys = np.arange(0, 6480, 10, dtype=np.int64)
    q = np.arange(0, 6480, 7, dtype=np.int64)
    index = native_scan.sorted_index(keys)
    assert index.lower_bound(q).tolist() == _oracle(keys, q)

    index._tree[:] = I64.max  # every node now says "all keys >= q"
    assert index.lower_bound(q).tolist() != _oracle(keys, q)


@needs_kernel
def test_small_batches_skip_the_kernel():
    """Below the cutoff the ctypes call never happens; at the cutoff it does."""
    index = native_scan.sorted_index(np.arange(1000, dtype=np.int64))

    def boom(*_args):
        raise AssertionError("kernel called")

    index._fn = boom
    assert index.lower_bound(np.arange(MIN - 1, dtype=np.int64)).tolist() == list(range(MIN - 1))
    with pytest.raises(AssertionError, match="kernel called"):
        index.lower_bound(np.arange(MIN, dtype=np.int64))


@needs_kernel
def test_pickle_refuses():
    """Held addresses must never be restored into another process."""
    with pytest.raises((TypeError, ValueError, pickle.PicklingError)):
        pickle.dumps(native_scan.sorted_index(np.arange(10, dtype=np.int64)))


# ── wiring: the two static-key lookup sites ─────────────────────────────


def _seed_hits(rng, seeds: int, edges: int) -> dict[str, dict[int, int]]:
    """Random seed -> {edge: count}; shared edges make multi-edge classes."""
    pool = rng.integers(1, 1 << 32, edges)
    return {
        f"s{i}": {int(e): 1 for e in rng.choice(pool, rng.integers(5, 60), replace=False)}
        for i in range(seeds)
    }


def test_canonicalizer_class_array_wired():
    from fuzzer_tool.core.scheduler_substrate import EdgeCanonicalizer

    rng = np.random.default_rng(3)
    canon = EdgeCanonicalizer()
    canon.refit(_seed_hits(rng, 40, 400))
    seen = list(canon._representative)
    probe = np.array(seen + [0, 7, 1 << 40, -5] + seen[::3], dtype=np.int64)

    want = [canon.class_of(int(e)) for e in probe]  # dict lookup, no search
    assert canon.class_array(probe).tolist() == want
    assert canon._index.uses_kernel == native_scan.sorted_index(probe).uses_kernel


def test_clusters_class_index_wired():
    from fuzzer_tool.core.edge_clusters import EdgeClusters

    rng = np.random.default_rng(4)
    clusters = EdgeClusters()
    clusters.refit({k: set(v) for k, v in _seed_hits(rng, 40, 400).items()})
    heads = clusters._heads.tolist()
    at = {h: i for i, h in enumerate(heads)}
    edges = list(clusters._canon._representative) + [0, 3, 1 << 40]
    probe = np.array(edges, dtype=np.int64)

    want = [at.get(clusters.class_of(e), -1) for e in edges]
    assert clusters.class_index(probe).tolist() == want
    assert clusters._head_index.uses_kernel == native_scan.sorted_index(probe).uses_kernel


def test_wired_sites_before_any_refit():
    from fuzzer_tool.core.edge_clusters import EdgeClusters
    from fuzzer_tool.core.scheduler_substrate import EdgeCanonicalizer

    probe = np.array([1, 2, 3], dtype=np.int64)
    assert EdgeCanonicalizer().class_array(probe).tolist() == [1, 2, 3]
    assert EdgeClusters().class_index(probe).tolist() == [-1, -1, -1]
