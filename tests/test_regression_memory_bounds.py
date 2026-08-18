"""Regression tests for unbounded memory growth in the new mutators.

Both defects here are the same mistake: a cache bounded by *entry count*
when the entries are large, or not bounded at all. On a libpng target this
took RSS from 47 MB to 942 MB.
"""

import zlib

import pytest

import fuzzer_tool.core.mutations.recompress as R
from fuzzer_tool.core.path_constraints import (
    MAX_ATTEMPTED,
    BranchRecord,
    PathConstraintSolver,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    R._cache_reset()
    yield
    R._cache_reset()


class TestInflateCacheIsByteBounded:
    """The cache was capped at 256 entries, but each entry can hold up to
    _MAX_INFLATE (4 MiB) of plaintext — a 1 GB ceiling. Counting entries is
    not a memory bound when the entries are unbounded."""

    def test_large_streams_do_not_accumulate(self):
        for i in range(60):
            R.recompress_zlib(zlib.compress(bytes([i % 251]) * (2 << 20)), max_len=4096)
        assert R.cache_stats()["bytes"] <= R._CACHE_MAX_BYTES

    def test_oversized_entries_are_not_cached_at_all(self):
        """One huge entry would otherwise evict every useful small one."""
        blob = zlib.compress(b"A" * (R._CACHE_MAX_ENTRY * 2))
        R.recompress_zlib(blob, max_len=4096)
        assert R.cache_stats()["entries"] == 0

    def test_small_entries_are_still_cached(self):
        """The cache must keep working — it exists to skip re-inflation."""
        for i in range(50):
            R.recompress_zlib(zlib.compress(b"payload %d" % i * 20), max_len=4096)
        assert R.cache_stats()["entries"] == 50

    def test_byte_budget_is_enforced_across_mixed_sizes(self):
        for i in range(40):
            size = (1 << 10) if i % 2 else (1 << 18)
            R.recompress_zlib(zlib.compress(bytes([i % 251]) * size), max_len=4096)
        assert R.cache_stats()["bytes"] <= R._CACHE_MAX_BYTES

    def test_eviction_is_oldest_first(self):
        payloads = [b"X" * (R._CACHE_MAX_ENTRY - 1) for _ in range(3)]
        keys = []
        for i, p in enumerate(payloads):
            blob = zlib.compress(p + bytes([i]))
            R.inflate_zlib(blob)
            keys.append((15, hash(blob)))
        # Budget is 8 MiB and each entry is ~256 KiB, so all three fit;
        # the point is that insertion order is preserved for eviction.
        assert list(R._inflate_cache)[0] == keys[0]

    def test_reset_clears_the_byte_counter_too(self):
        """A counter left non-zero after a clear would shrink the cache to
        nothing over time."""
        R.recompress_zlib(zlib.compress(b"data" * 100), max_len=4096)
        R._cache_reset()
        assert R.cache_stats() == {"entries": 0, "bytes": 0}

    def test_cache_still_returns_correct_plaintext(self):
        """Bounding must not corrupt what is served."""
        payload = b"the exact plaintext" * 40
        blob = zlib.compress(payload)
        assert R.inflate_zlib(blob) == payload
        assert R.inflate_zlib(blob) == payload  # second call hits the cache

    def test_zlib_and_gzip_entries_stay_separate(self):
        """The wbits key must survive the rewrite."""
        payload = b"shared payload" * 30
        R.inflate_zlib(zlib.compress(payload))
        assert R.inflate_gzip(zlib.compress(payload)) is None


class TestAttemptedSetIsBounded:
    """PathConstraintSolver._attempted grew for the life of the run — each
    entry holds operand bytes, ~200 bytes per distinct branch."""

    def test_set_stays_under_the_cap(self):
        solver = PathConstraintSolver()
        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        for i in range(MAX_ATTEMPTED + 20_000):
            rec = BranchRecord(
                (0x10).to_bytes(4, "little"),
                ((i * 7) & 0xFFFFFFFF).to_bytes(4, "little"),
                -1,
                4,
                i,
            )
            solver.negate(rec, data)
        assert len(solver._attempted) <= MAX_ATTEMPTED

    def test_frontier_still_suppresses_repeats_below_the_cap(self):
        """Bounding must not break the deduplication it exists for."""
        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        rec = BranchRecord(
            (0x10).to_bytes(4, "little"), (0x41424344).to_bytes(4, "little"), -1, 4, 1
        )
        solver = PathConstraintSolver()
        solver.negate(rec, data)
        assert solver.frontier([rec], data) == []


class TestFuzzerSeedKeyDoesNotPinBytes:
    """_seed_key_cache stored the full bytes object as a dict key, so every
    unique mutation was pinned in memory until the next minimization. During
    long stalls this grew RSS by GBs."""

    def test_seed_key_does_not_cache_full_bytes(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        f = Fuzzer.__new__(Fuzzer)
        f._corpus_manager = type("CM", (), {"seed_key": lambda s, d: "k"})()
        data = b"ABCDEFGH"
        key = f._seed_key(data)
        assert key == "k"
        assert not hasattr(f, "_seed_key_cache")

    def test_seed_key_deterministic(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        f = Fuzzer.__new__(Fuzzer)
        f._corpus_manager = type("CM", (), {"seed_key": lambda s, d: d[:4].hex()})()
        assert f._seed_key(b"test") == f._seed_key(b"test")
        assert f._seed_key(b"aaa") != f._seed_key(b"bbb")


class TestEdgeTrackerPrunesAllPerSeedMetrics:
    """_maybe_prune() dropped seed_edges but left seed_stack_depth,
    seed_path_hash, and the hw_* dicts to grow without bound."""

    def test_prune_removes_per_seed_metric_dicts(self):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        et = EdgeTracker(map_size=65536, max_tracked_seeds=2)
        for i in range(5):
            et.record_edges(
                seed_key=f"seed_{i}",
                hit_edges={i},
                stack_depth=100 + i,
                path_hash=0x1000 + i,
                hw_instructions=1000 + i,
                hw_branches=100 + i,
                hw_branch_misses=10 + i,
            )
        assert len(et.seed_edges) == 2
        # Only the newest 2 seeds should remain in the per-seed metric dicts
        assert len(et.seed_stack_depth) == 2
        assert len(et.seed_path_hash) == 2
        assert len(et.seed_hw_instructions) == 2
        assert len(et.seed_hw_branches) == 2
        assert len(et.seed_hw_branch_misses) == 2
        # The pruned seeds' entries must be gone
        for i in range(3):
            assert f"seed_{i}" not in et.seed_stack_depth
            assert f"seed_{i}" not in et.seed_path_hash
            assert f"seed_{i}" not in et.seed_hw_instructions
            assert f"seed_{i}" not in et.seed_hw_branches
            assert f"seed_{i}" not in et.seed_hw_branch_misses
