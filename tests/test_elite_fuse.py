"""Tests for the elite_fuse operator: fuses the two highest-coverage
corpus seeds into a hybrid third seed.
"""

from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine


class _SeedMeta(dict):
    """dict subclass so seed_meta.get(seed, {}) behaves like the real thing."""


def _fuzzer_with_corpus(corpus_and_coverage):
    """Build a minimal fuzzer-like object with a real corpus + seed_meta.

    corpus_and_coverage: list of (seed_bytes, coverage_edges) tuples.
    """

    class MinimalFuzzer:
        pass

    f = MinimalFuzzer()
    f.corpus = [seed for seed, _ in corpus_and_coverage]
    f.seed_meta = _SeedMeta({seed: {"coverage_edges": cov} for seed, cov in corpus_and_coverage})
    f.max_len = 65536
    f._rand_pool = RandPool(seed=1234)
    return f


class TestEliteSeeds:
    def test_empty_corpus_returns_empty(self):
        engine = OperatorEngine(_fuzzer_with_corpus([]))
        assert engine.elite_seeds() == []

    def test_single_seed_returns_empty(self):
        engine = OperatorEngine(_fuzzer_with_corpus([(b"AAAA", 10)]))
        assert engine.elite_seeds() == []

    def test_no_seed_meta_falls_back_to_tied_ranking(self):
        f = _fuzzer_with_corpus([(b"AAAA", 10), (b"BBBB", 20)])
        f.seed_meta = _SeedMeta()  # wiped after construction
        engine = OperatorEngine(f)
        # No coverage info to discriminate on, but there are still >= 2
        # corpus seeds -- the pool must not go empty just because
        # seed_meta hasn't been populated yet.
        pool = engine.elite_seeds()
        assert len(pool) == 2
        assert set(pool) == {b"AAAA", b"BBBB"}

    def test_ranks_by_coverage_descending(self):
        corpus = [
            (b"low_coverage_seed_1", 5),
            (b"high_coverage_seed_1", 500),
            (b"mid_coverage_seed_1", 50),
        ]
        engine = OperatorEngine(_fuzzer_with_corpus(corpus))
        pool = engine.elite_seeds()
        assert pool[0] == b"high_coverage_seed_1"
        assert pool[-1] == b"low_coverage_seed_1"

    def test_bytearray_corpus_entries_do_not_crash(self):
        """Real Fuzzer.corpus is list[bytes], but some callers (and the
        no-op sweep test) populate it with bytearray. seed_meta is keyed
        by bytes, so the ranking lookup must normalize rather than raise
        on an unhashable bytearray key."""
        f = _fuzzer_with_corpus([(b"AAAA", 1), (b"BBBB", 99)])
        f.corpus = [bytearray(s) for s in f.corpus]
        engine = OperatorEngine(f)
        pool = engine.elite_seeds()
        assert pool[0] == bytearray(b"BBBB")
        result = engine._op_elite_fuse(bytearray(b"BBBB"), 0, b"")
        assert result is not None

    def test_pool_capped_at_pool_size(self):
        from fuzzer_tool.services.operators import _ELITE_FUSE_POOL_SIZE

        corpus = [(f"seed_{i}".encode() * 4, i) for i in range(50)]
        engine = OperatorEngine(_fuzzer_with_corpus(corpus))
        pool = engine.elite_seeds()
        assert len(pool) == _ELITE_FUSE_POOL_SIZE

    def test_cache_rebuilds_only_on_corpus_growth(self):
        corpus = [(b"seed_aaaa", 5), (b"seed_bbbb", 500)]
        f = _fuzzer_with_corpus(corpus)
        engine = OperatorEngine(f)
        pool1 = engine.elite_seeds()
        # Mutate coverage in place without changing corpus length: cache
        # should NOT pick this up (same tradeoff as corpus_invariants()).
        f.seed_meta[b"seed_aaaa"]["coverage_edges"] = 9999
        pool2 = engine.elite_seeds()
        assert pool1 == pool2
        assert pool2[0] == b"seed_bbbb"  # stale ranking preserved
        # Now grow the corpus -- cache must rebuild.
        f.corpus.append(b"seed_cccc")
        f.seed_meta[b"seed_cccc"] = {"coverage_edges": 1}
        pool3 = engine.elite_seeds()
        assert pool3[0] == b"seed_aaaa"  # 9999 now visible post-rebuild


class TestOpEliteFuse:
    def test_returns_none_with_fewer_than_two_seeds(self):
        engine = OperatorEngine(_fuzzer_with_corpus([(b"AAAA", 10)]))
        assert engine._op_elite_fuse(bytearray(b"AAAA"), 0, b"AAAA") is None

    def test_returns_none_with_empty_corpus(self):
        engine = OperatorEngine(_fuzzer_with_corpus([]))
        assert engine._op_elite_fuse(bytearray(b""), 0, b"") is None

    def test_fuses_the_two_highest_coverage_seeds(self):
        corpus = [
            (b"AAAAAAAAAAAAAAAA", 1),  # decoy, low coverage
            (b"BBBBBBBBBBBBBBBB", 900),  # elite parent 1
            (b"CCCCCCCCCCCCCCCC", 1),  # decoy, low coverage
            (b"DDDDDDDDDDDDDDDD", 800),  # elite parent 2
        ]
        f = _fuzzer_with_corpus(corpus)
        engine = OperatorEngine(f)
        # Pin the elite pool directly to the two intended parents so this
        # test isolates fuse mechanics from ranking (covered separately in
        # TestEliteSeeds) -- with only 4 seeds total, the real pool (cap 8)
        # would include the decoys too, which isn't what's under test here.
        engine._elite_pool = [b"BBBBBBBBBBBBBBBB", b"DDDDDDDDDDDDDDDD"]
        engine._elite_pool_corpus_len = len(f.corpus)
        seen_bytesets = set()
        for _ in range(50):
            result = engine._op_elite_fuse(bytearray(b"BBBBBBBBBBBBBBBB"), 0, b"")
            assert result is not None
            result_bytes = set(result)
            seen_bytesets.add(frozenset(result_bytes))
            # Every produced byte must come from one of the two elite
            # parents (B or D) -- the decoys (A, C) must never contribute.
            assert result_bytes <= {ord("B"), ord("D")}
        # Over many trials we should see genuine fusion (both letters
        # present in at least one output), not always just one parent.
        assert any(len(bs) == 2 for bs in seen_bytesets)

    def test_respects_max_len(self):
        corpus = [
            (b"E" * 200, 500),
            (b"F" * 200, 400),
        ]
        f = _fuzzer_with_corpus(corpus)
        f.max_len = 50
        engine = OperatorEngine(f)
        for _ in range(20):
            result = engine._op_elite_fuse(bytearray(b"E" * 200), 0, b"")
            assert result is None or len(result) <= 50

    def test_registered_in_structural_category(self):
        assert "elite_fuse" in REGISTRY.categories()["structural"]
        assert "elite_fuse" in REGISTRY.names()

    def test_dispatch_resolves_to_handler(self):
        engine = OperatorEngine(_fuzzer_with_corpus([(b"AAAA", 1), (b"BBBB", 2)]))
        table = engine.build_dispatch()
        assert table["elite_fuse"] == engine._op_elite_fuse
