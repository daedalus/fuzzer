"""Tests for corpus compression (PPMD novelty scoring)."""

import pytest
from fuzzer_tool.core.corpus_compression import CorpusCompressor, PPMD_AVAILABLE


@pytest.mark.skipif(not PPMD_AVAILABLE, reason="pyppmd not installed")
class TestPPMDAvailable:
    def test_compression_ratio_repetitive(self):
        cc = CorpusCompressor()
        data = b"AAAA" * 100
        ratio = cc.compute_seed_ratio(data)
        assert ratio < 0.1  # highly compressible

    def test_compression_ratio_random(self):
        cc = CorpusCompressor()
        import random
        data = bytes(random.randint(0, 255) for _ in range(1000))
        ratio = cc.compute_seed_ratio(data)
        assert ratio > 0.5  # less compressible

    def test_novelty_score_repetitive(self):
        cc = CorpusCompressor()
        data = b"AAAA" * 100
        novelty = cc.compute_seed_novelty(data)
        assert novelty < 0.2  # low novelty (redundant)

    def test_novelty_score_random(self):
        cc = CorpusCompressor()
        import random
        data = bytes(random.randint(0, 255) for _ in range(1000))
        novelty = cc.compute_seed_novelty(data)
        assert novelty > 0.3  # higher novelty (diverse)

    def test_empty_data(self):
        cc = CorpusCompressor()
        assert cc.compute_seed_ratio(b"") == 1.0
        assert cc.compute_seed_novelty(b"") == 1.0

    def test_corpus_stats(self):
        cc = CorpusCompressor()
        corpus = [b"A" * 100, b"B" * 100, b"C" * 100]
        stats = cc.compute_corpus_stats(corpus)
        assert stats["mean_ratio"] < 1.0
        assert stats["total_raw"] == 300
        assert stats["corpus_ratio"] < 1.0

    def test_should_prune(self):
        cc = CorpusCompressor()
        # Repetitive data should be prunable
        assert cc.should_prune(b"A" * 100, threshold=0.5)
        # Random data should not be prunable
        import random
        assert not cc.should_prune(bytes(random.randint(0, 255) for _ in range(1000)), threshold=0.5)

    def test_rank_seeds(self):
        cc = CorpusCompressor()
        import random
        corpus = [
            b"A" * 100,  # low novelty
            bytes(random.randint(0, 255) for _ in range(1000)),  # high novelty
            b"B" * 100,  # low novelty
        ]
        ranked = cc.rank_seeds(corpus)
        assert len(ranked) == 3
        # Random data should rank higher (more novel) than repetitive
        assert ranked[0][0] == 1  # index 1 = random data


class TestPPMDUnavailable:
    def test_disabled_when_not_installed(self):
        if PPMD_AVAILABLE:
            pytest.skip("pyppmd is installed")
        cc = CorpusCompressor(enabled=True)
        assert not cc.enabled
        assert cc.compute_seed_ratio(b"test") == 1.0
        assert cc.compute_seed_novelty(b"test") == 1.0
        assert not cc.should_prune(b"test")

    def test_explicitly_disabled(self):
        cc = CorpusCompressor(enabled=False)
        assert not cc.enabled
        assert cc.compute_seed_ratio(b"test") == 1.0
