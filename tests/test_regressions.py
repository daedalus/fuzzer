"""Regression tests for bugs found during code review.

Each test targets a specific bug class to prevent recurrence:
- T1: hash_data() vs hashlib.sha256() mismatch in corpus eviction
- T2: weight cache staleness when corpus grows without edge changes
- T3: _max_mi_cache never invalidated after observe→record rename
- T5: dead constructor parameters with misleading docstrings
- T6: algebraic no-op in greedy loss formula
"""

import hashlib
import re
import tempfile
from pathlib import Path

import pytest


class TestHashConsistency:
    """Ensure hash_data() is used everywhere filenames are matched against content."""

    def test_hash_data_deterministic(self):
        from fuzzer_tool.adapters.filesystem import hash_data

        data = b"deterministic check"
        assert hash_data(data) == hash_data(data)

    def test_hash_data_16char_hex(self):
        from fuzzer_tool.adapters.filesystem import hash_data

        h = hash_data(b"test")
        assert len(h) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", h)

    def test_corpus_filenames_match_hash_data(self):
        """Corpus files are named id_{hash_data(content)}; any code that
        matches filenames against content must use hash_data()."""
        from fuzzer_tool.adapters.filesystem import hash_data, save_to_corpus

        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            seen = set()
            content = b"regression test seed"
            save_to_corpus(content, corpus_dir, seen)
            expected_name = f"id_{hash_data(content)}"
            assert (corpus_dir / "seeds" / expected_name).exists()

    def test_auto_minimize_kept_set_uses_hash_data(self):
        """auto_minimize_corpus must use hash_data(), not hashlib.sha256."""
        import inspect

        from fuzzer_tool.services.corpus_manager import CorpusManager

        source = inspect.getsource(CorpusManager.auto_minimize_corpus)
        assert "hashlib.sha256" not in source, (
            "auto_minimize_corpus must not use hashlib.sha256 directly; "
            "use hash_data() from fuzzer_tool.adapters.filesystem instead"
        )
        assert "hash_data" in source


class TestWeightCacheStaleness:
    """Weight cache must refresh when corpus grows, not just append."""

    def test_mi_weight_in_range_after_many_observations(self):
        """mutation_weight() must always return [0.1, 5.0] regardless of
        how many observations have been recorded."""
        from fuzzer_tool.core.mi import MutualInformationTracker

        t = MutualInformationTracker(min_observations=5)
        for i in range(200):
            pattern = bytes([i % 256, (i * 7) % 256])
            edge = bytes([1 if i % 3 == 0 else 0, 1 if i % 5 == 0 else 0])
            t.record(pattern, edge, map_size=2)
            if i > 10:
                w = t.mutation_weight(0, input_length=2)
                assert 0.1 <= w <= 5.0, (
                    f"mutation_weight returned {w} at observation {i}, "
                    f"outside documented range [0.1, 5.0]"
                )


class TestMICacheInvalidation:
    """_max_mi_cache must be invalidated on every record() call."""

    def test_cache_cleared_on_record(self):
        from fuzzer_tool.core.mi import MutualInformationTracker

        t = MutualInformationTracker(min_observations=5)

        # Position 0 byte=0 -> edges {0,1}, byte=1 -> edge {2} only
        # This creates genuine mutual information at position 0
        for i in range(60):
            b0 = i % 2
            if b0 == 0:
                edge = bytes([1, 1, 0])
            else:
                edge = bytes([0, 0, 1])
            t.record(bytes([b0, 0]), edge, map_size=3)

        # Trigger cache population via mutation_weight
        t.mutation_weight(0, input_length=2)
        assert len(t._max_mi_cache) > 0, "cache should be populated after mutation_weight"

        # New observations must clear the cache
        for i in range(60):
            b0 = (i + 1) % 2
            if b0 == 0:
                edge = bytes([0, 1, 1])
            else:
                edge = bytes([1, 0, 0])
            t.record(bytes([b0, 0]), edge, map_size=3)

        assert len(t._max_mi_cache) == 0, (
            "_max_mi_cache was not invalidated after record(); "
            "weights can exceed documented [0.1, 5.0] range"
        )


class TestRateDistortionLoss:
    """The greedy loss formula must count truly unique edges, not total edges."""

    def test_unique_seed_removed_last(self):
        """A seed covering edges no other seed covers must be removed last."""
        from fuzzer_tool.core.rate_distortion import RateDistortionCorpus

        rd = RateDistortionCorpus()
        seeds = {
            "redundant_a": {0, 1, 2},
            "redundant_b": {0, 1, 2},
            "unique": {3},  # only seed covering edge 3
        }
        curve = rd.compute_rate_distortion_curve(seeds, step_size=1)

        fracs = {s: f for s, f in curve}
        # At corpus_size=2, unique should still be present → coverage=1.0
        assert fracs[2] == 1.0, (
            f"At corpus_size=2, unique seed should still be present but coverage is {fracs[2]}"
        )

    def test_redundant_seed_removed_first(self):
        """A seed whose every edge is covered by others should be removed first."""
        from fuzzer_tool.core.rate_distortion import RateDistortionCorpus

        rd = RateDistortionCorpus()
        seeds = {
            "essential": {0, 1},
            "redundant": {0, 1},  # fully covered by essential
        }
        curve = rd.compute_rate_distortion_curve(seeds, step_size=1)

        fracs = {s: f for s, f in curve}
        assert fracs[1] == 1.0, (
            f"After removing redundant seed, coverage should be 1.0 but got {fracs[1]}"
        )


class TestDeadParameters:
    """Constructor parameters must be used, not stored and forgotten."""

    def test_renyi_no_smoothing_param(self):
        from fuzzer_tool.core.renyi import RenyiEntropy

        r = RenyiEntropy()
        assert not hasattr(r, "smoothing"), "RenyiEntropy stores 'smoothing' but never uses it"

    def test_renyi_docstring_no_smoothing_claim(self):
        from fuzzer_tool.core.renyi import RenyiEntropy

        doc = RenyiEntropy.__doc__ or ""
        assert "smoothing" not in doc.lower(), (
            "RenyiEntropy docstring mentions smoothing but it is not implemented"
        )
