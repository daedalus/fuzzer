"""Tests for CumulativeByteEntropy and its wiring into load_corpus().

Two things to pin down:
1. The tracker itself: bits() must agree with the existing pooled-corpus
   entropy formula (_corpus_byte_entropy in services/report.py), computed
   incrementally instead of by rescanning every seed each time.
2. load_corpus() must fold every seed it reads from disk into a tracker
   passed as entropy_tracker -- both full files and delta-reconstructed
   seeds -- and must never touch the tracker for the synthetic default
   seed, since that one is never read from disk.
"""

from __future__ import annotations

import math

from fuzzer_tool.adapters.filesystem import load_corpus, save_to_corpus
from fuzzer_tool.core.byte_entropy import (
    ENTROPY_SAMPLE_CAP,
    CumulativeByteEntropy,
    byte_entropy_bits,
)
from fuzzer_tool.services.report import _corpus_byte_entropy


def _reference_pooled_entropy(seeds: list[bytes], cap: int = ENTROPY_SAMPLE_CAP) -> float:
    """Same pooled-distribution formula as _corpus_byte_entropy, spelled
    out independently so this test doesn't just call the function it's
    meant to be an oracle for."""
    freq = [0] * 256
    total = 0
    for seed in seeds:
        for b in seed[:cap]:
            freq[b] += 1
            total += 1
    if total == 0:
        return 0.0
    ent = 0.0
    for count in freq:
        if count:
            pr = count / total
            ent -= pr * math.log2(pr)
    return ent


class TestCumulativeByteEntropyTracker:
    def test_empty_tracker_is_zero(self):
        tracker = CumulativeByteEntropy()
        assert tracker.bits() == 0.0
        assert len(tracker) == 0

    def test_single_seed_matches_byte_entropy_bits(self):
        tracker = CumulativeByteEntropy()
        data = bytes(range(256)) * 4  # uniform over all 256 values
        per_seed = tracker.add(data)
        assert per_seed == byte_entropy_bits(data)
        # One seed folded in: the running aggregate equals that seed's own
        # entropy, since the pooled distribution has nothing else in it.
        assert tracker.bits() == per_seed

    def test_add_returns_per_seed_entropy_not_running_total(self):
        tracker = CumulativeByteEntropy()
        low = b"\x00" * 1000
        high = bytes(range(256)) * 4
        ent_low = tracker.add(low)
        ent_high = tracker.add(high)
        # A near-zero-entropy seed's own reading must not be inflated by
        # whatever was already folded into the tracker.
        assert ent_low == byte_entropy_bits(low)
        assert ent_high == byte_entropy_bits(high)
        assert ent_low < ent_high

    def test_matches_reference_pooled_formula(self):
        import random

        rng = random.Random(42)
        seeds = [bytes(rng.randrange(256) for _ in range(200)) for _ in range(20)]
        tracker = CumulativeByteEntropy()
        for s in seeds:
            tracker.add(s)
        expected = _reference_pooled_entropy(seeds)
        assert abs(tracker.bits() - expected) < 1e-9

    def test_matches_report_corpus_byte_entropy_helper(self):
        import random

        rng = random.Random(7)
        seeds = [bytes(rng.randrange(256) for _ in range(300)) for _ in range(15)]
        tracker = CumulativeByteEntropy()
        for s in seeds:
            tracker.add(s)
        assert abs(tracker.bits() - _corpus_byte_entropy(seeds)) < 1e-9

    def test_empty_seed_is_a_no_op(self):
        tracker = CumulativeByteEntropy()
        assert tracker.add(b"") == 0.0
        assert len(tracker) == 0
        assert tracker.bits() == 0.0

    def test_cap_limits_bytes_folded_in(self):
        tracker = CumulativeByteEntropy()
        data = b"\xff" * (ENTROPY_SAMPLE_CAP + 500)
        tracker.add(data, cap=ENTROPY_SAMPLE_CAP)
        assert len(tracker) == ENTROPY_SAMPLE_CAP


class TestLoadCorpusEntropyTracker:
    def test_full_files_folded_into_tracker(self, tmp_path):
        a = b"AAAA" * 50
        b = bytes(range(256))
        seen = set()
        save_to_corpus(a, tmp_path, seen)
        save_to_corpus(b, tmp_path, seen)

        tracker = CumulativeByteEntropy()
        corpus, _, _ = load_corpus(tmp_path, entropy_tracker=tracker)
        assert len(corpus) == 2
        assert len(tracker) == len(a) + len(b)
        assert abs(tracker.bits() - _corpus_byte_entropy(corpus)) < 1e-9

    def test_delta_reconstructed_seeds_folded_into_tracker(self, tmp_path):
        # Build a delta chain: a -> b (delta). load_corpus must fold the
        # *resolved* bytes of b, not the diff, into the tracker.
        a = b"AAAA" * 20
        b = bytearray(a)
        b[0] = ord("Z")
        b = bytes(b)

        seen = set()
        save_to_corpus(a, tmp_path, seen)
        save_to_corpus(b, tmp_path, seen, parent=a, lineage_depth=0)

        tracker = CumulativeByteEntropy()
        corpus, _, _ = load_corpus(tmp_path, entropy_tracker=tracker)
        assert a in corpus
        assert b in corpus
        assert len(tracker) == len(a) + len(b)

    def test_default_synthetic_seed_not_folded_in(self, tmp_path):
        # Nonexistent corpus dir -> load_corpus returns the synthetic
        # b"AAAAAAAA" seed. It was never read from disk, so it must not
        # touch the tracker.
        empty_dir = tmp_path / "does_not_exist"
        tracker = CumulativeByteEntropy()
        corpus, _, _ = load_corpus(empty_dir, entropy_tracker=tracker, add_default=True)
        assert corpus == [b"AAAAAAAA"]
        assert len(tracker) == 0
        assert tracker.bits() == 0.0

    def test_no_tracker_passed_is_still_safe(self, tmp_path):
        # entropy_tracker is optional -- existing callers that don't pass
        # it must be unaffected.
        save_to_corpus(b"hello world", tmp_path, set())
        corpus, _, _ = load_corpus(tmp_path)
        assert len(corpus) == 1
