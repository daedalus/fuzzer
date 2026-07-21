"""Tests for adapters/filesystem.py — delta encoding, load_corpus with deltas."""

import json

from fuzzer_tool.adapters.filesystem import (
    apply_delta,
    compute_delta,
    hash_data,
    load_corpus,
    save_to_corpus,
)
from fuzzer_tool.core.bloom import BloomFilter


class TestComputeDelta:
    def test_identical_bytes(self):
        d = compute_delta(b"AAAA", b"AAAA")
        assert d == []  # empty diff, valid delta

    def test_single_byte_change(self):
        d = compute_delta(b"AAAA", b"AABA")
        assert d is not None
        assert len(d) == 1
        assert d[0] == [2, ord("B")]

    def test_multiple_changes(self):
        d = compute_delta(b"AAAAAAAA", b"AAABAAAD")
        assert d is not None
        assert len(d) == 2

    def test_different_lengths(self):
        assert compute_delta(b"AAAA", b"AAA") is None
        assert compute_delta(b"AAA", b"AAAA") is None

    def test_high_diff_ratio_returns_none(self):
        # >25% of bytes changed → None
        parent = b"A" * 100
        child = b"B" * 100
        assert compute_delta(parent, child) is None


class TestApplyDelta:
    def test_roundtrip(self):
        parent = b"AAAA"
        child = b"AABA"
        d = compute_delta(parent, child)
        assert d is not None
        assert apply_delta(parent, d) == child

    def test_empty_diff(self):
        assert apply_delta(b"AAAA", []) == b"AAAA"

    def test_multiple_positions(self):
        parent = b"AAAA"
        diff = [[1, ord("B")], [3, ord("D")]]
        assert apply_delta(parent, diff) == b"ABAD"


class TestHashData:
    def test_deterministic(self):
        assert hash_data(b"hello") == hash_data(b"hello")

    def test_different_inputs(self):
        assert hash_data(b"hello") != hash_data(b"world")

    def test_length(self):
        assert len(hash_data(b"test")) == 16


class TestLoadCorpusDelta:
    def test_load_full_files(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        (seeds / "a.bin").write_bytes(b"AAAA")
        (seeds / "b.bin").write_bytes(b"BBBB")
        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 2
        assert b"AAAA" in corpus
        assert b"BBBB" in corpus

    def test_load_delta_chain(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        deltas_dir = tmp_path / "deltas"
        deltas_dir.mkdir()
        # Full file
        parent_data = b"AAAA"
        parent_hash = hash_data(parent_data)
        (seeds / f"{parent_hash}.bin").write_bytes(parent_data)

        # Delta: change byte 2 to B
        child = b"AABA"
        d = compute_delta(parent_data, child)
        child_hash = hash_data(child)
        delta_file = deltas_dir / f"delta_{child_hash}.json"
        delta_file.write_text(json.dumps({"parent": parent_hash, "diff": d}))

        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 2
        assert child in corpus

    def test_delta_with_bloom(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        deltas_dir = tmp_path / "deltas"
        deltas_dir.mkdir()
        bloom = BloomFilter(capacity=100)
        parent_data = b"AAAA"
        parent_hash = hash_data(parent_data)
        (seeds / f"{parent_hash}.bin").write_bytes(parent_data)

        child = b"AABA"
        d = compute_delta(parent_data, child)
        child_hash = hash_data(child)
        delta_file = deltas_dir / f"delta_{child_hash}.json"
        delta_file.write_text(json.dumps({"parent": parent_hash, "diff": d}))

        corpus, seen = load_corpus(tmp_path, bloom=bloom)
        assert len(corpus) == 2

    def test_corrupt_delta_skipped(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        deltas_dir = tmp_path / "deltas"
        deltas_dir.mkdir()
        parent_data = b"AAAA"
        parent_hash = hash_data(parent_data)
        (seeds / f"{parent_hash}.bin").write_bytes(parent_data)

        child_hash = "deadbeef00000001"
        delta_file = deltas_dir / f"delta_{child_hash}.json"
        delta_file.write_text("not valid json {{{")

        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 1  # only the parent

    def test_missing_parent_skipped(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        deltas_dir = tmp_path / "deltas"
        deltas_dir.mkdir()
        # Delta references non-existent parent
        delta_file = deltas_dir / "delta_deadbeef00000001.json"
        delta_file.write_text(json.dumps({"parent": "nope", "diff": [[0, 65]]}))

        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 1  # default entry

    def test_empty_corpus_gets_default(self, tmp_path):
        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 1
        assert corpus[0] == b"AAAAAAAA"


class TestSaveToCorpus:
    def test_new_entry(self, tmp_path):
        seen = set()
        result = save_to_corpus(b"hello", tmp_path, seen)
        assert result is True
        assert b"hello" in [f.read_bytes() for f in (tmp_path / "seeds").rglob("id_*")]

    def test_duplicate_rejected(self, tmp_path):
        seen = set()
        save_to_corpus(b"hello", tmp_path, seen)
        result = save_to_corpus(b"hello", tmp_path, seen)
        assert result is False

    def test_delta_encoding(self, tmp_path):
        seen = set()
        # Save parent first
        parent = b"AAAA"
        save_to_corpus(parent, tmp_path, seen)

        # Save child (only 1 byte different → delta)
        child = b"AABA"
        result = save_to_corpus(child, tmp_path, seen, parent=parent, lineage_depth=0)
        assert result is True

        # Should have a delta file in deltas/
        delta_files = list((tmp_path / "deltas").glob("delta_*.json"))
        assert len(delta_files) == 1

        # Full file should be in seeds/
        full_files = list((tmp_path / "seeds").iterdir())
        assert len(full_files) == 1

    def test_with_bloom(self, tmp_path):
        bloom = BloomFilter(capacity=100)
        seen = set()
        save_to_corpus(b"test", tmp_path, seen, bloom=bloom)
        assert bloom.query(hash_data(b"test"))
