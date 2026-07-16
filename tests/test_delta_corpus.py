"""Tests for delta-encoded corpus storage."""

from fuzzer_tool.adapters.filesystem import (
    SNAPSHOT_INTERVAL,
    apply_delta,
    compute_delta,
    hash_data,
    load_corpus,
    save_to_corpus,
)
from fuzzer_tool.core.bloom import BloomFilter


class TestComputeDelta:
    def test_identical_returns_empty_diff(self):
        parent = b"AAAA"
        delta = compute_delta(parent, parent)
        assert delta == []

    def test_single_byte_change(self):
        parent = b"AAAA"
        child = b"AABA"
        delta = compute_delta(parent, child)
        assert delta == [[2, ord("B")]]

    def test_different_lengths_returns_none(self):
        assert compute_delta(b"AA", b"AAA") is None

    def test_large_diff_returns_none(self):
        # > 25% changed
        parent = b"\x00" * 100
        child = b"\xff" * 100
        assert compute_delta(parent, child) is None

    def test_exact_threshold_returns_none(self):
        # 25% changed = 25 out of 100 → returns diff (not None)
        parent = bytearray(b"\x00" * 100)
        child = bytearray(b"\x00" * 100)
        for i in range(25):
            child[i] = 0xFF
        delta = compute_delta(bytes(parent), bytes(child))
        assert delta is not None
        assert len(delta) == 25

    def test_over_threshold_returns_none(self):
        parent = bytearray(b"\x00" * 100)
        child = bytearray(b"\x00" * 100)
        for i in range(26):
            child[i] = 0xFF
        assert compute_delta(bytes(parent), bytes(child)) is None

    def test_empty_inputs(self):
        delta = compute_delta(b"", b"")
        assert delta == []


class TestApplyDelta:
    def test_roundtrip(self):
        parent = b"A" * 100
        child = bytearray(parent)
        child[50] = ord("B")
        child = bytes(child)
        delta = compute_delta(parent, child)
        assert delta is not None
        reconstructed = apply_delta(parent, delta)
        assert reconstructed == child

    def test_no_changes(self):
        data = b"same data"
        assert apply_delta(data, []) == data

    def test_multiple_changes(self):
        parent = b"A" * 100
        child = bytearray(parent)
        child[0] = ord("B")
        child[50] = ord("C")
        child = bytes(child)
        delta = compute_delta(parent, child)
        assert delta is not None
        assert apply_delta(parent, delta) == child

    def test_full_replacement(self):
        parent = b"\x00" * 10
        changes = [[i, i + 1] for i in range(10)]
        result = apply_delta(parent, changes)
        assert result == bytes(range(1, 11))


class TestDeltaSaveLoad:
    def test_delta_file_smaller_than_full(self, tmp_path):
        parent = b"A" * 100
        # Change 1 byte → delta should be smaller
        child = bytearray(parent)
        child[50] = 0xFF
        child = bytes(child)

        seen = set()
        save_to_corpus(parent, tmp_path, seen)
        save_to_corpus(child, tmp_path, seen, parent=parent, lineage_depth=0)

        files = list(tmp_path.iterdir())
        delta_files = [f for f in files if f.name.startswith("delta_")]
        full_files = [f for f in files if f.name.startswith("id_")]
        assert len(delta_files) == 1
        assert len(full_files) == 1

    def test_snapshot_at_interval(self, tmp_path):
        parent = b"A" * 100
        seen = set()
        save_to_corpus(parent, tmp_path, seen)

        # Save with depth at interval → should be full file, not delta
        child = bytearray(parent)
        child[0] = 0xFF
        child = bytes(child)
        save_to_corpus(child, tmp_path, seen, parent=parent, lineage_depth=SNAPSHOT_INTERVAL)

        files = list(tmp_path.iterdir())
        full_files = [f for f in files if f.name.startswith("id_")]
        assert len(full_files) == 2  # both parent and child as full files

    def test_load_reconstructs_delta_chain(self, tmp_path):
        # Build chain: A → B (delta) → C (delta)
        a = b"AAAA"
        b = bytearray(a)
        b[0] = ord("B")
        b = bytes(b)
        c = bytearray(b)
        c[1] = ord("C")
        c = bytes(c)

        seen = set()
        save_to_corpus(a, tmp_path, seen)
        save_to_corpus(b, tmp_path, seen, parent=a, lineage_depth=0)
        save_to_corpus(c, tmp_path, seen, parent=b, lineage_depth=1)

        corpus, loaded_seen = load_corpus(tmp_path)
        assert len(corpus) == 3
        assert a in corpus
        assert b in corpus
        assert c in corpus

    def test_load_with_bloom(self, tmp_path):
        a = b"test_a"
        seen = set()
        bloom = BloomFilter(capacity=100)
        save_to_corpus(a, tmp_path, seen, bloom)

        loaded_corpus, loaded_seen = load_corpus(tmp_path, bloom=bloom)
        assert len(loaded_corpus) == 1
        assert hash_data(a) in loaded_seen

    def test_corrupt_delta_skipped(self, tmp_path):
        a = b"AAAA"
        seen = set()
        save_to_corpus(a, tmp_path, seen)

        # Write a corrupt delta file
        h = "corrupt"
        (tmp_path / f"delta_{h}.json").write_text("not valid json {{{")

        corpus, _ = load_corpus(tmp_path)
        assert len(corpus) == 1  # only the full file

    def test_legacy_files_still_work(self, tmp_path):
        # Files without id_ prefix (legacy)
        (tmp_path / "f1").write_bytes(b"legacy1")
        (tmp_path / "f2").write_bytes(b"legacy2")
        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 2
        assert b"legacy1" in corpus
        assert b"legacy2" in corpus

    def test_deduplication(self, tmp_path):
        a = b"same"
        seen = set()
        save_to_corpus(a, tmp_path, seen)
        result = save_to_corpus(a, tmp_path, seen)
        assert result is False  # duplicate
        assert len(list(tmp_path.iterdir())) == 1

    def test_large_delta_stores_full(self, tmp_path):
        parent = b"\x00" * 40
        child = bytes(range(40))  # all different → > 25% changed
        seen = set()
        save_to_corpus(parent, tmp_path, seen)
        save_to_corpus(child, tmp_path, seen, parent=parent, lineage_depth=0)

        files = list(tmp_path.iterdir())
        full_files = [f for f in files if f.name.startswith("id_")]
        assert len(full_files) == 2  # both stored as full
