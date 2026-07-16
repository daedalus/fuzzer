"""Unit tests for services/parallel.py — sync cursor and worker logic."""

from pathlib import Path

import pytest

from fuzzer_tool.services.parallel import _sync_cursors, _sync_corpus_in


class MockFuzzer:
    """Minimal mock for testing _sync_corpus_in."""

    def __init__(self):
        self.seen_hashes = set()
        self.added = []

    def save_to_corpus(self, data):
        from fuzzer_tool.adapters.filesystem import hash_data

        h = hash_data(data)
        self.seen_hashes.add(h)
        self.added.append(data)


class TestSyncCursors:
    def setup_method(self):
        _sync_cursors.clear()

    def test_sync_pulls_new_files(self, tmp_path):
        parent = tmp_path / "work"
        parent.mkdir()
        w0 = parent / ".w0"
        w0.mkdir()
        (w0 / "id_aaa").write_bytes(b"seed_a")
        (w0 / "id_bbb").write_bytes(b"seed_b")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert len(fuzzer.added) == 2
        assert b"seed_a" in fuzzer.added
        assert b"seed_b" in fuzzer.added

    def test_sync_skips_already_seen(self, tmp_path):
        parent = tmp_path / "work"
        parent.mkdir()
        w0 = parent / ".w0"
        w0.mkdir()
        (w0 / "id_aaa").write_bytes(b"seed_a")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)
        assert len(fuzzer.added) == 1

        # Run again — cursor advanced, no new files
        _sync_corpus_in(parent, fuzzer, max_new=50)
        assert len(fuzzer.added) == 1

    def test_sync_respects_max_new(self, tmp_path):
        parent = tmp_path / "work"
        parent.mkdir()
        w0 = parent / ".w0"
        w0.mkdir()
        for i in range(10):
            (w0 / f"id_{i:04d}").write_bytes(f"seed_{i}".encode())

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=3)

        assert len(fuzzer.added) == 3

    def test_sync_skips_non_worker_dirs(self, tmp_path):
        parent = tmp_path / "work"
        parent.mkdir()
        # Not a worker dir
        (parent / "corpus").mkdir()
        (parent / "corpus" / "id_aaa").write_bytes(b"should_skip")
        # Worker dir
        w0 = parent / ".w0"
        w0.mkdir()
        (w0 / "id_bbb").write_bytes(b"should_pull")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert len(fuzzer.added) == 1
        assert fuzzer.added[0] == b"should_pull"

    def test_sync_skips_meta_files(self, tmp_path):
        parent = tmp_path / "work"
        parent.mkdir()
        w0 = parent / ".w0"
        w0.mkdir()
        (w0 / "id_aaa").write_bytes(b"seed")
        (w0 / "id_aaa.txt").write_text("meta")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert len(fuzzer.added) == 1

    def test_sync_multiple_workers(self, tmp_path):
        parent = tmp_path / "work"
        parent.mkdir()
        w0 = parent / ".w0"
        w1 = parent / ".w1"
        w0.mkdir()
        w1.mkdir()
        (w0 / "id_001").write_bytes(b"from_w0")
        (w1 / "id_002").write_bytes(b"from_w1")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert len(fuzzer.added) == 2
        assert set(fuzzer.added) == {b"from_w0", b"from_w1"}

    def test_sync_deduplicates_across_workers(self, tmp_path):
        parent = tmp_path / "work"
        parent.mkdir()
        w0 = parent / ".w0"
        w1 = parent / ".w1"
        w0.mkdir()
        w1.mkdir()
        (w0 / "id_same").write_bytes(b"dup")
        (w1 / "id_same").write_bytes(b"dup")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert len(fuzzer.added) == 1
