"""Tests for services/parallel.py — _sync_corpus_in and summary."""

from unittest.mock import MagicMock

from fuzzer_tool.services.parallel import _sync_corpus_in


class TestSyncCorpusIn:
    def test_no_sibling_dirs(self, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        fuzzer.save_to_corpus.assert_not_called()

    def test_sync_new_files(self, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        sibling = parent / ".w0"
        sibling.mkdir()
        (sibling / "file1.bin").write_bytes(b"data1")
        (sibling / "file2.bin").write_bytes(b"data2")

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        assert fuzzer.save_to_corpus.call_count == 2

    def test_sync_skips_txt_files(self, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        sibling = parent / ".w0"
        sibling.mkdir()
        (sibling / "data.bin").write_bytes(b"good")
        (sibling / "meta.txt").write_text("skip")

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        assert fuzzer.save_to_corpus.call_count == 1

    def test_sync_dedup(self, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        sibling = parent / ".w0"
        sibling.mkdir()
        (sibling / "file.bin").write_bytes(b"dup")

        fuzzer = MagicMock()
        # Pre-populate seen_hashes so dedup works
        from fuzzer_tool.adapters.filesystem import hash_data
        fuzzer.seen_hashes = {hash_data(b"dup")}
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        fuzzer.save_to_corpus.assert_not_called()

    def test_max_new_limit(self, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        sibling = parent / ".w0"
        sibling.mkdir()
        for i in range(5):
            (sibling / f"f{i}.bin").write_bytes(f"data{i}".encode())

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer, max_new=2)
        assert fuzzer.save_to_corpus.call_count == 2

    def test_non_w_dirs_skipped(self, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        sibling = parent / "other_dir"
        sibling.mkdir()
        (sibling / "file.bin").write_bytes(b"data")

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        fuzzer.save_to_corpus.assert_not_called()
