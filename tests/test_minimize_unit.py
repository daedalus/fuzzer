"""Unit tests for services/minimize.py — corpus minimization logic."""

import hashlib
import shutil
from pathlib import Path

import pytest

from fuzzer_tool.services.minimize import _commit_results, _minimize_by_hash


class TestMinimizeByHash:
    def test_deduplicates_identical_files(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.bin").write_bytes(b"same content")
        (corpus / "b.bin").write_bytes(b"same content")
        (corpus / "c.bin").write_bytes(b"different")

        files = sorted(corpus.glob("*.bin"))
        kept, removed = _minimize_by_hash(files, None, corpus)
        assert kept == 2
        assert removed == 1
        assert len(list(corpus.glob("*.bin"))) == 2

    def test_keeps_all_unique_files(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i in range(5):
            (corpus / f"f{i}.bin").write_bytes(f"content_{i}".encode())

        files = sorted(corpus.glob("*.bin"))
        kept, removed = _minimize_by_hash(files, None, corpus)
        assert kept == 5
        assert removed == 0

    def test_empty_corpus(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        kept, removed = _minimize_by_hash([], None, corpus)
        assert kept == 0
        assert removed == 0

    def test_moves_pruned_to_pruned_dir(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "first.bin").write_bytes(b"same content")
        (corpus / "second.bin").write_bytes(b"same content")
        (corpus / "unique.bin").write_bytes(b"different")

        files = sorted(corpus.glob("*.bin"))
        _minimize_by_hash(files, None, corpus)

        pruned = corpus / "pruned"
        assert pruned.exists()
        pruned_files = list(pruned.glob("*.bin"))
        assert len(pruned_files) == 1
        # The second occurrence (alphabetically) gets pruned
        assert pruned_files[0].name == "second.bin"


class TestCommitResults:
    def test_output_dir_copies_kept(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.bin").write_bytes(b"aaa")
        (corpus / "b.bin").write_bytes(b"bbb")
        output = tmp_path / "output"

        files = sorted(corpus.glob("*.bin"))
        kept = [str(files[0])]
        kept_count, removed = _commit_results(files, kept, str(output), corpus)

        assert kept_count == 1
        assert removed == 1
        assert output.exists()
        assert len(list(output.glob("*.bin"))) == 1

    def test_in_place_moves_to_pruned(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.bin").write_bytes(b"aaa")
        (corpus / "b.bin").write_bytes(b"bbb")

        files = sorted(corpus.glob("*.bin"))
        kept = [str(files[0])]
        kept_count, removed = _commit_results(files, kept, None, corpus)

        assert kept_count == 1
        assert removed == 1
        pruned = corpus / "pruned"
        assert pruned.exists()
        assert len(list(pruned.glob("*.bin"))) == 1

    def test_copies_meta_files(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.bin").write_bytes(b"aaa")
        (corpus / "a.txt").write_text("metadata")
        output = tmp_path / "output"

        files = [corpus / "a.bin"]
        kept = [str(corpus / "a.bin")]
        _commit_results(files, kept, str(output), corpus)

        assert (output / "a.txt").exists()

    def test_all_kept_no_removal(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.bin").write_bytes(b"aaa")

        files = [corpus / "a.bin"]
        kept = [str(corpus / "a.bin")]
        kept_count, removed = _commit_results(files, kept, None, corpus)

        assert kept_count == 1
        assert removed == 0
        # pruned/ dir is created but empty
        pruned = corpus / "pruned"
        assert pruned.exists()
        assert len(list(pruned.glob("*.bin"))) == 0
