"""Tests for filesystem adapter: save_crash, load_corpus, save_to_corpus."""

import hashlib

from fuzzer_tool.adapters.filesystem import (
    hash_data,
    load_corpus,
    save_crash,
    save_to_corpus,
)
from fuzzer_tool.core.bloom import BloomFilter


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class TestHashData:
    def test_returns_16_hex(self):
        h = hash_data(b"hello")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert hash_data(b"foo") == hash_data(b"foo")

    def test_different_inputs(self):
        assert hash_data(b"a") != hash_data(b"b")


class TestLoadCorpus:
    def test_empty_dir(self, tmp_path):
        corpus, seen = load_corpus(tmp_path)
        assert corpus == [b"AAAAAAAA"]
        assert seen == set()

    def test_loads_files(self, tmp_path):
        (tmp_path / "id_aaa").write_bytes(b"alpha")
        (tmp_path / "id_bbb").write_bytes(b"beta")
        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 2
        assert b"alpha" in corpus
        assert b"beta" in corpus
        assert _h(b"alpha") in seen
        assert _h(b"beta") in seen

    def test_deduplicates_by_hash(self, tmp_path):
        (tmp_path / "f1").write_bytes(b"same")
        (tmp_path / "f2").write_bytes(b"same")
        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 1
        assert len(seen) == 1

    def test_populates_bloom(self, tmp_path):
        (tmp_path / "f1").write_bytes(b"data")
        bloom = BloomFilter(capacity=100)
        load_corpus(tmp_path, bloom=bloom)
        assert bloom.query(_h(b"data"))

    def test_nonexistent_dir(self, tmp_path):
        corpus, seen = load_corpus(tmp_path / "nope")
        assert corpus == [b"AAAAAAAA"]
        assert seen == set()

    def test_ignores_directories(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "real").write_bytes(b"real")
        corpus, seen = load_corpus(tmp_path)
        assert len(corpus) == 1
        assert b"real" in corpus


class TestSaveToCorpus:
    def test_saves_new_input(self, tmp_path):
        seen = set()
        result = save_to_corpus(b"hello", tmp_path, seen)
        assert result is True
        assert _h(b"hello") in seen
        saved = list(tmp_path.iterdir())
        assert len(saved) == 1

    def test_rejects_duplicate(self, tmp_path):
        seen = set()
        save_to_corpus(b"hello", tmp_path, seen)
        result = save_to_corpus(b"hello", tmp_path, seen)
        assert result is False
        assert len(list(tmp_path.iterdir())) == 1

    def test_saves_different_inputs(self, tmp_path):
        seen = set()
        save_to_corpus(b"aaa", tmp_path, seen)
        save_to_corpus(b"bbb", tmp_path, seen)
        assert len(list(tmp_path.iterdir())) == 2

    def test_with_bloom_new(self, tmp_path):
        seen = set()
        bloom = BloomFilter(capacity=100)
        result = save_to_corpus(b"fresh", tmp_path, seen, bloom=bloom)
        assert result is True
        assert _h(b"fresh") in seen
        assert bloom.query(_h(b"fresh"))

    def test_with_bloom_duplicate(self, tmp_path):
        seen = set()
        bloom = BloomFilter(capacity=100)
        save_to_corpus(b"dup", tmp_path, seen, bloom=bloom)
        result = save_to_corpus(b"dup", tmp_path, seen, bloom=bloom)
        assert result is False

    def test_with_bloom_false_positive_path(self, tmp_path):
        seen = set()
        bloom = BloomFilter(capacity=10)
        for i in range(200):
            bloom.add(f"filler_{i}")
        h = _h(b"new_input")
        if bloom.query(h):
            result = save_to_corpus(b"new_input", tmp_path, seen, bloom=bloom)
            assert result is True
            assert h in seen

    def test_file_written_to_disk(self, tmp_path):
        seen = set()
        save_to_corpus(b"disk_check", tmp_path, seen)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].read_bytes() == b"disk_check"

    def test_creates_dir(self, tmp_path):
        subdir = tmp_path / "nested" / "corpus"
        seen = set()
        save_to_corpus(b"create", subdir, seen)
        assert subdir.exists()
        assert list(subdir.iterdir())


class TestSaveCrash:
    def test_saves_crash(self, tmp_path):
        hashes = set()
        sigs = {}
        result = save_crash(b"crash_data", -11, "SIGSEGV", tmp_path, hashes, sigs)
        assert result is True
        assert _h(b"crash_data") in hashes
        assert "signal:11" in sigs
        files = list(tmp_path.iterdir())
        assert len(files) == 4  # .bin + .txt + .sh + .hex

    def test_rejects_duplicate_crash(self, tmp_path):
        hashes = set()
        sigs = {}
        save_crash(b"dup", -11, "SIGSEGV", tmp_path, hashes, sigs)
        result = save_crash(b"dup", -11, "SIGSEGV", tmp_path, hashes, sigs)
        assert result is False

    def test_crash_metadata_with_sanitizer(self, tmp_path):
        hashes = set()
        sigs = {}
        stderr = "==1==ERROR: AddressSanitizer: heap-buffer-overflow\n"
        save_crash(b"asan_data", 0, stderr, tmp_path, hashes, sigs)
        files = list(tmp_path.iterdir())
        meta = [f for f in files if f.suffix == ".txt"][0]
        content = meta.read_text()
        assert "AddressSanitizer" in content
        assert "heap-buffer-overflow" in content

    def test_crash_metadata_without_sanitizer(self, tmp_path):
        hashes = set()
        sigs = {}
        save_crash(b"plain", -6, "Aborted", tmp_path, hashes, sigs)
        files = list(tmp_path.iterdir())
        meta = [f for f in files if f.suffix == ".txt"][0]
        content = meta.read_text()
        assert "returncode:    -6" in content
        assert sigs["signal:6"] == 1

    def test_crash_sig_count(self, tmp_path):
        hashes = set()
        sigs = {}
        save_crash(b"a", -11, "SIGSEGV", tmp_path, hashes, sigs)
        save_crash(b"b", -11, "SIGSEGV", tmp_path, hashes, sigs)
        assert sigs["signal:11"] == 2

    def test_different_signals(self, tmp_path):
        hashes = set()
        sigs = {}
        save_crash(b"a", -11, "SIGSEGV", tmp_path, hashes, sigs)
        save_crash(b"b", -6, "SIGABRT", tmp_path, hashes, sigs)
        assert "signal:11" in sigs
        assert "signal:6" in sigs
        assert len(sigs) == 2
