"""Unit tests for services/parallel.py — sync bookkeeping and worker logic.

Every test here used to write seeds FLAT at the worker-dir top level
(``.w0/id_aaa``), a layout ``save_to_corpus`` has never produced. They passed
against a sync function that listed the worker dir non-recursively, so the
suite certified a transfer path that moved zero real seeds. Seeds are written
through the shipping writer now, so the layout under test is the one on disk.
"""

from fuzzer_tool.adapters.filesystem import hash_data, save_to_corpus
from fuzzer_tool.services.parallel import _sync_corpus_in, _sync_seen


def write_seed(worker_dir, data: bytes) -> str:
    """Write a seed exactly as a worker does, returning its hash."""
    worker_dir.mkdir(parents=True, exist_ok=True)
    save_to_corpus(data, worker_dir, set())
    return hash_data(data)


class MockFuzzer:
    """Minimal mock for testing _sync_corpus_in."""

    def __init__(self):
        self.seen_hashes = set()
        self.added = []

    def save_to_corpus(self, data):
        self.seen_hashes.add(hash_data(data))
        self.added.append(data)


class TestSyncCorpus:
    def setup_method(self):
        _sync_seen.clear()

    def test_sync_pulls_new_seeds(self, tmp_path):
        parent = tmp_path / "work"
        w0 = parent / ".w0"
        write_seed(w0, b"seed_a")
        write_seed(w0, b"seed_b")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert set(fuzzer.added) == {b"seed_a", b"seed_b"}

    def test_sync_skips_already_seen(self, tmp_path):
        parent = tmp_path / "work"
        w0 = parent / ".w0"
        write_seed(w0, b"seed_a")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)
        assert len(fuzzer.added) == 1

        _sync_corpus_in(parent, fuzzer, max_new=50)
        assert len(fuzzer.added) == 1

    def test_sync_respects_max_new(self, tmp_path):
        parent = tmp_path / "work"
        w0 = parent / ".w0"
        for i in range(10):
            write_seed(w0, f"seed_{i}".encode())

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=3)

        assert len(fuzzer.added) == 3

    def test_sync_skips_non_worker_dirs(self, tmp_path):
        parent = tmp_path / "work"
        write_seed(parent / "corpus", b"should_skip")
        write_seed(parent / ".w0", b"should_pull")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert fuzzer.added == [b"should_pull"]

    def test_sync_multiple_workers(self, tmp_path):
        parent = tmp_path / "work"
        write_seed(parent / ".w0", b"from_w0")
        write_seed(parent / ".w1", b"from_w1")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert set(fuzzer.added) == {b"from_w0", b"from_w1"}

    def test_sync_deduplicates_across_workers(self, tmp_path):
        parent = tmp_path / "work"
        write_seed(parent / ".w0", b"dup")
        write_seed(parent / ".w1", b"dup")

        fuzzer = MockFuzzer()
        _sync_corpus_in(parent, fuzzer, max_new=50)

        assert fuzzer.added == [b"dup"]
