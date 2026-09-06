"""Tests for services/parallel.py — _sync_corpus_in and summary."""

from unittest.mock import MagicMock

from fuzzer_tool.adapters.filesystem import hash_data, save_to_corpus
from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.services.parallel import _sync_corpus_in, _sync_seen


class TestSyncCorpusIn:
    """Seeds are written through save_to_corpus, so the on-disk layout under
    test is ``<worker>/seeds/<hh>/id_<hash>`` rather than a flat listing."""

    def setup_method(self):
        _sync_seen.clear()

    @staticmethod
    def _seed(worker_dir, data: bytes):
        worker_dir.mkdir(parents=True, exist_ok=True)
        save_to_corpus(data, worker_dir, set())

    def test_no_sibling_dirs(self, tmp_path):
        parent = tmp_path / "parent"
        parent.mkdir()
        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        fuzzer.save_to_corpus.assert_not_called()

    def test_sync_new_seeds(self, tmp_path):
        parent = tmp_path / "parent"
        self._seed(parent / ".w0", b"data1")
        self._seed(parent / ".w0", b"data2")

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        assert fuzzer.save_to_corpus.call_count == 2

    def test_sync_skips_state_file(self, tmp_path):
        """state.pkl.gz is the only top-level file a worker dir holds, and the
        old flat listing imported it as a seed into every sibling."""
        parent = tmp_path / "parent"
        w0 = parent / ".w0"
        self._seed(w0, b"good")
        store = StateStore(w0)
        store.set("markov", {"junk": 1})
        assert store.save()
        assert (w0 / "state.pkl.gz").exists()

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)

        assert fuzzer.save_to_corpus.call_count == 1
        assert fuzzer.save_to_corpus.call_args[0][0] == b"good"

    def test_sync_skips_pruned(self, tmp_path):
        """A sibling's pruned entries were dropped on purpose; re-importing
        them would undo its minimization."""
        parent = tmp_path / "parent"
        w0 = parent / ".w0"
        self._seed(w0, b"live")
        pruned = w0 / "seeds" / "pruned" / "ab"
        pruned.mkdir(parents=True)
        (pruned / f"id_{hash_data(b'dropped')}").write_bytes(b"dropped")

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)

        assert fuzzer.save_to_corpus.call_count == 1
        assert fuzzer.save_to_corpus.call_args[0][0] == b"live"

    def test_sync_skips_own_dir(self, tmp_path):
        parent = tmp_path / "parent"
        self._seed(parent / ".w0", b"mine")
        self._seed(parent / ".w1", b"theirs")

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer, self_dir=parent / ".w0")

        assert fuzzer.save_to_corpus.call_count == 1
        assert fuzzer.save_to_corpus.call_args[0][0] == b"theirs"

    def test_sync_dedup(self, tmp_path):
        parent = tmp_path / "parent"
        self._seed(parent / ".w0", b"dup")

        fuzzer = MagicMock()
        fuzzer.seen_hashes = {hash_data(b"dup")}
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        fuzzer.save_to_corpus.assert_not_called()

    def test_max_new_limit(self, tmp_path):
        parent = tmp_path / "parent"
        for i in range(5):
            self._seed(parent / ".w0", f"data{i}".encode())

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer, max_new=2)
        assert fuzzer.save_to_corpus.call_count == 2

    def test_non_w_dirs_skipped(self, tmp_path):
        parent = tmp_path / "parent"
        self._seed(parent / "other_dir", b"data")

        fuzzer = MagicMock()
        fuzzer.seen_hashes = set()
        fuzzer.save_to_corpus = MagicMock()
        _sync_corpus_in(parent, fuzzer)
        fuzzer.save_to_corpus.assert_not_called()


class TestDistributeInitialCorpus:
    """Regression for P0-2: pre-existing corpus was never given to workers."""

    def test_round_robin_into_worker_dirs(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import discover_seed_files, save_to_corpus
        from fuzzer_tool.services.parallel import _distribute_initial_corpus

        parent = tmp_path / "corpus"
        parent.mkdir()
        # Loose seeds + one under seeds/ (discover_seed_files handles both).
        (parent / "seed_a").write_bytes(b"alpha")
        (parent / "seed_b").write_bytes(b"bravo")
        (parent / "seed_c").write_bytes(b"charlie")
        seeds_dir = parent / "seeds" / "ab"
        seeds_dir.mkdir(parents=True)
        save_to_corpus(b"delta", parent, set())

        n = _distribute_initial_corpus(parent, n_workers=2)
        assert n >= 4

        w0 = discover_seed_files(parent / ".w0")
        w1 = discover_seed_files(parent / ".w1")
        assert len(w0) >= 1
        assert len(w1) >= 1
        # Combined content covers the originals.
        bodies = {p.read_bytes() for p in w0 + w1}
        assert b"alpha" in bodies
        assert b"bravo" in bodies
        assert b"charlie" in bodies
        assert b"delta" in bodies

    def test_empty_corpus_is_noop(self, tmp_path):
        from fuzzer_tool.services.parallel import _distribute_initial_corpus

        parent = tmp_path / "empty"
        parent.mkdir()
        assert _distribute_initial_corpus(parent, n_workers=3) == 0
