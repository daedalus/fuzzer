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

    def test_seeds_reach_worker_dirs(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import discover_seed_files, save_to_corpus
        from fuzzer_tool.services.parallel import _distribute_initial_corpus

        parent = tmp_path / "corpus"
        parent.mkdir()
        # Loose seeds + one under seeds/ (discover_seed_files handles both).
        (parent / "seed_a").write_bytes(b"alpha")
        (parent / "seed_b").write_bytes(b"bravo")
        (parent / "seed_c").write_bytes(b"charlie")
        save_to_corpus(b"delta", parent, set())

        n = _distribute_initial_corpus(parent, n_workers=2)
        assert n >= 4

        bodies = set()
        for w in (".w0", ".w1"):
            bodies |= {p.read_bytes() for p in discover_seed_files(parent / w)}
        assert {b"alpha", b"bravo", b"charlie", b"delta"} <= bodies

    def test_empty_corpus_is_noop(self, tmp_path):
        from fuzzer_tool.services.parallel import _distribute_initial_corpus

        parent = tmp_path / "empty"
        parent.mkdir()
        assert _distribute_initial_corpus(parent, n_workers=3) == 0


class TestDistributeInitialCorpusOwnership:
    """The distributor must agree with the partition the sync path enforces."""

    def test_worker_dir_contents_are_not_redistributed(self, tmp_path):
        """``".w" not in path.parts`` matched nothing -- the parts are ``.w0``.

        Without a top-level ``seeds/`` tree, discover_seed_files walks into
        the worker directories, so every restart re-scattered each worker's
        own corpus across its siblings.
        """
        from fuzzer_tool.services.parallel import _distribute_initial_corpus

        parent = tmp_path / "corpus"
        (parent / ".w0").mkdir(parents=True)
        (parent / ".w1").mkdir(parents=True)
        (parent / ".w0" / "id_owned0").write_bytes(b"already-w0")
        (parent / ".w1" / "id_owned1").write_bytes(b"already-w1")
        (parent / "fresh").write_bytes(b"fresh-seed")

        n = _distribute_initial_corpus(parent, n_workers=2)
        assert n == 1, "only the fresh top-level seed should be distributed"

        distributed = {
            p.read_bytes()
            for w in (".w0", ".w1")
            for p in (parent / w).rglob("id_*")
            if p.is_file() and p.name not in ("id_owned0", "id_owned1")
        }
        assert distributed == {b"fresh-seed"}

    def test_is_under_worker_dir_predicate(self):
        from pathlib import Path

        from fuzzer_tool.services.parallel import _is_under_worker_dir

        assert _is_under_worker_dir(Path("/c/.w0/seed"))
        assert _is_under_worker_dir(Path("/c/.w12/seeds/ab/id_x"))
        assert not _is_under_worker_dir(Path("/c/seeds/ab/id_x"))
        assert not _is_under_worker_dir(Path("/c/.weights/id_x"))

    def test_assignment_is_stable_across_enumeration_order(self, tmp_path):
        """Assignment is by content, so file order and restarts cannot move a seed."""
        from fuzzer_tool.services.parallel import _distribute_initial_corpus

        payloads = [f"seed-{i}".encode() for i in range(12)]

        def place(names):
            root = tmp_path / f"c{len(list(tmp_path.iterdir()))}"
            root.mkdir()
            for name, body in zip(names, payloads, strict=True):
                (root / name).write_bytes(body)
            _distribute_initial_corpus(root, n_workers=4)
            return {
                p.read_bytes(): w
                for w in (".w0", ".w1", ".w2", ".w3")
                for p in (root / w).rglob("id_*")
                if p.is_file()
            }

        first = place([f"a{i:02d}" for i in range(12)])
        # Same content, names that enumerate in the opposite order.
        second = place([f"z{11 - i:02d}" for i in range(12)])
        assert first == second

    def test_fractal_partition_assignment_matches_sync_filter(self, tmp_path):
        """A seed must land on the worker ``accept_for_worker`` would keep it on."""
        from fuzzer_tool.core.parallel_fractal_partition import assign_worker
        from fuzzer_tool.services.parallel import _distribute_initial_corpus

        parent = tmp_path / "corpus"
        parent.mkdir()
        payloads = [f"payload-{i}".encode() for i in range(16)]
        for i, body in enumerate(payloads):
            (parent / f"seed_{i:02d}").write_bytes(body)

        n_workers, depth = 4, 3
        _distribute_initial_corpus(
            parent,
            n_workers=n_workers,
            fractal_partition=True,
            fractal_depth=depth,
        )

        for w in range(n_workers):
            for path in (parent / f".w{w}").rglob("id_*"):
                if not path.is_file():
                    continue
                data = path.read_bytes()
                assert assign_worker(data, n_workers, depth) == w, (
                    f"{data!r} placed on .w{w} but owned by "
                    f".w{assign_worker(data, n_workers, depth)}"
                )
