"""Tests for crashing seeds — inputs observed to crash the target are stored
under corpus/seeds/crashing/ and marked irreplaceable so no pruning path
(auto_minimize_corpus, trim_new_coverage, minimize) can remove them.
"""

import hashlib
import tempfile
import types
from pathlib import Path

from fuzzer_tool.adapters.filesystem import (
    hash_data,
    load_corpus,
    save_crashing_seed,
    save_to_corpus,
)
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.corpus_manager import CorpusManager


def _seed_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class MockFuzzer:
    """Minimal fuzzer mock for crashing-seed logic."""

    def __init__(self, corpus_dir: Path):
        self.ga = None
        self.qea = None
        self.corpus: list[bytes] = []
        self.seed_meta: dict[bytes, dict] = {}
        self._edge_tracker = EdgeTracker()
        self.shm_cov = None
        self.ptrace_cov = None
        self.max_corpus = 0
        self.max_corpus_bytes = 0
        self.corpus_dir = corpus_dir
        self._weight_cache = None
        self._cached_weights: dict = {}
        self._pruned_count = 0
        self._last_minimize_exec = 0
        self.exec_count = 0
        self._stop_requested = False
        self._use_bayesian = False
        self._seed_quality = None
        self.irreplaceable_hashes: set[str] = set()
        self.seen_hashes: set[str] = set()
        self.bloom = None
        self._total_corpus_attempts = 0
        self._last_hamming_distance = -1
        self._corpus_size_history: list[int] = []
        self._corpus_secretary = None
        _markov = types.SimpleNamespace()
        _markov.train = lambda data: None
        _markov.is_trained = lambda: False
        _markov.snapshot_and_check_plateau = lambda: False
        self.markov = _markov
        # Lineage gating read by auto_minimize_corpus — off for mocks.
        self._use_lineage = False

    def make_crash_capable(self):
        """Add the attributes CorpusManager.save_crash reads."""
        self.exec_count = 10
        self.target = "/bin/true"
        self._last_ops_used = []
        self._stats = types.SimpleNamespace(format_elapsed=lambda: "0s")
        self._last_regs = None
        self.ptrace_cov = None
        self._last_fault_addr = None
        self.crashes_dir = self.corpus_dir.parent / "crashes"
        self.crashes_dir.mkdir(parents=True, exist_ok=True)
        self.crash_hashes: set[str] = set()
        self.crash_sigs: dict = {}
        self.crash_frames: dict = {}
        self.save_smaller = False
        self.crash_blocklist: set = set()
        self.crash_allowlist: set = set()
        return self


class TestSaveCrashingSeed:
    def test_creates_file_and_marks_hash(self, tmp_path):
        data = b"CRASHER" + b"x" * 60
        seen: set = set()
        irr: set = set()
        assert save_crashing_seed(data, tmp_path, seen, irr) is True
        h = hash_data(data)
        f = tmp_path / "seeds" / "crashing" / h[:2] / f"id_{h}"
        assert f.is_file()
        assert f.read_bytes() == data
        assert h in irr
        assert h in seen

    def test_duplicate_input_still_gets_protected_copy(self, tmp_path):
        """Adversarial: input already known as a regular corpus seed still
        gets its protected copy under seeds/crashing/ and is marked."""
        data = b"DUPCRASH" + b"y" * 60
        seen: set = set()
        save_to_corpus(data, tmp_path, seen)
        assert hash_data(data) in seen
        irr: set = set()

        assert save_crashing_seed(data, tmp_path, seen, irr) is False  # not new
        h = hash_data(data)
        assert (tmp_path / "seeds" / "crashing" / h[:2] / f"id_{h}").is_file()
        assert h in irr

    def test_repeat_crash_on_same_input_is_a_stat(self, tmp_path):
        data = b"AGAIN" + b"z" * 60
        seen: set = set()
        irr: set = set()
        save_crashing_seed(data, tmp_path, seen, irr)
        before = {p.name for p in (tmp_path / "seeds" / "crashing").rglob("*")}
        assert save_crashing_seed(data, tmp_path, seen, irr) is False
        after = {p.name for p in (tmp_path / "seeds" / "crashing").rglob("*")}
        assert before == after

    def test_load_corpus_tracks_crashing_dir(self, tmp_path):
        data = b"LOADED" + b"w" * 60
        save_crashing_seed(data, tmp_path, set(), set())
        corpus, seen, irr = load_corpus(tmp_path, add_default=False)
        assert data in corpus
        assert hash_data(data) in irr


class TestCrashingSurvivesPruning:
    def _make_fuzzer(self, tmp_dir: Path) -> MockFuzzer:
        f = MockFuzzer(tmp_dir)
        f.max_corpus = 1  # small enough to trigger pruning
        return f

    def test_crashing_seed_survives_auto_minimize(self):
        """A crashing seed with zero unique edges is NOT pruned."""
        tmp = Path(tempfile.mkdtemp())
        f = self._make_fuzzer(tmp)
        et = f._edge_tracker
        et.cumulative_edges = {1, 2, 3}

        seed_a = b"unique___" + b"a" * 60
        seed_c = b"crashing_" + b"c" * 60

        ka = _seed_key(seed_a)
        kc = _seed_key(seed_c)
        et.seed_edges[ka] = {1, 2, 3}
        et.seed_hit_counts[ka] = {e: 1 for e in range(1, 4)}
        et.seed_edges[kc] = {1, 2}
        et.seed_hit_counts[kc] = {e: 1 for e in range(1, 3)}

        f.corpus = [seed_a, seed_c]
        f.seed_meta = {
            seed_a: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 100.0},
            seed_c: {"fuzz_count": 1, "coverage_edges": 2, "added_at": 101.0},
        }
        # Simulate a prior crash on seed_c: marked irreplaceable.
        f.irreplaceable_hashes.add(hash_data(seed_c))

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        assert seed_a in f.corpus, "unique-coverage seed should survive"
        assert seed_c in f.corpus, "crashing seed should survive despite redundancy"

    def test_trim_new_coverage_never_touches_crashing_seed(self):
        """Guard must fire before any target execution: a crashing seed that
        would otherwise be trimmable is left byte-exact."""
        tmp = Path(tempfile.mkdtemp())
        f = MockFuzzer(tmp)
        calls: list[bytes] = []

        class RecordingRunner:
            def run_target(self, data):
                calls.append(data)
                return (-2, "")

        f._runner = RecordingRunner()
        f.shm_cov = types.SimpleNamespace(get_edge_ids=lambda: {7})
        f._agg_cache_valid = True
        data = b"TRIMME" + b"t" * 80  # long enough to be trimmable
        f.irreplaceable_hashes.add(hash_data(data))

        mgr = CorpusManager(f)
        mgr.trim_new_coverage(data, parent=b"p" * 64)

        assert calls == [], "trim executed the target despite irreplaceable guard"
        assert data not in f.seed_meta or f.seed_meta.get(data) is not None


class TestCrashPathPersistsSeed:
    def test_save_crash_writes_protected_copy(self, tmp_path):
        """Regression: every detected crash lands in seeds/crashing/, even
        when the input was already a regular corpus entry."""
        tmp = Path(tempfile.mkdtemp())
        corpus_dir = tmp / "corpus"
        corpus_dir.mkdir()
        f = MockFuzzer(corpus_dir).make_crash_capable()

        data = b"PERSIST" + b"q" * 60
        # The input already lives in the corpus as a regular seed.
        save_to_corpus(data, corpus_dir, f.seen_hashes)
        f.corpus = [data]

        mgr = CorpusManager(f)
        name = mgr.save_crash(data, -11, "")
        assert isinstance(name, str)

        h = hash_data(data)
        protected = corpus_dir / "seeds" / "crashing" / h[:2] / f"id_{h}"
        assert protected.is_file(), "crashing copy missing from seeds/crashing/"
        assert hash_data(data) in f.irreplaceable_hashes
