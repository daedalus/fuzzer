"""Tests for irreplaceable seeds — seeds in corpus/seeds/irreplaceable/ that are never pruned."""

import hashlib
import tempfile
import types
from pathlib import Path

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.corpus_manager import CorpusManager, save_irreplaceable


def _seed_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class MockFuzzer:
    """Minimal fuzzer mock for testing irreplaceable seed logic."""

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


class TestIrreplaceableSeeds:
    """Verify that irreplaceable seeds survive auto_minimize_corpus pruning."""

    def _make_fuzzer(self, tmp_dir: Path) -> MockFuzzer:
        f = MockFuzzer(tmp_dir)
        f.max_corpus = 1  # small enough to trigger pruning
        return f

    def test_irreplaceable_survives_minimization(self):
        """An irreplaceable seed with zero unique edges is NOT pruned."""
        tmp = Path(tempfile.mkdtemp())
        f = self._make_fuzzer(tmp)
        et = f._edge_tracker
        et.cumulative_edges = {1, 2, 3}

        # Seed A covers {1,2,3}, Seed B covers {1,2} (redundant)
        seed_a = b"unique___" + b"a" * 60
        seed_b = b"redundant" + b"b" * 60

        ka = _seed_key(seed_a)
        kb = _seed_key(seed_b)

        et.seed_edges[ka] = {1, 2, 3}
        et.seed_hit_counts[ka] = {e: 1 for e in range(1, 4)}
        et.seed_edges[kb] = {1, 2}
        et.seed_hit_counts[kb] = {e: 1 for e in range(1, 3)}

        f.corpus = [seed_a, seed_b]
        f.seed_meta = {
            seed_a: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 100.0},
            seed_b: {"fuzz_count": 1, "coverage_edges": 2, "added_at": 101.0},
        }

        # Mark seed_b as irreplaceable (simulates loaded from irreplaceable/)
        f.irreplaceable_hashes.add(hash_data(seed_b))

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        # Both should survive: seed_a for unique coverage, seed_b as irreplaceable
        assert seed_a in f.corpus, "Seed A (unique coverage) should survive"
        assert seed_b in f.corpus, "Seed B (irreplaceable) should survive despite being redundant"

    def test_non_irreplaceable_redundant_is_pruned(self):
        """A redundant seed NOT marked irreplaceable IS pruned."""
        tmp = Path(tempfile.mkdtemp())
        f = self._make_fuzzer(tmp)
        et = f._edge_tracker
        et.cumulative_edges = {1, 2, 3}

        seed_a = b"unique___" + b"a" * 60
        seed_b = b"redundant" + b"b" * 60

        ka = _seed_key(seed_a)
        kb = _seed_key(seed_b)

        et.seed_edges[ka] = {1, 2, 3}
        et.seed_hit_counts[ka] = {e: 1 for e in range(1, 4)}
        et.seed_edges[kb] = {1, 2}
        et.seed_hit_counts[kb] = {e: 1 for e in range(1, 3)}

        f.corpus = [seed_a, seed_b]
        f.seed_meta = {
            seed_a: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 100.0},
            seed_b: {"fuzz_count": 1, "coverage_edges": 2, "added_at": 101.0},
        }
        # seed_b is NOT in irreplaceable_hashes

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        assert seed_a in f.corpus, "Seed A (unique) should survive"
        assert seed_b not in f.corpus, "Seed B (redundant, not irreplaceable) should be pruned"

    def test_irreplaceable_survives_aggressive_pruning(self):
        """Irreplaceable seeds survive even when ALL non-irreplaceable seeds are pruned."""
        tmp = Path(tempfile.mkdtemp())
        f = self._make_fuzzer(tmp)
        et = f._edge_tracker

        # 3 seeds: 2 non-irreplaceable + 1 irreplaceable, all covering the same edges.
        # With target_size=1, one of the non-irreplaceable seeds will be pruned.
        seed_a = b"redund1__" + b"a" * 60  # covers {1,2,3}
        seed_a2 = b"redund2__" + b"b" * 60  # covers {1,2,3}, redundant
        seed_irr = b"irreplace" + b"c" * 60  # covers just {1,2}, irreplaceable

        ka = _seed_key(seed_a)
        ka2 = _seed_key(seed_a2)
        kirr = _seed_key(seed_irr)

        et.cumulative_edges = {1, 2, 3}
        et.seed_edges[ka] = {1, 2, 3}
        et.seed_hit_counts[ka] = {e: 1 for e in range(1, 4)}
        et.seed_edges[ka2] = {1, 2, 3}
        et.seed_hit_counts[ka2] = {e: 1 for e in range(1, 4)}
        et.seed_edges[kirr] = {1, 2}
        et.seed_hit_counts[kirr] = {e: 1 for e in range(1, 3)}

        f.corpus = [seed_a, seed_a2, seed_irr]
        f.seed_meta = {
            seed_a: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 100.0},
            seed_a2: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 101.0},
            seed_irr: {"fuzz_count": 1, "coverage_edges": 2, "added_at": 102.0},
        }
        f.irreplaceable_hashes.add(hash_data(seed_irr))

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        assert seed_irr in f.corpus, "Irreplaceable seed should survive"
        # At least one non-irreplaceable seed was pruned
        assert len(f.corpus) <= 2, (
            f"Expected at most 2 seeds (1 irreplaceable + 1 non-irreplaceable), got {len(f.corpus)}"
        )

    def test_save_irreplaceable_creates_file_and_tracks_hash(self, tmp_path):
        """save_irreplaceable writes to irreplaceable/ and updates tracking sets."""
        data = b"keystone_seed_data"
        seen: set[str] = set()
        irreplaceable: set[str] = set()

        result = save_irreplaceable(data, tmp_path, seen, irreplaceable, bloom=None)
        assert result is True

        h = hash_data(data)
        assert h in irreplaceable
        assert h in seen

        # Verify file exists on disk
        irr_dir = tmp_path / "seeds" / "irreplaceable"
        assert irr_dir.exists()
        files = list(irr_dir.rglob("*"))
        assert any(f.read_bytes() == data for f in files if f.is_file())

    def test_load_corpus_loads_irreplaceable_together(self, tmp_path):
        """load_corpus loads from both seeds/ and seeds/irreplaceable/."""
        from fuzzer_tool.adapters.filesystem import load_corpus

        # Create seeds/ with one file
        seeds = tmp_path / "seeds"
        seeds.mkdir(parents=True)
        (seeds / "id_seed").write_bytes(b"seed_data")

        # Create seeds/irreplaceable/ with one file
        irr = tmp_path / "seeds" / "irreplaceable"
        irr.mkdir(parents=True)
        (irr / "id_keystone").write_bytes(b"keystone_data")

        corpus, seen, irr_hashes = load_corpus(tmp_path)

        assert len(corpus) == 2
        assert b"seed_data" in corpus
        assert b"keystone_data" in corpus
        assert hash_data(b"keystone_data") in irr_hashes
        assert hash_data(b"seed_data") not in irr_hashes

    def test_irreplaceable_dir_does_not_exist(self, tmp_path):
        """load_corpus handles missing seeds/irreplaceable/ directory gracefully."""
        from fuzzer_tool.adapters.filesystem import load_corpus

        seeds = tmp_path / "seeds"
        seeds.mkdir(parents=True)
        (seeds / "id_seed").write_bytes(b"seed_data")

        corpus, seen, irr_hashes = load_corpus(tmp_path, load_irreplaceable=True)
        assert len(corpus) == 1
        assert irr_hashes == set()
