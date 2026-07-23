"""Tests for corpus manager: coverage verification after minimization and prune structure."""

import hashlib
import tempfile
import types
from pathlib import Path

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.corpus_manager import CorpusManager


def _make_seed(data: bytes) -> bytes:
    return data


def _seed_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class MockFuzzer:
    """Minimal fuzzer mock exposing only what auto_minimize_corpus touches."""

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
        # Attributes needed by save_to_corpus
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


class TestCoverageVerification:
    """Verify that auto_minimize_corpus doesn't lose unique-edge coverage."""

    def test_unique_edge_preserved_after_scoring(self):
        """A seed covering a unique edge with low wasserstein weight is not dropped.

        Setup:
          - cumulative_edges = {1, 2, 3, 4, 5}
          - Seed A covers {1,2,3,4}  (high edge_count → high score)
          - Seed B covers {1,2,3}    (medium)
          - Seed C covers {5}        (low edge_count → low score, but UNIQUE)
        Expectation: Seed C survives minimization despite low score because
        the post-scoring verification re-adds it when edge 5 goes missing.
        """
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        et = f._edge_tracker
        et.cumulative_edges = {1, 2, 3, 4, 5}

        # Create seeds
        seed_a = b"seed_a_" + b"x" * 60
        seed_b = b"seed_b_" + b"y" * 60
        seed_c = b"seed_c_" + b"z" * 60  # unique edge {5}

        ka = _seed_key(seed_a)
        kb = _seed_key(seed_b)
        kc = _seed_key(seed_c)

        # Register in edge tracker
        et.seed_edges[ka] = {1, 2, 3, 4}
        et.seed_hit_counts[ka] = {e: 1 for e in range(1, 5)}
        et.seed_edges[kb] = {1, 2, 3}
        et.seed_hit_counts[kb] = {e: 1 for e in range(1, 4)}
        et.seed_edges[kc] = {5}
        et.seed_hit_counts[kc] = {5: 1}

        # Populate corpus
        f.corpus = [seed_a, seed_b, seed_c]
        f.seed_meta = {
            seed_a: {"fuzz_count": 1, "coverage_edges": 4, "added_at": 100.0},
            seed_b: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 101.0},
            seed_c: {"fuzz_count": 1, "coverage_edges": 1, "added_at": 102.0},
        }

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        # All three seeds should still be present (Seed C is needed for edge {5})
        remaining_keys = {_seed_key(s) for s in f.corpus}
        assert kc in remaining_keys, (
            f"Seed C (key={kc}) was dropped despite covering unique edge {{5}}. "
            f"Remaining keys: {remaining_keys}"
        )

        # All edges should be covered
        covered_after = set()
        for s in f.corpus:
            sk = _seed_key(s)
            covered_after.update(et.seed_edges.get(sk, set()))
        assert covered_after == et.cumulative_edges, (
            f"Coverage mismatch after minimization: "
            f"had {len(covered_after)}/{len(et.cumulative_edges)} edges. "
            f"Missing: {et.cumulative_edges - covered_after}"
        )

    def test_redundant_seeds_are_removed(self):
        """Seeds that don't add new edges can be dropped."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        et = f._edge_tracker
        et.cumulative_edges = {1, 2, 3}

        seed_a = b"unique_a_" + b"x" * 60
        seed_r = b"redundant_" + b"y" * 60  # same edges as seed_a

        ka = _seed_key(seed_a)
        kr = _seed_key(seed_r)

        et.seed_edges[ka] = {1, 2, 3}
        et.seed_hit_counts[ka] = {e: 1 for e in range(1, 4)}
        et.seed_edges[kr] = {1, 2, 3}
        et.seed_hit_counts[kr] = {e: 1 for e in range(1, 4)}

        f.corpus = [seed_a, seed_r]
        f.seed_meta = {
            seed_a: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 100.0},
            seed_r: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 101.0},
        }

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        # At least one seed remains, coverage is intact
        assert len(f.corpus) >= 1
        covered_after = set()
        for s in f.corpus:
            sk = _seed_key(s)
            covered_after.update(et.seed_edges.get(sk, set()))
        assert covered_after == et.cumulative_edges

    def test_empty_corpus_no_crash(self):
        """Minimizing an empty corpus does nothing."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()
        assert f.corpus == []

    def test_single_seed_preserved(self):
        """A sole seed covering all edges is never dropped."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        et = f._edge_tracker
        et.cumulative_edges = {1, 2, 3}

        seed = b"sole_seed_" + b"x" * 60
        k = _seed_key(seed)
        et.seed_edges[k] = {1, 2, 3}
        et.seed_hit_counts[k] = {e: 1 for e in range(1, 4)}
        f.corpus = [seed]
        f.seed_meta[seed] = {"fuzz_count": 1, "coverage_edges": 3, "added_at": 100.0}

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()
        assert seed in f.corpus

    def test_multiple_unique_edges_all_preserved(self):
        """Multiple seeds each covering unique edges are all kept."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        et = f._edge_tracker
        et.cumulative_edges = {10, 20, 30}

        seeds = {}
        for i, edge in enumerate([10, 20, 30]):
            data = f"seed_{i}_".encode() + b"x" * 60
            k = _seed_key(data)
            et.seed_edges[k] = {edge}
            et.seed_hit_counts[k] = {edge: 1}
            seeds[edge] = data
            f.corpus.append(data)
            f.seed_meta[data] = {"fuzz_count": 1, "coverage_edges": 1, "added_at": float(i)}

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        remaining_keys = {_seed_key(s) for s in f.corpus}
        for edge, data in seeds.items():
            assert _seed_key(data) in remaining_keys, (
                f"Seed covering edge {edge} was dropped. Keys: {remaining_keys}"
            )


class TestPruneDirectoryStructure:
    """Verify pruned files go into two-digit hash subdirectories."""

    def _touch(self, d: Path, name: str) -> Path:
        p = d / name
        p.write_bytes(b"x" * 100)
        return p

    def test_pruned_files_in_subdirs(self):
        """Pruned files land in pruned/<first_two_hash_digits>/ file."""
        with tempfile.TemporaryDirectory() as td:
            corpus_dir = Path(td)
            seeds_dir = corpus_dir / "seeds"
            seeds_dir.mkdir(parents=True)

            # Create seed files with known hashes
            s1 = b"\x00" * 100  # hash prefix check
            h1 = _seed_key(s1)
            self._touch(seeds_dir, f"id_{h1}")

            s2 = b"\x01" * 100
            h2 = _seed_key(s2)
            self._touch(seeds_dir, f"id_{h2}")

            f = MockFuzzer(corpus_dir)
            et = f._edge_tracker
            et.cumulative_edges = {1, 2}

            k1, k2 = h1, h2
            et.seed_edges[k1] = {1}
            et.seed_hit_counts[k1] = {1: 1}
            et.seed_edges[k2] = {1}
            et.seed_hit_counts[k2] = {1: 1}

            # Keep only s1, prune s2
            f.corpus = [s1, s2]
            f.seed_meta[s1] = {"fuzz_count": 1, "coverage_edges": 1, "added_at": 0.0}
            f.seed_meta[s2] = {"fuzz_count": 1, "coverage_edges": 1, "added_at": 1.0}

            f.max_corpus = 1
            mgr = CorpusManager(f)
            mgr.auto_minimize_corpus()

            # After minimization, s1 stays, s2 (h2) is pruned
            pruned_dir = seeds_dir / "pruned"
            assert pruned_dir.exists(), f"pruned dir not created at {pruned_dir}"

            # Pruned files should be in two-digit subdirectories
            pruned_files = list(pruned_dir.rglob("*"))
            assert len(pruned_files) > 0, f"No pruned files found in {pruned_dir}"
            # Each pruned file should be in a two-digit subdirectory
            for pf in pruned_files:
                if pf.is_file():
                    parent_dir = pf.parent.name
                    assert len(parent_dir) == 2 and parent_dir.isalnum(), (
                        f"Pruned file {pf} is not in a two-digit subdirectory"
                    )

    def test_delta_files_also_in_subdirs(self):
        """Delta pruned files also go into two-digit hash subdirectories."""
        with tempfile.TemporaryDirectory() as td:
            corpus_dir = Path(td)
            seeds_dir = corpus_dir / "seeds"
            seeds_dir.mkdir(parents=True)
            deltas_dir = corpus_dir / "deltas"
            deltas_dir.mkdir(parents=True)

            s1 = b"\xaa" * 100
            s2 = b"\xbb" * 100
            h1 = _seed_key(s1)
            h2 = _seed_key(s2)

            # Create delta file for s2 (will be pruned)
            (deltas_dir / f"delta_{h2}.json").write_text('{"parent":"abc","diff":[]}')

            f = MockFuzzer(corpus_dir)
            et = f._edge_tracker
            et.cumulative_edges = {1}

            et.seed_edges[h1] = {1}
            et.seed_hit_counts[h1] = {1: 1}
            et.seed_edges[h2] = {1}
            et.seed_hit_counts[h2] = {1: 1}

            f.corpus = [s1, s2]
            f.seed_meta[s1] = {"fuzz_count": 1, "coverage_edges": 1, "added_at": 0.0}
            f.seed_meta[s2] = {"fuzz_count": 1, "coverage_edges": 1, "added_at": 1.0}

            f.max_corpus = 1
            mgr = CorpusManager(f)
            mgr.auto_minimize_corpus()

            pruned_deltas = deltas_dir / "pruned"
            prefix = h2[:2]
            delta_pruned = pruned_deltas / prefix / f"delta_{h2}.json"
            assert delta_pruned.exists(), (
                f"Delta pruned file not at {delta_pruned}. "
                f"Contents: {list(pruned_deltas.rglob('*')) if pruned_deltas.exists() else 'no pruned dir'}"
            )


class TestQEACorpusInteraction:
    """QEA manages its own population; corpus/seed_meta must not be corrupted."""

    def test_save_to_corpus_skips_append_under_qea(self):
        """Under QEA, save_to_corpus populates seed_meta but does not grow f.corpus."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.qea = object()  # truthy — QEA is active
        mgr = CorpusManager(f)

        initial_corpus = [b"seed_a", b"seed_b"]
        f.corpus = list(initial_corpus)

        data = b"qea_discovered_seed_" + b"x" * 50
        mgr.save_to_corpus(data)

        # Corpus list should not have grown
        assert len(f.corpus) == len(initial_corpus), (
            f"Corpus grew under QEA: {len(f.corpus)} vs {len(initial_corpus)}"
        )

        # seed_meta should still be populated
        assert data in f.seed_meta, "QEA-discovered seed missing from seed_meta"

    def test_auto_minimize_skips_under_qea(self):
        """auto_minimize_corpus is a no-op when QEA is active (seed_meta preserved)."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.qea = object()  # truthy — QEA is active
        mgr = CorpusManager(f)

        seed_a = b"qea_seed_a_" + b"x" * 50
        seed_b = b"qea_seed_b_" + b"y" * 50
        f.corpus = [seed_a, seed_b]
        f.seed_meta = {
            seed_a: {"fuzz_count": 10, "coverage_edges": 5, "added_at": 100.0},
            seed_b: {"fuzz_count": 20, "coverage_edges": 3, "added_at": 200.0},
        }

        mgr.auto_minimize_corpus()

        # Corpus and seed_meta are untouched
        assert len(f.corpus) == 2
        assert seed_a in f.seed_meta
        assert seed_b in f.seed_meta
        assert f.seed_meta[seed_a]["fuzz_count"] == 10


class TestKnapsackRetention:
    """Byte-budget-aware corpus retention picks small high-density seeds first."""

    def test_knapsack_prefers_small_seeds_over_large(self):
        """With a byte budget, small high-density seeds outrank large low-density ones."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.max_corpus_bytes = 200
        mgr = CorpusManager(f)

        # Three seeds of varying sizes
        small = b"A" * 10    # 10 bytes — lowest coverage score
        medium = b"B" * 100  # 100 bytes — moderate
        large = b"C" * 1000  # 1000 bytes — highest coverage score

        # Set up corpus and seed_meta with coverage scores only
        f.corpus = [small, medium, large]
        f.seed_meta = {
            small: {"fuzz_count": 1, "coverage_edges": 1, "added_at": 100.0,
                     "edge_bitmap": bytearray(0), "redqueen_offsets": [], "momentum": 0.0,
                     "lineage_depth": 0, "hamming_distance": 0},
            medium: {"fuzz_count": 1, "coverage_edges": 5, "added_at": 200.0,
                      "edge_bitmap": bytearray(0), "redqueen_offsets": [], "momentum": 0.0,
                      "lineage_depth": 0, "hamming_distance": 0},
            large: {"fuzz_count": 1, "coverage_edges": 10, "added_at": 300.0,
                     "edge_bitmap": bytearray(0), "redqueen_offsets": [], "momentum": 0.0,
                     "lineage_depth": 0, "hamming_distance": 0},
        }
        # Give each seed a unique edge so they pass the mandatory set-cover
        for seed in f.corpus:
            sk = _seed_key(seed)
            f._edge_tracker.seed_edges[sk] = {hash(seed) % 65536}

        mgr.auto_minimize_corpus()

        # large (1000B) cannot fit in 200B budget; small+medium (110B) can
        assert large not in f.corpus, (
            "Large seed should be evicted under byte budget"
        )
        assert small in f.corpus, (
            "Small high-density seed should be retained"
        )
        assert medium in f.corpus, (
            "Medium seed should be retained"
        )

    def test_count_budget_unchanged_when_no_byte_budget(self):
        """Without max_corpus_bytes, count-budget behavior is unchanged."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.max_corpus = 2
        f.max_corpus_bytes = 0  # no byte budget
        mgr = CorpusManager(f)

        small = b"A" * 10
        medium = b"B" * 100
        large = b"C" * 1000

        f.corpus = [small, medium, large]
        f.seed_meta = {
            small: {"fuzz_count": 1, "coverage_edges": 1, "added_at": 100.0,
                     "edge_bitmap": bytearray(0), "redqueen_offsets": [], "momentum": 0.0,
                     "lineage_depth": 0, "hamming_distance": 0},
            medium: {"fuzz_count": 1, "coverage_edges": 5, "added_at": 200.0,
                      "edge_bitmap": bytearray(0), "redqueen_offsets": [], "momentum": 0.0,
                      "lineage_depth": 0, "hamming_distance": 0},
            large: {"fuzz_count": 1, "coverage_edges": 10, "added_at": 300.0,
                     "edge_bitmap": bytearray(0), "redqueen_offsets": [], "momentum": 0.0,
                     "lineage_depth": 0, "hamming_distance": 0},
        }
        for seed in f.corpus:
            sk = _seed_key(seed)
            f._edge_tracker.seed_edges[sk] = {hash(seed) % 65536}

        mgr.auto_minimize_corpus()

        # With count budget of 2, large (highest score) should be kept
        assert len(f.corpus) >= 2
        assert large in f.corpus, (
            "Large seed with highest score should be retained under count budget"
        )
