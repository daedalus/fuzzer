"""Tests for corpus manager: coverage verification after minimization and prune structure."""

import hashlib
import tempfile
import types
from pathlib import Path

from fuzzer_tool.core.analyzers.analyzer_corpus_flux import CorpusFlux
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.corpus_manager import CorpusManager


def _make_seed(data: bytes) -> bytes:
    return data


def _seed_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _cm_seed_key(data: bytes) -> str:
    """Mirror the CorpusManager.seed_key logic, respecting xxhash availability."""
    try:
        import xxhash  # noqa: F401

        return xxhash.xxh64(data).hexdigest()[:16]
    except ImportError:
        return hashlib.sha256(data).hexdigest()[:16]


class MockFuzzer:
    """Minimal fuzzer mock exposing only what auto_minimize_corpus touches."""

    def __init__(self, corpus_dir: Path):
        self.ga = None
        self.qea = None
        self.corpus: list[bytes] = []
        self.seed_meta: dict[bytes, dict] = {}
        self._edge_tracker = EdgeTracker()
        # save_to_corpus() folds each newly-admitted seed's bytes into the
        # campaign RandPool (corpus_manager.py, inject_entropy). A real
        # Fuzzer always has this; the mock needs it or every save raises.
        self._rng = RandPool(seed=42)
        self.shm_cov = None
        self.ptrace_cov = None
        self.max_corpus = 0
        self.max_corpus_bytes = 0
        self.corpus_dir = corpus_dir
        self._weight_cache = None
        self._cached_weights: dict = {}
        self._pruned_count = 0
        self._corpus_flux = CorpusFlux()
        self._last_minimize_exec = 0
        self.exec_count = 0
        self._stop_requested = False
        self._use_bayesian = False
        self._seed_quality = None
        self.irreplaceable_hashes: set[str] = set()
        # Attributes needed by save_to_corpus
        self.seen_hashes: set[str] = set()
        self.bloom = None
        self._total_corpus_attempts = 0
        self._duplicate_reject_count = 0
        self._last_hamming_distance = -1
        self._corpus_size_history: list[int] = []
        self._corpus_secretary = None
        _markov = types.SimpleNamespace()
        _markov.train = lambda data: None
        _markov.is_trained = lambda: False
        _markov.snapshot_and_check_plateau = lambda: False
        self.markov = _markov
        self._minimize_pending = False
        # Lineage tree gating — off by default in mocks; the CorpusManager
        # reads this in save_to_corpus/trim_new_coverage.
        self._use_lineage = False

    def mean_exec_time(self) -> float:
        """Corpus-wide mean target time per execution, in seconds.

        auto_minimize_corpus reads this for the >5000-edge set-cover tie-break;
        the mock has no timing, so zero (= "nothing measured yet") is correct
        and makes seed_exec_us fall back per seed.
        """
        total = sum(m.get("total_time", 0.0) for m in self.seed_meta.values())
        samples = sum(m.get("cost_samples", 0) for m in self.seed_meta.values())
        return total / samples if samples > 0 else 0.0

    def _defer_minimize(self):
        self._minimize_pending = True

    def _flush_pending_minimize(self):
        if self._minimize_pending:
            self._minimize_pending = False
            # The real Fuzzer delegates to self._corpus_manager.auto_minimize_corpus();
            # tests call the manager directly, so this is a no-op for the mock.
            pass


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

    def test_save_to_corpus_grows_under_qea_with_elo_arbitration(self):
        """Regression: QEA + Elo arbitration must keep f.corpus growing.

        `--elo all` enables QEA alongside corpus-based seed strategies
        (weighted/pareto/bayesian/boltzmann) that read f.corpus.  QEA's
        bypass previously froze the corpus at its initial size (displayed
        ``corpus: 1`` forever), starving those strategies onto a single
        seed and stalling the run.  With Elo arbitrating, append applies.
        """
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.qea = object()  # truthy — QEA is active
        f._use_elo = True  # Elo seed arbitration on (e.g. --elo all)
        mgr = CorpusManager(f)

        initial_corpus = [b"seed_a", b"seed_b"]
        f.corpus = list(initial_corpus)

        data = b"elo_all_discovered_seed_" + b"y" * 50
        mgr.save_to_corpus(data)

        assert len(f.corpus) == len(initial_corpus) + 1, (
            f"Corpus stayed frozen under QEA+Elo: {len(f.corpus)}"
        )
        assert f.corpus[-1] == data
        assert data in f.seed_meta

    def test_auto_minimize_skips_under_standalone_qea(self):
        """auto_minimize_corpus is a no-op under standalone QEA (no --elo).

        f.corpus is frozen at the initial seed set there (save_to_corpus
        never appends to it), so there's nothing live to minimize.
        """
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

    def test_auto_minimize_runs_under_qea_with_elo_arbitration(self):
        """Regression: `--qea --elo all` must still be able to minimize.

        Same condition as save_to_corpus()'s append gate: once Elo lifts
        QEA's corpus bypass, f.corpus is the real live pool again and
        stale/redundant seeds should be pruned like any other run.
        """
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.qea = object()  # truthy — QEA is active
        f._use_elo = True  # Elo seed arbitration on (e.g. --elo all)
        mgr = CorpusManager(f)

        keep = b"qea_elo_keep_" + b"x" * 60
        stale = b"qea_elo_stale_" + b"y" * 60
        f.corpus = [keep, stale]
        f.seed_meta = {
            keep: {"fuzz_count": 10, "coverage_edges": 5, "added_at": 100.0, "input_size": len(keep)},
            stale: {"fuzz_count": 200, "coverage_edges": 0, "added_at": 200.0, "input_size": len(stale)},
        }
        f.max_corpus = 1

        mgr.auto_minimize_corpus()

        assert len(f.corpus) == 1, "QEA+Elo run did not minimize down to max_corpus"
        assert keep in f.corpus

    def test_auto_minimize_runs_under_ga_and_qea_and_elo_together(self):
        """Regression: the --hail-mary combination (--ga --qea --elo all) must
        actually minimize, not just each flag individually.

        --elo all force-enables both --ga and --qea (see the "QEA and GA now
        run simultaneously" comment in cli/commands.py's cmd_fuzz), so this
        three-way combination is what --hail-mary itself produces, not just
        a hand-stacked edge case.
        """
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.ga = object()  # truthy — GA is active
        f.qea = object()  # truthy — QEA is active
        f._use_elo = True  # e.g. --elo all
        mgr = CorpusManager(f)

        keep = b"hailmary_keep_" + b"x" * 60
        stale = b"hailmary_stale_" + b"y" * 60
        f.corpus = [keep, stale]
        f.seed_meta = {
            keep: {"fuzz_count": 10, "coverage_edges": 5, "added_at": 100.0, "input_size": len(keep)},
            stale: {"fuzz_count": 200, "coverage_edges": 0, "added_at": 200.0, "input_size": len(stale)},
        }
        f.max_corpus = 1

        mgr.auto_minimize_corpus()

        assert len(f.corpus) == 1, "--ga --qea --elo all still disables corpus minimization"
        assert keep in f.corpus

    def test_auto_minimize_runs_under_ga(self):
        """Regression: --ga must not silently disable --minimize-every-execs.

        GALifecycle keeps its own population of Individuals (seed bytes +
        seed_key), independent of f.corpus/f.seed_meta indices, so pruning
        f.corpus is safe under GA.
        """
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.ga = object()  # truthy — GA is active
        mgr = CorpusManager(f)

        keep = b"ga_keep_" + b"x" * 60
        stale = b"ga_stale_" + b"y" * 60
        f.corpus = [keep, stale]
        f.seed_meta = {
            keep: {"fuzz_count": 10, "coverage_edges": 5, "added_at": 100.0, "input_size": len(keep)},
            stale: {"fuzz_count": 200, "coverage_edges": 0, "added_at": 200.0, "input_size": len(stale)},
        }
        f.max_corpus = 1

        mgr.auto_minimize_corpus()

        assert len(f.corpus) == 1, "--ga still disables corpus minimization"
        assert keep in f.corpus


class TestKnapsackRetention:
    """Byte-budget-aware corpus retention picks small high-density seeds first."""

    def test_knapsack_prefers_small_seeds_over_large(self):
        """With a byte budget, small high-density seeds outrank large low-density ones."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.max_corpus_bytes = 200
        mgr = CorpusManager(f)

        # Three seeds of varying sizes
        small = b"A" * 10  # 10 bytes — lowest coverage score
        medium = b"B" * 100  # 100 bytes — moderate
        large = b"C" * 1000  # 1000 bytes — highest coverage score

        # Set up corpus and seed_meta with coverage scores only
        f.corpus = [small, medium, large]
        f.seed_meta = {
            small: {
                "fuzz_count": 1,
                "coverage_edges": 1,
                "added_at": 100.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
            medium: {
                "fuzz_count": 1,
                "coverage_edges": 5,
                "added_at": 200.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
            large: {
                "fuzz_count": 1,
                "coverage_edges": 10,
                "added_at": 300.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
        }
        # Give each seed a unique edge so they pass the mandatory set-cover
        for seed in f.corpus:
            sk = _seed_key(seed)
            f._edge_tracker.seed_edges[sk] = {hash(seed) % 65536}

        mgr.auto_minimize_corpus()

        # large (1000B) cannot fit in 200B budget; small+medium (110B) can
        assert large not in f.corpus, "Large seed should be evicted under byte budget"
        assert small in f.corpus, "Small high-density seed should be retained"
        assert medium in f.corpus, "Medium seed should be retained"

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
            small: {
                "fuzz_count": 1,
                "coverage_edges": 1,
                "added_at": 100.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
            medium: {
                "fuzz_count": 1,
                "coverage_edges": 5,
                "added_at": 200.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
            large: {
                "fuzz_count": 1,
                "coverage_edges": 10,
                "added_at": 300.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
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


class TestMDSSelection:
    """--mds-select swaps top-K-by-score for weighted MDS local search."""

    def _meta(self, coverage_edges: int) -> dict:
        return {
            "fuzz_count": 1,
            "coverage_edges": coverage_edges,
            "added_at": 100.0,
            "edge_bitmap": bytearray(0),
            "redqueen_offsets": [],
            "momentum": 0.0,
            "lineage_depth": 0,
            "hamming_distance": 0,
        }

    def test_prefers_diverse_pair_over_single_near_duplicate_cluster(self):
        """hub_a/hub_b are near-duplicates of each other (should conflict);
        each is also similar enough to a distinct low-score seed to matter,
        but the real signal is: two near-identical high-scorers shouldn't
        both survive a tight budget when a farther-apart pair covers as
        much distinct ground for comparable combined weight.
        """
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.max_corpus = 2
        f._use_mds_select = True
        mgr = CorpusManager(f)

        hub_a = b"A" * 40
        hub_b = b"B" * 40  # near-duplicate of hub_a in edge coverage
        distinct = b"C" * 40  # covers unrelated edges

        f.corpus = [hub_a, hub_b, distinct]
        f.seed_meta = {
            hub_a: self._meta(8),
            hub_b: self._meta(7),
            distinct: self._meta(6),
        }

        et = f._edge_tracker
        ka, kb, kc = (_cm_seed_key(s) for s in (hub_a, hub_b, distinct))
        # hub_a/hub_b share almost all edges (near-duplicate signatures);
        # distinct covers a disjoint edge set.
        et.record_edges(ka, set(range(1, 20)))
        et.record_edges(kb, set(range(1, 19)))  # 18/19 overlap with hub_a
        et.record_edges(kc, set(range(100, 106)))

        mgr.auto_minimize_corpus()

        assert len(f.corpus) <= 2
        # distinct must survive: it is the only source of edges 100-105,
        # so it is mandatory via set-cover regardless of MDS selection.
        assert distinct in f.corpus

    def test_falls_back_to_topk_when_edge_tracker_empty(self):
        """No minhash signatures registered -> _mds_select_optional degrades
        to plain top-K instead of raising.
        """
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.max_corpus = 2
        f._use_mds_select = True
        mgr = CorpusManager(f)

        small = b"A" * 10
        medium = b"B" * 100
        large = b"C" * 1000

        f.corpus = [small, medium, large]
        f.seed_meta = {
            small: self._meta(1),
            medium: self._meta(5),
            large: self._meta(10),
        }
        # Deliberately do not register any seed_edges/minhash signatures.

        mgr.auto_minimize_corpus()

        assert len(f.corpus) <= 3
        assert large in f.corpus  # highest score, top-K fallback keeps it

    def test_off_by_default_matches_topk_behavior(self):
        """_use_mds_select defaults False on MockFuzzer -- unrelated to
        this feature's tests, but pins that the default path is untouched.
        """
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        assert getattr(f, "_use_mds_select", False) is False
        f.max_corpus = 2
        mgr = CorpusManager(f)

        small = b"A" * 10
        medium = b"B" * 100
        large = b"C" * 1000
        f.corpus = [small, medium, large]
        f.seed_meta = {
            small: self._meta(1),
            medium: self._meta(5),
            large: self._meta(10),
        }
        for seed in f.corpus:
            sk = _cm_seed_key(seed)
            f._edge_tracker.seed_edges[sk] = {hash(seed) % 65536}

        mgr.auto_minimize_corpus()
        assert large in f.corpus


class TestFreshSeedProtection:
    """Seeds with fuzz_count == 0 must survive minimization (Fix 1)."""

    def test_fresh_seeds_not_pruned(self):
        """A seed with fuzz_count=0 is excluded from the pruned pool."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        mgr = CorpusManager(f)
        et = f._edge_tracker
        et.cumulative_edges = {1, 2, 3}

        # Two seeds: one "mature" (fuzz_count=5, coverage_edges=3),
        # one "fresh" (fuzz_count=0, coverage_edges=0).
        mature = b"mature_seed_" + b"m" * 60
        fresh = b"fresh_seed_" + b"f" * 60

        f.corpus = [mature, fresh]
        km = _seed_key(mature)
        kf = _seed_key(fresh)
        et.seed_edges[km] = {1, 2, 3}
        et.seed_edges[kf] = {1}  # fresh seed has edges but were never fuzzed

        f.seed_meta = {
            mature: {
                "fuzz_count": 5,
                "coverage_edges": 3,
                "added_at": 100.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
            fresh: {
                "fuzz_count": 0,
                "coverage_edges": 0,
                "added_at": 200.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
        }

        # Force target_size = 1 so mature seed alone would fit,
        # but fresh must survive too.
        f.max_corpus = 1
        mgr.auto_minimize_corpus()

        assert fresh in f.corpus, "Fresh seed (fuzz_count=0) was wrongly pruned"
        assert mature in f.corpus, "Mature seed should also survive"

    def test_fresh_seeds_re_added_after_pruning(self):
        """Fresh seeds excluded before scoring are re-added after pruning."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        mgr = CorpusManager(f)
        et = f._edge_tracker
        et.cumulative_edges = {1}

        seed_a = b"seed_a_" + b"a" * 60
        seed_b = b"seed_b_" + b"b" * 60  # fresh

        f.corpus = [seed_a, seed_b]
        sk_a = _seed_key(seed_a)
        sk_b = _seed_key(seed_b)
        et.seed_edges[sk_a] = {1}
        et.seed_edges[sk_b] = {1}

        f.seed_meta = {
            seed_a: {
                "fuzz_count": 50,
                "coverage_edges": 0,
                "added_at": 100.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
            seed_b: {
                "fuzz_count": 0,
                "coverage_edges": 0,
                "added_at": 200.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "momentum": 0.0,
                "lineage_depth": 0,
                "hamming_distance": 0,
            },
        }

        # Force target_size = 1, seed_a has stale edges, seed_b is fresh
        f.max_corpus = 1
        mgr.auto_minimize_corpus()

        assert seed_b in f.corpus, "Fresh seed must survive minimization"
        # seed_a may or may not survive based on scoring; the test is that fresh survives.
        assert len(f.corpus) >= 1


class TestSaveToCorpusCoverageEdges:
    """New seeds get actual coverage_edges from EdgeTracker (Fix 3)."""

    def test_save_to_corpus_propagates_coverage_edges(self):
        """After edges are recorded in EdgeTracker, save_to_corpus copies them to seed_meta."""

        f = MockFuzzer(Path(tempfile.mkdtemp()))
        mgr = CorpusManager(f)
        et = f._edge_tracker
        et.cumulative_edges = {10, 20, 30}

        parent = b"parent_seed_" + b"p" * 60
        f.corpus.append(parent)
        f.seed_meta[parent] = {
            "fuzz_count": 1,
            "coverage_edges": 3,
            "added_at": 100.0,
            "edge_bitmap": bytearray(0),
            "redqueen_offsets": [],
            "momentum": 0.0,
            "lineage_depth": 0,
            "hamming_distance": 0,
        }

        # Simulate what fuzz_one does before save_to_corpus:
        # record the child seed's edges in EdgeTracker.
        child = b"child_seed_" + b"c" * 60
        # Use _cm_seed_key to match CorpusManager.seed_key (may use xxhash).
        child_key = _cm_seed_key(child)
        et.record_edges(child_key, {10, 20})
        et.cumulative_edges = {10, 20, 30}

        # Now save to corpus (as fuzz_one would after recording edges).
        mgr.save_to_corpus(child, parent=parent)

        assert child in f.seed_meta, "Seed must have seed_meta entry"
        assert f.seed_meta[child]["coverage_edges"] > 0, (
            f"Expected coverage_edges > 0, got {f.seed_meta[child]['coverage_edges']}"
        )

    def test_save_to_corpus_zero_edges_when_not_recorded(self):
        """On a parentless insert (no prior record_edges), coverage_edges stays 0."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        mgr = CorpusManager(f)
        f._edge_tracker.cumulative_edges = set()

        seed = b"synced_seed_" + b"s" * 60
        mgr.save_to_corpus(seed)

        assert seed in f.seed_meta
        assert f.seed_meta[seed]["coverage_edges"] == 0, (
            "Seed without prior record_edges should have 0 coverage_edges"
        )


class TestDeferredMinimize:
    """auto_minimize_corpus is deferred when called from save_to_corpus (Fix 4)."""

    def test_save_to_corpus_sets_pending_flag(self):
        """save_to_corpus sets _minimize_pending instead of calling minimize directly."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        mgr = CorpusManager(f)
        f._edge_tracker.cumulative_edges = set()

        assert not f._minimize_pending, "Flag should start False"

        # Add enough seeds to trigger max_corpus limit
        f.max_corpus = 1
        seed1 = b"seed_one_" + b"x" * 60
        mgr.save_to_corpus(seed1)
        assert not f._minimize_pending, "Below limit, no minimize needed"

        seed2 = b"seed_two_" + b"y" * 60
        mgr.save_to_corpus(seed2)
        assert f._minimize_pending, (
            "save_to_corpus should set _minimize_pending when corpus exceeds max_corpus"
        )

    def test_flush_runs_minimize(self):
        """_flush_pending_minimize clears the flag (real minimize is a no-op in mock)."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f._minimize_pending = True
        f._flush_pending_minimize()
        assert not f._minimize_pending, "flush must clear the flag"


class TestCorpusAddedCount:
    """save_to_corpus records an addition on CorpusFlux on every successful save.

    This is the addition-side channel CorpusFlux pairs with evictions to
    distinguish a stalled campaign from one in dynamic equilibrium -- see
    core/corpus_flux.py (P4-T6).
    """

    def test_records_addition_on_each_successful_save(self):
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        mgr = CorpusManager(f)
        f._edge_tracker.cumulative_edges = set()

        assert f._corpus_flux.total_additions == 0
        mgr.save_to_corpus(b"seed_one_" + b"x" * 60)
        assert f._corpus_flux.total_additions == 1
        mgr.save_to_corpus(b"seed_two_" + b"y" * 60)
        assert f._corpus_flux.total_additions == 2

    def test_does_not_record_addition_on_duplicate(self):
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        mgr = CorpusManager(f)
        f._edge_tracker.cumulative_edges = set()

        seed = b"seed_dup_" + b"x" * 60
        mgr.save_to_corpus(seed)
        assert f._corpus_flux.total_additions == 1
        # Same content again: save_to_corpus (the module-level helper) sees
        # the hash already in seen_hashes and does not re-save.
        mgr.save_to_corpus(seed)
        assert f._corpus_flux.total_additions == 1

    def test_eviction_wiring_present_at_both_prune_sites(self):
        """auto_minimize_corpus and deprioritize_near_duplicates both call
        record_eviction alongside their existing f._pruned_count bookkeeping.

        Behavioral tests exist elsewhere in this file for the (easy to
        trigger) addition/rejection paths; both prune paths need enough
        corpus/edge-tracker state to exercise for real that it isn't worth
        duplicating here, so this checks the wiring is present in source
        instead -- consistent with this repo's existing source-inspection
        regression tests (e.g. tests/test_lbr_wiring.py).
        """
        import inspect

        from fuzzer_tool.services import corpus_manager

        source = inspect.getsource(corpus_manager)
        assert "f._corpus_flux.record_eviction(removed)" in source
        assert "f._corpus_flux.record_eviction(len(to_remove))" in source

    def test_rejection_wiring_present_at_near_dup_site(self):
        """The Poisson-disk REJECT_NEAR_DUP branch calls record_rejection.

        See tests/test_regression_poisson_disk_admission.py for behavioral
        coverage of the admission decision itself.
        """
        import inspect

        from fuzzer_tool.services import corpus_manager

        source = inspect.getsource(corpus_manager)
        assert "f._corpus_flux.record_rejection()" in source


class TestSingletonEdgePreservation:
    """EdgeTracker._maybe_prune must not drop seeds that own unique edges."""

    def test_prune_preserves_singleton_edge_seed(self):
        """A seed with a unique edge survives even when over max_tracked_seeds."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        et = f._edge_tracker
        et.max_tracked_seeds = 2

        # Three seeds: A and B share edge 1, C has unique edge 2
        seed_a = b"seed_a_" + b"x" * 60
        seed_b = b"seed_b_" + b"y" * 60
        seed_c = b"seed_c_" + b"z" * 60

        ka = _seed_key(seed_a)
        kb = _seed_key(seed_b)
        kc = _seed_key(seed_c)

        et.seed_edges[ka] = {1}
        et.seed_edges[kb] = {1}
        et.seed_edges[kc] = {2}  # singleton edge

        et._maybe_prune()

        assert kc in et.seed_edges, (
            f"Singleton-edge seed C (key={kc}) was pruned from seed_edges. "
            f"Remaining: {list(et.seed_edges.keys())}"
        )

    def test_prune_drops_redundant_seed(self):
        """A fully-subsumed seed is the first to be pruned."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        et = f._edge_tracker
        et.max_tracked_seeds = 2

        seed_a = b"seed_a_" + b"x" * 60
        seed_b = b"seed_b_" + b"y" * 60
        seed_c = b"seed_c_" + b"z" * 60

        ka = _seed_key(seed_a)
        kb = _seed_key(seed_b)
        kc = _seed_key(seed_c)

        # A covers {1,2}; B and C both cover {1} (redundant)
        et.seed_edges[ka] = {1, 2}
        et.seed_edges[kb] = {1}
        et.seed_edges[kc] = {1}

        et._maybe_prune()

        # A must survive (covers unique edge 2)
        assert ka in et.seed_edges
        # One of B or C is dropped; the other may survive as filler
        assert len(et.seed_edges) == 2


class TestPostMinimizeCoverageVerification:
    """Post-pruning verification is a no-op when coverage is intact."""

    def test_post_minimize_no_regression_when_covered(self):
        """When all edges are covered, verification is a no-op."""
        f = MockFuzzer(Path(tempfile.mkdtemp()))
        et = f._edge_tracker
        et.cumulative_edges = {1, 2, 3}

        seed_a = b"seed_a_" + b"x" * 60
        ka = _seed_key(seed_a)
        et.seed_edges[ka] = {1, 2, 3}

        f.corpus = [seed_a]
        f.seed_meta = {seed_a: {"fuzz_count": 1, "coverage_edges": 3, "added_at": 100.0}}
        f.max_corpus = 1

        mgr = CorpusManager(f)
        mgr.auto_minimize_corpus()

        assert len(f.corpus) == 1
        assert seed_a in f.corpus
