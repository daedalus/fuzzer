"""Tests for bootstrap percolation corpus minimization."""

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.percolation import (
    CoverageRegime,
    bootstrap_minimize_corpus,
)


def _seed_key(seed: bytes) -> str:
    """Match CorpusManager.seed_key (xxhash 16-hex)."""
    import xxhash

    return xxhash.xxh64(seed).hexdigest()[:16]


def _make_corpus_and_tracker(spec: dict) -> tuple:
    """Build a corpus + EdgeTracker from a {seed_name: [edges]} spec."""
    corpus = []
    name_to_seed = {}
    for name in spec:
        seed = f"seed-{name}".encode()
        name_to_seed[name] = seed
        corpus.append(seed)

    et = EdgeTracker(max_tracked_seeds=100)
    for name, edges in spec.items():
        sk = _seed_key(name_to_seed[name])
        et.seed_edges[sk] = set(edges)
        et.seed_hit_counts[sk] = {}

    return corpus, et, name_to_seed


class TestBootstrapPercolation:
    def test_no_removal_when_all_seeds_have_unique_edges(self):
        corpus, et, _ = _make_corpus_and_tracker({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=1)
        assert set(kept) == set(corpus)
        assert removed == []

    def test_transitive_redundancy_removal(self):
        # Chain: A={1,2}, B={2,3}, C={3,4}, D={4,5}
        # B and C cover edge 3 only jointly, so each is redundant alone but
        # both cannot go. Removed one at a time (fewest edges, ties by corpus
        # order): B goes first, after which C uniquely owns edge 3 and stays.
        corpus, et, names = _make_corpus_and_tracker(
            {"A": [1, 2], "B": [2, 3], "C": [3, 4], "D": [4, 5]}
        )
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=1)
        assert set(kept) == {names["A"], names["C"], names["D"]}
        assert removed == [names["B"]]

    def test_k1_preserves_covered_edges_in_chain_fixture(self):
        corpus, et, names = _make_corpus_and_tracker(
            {"A": [1, 2], "B": [2, 3], "C": [3, 4], "D": [4, 5]}
        )
        kept, _ = bootstrap_minimize_corpus(corpus, et, k=1)
        covered = set().union(*(et.seed_edges[_seed_key(s)] for s in kept))
        assert covered == {1, 2, 3, 4, 5}

    def test_k1_never_loses_coverage_on_random_corpora(self):
        import random

        for trial in range(200):
            r = random.Random(trial)
            universe = r.randint(10, 80)
            spec = {
                f"s{i}": r.sample(range(universe), r.randint(1, min(12, universe)))
                for i in range(r.randint(2, 40))
            }
            corpus, et, _ = _make_corpus_and_tracker(spec)
            before = set().union(*(set(e) for e in spec.values()))
            kept, removed = bootstrap_minimize_corpus(corpus, et, k=1)
            after = set().union(*(et.seed_edges[_seed_key(s)] for s in kept)) if kept else set()
            assert after == before, f"trial {trial} lost {before - after}"
            assert set(kept) | set(removed) == set(corpus)
            assert not set(kept) & set(removed)
            # Fixed point: every survivor owns at least one edge.
            for s in kept:
                mine = et.seed_edges[_seed_key(s)]
                others = set().union(
                    *(et.seed_edges[_seed_key(o)] for o in kept if o != s)
                ) if len(kept) > 1 else set()
                assert mine - others, f"trial {trial}: redundant survivor"

    def test_k1_identical_seeds_keep_exactly_one(self):
        corpus, et, names = _make_corpus_and_tracker({"A": [1, 2], "B": [1, 2], "C": [1, 2]})
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=1)
        assert len(kept) == 1
        assert len(removed) == 2

    def test_k1_preserves_corpus_order_of_survivors(self):
        corpus, et, names = _make_corpus_and_tracker(
            {"A": [1], "B": [1, 2], "C": [2, 3], "D": [3]}
        )
        kept, _ = bootstrap_minimize_corpus(corpus, et, k=1)
        assert kept == [s for s in corpus if s in set(kept)]

    def test_k2_keeps_batch_k_rigid_semantics(self):
        # k >= 2 is the k-rigid core: every seed below the threshold goes in
        # the same round. B owns 0 edges and is removed; A and C each own 2+.
        corpus, et, names = _make_corpus_and_tracker(
            {"A": [1, 2, 3], "B": [3, 4], "C": [4, 5, 6]}
        )
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=2)
        assert set(kept) == {names["A"], names["C"]}
        assert removed == [names["B"]]

    def test_k_value_filters(self):
        corpus, et, names = _make_corpus_and_tracker({"A": [1, 2], "B": [2, 3], "C": [3, 4]})
        # k=1: A unique={1}, B unique={}, C unique={4} → remove B
        #       Remaining: A={1,2}, C={3,4} → both have 2 unique → fixed point.
        kept1, removed1 = bootstrap_minimize_corpus(corpus, et, k=1)
        assert set(kept1) == {names["A"], names["C"]}
        assert set(removed1) == {names["B"]}

        # k=2: A unique={1} (count=1<2), B unique={} (count=0<2),
        #       C unique={4} (count=1<2) → all removed.
        kept2, removed2 = bootstrap_minimize_corpus(corpus, et, k=2)
        assert kept2 == []
        assert set(removed2) == set(corpus)

    def test_k_value_removes_all_when_unreachable(self):
        corpus, et, _ = _make_corpus_and_tracker({"A": [1, 2], "B": [2, 3]})
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=2)
        assert kept == []
        assert set(removed) == set(corpus)

    def test_empty_corpus(self):
        kept, removed = bootstrap_minimize_corpus([], EdgeTracker(max_tracked_seeds=10))
        assert kept == []
        assert removed == []

    def test_singleton_corpus(self):
        corpus, et, _ = _make_corpus_and_tracker({"A": [1, 2]})
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=1)
        assert kept == corpus
        assert removed == []

    def test_seeds_with_no_tracked_edges_removed(self):
        corpus, et, _ = _make_corpus_and_tracker({"A": [1, 2]})
        orphan = b"orphan-seed"
        corpus.append(orphan)
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=1)
        assert orphan not in kept
        assert orphan in removed

    def test_idempotent(self):
        corpus, et, _ = _make_corpus_and_tracker(
            {"A": [1, 2], "B": [2, 3], "C": [3, 4], "D": [4, 5]}
        )
        kept1, _ = bootstrap_minimize_corpus(corpus, et, k=1)
        kept2, removed2 = bootstrap_minimize_corpus(kept1, et, k=1)
        assert set(kept2) == set(kept1)
        assert removed2 == []

    def test_no_seed_edges(self):
        et = EdgeTracker(max_tracked_seeds=10)
        corpus = [b"a", b"b", b"c"]
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=1)
        assert set(kept) == set(corpus)
        assert removed == []

    def test_coverage_regime_importable_from_percolation(self):
        assert CoverageRegime.SUBCRITICAL.value == "subcritical"
        assert CoverageRegime.CRITICAL.value == "critical"
        assert CoverageRegime.SUPERCRITICAL.value == "supercritical"
