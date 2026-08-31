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
        # Round 0: A unique={1}, B unique={}, C unique={}, D unique={5}
        #          → remove B, C
        # Round 1: A={1,2}, D={4,5} → A unique={1,2}, D unique={4,5} → fixed point
        corpus, et, names = _make_corpus_and_tracker(
            {"A": [1, 2], "B": [2, 3], "C": [3, 4], "D": [4, 5]}
        )
        kept, removed = bootstrap_minimize_corpus(corpus, et, k=1)
        assert set(kept) == {names["A"], names["D"]}
        assert set(removed) == {names["B"], names["C"]}

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
