"""Regression tests for incrementally-maintained lineage roots.

``MCTSSeedScheduler._roots`` derived the forest roots by scanning every node
in the tree, on every seed pick. That is O(corpus) on the fuzzer's
per-iteration hot path: measured at ~1.2ms on a 20k-node tree, which is the
entire per-execution budget at 800 eps, so enabling ``--mcts`` on a large
corpus roughly halved throughput.

``LineageTree`` now maintains the root set incrementally. These tests pin
both the equivalence (it must agree with the old scan under every
insert/prune sequence) and the complexity (it must not scale with corpus
size).
"""

import random
import time

import pytest

from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.core.schedulers.mcts import MCTSSeedScheduler


def _scan_roots(tree: LineageTree) -> list[str]:
    """The original O(n) definition, kept as the oracle."""
    return sorted(
        k for k, n in tree.nodes.items() if n.parent_key is None or n.parent_key not in tree.nodes
    )


def _random_tree(seed: int, size: int = 80, max_inactive: int = 50) -> LineageTree:
    rng = random.Random(seed)
    tree = LineageTree(max_inactive=max_inactive)
    keys: list[str] = []
    for i in range(size):
        key = f"k{i}"
        parent = None if (not keys or rng.random() < 0.2) else rng.choice(keys)
        tree.insert(parent, key, ["op"], [0], rng.randint(0, 3))
        keys.append(key)
        if rng.random() < 0.15:
            tree.prune_subtree(rng.choice(keys))
    return tree


class TestRootsEquivalence:
    def test_empty_tree_has_no_roots(self):
        assert LineageTree().roots() == []

    def test_single_root(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        assert tree.roots() == ["a"]

    def test_child_is_not_a_root(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", ["op"], [0], 1)
        assert tree.roots() == ["a"]

    def test_forest_is_multi_rooted(self):
        """Every imported corpus seed is inserted with parent_key=None."""
        tree = LineageTree()
        for k in ("a", "b", "c"):
            tree.insert(None, k, [], [], 1)
        assert sorted(tree.roots()) == ["a", "b", "c"]

    def test_unknown_parent_makes_an_orphan_root(self):
        tree = LineageTree()
        tree.insert("missing", "a", ["op"], [0], 1)
        assert tree.roots() == ["a"]

    def test_duplicate_insert_does_not_duplicate_a_root(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        tree.insert(None, "a", [], [], 5)
        assert tree.roots() == ["a"]

    @pytest.mark.parametrize("seed", range(25))
    def test_matches_full_scan_under_random_mutation(self, seed):
        tree = _random_tree(seed)
        assert sorted(tree.roots()) == _scan_roots(tree)

    @pytest.mark.parametrize("max_inactive", [0, 1, 5, 10_000])
    def test_matches_full_scan_across_pruning_pressure(self, max_inactive):
        """max_inactive=0 hard-drops immediately, exercising the _drop path."""
        tree = _random_tree(7, size=120, max_inactive=max_inactive)
        assert sorted(tree.roots()) == _scan_roots(tree)

    def test_dropped_node_promotes_its_children(self):
        """Hard-dropping a parent must make its surviving children roots."""
        tree = LineageTree(max_inactive=0)
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", ["op"], [0], 1)
        tree.insert("b", "c", ["op"], [0], 1)
        tree.prune_subtree("b")  # b and c go inactive, then hard-drop
        assert sorted(tree.roots()) == _scan_roots(tree)

    def test_rebuild_from_meta_resets_roots(self):
        """A stale root set would survive the rebuild and report dead keys."""
        tree = LineageTree()
        tree.insert(None, "old", [], [], 1)
        tree.rebuild_from_meta(
            {b"seed": {"parent_key": None, "new_edge_count": 1}},
            key_fn=lambda d: "fresh",
        )
        assert "old" not in tree.roots()
        assert sorted(tree.roots()) == _scan_roots(tree)

    def test_roots_never_reports_a_dropped_key(self):
        tree = _random_tree(3, size=100, max_inactive=0)
        assert all(k in tree.nodes for k in tree.roots())


class TestRootsComplexity:
    def _tree(self, n):
        rng = random.Random(1)
        tree = LineageTree()
        tree.insert(None, "r0", [], [], 1)
        keys = ["r0"]
        for i in range(n):
            key = f"k{i}"
            tree.insert(rng.choice(keys), key, ["havoc"], [0], rng.randint(0, 3))
            keys.append(key)
        return tree

    def _per_call_us(self, tree, calls=200):
        scheduler = MCTSSeedScheduler()
        eligible = set(tree.nodes)
        start = time.perf_counter()
        for _ in range(calls):
            scheduler.select(tree, eligible)
            scheduler.update(1.0)
        return (time.perf_counter() - start) / calls * 1e6

    def test_selection_cost_does_not_scale_with_corpus_size(self):
        """The defect this pins: cost grew linearly with the node count."""
        small = self._per_call_us(self._tree(500))
        large = self._per_call_us(self._tree(20_000))
        # Pre-fix this ratio was ~30x (37us -> 1206us). Allow generous slack
        # for a loaded machine while still failing on a reintroduced scan.
        assert large < small * 8, f"selection cost scaled with corpus: {small=} {large=}"

    def test_roots_lookup_is_not_linear(self):
        small, large = self._tree(500), self._tree(20_000)
        scheduler = MCTSSeedScheduler()

        def timed(tree):
            eligible = set(tree.nodes)
            start = time.perf_counter()
            for _ in range(500):
                scheduler._roots(tree, eligible)
            return time.perf_counter() - start

        assert timed(large) < timed(small) * 8

    def test_scheduler_still_selects_correctly_on_a_large_tree(self):
        """Speed must not have come at the cost of returning valid keys."""
        tree = self._tree(5_000)
        scheduler = MCTSSeedScheduler()
        eligible = set(tree.nodes)
        for _ in range(200):
            key = scheduler.select(tree, eligible)
            assert key is None or key in eligible
            scheduler.update(1.0)


class TestRebuildRepopulatesRoots:
    """``rebuild_from_meta`` builds nodes directly instead of calling
    ``insert``, so it must recompute the root set itself. If it does not,
    ``roots()`` comes back empty after a ``--resume`` and the scheduler's
    fallback treats every eligible seed as a root — silently degrading the
    tree descent back to the flat O(corpus) scan this change removed."""

    def test_roots_survive_a_rebuild(self):
        tree = LineageTree()
        tree.rebuild_from_meta(
            {
                b"a": {"parent_key": None, "new_edge_count": 2},
                b"b": {"parent_key": "a", "new_edge_count": 1},
                b"c": {"parent_key": None, "new_edge_count": 1},
            },
            key_fn=lambda d: d.decode(),
        )
        assert sorted(tree.roots()) == ["a", "c"]
        assert sorted(tree.roots()) == _scan_roots(tree)

    def test_orphan_with_missing_parent_becomes_a_root(self):
        """Only decidable after every node is built, hence the second pass."""
        tree = LineageTree()
        tree.rebuild_from_meta(
            {b"b": {"parent_key": "gone", "new_edge_count": 1}},
            key_fn=lambda d: d.decode(),
        )
        assert tree.roots() == ["b"]

    def test_rebuild_is_idempotent_for_roots(self):
        meta = {
            b"a": {"parent_key": None, "new_edge_count": 1},
            b"b": {"parent_key": "a", "new_edge_count": 1},
        }
        tree = LineageTree()
        tree.rebuild_from_meta(meta, key_fn=lambda d: d.decode())
        first = sorted(tree.roots())
        tree.rebuild_from_meta(meta, key_fn=lambda d: d.decode())
        assert sorted(tree.roots()) == first == ["a"]

    def test_scheduler_descends_a_rebuilt_tree(self):
        """End-to-end: a resumed tree must still drive a real UCT descent."""
        meta = {
            f"n{i}".encode(): {
                "parent_key": None if i == 0 else f"n{(i - 1) // 2}",
                "new_edge_count": 1,
            }
            for i in range(31)
        }
        tree = LineageTree()
        tree.rebuild_from_meta(meta, key_fn=lambda d: d.decode())
        assert tree.roots() == ["n0"]

        scheduler = MCTSSeedScheduler()
        eligible = set(tree.nodes)
        picked = set()
        for _ in range(200):
            key = scheduler.select(tree, eligible)
            if key is not None:
                picked.add(key)
            scheduler.update(1.0)
        assert len(picked) > 1, "resumed tree produced no meaningful descent"
