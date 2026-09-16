"""Tests for LineageTree.batch_lca_distances (Tarjan offline LCA).

See the "other tree structures in the codebase" section of
docs/handover/handover_trees.md for the motivating measurement: the
mutation lineage forest is not a Dyck-path parse tree, and a realistic
mostly-chain lineage has depth ~ Theta(n), so naive per-pair
``lca_distance`` calls (as seed_picker's diversity term makes, up to
n * sample_cap of them per pass) cost O(n^2) in the corpus size. This
file exists to nail down correctness against the pre-existing
``lca_distance`` before trusting the O(n) replacement.
"""

import random

import pytest

from fuzzer_tool.core.lineage import LineageTree


def _build_random_forest(n: int, n_roots: int, seed: int) -> tuple[LineageTree, list[str]]:
    rng = random.Random(seed)
    tree = LineageTree()
    keys: list[str] = []
    roots = [f"r{i}" for i in range(n_roots)]
    for r in roots:
        tree.insert(None, r, [], [], rng.randint(0, 3))
        keys.append(r)
    for i in range(n - n_roots):
        parent = rng.choice(keys)
        key = f"n{i}"
        tree.insert(parent, key, ["op"], [0], rng.randint(0, 3))
        keys.append(key)
    return tree, keys


def _build_chain_like_tree(n: int, branch_prob: float = 0.15, seed: int = 0):
    rng = random.Random(seed)
    tree = LineageTree()
    keys = ["root"]
    tree.insert(None, "root", [], [], 1)
    frontier = ["root"]
    for i in range(1, n):
        parent = rng.choice(frontier) if rng.random() < branch_prob else frontier[-1]
        key = f"n{i}"
        tree.insert(parent, key, ["op"], [0], rng.randint(0, 3))
        keys.append(key)
        frontier.append(key)
        if len(frontier) > 50:
            frontier.pop(0)
    return tree, keys


class TestBatchLcaDistancesCorrectness:
    """Cross-check against the pre-existing, trusted lca_distance."""

    @pytest.mark.parametrize(
        ("n", "n_roots"),
        [
            (50, 1),  # single tree
            (50, 5),  # small forest
            (200, 1),
            (200, 10),
            (500, 3),
            (30, 30),  # forest of singleton/near-singleton trees
        ],
    )
    def test_matches_pairwise_lca_distance(self, n, n_roots):
        tree, keys = _build_random_forest(n, n_roots, seed=n * 1000 + n_roots)
        rng = random.Random(7)
        pairs = [(rng.choice(keys), rng.choice(keys)) for _ in range(400)]
        pairs += [("unknown-a", keys[0]), (keys[0], "unknown-b"), (keys[0], keys[0])]

        batch = tree.batch_lca_distances(pairs)

        for a, b in pairs:
            if a not in tree.nodes or b not in tree.nodes:
                expected = -1
            elif a == b:
                expected = 0
            else:
                expected = tree.lca_distance(a, b)
            assert batch[(a, b)] == expected, f"mismatch for ({a}, {b})"

    def test_matches_on_mostly_chain_lineage(self):
        """The realistic shape: mostly-linear growth with occasional
        branching, where depth grows almost linearly with n -- exactly
        the case naive lca_distance is quadratic on."""
        tree, keys = _build_chain_like_tree(600, seed=3)
        rng = random.Random(9)
        pairs = [(rng.choice(keys), rng.choice(keys)) for _ in range(2000)]
        batch = tree.batch_lca_distances(pairs)
        for a, b in pairs:
            expected = tree.lca_distance(a, b)
            assert batch[(a, b)] == expected

    def test_distance_sum_matches_naive_loop(self):
        """A second, coarser check on a larger tree: the *sum* of all
        resolved distances must match a naive loop exactly -- catches
        any systematic off-by-one in the LCA depth (the union-before-
        query-check bug this method's first draft had, which added a
        constant +2 per pair whose true LCA was an ancestor of one of
        the two, not the two nodes' most recent common one)."""
        tree, keys = _build_chain_like_tree(1500, seed=5)
        rng = random.Random(11)
        pairs = [(rng.choice(keys), rng.choice(keys)) for _ in range(5000)]
        naive_sum = sum(max(tree.lca_distance(a, b), 0) for a, b in pairs)
        batch = tree.batch_lca_distances(pairs)
        # Sum over the original (possibly-duplicated) pair list, not
        # over batch.values() -- the returned dict is keyed by (a, b),
        # so a repeated pair in the input collapses to one entry there,
        # which would silently under-count relative to the naive loop.
        batch_sum = sum(max(batch[(a, b)], 0) for a, b in pairs)
        assert batch_sum == naive_sum


class TestBatchLcaDistancesEdgeCases:
    def test_empty_pairs(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        assert tree.batch_lca_distances([]) == {}

    def test_self_pair_known_key(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        assert tree.batch_lca_distances([("a", "a")]) == {("a", "a"): 0}

    def test_self_pair_unknown_key(self):
        tree = LineageTree()
        assert tree.batch_lca_distances([("ghost", "ghost")]) == {("ghost", "ghost"): -1}

    def test_unknown_key_pairs_resolve_to_minus_one(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        result = tree.batch_lca_distances([("a", "nope"), ("nope", "a"), ("x", "y")])
        assert result == {("a", "nope"): -1, ("nope", "a"): -1, ("x", "y"): -1}

    def test_disconnected_roots_in_same_forest(self):
        """Two separate trees in one forest: must resolve to -1, not to
        a bogus shared ancestor (the finished-set-leaking bug this
        method's first draft had across components)."""
        tree = LineageTree()
        tree.insert(None, "r1", [], [], 1)
        tree.insert("r1", "r1c", [], [], 1)
        tree.insert(None, "r2", [], [], 1)
        tree.insert("r2", "r2c", [], [], 1)
        result = tree.batch_lca_distances(
            [("r1", "r2"), ("r1c", "r2c"), ("r1", "r1c"), ("r2", "r2c")]
        )
        assert result[("r1", "r2")] == -1
        assert result[("r1c", "r2c")] == -1
        assert result[("r1", "r1c")] == 1
        assert result[("r2", "r2c")] == 1

    def test_duplicate_pairs_in_input(self):
        tree, keys = _build_random_forest(20, 1, seed=1)
        pairs = [(keys[0], keys[5])] * 10
        result = tree.batch_lca_distances(pairs)
        assert result[(keys[0], keys[5])] == tree.lca_distance(keys[0], keys[5])

    def test_reversed_pair_order_gives_same_distance(self):
        tree, keys = _build_random_forest(30, 1, seed=2)
        a, b = keys[3], keys[17]
        r1 = tree.batch_lca_distances([(a, b)])
        r2 = tree.batch_lca_distances([(b, a)])
        assert r1[(a, b)] == r2[(b, a)]

    def test_does_not_hang_on_corrupted_mutual_parent_cycle(self):
        """rebuild_from_meta on untrusted metadata can produce a
        parent-pointer cycle; batch_lca_distances must terminate and
        must not corrupt resolution of the rest of the forest."""
        tree = LineageTree()
        tree.insert(None, "root", [], [], 1)
        tree.insert("root", "a", [], [], 1)
        tree.insert("a", "b", [], [], 1)
        tree.insert("root", "c", [], [], 1)
        # Corrupt: a and b now point at each other, orphaned from root.
        tree.nodes["a"].parent_key = "b"
        tree.nodes["b"].parent_key = "a"
        tree._children.setdefault("b", set()).add("a")
        tree._children["root"].discard("a")

        result = tree.batch_lca_distances(
            [("a", "b"), ("root", "c"), ("c", "a"), ("root", "root")]
        )
        assert result[("root", "c")] == tree.lca_distance("root", "c")
        assert result[("c", "a")] == -1
        assert result[("root", "root")] == 0

    def test_isolated_self_parent_node(self):
        """A node whose parent_key equals its own key (possible after a
        corrupted rebuild) must not hang and must not poison other
        pairs."""
        tree = LineageTree()
        tree.insert(None, "root", [], [], 1)
        tree.nodes["root"].parent_key = None  # sanity: real root stays clean
        tree.insert("root", "child", [], [], 1)
        ghost = tree.nodes["child"]
        # Simulate a corrupted node with parent_key == its own key,
        # disconnected from the real tree.
        from fuzzer_tool.core.lineage import LineageNode

        tree.nodes["ghost"] = LineageNode(
            key="ghost", parent_key="ghost", depth=0, node_weight=0,
            child_ops=[], child_sites=[], seq=999,
        )
        tree._children.setdefault("ghost", set()).add("ghost")

        result = tree.batch_lca_distances([("root", "child"), ("ghost", "root")])
        assert result[("root", "child")] == 1
        assert result[("ghost", "root")] == -1


class TestBatchLcaMatchesSeedPickerDiversityPattern:
    """Reproduces the exact peer-pool-sampling access pattern
    services/seed_picker.py's ``_compute_weights`` diversity block uses
    (one lca_distance call per (seed, sampled-peer) pair, per-seed
    average, then a diversity multiplier) and checks that switching the
    distance lookup from per-pair ``lca_distance`` calls to one
    ``batch_lca_distances`` call produces bit-identical per-seed
    diversity multipliers. This is the property the seed_picker.py
    wiring change relies on -- if this drifts, seed scoring changes."""

    def test_per_seed_diversity_multiplier_matches(self):
        tree, all_sk = _build_chain_like_tree(300, seed=13)
        max_depth = max(node.depth for node in tree.nodes.values())
        rng = random.Random(21)
        sample_cap = 64
        n_sk = len(all_sk)
        pool_idx = rng.sample(range(n_sk), sample_cap + 1)

        def diversity_via(distance_fn):
            out = {}
            for i, sk_i in enumerate(all_sk):
                sample = [all_sk[j] for j in pool_idx if j != i][:sample_cap]
                valid = [d for d in (distance_fn(sk_i, k) for k in sample) if d >= 0]
                avg = sum(valid) / len(valid) if valid else 0.0
                out[sk_i] = 1.0 + 0.5 * min(avg / (2.0 * max_depth), 1.0)
            return out

        naive = diversity_via(tree.lca_distance)

        # batch path, mirroring seed_picker.py's wiring exactly
        all_pairs = []
        per_seed_samples = []
        for i, sk_i in enumerate(all_sk):
            sample = [all_sk[j] for j in pool_idx if j != i][:sample_cap]
            per_seed_samples.append(sample)
            all_pairs.extend((sk_i, k) for k in sample)
        batch_dist = tree.batch_lca_distances(all_pairs)
        batched = {}
        for i, sk_i in enumerate(all_sk):
            sample = per_seed_samples[i]
            valid = [d for d in (batch_dist[(sk_i, k)] for k in sample) if d >= 0]
            avg = sum(valid) / len(valid) if valid else 0.0
            batched[sk_i] = 1.0 + 0.5 * min(avg / (2.0 * max_depth), 1.0)

        assert naive == batched
