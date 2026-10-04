"""Lineage shape metrics (docs/handover/handover_maze_algorithms_2026-09-24.md, item 1).

Leaf / corridor fractions, max depth and mean unary-chain length over the
live lineage forest: the maze characterisation table, applied to seed
topology. A run's fingerprint says which Growing Tree policy its seed
picker behaved as (corridor-heavy = newest, bushy = random / oldest).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.lineage import LineageShape, LineageTree
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.stats import StatsReporter


def _chain(n: int) -> LineageTree:
    tree = LineageTree()
    tree.insert(None, "k0", [], [], 0)
    for i in range(1, n):
        tree.insert(f"k{i - 1}", f"k{i}", [], [], 0)
    return tree


def _star(leaves: int) -> LineageTree:
    tree = LineageTree()
    tree.insert(None, "root", [], [], 0)
    for i in range(leaves):
        tree.insert("root", f"leaf{i}", [], [], 0)
    return tree


def _balanced(depth: int) -> LineageTree:
    tree = LineageTree()
    tree.insert(None, "n1", [], [], 0)
    for i in range(2, 2 ** (depth + 1)):
        tree.insert(f"n{i // 2}", f"n{i}", [], [], 0)
    return tree


class TestShapeKnownTopologies:
    def test_empty_tree_is_all_zero(self):
        assert LineageTree().shape() == LineageShape(0, 0.0, 0.0, 0, 0.0)

    def test_single_root_is_one_leaf(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 0)

        assert tree.shape() == LineageShape(1, 1.0, 0.0, 0, 0.0)

    def test_chain_is_corridor(self):
        shape = _chain(10).shape()

        assert shape.live == 10
        assert shape.leaf_frac == pytest.approx(0.1)
        assert shape.corridor_frac == pytest.approx(0.9)
        assert shape.max_depth == 9
        assert shape.mean_chain == pytest.approx(9.0)

    def test_star_is_all_leaves(self):
        shape = _star(5).shape()

        assert shape.leaf_frac == pytest.approx(5 / 6)
        assert shape.corridor_frac == 0.0
        assert shape.max_depth == 1
        assert shape.mean_chain == 0.0

    def test_balanced_binary_tree_has_no_corridor(self):
        shape = _balanced(3).shape()

        assert shape.live == 15
        assert shape.leaf_frac == pytest.approx(8 / 15)
        assert shape.corridor_frac == 0.0
        assert shape.max_depth == 3

    def test_two_chains_off_a_fork_average_their_lengths(self):
        """r forks into a 3-chain (a1..a3) and a lone leaf b1."""
        tree = LineageTree()
        tree.insert(None, "r", [], [], 0)
        tree.insert("r", "a1", [], [], 0)
        tree.insert("a1", "a2", [], [], 0)
        tree.insert("a2", "a3", [], [], 0)
        tree.insert("r", "b1", [], [], 0)
        shape = tree.shape()

        # a1, a2 have one child each: one run of length 2. r forks; a3, b1 are leaves.
        assert shape.corridor_frac == pytest.approx(2 / 5)
        assert shape.mean_chain == pytest.approx(2.0)

    def test_forest_pools_every_root(self):
        tree = _chain(3)
        tree.insert(None, "other", [], [], 0)

        shape = tree.shape()

        assert shape.live == 4
        assert shape.leaf_frac == pytest.approx(2 / 4)


class TestShapeAdversarial:
    def test_pruned_nodes_are_not_live(self):
        tree = _chain(5)
        tree.prune_subtree("k3")

        shape = tree.shape()

        assert shape.live == 3
        assert shape.max_depth == 2
        assert shape.leaf_frac == pytest.approx(1 / 3)

    def test_everything_pruned_is_empty_not_a_division_error(self):
        tree = _chain(3)
        tree.prune_subtree("k0")

        assert tree.shape() == LineageShape(0, 0.0, 0.0, 0, 0.0)

    def test_deep_chain_does_not_recurse(self):
        """Iterative walk: a chain far past the interpreter recursion limit."""
        shape = _chain(5000).shape()

        assert shape.max_depth == 4999
        assert shape.mean_chain == pytest.approx(4999.0)

    def test_fractions_are_proportions(self):
        for tree in (_chain(7), _star(7), _balanced(4)):
            shape = tree.shape()

            assert 0.0 <= shape.leaf_frac <= 1.0
            assert 0.0 <= shape.corridor_frac <= 1.0
            assert shape.leaf_frac + shape.corridor_frac <= 1.0

    def test_rebuilt_forest_with_orphan_children_terminates(self):
        """A child whose parent never appears in seed_meta is an orphan root."""
        tree = LineageTree()
        tree.rebuild_from_meta(
            {b"a": {"parent_key": "gone"}, b"b": {"parent_key": "a"}},
            lambda raw: raw.decode(),
        )

        assert tree.shape().live == 2


FIND_RATE = 1 / 3
"""Chance a pick yields a child. Finds are rare in a real run; with a find on
every pick, round robin degenerates to a chain (its cursor always sits on the
newest seed). A fixed every-Nth-pick phase aliases with the cursor the same
way, so finds are drawn from a seeded stream instead."""

SIM_SEEDS = (1, 2, 3)


def _grow(policy: str, n: int, seed: int) -> LineageShape:
    """Growing Tree over *n* seeds: the picked seed is the parent of each find."""
    pick_rng = RandPool(seed=seed)
    find_rng = RandPool(seed=seed + 1000)
    tree = LineageTree()
    tree.insert(None, "k0", [], [], 0)
    keys = ["k0"]
    step = 0
    while len(keys) < n:
        if policy == "newest":
            parent = keys[-1]
        elif policy == "oldest":
            parent = keys[step % len(keys)]
        else:
            parent = keys[pick_rng.randint(0, len(keys) - 1)]
        step += 1
        if find_rng.random() >= FIND_RATE:
            continue
        tree.insert(parent, f"k{len(keys)}", [], [], 0)
        keys.append(f"k{len(keys)}")
    return tree.shape()


class TestFingerprintSeparatesPolicies:
    """Falsifier from the handover: if the metrics did not separate the
    policies they would not be a fingerprint. Newest must read as corridor.

    Measured limit: oldest and random overlap on every metric (leaf 0.47-0.69
    vs 0.47-0.50, corridor 0.10-0.31, depth 7-12), so the fingerprint says
    'newest-like or not', not which of the other two. Not asserted either way.
    """

    @pytest.mark.parametrize("seed", SIM_SEEDS)
    def test_newest_reads_as_corridor_others_do_not(self, seed):
        newest = _grow("newest", 200, seed)
        others = [_grow("oldest", 200, seed), _grow("random", 200, seed)]

        assert newest.corridor_frac > 0.9
        assert newest.max_depth == 199
        for other in others:
            assert other.corridor_frac < 0.5
            assert other.max_depth * 3 < newest.max_depth


def _stats_fixture(n: int):
    meta = {bytes([i + 1]): {"parent_key": f"{i - 1:02x}" if i else None} for i in range(n)}
    return SimpleNamespace(
        seed_meta=meta,
        _lineage=None,
        _seed_key=lambda raw: f"{raw[0] - 1:02x}",
        total_time=0.0,
        corpus=list(meta),
    )


class TestStatsLine:
    def test_summary_prints_shape_without_the_lineage_flag(self, capsys):
        f = _stats_fixture(4)

        StatsReporter._print_lineage_shape(StatsReporter.__new__(StatsReporter), f)

        out = capsys.readouterr().out
        assert "Lineage shape" in out
        assert "corridor 0.75" in out

    def test_summary_silent_when_no_seed_has_a_parent(self, capsys):
        f = SimpleNamespace(seed_meta={b"a": {}}, _lineage=None, _seed_key=lambda r: r.decode())

        StatsReporter._print_lineage_shape(StatsReporter.__new__(StatsReporter), f)

        assert capsys.readouterr().out == ""

    def test_summary_survives_non_dict_meta(self, capsys):
        f = SimpleNamespace(
            seed_meta={b"a": None, b"b": {"parent_key": None}},
            _lineage=None,
            _seed_key=lambda r: r.decode(),
        )

        StatsReporter._print_lineage_shape(StatsReporter.__new__(StatsReporter), f)

        assert capsys.readouterr().out == ""
