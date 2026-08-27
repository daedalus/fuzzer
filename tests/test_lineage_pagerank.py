"""Tests for ``LineageTree.pagerank_credit``.

``subtree_weight`` sums descendant weights discounted by depth, so it grows
with fan-out: enough one-edge children outrank a single high-yield child.
``pagerank_credit`` divides each child's contribution by its sibling count,
which makes a branch's rank independent of how many attempts it took.
"""

import pytest

from fuzzer_tool.core.lineage import GAMMA, LineageTree


def _two_branches(width: int, narrow_weight: int = 12) -> LineageTree:
    """A: one child worth ``narrow_weight``. B: ``width`` children worth 1."""
    t = LineageTree()
    t.insert(None, "A", [], [], 0)
    t.insert(None, "B", [], [], 0)
    t.insert("A", "A1", ["havoc"], [0], narrow_weight)
    for i in range(width):
        t.insert("B", f"B{i}", ["havoc"], [0], 1)
    return t


class TestFanOutInvariance:
    @pytest.mark.parametrize("width", [5, 13, 20, 50])
    def test_credit_ratio_does_not_move_with_fan_out(self, width):
        t = _two_branches(width)
        credit = t.pagerank_credit()
        assert credit["A"] / credit["B"] == pytest.approx(12.0, rel=1e-6)

    def test_subtree_weight_flips_where_credit_does_not(self):
        """Pins the behaviour the new measure exists to correct."""
        narrow = _two_branches(5)
        wide = _two_branches(50)
        assert narrow.subtree_weight("A") > narrow.subtree_weight("B")
        assert wide.subtree_weight("A") < wide.subtree_weight("B")
        for t in (narrow, wide):
            c = t.pagerank_credit()
            assert c["A"] > c["B"]

    def test_yield_per_mutation_orders_the_branches(self):
        t = LineageTree()
        t.insert(None, "root", [], [], 0)
        # Same total yield, different attempt counts.
        t.insert("root", "efficient", [], [], 0)
        t.insert("efficient", "e1", ["havoc"], [0], 20)
        t.insert("root", "wasteful", [], [], 0)
        for i in range(20):
            t.insert("wasteful", f"w{i}", ["havoc"], [0], 1)
        c = t.pagerank_credit()
        assert c["efficient"] > c["wasteful"]


class TestDistribution:
    def test_normalized(self):
        c = _two_branches(20).pagerank_credit()
        assert sum(c.values()) == pytest.approx(1.0)

    def test_all_nodes_present(self):
        t = _two_branches(7)
        assert set(t.pagerank_credit()) == set(t.nodes)

    def test_productive_leaf_outranks_its_barren_siblings(self):
        t = _two_branches(10)
        c = t.pagerank_credit()
        assert c["A1"] > c["B0"]

    def test_deterministic(self):
        t = _two_branches(9)
        assert t.pagerank_credit() == t.pagerank_credit()


class TestDegenerate:
    def test_empty_forest(self):
        assert LineageTree().pagerank_credit() == {}

    def test_no_weight_anywhere(self):
        t = LineageTree()
        t.insert(None, "r", [], [], 0)
        t.insert("r", "c", ["havoc"], [0], 0)
        assert t.pagerank_credit() == {"r": 0.0, "c": 0.0}

    def test_single_node(self):
        t = LineageTree()
        t.insert(None, "r", [], [], 5)
        assert t.pagerank_credit() == {"r": pytest.approx(1.0)}

    def test_pruned_nodes_leave_the_distribution(self):
        t = _two_branches(10)
        t.prune_subtree("B0")
        c = t.pagerank_credit()
        assert "B0" not in c
        assert sum(c.values()) == pytest.approx(1.0)

    def test_pruned_siblings_stop_diluting_survivors(self):
        """The divisor counts live children only — soft-deleted ones would
        otherwise keep taking a share of the parent's inbound credit."""
        t = _two_branches(10)
        before = t.pagerank_credit()["B1"]
        for i in (0, 2, 3, 4):
            t.prune_subtree(f"B{i}")
        assert t.pagerank_credit()["B1"] > before

    def test_damping_defaults_to_gamma(self):
        t = _two_branches(6)
        assert t.pagerank_credit() == t.pagerank_credit(damping=GAMMA)

    def test_deep_chain_terminates(self):
        t = LineageTree()
        t.insert(None, "n0", [], [], 0)
        for i in range(1, 400):
            t.insert(f"n{i - 1}", f"n{i}", ["havoc"], [0], 1)
        c = t.pagerank_credit()
        assert sum(c.values()) == pytest.approx(1.0)
        # Credit accumulates toward the root of a pure chain.
        assert c["n0"] > c["n399"]
