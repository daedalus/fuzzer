"""Exhaustion propagation on the lineage tree (``--lineage-exhaust``).

soksolv ``fess_deadends.go``: a node is dead only when its own moves are
spent AND every child is dead. Here: a seed is *spent* after ``budget``
fuzz rounds without new coverage; it is *exhausted* when spent and every
child is exhausted. New coverage under a node revives its whole ancestor
chain. Maintained incrementally (O(depth)), checked against a from-scratch
recursive oracle.
"""

from __future__ import annotations

import inspect

import pytest

from fuzzer_tool.core.analyzers.analyzer_recurrence import Novelty
from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.seed_picker import SeedPicker

BUDGET = 3


def _chain(depth: int) -> LineageTree:
    tree = LineageTree()
    parent = None
    for i in range(depth + 1):
        tree.insert(parent, f"k{i}", ["havoc"], [0], 0)
        parent = f"k{i}"
    return tree


def _star(n_children: int) -> LineageTree:
    tree = LineageTree()
    tree.insert(None, "r", [], [], 0)
    for i in range(n_children):
        tree.insert("r", f"c{i}", ["havoc"], [0], 0)
    return tree


def _spend(tree: LineageTree, key: str) -> None:
    for _ in range(BUDGET):
        tree.record_fuzz(key, Novelty.NONE, BUDGET)


def _oracle(tree: LineageTree, spent: set[str]) -> set[str]:
    """Exhausted keys recomputed from scratch: spent and all children exhausted."""
    memo: dict[str, bool] = {}

    def dead(k: str) -> bool:
        if k in memo:
            return memo[k]
        memo[k] = False  # cycle guard: an unresolved cycle stays alive
        kids = [c for c in tree._children.get(k, ()) if c in tree.nodes]
        memo[k] = k in spent and all(dead(c) for c in kids)
        return memo[k]

    return {k for k in tree.nodes if dead(k)}


def _exhausted(tree: LineageTree) -> set[str]:
    return {k for k in tree.nodes if tree.is_exhausted(k)}


class TestCore:
    def test_leaf_exhausts_after_budget(self):
        tree = _chain(2)
        for _ in range(BUDGET - 1):
            tree.record_fuzz("k2", Novelty.NONE, BUDGET)
        assert not tree.is_exhausted("k2")
        tree.record_fuzz("k2", Novelty.NONE, BUDGET)
        assert tree.is_exhausted("k2")

    def test_unfuzzed_child_keeps_parent_alive(self):
        """Falsification of the DFS rule: an unfuzzed child is not dead."""
        tree = _star(2)
        _spend(tree, "r")
        _spend(tree, "c0")
        assert not tree.is_exhausted("r")
        _spend(tree, "c1")
        assert tree.is_exhausted("r")

    def test_parent_not_exhausted_until_own_budget_spent(self):
        tree = _star(1)
        _spend(tree, "c0")
        assert tree.is_exhausted("c0")
        assert not tree.is_exhausted("r")
        _spend(tree, "r")
        assert tree.is_exhausted("r")

    def test_propagation_is_order_independent(self):
        top_down = _chain(4)
        bottom_up = _chain(4)
        for i in range(5):
            _spend(top_down, f"k{i}")
            _spend(bottom_up, f"k{4 - i}")
        assert _exhausted(top_down) == _exhausted(bottom_up) == {f"k{i}" for i in range(5)}

    def test_new_coverage_revives_ancestors(self):
        tree = _chain(3)
        for i in range(4):
            _spend(tree, f"k{i}")
        assert tree.is_exhausted("k0")
        tree.record_fuzz("k3", Novelty.NEW, BUDGET)
        assert _exhausted(tree) == set()
        # Ancestors stay spent: re-spending the leaf re-kills the chain.
        _spend(tree, "k3")
        assert _exhausted(tree) == {f"k{i}" for i in range(4)}

    def test_insert_under_exhausted_node_revives_chain(self):
        tree = _chain(2)
        for i in range(3):
            _spend(tree, f"k{i}")
        tree.insert("k2", "k3", ["havoc"], [0], 1)
        assert _exhausted(tree) == set()

    def test_mark_exhausted_is_idempotent(self):
        tree = _star(1)
        tree.mark_exhausted("c0")
        tree.mark_exhausted("c0")
        tree.mark_exhausted("r")
        assert tree.nodes["r"].live == 0
        tree.revive("c0")
        assert tree.nodes["r"].live == 1
        assert not tree.is_exhausted("r")


class TestOracle:
    def _run(self, seed: int, n: int, steps: int) -> tuple[set[str], set[str]]:
        rng = RandPool(seed)
        tree = LineageTree()
        keys: list[str] = []
        spent: set[str] = set()
        for i in range(n):
            parent = keys[rng.randrange(len(keys))] if keys else None
            tree.insert(parent, f"n{i}", ["havoc"], [0], 0)
            keys.append(f"n{i}")
        for _ in range(steps):
            k = keys[rng.randrange(len(keys))]
            if rng.random() < 0.1:
                tree.revive(k)
                spent.discard(k)
                continue
            tree.mark_exhausted(k)
            spent.add(k)
        return _exhausted(tree), _oracle(tree, spent)

    def test_control_oracle_against_itself(self):
        """Hard Rule 46: the oracle must agree with a second run of itself."""
        _, a = self._run(7, 300, 600)
        _, b = self._run(7, 300, 600)
        assert a == b
        assert a  # non-trivial: something is exhausted

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_incremental_matches_oracle(self, seed):
        got, want = self._run(seed, 300, 600)
        assert got == want


class TestAdversarial:
    def test_unknown_key_is_noop(self):
        tree = _chain(1)
        tree.mark_exhausted("nope")
        tree.revive("nope")
        tree.record_fuzz("nope", Novelty.NONE, 1)
        assert not tree.is_exhausted("nope")

    def test_parent_cycle_terminates_and_stays_alive(self):
        """A corrupt a<->b cycle must not hang; an unresolved cycle stays alive."""
        tree = LineageTree()
        tree.insert(None, "a", [], [], 0)
        tree.insert("a", "b", [], [], 0)
        tree.nodes["a"].parent_key = "b"
        tree._children.setdefault("b", set()).add("a")
        tree.nodes["b"].live += 1
        tree.mark_exhausted("a")
        tree.mark_exhausted("b")
        assert not tree.is_exhausted("a")
        assert not tree.is_exhausted("b")
        tree.revive("a")

    def test_dropped_live_child_releases_parent(self):
        tree = _star(1)
        tree.mark_exhausted("r")
        assert not tree.is_exhausted("r")
        tree.prune_subtree("c0")
        assert tree.is_exhausted("r")

    def test_live_counter_never_negative(self):
        tree = _star(3)
        for _ in range(3):
            for k in ("c0", "c1", "c2", "r"):
                tree.mark_exhausted(k)
                tree.revive(k)
                tree.mark_exhausted(k)
        assert tree.nodes["r"].live == 0


class TestPersistence:
    def test_round_trip_through_rebuild(self):
        tree = _star(2)
        _spend(tree, "c0")
        _spend(tree, "c1")
        _spend(tree, "r")
        tree.record_fuzz("c1", Novelty.NONE, BUDGET)
        state = tree.exhaust_state()

        meta = {
            "r": {},
            "c0": {"parent_key": "r"},
            "c1": {"parent_key": "r"},
        }
        fresh = LineageTree()
        fresh.rebuild_from_meta(meta, lambda s: s)
        assert _exhausted(fresh) == set()
        fresh.load_exhaust(state, BUDGET)
        assert _exhausted(fresh) == _exhausted(tree) == {"r", "c0", "c1"}

    def test_load_ignores_garbage(self):
        tree = _star(1)
        tree.load_exhaust({"c0": "x", "zz": 9, 3: 4}, BUDGET)
        tree.load_exhaust("not a dict", BUDGET)
        assert _exhausted(tree) == set()


class _Fuzzer:
    def __init__(self, tree, exhaust):
        self._use_lineage_backtrack = True
        self._lineage = tree
        self.seed_meta = {}
        self._lineage_backtrack_decay = 0.5
        self._lineage_backtrack_min_fuzz = BUDGET
        self._lineage_exhaust = exhaust


def _weight(f, key, w=1.0, fuzz_count=100):
    return SeedPicker._weight_lineage_backtrack(
        object.__new__(SeedPicker), key, w, fuzz_count, f, {}
    )


class TestBacktrackWeight:
    def test_exhausted_node_penalised_by_depth(self):
        tree = _chain(2)
        for i in range(3):
            _spend(tree, f"k{i}")
        f = _Fuzzer(tree, True)
        assert _weight(f, "k2") == pytest.approx(0.5**2)

    def test_live_subtree_not_penalised(self):
        """Falsification vs DFS: no coverage meta, but an unfuzzed child is alive."""
        tree = _chain(2)
        _spend(tree, "k1")
        f = _Fuzzer(tree, True)
        assert _weight(f, "k1") == 1.0
        # The DFS path (flag off) penalises the same node.
        f_dfs = _Fuzzer(tree, False)
        assert _weight(f_dfs, "k1") < 1.0


class TestWiring:
    def test_cli_flag_reaches_fuzzer(self):
        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert '"--lineage-exhaust"' in src
        assert "lineage_exhaust=getattr" in src
        assert "lineage_exhaust" not in commands._HAIL_MARY_FLAGS

    def test_fuzzer_gates_on_backtrack(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        src = inspect.getsource(Fuzzer.__init__)
        assert "lineage_exhaust and self._use_lineage_backtrack" in src

    def test_round_feeds_outcome(self):
        from fuzzer_tool.services.fuzz_round import FuzzRound

        assert "_record_lineage_exhaust" in inspect.getsource(FuzzRound._credit_seed)
