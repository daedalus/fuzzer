"""Regression: lineage walks must terminate on self/cyclic parent chains.

Found live on a fresh-corpus `--elo all` campaign: a no-op mutation can
make ``_seed_key(parent) == _seed_key(child)``, so insert() filed a node
whose parent_key pointed at ITSELF while being promoted to a root. The
first `_compute_weights` LCA-diversity pass then spun forever inside
``lca``'s final walk (main thread, no stats line, SIGTERM ignored by the
graceful-shutdown handler).

Contract after the fix:
- insert() stores parent_key=None when promoted to a root (no dangling
  or self references);
- ancestors()/lca() bound their walks by the node count, so even an
  externally-corrupted forest degrades to "disconnected" instead of
  hanging the fuzzer.
"""

from fuzzer_tool.core.lineage import LineageNode, LineageTree


def _node(key, parent_key, depth):
    return LineageNode(
        key=key,
        parent_key=parent_key,
        depth=depth,
        node_weight=0,
        child_ops=[],
        child_sites=[],
        seq=0,
    )


class TestSelfParentInsert:
    def test_insert_self_parent_promotes_to_root(self):
        """child_key == parent_key (no-op mutation hash) must store a root,
        not a self-referential node."""
        t = LineageTree()
        t.insert("k", "k", [], [], 3)
        assert t.nodes["k"].parent_key is None
        assert t.nodes["k"].depth == 0
        assert "k" in t._root_keys

    def test_unresolvable_parent_stores_none(self):
        """Dangling parents must not be stored either — same spin risk."""
        t = LineageTree()
        t.insert("ghost", "c", [], [], 1)
        assert t.nodes["c"].parent_key is None


class TestBoundedWalks:
    def test_lca_with_legacy_self_parent_terminates(self):
        """A tree carrying a pre-fix self-parent (e.g. loaded from old
        state) must answer lca() instead of spinning."""
        t = LineageTree()
        t.nodes["k"] = _node("k", "k", 0)  # corrupt: self-parent
        t.nodes["x"] = _node("x", None, 0)
        assert t.lca("k", "x") is None

    def test_ancestors_bounded_on_injected_cycle(self):
        t = LineageTree()
        t.nodes["c"] = _node("c", "d", 5)
        t.nodes["d"] = _node("d", "c", 5)
        seen = {n.key for n in t.ancestors("c")}
        assert seen <= {"c", "d"}

    def test_lca_bounded_on_injected_cycle(self):
        t = LineageTree()
        t.nodes["c"] = _node("c", "d", 5)
        t.nodes["d"] = _node("d", "c", 5)
        t.nodes["e"] = _node("e", None, 0)
        assert t.lca("c", "e") is None


class TestNormalTreesUnchanged:
    def test_lca_still_finds_common_ancestor(self):
        t = LineageTree()
        t.insert(None, "root", [], [], 1)
        t.insert("root", "l", [], [], 1)
        t.insert("root", "r", [], [], 1)
        assert t.lca("l", "r") == "root"
        assert t.lca_distance("l", "r") == 2

    def test_disconnected_trees_return_none(self):
        t = LineageTree()
        t.insert(None, "a", [], [], 1)
        t.insert(None, "b", [], [], 1)
        assert t.lca("a", "b") is None
