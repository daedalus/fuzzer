"""Regression tests: lineage backtracking (--lineage-backtrack).

Backtracking widens exploration by detecting an *exhausted* lineage branch
— a seed whose subtree has gained no coverage since the last credit reset,
and which has been fuzzed enough times that the absence is evidence rather
than noise — and penalising it geometrically by depth. Selection weight
shifts back toward shallow seeds with unexplored siblings.

The properties that must hold, and that a silent regression would break:

  * off by default, and inert without --lineage (it needs the tree to know
    what a branch is);
  * productive branches are never penalised, however deep — otherwise the
    feature would punish exactly the lineage worth pursuing;
  * the penalty grows with depth, which is what makes it a *backtrack*
    rather than a flat discount;
  * a young seed is not judged exhausted merely for having no edges yet.

Each of these fails silently if broken: the run still completes, at a
plausible speed, just with worse seed choices.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.services.seed_picker import SeedPicker


class _Fuzzer:
    """Minimal stand-in exposing only what the weight function reads."""

    def __init__(self, *, backtrack=True, tree=None, meta=None, decay=0.7, min_fuzz=8):
        self._use_lineage_backtrack = backtrack
        self._lineage = tree
        self.seed_meta = meta or {}
        self._lineage_backtrack_decay = decay
        self._lineage_backtrack_min_fuzz = min_fuzz


def _weight(f, key, w, fuzz_count, key_to_seed):
    return SeedPicker._weight_lineage_backtrack(
        object.__new__(SeedPicker), key, w, fuzz_count, f, key_to_seed
    )


def _tree_with_chain(depth: int, edges_per_node: int = 0):
    """Linear chain root -> c1 -> ... of the requested depth.

    edges_per_node feeds coverage meta so recent_credit can be non-zero.
    """
    tree = LineageTree()
    meta = {}
    key_to_seed = {}
    parent = None
    for i in range(depth + 1):
        key = f"k{i}"
        seed = f"seed{i}".encode()
        tree.insert(parent, key, ["havoc"], [0], edges_per_node)
        meta[seed] = {
            "coverage_edges": edges_per_node,
            "coverage_edges_baseline": 0,
        }
        key_to_seed[key] = seed
        parent = key
    return tree, meta, key_to_seed


class TestGating:
    def test_noop_when_flag_disabled(self):
        tree, meta, k2s = _tree_with_chain(3)
        f = _Fuzzer(backtrack=False, tree=tree, meta=meta)
        assert _weight(f, "k3", 10.0, 100, k2s) == 10.0

    def test_noop_without_a_lineage_tree(self):
        f = _Fuzzer(backtrack=True, tree=None, meta={})
        assert _weight(f, "k3", 10.0, 100, {}) == 10.0

    def test_noop_for_unknown_key(self):
        tree, meta, k2s = _tree_with_chain(3)
        f = _Fuzzer(tree=tree, meta=meta)
        assert _weight(f, "does-not-exist", 10.0, 100, k2s) == 10.0

    def test_root_is_never_penalised(self):
        """Depth 0 has nowhere to back off to."""
        tree, meta, k2s = _tree_with_chain(3)
        f = _Fuzzer(tree=tree, meta=meta)
        assert _weight(f, "k0", 10.0, 100, k2s) == 10.0


class TestExhaustionDetection:
    def test_young_seed_is_not_judged_exhausted(self):
        """Below min_fuzz, "no new edges" is not yet evidence."""
        tree, meta, k2s = _tree_with_chain(3)
        f = _Fuzzer(tree=tree, meta=meta, min_fuzz=8)
        assert _weight(f, "k3", 10.0, 1, k2s) == 10.0

    def test_exhausted_branch_is_penalised(self):
        tree, meta, k2s = _tree_with_chain(3, edges_per_node=0)
        f = _Fuzzer(tree=tree, meta=meta, min_fuzz=8)
        assert _weight(f, "k3", 10.0, 100, k2s) < 10.0

    def test_productive_branch_is_not_penalised(self):
        """A deep branch still gaining coverage keeps its full weight —
        penalising it would punish the lineage most worth pursuing."""
        tree, meta, k2s = _tree_with_chain(3, edges_per_node=25)
        f = _Fuzzer(tree=tree, meta=meta, min_fuzz=8)
        assert _weight(f, "k3", 10.0, 100, k2s) == 10.0


class TestDepthBehaviour:
    def test_penalty_increases_with_depth(self):
        """This monotonicity is what makes it a backtrack: descending a dead
        branch must get progressively less attractive."""
        tree, meta, k2s = _tree_with_chain(5, edges_per_node=0)
        f = _Fuzzer(tree=tree, meta=meta, min_fuzz=8)
        weights = [_weight(f, f"k{d}", 10.0, 100, k2s) for d in range(1, 6)]
        assert weights == sorted(weights, reverse=True), weights
        assert weights[-1] < weights[0]

    def test_penalty_matches_configured_decay(self):
        tree, meta, k2s = _tree_with_chain(3, edges_per_node=0)
        f = _Fuzzer(tree=tree, meta=meta, decay=0.5, min_fuzz=8)
        node_depth = tree.nodes["k2"].depth
        assert _weight(f, "k2", 1.0, 100, k2s) == pytest.approx(0.5**node_depth)

    def test_penalty_has_a_floor(self):
        """Very deep dead branches are de-prioritised, not erased — the
        seed must stay reachable."""
        tree, meta, k2s = _tree_with_chain(40, edges_per_node=0)
        f = _Fuzzer(tree=tree, meta=meta, decay=0.5, min_fuzz=8)
        w = _weight(f, "k40", 1.0, 100, k2s)
        assert w > 0.0
        assert w >= 0.05 * 1.0 * 0.999


class TestWiring:
    def test_backtrack_requires_lineage(self):
        """--lineage-backtrack without --lineage must stay inert rather
        than half-enable and read a tree that was never built."""
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        src = inspect.getsource(Fuzzer.__init__)
        assert "lineage_backtrack and lineage" in src

    def test_cli_flag_exists_and_reaches_the_fuzzer(self):
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert '"--lineage-backtrack"' in src
        assert "lineage_backtrack=getattr" in src

    def test_weight_function_is_called_in_the_weight_pass(self):
        import inspect

        src = inspect.getsource(SeedPicker._compute_weights)
        assert "_weight_lineage_backtrack" in src
