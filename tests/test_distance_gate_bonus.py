"""Tests for the opt-in control-dependence discount in core/distance.py.

Drives ``TargetDistance._compute_bb_values`` directly against a
hand-built FunctionCFG (no ELF, no disassembly -- the discount is a pure
post-processing step over whatever harmonic-BFS values
``_compute_bb_values`` already produced). This mirrors the white-box
style of ``tests/test_distance_unit.py``.

CFG shape -- a loop with the target inside it::

    0(entry) -> 1(header) -> 2(target, loop body) -> 1  (back edge)
                          \\-> 3(exit)

``_compute_bb_values`` BFS-explores the *reversed* CFG (predecessor
edges, via ``core/dominators.py::predecessors``) starting at each target
block, so the value assigned to block b is the shortest *forward* path
length from b to the target -- exactly the AFLGo distance the module
docstring promises. (An earlier version walked ``successors`` forward
from the target instead, which computes the unrelated quantity "distance
from target to b"; see git history / the dominator-gate handover doc for
the bug this replaced.) This shape exercises both branches of the
discount:

  - block 1 (loop header) IS a dominator of the target AND has a
    baseline value (one predecessor hop via the back edge, 2->1) ->
    gets discounted.
  - block 0 (entry) IS a dominator of the target, reached via two
    predecessor hops (2->1->0) -> also gets discounted. Because every
    dominator of a reachable target lies on *some* forward path to it
    by definition, the reversed-CFG BFS is guaranteed to discover it --
    unlike the old forward-from-target walk, which could (and did, for
    block 0 here) miss real dominators entirely.
  - block 3 (loop exit) is forward-reachable *from* block 1 but cannot
    reach the target at all (it has no successors) -> correctly gets no
    baseline value, and is not a dominator either (skippable: you can
    reach 3 without ever visiting 2).
"""

import pytest

from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG
from fuzzer_tool.core.distance import TargetDistance

FUNC_NAME = "target_fn"


def _make_cfg() -> FunctionCFG:
    edges = {0: [1], 1: [2, 3], 2: [1], 3: []}
    blocks = {
        a: BasicBlock(start=a, end=a + 1, successors=list(succs))
        for a, succs in edges.items()
    }
    blocks[0].is_entry = True
    blocks[3].is_exit = True
    return FunctionCFG(name=FUNC_NAME, start=0, end=4, blocks=blocks)


def _make_td(gate_bonus: float) -> TargetDistance:
    td = TargetDistance("/nonexistent", targets=[FUNC_NAME], gate_bonus=gate_bonus)
    td.functions = {FUNC_NAME: (0, 4)}
    td._func_addrs_sorted = [(0, FUNC_NAME)]
    td.target_addrs = {2}
    td._cfgs = {FUNC_NAME: _make_cfg()}
    td._compute_bb_values()
    return td


class TestGateBonusDisabledByDefault:
    def test_default_is_zero(self):
        td = TargetDistance("/nonexistent", targets=[FUNC_NAME])
        assert td._gate_bonus == 0.0

    def test_zero_gate_bonus_leaves_bfs_values_untouched(self):
        td = _make_td(gate_bonus=0.0)
        # Hand-derived harmonic-BFS values (reversed CFG, backward from
        # target=2): 2(target)=0.0; 1 is one predecessor hop away
        # (2->1) -> 1+d=2.0; 0 is two predecessor hops away (2->1->0)
        # -> 1+d=3.0; 3 has no path to the target at all (dead end past
        # the loop exit) so it gets no baseline value.
        assert td._bb_value[2] == 0.0
        assert td._bb_value[1] == 2.0
        assert td._bb_value[0] == 3.0
        assert 3 not in td._bb_value
        assert td._bb_gate == set()
        assert not td.is_gate(0)
        assert not td.is_gate(1)


class TestGateBonusDiscountsOnlyTrueGates:
    def test_gates_are_the_full_dominator_chain(self):
        td = _make_td(gate_bonus=0.5)
        # dominator_chain(2) = [2, 1, 0]; gates exclude the target itself.
        assert td._bb_gate == {0, 1}
        assert 3 not in td._bb_gate  # skippable, not a dominator

    def test_reachable_gate_is_discounted(self):
        td = _make_td(gate_bonus=0.5)
        assert td._bb_value[1] == 2.0 * 0.5

    def test_every_dominator_gets_a_real_bfs_value(self):
        # Block 0 is a dominator two predecessor-hops from the target.
        # With the reversed-CFG BFS this is always discovered (a
        # dominator, by definition, sits on some forward path to a
        # reachable target, so it is backward-reachable from it too) --
        # there is no "structural gate with no baseline value" case left
        # for a genuine dominator to fall into.
        td = _make_td(gate_bonus=0.5)
        assert 0 in td._bb_value
        assert td._bb_value[0] == 3.0 * 0.5
        assert td.is_gate(0)

    def test_non_gate_block_unchanged(self):
        td = _make_td(gate_bonus=0.5)
        assert 3 not in td._bb_value
        assert not td.is_gate(3)

    def test_target_block_stays_zero(self):
        td = _make_td(gate_bonus=0.5)
        assert td._bb_value[2] == 0.0
        assert not td.is_gate(2)  # a target doesn't list itself as its own gate


class TestGateBonusValidation:
    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            TargetDistance("/nonexistent", targets=[FUNC_NAME], gate_bonus=1.5)
        with pytest.raises(ValueError):
            TargetDistance("/nonexistent", targets=[FUNC_NAME], gate_bonus=-0.1)
