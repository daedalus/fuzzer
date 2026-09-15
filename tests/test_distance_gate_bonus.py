"""Tests for the opt-in control-dependence discount in core/distance.py.

Drives ``TargetDistance._compute_bb_values`` directly against a
hand-built FunctionCFG (no ELF, no disassembly -- the discount is a pure
post-processing step over whatever harmonic-BFS values
``_compute_bb_values`` already produced). This mirrors the white-box
style of ``tests/test_distance_unit.py``.

CFG shape -- a loop with the target inside it::

    0(entry) -> 1(header) -> 2(target, loop body) -> 1  (back edge)
                          \\-> 3(exit)

``_compute_bb_values``'s existing harmonic-BFS walks *forward* from the
target over ``successors`` (see the open question recorded in
``docs/handover/handover_dominator_gate_2026-09-15.md`` about whether
this matches the module's own "reversed CFG" docstring claim), so the
set of blocks it assigns a baseline value to is whatever the target can
reach going forward -- which for a loop includes the loop header via
the back edge but does NOT include the header's other predecessor
(block 0, outside the loop, never forward-reachable from inside it).
This shape was chosen deliberately to exercise both branches of the
discount:

  - block 1 (loop header) IS a dominator of the target AND already has
    a baseline value (reached via the back edge) -> gets discounted.
  - block 0 (entry) IS a dominator of the target but has NO baseline
    value (never forward-reachable from inside the loop) -> structurally
    still a gate (``is_gate`` is True) but nothing to discount, and the
    code must not invent a value for it.
  - block 3 (loop exit) has a baseline value but is NOT a dominator
    (skippable -- you can reach 3 without ever hitting 2) -> must stay
    untouched by the discount.
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
        # Hand-derived harmonic-BFS values (forward from target=2):
        # 2(target)=0.0; 1 reached via the back edge at d=1 -> 1/(1/2)=2.0;
        # 3 reached from 1 at d=2 -> 1/(1/3)=3.0; 0 never forward-reachable
        # from 2, so absent entirely.
        assert td._bb_value[2] == 0.0
        assert td._bb_value[1] == 2.0
        assert td._bb_value[3] == 3.0
        assert 0 not in td._bb_value
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

    def test_unreachable_gate_has_no_invented_value(self):
        td = _make_td(gate_bonus=0.5)
        # Block 0 is structurally a gate (is_gate True) even though the
        # existing BFS never assigned it a baseline distance -- the
        # discount must skip it, not fabricate one.
        assert 0 not in td._bb_value
        assert td.is_gate(0)

    def test_non_gate_block_unchanged(self):
        td = _make_td(gate_bonus=0.5)
        assert td._bb_value[3] == 3.0
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
