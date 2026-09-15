"""Regression test: acyclic-region ancestors must get a real CFG distance.

``_compute_bb_values`` used to BFS forward from the target block over
``successors`` (see the dominator-gate handover doc's "open question"),
which only assigns a baseline value to whatever the target can reach
going forward. In a plain acyclic diamond, a target's own ancestors --
the blocks that lead to it, which directed fuzzing most wants to
prioritize -- are never forward-reachable *from* the target, so they
silently got no CFG value at all and fell back to the coarser
per-function call-graph distance. Fixed by walking the reversed CFG
(predecessor edges) from the target instead, matching the module's own
"BFS on the reversed intra-procedural CFG" docstring claim.

This deliberately uses a loop-free diamond (unlike
``test_distance_gate_bonus.py``'s CFG, which has a back edge) because a
back edge could make an ancestor forward-reachable from the target too
and mask this exact bug -- which is exactly what happened when this
fix was first drafted.
"""

from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG
from fuzzer_tool.core.distance import TargetDistance

FUNC_NAME = "diamond_fn"


def _make_diamond_cfg() -> FunctionCFG:
    # 0(entry) -> 1, 0 -> 2; 1 -> 3(target); 2 -> 3(target). No back edges.
    edges = {0: [1, 2], 1: [3], 2: [3], 3: []}
    blocks = {
        a: BasicBlock(start=a, end=a + 1, successors=list(succs))
        for a, succs in edges.items()
    }
    blocks[0].is_entry = True
    blocks[3].is_exit = True
    return FunctionCFG(name=FUNC_NAME, start=0, end=4, blocks=blocks)


def _make_td() -> TargetDistance:
    td = TargetDistance("/nonexistent", targets=[FUNC_NAME])
    td.functions = {FUNC_NAME: (0, 4)}
    td._func_addrs_sorted = [(0, FUNC_NAME)]
    td.target_addrs = {3}
    td._cfgs = {FUNC_NAME: _make_diamond_cfg()}
    td._compute_bb_values()
    return td


class TestAncestorsGetRealDistanceInAcyclicRegion:
    def test_target_is_zero(self):
        td = _make_td()
        assert td._bb_value[3] == 0.0

    def test_direct_predecessors_are_one_hop(self):
        td = _make_td()
        # Both 1 and 2 are one predecessor-hop from the target -> 1+d=2.0.
        assert td._bb_value[1] == 2.0
        assert td._bb_value[2] == 2.0

    def test_entry_ancestor_is_two_hops(self):
        # This is the case the old forward-from-target BFS missed
        # entirely: 0 has no back edge making it forward-reachable
        # *from* 3, but it plainly can reach 3 (via either arm of the
        # diamond), so it must carry a real distance value.
        td = _make_td()
        assert 0 in td._bb_value
        assert td._bb_value[0] == 3.0
