"""Unit tests for core/dominators.py.

Graphs are built directly as FunctionCFG/BasicBlock objects (no
disassembly involved — dominance is a pure graph-structure question).
Every idom/gate relationship below is hand-derived from the diagram in
each test's docstring, independent of the algorithm under test, per the
repo rule that equivalence assertions must not validate code against
itself.
"""

from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG
from fuzzer_tool.core.dominators import (
    compute_idom,
    dominates,
    dominator_chain,
    gate_blocks,
)


def _cfg(edges: dict[int, list[int]], entry: int) -> FunctionCFG:
    """Build a FunctionCFG from a plain successor map.

    Every address mentioned anywhere (as a key or in a successor list)
    gets a block; ``entry`` is flagged ``is_entry``.
    """
    addrs = set(edges) | {a for succs in edges.values() for a in succs}
    blocks = {
        a: BasicBlock(start=a, end=a + 1, successors=list(edges.get(a, [])))
        for a in addrs
    }
    blocks[entry].is_entry = True
    return FunctionCFG(name="f", start=min(addrs), end=max(addrs) + 1, blocks=blocks)


class TestChain:
    """0 -> 1 -> 2 -> 3 (straight line, no branches).

    Every block's sole dominator chain is the prefix of the chain up to
    entry: idom(1)=0, idom(2)=1, idom(3)=2.
    """

    CFG = _cfg({0: [1], 1: [2], 2: [3], 3: []}, entry=0)

    def test_idom(self):
        idom = compute_idom(self.CFG)
        assert idom == {0: 0, 1: 0, 2: 1, 3: 2}

    def test_dominates_transitively(self):
        idom = compute_idom(self.CFG)
        assert dominates(idom, 0, 3)
        assert dominates(idom, 1, 3)
        assert not dominates(idom, 2, 1)  # wrong direction

    def test_dominator_chain(self):
        idom = compute_idom(self.CFG)
        assert dominator_chain(idom, 3) == [3, 2, 1, 0]


class TestDiamond:
    """0 -> {1, 2} -> 3 (if/else merging back).

    Neither 1 nor 2 dominates 3 (each is skippable via the other arm);
    0 is the immediate dominator of everything downstream of the merge.
    """

    CFG = _cfg({0: [1, 2], 1: [3], 2: [3], 3: []}, entry=0)

    def test_idom(self):
        idom = compute_idom(self.CFG)
        assert idom == {0: 0, 1: 0, 2: 0, 3: 0}

    def test_neither_arm_dominates_merge(self):
        idom = compute_idom(self.CFG)
        assert not dominates(idom, 1, 3)
        assert not dominates(idom, 2, 3)

    def test_gate_blocks_is_just_entry(self):
        # 3 is the only target; its sole proper dominator is 0.
        assert gate_blocks(self.CFG, targets={3}) == {0}


class TestNestedDiamondGate:
    """0 -> 1 -> {2, 3} -> 4 -> 5 (a diamond gated behind a mandatory block).

    1 and 4 dominate the target (5); 2 and 3 don't (each is skippable via
    the other arm of the inner diamond). This is the shape the gate
    discount exists for: 2 and 3 are BFS-equidistant from 5 but only 1
    and 4 are mandatory.
    """

    CFG = _cfg({0: [1], 1: [2, 3], 2: [4], 3: [4], 4: [5], 5: []}, entry=0)

    def test_gate_blocks_excludes_inner_diamond_arms(self):
        gates = gate_blocks(self.CFG, targets={5})
        assert gates == {0, 1, 4}
        assert 2 not in gates
        assert 3 not in gates


class TestLoop:
    """0 -> 1 -> 2 -> 1 (back edge), 2 -> 3 (loop exit).

    The back edge must not confuse the fixed point: 1 still dominates
    both 2 and 3 (the only way in is through 1), and 1's own idom stays
    0 despite 2 also being one of 1's predecessors.
    """

    CFG = _cfg({0: [1], 1: [2], 2: [1, 3], 3: []}, entry=0)

    def test_idom(self):
        idom = compute_idom(self.CFG)
        assert idom == {0: 0, 1: 0, 2: 1, 3: 2}

    def test_no_spurious_cycle(self):
        idom = compute_idom(self.CFG)
        assert dominates(idom, 1, 3)
        assert not dominates(idom, 2, 0)


class TestUnreachable:
    """0 -> 1 (entry side); 9 -> 1 is a dangling predecessor with no path
    from entry to 9 — 9 must not appear in idom, and must not corrupt 1's
    dominator (still 0, not confused by the unreachable extra predecessor).
    """

    CFG = _cfg({0: [1], 9: [1], 1: []}, entry=0)

    def test_unreachable_block_absent(self):
        idom = compute_idom(self.CFG)
        assert 9 not in idom
        assert idom[1] == 0

    def test_dominates_false_for_unreachable(self):
        idom = compute_idom(self.CFG)
        assert not dominates(idom, 0, 9)
        assert dominator_chain(idom, 9) == []


class TestGateBlocksMultiTarget:
    """Diamond with a target on each arm: 0 -> {1, 2}, both terminal.

    Neither arm gates the other's target — only the shared entry does.
    """

    CFG = _cfg({0: [1, 2], 1: [], 2: []}, entry=0)

    def test_union_of_both_chains(self):
        gates = gate_blocks(self.CFG, targets={1, 2})
        assert gates == {0}

    def test_targets_excluded_from_their_own_gate_set(self):
        gates = gate_blocks(self.CFG, targets={1, 2})
        assert 1 not in gates
        assert 2 not in gates


class TestDefaultEntryFallback:
    """No block flagged is_entry: compute_idom falls back to min(blocks)."""

    CFG = _cfg({0: [1], 1: []}, entry=0)

    def test_falls_back_to_lowest_start(self):
        self.CFG.blocks[0].is_entry = False
        idom = compute_idom(self.CFG)
        assert idom == {0: 0, 1: 0}
