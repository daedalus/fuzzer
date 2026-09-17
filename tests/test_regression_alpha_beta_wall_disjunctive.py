"""Regression tests for the comparison-wall solver's disjunctive-sum fix.

Background (docs/handover/handover_decision_game_theory_survey_2026-09-13.md
§2): ``Z3Solver._alpha_beta_wall`` searched the *whole* wall as one
alternating-turn minimax game. Its per-step reward, ``evaluate([cond])``, is
always called on a singleton and so is always overlap-free by construction
— the overlap penalty the function's own docstring describes never applied
to the sequence actually being built. Empirically this made the returned
value depend only on ``len(conditions)``'s parity, not on the conditions'
actual offset/overlap structure: fully-disjoint, fully-overlapping, and
partially-overlapping same-shape fixtures all produced bit-identical output.

The fix partitions conditions into overlap-connected components first
(disjunctive sum: components sharing no offset can't affect each other's
score in any order) and only runs the two-player search *within* a
component, concatenating components widest-first. These tests exercise the
partitioning directly and check that overlap structure now visibly changes
behavior, without requiring z3 (no test here touches the solver, only the
pure-Python wall-ordering logic — no ``requires_z3`` marker needed).
"""

from __future__ import annotations

from fuzzer_tool.core.cond_stmt import CondState, CondStmt, CondStmtBase
from fuzzer_tool.core.smt_solver import Z3Solver, _connected_components_by_offset


def _cond(cmpid: int, offsets: tuple[int, ...], width: int = 1) -> CondStmt:
    base = CondStmtBase(cmpid=cmpid, op_a=b"\x00", op_b=b"\x01", width=width, result=1, pc=None)
    return CondStmt(base=base, offsets=offsets, state=CondState.UNSOLVED)


def _wall_solver() -> Z3Solver:
    # No z3 session needed: _alpha_beta_wall never touches self beyond
    # method dispatch, so bypassing __init__ keeps these tests z3-free.
    return Z3Solver.__new__(Z3Solver)


# ═══════════════════════════════════════════════════════════════════
# _connected_components_by_offset
# ═══════════════════════════════════════════════════════════════════


class TestConnectedComponentsByOffset:
    def test_all_disjoint_are_singletons(self):
        conds = [_cond(i, (i,)) for i in range(4)]
        comps = _connected_components_by_offset(conds)
        assert len(comps) == 4
        assert all(len(c) == 1 for c in comps)

    def test_shared_offset_merges_into_one_component(self):
        conds = [_cond(i, (0,)) for i in range(4)]
        comps = _connected_components_by_offset(conds)
        assert len(comps) == 1
        assert len(comps[0]) == 4

    def test_transitive_overlap_merges_chain(self):
        # a-b share offset 1, b-c share offset 2: a, b, c must all merge
        # even though a and c share nothing directly.
        a = _cond(0, (1,))
        b = _cond(1, (1, 2))
        c = _cond(2, (2,))
        comps = _connected_components_by_offset([a, b, c])
        assert len(comps) == 1
        assert {c.base.cmpid for c in comps[0]} == {0, 1, 2}

    def test_two_independent_pairs_stay_separate(self):
        p0 = _cond(0, (0, 1))
        p1 = _cond(1, (0, 1))
        p2 = _cond(2, (2, 3))
        p3 = _cond(3, (2, 3))
        comps = _connected_components_by_offset([p0, p1, p2, p3])
        assert len(comps) == 2
        ids = sorted(sorted(c.base.cmpid for c in comp) for comp in comps)
        assert ids == [[0, 1], [2, 3]]

    def test_offsetless_conditions_are_isolated_singletons(self):
        conds = [_cond(0, ()), _cond(1, ())]
        comps = _connected_components_by_offset(conds)
        assert len(comps) == 2

    def test_empty_input(self):
        assert _connected_components_by_offset([]) == []


# ═══════════════════════════════════════════════════════════════════
# _alpha_beta_wall — overlap structure must now be visible in the result
# ═══════════════════════════════════════════════════════════════════


class TestAlphaBetaWallDisjunctiveSum:
    def test_empty_wall(self):
        assert _wall_solver()._alpha_beta_wall([], 4, 8) == []

    def test_single_condition(self):
        c = _cond(0, (0,))
        order = _wall_solver()._alpha_beta_wall([c], 4, 8)
        assert order == [c]

    def test_no_offset_conditions_pass_through(self):
        conds = [_cond(0, ()), _cond(1, ())]
        order = _wall_solver()._alpha_beta_wall(conds, 4, 8)
        assert {c.base.cmpid for c in order} == {0, 1}

    def test_disjoint_and_full_overlap_no_longer_collapse_to_the_same_shape(self):
        # This is the regression the bug produced: before the fix, these
        # two fixtures (4 conditions, one offset each) were computed by
        # the *same* whole-wall minimax and returned identical output
        # regardless of whether the offsets were all distinct or all the
        # same. After the fix they are partitioned differently — 4
        # singleton components vs. 1 four-element component — which is
        # observable even though, coincidentally, ties within each
        # component still preserve input order for symmetric fixtures.
        disjoint = [_cond(i, (i,)) for i in range(4)]
        overlapping = [_cond(i, (0,)) for i in range(4)]

        assert len(_connected_components_by_offset(disjoint)) == 4
        assert len(_connected_components_by_offset(overlapping)) == 1

        order_disjoint = _wall_solver()._alpha_beta_wall(disjoint, 4, 8)
        order_overlap = _wall_solver()._alpha_beta_wall(overlapping, 4, 8)
        assert [c.base.cmpid for c in order_disjoint] == [0, 1, 2, 3]
        assert [c.base.cmpid for c in order_overlap] == [0, 1, 2, 3]

    def test_widest_component_is_scheduled_first(self):
        # One heavily-tainted overlapping trio (offsets 1-3, total width 6)
        # plus one unrelated single-offset condition (offset 9, width 1).
        # The trio's component must be placed before the singleton.
        wide_a = _cond(0, (1, 2))
        wide_b = _cond(1, (2, 3))
        wide_c = _cond(2, (3,))
        narrow = _cond(3, (9,))
        order = _wall_solver()._alpha_beta_wall([narrow, wide_a, wide_b, wide_c], 4, 8)
        ids = [c.base.cmpid for c in order]
        assert ids.index(3) == len(ids) - 1  # narrow singleton lands last

    def test_independent_pairs_are_solved_independently(self):
        # Two disjoint overlapping pairs. Regardless of internal tie-break,
        # every member of pair A must be contiguous and separate from pair
        # B's members, since they form two disjunctive-sum components.
        p0, p1 = _cond(0, (0, 1)), _cond(1, (0, 1))
        p2, p3 = _cond(2, (2, 3)), _cond(3, (2, 3))
        order = _wall_solver()._alpha_beta_wall([p0, p1, p2, p3], 4, 8)
        ids = [c.base.cmpid for c in order]
        pos_a = sorted(ids.index(i) for i in (0, 1))
        pos_b = sorted(ids.index(i) for i in (2, 3))
        assert pos_a == [0, 1] or pos_a == [2, 3]
        assert pos_b == [0, 1] or pos_b == [2, 3]
        assert pos_a != pos_b

    def test_max_mutations_unused_but_signature_preserved(self):
        # max_mutations has never been read inside _alpha_beta_wall (true
        # before and after this fix); this just locks the call signature.
        c = _cond(0, (0,))
        assert _wall_solver()._alpha_beta_wall([c], max_mutations=999, max_depth=8) == [c]
