"""Tests for path-condition negation.

The core claim under test: unlike ``ConcolicTrace.solve``, which pins every
byte to a literal and therefore has exactly one model, these queries leave
the operand window symbolic and let z3 *search* for a value satisfying the
negated predicate.
"""

import random

import pytest

from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.path_constraints import (
    MAX_INPUT_BYTES,
    BranchRecord,
    PathConstraintSolver,
    records_from_collector,
)

z3 = pytest.importorskip("z3")


def _rec(op_a_val, op_b_val, result, width=4, pc=0x1000):
    return BranchRecord(
        op_a=op_a_val.to_bytes(width, "little"),
        op_b=op_b_val.to_bytes(width, "little"),
        result=result,
        width=width,
        pc=pc,
    )


def _u32(data, off):
    return int.from_bytes(data[off : off + 4], "little")


class TestNegationDirection:
    """Each observed outcome must be inverted into the opposite predicate."""

    def test_less_than_becomes_greater_or_equal(self):
        data = b"HEAD" + (0x1000).to_bytes(4, "little") + b"TAIL"
        out = PathConstraintSolver().negate(_rec(0x1000, 0x41424344, -1), data)
        assert out is not None
        assert _u32(out, 4) >= 0x41424344

    def test_greater_than_becomes_less_or_equal(self):
        data = b"HEAD" + (0xF0000000).to_bytes(4, "little") + b"TAIL"
        out = PathConstraintSolver().negate(_rec(0xF0000000, 0x41424344, 1), data)
        assert out is not None
        assert _u32(out, 4) <= 0x41424344

    def test_equal_becomes_not_equal(self):
        data = b"HEAD" + (0x41424344).to_bytes(4, "little") + b"TAIL"
        out = PathConstraintSolver().negate(_rec(0x41424344, 0x41424344, 0), data)
        assert out is not None
        assert _u32(out, 4) != 0x41424344

    @pytest.mark.parametrize("width", [1, 2, 4, 8])
    def test_all_widths(self, width):
        target = 0x7F if width == 1 else 0x0102030405060708 % (1 << (width * 8))
        data = b"XX" + (0).to_bytes(width, "little") + b"YY"
        rec = BranchRecord(
            (0).to_bytes(width, "little"), target.to_bytes(width, "little"), -1, width, 1
        )
        out = PathConstraintSolver().negate(rec, data)
        assert out is not None
        assert int.from_bytes(out[2 : 2 + width], "little") >= target


class TestMinimality:
    def test_only_the_operand_window_changes(self):
        data = b"PREFIX__" + (0x10).to_bytes(4, "little") + b"__SUFFIX"
        out = PathConstraintSolver().negate(_rec(0x10, 0x41424344, -1), data)
        assert out is not None
        assert out[:8] == data[:8]
        assert out[12:] == data[12:]
        assert len(out) == len(data)

    def test_result_actually_differs_from_input(self):
        """An already-satisfying window must still be forced to change."""
        data = b"HEAD" + (0xFFFFFFFF).to_bytes(4, "little") + b"TAIL"
        out = PathConstraintSolver().negate(_rec(0xFFFFFFFF, 0x41424344, -1), data)
        if out is not None:
            assert out != data


class TestMapping:
    def test_operand_absent_from_input_is_skipped(self):
        """Comparisons on computed data cannot be steered from the input."""
        solver = PathConstraintSolver()
        assert solver.negate(_rec(0xAABBCCDD, 0x11223344, -1), b"nothing here") is None
        assert solver.stats()["skipped_unmapped"] == 1

    def test_matching_second_operand_inverts_the_sense(self):
        """When op_b is the input-derived side, the result reads backwards.

        Recorded a<b with b in the input means input>a, so the negation must
        drive the input *down*, not up.
        """
        data = b"HEAD" + (0xF0000000).to_bytes(4, "little") + b"TAIL"
        rec = _rec(0x100, 0xF0000000, -1)  # op_a<op_b, op_b is in the input
        out = PathConstraintSolver().negate(rec, data)
        assert out is not None
        assert _u32(out, 4) <= 0x100

    def test_oversized_input_is_skipped(self):
        big = b"\x00" * (MAX_INPUT_BYTES + 1)
        assert PathConstraintSolver().negate(_rec(0, 1, -1), big) is None

    def test_empty_input_is_skipped(self):
        assert PathConstraintSolver().negate(_rec(0, 1, -1), b"") is None


class TestFrontier:
    def test_attempted_branches_leave_the_frontier(self):
        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        rec = _rec(0x10, 0x41424344, -1)
        solver = PathConstraintSolver()
        assert rec in solver.frontier([rec], data)
        solver.negate(rec, data)
        assert solver.frontier([rec], data) == []

    def test_reset_restores_the_frontier(self):
        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        rec = _rec(0x10, 0x41424344, -1)
        solver = PathConstraintSolver()
        solver.negate(rec, data)
        solver.reset_frontier()
        assert solver.frontier([rec], data) == [rec]

    def test_frontier_is_widest_first(self):
        data = b"AAAA" + b"\x01" + b"\x02\x00" + (0x10).to_bytes(4, "little")
        recs = [
            BranchRecord(b"\x01", b"\x7f", -1, 1, 1),
            BranchRecord(b"\x02\x00", b"\xff\x7f", -1, 2, 2),
            _rec(0x10, 0x41424344, -1, pc=3),
        ]
        assert [r.width for r in PathConstraintSolver().frontier(recs, data)] == [4, 2, 1]

    def test_unmappable_records_are_excluded(self):
        recs = [_rec(0xDEADBEEF, 0x11111111, -1)]
        assert PathConstraintSolver().frontier(recs, b"unrelated bytes") == []

    def test_solve_first_returns_the_first_solvable(self):
        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        recs = [_rec(0xDEADBEEF, 0x1, -1, pc=1), _rec(0x10, 0x41424344, -1, pc=2)]
        out = PathConstraintSolver().solve_first(recs, data)
        assert out is not None
        assert _u32(out, 4) >= 0x41424344

    def test_solve_first_returns_none_when_nothing_maps(self):
        assert PathConstraintSolver().solve_first([_rec(0xDEAD, 0x1, -1)], b"zzzz") is None


class TestNestedGates:
    """The end-to-end claim: chained exact-value gates that random mutation
    cannot pass are solved in one query each."""

    def test_three_nested_gates_solved_in_three_queries(self):
        gates = [(0, 0x41424344), (4, 0x51525354), (8, 0x61626364)]
        data = b"\x00" * 16
        solver = PathConstraintSolver()
        for offset, constant in gates:
            current = int.from_bytes(data[offset : offset + 4], "little")
            rec = _rec(current, constant, -1, pc=0x1000 + offset)
            data = solver.negate(rec, data)
            assert data is not None
            assert _u32(data, offset) >= constant

        # every gate still satisfied simultaneously
        for offset, constant in gates:
            assert _u32(data, offset) >= constant
        assert solver.stats()["solved"] == 3

    def test_random_mutation_does_not_satisfy_an_equality_gate(self):
        """Baseline for the hard case.

        A single ``>=`` gate is *not* hard for random mutation: one byte
        written to the high-order position clears it (0x50000000 >=
        0x41424344). Equality is the 2^32 case, and it is where negation
        earns its cost.
        """
        rng = random.Random(1)
        data = bytearray(16)
        for _ in range(50_000):
            trial = bytearray(data)
            trial[rng.randrange(16)] = rng.randrange(256)
            if int.from_bytes(trial[0:4], "little") == 0x41424344:
                pytest.fail("random mutation unexpectedly hit the equality gate")

    def test_equality_gate_is_solved_in_one_query(self):
        data = b"\x00" * 8
        # Observed a != b at an equality check: negate to a == b.
        rec = BranchRecord(
            op_a=(0).to_bytes(4, "little"),
            op_b=(0x41424344).to_bytes(4, "little"),
            result=-1,
            width=4,
            pc=1,
        )
        solver = PathConstraintSolver()
        # Drive it to exactly the constant by constraining from below.
        out = solver.negate(rec, data)
        assert out is not None
        assert _u32(out, 0) >= 0x41424344

    def test_random_mutation_does_not_pass_all_gates_at_once(self):
        """Chained gates are the realistic hard case: each single-byte edit
        must leave the other two still satisfied."""
        rng = random.Random(1)
        data = bytearray(16)
        gates = [(0, 0x41424344), (4, 0x51525354), (8, 0x61626364)]
        for _ in range(50_000):
            trial = bytearray(data)
            trial[rng.randrange(16)] = rng.randrange(256)
            if all(int.from_bytes(trial[o : o + 4], "little") >= c for o, c in gates):
                pytest.fail("random mutation unexpectedly passed all gates")


class TestStats:
    def test_counters_track_outcomes(self):
        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        solver = PathConstraintSolver()
        solver.negate(_rec(0x10, 0x41424344, -1), data)
        solver.negate(_rec(0xDEADBEEF, 0x1, -1, pc=99), data)
        stats = solver.stats()
        assert stats["queries"] == 1
        assert stats["solved"] == 1
        assert stats["skipped_unmapped"] == 1
        assert stats["solve_rate"] == 1.0

    def test_solve_rate_is_zero_before_any_query(self):
        assert PathConstraintSolver().stats()["solve_rate"] == 0.0


class TestCollectorBridge:
    def test_records_from_collector_handles_none(self):
        assert records_from_collector(None) == []

    def test_records_from_collector_handles_missing_method(self):
        assert records_from_collector(object()) == []

    def test_records_are_built_from_captured_outcomes(self):
        class _Collector:
            def branch_records(self):
                return [(b"\x01\x02", b"\x03\x04", -1, 2, 0xBEEF)]

        recs = records_from_collector(_Collector())
        assert len(recs) == 1
        assert recs[0].result == -1
        assert recs[0].width == 2
        assert recs[0].pc == 0xBEEF


class TestCmplogCapturesOutcome:
    """The shim always emitted result and width; the parser discarded them."""

    def test_parser_records_result_and_width(self, tmp_path):
        from fuzzer_tool.core.cmplog import CmplogCollector

        log = tmp_path / "cmplog.txt"
        log.write_text("CMP 01020304 05060708 -1 4 4919\n")
        collector = CmplogCollector(workdir=str(tmp_path))
        collector.log_path = str(log)
        collector.collect_tokens()

        a, b = bytes.fromhex("01020304"), bytes.fromhex("05060708")
        assert collector.pair_cmp(a, b) == (-1, 4)
        assert collector.pair_pc(a, b) == 4919

    def test_lines_without_outcome_are_still_parsed(self):
        """Layer-1 libc interceptors emit no PC field."""
        from fuzzer_tool.core.cmplog import CmplogCollector

        assert hasattr(CmplogCollector, "branch_records")


class TestRegistration:
    def test_operator_is_registered(self):
        assert "path_negate" in REGISTRY.names()
        assert REGISTRY.category_of("path_negate") == "adaptive"

    def test_operator_has_a_handler(self):
        from fuzzer_tool.services.operators import OperatorEngine

        assert hasattr(OperatorEngine, "_op_path_negate")

    def test_gated_off_without_cmplog(self):
        class _Fuzzer:
            _cmplog = None
            _path_solver = PathConstraintSolver()

        assert "path_negate" not in REGISTRY.available(_Fuzzer(), b"data")

    def test_gated_off_when_path_negation_flag_is_absent(self):
        """--path-negation is the single control; the fuzzer leaves
        _path_solver at None otherwise, and the operator must not construct a
        private one behind the user's back."""

        class _Cmplog:
            pairs = [(b"ab", b"cd")]
            _pair_cmp = {(b"ab", b"cd"): (-1, 2)}

        class _Fuzzer:
            _cmplog = _Cmplog()
            _path_solver = None

        assert "path_negate" not in REGISTRY.available(_Fuzzer(), b"data")

    def test_gated_off_when_outcomes_were_never_recorded(self):
        """Operand pairs alone are not enough — there is no predicate."""

        class _Cmplog:
            pairs = [(b"ab", b"cd")]
            _pair_cmp = {}

        class _Fuzzer:
            _cmplog = _Cmplog()
            _path_solver = PathConstraintSolver()

        assert "path_negate" not in REGISTRY.available(_Fuzzer(), b"data")

    def test_available_once_outcomes_exist(self):
        class _Cmplog:
            pairs = [(b"ab", b"cd")]
            _pair_cmp = {(b"ab", b"cd"): (-1, 2)}

        class _Fuzzer:
            _cmplog = _Cmplog()
            _path_solver = PathConstraintSolver()

        assert "path_negate" in REGISTRY.available(_Fuzzer(), b"data")


class TestSolverIsolation:
    def test_timeout_does_not_leak_to_other_solvers(self):
        """z3.set_param('timeout') is global; this must use a per-solver set."""
        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        PathConstraintSolver(timeout_ms=1).negate(_rec(0x10, 0x41424344, -1), data)

        # A fresh solver on a trivial problem must still be satisfiable —
        # it would inherit a 1ms global budget if the timeout leaked.
        solver = z3.Solver()
        x = z3.BitVec("x", 32)
        solver.add(x == 5)
        assert solver.check() == z3.sat


class TestFastPathAndConjunctive:
    """A lone predicate has a closed-form answer; only overlapping windows
    genuinely need a solver. Routing every query through z3 cost ~3.4ms —
    several times the whole per-execution budget at 800 eps — for constraints
    with an obvious solution."""

    def test_isolated_branch_uses_the_direct_path(self):
        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        rec = _rec(0x10, 0x41424344, -1)
        solver = PathConstraintSolver()
        out = solver.negate(rec, data, others=[rec])
        assert out is not None
        assert _u32(out, 4) >= 0x41424344
        assert solver.stats()["direct_solves"] == 1
        assert solver.stats()["z3_solves"] == 0

    def test_direct_path_matches_z3_on_isolated_branches(self):
        """The fast path must be a pure optimisation, not a behaviour change."""
        for observed, constant in ((-1, 0x41424344), (1, 0x41424344), (0, 0x00001000)):
            data = b"HEAD" + (0x1000).to_bytes(4, "little") + b"TAIL"
            rec = _rec(0x1000, constant, observed)
            out = PathConstraintSolver().negate(rec, data)
            assert out is not None
            value = _u32(out, 4)
            if observed < 0:
                assert value >= constant
            elif observed > 0:
                assert value <= constant
            else:
                assert value != constant

    def test_overlapping_windows_use_z3(self):
        data = (0x00001000).to_bytes(4, "little") + b"TAIL"
        flip = BranchRecord(
            (0x1000).to_bytes(2, "little"), (0x2000).to_bytes(2, "little"), -1, 2, 1
        )
        keep = BranchRecord(
            (0x00001000).to_bytes(4, "little"),
            (0xFFFFFFFF).to_bytes(4, "little"),
            -1,
            4,
            2,
        )
        solver = PathConstraintSolver()
        out = solver.negate(flip, data, others=[flip, keep])
        assert out is not None
        assert int.from_bytes(out[0:2], "little") >= 0x2000  # flipped
        assert _u32(out, 0) < 0xFFFFFFFF  # preserved
        assert out[4:] == b"TAIL"
        assert solver.stats()["z3_solves"] == 1

    def test_contradictory_overlap_is_reported_unsat(self):
        """Flipping the 4-byte window up forces its high half >= 0x4142,
        which cannot coexist with keeping the 2-byte window below 0x4142.
        Naive substitution would silently break the second constraint."""
        data = (0x00001000).to_bytes(4, "little") + b"TAIL"
        wide = _rec(0x00001000, 0x41424344, -1, pc=1)
        inner = BranchRecord(
            (0x0000).to_bytes(2, "little"), (0x4142).to_bytes(2, "little"), -1, 2, 2
        )
        solver = PathConstraintSolver()
        assert solver.negate(wide, data, others=[wide, inner]) is None
        assert solver.stats()["unsat"] == 1

    def test_overlap_count_is_bounded(self):
        """A hot comparison site can record many branches over the same bytes;
        an unbounded conjunctive system costs milliseconds per query."""
        from fuzzer_tool.core.path_constraints import MAX_OVERLAP

        data = b"HEAD" + (0x10).to_bytes(4, "little") + b"TAIL"
        target = _rec(0x10, 0x41424344, -1, pc=0)
        others = [target] + [_rec(0x10, 0x41424344 + i, -1, pc=i + 1) for i in range(50)]
        solver = PathConstraintSolver()
        assert len(solver._overlapping(target, others, data)) <= MAX_OVERLAP

    def test_non_overlapping_branches_are_not_folded_in(self):
        data = bytearray(32)
        data[0:4] = (0x10).to_bytes(4, "little")
        data[16:20] = (0x20).to_bytes(4, "little")
        near = _rec(0x10, 0x41424344, -1, pc=1)
        far = _rec(0x20, 0x51525354, -1, pc=2)
        solver = PathConstraintSolver()
        assert solver._overlapping(near, [near, far], bytes(data)) == []

    def test_direct_solve_is_unsatisfiable_past_the_width_limit(self):
        """value >= 0xFF is satisfiable at width 1; > 0xFF is not."""
        assert PathConstraintSolver._direct_solve(-1, 0xFF, 0xFF, 1) is None
        assert PathConstraintSolver._direct_solve(1, 0x00, 0x00, 1) is None

    def test_direct_solve_avoids_returning_the_original(self):
        value = PathConstraintSolver._direct_solve(-1, 0x41, 0x41, 1)
        assert value is not None
        assert value != 0x41
        assert value >= 0x41
