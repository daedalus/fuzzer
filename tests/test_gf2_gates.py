"""Tests for targets/gf2_gates.c and tools/gf2_gates_experiment.py.

Ground truth comes from the plain build of the target; the solver arms are checked
against it, never against their own bookkeeping.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import gf2_gates_experiment as exp  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")

SEED = 1
BUDGET = 500
FULL = exp.SOLVED_ALL


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    lib = tmp_path_factory.mktemp("gf2") / "gf2_gates_ref.so"
    src = exp.REPO / "targets" / "gf2_gates.c"
    subprocess.run(
        ["cc", "-O2", "-shared", "-fPIC", "-DGATES_PLAIN", "-o", str(lib), str(src)], check=True
    )
    return exp.Oracle(lib)


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


class TestSolveSystem:
    def test_solution_satisfies_every_row(self):
        masks = [0b0111, 0b1010, 0b0101]
        rhs = 0b101
        x = exp.solve_system(masks, rhs)
        assert x is not None
        assert [_parity(m & x) for m in masks] == [rhs >> j & 1 for j in range(3)]

    def test_inconsistent_system_is_none(self):
        # Rows 0 and 1 are equal but demand different parities.
        assert exp.solve_system([0b11, 0b11], 0b01) is None

    def test_zero_target_gives_zero(self):
        assert exp.solve_system([0b101, 0b010], 0) == 0

    def test_singular_map_with_free_variable(self):
        # One equation over three unknowns: any solution must satisfy it.
        x = exp.solve_system([0b111], 1)
        assert x is not None and _parity(x) == 1

    def test_empty_row_with_nonzero_target_is_none(self):
        assert exp.solve_system([0], 1) is None


class TestOracle:
    def test_zero_input_opens_nothing(self, oracle):
        assert oracle.run(exp.BASE) == 0

    def test_short_input_opens_nothing(self, oracle):
        assert oracle.run(b"K" * (exp.INPUT_LEN - 1)) == 0

    def test_k_byte_opens_only_g0(self, oracle):
        assert oracle.run(b"K" + bytes(exp.INPUT_LEN - 1)) == 1


class TestArms:
    def test_operand_map_opens_every_gate(self, oracle):
        mask, used = exp.run_one(oracle, "operand_map", SEED, BUDGET)
        assert mask == FULL
        assert used is not None and used <= BUDGET

    def test_probe_affine_cannot_open_rare_event_gates(self, oracle):
        # G2/G3 branch bits are equality events, not affine: the probe must decline.
        mask, _ = exp.run_one(oracle, "probe_affine", SEED, BUDGET)
        assert mask >> 2 & 1 == 0 and mask >> 3 & 1 == 0

    def test_identity_control_solves_nothing(self, oracle):
        mask, used = exp.run_one(oracle, "identity", SEED, BUDGET)
        assert mask == 0 and used is None

    def test_g3_solution_breaks_under_a_changed_target_bit(self, oracle):
        # Adversarial: solve for the wrong target; the oracle must keep G3 closed.
        bits = exp.FIELDS[3][1]
        oracle.run(exp.put_field(3, 0))
        const = oracle.operand(3)
        solver = exp.IncrementalXorMapSolver(bits)
        for i in range(bits):
            oracle.run(exp.put_field(3, 1 << i))
            solver.add_pair(1 << i, oracle.operand(3) ^ const)
        sol, _ = solver.solve()
        rows = [sum(1 << i for i in idx) for idx in sol]
        wrong = exp.solve_system(rows, (oracle.target(3) ^ 1) ^ const)
        assert wrong is not None
        assert oracle.run(exp.put_field(3, wrong)) >> 3 & 1 == 0
