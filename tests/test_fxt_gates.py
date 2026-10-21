"""Tests for targets/fxt_gates.c against tools/fxt_gates_ref.py.

Ground truth is the plain build of the target. Rank gates are also checked against
itertools enumeration, so a shared mistake in C and Python still fails.
"""

import ctypes
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.core.rand_pool import RandPool

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import fxt_gates_ref as ref  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="needs clang")

FULL = (1 << ref.GATE_COUNT) - 1
RANDOM_INPUTS = 300
PERTURBED_INPUTS = 300
GATE = {name: g for g, (name, _, _) in enumerate(ref.GATES)}


class Oracle:
    """ctypes view of the GATES_PLAIN build."""

    def __init__(self, lib: Path):
        self._lib = ctypes.CDLL(str(lib))
        self._lib.fuzz_gates.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self._lib.fuzz_gates.restype = ctypes.c_uint64
        self._lib.gates_operand.restype = ctypes.POINTER(ctypes.c_uint64)
        self._lib.gates_target.restype = ctypes.POINTER(ctypes.c_uint64)
        self._lib.gates_witness.restype = ctypes.POINTER(ctypes.c_ubyte)

    def run(self, buf: bytes) -> int:
        return self._lib.fuzz_gates(buf, len(buf))

    def operands(self) -> list[int]:
        op = self._lib.gates_operand()
        return [op[g] for g in range(ref.GATE_COUNT)]

    def targets(self) -> list[int]:
        t = self._lib.gates_target()
        return [t[g] for g in range(ref.GATE_COUNT)]

    def witness(self) -> bytes:
        return bytes(self._lib.gates_witness()[: ref.INPUT_LEN])


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    lib = tmp_path_factory.mktemp("fxt") / "fxt_gates_ref.so"
    src = REPO / "targets" / "fxt_gates.c"
    subprocess.run(
        ["clang", "-O2", "-shared", "-fPIC", "-DGATES_PLAIN", "-o", str(lib), str(src)], check=True
    )
    return Oracle(lib)


def _put(buf: bytes, g: int, field: bytes) -> bytes:
    """``buf`` with gate ``g``'s field replaced."""
    o = ref.OFFSET[g]
    return buf[:o] + field + buf[o + ref.FIELD_LEN[g] :]


def _expected(buf: bytes) -> list[int]:
    """Reference operand vector: SENTINEL where the field is invalid."""
    out = []
    for g, f in enumerate(ref.fields(buf)):
        valid, op = ref.gate(g, f)
        out.append(op if valid else ref.SENTINEL)
    return out


def _c_operand(oracle: Oracle, buf: bytes, g: int) -> int:
    oracle.run(buf)
    return oracle.operands()[g]


class TestOracle:
    def test_witness_opens_every_gate(self, oracle):
        assert oracle.run(oracle.witness()) == FULL

    def test_reference_solves_witness(self, oracle):
        # Python reaches the C targets on its own.
        targets = oracle.targets()
        for g, f in enumerate(ref.fields(oracle.witness())):
            assert ref.gate(g, f) == (True, targets[g]), ref.GATES[g][0]

    def test_short_input_opens_nothing(self, oracle):
        w = oracle.witness()
        assert oracle.run(w[:-1]) == 0
        assert oracle.operands() == [ref.SENTINEL] * ref.GATE_COUNT

    def test_trailing_bytes_ignored(self, oracle):
        assert oracle.run(oracle.witness() + b"\xff" * 64) == FULL


class TestReference:
    def test_matches_c_on_random_inputs(self, oracle):
        rng = RandPool(seed=3)
        for _ in range(RANDOM_INPUTS):
            buf = rng.randbytes(ref.INPUT_LEN)
            oracle.run(buf)
            assert oracle.operands() == _expected(buf), buf.hex()

    def test_matches_c_near_witness(self, oracle):
        # Valid fields are rare at random: perturb the witness to reach the rank paths.
        rng = RandPool(seed=4)
        w = oracle.witness()
        for _ in range(PERTURBED_INPUTS):
            buf = bytearray(w)
            for i in rng.randrange_list(ref.INPUT_LEN, 8):
                buf[i] ^= 1 << rng.randrange(8)
            oracle.run(bytes(buf))
            assert oracle.operands() == _expected(bytes(buf)), buf.hex()

    @pytest.mark.parametrize("fill", [0x00, 0xFF, 0x80, 0x7F])
    def test_matches_c_on_constant_fill(self, oracle, fill):
        buf = bytes([fill]) * ref.INPUT_LEN
        oracle.run(buf)
        assert oracle.operands() == _expected(buf)


class TestLayout:
    def test_field_change_moves_only_its_gate(self, oracle):
        # Adversarial: a byte flip in gate g's field leaves every other gate solved.
        w = oracle.witness()
        for g in range(ref.GATE_COUNT):
            for k in range(ref.FIELD_LEN[g]):
                buf = bytearray(w)
                buf[ref.OFFSET[g] + k] ^= 0xFF
                assert oracle.run(bytes(buf)) | (1 << g) == FULL, (ref.GATES[g][0], k)

    def test_field_lengths_cover_input(self):
        assert ref.OFFSET[-1] + ref.FIELD_LEN[-1] == ref.INPUT_LEN


class TestRankBijection:
    """Ranks equal the lex index of an itertools enumeration (independent of both codes)."""

    def test_perm_lex(self, oracle):
        g, w = GATE["perm_lex"], oracle.witness()
        for i, p in enumerate(itertools.permutations(range(8))):
            x = sum(v << (3 * j) for j, v in enumerate(p))
            assert _c_operand(oracle, _put(w, g, x.to_bytes(4, "little")), g) == i

    def test_multiset_perm(self, oracle):
        g, w = GATE["multiset_perm"], oracle.witness()
        words = sorted(set(itertools.permutations((0, 0, 1, 1, 2, 2, 3, 3))))
        for i, s in enumerate(words):
            x = sum(v << (2 * j) for j, v in enumerate(s))
            assert _c_operand(oracle, _put(w, g, x.to_bytes(2, "little")), g) == i

    def test_regression_dyck_ballot_closed_form(self, oracle):
        # ballot() was exponential recursion; the closed form must keep the lex ranks.
        g, w = GATE["dyck_lex"], oracle.witness()
        words = sorted("".join(s) for s in itertools.product("()", repeat=16) if _balanced(s))
        for i, s in enumerate(words):
            x = sum(1 << j for j, c in enumerate(s) if c == "(")
            field = x.to_bytes(2, "little")
            assert _c_operand(oracle, _put(w, g, field), g) == i
            assert ref.gate(g, field) == (True, i)

    def test_regression_rudin_shapiro_table(self, oracle):
        # The O(n) loop was 93% of exec time; the table must keep every 16-bit partial sum.
        g, w = GATE["rudin_shapiro"], oracle.witness()
        for n in range(1 << 16):
            field = n.to_bytes(2, "little")
            assert _c_operand(oracle, _put(w, g, field), g) == ref.gate(g, field)[1], n

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 0x7FFF, 0xFFFF])
    def test_rudin_shapiro_edges(self, oracle, n):
        # Direct definition, independent of both tables; negative sums wrap mod 2^32.
        want = sum(1 - 2 * ((k & (k >> 1)).bit_count() & 1) for k in range(n)) & ref.M32
        field = n.to_bytes(2, "little")
        assert (
            _c_operand(
                oracle, _put(oracle.witness(), GATE["rudin_shapiro"], field), GATE["rudin_shapiro"]
            )
            == want
        )

    def test_unbalanced_dyck_is_invalid(self, oracle):
        # Falsification: ")(" prefix and an unclosed word both fail the precondition.
        g, w = GATE["dyck_lex"], oracle.witness()
        for x in (0x0000, 0xFFFF, 0x7FFF):
            field = x.to_bytes(2, "little")
            assert _c_operand(oracle, _put(w, g, field), g) == ref.SENTINEL
            assert ref.gate(g, field)[0] is False


def _balanced(s: tuple[str, ...]) -> bool:
    depth = 0
    for c in s:
        depth += 1 if c == "(" else -1
        if depth < 0:
            return False
    return depth == 0
