"""Reversible-gate mutations (CNOT / CCNOT / CSWAP) and the rev_circuit op."""

import pytest

from fuzzer_tool.core.mutations.reversible import Gate, apply_gate, rev_circuit
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng


def _bit(buf, b):
    return (buf[b >> 3] >> (b & 7)) & 1


def _popcount(buf):
    return sum(bin(x).count("1") for x in buf)


def _random_circuit(rng, nbits, depth):
    """Gate list with distinct bit operands, drawn from RandPool."""
    circuit = []
    for _ in range(depth):
        gate = Gate(rng.randint(0, len(Gate) - 1))
        bits = tuple(rng.sample(range(nbits), 2 if gate is Gate.CNOT else 3))
        circuit.append((gate, bits))
    return circuit


# ── primitives ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("ctrl, expected", [(0, 0x00), (1, 0x02)])
def test_cnot_flips_target_iff_control(ctrl, expected):
    buf = bytearray([ctrl])
    apply_gate(buf, Gate.CNOT, (0, 1))
    assert buf[0] & 0x02 == expected


@pytest.mark.parametrize("a, b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_ccnot_needs_both_controls(a, b):
    buf = bytearray([a | (b << 1)])
    apply_gate(buf, Gate.CCNOT, (0, 1, 2))
    assert _bit(buf, 2) == (a & b)


@pytest.mark.parametrize("ctrl", [0, 1])
def test_cswap_swaps_iff_control(ctrl):
    buf = bytearray([ctrl | 0x02])  # x=1, y=0
    apply_gate(buf, Gate.CSWAP, (0, 1, 2))
    assert (_bit(buf, 1), _bit(buf, 2)) == ((0, 1) if ctrl else (1, 0))


# ── falsification: reversibility and conservation ───────────────────────


def test_reversed_circuit_restores_input():
    """Every gate is self-inverse, so the reversed circuit undoes it."""
    rng = RandPool(seed=1234)
    data = bytes(rng.randbytes(16))
    circuit = _random_circuit(rng, len(data) * 8, 64)

    buf = bytearray(data)
    for gate, bits in circuit:
        apply_gate(buf, gate, bits)
    assert buf != bytearray(data)  # control: the circuit did something

    for gate, bits in reversed(circuit):
        apply_gate(buf, gate, bits)
    assert bytes(buf) == data


def test_cswap_preserves_popcount():
    rng = RandPool(seed=99)
    buf = bytearray(rng.randbytes(16))
    before = _popcount(buf)
    for _ in range(256):
        apply_gate(buf, Gate.CSWAP, tuple(rng.sample(range(128), 3)))
    assert _popcount(buf) == before


# ── rev_circuit (scripted) ──────────────────────────────────────────────


def test_rev_circuit_scripted_cnot():
    """depth=1, gate=CNOT, control bit 0 (set), target bit 9."""
    data = b"\x01\x00"
    rng = ScriptedRng(randints=[1, int(Gate.CNOT), 0, 9])
    expected = bytearray(data)
    expected[9 >> 3] ^= 1 << (9 & 7)
    assert rev_circuit(data, rng) == bytes(expected)


def test_rev_circuit_tagged_control():
    """With ctrl_bytes, the control comes from a tagged byte."""
    data = b"\x00\x00\x80"
    # depth=1, gate=CNOT, choice -> byte 2, control bit 7, target bit 0
    rng = ScriptedRng(randints=[1, int(Gate.CNOT), 7, 0], choice_idxs=[0])
    assert rev_circuit(data, rng, ctrl_bytes=(2,)) == b"\x01\x00\x80"


# ── adversarial ─────────────────────────────────────────────────────────


def test_rev_circuit_empty():
    assert rev_circuit(b"", ScriptedRng()) == b""


def test_rev_circuit_duplicate_operands_skipped():
    """control == target is not a reversible gate; it must be skipped."""
    data = b"\x01"
    rng = ScriptedRng(randints=[1, int(Gate.CNOT), 0, 0] * 4)
    assert rev_circuit(data, rng) == data


def test_rev_circuit_all_zero_is_unchanged():
    """All controls read 0, so nothing can fire: input returned as-is."""
    data = bytes(8)
    rng = RandPool(seed=7)
    assert rev_circuit(data, rng) == data


def test_rev_circuit_single_byte_and_length():
    rng = RandPool(seed=3)
    for _ in range(64):
        out = rev_circuit(b"\xff", rng)
        assert len(out) == 1


# ── registry / dispatch wiring ──────────────────────────────────────────


def test_rev_circuit_registered_in_bit_band():
    assert "rev_circuit" in REGISTRY.names()
    assert "rev_circuit" in OPERATOR_CATEGORIES["bit"]


def test_rev_circuit_handler_dispatches(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = Fuzzer(
        target="/bin/true",
        corpus_dir=str(tmp_path / "corpus"),
        crashes_dir=str(tmp_path / "crashes"),
        max_len=64,
    )
    dispatch = f._operators.build_dispatch()
    out = dispatch["rev_circuit"](bytearray(b"\xff" * 8), 0, b"\xff" * 8)
    assert out is not None and len(out) == 8


# ── bit_swap_<w> arms honour their width ────────────────────────────────


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_regression_bit_swap_width(tmp_path, width):
    """bit_swap_<8w> swaps inside a width-byte window, not a random one."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    f = Fuzzer(
        target="/bin/true",
        corpus_dir=str(tmp_path / "corpus"),
        crashes_dir=str(tmp_path / "crashes"),
        max_len=64,
    )
    top = 8 * width - 1
    f._rng = ScriptedRng(randints=[0, 0, top])  # start, pos1, pos2
    handler = getattr(f._operators, f"_op_bit_swap_{8 * width}")

    out = handler(bytearray(b"\x01" + bytes(9)), 0, b"")

    expected = bytearray(10)
    expected[top >> 3] |= 1 << (top & 7)
    assert out == expected
