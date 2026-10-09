"""Random fill and per-byte loops in op handlers are batched.

``bytes(rng.randint(0, 255) for _ in range(n))`` drew n scalars; the
handlers now take one ``randbytes(n)``. Scripted RNGs pin the exact draw
sequence (Hard Rule 39): a handler still drawing per byte exhausts the
``randints`` script and raises StopIteration.
"""

import math
from collections import Counter

import pytest

from fuzzer_tool.core.mutations.generic import SIMD_BOUNDARIES
from fuzzer_tool.services import operators as operators_mod
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng

_BLOB = bytes(range(0xA0, 0xA0 + 64))


def _engine(**script) -> OperatorEngine:
    return OperatorEngine(make_minimal_fuzzer(pool=ScriptedRng(**script)))


def test_simd_boundary_fills_empty_in_one_draw():
    n = SIMD_BOUNDARIES[1]
    buf = bytearray()
    _engine(choice_idxs=[1], randbytes=[_BLOB[:n]])._op_simd_boundary(buf, 0, b"")
    assert buf == _BLOB[:n]


def test_varsize_insert_in_one_draw():
    size, pos = 3, 1
    buf = bytearray(b"AB")
    _engine(randints=[0, size, pos], randbytes=[_BLOB[:size]])._op_varsize(buf, 0, b"")
    assert buf == b"A" + _BLOB[:size] + b"B"


def test_block_insert_in_one_draw():
    # randints: idx, choose_len bucket (<90 -> short), length.
    idx, size = 2, 5
    buf = bytearray(b"WXYZ")
    _engine(randints=[idx, 0, size], randbytes=[_BLOB[:size]])._op_block_insert(buf, 0, b"")
    assert buf == b"WX" + _BLOB[:size] + b"YZ"


def test_length_grow_in_one_draw():
    size = 7
    buf = bytearray(b"seed")
    _engine(randints=[size], randbytes=[_BLOB[:size]])._op_length_grow(buf, 0, b"")
    assert buf == b"seed" + _BLOB[:size]


def test_length_boundary_fills_empty_in_one_draw():
    size = 9
    buf = bytearray()
    _engine(randints=[size], randbytes=[_BLOB[:size]])._op_length_boundary(buf, 0, b"")
    assert buf == _BLOB[:size]


def test_havoc_fills_empty_in_one_draw():
    size = 4
    buf = bytearray()
    _engine(randints=[size], randbytes=[_BLOB[:size]])._apply_single_mutation(buf)
    assert buf == _BLOB[:size]


@pytest.mark.parametrize("length", [8, 40, 401])
def test_skipdet_inverts_block(length):
    """Falsification: the block is XOR 0xFF, the rest untouched."""
    block, start = length // 4, 1
    data = bytes(i * 37 & 0xFF for i in range(length))
    buf = bytearray(data)
    _engine(randints=[block, start])._op_skipdet_probe(buf, 0, b"")

    expected = bytearray(data)
    for i in range(start, start + block):
        expected[i] = data[i] ^ 0xFF
    assert buf == expected


def test_colorization_fallback_draws_indices_once():
    """Adversarial: a repeated index is recoloured once per draw, as before."""
    data = b"0123456789"
    picks = 5  # len // randint(2, 10) with the draw pinned at 2
    buf = bytearray(data)
    eng = _engine(randints=[2], batch_value=3)
    eng.ctx.cmplog_pairs = None
    eng._op_colorization(buf, 0, b"")

    expected = bytearray(data)
    for _ in range(picks):
        expected[3] = operators_mod._COLORIZE_TBL[expected[3]]
    assert buf == expected


def _entropy_oracle(data: bytes) -> float:
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values()) / 8.0


@pytest.mark.parametrize(
    "data",
    [b"x", b"\x00" * 100, bytes(range(256)), bytes(range(256)) * 3 + b"ab", b"abracadabra"],
)
def test_byte_entropy_norm_matches_oracle(data):
    assert operators_mod._byte_entropy_norm(data) == pytest.approx(_entropy_oracle(data))


def test_byte_entropy_norm_empty():
    """Adversarial: empty input must not divide by zero."""
    assert operators_mod._byte_entropy_norm(b"") == 0.0
