"""Tests for the zigzag_encode and float16_edge byte operators.

Expected bytes come from arithmetic (ZigZag's defining formula), the LEB128
helper and ``struct``'s IEEE half — never from the modules under test. Draws
are scripted (Hard Rule 39).
"""

import math
import struct

import pytest

from fuzzer_tool.core.mutations.float16 import BF16_EDGES, F16_EDGES, ORDERS, TABLES, float16_edge
from fuzzer_tool.core.mutations.generic import encode_uleb128
from fuzzer_tool.core.mutations.zigzag import WIDTHS, ZIGZAG_EDGES, ZIGZAG_MODES, zigzag_encode
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

_MAX = 4096
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1


def _zz(n: int) -> int:
    """Protobuf sint64 definition: 0,-1,1,-2 -> 0,1,2,3."""
    return 2 * n if n >= 0 else -2 * n - 1


def _mode(name):
    return [m.__name__ for m in ZIGZAG_MODES].index(name)


class _MockFuzzer:
    def __init__(self, seed=1, max_len=_MAX):
        self._rng = RandPool(seed=seed)
        self.max_len = max_len


# ── zigzag_encode ──────────────────────────────────────────────────────


class TestZigzag:
    def test_rewrite_signed_field(self):
        field = (-300).to_bytes(4, "little", signed=True)
        rng = ScriptedRng(choice_idxs=[_mode("rewrite"), WIDTHS.index(4)])
        out = zigzag_encode(b"<" + field + b">", 1, rng, _MAX)
        assert out == b"<" + encode_uleb128(_zz(-300)) + b">"

    @pytest.mark.parametrize("edge", [_I64_MIN, _I64_MAX, -1])
    def test_insert_edge(self, edge):
        rng = ScriptedRng(choice_idxs=[_mode("insert_edge"), ZIGZAG_EDGES.index(edge)])
        out = zigzag_encode(b"ab", 1, rng, _MAX)
        assert out == b"a" + encode_uleb128(_zz(edge)) + b"b"

    def test_int64_min_is_ten_byte_varint(self):
        # The longest legal varint: decoders that cap at 9 bytes overflow here.
        assert len(encode_uleb128(_zz(_I64_MIN))) == 10

    # Falsification: a field running past the end is not a field.
    def test_falsify_rewrite_past_end_declines(self):
        rng = ScriptedRng(choice_idxs=[_mode("rewrite"), WIDTHS.index(8)])
        assert zigzag_encode(b"abc", 1, rng, _MAX) is None

    def test_falsify_over_budget_declines(self):
        rng = ScriptedRng(choice_idxs=[_mode("insert_edge"), ZIGZAG_EDGES.index(_I64_MIN)])
        assert zigzag_encode(b"ab", 0, rng, 5) is None

    def test_empty_declines(self):
        assert zigzag_encode(b"", 0, ScriptedRng(), _MAX) is None

    @pytest.mark.parametrize("max_len", [1, 2, 9, 16])
    def test_adversarial_never_exceeds_max_len(self, max_len):
        blobs = [bytes(range(256)), b"\xff" * 40, b"\x80" * 40, b"\x00"]
        for seed in range(40):
            rng = RandPool(seed=seed)
            for blob in blobs:
                out = zigzag_encode(blob[:max_len], seed, rng, max_len)
                assert out is None or len(out) <= max_len


# ── float16_edge ───────────────────────────────────────────────────────


def _f16(i: int) -> int:
    return F16_EDGES.index(i)


class TestFloat16:
    @pytest.mark.parametrize(
        ("bits", "check"),
        [
            (0x7C00, math.isinf),
            (0x7E00, math.isnan),
            (0x0001, lambda v: v == 2.0**-24),
            (0x7BFF, lambda v: v == 65504.0),
        ],
    )
    def test_f16_edge_le(self, bits, check):
        rng = ScriptedRng(choice_idxs=[TABLES.index(F16_EDGES), _f16(bits), ORDERS.index("little")])
        out = float16_edge(b"xxxx", 1, rng, _MAX)
        assert out[0:1] == b"x" and out[3:] == b"x"
        assert check(struct.unpack("<e", out[1:3])[0])

    def test_bf16_edge_be_is_float32_top_half(self):
        bits = 0x7F80  # bfloat16 +inf
        rng = ScriptedRng(
            choice_idxs=[TABLES.index(BF16_EDGES), BF16_EDGES.index(bits), ORDERS.index("big")]
        )
        out = float16_edge(b"\x00\x00", 0, rng, _MAX)
        assert out == struct.pack(">f", math.inf)[:2]

    def test_pos_clamped_to_last_pair(self):
        rng = ScriptedRng(choice_idxs=[TABLES.index(F16_EDGES), _f16(0x7C00), ORDERS.index("big")])
        assert float16_edge(b"abc", 2, rng, _MAX) == b"a" + struct.pack(">e", math.inf)

    # Falsification: writing the bytes already there is no mutation.
    def test_falsify_same_bytes_declines(self):
        rng = ScriptedRng(
            choice_idxs=[TABLES.index(F16_EDGES), _f16(0x7C00), ORDERS.index("little")]
        )
        assert float16_edge(struct.pack("<e", math.inf), 0, rng, _MAX) is None

    def test_short_declines(self):
        assert float16_edge(b"a", 0, ScriptedRng(), _MAX) is None

    @pytest.mark.parametrize("n", [2, 3, 17])
    def test_adversarial_preserves_length(self, n):
        for seed in range(40):
            rng = RandPool(seed=seed)
            blob = rng.randbytes(n)
            out = float16_edge(blob, seed * 7, rng, _MAX)
            assert out is None or len(out) == n


# ── wiring ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("op", ["zigzag_encode", "float16_edge"])
def test_registered_in_byte_band_with_handler(op):
    assert REGISTRY.category_of(op) == "byte"
    dispatch = OperatorEngine(_MockFuzzer()).build_dispatch()
    buf = bytearray(b"\x01\x02\x03\x04\x05\x06\x07\x08")
    out = dispatch[op](buf, 0, bytes(buf))
    assert out is None or len(out) <= _MAX
