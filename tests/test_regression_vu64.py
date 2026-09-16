"""Regression tests for the vu64 op mutator.

vu64 is a variable-length u64 encoding ported from aki-akaguma/vu64:
the first byte's leading-ones count signals the total byte length (1-9),
with little-endian data layout. Unlike LEB128, zero encodes as a single
byte and the prefix carries both length and data bits.
"""

import pytest

from fuzzer_tool.core.mutations.vu64 import (
    vu64_decode_bytes,
    vu64_encode,
    vu64_encode_value,
    vu64_encoded_len,
)
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine


class _MockFuzzer:
    def __init__(self, seed=1, max_len=4096):
        self._rng = RandPool(seed=seed)
        self.max_len = max_len


def test_vu64_encode_value_matches_rust_test_vectors():
    """Ported vectors from the vu64 Rust crate."""
    vectors = [
        (0, b"\x00"),
        (0x7F, b"\x7f"),
        (0x0F0F, b"\x8f\x3c"),
        (0x0F0F_F0F0, b"\xe0\x0f\xff\xf0"),
        (0x0F0F_F0F0_0F0F, b"\xfd\x87\x07\x78\xf8\x87\x07"),
        (0x0F0F_F0F0_0F0F_F0F0, b"\xff\xf0\xf0\x0f\x0f\xf0\xf0\x0f\x0f"),
        (0xFFFFFFFFFFFFFFFF, b"\xff\xff\xff\xff\xff\xff\xff\xff\xff"),
    ]
    for value, expected in vectors:
        assert vu64_encode_value(value) == expected


def test_vu64_decode_bytes_roundtrips_all_lengths():
    """Every u64 value round-trips through its vu64 encoding."""
    values = [0, 1, 0x7F, 0x80, 0x3FFF, 0x4000, 0x0F0F, 0x0F0F_F0F0]
    values.extend(
        1 << shift for shift in (7, 8, 14, 15, 21, 22, 28, 29, 35, 36, 42, 43, 49, 50, 56, 57, 63)
    )
    values.append(0xFFFFFFFFFFFFFFFF)
    for value in values:
        encoded = vu64_encode_value(value)
        decoded, consumed = vu64_decode_bytes(encoded)
        assert decoded == value
        assert consumed == len(encoded)
        assert consumed == vu64_encoded_len(value)


def test_vu64_decode_rejects_truncated_and_invalid_input():
    with pytest.raises(ValueError):
        vu64_decode_bytes(b"")
    with pytest.raises(ValueError):
        vu64_decode_bytes(b"\x80")
    with pytest.raises(ValueError):
        vu64_decode_bytes(b"\xff")


def test_vu64_encode_rewrites_input_and_roundtrips():
    """The operator rewrites a little-endian integer as vu64."""
    from tests.support.scripted_rng import ScriptedRng

    rng = ScriptedRng(randints=[0])
    data = b"\x80\x00"
    result = vu64_encode(data, rng)
    assert result != data
    value, consumed = vu64_decode_bytes(result)
    assert value == 0x80
    assert consumed == 2


def test_vu64_encode_respects_max_len():
    rng = RandPool(seed=1)
    data = b"\x0f\x0f\x00\x00"
    result = vu64_encode(data, rng, max_len=2)
    assert len(result) <= 2


def test_vu64_operator_registered_in_byte_band():
    assert "vu64_encode" in REGISTRY.names()
    assert REGISTRY.category_of("vu64_encode") == "byte"
    assert "vu64_encode" in OPERATOR_CATEGORIES["byte"]


def test_vu64_operator_has_dispatch_handler():
    engine = OperatorEngine(_MockFuzzer())
    dispatch = engine.build_dispatch()
    assert callable(dispatch["vu64_encode"])


def test_vu64_operator_mutates_non_empty_input():
    fuzzer = _MockFuzzer(seed=1)
    engine = OperatorEngine(fuzzer)
    dispatch = engine.build_dispatch()
    buf = bytearray(b"ABCDEFGH")
    result = dispatch["vu64_encode"](buf, 0, bytes(buf))
    assert result is None or result != buf
