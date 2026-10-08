"""Redqueen I2S through a line code: ``LineCodeEncoder`` (core/rq_encodings.py).

    input  man(b"MAGI")  --decode-->  b"MAGI"  ==  b"WXYZ"   (cmplog operands)
    search man(b"MAGI");  write man(b"WXYZ")

Expected bytes come from the bit-by-bit reference coders in
``tests/test_line_code.py``, not from the module's tables.
"""

import pytest

from fuzzer_tool.core.rq_encodings import (
    BUILTIN_ENCODERS,
    LineEncoders,
    generate_mutations,
    set_line_encoders,
)
from tests.test_line_code import _REFS

_STR_A, _STR_B = b"MAGI", b"WXYZ"


@pytest.fixture(autouse=True)
def _armed():
    """Arm for the test, disarm after: the encoder list is module state."""
    set_line_encoders(LineEncoders.ON)
    yield
    set_line_encoders(LineEncoders.OFF)


def _apply(data: bytes, mutation) -> bytes:
    offsets, repls, _enc = mutation
    buf = bytearray(data)
    for off, chunk in zip(offsets, repls, strict=True):
        buf[off : off + len(chunk)] = chunk
    return bytes(buf)


def _hits(op_a, op_b, size, kind, data, name):
    return [m for m in generate_mutations(op_a, op_b, size, kind, data) if m[2].name() == name]


@pytest.mark.parametrize("codec", ["man_ieee", "man_thomas", "gray"])
def test_solves_decoded_string_compare(codec):
    data = b"\x00" + _REFS[codec](_STR_A) + b"\x00"
    hits = _hits(_STR_A, _STR_B, 8 * len(_STR_A), "STR", data, f"line_{codec}")
    assert hits
    assert _apply(data, hits[0]) == b"\x00" + _REFS[codec](_STR_B) + b"\x00"


def test_solves_msb_first_integer_compare():
    # Decoder shifts bits MSB-first into a uint32: the cmplog operand is the
    # little-endian integer, the wire carries it big-endian.
    a, b = 0x11223344, 0x55667788
    data = b"\xaa" + _REFS["man_ieee"](a.to_bytes(4, "big"))
    hits = _hits(
        a.to_bytes(4, "little"), b.to_bytes(4, "little"), 32, "CMP", data, "line_man_ieee_r"
    )
    assert hits
    assert _REFS["man_ieee"](b.to_bytes(4, "big")) in [m[1][0] for m in hits]


# Falsification: no line-coded operand in the input, no line-code hit.
def test_falsify_plain_input_no_hit():
    data = b"xx" + _STR_A + b"yy"
    names = {m[2].name() for m in generate_mutations(_STR_A, _STR_B, 32, "STR", data)}
    assert not {n for n in names if n.startswith("line_man")}


def test_falsify_not_applicable():
    by = {e.name(): e for e in BUILTIN_ENCODERS}
    # Byte order is meaningless for a memcmp; wide operands blow the budget.
    assert not by["line_man_ieee_r"].is_applicable(32, "STR", _STR_A, _STR_B)
    assert not by["line_man_ieee"].is_applicable(512, "STR", b"a" * 64, b"b" * 64)
    assert not by["line_gray"].is_applicable(8, "CMP", b"", b"")


# Adversarial: a coded operand split by a violation must not match.
def test_adversarial_broken_code_no_hit():
    coded = _REFS["man_ieee"](_STR_A)
    data = coded[:3] + b"\x00" + coded[3:]
    assert not _hits(_STR_A, _STR_B, 32, "STR", data, "line_man_ieee")


# ── Gate ───────────────────────────────────────────────────────────────


def _line_names() -> list[str]:
    return [e.name() for e in BUILTIN_ENCODERS if e.name().startswith("line_")]


def test_arming_is_idempotent():
    armed = _line_names()
    set_line_encoders(LineEncoders.ON)
    assert _line_names() == armed
    assert len(armed) == len(set(armed))


# Falsification: disarmed, a line-coded operand is not solved, even for a
# pair whose armed result was cached before.
def test_falsify_disarmed_no_hit():
    data = b"\x00" + _REFS["man_ieee"](_STR_A)
    assert _hits(_STR_A, _STR_B, 32, "STR", data, "line_man_ieee")

    set_line_encoders(LineEncoders.OFF)
    assert not _line_names()
    assert not _hits(_STR_A, _STR_B, 32, "STR", data, "line_man_ieee")


@pytest.mark.parametrize("flag", [False, True])
def test_fuzzer_flag_sets_encoders(tmp_path, flag):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "corpus", tmp_path / "crashes"
    corpus.mkdir()
    crashes.mkdir()
    # Start from the opposite state: construction must override it.
    set_line_encoders(LineEncoders.OFF if flag else LineEncoders.ON)
    Fuzzer(target="/bin/true", corpus_dir=str(corpus), crashes_dir=str(crashes), op_line_code=flag)
    assert bool(_line_names()) is flag
