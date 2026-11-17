"""Float-aware redqueen: ``FloatAsciiEncoder`` (core/rq_encodings.py).

The shim logs float/double compares as 4/8-byte CMP records of the raw IEEE
bits (``__sanitizer_cov_trace_cmpf/cmpd``). A text input reaches the compare
through strtod, so the operand appears as a numeral:

    input "x=3.25;"  --strtod-->  3.25 == 4.5   (cmpd operands)
    write "x=4.5;"   (and the ULP neighbours: CMP variants are +-1 on the bits)

Expected bytes are built with struct/format here, not by the module.
"""

import math
import struct

import pytest

from fuzzer_tool.core.rq_encodings import generate_mutations


def _d(x: float) -> bytes:
    return struct.pack(">d", x)


def _f(x: float) -> bytes:
    return struct.pack(">f", x)


def _float_repls(op_a, op_b, size, data) -> list[bytes]:
    muts = generate_mutations(op_a, op_b, size, "CMP", data)
    return [m[1][0] for m in muts if m[2].name().startswith("float_ascii")]


def test_solves_double_text_compare():
    repls = _float_repls(_d(3.25), _d(4.5), 64, b"x=3.25;")
    assert b"4.5" in repls


def test_ulp_neighbours_offered():
    repls = _float_repls(_d(3.25), _d(4.5), 64, b"x=3.25;")
    assert repr(math.nextafter(4.5, math.inf)).encode() in repls
    assert repr(math.nextafter(4.5, -math.inf)).encode() in repls


def test_solves_float32_shortest_rendering():
    """0.1f prints as 0.1 at float32 precision, not 0.10000000149011612."""
    repls = _float_repls(_f(0.1), _f(2.5), 32, b"v 0.1 end")
    assert b"2.5" in repls


def test_g_style_integral_value():
    """strtod("100") == 100.0: the %g rendering has no '.0'."""
    repls = _float_repls(_d(100.0), _d(250.0), 64, b"n=100&")
    assert b"250" in repls


def test_integer_compare_not_treated_as_float():
    """Falsification: an int bit pattern decodes to an ugly float; no float hits."""
    a, b = 0x41424344, 0x41424345
    data = repr(struct.unpack(">f", a.to_bytes(4, "big"))[0]).encode()
    assert _float_repls(a.to_bytes(4, "big"), b.to_bytes(4, "big"), 32, data) == []


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 5e-324])
def test_non_finite_and_denormal_skipped(bad):
    """Adversarial: NaN/inf/denormal operands never yield float renderings."""
    assert _float_repls(_d(bad), _d(4.5), 64, b"x=nan inf 5e-324") == []
    assert _float_repls(_d(4.5), _d(bad), 64, b"x=4.5") == []


def test_string_compare_ignored():
    assert [
        m
        for m in generate_mutations(b"3.25", b"4.5", 512, "STR", b"3.25")
        if m[2].name().startswith("float_ascii")
    ] == []
