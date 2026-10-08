"""Tests for the ``line_code`` operator (core/mutations/line_code.py).

Expected bytes come from the bit-by-bit reference coders below, written from
each code's definition -- never from the module's lookup tables. Draws are
scripted (Hard Rule 39).
"""

import pytest

from fuzzer_tool.core.mutations.line_code import CODECS, LINE_MODES, line_code
from fuzzer_tool.core.operator_registry import _CATEGORIES, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

_MAX = 4096

# 4B5B data symbols, FDDI / 100BASE-TX (ANSI X3.148).
_4B5B = (
    "11110", "01001", "10100", "10101", "01010", "01011", "01110", "01111",
    "10010", "10011", "10110", "10111", "11010", "11011", "11100", "11101",
)  # fmt: skip


# ── Reference coders (definitions, bit by bit) ─────────────────────────


def _bits(data: bytes) -> list[int]:
    return [(b >> (7 - i)) & 1 for b in data for i in range(8)]


def _pack(bits: list[int]) -> bytes:
    bits = bits + [0] * (-len(bits) % 8)
    return bytes(int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8))


def _ref_man(data: bytes, one: tuple[int, int]) -> bytes:
    zero = (one[1], one[0])
    return _pack([h for b in _bits(data) for h in (one if b else zero)])


def _ref_diff_man(data: bytes) -> bytes:
    """Transition at bit start means 0; always a mid-bit transition."""
    level, out = 0, []
    for b in _bits(data):
        first = level ^ (b == 0)
        level = first ^ 1
        out += [first, level]
    return _pack(out)


def _ref_bmc(data: bytes) -> bytes:
    """Biphase mark: transition at every bit start, extra mid-bit for 1."""
    level, out = 0, []
    for b in _bits(data):
        first = level ^ 1
        level = first ^ b
        out += [first, level]
    return _pack(out)


def _ref_nrzi(data: bytes) -> bytes:
    """USB/HDLC NRZI: 0 toggles the line, 1 holds it."""
    level, out = 0, []
    for b in _bits(data):
        level ^= b == 0
        out.append(level)
    return _pack(out)


def _ref_stuff(data: bytes, run: int) -> bytes:
    """Insert a 0 after *run* consecutive 1s."""
    ones, out = 0, []
    for b in _bits(data):
        out.append(b)
        ones = ones + 1 if b else 0
        if ones == run:
            out.append(0)
            ones = 0
    return _pack(out)


def _ref_4b5b(data: bytes) -> bytes:
    s = "".join(_4B5B[b >> 4] + _4B5B[b & 0xF] for b in data)
    return _pack([int(c) for c in s])


def _ref_ppp(data: bytes) -> bytes:
    """RFC 1662 async HDLC, default ACCM: escape 7E, 7D and 00-1F."""
    out = bytearray()
    for b in data:
        out += bytes((0x7D, b ^ 0x20)) if b in (0x7E, 0x7D) or b < 0x20 else bytes((b,))
    return bytes(out)


def _ref_slip(data: bytes) -> bytes:
    """RFC 1055: END C0 -> DB DC, ESC DB -> DB DD."""
    return data.replace(b"\xdb", b"\xdb\xdd").replace(b"\xc0", b"\xdb\xdc")


def _ref_cobs(data: bytes) -> bytes:
    """Split on zeros; each chunk as blocks of at most 254 bytes."""
    out = bytearray()
    for chunk in data.split(b"\x00"):
        while len(chunk) >= 254:
            out += b"\xff" + chunk[:254]
            chunk = chunk[254:]
        out += bytes((len(chunk) + 1,)) + chunk
    return bytes(out)


_REFS = {
    "man_ieee": lambda d: _ref_man(d, (0, 1)),
    "man_thomas": lambda d: _ref_man(d, (1, 0)),
    "diff_man": _ref_diff_man,
    "bmc": _ref_bmc,
    "nrzi": _ref_nrzi,
    "gray": lambda d: bytes(b ^ (b >> 1) for b in d),
    "hdlc_bits": lambda d: _ref_stuff(d, 5),
    "usb_bits": lambda d: _ref_stuff(d, 6),
    "4b5b": _ref_4b5b,
    "ppp": _ref_ppp,
    "slip": _ref_slip,
    "cobs": _ref_cobs,
}

_SPANS = (b"", b"\x00", b"\xff", b"\x7e\x7d\xc0\xdb", bytes(range(64)), b"\xff" * 300)

_CODEC = {c.name: c for c in CODECS}


def _mode(name: str) -> int:
    return [m.__name__ for m in LINE_MODES].index(name)


def _idx(name: str) -> int:
    return [c.name for c in CODECS].index(name)


# ── Codecs ─────────────────────────────────────────────────────────────


def test_every_codec_has_a_reference():
    assert set(_CODEC) == set(_REFS)


# Control (Hard Rule 46): the oracle must be self-consistent before it
# judges anything -- Manchester's two polarities are bitwise complements.
def test_control_reference_polarities_complement():
    span = bytes(range(64))
    ieee, thomas = _REFS["man_ieee"](span), _REFS["man_thomas"](span)
    assert bytes(a ^ b for a, b in zip(ieee, thomas, strict=True)) == b"\xff" * len(ieee)


@pytest.mark.parametrize("name", sorted(_REFS))
@pytest.mark.parametrize("span", _SPANS, ids=lambda s: f"len{len(s)}")
def test_encode_matches_reference(name, span):
    assert _CODEC[name].enc(span) == _REFS[name](span)


@pytest.mark.parametrize("name", sorted(_REFS))
@pytest.mark.parametrize("span", _SPANS, ids=lambda s: f"len{len(s)}")
def test_decode_inverts_encode(name, span):
    coded = _REFS[name](span)
    assert _CODEC[name].dec(coded) == (span, len(coded))


def test_manchester_decode_stops_at_violation():
    coded = _REFS["man_ieee"](b"hi")
    assert _CODEC["man_ieee"].dec(coded + b"\x00" + coded) == (b"hi", len(coded))


def test_hdlc_decode_stops_at_flag():
    # 0x10 0x20 stuffs nothing and ends in 0, so the flag starts byte 2.
    out, used = _CODEC["hdlc_bits"].dec(b"\x10\x20\x7e\x10")
    assert (out, used) == (b"\x10\x20", 2)


@pytest.mark.parametrize(
    ("name", "coded", "expect"),
    [
        ("ppp", b"a\x7d\x5eb\x7ec", (b"a\x7eb", 4)),  # raw flag ends the frame
        ("ppp", b"ab\x7d", (b"ab", 2)),  # dangling escape
        ("slip", b"a\xdb\xdcb\xdb\x00", (b"a\xc0b", 4)),  # bad escape
        ("cobs", b"\x03ab\x02c\x00\x02d", (b"ab\x00c", 5)),  # zero inside
        ("cobs", b"\x02a\x09bc", (b"a", 2)),  # code overruns the window
    ],
)
def test_byte_stuffing_prefix(name, coded, expect):
    assert _CODEC[name].dec(coded) == expect


@pytest.mark.parametrize("name", sorted(_REFS))
def test_bad_symbols_break_decoding(name):
    """Each listed violation, inserted mid-stream, cuts the decoded prefix."""
    span = b"\x10\x20\x30\x40"
    coded = _REFS[name](span)
    for bad in _CODEC[name].bad:
        out, _ = _CODEC[name].dec(coded + bad + coded)
        assert len(out) < 2 * len(span), bad


@pytest.mark.parametrize("name", ["nrzi", "gray"])
def test_bijections_have_no_violation(name):
    assert _CODEC[name].bad == ()


# ── Modes ──────────────────────────────────────────────────────────────


def test_encode_mode():
    rng = ScriptedRng(choice_idxs=[_mode("encode"), _idx("man_ieee")], randints=[2])
    out = line_code(b"<hi>", 1, rng, _MAX)
    assert out == b"<" + _REFS["man_ieee"](b"hi") + b">"


def test_decode_mode():
    data = b"<" + _REFS["man_thomas"](b"hi") + b">"
    rng = ScriptedRng(choice_idxs=[_mode("decode"), _idx("man_thomas")])
    assert line_code(data, 1, rng, _MAX) == b"<hi>"


def test_recode_mode_flips_one_data_bit():
    data = b"<" + _REFS["bmc"](b"hi") + b"\x00"
    rng = ScriptedRng(choice_idxs=[_mode("recode"), _idx("bmc")], randints=[15])
    flipped = bytes((ord("h"), ord("i") ^ 1))
    assert line_code(data, 1, rng, _MAX) == b"<" + _REFS["bmc"](flipped) + b"\x00"


def test_violate_mode_inserts_bad_symbol():
    bad = _CODEC["cobs"].bad
    rng = ScriptedRng(choice_idxs=[_mode("violate"), _idx("cobs"), 1])
    assert line_code(b"ab", 1, rng, _MAX) == b"a" + bad[1] + b"b"


def test_pos_wraps_modulo_length():
    rng = ScriptedRng(choice_idxs=[_mode("encode"), _idx("gray")], randints=[1])
    assert line_code(b"ab\x03", 5, rng, _MAX) == b"ab" + bytes((3 ^ 1,))


# Falsification: each mode must decline when its precondition is false.
def test_falsify_decode_on_uncoded_declines():
    rng = ScriptedRng(choice_idxs=[_mode("decode"), _idx("man_ieee")])
    assert line_code(b"\x00\x00\x00", 0, rng, _MAX) is None


def test_falsify_violate_bijection_declines():
    rng = ScriptedRng(choice_idxs=[_mode("violate"), _idx("gray")])
    assert line_code(b"abc", 0, rng, _MAX) is None


def test_falsify_unchanged_encode_declines():
    # Gray code of 0x00 is 0x00: no mutation, no credit.
    rng = ScriptedRng(choice_idxs=[_mode("encode"), _idx("gray")], randints=[1])
    assert line_code(b"\x00", 0, rng, _MAX) is None


def test_falsify_over_budget_declines():
    rng = ScriptedRng(choice_idxs=[_mode("encode"), _idx("man_ieee")], randints=[4])
    assert line_code(b"abcd", 0, rng, 7) is None


def test_empty_declines():
    assert line_code(b"", 0, ScriptedRng(), _MAX) is None


@pytest.mark.parametrize("max_len", [1, 2, 9, 64])
def test_adversarial_never_exceeds_max_len(max_len):
    blobs = [bytes(range(256)), b"\xff" * 80, b"\x00" * 80, b"\x55" * 80, b"\x7e\x7d\xc0\xdb" * 20]
    for seed in range(60):
        rng = RandPool(seed=seed)
        for blob in blobs:
            out = line_code(blob[:max_len], seed, rng, max_len)
            assert out is None or len(out) <= max_len


@pytest.mark.parametrize("name", sorted(_REFS))
def test_adversarial_decoders_on_random_bytes(name):
    rng = RandPool(seed=7)
    for n in (0, 1, 2, 3, 5, 17, 128):
        blob = rng.randbytes(n)
        out, used = _CODEC[name].dec(blob)
        assert 0 <= used <= n
        assert isinstance(out, bytes)


# ── Wiring ─────────────────────────────────────────────────────────────


class _Fuzzer:
    def __init__(self, on: bool, seed: int = 1):
        self.op_line_code = on
        self._rng = RandPool(seed=seed)
        self.max_len = _MAX


def test_registered_in_structural_band():
    assert "line_code" in _CATEGORIES["structural"]


def test_gated_on_flag():
    assert "line_code" not in REGISTRY.available(_Fuzzer(False), b"abc")
    assert "line_code" in REGISTRY.available(_Fuzzer(True), b"abc")


def test_handler_dispatches():
    dispatch = OperatorEngine(_Fuzzer(True)).build_dispatch()
    buf = bytearray(b"<" + _REFS["man_ieee"](b"hi") + b">")
    out = dispatch["line_code"](buf, 1, bytes(buf))
    assert out is None or len(out) <= _MAX
