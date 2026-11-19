"""inet_forms: dotted-quad IPv4 literals rewritten to other inet_aton(3) spellings."""

import socket

import pytest

from fuzzer_tool.core.mutations.inet_forms import (
    _QUAD,
    ADDR_BITS,
    MAX_PAD,
    MAX_PARTS,
    MAX_QUAD_PROBES,
    MAX_WRAP,
    OCTET,
    QUAD_HEAD,
    QUAD_NP_MIN_LEN,
    SKEW_ONE_IN,
    Radix,
    Skew,
    find_quad,
    fmt_part,
    has_quad,
    inet_forms,
    parse_aton,
    split_addr,
)
from fuzzer_tool.core.operator_registry import _CATEGORIES, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_RNG = RandPool(seed=0x1A70)
_BIG = 1 << 20
_RADIXES = tuple(Radix)
_SKEWS = tuple(Skew)
_KEEP = 1  # first draw != 0: equivalent rewrite
_PAD_FILL = b"#"


def _quad(addr: int) -> bytes:
    return ".".join(str(addr >> s & OCTET) for s in (24, 16, 8, 0)).encode()


def _wrap(lit: bytes) -> bytes:
    return b"ip=" + lit + b";"


def _lit(out: bytes) -> bytes:
    """Literal between the _wrap() prefix and suffix."""
    return out[len(b"ip=") : -len(b";")]


def _equiv_draws(parts: int, styles) -> list[int]:
    draws = [_KEEP, parts]
    for radix, pad in styles:
        draws += [_RADIXES.index(radix), pad]
    return draws


def _lax(lit: bytes) -> int:
    """Decimal-only reader that masks each part to its width (lax parser model)."""
    vals = [int(p) for p in lit.split(b".")]
    tail = ADDR_BITS - 8 * (len(vals) - 1)
    out = 0
    for v in vals[:-1]:
        out = out << 8 | v & OCTET
    return out << tail | vals[-1] & ((1 << tail) - 1)


# ── parse_aton oracle ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("127.0.0.1", 0x7F000001),
        ("127.1", 0x7F000001),
        ("127.0.1", 0x7F000001),
        ("0x7f.1", 0x7F000001),
        ("0177.0.0.01", 0x7F000001),
        ("2130706433", 0x7F000001),
        ("0X7F000001", 0x7F000001),
        ("017700000001", 0x7F000001),
        ("1.2.65535", 0x0102FFFF),
        ("1.16777215", 0x01FFFFFF),
        ("4294967295", 0xFFFFFFFF),
        ("0", 0),
        ("00", 0),
        ("1.2.3.4 ", 0x01020304),
    ],
)
def test_parse_aton_accepts(text, want):
    assert parse_aton(text) == want
    assert parse_aton(text.encode()) == want


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1.2.3.256",
        "1.2.65536",
        "1.16777216",
        "4294967296",
        "256.1",
        "1.2.3.4.5",
        "1..2",
        ".1",
        "1.",
        "08",
        "0x",
        "0xg",
        "+1",
        "1.2.3.4x",
        "1 .2",
        "١.2.3.4",  # non-ASCII digit
        b"1.2.\xff.4",
    ],
)
def test_parse_aton_rejects(text):
    assert parse_aton(text) is None


def test_control_parse_aton_matches_libc_on_canonical():
    # Hard Rule 46: oracle vs libc on unambiguous input first.
    for addr in _RNG.randint_list(0, (1 << ADDR_BITS) - 1, 200):
        lit = _quad(addr).decode()
        assert parse_aton(lit) == addr
        assert socket.inet_aton(lit) == addr.to_bytes(4, "big")


# ── helpers ────────────────────────────────────────────────────────────


def test_split_addr_tail_widths():
    addr = 0x7F000001
    assert split_addr(addr, 4) == [0x7F, 0, 0, 1]
    assert split_addr(addr, 3) == [0x7F, 0, 1]
    assert split_addr(addr, 2) == [0x7F, 1]
    assert split_addr(addr, 1) == [addr]


def test_fmt_part_radixes():
    assert fmt_part(8, Radix.DEC, 2) == "8"  # decimal never padded
    assert fmt_part(8, Radix.OCT, 0) == "010"
    assert fmt_part(8, Radix.OCT, 2) == "00010"
    assert fmt_part(0, Radix.OCT, 0) == "00"
    assert fmt_part(255, Radix.HEX_LO, 1) == "0x0ff"
    assert fmt_part(255, Radix.HEX_UP, 0) == "0XFF"


def test_has_quad():
    assert has_quad(b"GET http://10.0.0.1/ HTTP/1.1")
    assert has_quad(b"1.2.3.4")
    assert not has_quad(b"")


# ── equivalent rewrites ────────────────────────────────────────────────


def test_single_hex_integer_exact():
    data = _wrap(b"192.168.1.10")
    rng = ScriptedRng(randints=_equiv_draws(1, [(Radix.HEX_LO, 0)]))
    assert inet_forms(data, 0, rng, _BIG) == _wrap(b"0xc0a8010a")


def test_short_form_mixed_radix_exact():
    data = _wrap(b"127.0.0.1")
    styles = [(Radix.OCT, 1), (Radix.HEX_UP, 0)]
    rng = ScriptedRng(randints=_equiv_draws(2, styles))
    assert inet_forms(data, 0, rng, _BIG) == _wrap(b"00177.0X1")


def test_every_equivalent_rewrite_round_trips():
    # Draws come from RandPool, scripted: deterministic, no retry loop (Rule 39).
    addrs = _RNG.randint_list(0, (1 << ADDR_BITS) - 1, 400)
    for addr in addrs:
        parts = _RNG.randint(1, MAX_PARTS)
        styles = [
            (_RADIXES[_RNG.randint(0, len(_RADIXES) - 1)], _RNG.randint(0, MAX_PAD))
            for _ in range(parts)
        ]
        rng = ScriptedRng(randints=_equiv_draws(parts, styles))
        out = inet_forms(_wrap(_quad(addr)), 0, rng, _BIG)
        if out is None:  # identical spelling: only 4 plain decimal parts
            assert parts == MAX_PARTS
            assert all(r is Radix.DEC for r, _ in styles)
            continue

        lit = _lit(out)
        assert parse_aton(lit) == addr
        assert socket.inet_aton(lit.decode()) == addr.to_bytes(4, "big")


# ── divergent (strict vs lax) rewrites ─────────────────────────────────


def _skew_draws(skew: Skew, extra=()) -> list[int]:
    return [0, _SKEWS.index(skew), *extra]


def test_wrap_adds_multiple_of_256():
    data = _wrap(b"10.0.0.1")
    out = inet_forms(data, 0, ScriptedRng(randints=_skew_draws(Skew.WRAP, [3, 2])), _BIG)
    assert out == _wrap(b"10.0.0.513")


def test_dec_pad_zero_prefixes_parts_past_seven():
    out = inet_forms(_wrap(b"10.0.0.9"), 0, ScriptedRng(randints=_skew_draws(Skew.DEC_PAD)), _BIG)
    assert out == _wrap(b"010.0.0.09")


def test_int_wrap_adds_two_to_the_32():
    out = inet_forms(_wrap(b"0.0.0.1"), 0, ScriptedRng(randints=_skew_draws(Skew.INT_WRAP)), _BIG)
    assert out == _wrap(str((1 << ADDR_BITS) + 1).encode())


def test_every_skew_splits_strict_and_lax():
    for addr in _RNG.randint_list(0, (1 << ADDR_BITS) - 1, 200):
        skew = _SKEWS[_RNG.randint(0, len(_SKEWS) - 1)]
        extra = [_RNG.randint(0, MAX_PARTS - 1), _RNG.randint(1, MAX_WRAP)]
        rng = ScriptedRng(randints=_skew_draws(skew, extra))
        out = inet_forms(_wrap(_quad(addr)), 0, rng, _BIG)
        if out is None:  # DEC_PAD with every part < 8: nothing diverges
            assert skew is Skew.DEC_PAD
            assert all(p < 8 for p in split_addr(addr, MAX_PARTS))
            continue

        lit = _lit(out)
        assert _lax(lit) == addr
        assert parse_aton(lit) != addr


def test_skew_drawn_one_in_n():
    # Any first draw other than 0 stays on the equivalent path.
    rng = ScriptedRng(randints=[SKEW_ONE_IN - 1, 1, _RADIXES.index(Radix.DEC), 0])
    assert inet_forms(_wrap(b"0.0.1.0"), 0, rng, _BIG) == _wrap(b"256")


# ── site choice ────────────────────────────────────────────────────────


def test_byte_idx_picks_quad_at_or_after_site():
    data = b"1.1.1.1 2.2.2.2"
    rng = ScriptedRng(randints=_equiv_draws(1, [(Radix.DEC, 0)]))
    assert inet_forms(data, data.index(b"2"), rng, _BIG) == b"1.1.1.1 " + str(0x02020202).encode()


def test_byte_idx_past_last_quad_wraps_to_first():
    data = b"1.1.1.1 tail"
    rng = ScriptedRng(randints=_equiv_draws(1, [(Radix.DEC, 0)]))
    assert inet_forms(data, len(data) - 1, rng, _BIG) == str(0x01010101).encode() + b" tail"


# ── falsification / adversarial ────────────────────────────────────────


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"1.2.3",
        b"256.1.1.1",
        b"1.2.3.4.5",
        b"v1.2.3.4",
        b"01.2.3.4",
        b"1.2.3.4a",
        b"no address here",
    ],
)
def test_falsification_no_quad_declines(data):
    assert not has_quad(data)
    # Any draw would be wrong: an empty script raises if the op draws at all.
    assert inet_forms(data, 0, ScriptedRng(), _BIG) is None


def test_adversarial_edges_growth_and_large_input():
    # Quad flush with both buffer ends.
    rng = ScriptedRng(randints=_equiv_draws(1, [(Radix.HEX_UP, 0)]))
    assert inet_forms(b"1.2.3.4", 3, rng, _BIG) == b"0X1020304"

    # Growth past max_len declines rather than truncating.
    rng = ScriptedRng(randints=_equiv_draws(1, [(Radix.OCT, MAX_PAD)]))
    assert inet_forms(b"1.2.3.4", 0, rng, len(b"1.2.3.4")) is None

    # 64 KiB of filler, quad at the very end, byte_idx far past it.
    data = _PAD_FILL * (1 << 16) + b"8.8.8.8"
    rng = ScriptedRng(randints=_equiv_draws(1, [(Radix.DEC, 0)]))
    out = inet_forms(data, 5, rng, _BIG)
    assert out == _PAD_FILL * (1 << 16) + str(0x08080808).encode()


def test_adversarial_trailing_dot_sentence_still_found():
    # "at 1.2.3.4." ends a sentence; the dot is not a fifth part.
    rng = ScriptedRng(randints=_equiv_draws(1, [(Radix.DEC, 0)]))
    out = inet_forms(b"at 1.2.3.4.", 0, rng, _BIG)
    assert out == b"at " + str(0x01020304).encode() + b"."


def test_adversarial_huge_digit_runs():
    # Past CPython's int-string limit: must reject, not raise.
    assert parse_aton("9" * 5000) is None
    assert parse_aton("1." + "7" * 5000) is None
    # Leading zeros are free: strtoul skips them.
    assert parse_aton("0x" + "0" * 5000 + "1") == 1


# ── vectorised quad finder vs the plain regex ──────────────────────────


def _corpus_cases():
    """Inputs on both sides of QUAD_NP_MIN_LEN, quads straddling the QUAD_HEAD cut."""
    quads = [b"1.2.3.4", b"255.255.255.255", b"10.0.0.1"]
    fakes = [b"1.2.3", b"1.2.3.4.5", b"v1.2.3.4", b"999.1.1.1", b"1.2.3 1.2", b"01.2.3.4"]
    cases = [b"", b"1.2.3.4", bytes(QUAD_HEAD)]
    for n in (QUAD_HEAD + 1, QUAD_NP_MIN_LEN - 1, QUAD_NP_MIN_LEN, 4 * QUAD_NP_MIN_LEN):
        base = bytearray(b"\x20" * n)
        for i, f in enumerate(fakes):
            at = (i * 97) % max(1, n - 20)
            base[at : at + len(f)] = f
        cases.append(bytes(base))
        for q in quads:
            for at in (0, QUAD_HEAD - 20, QUAD_HEAD - 9, QUAD_HEAD - 3, n - len(q)):
                if 0 <= at <= n - len(q):
                    planted = bytearray(base)
                    planted[at : at + len(q)] = q
                    cases.append(bytes(planted))
    cases.append(b"1.2.3.4.5 " * (MAX_QUAD_PROBES * 8))  # probe cap -> regex
    cases.append(b"1.2.3.4 " * 4096)
    cases.append(bytes(_RNG.randbytes(1 << 14)))
    return cases


def test_control_regex_search_against_itself():
    for data in _corpus_cases():
        a, b = _QUAD.search(data), _QUAD.search(bytes(data))
        assert (a and a.span()) == (b and b.span())


@pytest.mark.parametrize("lo", [0, 1, QUAD_HEAD - 2, QUAD_NP_MIN_LEN + 3])
def test_find_quad_matches_regex(lo):
    for data in _corpus_cases():
        want = _QUAD.search(data, lo)
        got = find_quad(data, lo)
        assert (want and want.span()) == (got and got.span())


# ── registry / handler wiring ──────────────────────────────────────────


def test_registered_in_structural_category():
    assert "inet_forms" in _CATEGORIES["structural"]


def test_gated_on_dotted_quad():
    assert "inet_forms" in REGISTRY.available(None, b"GET http://10.0.0.1/")
    assert "inet_forms" not in REGISTRY.available(None, b"GET http://host/")


def test_handler_rewrites_and_declines():
    from fuzzer_tool.services.operators import OperatorEngine
    from tests.support.operator_env import make_minimal_fuzzer

    engine = OperatorEngine(make_minimal_fuzzer())
    engine.f._rng = ScriptedRng(randints=_equiv_draws(1, [(Radix.HEX_LO, 0)]))
    data = _wrap(b"10.0.0.1")
    assert bytes(engine._op_inet_forms(bytearray(data), 0, data)) == _wrap(b"0xa000001")
    assert bytes(engine._op_inet_forms(bytearray(b"none"), 0, b"none")) == b"none"
