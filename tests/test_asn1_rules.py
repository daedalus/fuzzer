"""Tests for the ASN.1 encoding-rule mutators: BER forms, CER forms, DER
canonicity, ASN.1 value edges, PER length determinants.

Expected bytes come from the independent encoder below, never from
``core/mutations/der.py``. RNG draws are scripted (Hard Rule 39).
"""

import pytest

from fuzzer_tool.core.mutations import asn1_values, ber_forms, cer_forms, der_canon, per
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_MAX = 4096
_EOC = b"\x00\x00"


def _tlv(tag: int, value: bytes) -> bytes:
    """Independent DER encoder (short + long form)."""
    n = len(value)
    if n < 0x80:
        return bytes((tag, n)) + value
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes((tag, 0x80 | len(raw))) + raw + value


def _idx(modes, name: str) -> int:
    return [m.__name__ for m in modes].index(name)


def _rng(*idxs) -> ScriptedRng:
    return ScriptedRng(choice_idxs=list(idxs))


# ── der_canon_mutate ─────────────────────────────────────────────────────


def test_der_bool_noncanon():
    data = _tlv(0x30, _tlv(0x01, b"\xff"))
    k = der_canon.BER_TRUES.index(0x01)
    out = der_canon.der_canon_mutate(data, _rng(_idx(der_canon.MODES, "bool_noncanon"), 0, k), _MAX)
    assert out == _tlv(0x30, _tlv(0x01, b"\x01"))


@pytest.mark.parametrize(("value", "padded"), [(b"\x05", b"\x00\x05"), (b"\x80", b"\xff\x80")])
def test_der_int_pad_sign_extends(value, padded):
    data = _tlv(0x30, _tlv(0x02, value))
    out = der_canon.der_canon_mutate(data, _rng(_idx(der_canon.MODES, "int_pad"), 0), _MAX)
    assert out == _tlv(0x30, _tlv(0x02, padded))


def test_der_bits_pad_sets_unused_bits():
    data = _tlv(0x30, _tlv(0x03, b"\x00\xaa\xa0"))
    k = der_canon.UNUSED_BITS.index(3)
    out = der_canon.der_canon_mutate(data, _rng(_idx(der_canon.MODES, "bits_pad"), 0, k), _MAX)
    assert out == _tlv(0x30, _tlv(0x03, b"\x03\xaa\xa7"))


def test_der_set_unsort_reverses_der_order():
    a, b = _tlv(0x02, b"\x01"), _tlv(0x02, b"\x02")
    data = _tlv(0x31, a + b)
    out = der_canon.der_canon_mutate(data, _rng(_idx(der_canon.MODES, "set_unsort"), 0), _MAX)
    assert out == _tlv(0x31, b + a)


def test_der_canon_declines_without_site():
    data = _tlv(0x30, _tlv(0x05, b""))
    for i in range(len(der_canon.MODES)):
        assert der_canon.der_canon_mutate(data, _rng(i, 0, 0), _MAX) is None


# ── ber_form_mutate ──────────────────────────────────────────────────────


def test_ber_indef_eoc_appends_terminator():
    inner = _tlv(0x02, b"\x01")
    data = _tlv(0x30, _tlv(0x30, inner))
    # pre-order sites: outer SEQUENCE (0), inner SEQUENCE (1)
    out = ber_forms.ber_form_mutate(data, _rng(_idx(ber_forms.MODES, "indef_eoc"), 1), _MAX)
    assert out == _tlv(0x30, b"\x30\x80" + inner + _EOC)


def test_ber_string_chunk_octets():
    data = _tlv(0x30, _tlv(0x04, b"abcd"))
    k = ber_forms.CHUNKS.index(2)
    out = ber_forms.ber_form_mutate(data, _rng(_idx(ber_forms.MODES, "string_chunk"), 0, k), _MAX)
    assert out == _tlv(0x30, _tlv(0x24, _tlv(0x04, b"ab") + _tlv(0x04, b"cd")))


def test_ber_string_chunk_bitstring_keeps_unused_on_last():
    data = _tlv(0x30, _tlv(0x03, b"\x04\xaa\xb0"))
    k = ber_forms.CHUNKS.index(2)
    out = ber_forms.ber_form_mutate(data, _rng(_idx(ber_forms.MODES, "string_chunk"), 0, k), _MAX)
    assert out == _tlv(0x30, _tlv(0x23, _tlv(0x03, b"\x00\xaa") + _tlv(0x03, b"\x04\xb0")))


@pytest.mark.parametrize(("form", "tag"), [(0, b"\x1f\x02"), (1, b"\x1f\x80\x02")])
def test_ber_tag_highform(form, tag):
    data = _tlv(0x30, _tlv(0x02, b"\x07"))
    out = ber_forms.ber_form_mutate(
        data, _rng(_idx(ber_forms.MODES, "tag_highform"), 1, form), _MAX
    )
    assert out == _tlv(0x30, tag + b"\x01\x07")


def test_ber_len_overlong():
    data = _tlv(0x30, _tlv(0x02, b"\x07"))
    k = ber_forms.PAD_BYTES.index(2)
    out = ber_forms.ber_form_mutate(data, _rng(_idx(ber_forms.MODES, "len_overlong"), 1, k), _MAX)
    assert out == _tlv(0x30, b"\x02\x83\x00\x00\x01\x07")


def test_ber_eoc_inject_in_definite():
    a, b = _tlv(0x02, b"\x01"), _tlv(0x02, b"\x02")
    data = _tlv(0x30, a + b)
    out = ber_forms.ber_form_mutate(data, _rng(_idx(ber_forms.MODES, "eoc_inject"), 0, 1), _MAX)
    assert out == _tlv(0x30, a + _EOC + b)


# ── cer_form_mutate ──────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [999, 1000, 1001])
def test_cer_frag_edge_segments_at_boundary(size):
    value = bytes(range(256)) * 4 + b"\x01"  # 1025 bytes
    data = _tlv(0x30, _tlv(0x04, value))
    k = cer_forms.SEGMENT_SIZES.index(size)
    out = cer_forms.cer_form_mutate(data, _rng(_idx(cer_forms.MODES, "frag_edge"), 0, k), _MAX)
    segs = b"".join(_tlv(0x04, value[i : i + size]) for i in range(0, len(value), size))
    assert out == _tlv(0x30, b"\x24\x80" + segs + _EOC)


def test_cer_frag_edge_grows_short_value_past_segment():
    data = _tlv(0x30, _tlv(0x04, b"ab"))
    k = cer_forms.SEGMENT_SIZES.index(1000)
    out = cer_forms.cer_form_mutate(data, _rng(_idx(cer_forms.MODES, "frag_edge"), 0, k), _MAX)
    value = b"ab" + bytes(cer_forms.CER_SEGMENT + 1 - 2)
    segs = _tlv(0x04, value[:1000]) + _tlv(0x04, value[1000:])
    assert out == _tlv(0x30, b"\x24\x80" + segs + _EOC)


def test_cer_frag_edge_declines_over_budget():
    data = _tlv(0x30, _tlv(0x04, b"ab"))
    k = cer_forms.SEGMENT_SIZES.index(1000)
    assert cer_forms.cer_form_mutate(data, _rng(0, 0, k), 64) is None


def test_cer_indef_all_converts_every_constructed():
    inner = _tlv(0x02, b"\x01")
    data = _tlv(0x30, _tlv(0x31, inner))
    out = cer_forms.cer_form_mutate(data, _rng(_idx(cer_forms.MODES, "indef_all")), _MAX)
    assert out == b"\x30\x80\x31\x80" + inner + _EOC + _EOC


def test_cer_frag_nested():
    data = _tlv(0x30, _tlv(0x04, b"ab"))
    out = cer_forms.cer_form_mutate(data, _rng(_idx(cer_forms.MODES, "frag_nested"), 0), _MAX)
    nested = b"\x24\x80" + _tlv(0x04, b"ab") + _EOC
    assert out == _tlv(0x30, b"\x24\x80" + nested + _EOC)


# ── asn1_value_mutate ────────────────────────────────────────────────────


def test_asn1_int_edge():
    data = _tlv(0x30, _tlv(0x02, b"\x05"))
    k = asn1_values.INTEGERS.index(b"\x00\x80")
    out = asn1_values.asn1_value_mutate(data, _rng(_idx(asn1_values.MODES, "int_edge"), 0, k), _MAX)
    assert out == _tlv(0x30, _tlv(0x02, b"\x00\x80"))


def test_asn1_oid_arc_rewrites_one_subidentifier():
    oid = bytes.fromhex("2a864886f70d")  # 1.2.840.113549
    data = _tlv(0x30, _tlv(0x06, oid))
    k = asn1_values.OID_ARCS.index(128)
    # site 0, subidentifier 1 (840), arc value 128 -> 81 00
    out = asn1_values.asn1_value_mutate(
        data, _rng(_idx(asn1_values.MODES, "oid_arc"), 0, 1, k), _MAX
    )
    assert out == _tlv(0x30, _tlv(0x06, b"\x2a\x81\x00\x86\xf7\x0d"))


def test_asn1_oid_arc_skips_dangling_continuation():
    data = _tlv(0x30, _tlv(0x06, b"\x2a\x86"))
    out = asn1_values.asn1_value_mutate(data, _rng(_idx(asn1_values.MODES, "oid_arc"), 0), _MAX)
    assert out is None


def test_asn1_real_swap_retypes_integer():
    data = _tlv(0x30, _tlv(0x02, b"\x05"))
    k = asn1_values.REALS.index(b"\x42")  # NaN
    out = asn1_values.asn1_value_mutate(
        data, _rng(_idx(asn1_values.MODES, "real_swap"), 0, k), _MAX
    )
    assert out == _tlv(0x30, _tlv(0x09, b"\x42"))


def test_asn1_str_charset():
    data = _tlv(0x30, _tlv(0x13, b"abc"))
    k = asn1_values.CHARSET_BREAKS[0x13].index(b"a@b")
    out = asn1_values.asn1_value_mutate(
        data, _rng(_idx(asn1_values.MODES, "str_charset"), 0, k), _MAX
    )
    assert out == _tlv(0x30, _tlv(0x13, b"a@b"))


def test_asn1_fixed_len_null_with_content():
    data = _tlv(0x30, _tlv(0x05, b""))
    k = asn1_values.FIXED_LEN_BREAKS[0x05].index(b"\x00")
    out = asn1_values.asn1_value_mutate(
        data, _rng(_idx(asn1_values.MODES, "fixed_len"), 0, k), _MAX
    )
    assert out == _tlv(0x30, _tlv(0x05, b"\x00"))


# ── per_len_mutate ───────────────────────────────────────────────────────

_PER = b"\x02ab" + b"\x03xyz"  # two octet-aligned length-prefixed fields


def test_parse_per_chain():
    assert per.parse_per(_PER) == [(0, 1, 2), (3, 1, 3)]
    assert per.parse_per(b"\x82\x00" + b"q" * 0x200 + b"\x01z") == [(0, 2, 0x200), (0x202, 1, 1)]


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"\x02ab",
        b"\x05ab",
        b"\x00\x00\x00\x00",
        b"\xc5" + b"a" * 8,
        b"\x81",
        b"\x01a" * (per.MAX_FIELDS + 1),  # chain longer than the cap
    ],
)
def test_parse_per_rejects(blob):
    assert per.parse_per(blob) is None


def test_parse_per_accepts_chain_at_cap():
    assert len(per.parse_per(b"\x01a" * per.MAX_FIELDS)) == per.MAX_FIELDS


def test_per_form_flip_short_to_long():
    out = per.per_len_mutate(_PER, _rng(_idx(per.MODES, "form_flip"), 1), _MAX)
    assert out == b"\x02ab" + b"\x80\x03xyz"


def test_per_form_flip_long_to_short():
    data = b"\x80\x02ab" + b"\x03xyz"
    out = per.per_len_mutate(data, _rng(_idx(per.MODES, "form_flip"), 0), _MAX)
    assert out == _PER


def test_per_len_skew_desyncs_length():
    # candidates for n=2: (n-1, n+1, *LEN_EDGES); pick n+1
    out = per.per_len_mutate(_PER, _rng(_idx(per.MODES, "len_skew"), 0, 1), _MAX)
    assert out == b"\x03ab" + b"\x03xyz"


def test_per_len_skew_long_edge():
    k = 2 + per.LEN_EDGES.index(0x3FFF)
    out = per.per_len_mutate(_PER, _rng(_idx(per.MODES, "len_skew"), 0, k), _MAX)
    assert out == b"\xbf\xffab" + b"\x03xyz"


def test_per_frag_header():
    k = per.FRAG_COUNTS.index(5)  # reserved fragment count
    out = per.per_len_mutate(_PER, _rng(_idx(per.MODES, "frag_header"), 1, k), _MAX)
    assert out == b"\x02ab" + b"\xc5xyz"


def test_per_bit_insert_shifts_tail():
    # insert a 1 bit before field 1: 03 78 79 7a -> 1|00000011 0111... padded
    out = per.per_len_mutate(_PER, _rng(_idx(per.MODES, "bit_insert"), 1, 1), _MAX)
    tail = int.from_bytes(b"\x03xyz", "big")
    expect = ((1 << 32) | tail) << 7
    assert out == b"\x02ab" + expect.to_bytes(5, "big")


def test_per_declines_non_per():
    assert per.per_len_mutate(b"\x30\x82\xff", _rng(0, 0), _MAX) is None


# ── adversarial: arbitrary bytes never raise, never exceed max_len ──────

_MUTATORS = (
    der_canon.der_canon_mutate,
    ber_forms.ber_form_mutate,
    cer_forms.cer_form_mutate,
    asn1_values.asn1_value_mutate,
    per.per_len_mutate,
)

_BLOBS = (
    b"",
    b"\x30",
    b"\x30\x80" * 50,
    b"\x30\x84\xff\xff\xff\xff",
    _tlv(0x31, _tlv(0x02, b"\x01") * 3),
    _tlv(0x30, _tlv(0x03, b"") + _tlv(0x03, b"\x09") + _tlv(0x06, b"") + _tlv(0x01, b"")),
    _tlv(0x30, _tlv(0x04, b"x" * 300) + _tlv(0x0C, b"\xff") + _tlv(0x1E, b"a")),
    bytes(range(256)),
    _PER,
    b"\xc1" + b"\x00" * 64,
)


@pytest.mark.parametrize("fn", _MUTATORS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("max_len", [1, 8, 64, 300, 4096])
def test_adversarial_bounded(fn, max_len):
    for seed in range(25):
        rng = RandPool(seed=seed)
        for blob in _BLOBS:
            src = blob[:max_len]
            out = fn(src, rng, max_len)
            assert out is None or (len(out) <= max_len and out != src)


# ── registry wiring ──────────────────────────────────────────────────────

_TLV_OPS = ("ber_form_mutate", "cer_form_mutate", "der_canon_mutate", "asn1_value_mutate")


def test_ops_registered_in_format_band():
    from fuzzer_tool.core.operator_registry import REGISTRY

    for op in (*_TLV_OPS, "per_len_mutate"):
        assert REGISTRY.category_of(op) == "format", op


def test_ops_have_handlers():
    from fuzzer_tool.services.operators import OperatorEngine

    for op in (*_TLV_OPS, "per_len_mutate"):
        assert callable(getattr(OperatorEngine, f"_op_{op}", None)), op


def test_sniffers_gate_on_shape():
    from fuzzer_tool.core.operator_registry import _FORMAT_SNIFFERS

    der = _tlv(0x30, _tlv(0x02, b"\x01"))
    for op in _TLV_OPS:
        assert _FORMAT_SNIFFERS[op](der), op
        assert not _FORMAT_SNIFFERS[op](b"PK\x03\x04"), op
    assert _FORMAT_SNIFFERS["per_len_mutate"](_PER)
    assert not _FORMAT_SNIFFERS["per_len_mutate"](b"\x05ab")
