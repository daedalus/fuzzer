"""Behavior tests for the BER/DER (ASN.1 TLV) op mutators.

Covers core/mutations/der.py: parse/serialize round-trips (byte-minimal,
including BER long-form headers), length/tag/reorder/insert effects with
independently hand-computed expected bytes, degenerate-input safety, and
RandPool compatibility (the real fuzzer rng).
"""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.der import (
    DerMutator,
    _encode_length,
    _long_length,
    parse_der,
    serialize_der,
)
from fuzzer_tool.core.rand_pool import RandPool

# SEQUENCE { INTEGER 5, OID 1.3.5 }
SEQ = b"\x30\x08\x02\x01\x05\x06\x03\x2a\x03\x04"
# Same content with a BER long-form root length (non-minimal for 8).
SEQ_LONG = b"\x30\x81\x08\x02\x01\x05\x06\x03\x2a\x03\x04"


class _Scripted:
    """Deterministic rng stub: randint pops preset values (defaults to a)."""

    def __init__(self, values):
        self._v = list(values)

    def randint(self, a, b):
        return self._v.pop(0) if self._v else a

    def randbytes(self, n):
        return b"\xaa" * n

    def shuffle(self, seq):
        seq.reverse()


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def test_encode_length_boundaries():
    assert _encode_length(0) == b"\x00"
    assert _encode_length(127) == b"\x7f"
    assert _encode_length(128) == b"\x81\x80"
    assert _encode_length(0x1234) == b"\x82\x12\x34"
    assert _encode_length(5, indefinite=True) == b"\x80"
    assert _long_length(0) == b"\x81\x00"
    assert _long_length(8) == b"\x81\x08"


def test_parse_roundtrip_canonical():
    nodes = parse_der(SEQ)
    assert nodes is not None
    assert serialize_der(nodes) == SEQ


def test_parse_roundtrip_preserves_ber_long_form():
    """Untouched subtrees re-serialize byte-identically, BER headers included."""
    assert serialize_der(parse_der(SEQ_LONG)) == SEQ_LONG


def test_parse_rejects_non_der():
    assert parse_der(b"") is None
    assert parse_der(b"\x30") is None  # truncated: no length byte
    assert parse_der(b"\x30\x81") is None  # long form: missing length bytes
    assert parse_der(b"\x30\x05\x02") is None  # declared length > remaining
    assert parse_der(b"\x30\x03\x02\x01\x05\xff") is None  # trailing junk
    assert parse_der(b"PK\x03\x04") is None
    assert parse_der(b"\x02\x01\x05") is not None  # bare INTEGER is fine


def test_parse_deep_nesting_stays_bounded():
    """50 nested SEQUENCEs parse (depth-capped to opaque leaves) and round-trip."""
    data = b"\x30" * 50
    nodes = parse_der(data)
    assert nodes is not None
    assert serialize_der(nodes) == data


def test_len_flip_short_to_long():
    """Short->long form flip: 30 08 ... becomes 30 81 08 ... (BER-valid, non-minimal)."""
    out = DerMutator().mutate_length(SEQ, rng=_Scripted([0, 0]))
    assert out == b"\x30\x81\x08\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_len_flip_long_to_short():
    """Long->short form flip normalizes back to minimal DER."""
    out = DerMutator().mutate_length(SEQ_LONG, rng=_Scripted([0, 0]))
    assert out == SEQ


def test_len_indefinite_on_constructed():
    """Constructed SEQUENCE can be re-encoded with a BER indefinite length."""
    out = DerMutator().mutate_length(SEQ, rng=_Scripted([0, 3]))
    assert out == b"\x30\x80\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_len_grow_appends_and_reencodes():
    """Grow appends bytes and the root length field tracks the new size."""
    out = DerMutator().mutate_length(SEQ, rng=_Scripted([0, 1]))
    assert out == b"\x30\x09\x02\x01\x05\x06\x03\x2a\x03\x04\xaa"


def test_len_shrink_truncates_value():
    out = DerMutator().mutate_length(SEQ, rng=_Scripted([0, 2]))
    assert out == b"\x30\x00"


def test_len_flip_already_minimal_long_returns_none():
    """A minimal long-form length has nothing to flip to — no-op -> None."""
    data = b"\x02\x81\xc8" + bytes(200)
    assert DerMutator().mutate_length(data, rng=_Scripted([0, 0])) is None


def test_tag_class_flip():
    out = DerMutator().mutate_tag(SEQ, rng=_Scripted([0, 0]))
    assert out == b"\xf0\x08\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_tag_constructed_toggle():
    out = DerMutator().mutate_tag(SEQ, rng=_Scripted([0, 1]))
    assert out == b"\x10\x08\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_tag_number_swap():
    """Number swap replaces the low 5 bits; expected = (0x30 & 0xE0) | 0x01."""
    out = DerMutator().mutate_tag(SEQ, rng=_Scripted([0, 2]))
    assert out == b"\x21\x08\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_tag_extend_to_two_bytes():
    out = DerMutator().mutate_tag(SEQ, rng=_Scripted([0, 3]))
    assert out == b"\x3f\x01\x08\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_reorder_shuffle_reverses_siblings():
    out = DerMutator().reorder_children(SEQ, rng=_Scripted([0, 0]))
    assert out == b"\x30\x08\x06\x03\x2a\x03\x04\x02\x01\x05"


def test_reorder_duplicate_child():
    out = DerMutator().reorder_children(SEQ, rng=_Scripted([0, 1]))
    assert out == b"\x30\x0b\x02\x01\x05\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_reorder_remove_child():
    out = DerMutator().reorder_children(SEQ, rng=_Scripted([0, 2]))
    assert out == b"\x30\x05\x06\x03\x2a\x03\x04"


def test_reorder_requires_container():
    assert DerMutator().reorder_children(b"\x02\x01\x05", rng=_rng()) is None


def test_insert_null_at_front():
    """Inserting a NULL (fresh[0]) before the first sibling re-encodes the length."""
    out = DerMutator().insert_tlv(SEQ, rng=_Scripted([0, 0, 0, 0, 0, 0]))
    assert out == b"\x30\x0a\x05\x00\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_insert_truncated_tlv():
    """Splicing a truncated header (04 05, no value) is the DER-parser killer."""
    out = DerMutator().insert_tlv(SEQ, rng=_Scripted([0, 0, 0, 0, 3, 0]))
    assert out == b"\x30\x0a\x04\x05\x02\x01\x05\x06\x03\x2a\x03\x04"


def test_generator_produces_parseable_der():
    mut = DerMutator()
    outs = {mut._generate_random_der(4096, rng=_rng(s)) for s in range(10)}
    for out in outs:
        assert parse_der(out) is not None
        assert len(out) <= 4096
    assert len(outs) > 1  # not a constant generator


def test_der_gram_generates_parseable_der():
    """dictionaries/der.gram generates real DER bytes via the engine (not 0x3F
    junk); len/inner alternatives are picked independently, so at least some
    generations must be fully parseable."""
    from pathlib import Path

    from fuzzer_tool.core.grammar import load_grammar

    gram = load_grammar(Path(__file__).resolve().parent.parent / "dictionaries" / "der.gram")
    outs = {gram.generate("der", max_len=128) for _ in range(60)}
    assert len(outs) > 1
    for out in outs:
        assert out and out[0] == 0x30
        assert len(out) <= 128
    assert any(parse_der(out) is not None for out in outs)


def test_mutators_never_raise_on_degenerate_input():
    mut = DerMutator()
    degenerate = (b"", b"\x30", b"\x30\x81", b"\x30\x80", b"\xff", b"\x30" * 50, b"\xff" * 16)
    for data in degenerate:
        for fn in (
            mut.mutate_length,
            mut.mutate_tag,
            mut.reorder_children,
            mut.insert_tlv,
        ):
            fn(data, rng=_rng())


def test_mutators_work_with_randpool():
    """The real fuzzer passes RandPool, which lacks getrandbits."""
    mut = DerMutator()
    rng = RandPool(seed=7)
    for fn in (
        mut.mutate_length,
        mut.mutate_tag,
        mut.reorder_children,
        mut.insert_tlv,
    ):
        out = fn(SEQ, max_len=4096, rng=rng)
        assert out is None or isinstance(out, bytes)


def test_diversity_each_method():
    """Each op must produce more than one distinct output over seeds."""
    mut = DerMutator()
    for fn in (
        mut.mutate_length,
        mut.mutate_tag,
        mut.reorder_children,
        mut.insert_tlv,
    ):
        outs = {fn(SEQ, rng=_rng(s)) for s in range(40)}
        assert len(outs) >= 2, fn.__name__
