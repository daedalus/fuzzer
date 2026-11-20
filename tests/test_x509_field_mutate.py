"""Tests for core/mutations/x509_fields.x509_field_mutate (openssl_read.c).

Certificates are built here with an independent TLV encoder, so expected
bytes never come from the DER code under test.
"""

import pytest

from fuzzer_tool.core.mutations.x509_fields import (
    EXTENSIONS,
    MODES,
    OIDS,
    SERIALS,
    STRING_TAGS,
    TIMES,
    UNUSED_BITS,
    VERSIONS,
    x509_field_mutate,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_MAX = 4096


def _tlv(tag: int, value: bytes) -> bytes:
    """Independent DER encoder (short + long form)."""
    n = len(value)
    if n < 0x80:
        return bytes((tag, n)) + value
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes((tag, 0x80 | len(raw))) + raw + value


_SHA256_RSA = bytes.fromhex("2a864886f70d01010b")
_CN = bytes.fromhex("550403")
_ALG = _tlv(0x30, _tlv(0x06, _SHA256_RSA) + b"\x05\x00")
_NAME = _tlv(0x30, _tlv(0x31, _tlv(0x30, _tlv(0x06, _CN) + _tlv(0x0C, b"x"))))
_VALIDITY = _tlv(0x30, _tlv(0x17, b"250101000000Z") + _tlv(0x17, b"350101000000Z"))
_SPKI = _tlv(0x30, _ALG + _tlv(0x03, b"\x00\x01\x02"))
_VERSION = _tlv(0xA0, _tlv(0x02, b"\x02"))
_SERIAL = _tlv(0x02, b"\x05")
_SIG = _tlv(0x03, b"\x00\xaa")


def _cert(version: bytes = _VERSION, serial: bytes = _SERIAL, tail: bytes = b"") -> bytes:
    tbs = _tlv(0x30, version + serial + _ALG + _NAME + _VALIDITY + _NAME + _SPKI + tail)
    return _tlv(0x30, tbs + _ALG + _SIG)


def _mode(name: str) -> int:
    return [m.__name__ for m in MODES].index(name)


def test_version_edge_rewrites_explicit_version():
    rng = ScriptedRng(choice_idxs=[_mode("version_edge"), VERSIONS.index(b"\xff")])
    expected = _cert(version=_tlv(0xA0, _tlv(0x02, b"\xff")))
    assert x509_field_mutate(_cert(), rng, _MAX) == expected


def test_version_edge_inserts_missing_version():
    rng = ScriptedRng(choice_idxs=[_mode("version_edge"), VERSIONS.index(b"\x03")])
    expected = _cert(version=_tlv(0xA0, _tlv(0x02, b"\x03")))
    assert x509_field_mutate(_cert(version=b""), rng, _MAX) == expected


def test_serial_edge_writes_oversized_serial():
    big = next(s for s in SERIALS if len(s) > 20)  # RFC 5280: <= 20 octets
    rng = ScriptedRng(choice_idxs=[_mode("serial_edge"), SERIALS.index(big)])
    assert x509_field_mutate(_cert(), rng, _MAX) == _cert(serial=_tlv(0x02, big))


def test_oid_swap_replaces_chosen_oid():
    # Sites in document order: tbs sig alg, issuer CN, subject CN, spki alg, outer alg.
    new = OIDS[0]
    rng = ScriptedRng(choice_idxs=[_mode("oid_swap"), 1, 0])
    out = x509_field_mutate(_cert(), rng, _MAX)
    issuer = _tlv(0x30, _tlv(0x31, _tlv(0x30, _tlv(0x06, new) + _tlv(0x0C, b"x"))))
    tbs = _tlv(0x30, _VERSION + _SERIAL + _ALG + issuer + _VALIDITY + _NAME + _SPKI)
    assert out == _tlv(0x30, tbs + _ALG + _SIG)


def test_time_edge_swaps_tag_and_value():
    tag, value = TIMES[0]
    rng = ScriptedRng(choice_idxs=[_mode("time_edge"), 0, 0])
    out = x509_field_mutate(_cert(), rng, _MAX)
    validity = _tlv(0x30, _tlv(tag, value) + _tlv(0x17, b"350101000000Z"))
    tbs = _tlv(0x30, _VERSION + _SERIAL + _ALG + _NAME + validity + _NAME + _SPKI)
    assert out == _tlv(0x30, tbs + _ALG + _SIG)


def test_string_swap_changes_only_the_tag():
    bmp = 0x1E
    rng = ScriptedRng(choice_idxs=[_mode("string_swap"), 1, STRING_TAGS.index(bmp)])
    out = x509_field_mutate(_cert(), rng, _MAX)
    subject = _tlv(0x30, _tlv(0x31, _tlv(0x30, _tlv(0x06, _CN) + _tlv(bmp, b"x"))))
    tbs = _tlv(0x30, _VERSION + _SERIAL + _ALG + _NAME + _VALIDITY + subject + _SPKI)
    assert out == _tlv(0x30, tbs + _ALG + _SIG)


def test_bitstring_unused_sets_first_octet():
    rng = ScriptedRng(choice_idxs=[_mode("bitstring_unused"), 1, UNUSED_BITS.index(8)])
    # Sites in document order: spki key (0), signatureValue (1).
    out = x509_field_mutate(_cert(), rng, _MAX)
    assert out == _cert()[: -len(_SIG)] + _tlv(0x03, b"\x08\xaa")


def test_ext_inject_creates_extensions_block():
    oid, inner = EXTENSIONS[0]
    rng = ScriptedRng(choice_idxs=[_mode("ext_inject"), 0])
    ext = _tlv(0x30, _tlv(0x06, oid) + b"\x01\x01\xff" + _tlv(0x04, inner))
    expected = _cert(tail=_tlv(0xA3, _tlv(0x30, ext)))
    assert x509_field_mutate(_cert(), rng, _MAX) == expected


def test_ext_inject_appends_to_existing_extensions():
    oid, inner = EXTENSIONS[0]
    old = _tlv(0x30, _tlv(0x06, bytes.fromhex("551d0e")) + _tlv(0x04, b"\x04\x00"))
    new = _tlv(0x30, _tlv(0x06, oid) + b"\x01\x01\xff" + _tlv(0x04, inner))
    rng = ScriptedRng(choice_idxs=[_mode("ext_inject"), 0])
    out = x509_field_mutate(_cert(tail=_tlv(0xA3, _tlv(0x30, old))), rng, _MAX)
    assert out == _cert(tail=_tlv(0xA3, _tlv(0x30, old + new)))


# Falsification: non-certificate shapes decline the cert-only modes.
@pytest.mark.parametrize("mode", ["version_edge", "serial_edge", "ext_inject"])
def test_regression_non_cert_declines_cert_modes(mode):
    seq = _tlv(0x30, _tlv(0x02, b"\x01"))
    assert x509_field_mutate(seq, ScriptedRng(choice_idxs=[_mode(mode), 0]), _MAX) is None


def test_unparseable_declines():
    assert x509_field_mutate(b"\x30\x82\xff", ScriptedRng(choice_idxs=[0]), _MAX) is None
    assert x509_field_mutate(b"", ScriptedRng(choice_idxs=[0]), _MAX) is None


def test_over_budget_declines():
    big = next(s for s in SERIALS if len(s) > 20)
    rng = ScriptedRng(choice_idxs=[_mode("serial_edge"), SERIALS.index(big)])
    assert x509_field_mutate(_cert(), rng, len(_cert())) is None


def test_extension_values_are_valid_der():
    """Injected values must parse, or the mutator only hits the outer decoder."""
    from fuzzer_tool.core.mutations.der import parse_der

    for _oid, inner in EXTENSIONS:
        assert parse_der(inner) is not None


@pytest.mark.parametrize("max_len", [1, 8, 64, 300, 4096])
def test_adversarial_never_exceeds_max_len(max_len):
    blobs = [
        _cert(),
        _cert(version=b""),
        b"\x30\x80" * 50,
        _tlv(0x30, _tlv(0x30, b"")),
        bytes(range(256)),
        b"\x30\x84\xff\xff\xff\xff",
    ]
    for seed in range(40):
        rng = RandPool(seed=seed)
        for blob in blobs:
            out = x509_field_mutate(blob[:max_len], rng, max_len)
            assert out is None or len(out) <= max_len


def test_registered_and_gated_on_cert_shape():
    from fuzzer_tool.core.operator_registry import _FORMAT_SNIFFERS

    sniff = _FORMAT_SNIFFERS.get("x509_field_mutate")
    assert sniff is not None, "x509_field_mutate has no format sniffer"
    assert sniff(_cert())
    assert not sniff(b"\x00" + _cert())
    assert not sniff(_tlv(0x30, _tlv(0x02, b"\x01")))
