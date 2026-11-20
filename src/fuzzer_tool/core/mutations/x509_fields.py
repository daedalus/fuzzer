"""X.509 field-level edits for ``openssl_read.c``.

The generic DER operators (``core/mutations/der.py``) know TLVs, not
certificates: they flip tags and lengths but never write a value an X.509
decoder branches on. This walks the parsed tree and edits the fields that
do (RFC 5280 §4.1):

    Certificate  SEQUENCE { tbsCertificate, signatureAlgorithm, signature }
    tbs          SEQUENCE { [0] version, serial, sigAlg, issuer, validity,
                            subject, spki, [3] extensions ... }

    mode              edit                                  decoder branch
    ----------------  ------------------------------------  -------------------
    version_edge      [0] INTEGER := 0..3/ff/empty/2-byte   v1/v2/v3 gating
    serial_edge       serial := empty/negative/21 octets    serial length checks
    oid_swap          any OID := known alg/ext/attr/curve   algorithm dispatch
    time_edge         UTCTime/GeneralizedTime := edge       time parsing
    string_swap       name string tag := other string type  string conversion
    bitstring_unused  BIT STRING unused-bits := 1/7/8/ff    bit-string checks
    ext_inject        add critical extension (created [3])  extension decoders

oid/time/string/bitstring work on any DER (CRLs, CSRs, PKCS#7 carry the
same fields); version/serial/ext_inject need a certificate-shaped tree.
Returns None to decline: unparseable input, no site, no-op, or over budget.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.der import DerNode, parse_der, serialize_der

# ASN.1 universal tags (X.680).
TAG_BOOLEAN = 0x01
TAG_INTEGER = 0x02
TAG_BIT_STRING = 0x03
TAG_OCTET_STRING = 0x04
TAG_OID = 0x06
TAG_SEQUENCE = 0x30
TAG_UTC_TIME = 0x17
TAG_GENERALIZED_TIME = 0x18

# Context tags inside tbsCertificate (RFC 5280 §4.1).
TAG_VERSION = 0xA0
TAG_EXTENSIONS = 0xA3

# RFC 5280 §4.1.2.2: serials are at most 20 octets.
MAX_SERIAL_OCTETS = 20

_DER_TRUE = b"\x01\x01\xff"

# v1=0, v2=1, v3=2; 3 and ff are undefined; empty and non-minimal are invalid.
VERSIONS = (b"\x00", b"\x01", b"\x02", b"\x03", b"\xff", b"", b"\x00\x02")

SERIALS = (
    b"",  # empty INTEGER: invalid
    b"\x00",  # zero: forbidden by RFC 5280, accepted in practice
    b"\x80",  # negative
    b"\xff" * 8,  # negative, wide
    b"\x00\x01",  # non-minimal
    b"\x01" * MAX_SERIAL_OCTETS,  # at the limit
    b"\x01" * (MAX_SERIAL_OCTETS + 1),  # one over
)

# Encoded OID bodies: signature/key algorithms, curves, extensions, name
# attributes, plus malformed encodings (empty, dangling continuation, huge arc).
OIDS = (
    bytes.fromhex("2a864886f70d010101"),  # rsaEncryption
    bytes.fromhex("2a864886f70d01010b"),  # sha256WithRSAEncryption
    bytes.fromhex("2a864886f70d01010a"),  # RSASSA-PSS
    bytes.fromhex("2a8648ce3d0201"),  # id-ecPublicKey
    bytes.fromhex("2a8648ce3d040302"),  # ecdsa-with-SHA256
    bytes.fromhex("2a8648ce3d030107"),  # prime256v1
    bytes.fromhex("2b8104000a"),  # secp256k1
    bytes.fromhex("2b6570"),  # Ed25519
    bytes.fromhex("2a8648ce380401"),  # DSA
    bytes.fromhex("551d13"),  # basicConstraints
    bytes.fromhex("551d0f"),  # keyUsage
    bytes.fromhex("551d11"),  # subjectAltName
    bytes.fromhex("551d1e"),  # nameConstraints
    bytes.fromhex("551d20"),  # certificatePolicies
    bytes.fromhex("551d25"),  # extKeyUsage
    bytes.fromhex("550403"),  # commonName
    bytes.fromhex("2a864886f70d010901"),  # emailAddress
    b"",
    b"\x80",
    b"\x2a\x86",
    b"\x2a" + b"\xff" * 12 + b"\x7f",
)

# (tag, value): wrong form, invalid dates, two-digit-year pivot, tag/format mismatch.
TIMES = (
    (TAG_UTC_TIME, b"500101000000Z"),  # 1950: pivot low edge
    (TAG_UTC_TIME, b"491231235959Z"),  # 2049: pivot high edge
    (TAG_UTC_TIME, b"250230000000Z"),  # Feb 30
    (TAG_UTC_TIME, b"251301000000Z"),  # month 13
    (TAG_UTC_TIME, b"2501010000Z"),  # no seconds
    (TAG_UTC_TIME, b"250101000000+0100"),  # offset, not Z
    (TAG_UTC_TIME, b"250101000000"),  # no zone
    (TAG_UTC_TIME, b""),
    (TAG_GENERALIZED_TIME, b"20500101000000Z"),
    (TAG_GENERALIZED_TIME, b"99991231235959Z"),
    (TAG_GENERALIZED_TIME, b"20250101000000.999Z"),  # fractional seconds
    (TAG_GENERALIZED_TIME, b"250101000000Z"),  # UTCTime body, GeneralizedTime tag
    (TAG_UTC_TIME, b"20250101000000Z"),  # GeneralizedTime body, UTCTime tag
)

# UTF8, Numeric, Printable, T61, IA5, Universal, BMP, Visible.
STRING_TAGS = (0x0C, 0x12, 0x13, 0x14, 0x16, 0x1C, 0x1E, 0x1A)
_STRING_TAG_SET = frozenset(STRING_TAGS)

# Unused-bit counts: 1..7 legal, 8 and ff illegal.
UNUSED_BITS = (1, 7, 8, 0xFF)

# (extnID body, extnValue DER): edge values for the decoders OpenSSL caches.
EXTENSIONS = (
    # basicConstraints CA:TRUE pathLen = huge
    (bytes.fromhex("551d13"), b"\x30\x0a\x01\x01\xff\x02\x05\x7f\xff\xff\xff\xff"),
    # basicConstraints CA:TRUE pathLen negative
    (bytes.fromhex("551d13"), b"\x30\x06\x01\x01\xff\x02\x01\xff"),
    # keyUsage with 8 unused bits
    (bytes.fromhex("551d0f"), b"\x03\x02\x08\xff"),
    # subjectAltName: empty dNSName, iPAddress of 5 bytes, rfc822Name
    (bytes.fromhex("551d11"), b"\x30\x0b\x82\x00\x87\x05\x01\x02\x03\x04\x05\x81\x00"),
    # nameConstraints: permitted dNSName "", excluded 5-byte iPAddress
    (bytes.fromhex("551d1e"), bytes.fromhex("3011a00430028200a109300787050102030405")),
    # certificatePolicies: anyPolicy with an empty qualifier list (SIZE 1..MAX)
    (bytes.fromhex("551d20"), bytes.fromhex("300a30080604551d20003000")),
    # extKeyUsage: empty sequence
    (bytes.fromhex("551d25"), b"\x30\x00"),
    # authorityKeyIdentifier: keyIdentifier + serial without issuer
    (bytes.fromhex("551d23"), b"\x30\x07\x80\x01\x00\x82\x02\x00\x01"),
)


def _walk(nodes: list[DerNode]) -> list[DerNode]:
    """Pre-order (document-order) node list."""
    out: list[DerNode] = []
    stack = list(reversed(nodes))
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(reversed(n.children))
    return out


def _tag_sites(nodes: list[DerNode], tags) -> list[DerNode]:
    return [n for n in _walk(nodes) if len(n.tag) == 1 and n.tag[0] in tags]


def _tbs(nodes: list[DerNode]) -> DerNode | None:
    """tbsCertificate, or None when the tree is not certificate-shaped."""
    if len(nodes) != 1:
        return None
    cert = nodes[0]
    if cert.tag[0] != TAG_SEQUENCE or not cert.parsed_children or len(cert.children) < 3:
        return None
    tbs = cert.children[0]
    if tbs.tag[0] != TAG_SEQUENCE or not tbs.parsed_children or len(tbs.children) < 6:
        return None
    return tbs


def _tlv(tag: int, value: bytes) -> bytes:
    return bytes((tag,)) + _len(len(value)) + value


def _len(n: int) -> bytes:
    if n < 0x80:
        return bytes((n,))
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes((0x80 | len(raw),)) + raw


def version_edge(nodes, rng) -> bytes | None:
    tbs = _tbs(nodes)
    if tbs is None:
        return None

    value = rng.choice(VERSIONS)
    first = tbs.children[0]

    # Absent (v1 default): insert an explicit [0] in front.
    if first.tag[0] != TAG_VERSION:
        version = _tlv(TAG_VERSION, _tlv(TAG_INTEGER, value))
        return serialize_der(nodes, target=tbs, tvalue=version + tbs.value)

    if not first.children:
        return None
    return serialize_der(nodes, target=first.children[0], tvalue=value)


def serial_edge(nodes, rng) -> bytes | None:
    tbs = _tbs(nodes)
    if tbs is None:
        return None

    idx = 1 if tbs.children[0].tag[0] == TAG_VERSION else 0
    serial = tbs.children[idx]
    if serial.tag[0] != TAG_INTEGER:
        return None
    return serialize_der(nodes, target=serial, tvalue=rng.choice(SERIALS))


def oid_swap(nodes, rng) -> bytes | None:
    sites = _tag_sites(nodes, (TAG_OID,))
    if not sites:
        return None
    return serialize_der(nodes, target=rng.choice(sites), tvalue=rng.choice(OIDS))


def time_edge(nodes, rng) -> bytes | None:
    sites = _tag_sites(nodes, (TAG_UTC_TIME, TAG_GENERALIZED_TIME))
    if not sites:
        return None

    target = rng.choice(sites)
    tag, value = rng.choice(TIMES)
    return serialize_der(nodes, target=target, tvalue=value, ttag=bytes((tag,)))


def string_swap(nodes, rng) -> bytes | None:
    sites = _tag_sites(nodes, _STRING_TAG_SET)
    if not sites:
        return None
    return serialize_der(nodes, target=rng.choice(sites), ttag=bytes((rng.choice(STRING_TAGS),)))


def bitstring_unused(nodes, rng) -> bytes | None:
    sites = [n for n in _tag_sites(nodes, (TAG_BIT_STRING,)) if n.value]
    if not sites:
        return None

    target = rng.choice(sites)
    value = bytes((rng.choice(UNUSED_BITS),)) + target.value[1:]
    return serialize_der(nodes, target=target, tvalue=value)


def ext_inject(nodes, rng) -> bytes | None:
    tbs = _tbs(nodes)
    if tbs is None:
        return None

    oid, inner = rng.choice(EXTENSIONS)
    ext = _tlv(TAG_SEQUENCE, _tlv(TAG_OID, oid) + _DER_TRUE + _tlv(TAG_OCTET_STRING, inner))

    # Existing [3] { SEQUENCE OF Extension }: append to the inner SEQUENCE.
    block = next((c for c in tbs.children if c.tag[0] == TAG_EXTENSIONS), None)
    if block is not None and block.children:
        exts = block.children[0]
        return serialize_der(nodes, target=exts, tvalue=exts.value + ext)

    # None yet: append a fresh [3] block to tbsCertificate.
    wrapper = _tlv(TAG_EXTENSIONS, _tlv(TAG_SEQUENCE, ext))
    return serialize_der(nodes, target=tbs, tvalue=tbs.value + wrapper)


MODES = (
    version_edge,
    serial_edge,
    oid_swap,
    time_edge,
    string_swap,
    bitstring_unused,
    ext_inject,
)


def x509_field_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one X.509 field edit; None when it cannot apply."""
    nodes = parse_der(data)
    if not nodes:
        return None

    out = rng.choice(MODES)(nodes, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
