"""Text-layer mutations: transport encodings, escapes, float spellings, UTF-16/32.

Four operators, each aimed at a decoder layer the byte-level operators only
reach by luck:

    operator         layer it attacks
    ---------------  ------------------------------------------------------
    encoding_wrap    base64/32/85 / hex / percent / quoted-printable / HTML
                     entity / Modified UTF-8 decoders (en- or de-code a span)
    escape_mutate    string-escape lexers (bad escapes, broken quoting)
    ascii_float      strtod-style parsers (denormals, overflow, 17 digits)
    utf16_transcode  wide-char paths (UTF-16/32 BOMs, odd length, lone
                     surrogates, code units past U+10FFFF)

All functions take ``(data, byte_idx, rng, max_len)`` and return the new
bytes, or None to decline: nothing to work on, or the result would not fit.
Declining beats clamping -- a truncated encoding is not the mutation the
operator is credited for (same rule as ``utf8.seq_mutate``).

Only ``rng.randint`` / ``rng.choice`` are drawn, so ``ScriptedRng`` and
``ExhaustivePool`` can drive every path.
"""

import base64
import binascii
import html
import re
import urllib.parse

# Longest span a codec rewrites. Percent-encoding triples it; bounded so one
# call cannot blow the budget on a large seed.
_MAX_SPAN = 64

# ── encoding_wrap ──────────────────────────────────────────────────────

_B64_ALPHABET = re.compile(rb"[^A-Za-z0-9+/]")
_B64_QUANTUM = 4
_HEX_RUN = re.compile(rb"(?:[0-9A-Fa-f]{2})+")


def b64_encode(span: bytes) -> bytes | None:
    return base64.b64encode(span)


def b64_decode(span: bytes) -> bytes | None:
    """Decode the alphabet bytes of *span*, re-padded; None if not base64."""
    core = _B64_ALPHABET.sub(b"", span)
    if not core:
        return None

    core += b"=" * (-len(core) % _B64_QUANTUM)
    try:
        return base64.b64decode(core, validate=True)
    except binascii.Error:
        return None


def hex_encode(span: bytes) -> bytes | None:
    return span.hex().encode()


def hex_decode(span: bytes) -> bytes | None:
    m = _HEX_RUN.match(span)
    return bytes.fromhex(m.group().decode()) if m else None


def pct_encode(span: bytes) -> bytes | None:
    """Encode every byte, unreserved ones too: b"a/" -> b"%61%2F"."""
    return b"".join(b"%%%02X" % b for b in span)


def pct_decode(span: bytes) -> bytes | None:
    return urllib.parse.unquote_to_bytes(span)


def pct_double(span: bytes) -> bytes | None:
    """Double encoding: b"/" -> b"%252F" (the classic filter bypass)."""
    return pct_encode(span).replace(b"%", b"%25")


_B32_ALPHABET = re.compile(rb"[^A-Z2-7]")
_B32_QUANTUM = 8
_A85_FRAME = (b"<~", b"~>")
# Named, decimal or hex character reference: &lt; &#65; &#x42;
_HTML_ENTITY = re.compile(rb"&(?:[A-Za-z][A-Za-z0-9]*|#[0-9]+|#[xX][0-9A-Fa-f]+);")


def b32_encode(span: bytes) -> bytes | None:
    return base64.b32encode(span)


def b32_decode(span: bytes) -> bytes | None:
    """Decode the alphabet bytes of *span*, re-padded; None if not base32."""
    core = _B32_ALPHABET.sub(b"", span)
    if not core:
        return None

    core += b"=" * (-len(core) % _B32_QUANTUM)
    try:
        return base64.b32decode(core)
    except binascii.Error:
        return None


def b85_encode(span: bytes) -> bytes | None:
    """RFC 1924 / git-style base85."""
    return base64.b85encode(span)


def a85_encode(span: bytes) -> bytes | None:
    """Adobe Ascii85, <~ ~> framed (PDF, PostScript)."""
    return base64.a85encode(span, adobe=True)


def a85_decode(span: bytes) -> bytes | None:
    """Ascii85, framed or bare; None if not Ascii85."""
    adobe = span.startswith(_A85_FRAME[0]) and span.endswith(_A85_FRAME[1])
    try:
        out = base64.a85decode(span, adobe=adobe)
    except ValueError:
        return None
    return out or None


def qp_encode(span: bytes) -> bytes | None:
    """Encode every byte, printable ones too: b"a=" -> b"=61=3D"."""
    return b"".join(b"=%02X" % b for b in span)


def qp_decode(span: bytes) -> bytes | None:
    return binascii.a2b_qp(span)


def html_encode(span: bytes) -> bytes | None:
    """Hex character references for every byte: b"a" -> b"&#x61;"."""
    return b"".join(b"&#x%X;" % b for b in span)


def _unescape(m: re.Match) -> bytes:
    return html.unescape(m.group().decode()).encode("utf-8", "surrogatepass")


def html_decode(span: bytes) -> bytes | None:
    """Resolve character references; None if *span* holds none."""
    out, n = _HTML_ENTITY.subn(_unescape, span)
    return out if n else None


# Modified UTF-8 (Java/JNI, DEX): NUL as C0 80, astral as a CESU-8 pair.
_MUTF8_NUL = b"\xc0\x80"
_UTF8_ASTRAL = re.compile(rb"[\xf0-\xf4][\x80-\xbf]{3}")
_CESU_PAIR = re.compile(rb"\xed[\xa0-\xaf][\x80-\xbf]\xed[\xb0-\xbf][\x80-\xbf]")
_ASTRAL_BASE = 0x10000


def _to_cesu(m: re.Match) -> bytes:
    """F0 9F 98 80 (U+1F600) -> ED A0 BD ED B8 80; invalid leads stay."""
    try:
        cp = ord(m.group().decode("utf-8"))
    except UnicodeDecodeError:
        return m.group()

    rest = cp - _ASTRAL_BASE
    hi, lo = chr(0xD800 + (rest >> 10)), chr(0xDC00 + (rest & 0x3FF))
    return (hi + lo).encode("utf-8", "surrogatepass")


def _from_cesu(m: re.Match) -> bytes:
    pair = m.group().decode("utf-8", "surrogatepass")
    return pair.encode("utf-16-le", "surrogatepass").decode("utf-16-le").encode()


def mutf8_encode(span: bytes) -> bytes | None:
    return _UTF8_ASTRAL.sub(_to_cesu, span.replace(b"\x00", _MUTF8_NUL))


def mutf8_decode(span: bytes) -> bytes | None:
    return _CESU_PAIR.sub(_from_cesu, span.replace(_MUTF8_NUL, b"\x00"))


# Append only: tests and replay index this tuple by position.
CODECS = (
    b64_encode,
    b64_decode,
    hex_encode,
    hex_decode,
    pct_encode,
    pct_decode,
    pct_double,
    b32_encode,
    b32_decode,
    b85_encode,
    a85_encode,
    a85_decode,
    qp_encode,
    qp_decode,
    html_encode,
    html_decode,
    mutf8_encode,
    mutf8_decode,
)


def _splice(data: bytes, start: int, end: int, repl: bytes, max_len: int) -> bytes | None:
    """Replace data[start:end] with *repl*; None if unchanged or too long."""
    if repl == data[start:end]:
        return None
    if len(data) - (end - start) + len(repl) > max_len:
        return None
    return data[:start] + repl + data[end:]


def encoding_wrap(data: bytes, byte_idx: int, rng, max_len: int) -> bytes | None:
    """En- or de-code the span starting at *byte_idx* with one codec."""
    n = len(data)
    if not n:
        return None

    start = byte_idx % n
    length = rng.randint(1, min(_MAX_SPAN, n - start))
    codec = rng.choice(CODECS)

    repl = codec(data[start : start + length])
    if repl is None:
        return None
    return _splice(data, start, start + length, repl, max_len)


# ── escape_mutate ──────────────────────────────────────────────────────

# Escapes a lexer must reject or range-check: truncated \x/\u, surrogate
# halves, code points past U+10FFFF, octal overflow, open named escapes.
ESCAPES = (
    b"\\x",
    b"\\x0",
    b"\\u",
    b"\\u12",
    b"\\uD800",
    b"\\uDC00\\uD800",
    b"\\U0010FFFF",
    b"\\U00110000",
    b"\\u{110000}",
    b"\\0",
    b"\\777",
    b"\\N{",
    b"\\c",
    b"\\",
)

_QUOTE = re.compile(rb"[\"']")
_BACKSLASH = re.compile(rb"\\")


def _find_from(data: bytes, pos: int, wanted: re.Pattern) -> int | None:
    """First match of *wanted* at or after *pos*, wrapping to 0.

    Regex search, not a byte loop: 10x faster on a 20 KB buffer.
    """
    m = wanted.search(data, pos) or wanted.search(data, 0, pos)
    return m.start() if m else None


def insert_escape(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    return _splice(data, pos, pos, rng.choice(ESCAPES), max_len)


def drop_quote(data: bytes, pos: int, _rng, max_len: int) -> bytes | None:
    """Delete a quote so the string runs to EOF (unterminated literal)."""
    i = _find_from(data, pos, _QUOTE)
    return None if i is None else data[:i] + data[i + 1 :]


def escape_quote(data: bytes, pos: int, _rng, max_len: int) -> bytes | None:
    """Backslash a closing quote so the lexer reads past it."""
    i = _find_from(data, pos, _QUOTE)
    return None if i is None else _splice(data, i, i, b"\\", max_len)


def strip_escape(data: bytes, pos: int, _rng, max_len: int) -> bytes | None:
    """Remove a backslash, exposing the raw byte it guarded."""
    i = _find_from(data, pos, _BACKSLASH)
    return None if i is None else data[:i] + data[i + 1 :]


def tail_backslash(data: bytes, _pos: int, _rng, max_len: int) -> bytes | None:
    """Dangling escape at EOF: the lexer's read-one-more path."""
    n = len(data)
    return _splice(data, n, n, b"\\", max_len)


ESCAPE_MODES = (insert_escape, drop_quote, escape_quote, strip_escape, tail_backslash)


def escape_mutate(data: bytes, byte_idx: int, rng, max_len: int) -> bytes | None:
    """Break string-escape or quoting structure near *byte_idx*."""
    if not data:
        return None

    mode = rng.choice(ESCAPE_MODES)
    return mode(data, byte_idx % (len(data) + 1), rng, max_len)


# ── ascii_float ────────────────────────────────────────────────────────

# strtod edge cases. The two 2.225...e-308 spellings hung PHP and Java
# (CVE-2010-4645); 2.47...e-324 rounds exactly half to the smallest
# denormal; the 400-digit forms blow fixed-size digit buffers.
FLOAT_EDGES = (
    b"-0",
    b"-0.0",
    b"0e0",
    b"1e309",
    b"-1e309",
    b"1e-400",
    b"1.7976931348623157e308",
    b"1.7976931348623159e308",
    b"4.9e-324",
    b"2.4703282292062327e-324",
    b"2.2250738585072011e-308",
    b"2.2250738585072012e-308",
    b"9007199254740993",
    b"0.1e99999999999999999999",
    b"1" + b"0" * 400,
    b"0." + b"0" * 400 + b"1",
    b"NaN",
    b"-Infinity",
    b"0x1p-1074",
    b"1.",
    b".1",
    b"1e",
    b"1e+",
    b"01.5",
    b"1_000.5",
)

_NUM = re.compile(rb"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NUM_BYTES = frozenset(b"0123456789.+-eE")
# How far back from byte_idx to look for the start of the covering number.
_MAX_BACKOFF = 64


def _number_at(data: bytes, pos: int) -> re.Match | None:
    """Numeric token covering or following *pos*, else the first one."""
    start = pos
    floor = max(0, pos - _MAX_BACKOFF) if data[pos] in _NUM_BYTES else pos
    while start > floor and data[start - 1] in _NUM_BYTES:
        start -= 1

    return _NUM.search(data, start) or _NUM.search(data)


def ascii_float(data: bytes, byte_idx: int, rng, max_len: int) -> bytes | None:
    """Replace the ASCII number near *byte_idx* with a strtod edge case."""
    if not data:
        return None

    m = _number_at(data, byte_idx % len(data))
    if m is None:
        return None
    return _splice(data, m.start(), m.end(), rng.choice(FLOAT_EDGES), max_len)


# ── utf16_transcode ────────────────────────────────────────────────────

_BOM_LE = b"\xff\xfe"
_BOM_BE = b"\xfe\xff"
_LONE_HIGH_SURROGATE_LE = b"\x00\xd8"  # U+D800, little-endian
_BOM32_LE = b"\xff\xfe\x00\x00"
_BOM32_BE = b"\x00\x00\xfe\xff"
_PAST_MAX_CP_LE = (0x110000).to_bytes(4, "little")  # one past U+10FFFF


def _to_utf16(raw: bytes, encoding: str) -> bytes:
    """Transcode; undecodable bytes survive as lone low surrogates."""
    return raw.decode("utf-8", "surrogateescape").encode(encoding, "surrogatepass")


def _span(data: bytes, pos: int, rng) -> tuple[int, int]:
    length = rng.randint(1, min(_MAX_SPAN, len(data) - pos))
    return pos, pos + length


def span_le(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    s, e = _span(data, pos, rng)
    return _splice(data, s, e, _BOM_LE + _to_utf16(data[s:e], "utf-16-le"), max_len)


def span_be(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    s, e = _span(data, pos, rng)
    return _splice(data, s, e, _BOM_BE + _to_utf16(data[s:e], "utf-16-be"), max_len)


def whole_le(data: bytes, _pos: int, _rng, max_len: int) -> bytes | None:
    return _splice(data, 0, len(data), _to_utf16(data, "utf-16-le"), max_len)


def odd_trunc(data: bytes, _pos: int, _rng, max_len: int) -> bytes | None:
    """UTF-16 with its last byte cut: the half-code-unit read path."""
    return _splice(data, 0, len(data), _to_utf16(data, "utf-16-le")[:-1], max_len)


def lone_surrogate(data: bytes, pos: int, _rng, max_len: int) -> bytes | None:
    return _splice(data, pos, pos, _LONE_HIGH_SURROGATE_LE, max_len)


def utf32_span_le(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    s, e = _span(data, pos, rng)
    return _splice(data, s, e, _BOM32_LE + _to_utf16(data[s:e], "utf-32-le"), max_len)


def utf32_span_be(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    s, e = _span(data, pos, rng)
    return _splice(data, s, e, _BOM32_BE + _to_utf16(data[s:e], "utf-32-be"), max_len)


def utf32_oob(data: bytes, pos: int, _rng, max_len: int) -> bytes | None:
    """UTF-32 unit past U+10FFFF: the range check UTF-16 cannot reach."""
    return _splice(data, pos, pos, _PAST_MAX_CP_LE, max_len)


# Append only: tests and replay index this tuple by position.
UTF16_MODES = (
    span_le,
    span_be,
    whole_le,
    odd_trunc,
    lone_surrogate,
    utf32_span_le,
    utf32_span_be,
    utf32_oob,
)


def utf16_transcode(data: bytes, byte_idx: int, rng, max_len: int) -> bytes | None:
    """Re-encode the buffer or a span of it as UTF-16, or break UTF-16."""
    if not data:
        return None

    mode = rng.choice(UTF16_MODES)
    return mode(data, byte_idx % len(data), rng, max_len)
