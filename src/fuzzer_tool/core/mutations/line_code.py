"""Line codes: Manchester and friends.

Physical-layer and framing codes a decoder unwraps before it parses:

    codec       users                             violation
    ----------  --------------------------------  ---------------------------
    man_ieee    10BASE-T, RFID (0 -> 10, 1 -> 01)  00 / 11 half-bit pair
    man_thomas  G.E. Thomas (1 -> 10, 0 -> 01)     00 / 11 half-bit pair
    diff_man    Token Ring (start transition = 0)  no mid-bit transition
    bmc         S/PDIF, AES3, USB-PD, LTC          no bit-start transition
    nrzi        USB, HDLC (0 toggles the line)     none (bijection)
    gray        rotary encoders (per byte)         none (bijection)
    hdlc_bits   HDLC: 0 after five 1s              flag 7E, abort FF
    usb_bits    USB: 0 after six 1s                seven 1s
    4b5b        FDDI, 100BASE-TX                   Q 00000 / I 11111
    ppp         RFC 1662 async HDLC (7D escape)    raw 7E, 7D 7E abort
    slip        RFC 1055 (DB escape)               raw C0, DB + other
    cobs        consistent overhead byte stuffing  zero byte, code overrun

Bits are MSB-first, level codes start from line level 0, and encoded bit
streams are zero-padded to a byte.

Four modes: ``encode`` a span, ``decode`` the valid prefix at a position,
``recode`` (decode, flip one data bit, re-encode: the frame stays valid
while its payload moves) and ``violate`` (insert a code violation).

Functions take ``(data, byte_idx, rng, max_len)`` and return the new bytes,
or None to decline (same contract as ``text_codec``). Only ``rng.randint``
/ ``rng.choice`` are drawn.
"""

import re
from collections.abc import Callable
from typing import NamedTuple

# Data bytes one call encodes; Manchester doubles it, so decode reads twice.
_MAX_SPAN = 64
_MAX_WINDOW = 2 * _MAX_SPAN

_NIBBLE_BITS = 4
_HEX = b"0123456789abcdef"
_INVALID = ord("x")


class Codec(NamedTuple):
    """One line code: encoder, valid-prefix decoder, violation symbols."""

    name: str
    enc: Callable[[bytes], bytes]
    # Returns (decoded, coded bytes consumed); stops at the first violation.
    dec: Callable[[bytes], tuple[bytes, int]]
    bad: tuple[bytes, ...]


# ── Bit strings ────────────────────────────────────────────────────────


def _to_bits(data: bytes) -> str:
    """b"\\xa0" -> "10100000"."""
    if not data:
        return ""
    return format(int.from_bytes(data, "big"), f"0{8 * len(data)}b")


def _from_bits(bits: str) -> bytes:
    """Zero-pad to a byte and pack."""
    bits += "0" * (-len(bits) % 8)
    if not bits:
        return b""
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


# ── Manchester: one hex digit <-> one coded byte, via translate ────────


def _manchester(one: str) -> tuple[Callable, Callable]:
    """Codec pair for the polarity that sends data 1 as *one* ("01"/"10")."""
    zero = one[::-1]
    code = [int("".join(one if (n >> i) & 1 else zero for i in (3, 2, 1, 0)), 2) for n in range(16)]

    enc_tab = bytearray(range(256))
    dec_tab = bytearray([_INVALID]) * 256
    for nib, c in enumerate(code):
        enc_tab[_HEX[nib]] = c
        dec_tab[c] = _HEX[nib]

    def enc(span: bytes) -> bytes:
        return span.hex().encode().translate(enc_tab)

    def dec(window: bytes) -> tuple[bytes, int]:
        digits = window.translate(dec_tab)
        cut = digits.find(_INVALID)
        n = len(digits) if cut < 0 else cut
        n -= n % 2
        return bytes.fromhex(digits[:n].decode()), n

    return enc, dec


_MAN_IEEE_ENC, _MAN_IEEE_DEC = _manchester("01")


# ── Level codes, as whole-int prefix XORs ─────────────────────────────
#
#   diff_man: first half f = ~prefix_xor(b), pair (f, ~f)  =  man_ieee(prefix_xor(b))
#   bmc:      line level L = prefix_xor(~b) (the NRZI stream),
#             pair (F, L) with F = ~(L >> 1): a transition at every bit start


def _prefix_xor(x: int, nbits: int) -> int:
    """Bit i (MSB-first) becomes the XOR of bits 0..i."""
    shift = 1
    while shift < nbits:
        x ^= x >> shift
        shift <<= 1
    return x


def _diff_enc(span: bytes) -> bytes:
    nbits = 8 * len(span)
    x = _prefix_xor(int.from_bytes(span, "big"), nbits)
    return _MAN_IEEE_ENC(x.to_bytes(len(span), "big"))


def _diff_dec(window: bytes) -> tuple[bytes, int]:
    """Same validity as Manchester (a mid-bit transition); then undo the XOR."""
    x, used = _MAN_IEEE_DEC(window)
    b = int.from_bytes(x, "big")
    return (b ^ (b >> 1)).to_bytes(len(x), "big"), used


def _spread(shift: int, nibble_shift: int) -> bytes:
    """Byte -> one nibble's bits at every other position, offset by *shift*."""
    tab = bytearray(256)
    for b in range(256):
        nib = (b >> nibble_shift) & 0xF
        tab[b] = sum(((nib >> i) & 1) << (2 * i + shift) for i in range(4))
    return bytes(tab)


def _gather(shift: int, nibble_shift: int) -> bytes:
    """Inverse of ``_spread``: every other bit -> one nibble."""
    tab = bytearray(256)
    for b in range(256):
        tab[b] = sum(((b >> (2 * i + shift)) & 1) << i for i in range(4)) << nibble_shift
    return bytes(tab)


# Coded pair per data byte: hi = F7 L7 .. F4 L4, lo = F3 L3 .. F0 L0.
_SPREAD = {(s, n): _spread(s, n) for s in (0, 1) for n in (0, _NIBBLE_BITS)}
_GATHER = {(s, n): _gather(s, n) for s in (0, 1) for n in (0, _NIBBLE_BITS)}


def _interleave(f: bytes, line: bytes) -> bytes:
    n = len(f)
    out = bytearray(2 * n)
    for half, nib in ((0, _NIBBLE_BITS), (1, 0)):
        hi = int.from_bytes(f.translate(_SPREAD[1, nib]), "big")
        lo = int.from_bytes(line.translate(_SPREAD[0, nib]), "big")
        out[half::2] = (hi | lo).to_bytes(n, "big")
    return bytes(out)


def _deinterleave(coded: bytes, shift: int) -> int:
    """Bits at *shift* parity of each coded pair, packed as one int."""
    hi = int.from_bytes(coded[0::2].translate(_GATHER[shift, _NIBBLE_BITS]), "big")
    return hi | int.from_bytes(coded[1::2].translate(_GATHER[shift, 0]), "big")


def _bmc_enc(span: bytes) -> bytes:
    nbits = 8 * len(span)
    mask = (1 << nbits) - 1
    line = _prefix_xor(~int.from_bytes(span, "big") & mask, nbits)
    first = ~(line >> 1) & mask
    return _interleave(first.to_bytes(len(span), "big"), line.to_bytes(len(span), "big"))


def _bmc_dec(window: bytes) -> tuple[bytes, int]:
    n = len(window) // 2
    nbits = 8 * n
    coded = window[: 2 * n]
    first, line = _deinterleave(coded, 1), _deinterleave(coded, 0)

    # Violation: no transition at a bit start (first half equals prior level).
    bad = ~(first ^ (line >> 1)) & ((1 << nbits) - 1)
    valid = nbits - bad.bit_length() if bad else nbits
    keep = valid // 8
    data = (first ^ line).to_bytes(n, "big") if n else b""
    return data[:keep], 2 * keep


# ── Bijections: NRZI (prefix XOR on one big int), Gray (per byte) ──────


def _nrzi_enc(span: bytes) -> bytes:
    """Line level is the running XOR of toggles; a toggle is a data 0."""
    nbits = 8 * len(span)
    if not nbits:
        return b""

    level = ~int.from_bytes(span, "big") & ((1 << nbits) - 1)
    shift = 1
    while shift < nbits:
        level ^= level >> shift
        shift <<= 1
    return level.to_bytes(len(span), "big")


def _nrzi_dec(window: bytes) -> tuple[bytes, int]:
    nbits = 8 * len(window)
    if not nbits:
        return b"", 0

    level = int.from_bytes(window, "big")
    data = ~(level ^ (level >> 1)) & ((1 << nbits) - 1)
    return data.to_bytes(len(window), "big"), len(window)


_GRAY = bytes(b ^ (b >> 1) for b in range(256))
_GRAY_INV = bytes(_GRAY.index(b) for b in range(256))


def _gray_enc(span: bytes) -> bytes:
    return span.translate(_GRAY)


def _gray_dec(window: bytes) -> tuple[bytes, int]:
    return window.translate(_GRAY_INV), len(window)


# ── Bit stuffing: str.replace on the bit string ────────────────────────


def _stuffing(run: int) -> tuple[Callable, Callable]:
    """Codec pair inserting a 0 after *run* consecutive 1s."""
    ones = "1" * run
    stuffed = ones + "0"
    violation = ones + "1"

    def enc(span: bytes) -> bytes:
        return _from_bits(_to_bits(span).replace(ones, stuffed))

    def dec(window: bytes) -> tuple[bytes, int]:
        bits = _to_bits(window)
        cut = bits.find(violation)
        cut = len(bits) if cut < 0 else cut

        body = bits[:cut].replace(stuffed, ones)
        return _from_bits(body[: len(body) - len(body) % 8]), cut // 8

    return enc, dec


# ── 4B5B ───────────────────────────────────────────────────────────────

_4B5B_GROUP = 5
_4B5B = (
    "11110", "01001", "10100", "10101", "01010", "01011", "01110", "01111",
    "10010", "10011", "10110", "10111", "11010", "11011", "11100", "11101",
)  # fmt: skip
_4B5B_BYTE = tuple(_4B5B[b >> _NIBBLE_BITS] + _4B5B[b & 0xF] for b in range(256))


def _4b5b_enc(span: bytes) -> bytes:
    return _from_bits("".join(map(_4B5B_BYTE.__getitem__, span)))


_4B5B_PREFIX = re.compile("(?:" + "|".join(_4B5B) + ")*")
_4B5B_PAIR = {_4B5B_BYTE[b]: b for b in range(256)}
_4B5B_TEN = re.compile(".{10}")


def _4b5b_dec(window: bytes) -> tuple[bytes, int]:
    bits = _to_bits(window)
    groups = _4B5B_PREFIX.match(bits).end() // _4B5B_GROUP
    n = groups - groups % 2
    out = bytes(map(_4B5B_PAIR.__getitem__, _4B5B_TEN.findall(bits, 0, n * _4B5B_GROUP)))
    # Bytes holding the decoded groups: 10 bits per data byte, rounded up.
    return out, min(-(-n * _4B5B_GROUP // 8), len(window))


# ── Byte stuffing: regex prefix match + unescape ───────────────────────

_PPP_XOR = 0x20
_PPP_NEEDS_ESC = re.compile(rb"[\x00-\x1f\x7d\x7e]")
_PPP_PREFIX = re.compile(rb"(?:\x7d[^\x7e]|[^\x7d\x7e])*", re.S)
_PPP_ESCAPED = re.compile(rb"\x7d(.)", re.S)


def _ppp_enc(span: bytes) -> bytes:
    return _PPP_NEEDS_ESC.sub(lambda m: bytes((0x7D, m[0][0] ^ _PPP_XOR)), span)


def _ppp_dec(window: bytes) -> tuple[bytes, int]:
    used = _PPP_PREFIX.match(window).end()
    return _PPP_ESCAPED.sub(lambda m: bytes((m[1][0] ^ _PPP_XOR,)), window[:used]), used


_SLIP_PREFIX = re.compile(rb"(?:\xdb[\xdc\xdd]|[^\xc0\xdb])*")
_SLIP_ESCAPED = re.compile(rb"\xdb([\xdc\xdd])")
_SLIP_UNESC = {b"\xdc": b"\xc0", b"\xdd": b"\xdb"}


def _slip_enc(span: bytes) -> bytes:
    return span.replace(b"\xdb", b"\xdb\xdd").replace(b"\xc0", b"\xdb\xdc")


def _slip_dec(window: bytes) -> tuple[bytes, int]:
    used = _SLIP_PREFIX.match(window).end()
    return _SLIP_ESCAPED.sub(lambda m: _SLIP_UNESC[m[1]], window[:used]), used


# Longest COBS block: code 0xFF, 254 data bytes and no implied zero.
_COBS_FULL = 0xFF


def _cobs_enc(span: bytes) -> bytes:
    out = bytearray()
    for chunk in span.split(b"\x00"):
        while len(chunk) >= _COBS_FULL - 1:
            out += bytes((_COBS_FULL,)) + chunk[: _COBS_FULL - 1]
            chunk = chunk[_COBS_FULL - 1 :]
        out += bytes((len(chunk) + 1,)) + chunk
    return bytes(out)


def _cobs_dec(window: bytes) -> tuple[bytes, int]:
    """Implied zeros are emitted only between two whole blocks."""
    out, i, n, zero = bytearray(), 0, len(window), False
    while i < n:
        code = window[i]
        if code == 0 or i + code > n:
            break
        if zero:
            out.append(0)
        out += window[i + 1 : i + code]
        zero = code != _COBS_FULL
        i += code
    return bytes(out), i


_MAN_BAD = (b"\x00", b"\xff")
# Append only: tests index this tuple by name.
CODECS = (
    Codec("man_ieee", _MAN_IEEE_ENC, _MAN_IEEE_DEC, _MAN_BAD),
    Codec("man_thomas", *_manchester("10"), _MAN_BAD),
    Codec("diff_man", _diff_enc, _diff_dec, _MAN_BAD),
    Codec("bmc", _bmc_enc, _bmc_dec, _MAN_BAD),
    Codec("nrzi", _nrzi_enc, _nrzi_dec, ()),
    Codec("gray", _gray_enc, _gray_dec, ()),
    Codec("hdlc_bits", *_stuffing(5), (b"\x7e", b"\xff")),
    Codec("usb_bits", *_stuffing(6), (b"\xff", b"\xfe")),
    Codec("4b5b", _4b5b_enc, _4b5b_dec, (b"\x00\x00", b"\xff\xff")),
    Codec("ppp", _ppp_enc, _ppp_dec, (b"\x7e", b"\x7d\x7e")),
    Codec("slip", _slip_enc, _slip_dec, (b"\xc0", b"\xdb\x00")),
    Codec("cobs", _cobs_enc, _cobs_dec, (b"\x00", b"\xff")),
)


# ── Modes ──────────────────────────────────────────────────────────────


def _splice(data: bytes, start: int, end: int, repl: bytes, max_len: int) -> bytes | None:
    """Replace data[start:end] with *repl*; None if unchanged or too long."""
    if repl == data[start:end]:
        return None
    if len(data) - (end - start) + len(repl) > max_len:
        return None
    return data[:start] + repl + data[end:]


def encode(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    """Line-code the span at *pos*."""
    codec = rng.choice(CODECS)
    end = pos + rng.randint(1, min(_MAX_SPAN, len(data) - pos))
    return _splice(data, pos, end, codec.enc(data[pos:end]), max_len)


def decode(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    """Replace the valid coded prefix at *pos* with its payload."""
    codec = rng.choice(CODECS)
    out, used = codec.dec(data[pos : pos + _MAX_WINDOW])
    if not used:
        return None
    return _splice(data, pos, pos + used, out, max_len)


def recode(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    """Flip one payload bit under the code, keeping the frame valid."""
    codec = rng.choice(CODECS)
    out, used = codec.dec(data[pos : pos + _MAX_WINDOW])
    if not out:
        return None

    bit = rng.randint(0, 8 * len(out) - 1)
    flipped = bytearray(out)
    flipped[bit >> 3] ^= 0x80 >> (bit & 7)
    return _splice(data, pos, pos + used, codec.enc(bytes(flipped)), max_len)


def violate(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    """Insert a symbol the code forbids at *pos*."""
    codec = rng.choice(CODECS)
    if not codec.bad:
        return None
    return _splice(data, pos, pos, rng.choice(codec.bad), max_len)


# Append only: tests index this tuple by name.
LINE_MODES = (encode, decode, recode, violate)


def line_code(data: bytes, byte_idx: int, rng, max_len: int) -> bytes | None:
    """Encode, decode, recode or violate a line code at *byte_idx*."""
    if not data:
        return None

    mode = rng.choice(LINE_MODES)
    return mode(data, byte_idx % len(data), rng, max_len)
