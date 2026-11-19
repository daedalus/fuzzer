"""inet_aton(3) spellings of a dotted-quad IPv4 literal.

``inet_aton`` accepts far more than ``a.b.c.d``: 1-4 parts, the last one
filling the remaining bits, each in decimal, octal (leading ``0``) or hex
(``0x``/``0X``)::

    127.0.0.1  ==  127.1  ==  0x7f.0.1  ==  0177.0.0.01  ==  2130706433

URL filters, SSRF allow-lists and ACLs that match the dotted quad textually
miss these; parsers that re-implement the grammar disagree with libc on them.
This operator finds a quad and rewrites it:

- equivalent (``SKEW_ONE_IN - 1`` of calls): random part count, per-part
  radix and zero padding; always parses back to the same address.
- skewed (1 in ``SKEW_ONE_IN``): a spelling strict and lax parsers read
  differently -- a part + k*256 (``&0xff`` wrap), decimal parts zero-padded
  (octal to inet_aton, decimal to atoi), or the integer + 2^32.

:func:`parse_aton` is an independent inet_aton model, the test oracle.

Only ``rng.randint`` is drawn, so ``ScriptedRng`` drives every path.
"""

from __future__ import annotations

import enum
import re

import numpy as np

ADDR_BITS = 32
OCTET = 0xFF
OCTET_BITS = 8
MAX_PARTS = 4
#: Extra leading zeros on an octal / hex part (``0x0001``).
MAX_PAD = 3
#: Largest k in a WRAP part + k*256.
MAX_WRAP = 3
#: One call in SKEW_ONE_IN writes a strict/lax-divergent spelling.
SKEW_ONE_IN = 4
#: Longest dotted quad ("255.255.255.255"): back-off bound for the site scan.
QUAD_MAX_LEN = 15
#: Smallest decimal part whose zero-padded spelling changes value (``08``).
_OCTAL_BASE = 8
#: Significant digits past which a part overflows 32 bits in any radix
#: (octal needs 11); longer runs are rejected before int() sees them.
_MAX_DIGITS = 11

#: Scan length from which the numpy prefilter is used; below, the regex wins
#: (min-of-9: 512 B regex 22 us vs 31 us; 1 KiB 37-46 us vs 31 us).
QUAD_NP_MIN_LEN = 1024
#: Bytes the regex scans before the prefilter: early quads exit here.
QUAD_HEAD = 256
#: Candidate chains probed one by one; denser input (SNMP OIDs, "1.2.3.4.5 ..."
#: at 3 anchored matches per dot) goes back to the regex from the first chain.
MAX_QUAD_PROBES = 64
#: Most digits in a canonical octet: dot-to-dot gap is 2..OCT_DIGITS+1.
_OCT_DIGITS = 3
_DOT, _ZERO, _DIGITS = ord("."), ord("0"), 10
#: A head match is trusted only this far before the cut: a whole quad plus
#: the 2-byte lookahead, so no attempt before it saw the truncation.
_CUT_MARGIN = QUAD_MAX_LEN + 2

_ADDR_LIMIT = 1 << ADDR_BITS
_SPACE = b" \t\n\r\x0b\x0c"
_QUAD_BYTES = frozenset(b"0123456789.")

# Canonical octet: 0-255, no leading zero (unambiguous to every parser).
_OCT = rb"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
# Not glued to a word or another part: "v1.2.3.4" and "1.2.3.4.5" are not quads.
_QUAD = re.compile(rb"(?<![\w.])" + _OCT + rb"(?:\." + _OCT + rb"){3}(?!\w|\.[0-9])")

_HEX = re.compile(rb"0[xX]([0-9a-fA-F]+)")
_OCTAL = re.compile(rb"0([0-7]*)")
_DEC = re.compile(rb"[1-9][0-9]*")
_HEX_RADIX, _OCT_RADIX, _DEC_RADIX = 16, 8, 10


class Radix(enum.Enum):
    """Part spelling: (prefix, format spec). Padding goes after the prefix."""

    DEC = ("", "d")
    OCT = ("0", "o")
    HEX_LO = ("0x", "x")
    HEX_UP = ("0X", "X")


class Skew(enum.Enum):
    """Spellings strict (inet_aton) and lax (atoi + mask) parsers split on."""

    WRAP = enum.auto()
    DEC_PAD = enum.auto()
    INT_WRAP = enum.auto()


_RADIXES = tuple(Radix)
_SKEWS = tuple(Skew)


# ── oracle ─────────────────────────────────────────────────────────────


def _part(field: bytes) -> int | None:
    """One part's value; None if it is not a well-formed number."""
    for pat, base in ((_HEX, _HEX_RADIX), (_OCTAL, _OCT_RADIX), (_DEC, _DEC_RADIX)):
        m = pat.fullmatch(field)
        if m is None:
            continue

        digits = (m.group(1) if m.re.groups else m.group()).lstrip(b"0")
        if len(digits) > _MAX_DIGITS:
            return _ADDR_LIMIT  # out of range: _pack rejects it
        return int(digits or b"0", base)
    return None


def _pack(vals: list[int]) -> int | None:
    """Join parts: all but the last are octets, the last fills the rest."""
    tail_bits = ADDR_BITS - OCTET_BITS * (len(vals) - 1)
    if vals[-1] >> tail_bits or any(v > OCTET for v in vals[:-1]):
        return None

    addr = 0
    for v in vals[:-1]:
        addr = addr << OCTET_BITS | v
    return addr << tail_bits | vals[-1]


def parse_aton(text: str | bytes) -> int | None:
    """inet_aton(3) model: 32-bit address, or None if libc would reject *text*.

    Trailing whitespace is accepted (glibc stops there); an empty part,
    ``0x`` without digits and octal 8/9 are rejected.
    """
    if isinstance(text, str):
        if not text.isascii():
            return None
        text = text.encode()

    fields = text.rstrip(_SPACE).split(b".")
    if len(fields) > MAX_PARTS:
        return None

    vals = []
    for field in fields:
        v = _part(field)
        if v is None:
            return None
        vals.append(v)
    return _pack(vals)


# ── rewrite ────────────────────────────────────────────────────────────


def _chains(data: bytes, lo: int) -> list[int]:
    """Dots at/after *lo* starting digit.digit.digit.digit: a quad's necessary shape.

    ``dot[p]``: a dot between digits. ``step(m)[p]``: a dot at p with an
    ``m`` dot 2..4 bytes later and only digits between. Three chained dots
    make a candidate.
    """
    base = max(0, lo - 1)
    a = np.frombuffer(data, np.uint8, offset=base)
    n, pad = len(a), _OCT_DIGITS + 1
    dig = np.zeros(n + pad, bool)
    dig[:n] = (a - _ZERO) < _DIGITS  # uint8 wrap: one compare
    dot = np.zeros(n + pad, bool)
    dot[1 : n - 1] = (a[1:-1] == _DOT) & dig[: n - 2] & dig[2:n]

    runs = [dig[1 : n + 1]]
    for k in range(2, pad):
        runs.append(runs[-1] & dig[k : n + k])

    def step(m: np.ndarray) -> np.ndarray:
        out = np.zeros(n + pad, bool)
        hit = out[:n]
        for k, run in enumerate(runs, 2):
            hit |= m[k : n + k] & run
        hit &= dot[:n]
        return out

    first = np.flatnonzero(step(step(dot))) + base
    return first[first >= lo].tolist()


def _probe(data: bytes, lo: int, chains: list[int]) -> re.Match | None:
    """Anchored match at the 1-3 starts before each candidate's first dot."""
    for p in chains:
        for s in range(max(lo, p - _OCT_DIGITS), p):
            m = _QUAD.match(data, s)
            if m is not None:
                return m
    return None


def find_quad(data: bytes, lo: int = 0) -> re.Match | None:
    """``_QUAD.search(data, lo)``: regex head, then a numpy prefilter.

    sre costs ~50 ns/byte here; the prefilter leaves a few anchored matches.
    """
    if len(data) - lo < QUAD_NP_MIN_LEN:
        return _QUAD.search(data, lo)

    head = lo + QUAD_HEAD

    m = _QUAD.search(data, lo, head)
    if m is not None and m.end() + _CUT_MARGIN <= head:
        return m

    chains = _chains(data, lo)
    if len(chains) <= MAX_QUAD_PROBES:
        return _probe(data, lo, chains)
    return _QUAD.search(data, max(lo, chains[0] - _OCT_DIGITS))


def has_quad(data: bytes) -> bool:
    """Availability: *data* holds a canonical dotted quad."""
    return find_quad(data) is not None


def split_addr(addr: int, parts: int) -> list[int]:
    """*addr* as *parts* inet_aton parts: leading octets, then the tail."""
    tail_bits = ADDR_BITS - OCTET_BITS * (parts - 1)
    head = [addr >> (ADDR_BITS - OCTET_BITS * (i + 1)) & OCTET for i in range(parts - 1)]
    return [*head, addr & ((1 << tail_bits) - 1)]


def fmt_part(value: int, radix: Radix, pad: int) -> str:
    """Spell one part: ``fmt_part(8, Radix.OCT, 1) == "0010"``.

    Decimal is never padded: a leading zero would make it octal.
    """
    prefix, spec = radix.value
    zeros = "0" * pad if prefix else ""
    return prefix + zeros + format(value, spec)


def _quad_at(data: bytes, pos: int) -> re.Match | None:
    """Quad covering or following *pos*, else the first one."""
    start = pos
    floor = max(0, pos - QUAD_MAX_LEN)
    while start > floor and data[start - 1] in _QUAD_BYTES:
        start -= 1
    return find_quad(data, start) or find_quad(data)


def _equivalent(addr: int, rng) -> bytes:
    parts = rng.randint(1, MAX_PARTS)
    out = []
    for value in split_addr(addr, parts):
        radix = _RADIXES[rng.randint(0, len(_RADIXES) - 1)]
        out.append(fmt_part(value, radix, rng.randint(0, MAX_PAD)))
    return ".".join(out).encode()


def _skewed(addr: int, rng) -> bytes | None:
    skew = _SKEWS[rng.randint(0, len(_SKEWS) - 1)]
    if skew is Skew.INT_WRAP:
        return str(addr + _ADDR_LIMIT).encode()

    octets = split_addr(addr, MAX_PARTS)
    if skew is Skew.WRAP:
        i = rng.randint(0, MAX_PARTS - 1)
        octets[i] += rng.randint(1, MAX_WRAP) << OCTET_BITS
        return b".".join(b"%d" % v for v in octets)

    # DEC_PAD: "09" / "010" -- invalid or a different value to inet_aton.
    if max(octets) < _OCTAL_BASE:
        return None
    return b".".join(b"0%d" % v if v >= _OCTAL_BASE else b"%d" % v for v in octets)


def inet_forms(data: bytes, byte_idx: int, rng, max_len: int) -> bytes | None:
    """Rewrite the dotted quad near *byte_idx* to another inet_aton spelling.

    Args:
        data: Input buffer.
        byte_idx: Planned mutation site; the quad at/after it is taken.
        rng: ``RandPool`` (or ScriptedRng in tests).
        max_len: Output ceiling.

    Returns:
        New bytes, or None: no quad, unchanged spelling, or too long.
    """
    if not data:
        return None

    m = _quad_at(data, byte_idx % len(data))
    if m is None:
        return None

    lit = m.group()
    addr = _pack([int(p) for p in lit.split(b".")])
    skew = rng.randint(0, SKEW_ONE_IN - 1) == 0
    repl = _skewed(addr, rng) if skew else _equivalent(addr, rng)
    if repl is None or repl == lit:
        return None
    if len(data) - len(lit) + len(repl) > max_len:
        return None
    return data[: m.start()] + repl + data[m.end() :]
