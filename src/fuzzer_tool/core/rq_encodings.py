"""Redqueen-style encoding-based input-to-state transform engine.

Port of the encoding strategies from Redqueen's encoding.py (NDSS 2019).
Each encoding strategy represents a way the target program might compare
input-derived data against a constant or computed value — e.g. after
sign-extension, zero-extension, ASCII decimal conversion, etc.

Usage:
    from fuzzer_tool.core.rq_encodings import generate_mutations
    mutations = generate_mutations(op_a, op_b, cmp_size, cmp_type, input_data)
    for (offset_tuple, repl_tuple, encoder) in mutations:
        # apply replacement at offsets
"""

import base64
import functools
import logging
import struct
import zlib
from collections.abc import Callable
from itertools import product

from fuzzer_tool.core.gf2_common import apply_bitmask_map, invert_bitmask_map
from fuzzer_tool.core.lru import LRUCache
from fuzzer_tool.core.mutations.generic import encode_sleb128, encode_uleb128

log = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────

# Local bindings for hot-path micro-optimization
_int_from_bytes = int.from_bytes
_bytes_ljust = bytes.ljust
_struct_unpack = struct.unpack
_struct_pack = struct.pack

_UNPACK_KEYS = {1: "B", 2: "H", 4: "L", 8: "Q"}


def _to_int(val: bytes, signed: bool = False) -> int:
    """Interpret *val* as a little-endian integer.

    Handles arbitrary-length operands from cmplog trace data (e.g. a 7-byte
    memcmp(foo, bar, 7)) by zero-extending (or sign-extending) to the nearest
    supported width.
    """
    if not val:
        return 0
    return _int_from_bytes(val, "little", signed=signed)


def _reverse_if(val: bytes, do_reverse: bool) -> bytes:
    return val[::-1] if do_reverse else val


# ── Encoder base ───────────────────────────────────────────────────────


class Encoder:
    """An encoding strategy that may explain a comparison operands.

    Subclasses override ``is_applicable`` and ``encode``.
    """

    #: Encode only the operand itself, never its +-i neighbours (for encoders
    #: where a nearby value maps to an unrelated one, or inversion is costly).
    value_only = False

    def is_applicable(  # noqa: ARG002
        self, cmp_size: int, cmp_type: str, lhs: bytes, rhs: bytes
    ) -> bool:
        """Return True if this encoding could apply to *lhs* /*rhs*."""
        return True

    def encode(self, val: bytes) -> list[bytes]:
        """Return one or more encoded byte sequences from *val*.

        Most encodings return a single value; ``SplitEncoding`` and
        multi-byte variants return multiple discontiguous chunks.
        """
        return [val]

    def pattern(self, val: bytes) -> list[bytes]:
        """Chunks to search for; ``encode`` unless trailing output is ambiguous."""
        return self.encode(val)

    def size(self) -> int:
        """Return the number of discontiguous chunks produced by ``encode``."""
        return 1

    def name(self) -> str:
        return self.__class__.__name__

    def description(self) -> str:
        return self.__doc__ or ""


# ── Concrete encoders ──────────────────────────────────────────────────


class PlainEncoder(Encoder):
    """Direct value substitution — the operands are compared as-is."""

    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_type != "STR"

    def encode(self, val):
        return [_reverse_if(val, self.reverse)]

    def name(self):
        return f"plain_{'r' if self.reverse else 'p'}"


class ZextEncoder(Encoder):
    """Zero extension — the upper bytes of the operands are zero.

    E.g. a 32-bit comparison where the upper 24 bits are zero means the
    value was zero-extended from 8 bits.
    """

    def __init__(self, keep_bytes: int, reverse: bool = False):
        self.keep_bytes = keep_bytes
        self.reverse = reverse

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):
        if cmp_type == "STR":
            return False
        L = cmp_size // 8  # total bytes
        if self.keep_bytes >= L:
            return False
        for v in (lhs, rhs):
            vv = _reverse_if(v, self.reverse)
            if vv[: L - self.keep_bytes] != b"\x00" * (L - self.keep_bytes):
                return False
        return True

    def encode(self, val):
        vv = _reverse_if(val, self.reverse)
        return [vv[-self.keep_bytes :]]

    def name(self):
        return f"zext_{'r' if self.reverse else 'p'}_{self.keep_bytes}"


class SextEncoder(Encoder):
    """Sign extension — the upper bytes are all 0x00 or 0xFF.

    E.g. a 32-bit comparison where the upper 24 bits are all 0xFF means
    the value was sign-extended from 8 bits (negative).
    """

    def __init__(self, keep_bytes: int, reverse: bool = False):
        self.keep_bytes = keep_bytes
        self.reverse = reverse

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):
        if cmp_type == "STR":
            return False
        L = cmp_size // 8
        if self.keep_bytes >= L:
            return False
        for v in (lhs, rhs):
            vv = _reverse_if(v, self.reverse)
            head = vv[: L - self.keep_bytes]
            if head == b"\x00" * len(head):
                continue
            if head == b"\xff" * len(head) and (vv[L - self.keep_bytes] & 0x80):
                continue
            return False
        return True

    def encode(self, val):
        vv = _reverse_if(val, self.reverse)
        return [vv[-self.keep_bytes :]]

    def name(self):
        return f"sext_{'r' if self.reverse else 'p'}_{self.keep_bytes}"


class AsciiEncoder(Encoder):
    """ASCII number representation — the value is compared as a text numeral.

    E.g. ``0x41 0x42`` (65 -> "65") compared against ``0x36 0x35`` ("65").
    """

    def __init__(self, base: int = 10, signed: bool = False):
        self.base = base
        self.signed = signed

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_type != "STR"

    def encode(self, val):
        intval = _to_int(val, self.signed)
        if self.base == 16:
            return [f"{intval:x}".encode()]
        if self.base == 8:
            return [f"{intval:o}".encode()]
        return [f"{intval:d}".encode()]

    def name(self):
        return f"ascii_{'s' if self.signed else 'u'}_{self.base}"


class CStringEncoder(Encoder):
    """Null-terminated string — the comparison is on non-null string content."""

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        if cmp_type != "STR":
            return False
        if len(lhs) < 2 or len(rhs) < 2:
            return False
        return lhs[0:1] != b"\x00" and rhs[0:1] != b"\x00"

    def encode(self, val):
        idx = val.find(b"\x00")
        return [val[: max(2, idx)]] if idx >= 0 else [val]

    def name(self):
        return "cstr"


class CStrChrEncoder(Encoder):
    """Single-character comparison — like strchr() return value.

    The RHS is a null-terminated single character, the LHS is the full
    string.  The target did something like ``strchr(input, c) != NULL``.
    """

    def __init__(self, skip: int = 0):
        self.skip = skip

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        if cmp_type != "STR":
            return False
        if len(lhs) <= self.skip or len(rhs) < 2:
            return False
        if rhs[0:1] == b"\x00":
            return False
        return rhs[1:] == b"\x00" * (len(rhs) - 1)

    def encode(self, val):
        return [val[self.skip : self.skip + 1]]

    def name(self):
        return f"cstrchr_{self.skip}"


class MemEncoder(Encoder):
    """Fixed-length memory comparison — like memcmp() with a constant length."""

    def __init__(self, length: int):
        self.length = length

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_type == "STR" and len(lhs) >= self.length

    def encode(self, val):
        return [val[: self.length]]

    def name(self):
        return f"mem_{self.length}"


class SplitEncoder(Encoder):
    """64-bit split into two 32-bit halves — for double-word comparisons.

    Used when the target splits a 64-bit comparison into two 32-bit
    compare instructions (common on 32-bit architectures or certain
    compiler codegen).
    """

    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_size == 64

    def encode(self, val):
        vv = _reverse_if(val, self.reverse)
        return [vv[:4], vv[4:8]]

    def size(self):
        return 2

    def name(self):
        return f"split_{'r' if self.reverse else 'p'}"


# ── Decoder-layer encoders ─────────────────────────────────────────────
#
# The target decodes the input before comparing: cmplog sees the decoded
# operand, the input holds its encoded form. Encoding operand_a finds it,
# encoding operand_b writes the solved form back. E.g. base64:
#
#   input   "aGVsbG8gd29ybGQh"  --decode-->  "hello world!"  ==  "HELLO_WORLD"
#   search   b64("hello world")              cmplog operands
#   write    b64("HELLO_WORLD")

_B64_MIN = 3  # one full quantum; shorter patterns match by accident
_HEX_MIN = 2
_UTF16_MIN = 4  # two code units


class Base64Encoder(Encoder):
    """Base64 text decoded before a string compare."""

    def __init__(self, urlsafe: bool = False):
        self.urlsafe = urlsafe
        self._b64 = base64.urlsafe_b64encode if urlsafe else base64.b64encode

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_type == "STR" and len(lhs) >= _B64_MIN

    def encode(self, val):
        return [self._b64(val).rstrip(b"=")]

    def pattern(self, val):
        # Only chars fully set by *val*: the last partial one carries bits
        # of the next input byte. 11 bytes = 88 bits -> 14 full chars.
        return [self.encode(val)[0][: len(val) * 8 // 6]]

    def name(self):
        return f"b64_{'url' if self.urlsafe else 'std'}"


class HexEncoder(Encoder):
    """Hex string decoded before a memory compare."""

    def __init__(self, upper: bool = False):
        self.upper = upper

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_type == "STR" and len(lhs) >= _HEX_MIN

    def encode(self, val):
        h = val.hex().encode()
        return [h.upper() if self.upper else h]

    def name(self):
        return f"hex_{'u' if self.upper else 'l'}"


class Utf16Encoder(Encoder):
    """UTF-16 input narrowed to bytes before a string compare."""

    def __init__(self, big_endian: bool = False):
        self.codec = "utf-16-be" if big_endian else "utf-16-le"

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_type == "STR"

    def encode(self, val):
        return [val.decode("latin-1").encode(self.codec)]

    def name(self):
        return f"utf16_{self.codec[-2:]}"


def _is_wide_ascii(val: bytes) -> bool:
    """UTF-16LE of non-NUL Latin-1: b"a\\0b\\0"."""
    if len(val) < _UTF16_MIN or len(val) % 2:
        return False
    return not any(val[1::2]) and all(val[0::2])


class Utf16NarrowEncoder(Encoder):
    """Narrow input widened to UTF-16LE before a wide compare (wcscmp)."""

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_type == "STR" and _is_wide_ascii(lhs) and _is_wide_ascii(rhs)

    def encode(self, val):
        return [val[0::2]]

    def name(self):
        return "utf16_narrow"


class CaseEncoder(Encoder):
    """Input case-folded (tolower/toupper) before a string compare."""

    def __init__(self, upper: bool = False):
        self.upper = upper

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        return cmp_type == "STR" and self.encode(lhs)[0] != lhs

    def encode(self, val):
        return [val.upper() if self.upper else val.lower()]

    def name(self):
        return f"case_{'u' if self.upper else 'l'}"


class Leb128Encoder(Encoder):
    """LEB128 varint decoded before an integer compare (DWARF, wasm, dex)."""

    def __init__(self, signed: bool = False):
        self.signed = signed

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        if cmp_type == "STR":
            return False

        # One-byte unsigned varints are plain bytes, already zext_1's job.
        value = _to_int(lhs, self.signed)
        return value < 0 if self.signed else value > 0x7F

    def encode(self, val):
        if self.signed:
            return [encode_sleb128(_to_int(val, signed=True))]
        return [encode_uleb128(_to_int(val))]

    def name(self):
        return "sleb128" if self.signed else "uleb128"


# CRC-32 over a 4-byte field is affine over GF(2): crc(x) = L·x ^ crc(0).
# L is invertible, so every 32-bit compare operand has exactly one preimage.
_CRC_FIELD_BYTES = 4
_CRC_BITS = 8 * _CRC_FIELD_BYTES
_CRC_ZERO = zlib.crc32(bytes(_CRC_FIELD_BYTES))

# A CRC output fits 24 bits with p=1/256; small-int compares (len == 12) always do.
# Skipping them keeps the two extra input scans off the common pair.
_CRC_MIN_OPERAND = 1 << 24


def _crc_inverse_rows() -> list[int]:
    """Rows of L^-1; row j selects the crc bits that make input bit j."""
    cols = [
        zlib.crc32((1 << i).to_bytes(_CRC_FIELD_BYTES, "little")) ^ _CRC_ZERO
        for i in range(_CRC_BITS)
    ]
    rows = [sum(((cols[i] >> j) & 1) << i for i in range(_CRC_BITS)) for j in range(_CRC_BITS)]
    inv = invert_bitmask_map(rows, _CRC_BITS)
    assert inv is not None, "CRC-32 4-byte map is a bijection"
    return inv


_CRC_INV_ROWS = _crc_inverse_rows()


class Crc32Encoder(Encoder):
    """CRC-32 of a 4-byte field compared to a constant (Fuzzification AntiHybrid).

    ``if (crc32(value) == OUTPUT_CRC)``: the operand is crc(x), not x, so the
    pattern searched for is the preimage x and the replacement is the
    preimage of the constant.
    """

    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        if cmp_type == "STR" or cmp_size != _CRC_BITS:
            return False
        return _to_int(lhs) >= _CRC_MIN_OPERAND and _to_int(rhs) >= _CRC_MIN_OPERAND

    def encode(self, val):
        x = apply_bitmask_map(_CRC_INV_ROWS, _to_int(val) ^ _CRC_ZERO)
        return [_reverse_if(x.to_bytes(_CRC_FIELD_BYTES, "little"), self.reverse)]

    def name(self):
        return f"crc32_{'r' if self.reverse else 'p'}"


# FNV-1a mixes XOR with multiplication, so there is no GF(2) matrix inverse
# the way CRC-32 has.  Meet-in-the-middle recovers the 4-byte preimages in
# ~2^16 steps per side (~12 ms in pure Python); results are cached per hash.
# 4 bytes -> 32 bits is NOT a bijection: ~1 preimage on average, but about a
# third of targets have none and the rest have 1-5.
_FNV_FIELD_BYTES = 4
_FNV_BITS = 8 * _FNV_FIELD_BYTES
_FNV_OFFSET = 2166136261
_FNV_PRIME = 16777619
# Same small-int filter as CRC: skip compares whose operands fit in 24 bits.
_FNV_MIN_OPERAND = 1 << 24
# Modular inverse of FNV_PRIME mod 2^32 (odd => invertible).
_FNV_PRIME_INV = pow(_FNV_PRIME, -1, 1 << 32)

# Per-hash preimage cache (hash_int -> list of 4-byte little-endian preimages).
# Bounded independently of the mutations LRU so encode() stays O(1) after
# the first inversion of a given operand.
_FNV_INV_CACHE_MAX = 4096
_fnv_inv_cache: LRUCache = LRUCache(_FNV_INV_CACHE_MAX)


def _fnv1a(data: bytes) -> int:
    """Independent reference FNV-1a (32-bit), matching antifuzz_demo.c."""
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


@functools.cache
def _fnv1a_fwd_table() -> dict[int, int]:
    """Map hash after 2 bytes -> those two bytes (packed LE); injective, no collisions."""
    fwd: dict[int, int] = {}
    for ab in range(1 << 16):
        h = _FNV_OFFSET
        h = ((h ^ (ab & 0xFF)) * _FNV_PRIME) & 0xFFFFFFFF
        h = ((h ^ ((ab >> 8) & 0xFF)) * _FNV_PRIME) & 0xFFFFFFFF
        fwd[h] = ab
    return fwd


def _fnv1a_invert_all(target: int) -> list[bytes]:
    """Recover all 4-byte little-endian preimages of *target* under FNV-1a.

    Not a bijection: ~1 preimage on average, none for about a third of
    targets.  Meet-in-the-middle enumerates them all in ~2^16 steps per side.
    Results are cached per hash.
    """
    cached = _fnv_inv_cache.get(target)
    if cached is not None:
        return cached

    fwd = _fnv1a_fwd_table()

    out: list[bytes] = []
    for cd in range(1 << 16):
        h = target
        b3 = (cd >> 8) & 0xFF
        h = ((h * _FNV_PRIME_INV) & 0xFFFFFFFF) ^ b3
        b2 = cd & 0xFF
        h = ((h * _FNV_PRIME_INV) & 0xFFFFFFFF) ^ b2
        ab = fwd.get(h)
        if ab is not None:
            out.append(bytes((ab & 0xFF, (ab >> 8) & 0xFF, b2, b3)))

    # Not surjective: some 32-bit values have no 4-byte preimage.
    _fnv_inv_cache[target] = out
    return out


class Fnv1aEncoder(Encoder):
    """FNV-1a of a 4-byte field compared to a constant (AntiFuzz §4.4).

    ``if (fnv1a(value) == OUTPUT_HASH)``: the operand is the hash, not the
    field, so the pattern searched for is the preimage and the replacement
    is the preimage of the constant (the constant only: its +-i neighbours
    hash to unrelated values).  Defeats the hashed magic check in
    ``targets/antifuzz_demo.c`` (``fnv1a(input[0:4]) == fnv1a("crsh")``).
    """

    value_only = True

    def __init__(self, reverse: bool = False):
        self.reverse = reverse

    def is_applicable(self, cmp_size, cmp_type, lhs, rhs):  # noqa: ARG002
        if cmp_type == "STR" or cmp_size != _FNV_BITS:
            return False
        return _to_int(lhs) >= _FNV_MIN_OPERAND and _to_int(rhs) >= _FNV_MIN_OPERAND

    def encode(self, val):
        # All preimages: engine expands size()==1 multi-result as alternatives.
        return [_reverse_if(p, self.reverse) for p in _fnv1a_invert_all(_to_int(val))]

    def name(self):
        return f"fnv1a_{'r' if self.reverse else 'p'}"


# ── Engine ─────────────────────────────────────────────────────────────


# All built-in encoders.
# Mirrors the Encoders list in Redqueen encoding.py lines 235-242.
BUILTIN_ENCODERS: list[Encoder] = []

for bytes_ in (1, 2, 4):
    for rev in (False, True):
        BUILTIN_ENCODERS.append(ZextEncoder(bytes_, rev))
        BUILTIN_ENCODERS.append(SextEncoder(bytes_, rev))

for base_ in (8, 10, 16):
    for sign in (False, True):
        BUILTIN_ENCODERS.append(AsciiEncoder(base_, sign))

for rev in (False, True):
    BUILTIN_ENCODERS.append(PlainEncoder(rev))
    BUILTIN_ENCODERS.append(SplitEncoder(rev))

BUILTIN_ENCODERS.append(CStringEncoder())

for length in range(4, 16):
    BUILTIN_ENCODERS.append(MemEncoder(length))

for length in range(0, 4):
    BUILTIN_ENCODERS.append(CStrChrEncoder(length))

# Decoder-layer encoders (not in Redqueen).
for flag in (False, True):
    BUILTIN_ENCODERS.append(Base64Encoder(flag))
    BUILTIN_ENCODERS.append(HexEncoder(flag))
    BUILTIN_ENCODERS.append(Utf16Encoder(flag))
    BUILTIN_ENCODERS.append(CaseEncoder(flag))
    BUILTIN_ENCODERS.append(Leb128Encoder(flag))
BUILTIN_ENCODERS.append(Utf16NarrowEncoder())
for rev in (False, True):
    BUILTIN_ENCODERS.append(Crc32Encoder(rev))
for rev in (False, True):
    BUILTIN_ENCODERS.append(Fnv1aEncoder(rev))

MAX_MUTATIONS_PER_PAIR = 256

# Cache of input-independent encoder results for generate_mutations().
# Key: (cmp_size, cmp_type, operand_a, operand_b, hammer).
# Value: {encoder: (pattern_alts, repl_variants | None)} — only the
# applicable encoders appear, and replacement variants are filled in
# lazily on the first call that actually finds the pattern in the input.
# LRU-bounded so stale cmplog pairs age out while hot ones stay.
_RQ_MUTATIONS_CACHE_MAX = 20000
_rq_mutations_cache: LRUCache = LRUCache(_RQ_MUTATIONS_CACHE_MAX)


def find_offsets(data: bytes, pattern: bytes) -> list[int]:
    """Find all occurrences of *pattern* in *data* (including overlaps)."""
    if not pattern:
        return []
    offsets = []
    _find = data.find
    start = 0
    while True:
        start = _find(pattern, start)
        if start == -1:
            return offsets
        offsets.append(start)
        start += 1


def _applicable_encoders(cmp_size: int, cmp_type: str, operand_a: bytes, operand_b: bytes) -> dict:
    """First touch of a pair: {encoder: (pattern chunks, None)} for applicable encoders."""
    # Evaluate every encoder once. Only the applicable
    # ones are kept — the common case iterates ~10 encoders instead
    # of all 39, with no per-encoder dict lookups on later calls.
    enc_cache = {}
    for enc in BUILTIN_ENCODERS:
        if not enc.is_applicable(cmp_size, cmp_type, operand_a, operand_b):
            continue
        # Encode operand_a to get the pattern chunks to search for.
        # Replacement variants are computed lazily on the first hit:
        # most pairs' patterns never appear in the input, and encoding
        # up to 129 variants is the most expensive step.
        encoded = enc.pattern(operand_a)
        # size()==1 with multiple encode results = alternative single-chunk
        # patterns (FNV-1a has 0-5 preimages).  Multi-chunk encoders
        # (SplitEncoder) keep size()>1 and a single concurrent pattern.
        if enc.size() == 1 and len(encoded) != 1:
            pattern_alts = [tuple([c]) for c in encoded if c]
        else:
            pattern_alts = [tuple(encoded)] if encoded and all(encoded) else []
        enc_cache[enc] = (pattern_alts, None)
    return enc_cache


def generate_mutations(
    operand_a: bytes,
    operand_b: bytes,
    cmp_size: int,
    cmp_type: str,
    input_data: bytes,
    *,
    hammer: bool = False,
    is_hash: Callable | None = None,
) -> list[tuple[tuple[int, ...], tuple[bytes, ...], Encoder]]:
    """Generate I2S mutations for a single cmplog pair.

    For each applicable encoder, finds occurrences of the encoded form of
    *operand_a* in *input_data*, then generates replacement variants from
    the encoded form of *operand_b*.

    The encoder applicability checks, encoded pattern chunks, and encoded
    replacement variants depend only on the pair — not on *input_data* —
    so they are cached per pair (one lookup per call). Only the offset
    search and permutation loop run per call.

    Args:
        operand_a: The first operand captured from the CMP instruction.
        operand_b: The second operand (the value we want to replace with).
        cmp_size: Comparison width in bits (8, 16, 32, 64, or 512 for strings).
        cmp_type: ``"CMP"``, ``"SUB"``, ``"STR"``, or ``"LEA"``.
        input_data: The current fuzz input for offset search.
        hammer: If True, generate more aggressive +/- offsets (for LEA/SUB).

    Returns:
        List of ``(offset_tuple, replacement_tuple, encoder)`` tuples.
        Each tuple can be applied to the input data.
    """
    # Local bindings for hot path
    _find_offsets = find_offsets
    _get_encoded = _get_encoded_variants
    _product = product
    _cache = _rq_mutations_cache
    MAX = MAX_MUTATIONS_PER_PAIR

    # Pre-allocate mutations list — capped at MAX per encoder × encoder count
    mutations: list[tuple[tuple[int, ...], tuple[bytes, ...], Encoder]] = []
    seen: set[tuple] = set()

    # Skip hash-like comparisons that can't be cracked by I2S substitution
    if is_hash is not None and is_hash(operand_a, operand_b):
        return mutations

    pair_key = (cmp_size, cmp_type, operand_a, operand_b, hammer)
    enc_cache = _cache.get(pair_key)
    if enc_cache is None:
        enc_cache = _applicable_encoders(cmp_size, cmp_type, operand_a, operand_b)
        _cache[pair_key] = enc_cache

    for enc, (pattern_alts, repl_variants) in enc_cache.items():
        count = 0
        for pattern_chunks in pattern_alts:
            if count >= MAX:
                break
            if not pattern_chunks:
                continue

            # Find offsets for each pattern chunk.
            offset_lists = []
            all_found = True
            for chunk in pattern_chunks:
                offsets = _find_offsets(input_data, chunk)
                if not offsets:
                    all_found = False
                    break
                offset_lists.append(offsets)

            if not all_found:
                continue

            if repl_variants is None:
                # Generate replacement variants from operand_b THROUGH the same
                # encoder, only now that the pattern was actually found.
                repl_variants = tuple(_get_encoded(enc, cmp_type, cmp_size, operand_b, hammer))
                enc_cache[enc] = (pattern_alts, repl_variants)

            pattern_key = pattern_chunks

            # Generate up to MAX permutations
            for offset_combo in _product(*offset_lists):
                if count >= MAX:
                    break
                for repl in repl_variants:
                    if pattern_key != repl:
                        k = (offset_combo, repl)
                        if k not in seen:
                            seen.add(k)
                            mutations.append((offset_combo, repl, enc))
                            count += 1

    return mutations


def _get_encoded_variants(
    enc: Encoder, cmp_type: str, cmp_size: int, val: bytes, hammer: bool
) -> list[tuple[bytes, ...]]:
    """Generate replacement variants, encoding *val* through *enc*.

    Produces raw value variants first, then encodes each one through *enc*.
    This ensures multi-chunk encoders (like SplitEncoder) produce the same
    number of chunks for both pattern and replacement sides.
    """
    # Local bindings for hot path
    _enc_encode = enc.encode

    # Generate raw value variants into a pre-allocated list
    raw_variants: list[bytes]
    if enc.value_only:
        raw_variants = [val]
    elif cmp_type == "STR":
        raw_variants = [val, val + b"\x00", val + b"\n", b'"' + val + b'"', b"'" + val + b"'"]
    elif cmp_type == "SUB":
        bytes_len = cmp_size // 8
        key = _UNPACK_KEYS.get(bytes_len)
        if key is None:
            raw_variants = [val]
        else:
            padded = val.rjust(bytes_len, b"\x00")
            base_val = _struct_unpack(">" + key, padded)[0]
            max_val = (1 << (8 * bytes_len)) - 1
            # Pre-allocate array for 32 variants (-16 to 15)
            raw_variants = [None] * 32  # type: ignore[list-item]
            for i in range(32):
                raw_variants[i] = _struct_pack(">" + key, (base_val - 16 + i) % (max_val + 1))  # type: ignore[assignment]
    else:
        bytes_len = cmp_size // 8
        key = _UNPACK_KEYS.get(bytes_len)
        if key is None:
            raw_variants = [val]
        else:
            padded = val.rjust(bytes_len, b"\x00")
            base_val = _struct_unpack(">" + key, padded)[0]
            max_val = (1 << (8 * bytes_len)) - 1
            max_offset = 64 if hammer else 1
            # Pre-allocate for val + 2 per offset
            n_variants = 1 + 2 * max_offset
            raw_variants = [None] * n_variants  # type: ignore[list-item]
            raw_variants[0] = val  # type: ignore[assignment]
            for i in range(1, max_offset + 1):
                idx = 1 + 2 * (i - 1)
                raw_variants[idx] = _struct_pack(">" + key, (base_val + i) % (max_val + 1))  # type: ignore[assignment]
                raw_variants[idx + 1] = _struct_pack(">" + key, (base_val - i) % (max_val + 1))  # type: ignore[assignment]

    # Encode each raw variant through the encoder and deduplicate.
    # size()==1 with multiple encode results = alternative single-chunk
    # replacements (FNV-1a preimages), not concurrent multi-chunk.
    seen: set[tuple] = set()
    result: list[tuple[bytes, ...]] = []
    multi_alt = enc.size() == 1
    for rv in raw_variants:
        parts = _enc_encode(rv)
        if multi_alt and len(parts) != 1:
            for p in parts:
                if not p:
                    continue
                encoded = (p,)
                if encoded not in seen:
                    seen.add(encoded)
                    result.append(encoded)
        else:
            if not parts or any(not p for p in parts):
                continue
            encoded = tuple(parts)
            if encoded not in seen:
                seen.add(encoded)
                result.append(encoded)
    return result


def encoders_summary() -> list[dict]:
    """Return a human-readable list of all registered encoders."""
    return [{"name": e.name(), "desc": e.description(), "size": e.size()} for e in BUILTIN_ENCODERS]
