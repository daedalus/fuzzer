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

import logging
import struct
from collections.abc import Callable
from itertools import product

log = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────

_UNPACK_KEYS = {1: "B", 2: "H", 4: "L", 8: "Q"}


def _to_int(val: bytes, signed: bool = False) -> int:
    """Interpret *val* as a little-endian integer."""
    key = _UNPACK_KEYS.get(len(val))
    if key is None:
        raise ValueError(f"cannot unpack {len(val)}-byte value")
    return struct.unpack("<" + (key.lower() if signed else key), val)[0]


def _reverse_if(val: bytes, do_reverse: bool) -> bytes:
    return val[::-1] if do_reverse else val


# ── Encoder base ───────────────────────────────────────────────────────


class Encoder:
    """An encoding strategy that may explain a comparison operands.

    Subclasses override ``is_applicable`` and ``encode``.
    """

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
        return [val[:max(2, idx)]] if idx >= 0 else [val]

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

MAX_MUTATIONS_PER_PAIR = 256


def find_offsets(data: bytes, pattern: bytes) -> list[int]:
    """Find all occurrences of *pattern* in *data* (including overlaps)."""
    if not pattern:
        return []
    offsets = []
    start = 0
    while True:
        start = data.find(pattern, start)
        if start == -1:
            return offsets
        offsets.append(start)
        start += 1


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
    mutations: list[tuple[tuple[int, ...], tuple[bytes, ...], Encoder]] = []
    seen: set[tuple] = set()

    # Skip hash-like comparisons that can't be cracked by I2S substitution
    if is_hash is not None and is_hash(operand_a, operand_b):
        return mutations

    for enc in BUILTIN_ENCODERS:
        if not enc.is_applicable(cmp_size, cmp_type, operand_a, operand_b):
            continue

        # Encode operand_a to get the pattern chunks to search for.
        pattern_chunks = enc.encode(operand_a)
        if not pattern_chunks:
            continue

        # Find offsets for each pattern chunk.
        offset_lists = []
        all_found = True
        for chunk in pattern_chunks:
            offsets = find_offsets(input_data, chunk)
            if not offsets:
                all_found = False
                break
            offset_lists.append(offsets)

        if not all_found:
            continue

        # Generate replacement variants from operand_b THROUGH the same encoder.
        # This is critical — SplitEncoder must produce 2-chunk tuples for both
        # pattern AND replacement.
        repl_variants = _get_encoded_variants(enc, cmp_type, cmp_size, operand_b, hammer)

        # Generate up to MAX_MUTATIONS_PER_PAIR permutations
        count = 0
        for offset_combo in product(*offset_lists):
            if count >= MAX_MUTATIONS_PER_PAIR:
                break
            for repl in repl_variants:
                if tuple(pattern_chunks) != repl:
                    key = (offset_combo, repl)
                    if key not in seen:
                        seen.add(key)
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
    # Generate raw value variants
    raw_variants: list[bytes]
    if cmp_type == "STR":
        raw_variants = [val]
        raw_variants.append(val + b"\x00")
        raw_variants.append(val + b"\n")
        raw_variants.append(b'"' + val + b'"')
        raw_variants.append(b"'" + val + b"'")
    elif cmp_type == "SUB":
        bytes_len = cmp_size // 8
        key = _UNPACK_KEYS.get(bytes_len)
        if key is None:
            raw_variants = [val]
        else:
            base_val = struct.unpack(">" + key, val)[0]
            max_val = (1 << (8 * bytes_len)) - 1
            raw_variants = []
            for i in range(-16, 16):
                raw_variants.append(struct.pack(">" + key, (base_val + i) % (max_val + 1)))
    else:
        bytes_len = cmp_size // 8
        key = _UNPACK_KEYS.get(bytes_len)
        if key is None:
            raw_variants = [val]
        else:
            base_val = struct.unpack(">" + key, val)[0]
            max_val = (1 << (8 * bytes_len)) - 1
            max_offset = 64 if hammer else 1
            raw_variants = [val]
            for i in range(1, max_offset + 1):
                raw_variants.append(struct.pack(">" + key, (base_val + i) % (max_val + 1)))
                raw_variants.append(struct.pack(">" + key, (base_val - i) % (max_val + 1)))

    # Encode each raw variant through the encoder and deduplicate
    seen: set[tuple] = set()
    result: list[tuple[bytes, ...]] = []
    for rv in raw_variants:
        encoded = tuple(enc.encode(rv))
        if encoded not in seen:
            seen.add(encoded)
            result.append(encoded)
    return result


def encoders_summary() -> list[dict]:
    """Return a human-readable list of all registered encoders."""
    return [
        {"name": e.name(), "desc": e.description(), "size": e.size()}
        for e in BUILTIN_ENCODERS
    ]
