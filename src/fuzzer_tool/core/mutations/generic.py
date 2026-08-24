"""Mutation operators and dictionary handling."""

import random


# Helper: resolve rng parameter to RandPool or stdlib random
def _get_rng(rng=None):
    return rng or random


INTERESTING_8 = [
    -128,  # Overflow signed 8-bit when decremented
    -1,
    0,
    1,
    2,  # Power-of-2 ± 1 (from Radamsa)
    3,  # Power-of-2 ± 1 (from Radamsa)
    16,  # One-off with common buffer size
    32,  # One-off with common buffer size
    64,  # One-off with common buffer size
    100,  # One-off with common buffer size
    127,  # Overflow signed 8-bit when incremented
    129,  # Power-of-2 ± 1 (from Radamsa)
]

INTERESTING_16 = [
    -32768,  # Overflow signed 16-bit when decremented
    -129,  # Overflow signed 8-bit
    128,  # Overflow signed 8-bit
    255,  # Overflow unsigned 8-bit when incremented
    256,  # Overflow unsigned 8-bit
    257,  # Power-of-2 ± 1 (from Radamsa)
    512,  # One-off with common buffer size
    1000,  # One-off with common buffer size
    1024,  # One-off with common buffer size
    4096,  # One-off with common buffer size
    32767,  # Overflow signed 16-bit when incremented
    32769,  # Power-of-2 ± 1 (from Radamsa)
]

INTERESTING_32 = [
    -2147483648,  # Overflow signed 32-bit when decremented
    -100663046,  # Large negative number (endian-agnostic)
    -32769,  # Overflow signed 16-bit
    32768,  # Overflow signed 16-bit
    32769,  # Power-of-2 ± 1 (from Radamsa)
    65535,  # Overflow unsigned 16-bit when incremented
    65536,  # Overflow unsigned 16-bit
    65537,  # Power-of-2 ± 1 (from Radamsa)
    100663045,  # Large positive number (endian-agnostic)
    2139095040,  # Float infinity
    2147483647,  # Overflow signed 32-bit when incremented
    2147483649,  # Power-of-2 ± 1 (from Radamsa)
]

INTERESTING_UNSIGNED_8 = [
    0,
    1,
    2,
    3,
    4,
    5,  # Small values — trigger len < nlen underflows
    0xFE,
    0xFF,  # Near unsigned 8-bit max
]

INTERESTING_UNSIGNED_16 = [
    0,
    1,
    2,
    3,
    4,
    5,
    0xFFFE,
    0xFFFF,  # Unsigned 16-bit max
    0x7FFE,
    0x7FFF,  # Near signed 16-bit max
]

INTERESTING_UNSIGNED_32 = [
    0,
    1,
    2,
    3,
    4,
    5,
    0xFFFFFFFE,
    0xFFFFFFFF,  # Unsigned 32-bit max (SIZE_MAX on 32-bit)
    0x7FFFFFFE,
    0x7FFFFFFF,  # Near signed 32-bit max
    0x100,
    0x400,
    0x1000,  # Common buffer boundaries
]

LENGTH_BOUNDARIES = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    31,
    32,
    63,
    64,
    127,
    128,
    255,
    256,
    512,
    1024,
    4096,
]

# Data lengths near AVX2/SSE2 SIMD boundaries — exercises _mm256_loadu_si256
# overread guards and scalar fallback paths
SIMD_BOUNDARIES = [15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65]

# Regex backtracking bomb patterns — stress regcomp/regexec
REGEX_BOMBS = [
    "(a+)+",
    "(?:a|b?)*",
    "(?:x{1,}){1,}",
    "^(a+)+$",
    "((a){1,}){1,}",
    "(a|ab)+",
    "([a-zA-Z]+)*$",
    "(?:a{2,})+",
    "(a?){1,}a{1,}",
    "(?:xx|x)+",
]

ARITHMETIC_DELTAS = [1, 2, 4, 8, 16, 32, 64, 128]

ARITH_MAX = 35

# ---------------------------------------------------------------------------
# Length selection and corpus-literal helpers (ported from go-fuzz)
# ---------------------------------------------------------------------------


def choose_len(n: int, rng=None) -> int:
    """Choose a range length, preferring short ranges.

    Mirrors go-fuzz's chooseLen: 90% chance of 1-8, 9% of 1-32, 1% any.
    """
    rng = _get_rng(rng)
    x = rng.randint(0, 99)
    if x < 90:
        return rng.randint(1, min(8, n))
    if x < 99:
        return rng.randint(1, min(32, n))
    return rng.randint(1, n)


def extract_corpus_literals(corpus: list[bytes]) -> tuple[list[bytes], list[bytes]]:
    """Extract integer and string literals from the corpus.

    Returns (int_lits, str_lits) where int_lits are digit sequences of
    length >= 2 (optionally with leading '-') and str_lits are printable
    ASCII runs of length >= 3 that are not pure digits.
    """
    int_lits: list[bytes] = []
    str_lits: list[bytes] = []
    seen_int = set()
    seen_str = set()
    for raw in corpus:
        # Coerce to bytes. The annotation says list[bytes], but the live
        # corpus holds bytearray -- Fuzzer.corpus is bytearray from startup
        # onward -- and slicing a bytearray yields a bytearray, which is
        # unhashable. The `lit not in seen_int` membership test below then
        # raises TypeError on the very first literal found, so this function
        # never returned anything on a real corpus.
        raw = bytes(raw)
        i = 0
        while i < len(raw):
            # Digit runs take priority over everything else.
            if 0x30 <= raw[i] <= 0x39:
                j = i + 1
                while j < len(raw) and 0x30 <= raw[j] <= 0x39:
                    j += 1
                if j - i >= 2:
                    lit = raw[i:j]
                    if lit not in seen_int:
                        seen_int.add(lit)
                        int_lits.append(lit)
                i = j
                continue
            # Integer literal: optional '-', then >=2 digits.
            if (
                raw[i] == 45
                and i + 2 < len(raw)
                and all(0x30 <= b <= 0x39 for b in raw[i + 1 : i + 3])
            ):
                j = i + 1
                while j < len(raw) and 0x30 <= raw[j] <= 0x39:
                    j += 1
                lit = raw[i:j]
                if lit not in seen_int:
                    seen_int.add(lit)
                    int_lits.append(lit)
                i = j
                continue
            # Alpha run: letters and underscore, length >= 3.
            if (0x61 <= raw[i] <= 0x7A) or (0x41 <= raw[i] <= 0x5A) or raw[i] == 0x5F:
                j = i + 1
                while j < len(raw) and (
                    (0x61 <= raw[j] <= 0x7A) or (0x41 <= raw[j] <= 0x5A) or raw[j] == 0x5F
                ):
                    j += 1
                if j - i >= 3:
                    lit = raw[i:j]
                    if lit not in seen_str:
                        seen_str.add(lit)
                        str_lits.append(lit)
                i = j
                continue
            # Symbol run: printable non-alphanum, length >= 3.
            if 0x20 <= raw[i] <= 0x7E and not (
                (0x30 <= raw[i] <= 0x39) or (0x61 <= raw[i] <= 0x7A) or (0x41 <= raw[i] <= 0x5A)
            ):
                j = i + 1
                while (
                    j < len(raw)
                    and 0x20 <= raw[j] <= 0x7E
                    and not (
                        (0x30 <= raw[j] <= 0x39)
                        or (0x61 <= raw[j] <= 0x7A)
                        or (0x41 <= raw[j] <= 0x5A)
                    )
                ):
                    j += 1
                if j - i >= 3:
                    lit = raw[i:j]
                    if lit not in seen_str:
                        seen_str.add(lit)
                        str_lits.append(lit)
                i = j
                continue
            i += 1
    return int_lits, str_lits


def splice_common_prefix(a: bytes, b: bytes, rng=None) -> bytes:
    """Splice a donor into a base, aligned by common prefix/suffix.

    Ported from go-fuzz case 16.  Falls back to returning *a* when the
    differing middle is too small (< 4 bytes).
    """
    if len(a) < 4 or len(b) < 4:
        return a
    rng = _get_rng(rng)
    idx0 = 0
    while idx0 < len(a) and idx0 < len(b) and a[idx0] == b[idx0]:
        idx0 += 1
    idx1 = 0
    while idx1 < len(a) and idx1 < len(b) and a[len(a) - idx1 - 1] == b[len(b) - idx1 - 1]:
        idx1 += 1
    diff = min(len(a) - idx0 - idx1, len(b) - idx0 - idx1)
    if diff < 4:
        return a
    cut = idx0 + rng.randint(0, diff - 2) + 1
    n = rng.randint(0, min(diff - (cut - idx0), len(b) - idx0))
    return a[:cut] + b[idx0 : idx0 + n] + a[cut + n :]


# ---------------------------------------------------------------------------
# Security-sensitive strings (ported from honggfuzz mangle_SpecialStrings)
# ---------------------------------------------------------------------------

SPECIAL_STRINGS: list[bytes] = [
    # Format string attacks
    b"%s",
    b"%n",
    b"%x",
    b"%p",
    b"%9999999s",
    b"%08x",
    # SQL injection
    b"'",
    b'"',
    b"`",
    b"1=1",
    b"--",
    b"/*",
    b"*/",
    b" OR ",
    b" AND ",
    b"UNION SELECT",
    # Path traversal
    b"../",
    b"..\\",
    b"../../../../../../../../etc/passwd",
    b"boot.ini",
    b"/bin/sh",
    # XML/HTML injection
    b"<",
    b">",
    b"<script>",
    b"javascript:",
    b"CDATA",
    b"<!--",
    b"-->",
    # JSON / edge-case literals
    b"null",
    b"true",
    b"false",
    b"NaN",
    b"Infinity",
    b"undefined",
    b"{}",
    b"[]",
    # Command injection
    b"|",
    b";",
    b"`",
    b"$(",
    b"&&",
    b"||",
    # Terminators / control characters
    b"\n",
    b"\r\n",
    b"\x00",
    b"\xff",
]

# ---------------------------------------------------------------------------
# Magic / boundary values table (ported from honggfuzz mangle_Magic)
# Covers 1/2/4/8-byte widths, little-endian and big-endian.
# ---------------------------------------------------------------------------

# 1-byte magic values
_MAGIC_8 = [
    0x00,
    0x01,
    0x02,
    0x03,
    0x04,
    0x05,
    0x06,
    0x07,
    0x08,
    0x09,
    0x0A,
    0x0B,
    0x0C,
    0x0D,
    0x0E,
    0x0F,
    0x10,
    0x20,
    0x40,
    0x7E,
    0x7F,
    0x80,
    0x81,
    0xC0,
    0xFE,
    0xFF,
]

# 2-byte magic values (big-endian); LE variants generated automatically
_MAGIC_16_BE = [
    0x0001,
    0x0002,
    0x0003,
    0x0004,
    0x0005,
    0x0006,
    0x0007,
    0x0008,
    0x0009,
    0x000A,
    0x000B,
    0x000C,
    0x000D,
    0x000E,
    0x000F,
    0x0010,
    0x0020,
    0x0040,
    0x007E,
    0x007F,
    0x0080,
    0x0081,
    0x00C0,
    0x00FE,
    0x00FF,
    0x7EFF,
    0x7F00,
    0x7FFF,
    0x8000,
    0x8001,
    0xFEFE,
    0xFFFF,
]

# 4-byte magic values (big-endian)
_MAGIC_32_BE = [
    0x00000001,
    0x00000010,
    0x00000080,
    0x000000FF,
    0x00007F00,
    0x00007FFF,
    0x00008000,
    0x00008001,
    0x0000FFFE,
    0x0000FFFF,
    0x7FFFFFFE,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
]

# 8-byte magic values (big-endian)
_MAGIC_64_BE = [
    0x0000000000000001,
    0x0000000000000010,
    0x0000000000000080,
    0x00000000000000FF,
    0x000000007FFFFFFF,
    0x0000000080000000,
    0x0000000080000001,
    0x00000000FFFFFFFE,
    0x00000000FFFFFFFF,
    0x7FFFFFFFFFFFFFFE,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0x8000000000000001,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
]


def _build_magic_table() -> list[tuple[int, bytes]]:
    """Build the full magic values table as (width, packed_bytes) pairs.

    Each entry is packed in both LE and BE for random endianness selection.
    Returns a flat list suitable for random.choice().
    """
    table: list[tuple[int, bytes]] = []
    for val in _MAGIC_8:
        table.append((1, bytes([val])))
    for val in _MAGIC_16_BE:
        table.append((2, val.to_bytes(2, "little")))
        table.append((2, val.to_bytes(2, "big")))
    for val in _MAGIC_32_BE:
        table.append((4, val.to_bytes(4, "little")))
        table.append((4, val.to_bytes(4, "big")))
    for val in _MAGIC_64_BE:
        table.append((8, val.to_bytes(8, "little")))
        table.append((8, val.to_bytes(8, "big")))
    return table


MAGIC_TABLE = _build_magic_table()


def ascii_num_arithmetic(data: bytes, rng=None) -> bytes | None:
    """Find an existing ASCII digit sequence and apply arithmetic to it.

    Scans from a random offset for the first digit sequence, parses the
    integer, applies one of: +1, -1, *2, /2, NOT, random replace,
    +random, -random.  Returns None if no digit sequence is found.

    Ported from honggfuzz mangle_ASCIINumChange.
    """
    if not data:
        return None
    r = _get_rng(rng)
    # Find a random digit sequence
    start = r.randint(0, len(data) - 1)
    # Scan forward from start, wrapping around
    idx = start
    for _ in range(len(data)):
        if data[idx : idx + 1].isdigit():
            # Find end of digit sequence
            end = idx + 1
            while end < len(data) and data[end : end + 1].isdigit():
                end += 1
            # Parse the number (limit to 20 digits to avoid overflow)
            seq_len = min(end - idx, 20)
            end = idx + seq_len
            old_val = int(data[idx:end])
            # Apply one of 8 operations
            op = r.randint(0, 7)
            if op == 0:  # +1
                new_val = old_val + 1
            elif op == 1:  # -1
                new_val = max(0, old_val - 1)
            elif op == 2:  # *2
                new_val = old_val * 2
            elif op == 3:  # /2
                new_val = old_val // 2
            elif op == 4:  # NOT (~val)
                new_val = ~old_val
            elif op == 5:  # random replace
                new_val = r.randint(0, 999999999)
            elif op == 6:  # +random
                new_val = old_val + r.randint(1, 256)
            else:  # -random
                new_val = max(0, old_val - r.randint(1, 256))
            new_str = str(new_val).encode("ascii")
            # Replace with proper inflate/deflate
            result = bytearray(data)
            if len(new_str) < seq_len:
                result[idx : idx + seq_len] = new_str + b"\x00" * (seq_len - len(new_str))
            elif len(new_str) > seq_len:
                result[idx:end] = new_str
            else:
                result[idx:end] = new_str
            return bytes(result)
        idx = (idx + 1) % len(data)
    return None


# ---------------------------------------------------------------------------
# Chunk shuffle (ported from honggfuzz mangle_ChunkShuffle)
# ---------------------------------------------------------------------------


def block_shuffle_variable(data: bytes, rng=None) -> bytes:
    """Shuffle variable-width blocks via order-statistics spacings trick.

    Divides the input into k random blocks (2 ≤ k ≤ 5) using cut points
    generated via the normalized-Exponential spacing trick (see
    order_statistics.py Part 3). The cut points have the same joint
    distribution as sorted Uniform(0, len(data)) draws, but without
    needing a sort. Blocks are then randomly permuted.

    Unlike chunk_shuffle (fixed-width chunks), this operator produces
    variable-width blocks that can rearrange structural elements at
    any granularity — useful for formats where field widths vary.

    Args:
        data: Input bytes.

    Returns:
        Bytes with variable-width blocks rearranged.
    """
    if len(data) < 8:
        return data
    r = _get_rng(rng)
    k = r.randint(2, 5)  # number of blocks
    n_cuts = k - 1

    # Spacings trick: n_cuts+1 i.i.d. Exponential(1) draws, normalized,
    # give the same joint law as the gaps between n_cuts sorted
    # Uniform(0,1) points — i.e. Dirichlet(1,...,1) (Part 3 of
    # order_statistics.py). Cumulative sums give sorted cut points
    # without an explicit sort.
    exps = [r.expovariate(1.0) for _ in range(n_cuts + 1)]
    s = sum(exps)
    cum = 0.0
    cuts: list[int] = []
    for i in range(n_cuts):
        cum += exps[i]
        pos = int(len(data) * cum / s)
        pos = max(1, min(pos, len(data) - 1))
        cuts.append(pos)
    cuts = sorted(set(cuts))  # deduplicate in case of integer collisions

    # Build blocks from cut points
    blocks: list[bytes] = []
    prev = 0
    for c in cuts:
        blocks.append(data[prev:c])
        prev = c
    blocks.append(data[prev:])

    # Shuffle all blocks
    r.shuffle(blocks)
    return b"".join(blocks)


def chunk_shuffle(data: bytes, rng=None, stride: int | None = None) -> bytes:
    """Shuffle fixed-size chunks, preserving chunk boundaries.

    Divides the input into chunks of 1-4 bytes (chosen randomly), then
    swaps random chunk pairs.  Useful for binary formats with fixed-width
    fields where byte-level shuffling would break alignment.

    When ``stride`` is given (an inferred record size, e.g. from
    ``estimate_record_size``), chunks are stride-sized instead — this swaps
    whole records, which keeps record-internal field alignment intact while
    reordering records.

    Ported from honggfuzz mangle_ChunkShuffle.
    """
    if len(data) < 8:
        return data
    r = _get_rng(rng)
    chunk_size = (
        stride
        if (stride is not None and stride >= 2 and len(data) // stride >= 2)
        else r.randint(1, 4)
    )
    num_chunks = len(data) // chunk_size
    if num_chunks < 2:
        return data
    result = bytearray(data)
    # Swap random chunk pairs
    n_swaps = r.randint(1, max(1, num_chunks // 2))
    for _ in range(n_swaps):
        i = r.randint(0, num_chunks - 1)
        j = r.randint(0, num_chunks - 1)
        if i != j:
            off_i = i * chunk_size
            off_j = j * chunk_size
            tmp = result[off_i : off_i + chunk_size]
            result[off_i : off_i + chunk_size] = result[off_j : off_j + chunk_size]
            result[off_j : off_j + chunk_size] = tmp
    return bytes(result)


# Compound dictionary separators (from honggfuzz mangle_DictionaryInsert)
DICT_COMPOUND_SEPARATORS: list[bytes] = [
    b"",
    b" ",
    b"\t",
    b"\n",
    b"\r\n",
    b",",
    b";",
    b":",
    b"=",
    b"&",
    b"|",
    b"(",
    b")",
    b".",
    b'"',
    b"'",
]

# Punctuation characters for punctuation_insert (from honggfuzz mangle_Punctuation)
PUNCTUATION_CHARS: list[int] = [
    0x21,
    0x22,
    0x23,
    0x24,
    0x25,
    0x26,
    0x27,
    0x28,  # ! " # $ % & ' (
    0x29,
    0x2A,
    0x2B,
    0x2C,
    0x2D,
    0x2E,
    0x2F,  # ) * + , - . /
    0x3A,
    0x3B,
    0x3C,
    0x3D,
    0x3E,
    0x3F,
    0x40,  # : ; < = > ? @
    0x5B,
    0x5C,
    0x5D,
    0x5E,
    0x5F,
    0x60,  # [ \ ] ^ _ `
    0x7B,
    0x7C,
    0x7D,
    0x7E,  # { | } ~
]

MUTATIONS = [
    "bit_flip",
    "byte_flip",
    "interesting_8",
    "interesting_16",
    "interesting_32",
    "arithmetic",
    "random_bytes",
    "block_insert",
    "block_delete",
    "block_duplicate",
    "splice",
    "havoc",
    "crossover",
    "length_grow",
    "length_shrink",
    "repeat_clone",
    "truncate",
    "length_boundary",
    "swap_regions",
    "swap_bytes",
    "endianness_swap",
    "type_replace",
    "ascii_num",
    "byte_shuffle",
    "byte_delete",
    "byte_insert",
    "insert_ascii_num",
    "transpose_16",
    "transpose_32",
    "transpose_64",
    "bit_transpose_8",
    "bit_transpose_16",
    "bit_transpose_32",
    "bit_transpose_64",
    "bit_offset_flip",
    "bit_offset_span",
    "bit_rotate",
    "bit_shift",
    "span_invert",
    "bit_repack",
    "simd_boundary",
    "regex_bomb",
    "clone_fixed",
    "overwrite_copy",
    "overwrite_fixed",
    "redqueen_xform",
    "fuse_this",
    "fuse_next",
    "fuse_old",
    "tree_mutate",
    "utf8_widen",
    "utf8_insert",
    "line_mutate",
    "skipdet_probe",
    "auto_extras",
    "tlv_mutate",
    "token_shuffle",
    "gradient_cmp",
    "special_strings",
    "magic_values",
    "ascii_num_arithmetic",
    "chunk_shuffle",
    "block_shuffle_variable",
    "dict_compound",
    "punctuation_insert",
    "splice_diff_located",
    "radamsa_num",
]

# Format-aware mutations: structure-aware operators for specific file formats.
# Every scheduler (mc, mopt, replicator, elo) must register all of these.
FORMAT_MUTATIONS = [
    "png_chunk_mutate",
    "png_crc_fix",
    "jpeg_chunk_mutate",
    "jpeg_crc_fix",
    "bmp_chunk_mutate",
    "gzip_chunk_mutate",
    "zlib_chunk_mutate",
    "format_lock",
    "pgs_chunk_mutate",
    "isobmff_chunk_mutate",
    "nal_chunk_mutate",
    "protobuf_chunk_mutate",
    "gif_chunk_mutate",
    "webp_chunk_mutate",
    "webm_chunk_mutate",
    "zip_chunk_mutate",
    "x86_chunk_mutate",
    "arm_chunk_mutate",
]


def splice(a: bytes, b: bytes, rng=None) -> bytes:
    """Cross two inputs at random offsets to produce a structural hybrid.

    Takes the prefix of *a* up to a random cut point, then appends the
    suffix of *b* from a random cut point.  Returns *a* unchanged when
    either input is too short (< 2 bytes) to produce a meaningful splice.

    Args:
        a: First input.
        b: Second input.

    Returns:
        Spliced bytes combining prefix of *a* with suffix of *b*.
    """
    if len(a) < 2 or len(b) < 2:
        return a
    cut_a = _get_rng(rng).randint(1, len(a) - 1)
    cut_b = _get_rng(rng).randint(1, len(b) - 1)
    return a[:cut_a] + b[cut_b:]


def crossover(a: bytes, b: bytes, rng=None) -> bytes:
    """Two-point crossover: exchange a middle segment between two inputs.

    Picks two random cut points in *a* and replaces the segment between
    them with the corresponding segment from *b*.  Returns *a* unchanged
    when either input is too short (< 4 bytes).

    Args:
        a: First input (base).
        b: Second input (donor).

    Returns:
        Hybrid bytes with a middle segment swapped from *b*.
    """
    if len(a) < 4 or len(b) < 4:
        return a
    cut1 = _get_rng(rng).randint(1, len(a) - 3)
    cut2 = _get_rng(rng).randint(cut1 + 1, len(a) - 1)
    seg_len = cut2 - cut1
    b_start = _get_rng(rng).randint(0, max(0, len(b) - seg_len))
    result = bytearray(a)
    result[cut1:cut2] = b[b_start : b_start + seg_len]
    return bytes(result)


DICT_MUTATIONS = [
    "dict_insert",
    "dict_replace",
    "dict_overwrite",
    "dict_prepend",
    "dict_append",
    "checksum_repair",
    "token_dup",
]


def _decode_dict_escapes(token: str) -> bytes:
    """Decode AFL dictionary escapes in *token* (quotes already stripped).

    Single left-to-right pass rather than a regex sweep, because the escapes
    are not independent: ``\\\\x41`` is an escaped backslash followed by the
    literal text ``x41``, but a regex scanning for ``\\x[0-9a-f]{2}`` matches
    at offset 1 and yields a backslash followed by ``A``. Consuming ``\\\\``
    as one unit before looking for ``\\x`` is the only way to get that right.

    Recognised: ``\\xNN`` (hex byte), ``\\\\`` (backslash), ``\\"`` (quote).
    An unrecognised escape keeps its backslash literally, as AFL does --
    dictionaries in the wild carry things like ``\\1a`` that are not escapes
    at all, and dropping the backslash would silently alter the token.
    """
    out = bytearray()
    i = 0
    n = len(token)
    while i < n:
        if token[i] != "\\":
            # Copy the whole run up to the next backslash in one encode().
            j = token.find("\\", i)
            if j == -1:
                j = n
            out.extend(token[i:j].encode("utf-8"))
            i = j
            continue
        if i + 1 >= n:  # trailing lone backslash
            out.append(0x5C)
            break
        nxt = token[i + 1]
        if nxt == "x" and i + 3 < n and _IS_HEX(token[i + 2]) and _IS_HEX(token[i + 3]):
            out.append(int(token[i + 2 : i + 4], 16))
            i += 4
        elif nxt == "\\":
            out.append(0x5C)
            i += 2
        elif nxt == '"':
            out.append(0x22)
            i += 2
        else:
            out.append(0x5C)
            i += 1
    return bytes(out)


def _IS_HEX(c: str) -> bool:
    return c in "0123456789abcdefABCDEF"


def parse_dict_line(line: str) -> bytes | None:
    """Parse a single dictionary line.

    Handles the AFL dictionary format: an optional ``name`` (or ``name@level``)
    followed by ``=``, then the token itself enclosed in double quotes, with
    ``\\xNN`` / ``\\\\`` / ``\\"`` escapes inside it. A bare quoted token with
    no ``name=`` prefix is equally valid and is what most of the dictionaries
    under ``dictionaries/`` actually use.

    The enclosing quotes are DELIMITERS, not content. They were previously
    kept, so every token from a standard AFL dictionary carried a spurious
    0x22 on each end -- ``"IDAT"`` rather than ``IDAT`` -- and therefore
    matched nothing in the target. That affected 12,169 of the 18,311 tokens
    in the shipped dictionaries (66.5%). No test caught it because the
    existing ones only asserted the result was non-None `bytes`, never what
    the bytes were; the tests added alongside this fix assert values.

    The token is located by its quotes rather than by splitting on the first
    ``=``, which mangled any token containing one: ``"a=b"`` split into
    ``"a`` and ``b"`` and returned the latter.

    A line with NO quotes is taken to be a bare token in its entirety, and is
    NOT split on ``=``. AFL only ever puts an unquoted ``=`` between a name
    and its quoted value, so a line without quotes cannot be a name/value
    pair -- and the two shipped dictionaries that use the bare form
    (``ruby.dict``, ``rar.dict``, 5,770 lines between them) confirm it: every
    one is a token, 296 of which contain an ``=``. Splitting them turned
    ``!=`` into an empty token and ``==`` into ``=``.

    Args:
        line: Raw line from dictionary file.

    Returns:
        Parsed token bytes, or None if the line is empty, a comment, or
        carries no token (``NAME=""``) -- a zero-length token cannot
        contribute to a mutation, so it is dropped rather than stored.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    start = line.find('"')
    end = line.rfind('"')
    token = line[start + 1 : end] if (start != -1 and end > start) else line
    return _decode_dict_escapes(token) or None


def load_dictionary(path: str) -> list[bytes]:
    """Load tokens from a dictionary file.

    Args:
        path: Path to dictionary file.

    Returns:
        List of token byte sequences.

    Raises:
        FileNotFoundError: If dictionary file does not exist.
    """
    d = []
    with open(path, errors="replace") as f:
        for line in f:
            tok = parse_dict_line(line)
            if tok is not None:
                d.append(tok)
    return d


def minimize_bytes(data: bytes, interesting_fn, max_stages: int = 128) -> bytes:
    """Delta-debugging style minimizer: binary-search for the smallest input
    that still triggers the same behavior.

    Args:
        data: The original input to minimize.
        interesting_fn: Callable(bytes) -> bool, returns True if input is still interesting.
        max_stages: Maximum number of reduction stages before stopping.

    Returns:
        Minimized input that still triggers the same behavior.
    """
    if not data or not interesting_fn(data):
        return data

    best = bytearray(data)
    stage = 0

    while stage < max_stages and len(best) > 1:
        improved = False

        for chunk_size in _divisor_sizes(len(best)):
            if chunk_size > len(best):
                continue
            offset = 0
            while offset + chunk_size <= len(best):
                candidate = best[:offset] + best[offset + chunk_size :]
                if candidate and interesting_fn(bytes(candidate)):
                    best = candidate
                    improved = True
                    break
                offset += chunk_size
            if improved:
                break

        if not improved:
            break
        stage += 1

    return bytes(best)


def _divisor_sizes(n: int) -> list[int]:
    """Return reduction chunk sizes for delta-debugging, from large to small.

    Uses halving then 1/4, 1/8, ..., then individual bytes.
    """
    sizes = set()
    s = n // 2
    while s >= 1:
        sizes.add(s)
        s //= 2
    sizes.add(1)
    return sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------
# Type-aware mutation (ported from AFL++ redqueen.c type_replace)
# ---------------------------------------------------------------------------


# Character class ranges: (start, end) inclusive
_CHAR_CLASSES = [
    (0x41, 0x46),  # A-F
    (0x61, 0x66),  # a-f
    (0x32, 0x39),  # 2-9
    (0x47, 0x5A),  # G-Z
    (0x67, 0x7A),  # g-z
    (0x21, 0x2A),  # ! to *
    (0x2C, 0x2E),  # , to .
    (0x3A, 0x40),  # : to @
    (0x5B, 0x60),  # [ to `
    (0x7B, 0x7E),  # { to ~
]

# Direct swaps within/between classes
_SWAP_MAP = {
    0x2B: 0x2F,  # + <-> /
    0x2F: 0x2B,
    0x20: 0x09,  # space <-> tab
    0x09: 0x20,
    0x0D: 0x0A,  # CR <-> LF
    0x0A: 0x0D,
    0x00: 0x01,  # NUL <-> SOH
    0x01: 0x00,
    0xFF: 0x00,
}


def _in_class(b: int) -> tuple[int, int] | None:
    """Return the character class range for a byte, or None."""
    for start, end in _CHAR_CLASSES:
        if start <= b <= end:
            return (start, end)
    return None


# Precomputed translate table: byte -> deterministic replacement within same class.
# Avoids per-byte Python branching in type_replace(). Each entry is a fixed
# but different value from the same character class.
_TYPE_REPLACE_TABLE = bytes.maketrans(b"", b"")  # identity as base
_TYPE_REPLACE_TBL = bytearray(256)
for _b in range(256):
    if _b in _SWAP_MAP:
        _TYPE_REPLACE_TBL[_b] = _SWAP_MAP[_b]
    elif 0x30 <= _b <= 0x39:
        _TYPE_REPLACE_TBL[_b] = 0x30 + (_b * 7 + 3) % 10
    elif 0x41 <= _b <= 0x5A:
        _TYPE_REPLACE_TBL[_b] = 0x41 + (_b * 13 + 5) % 26
    elif 0x61 <= _b <= 0x7A:
        _TYPE_REPLACE_TBL[_b] = 0x61 + (_b * 17 + 7) % 26
    elif _b < 32:
        _TYPE_REPLACE_TBL[_b] = _b ^ 0x1F
    else:
        _TYPE_REPLACE_TBL[_b] = _b ^ 0x7F
_TYPE_REPLACE_TABLE = bytes(_TYPE_REPLACE_TBL)


def type_replace_byte(b: int, rng=None) -> int:
    """Replace a byte with a different value from the same character class.

    Preserves the 'type' of the byte: hex digits stay hex, digits stay
    digits, uppercase stays uppercase, etc. This is useful for fuzzing
    text-based formats where structural tokens must remain valid.

    Ported from AFL++ redqueen.c type_replace.

    Args:
        b: Original byte value (0-255).

    Returns:
        A different byte value from the same character class.
    """
    # Direct swaps
    if b in _SWAP_MAP:
        return _SWAP_MAP[b]

    # Character class ranges
    cls = _in_class(b)
    if cls:
        start, end = cls
        size = end - start
        if size == 0:
            # Single-member class (like '0' or '1'), flip to the other
            if b == 0x30:
                return 0x31  # '0' -> '1'
            if b == 0x31:
                return 0x30  # '1' -> '0'
            return b ^ 0x01
        c = b
        while c == b:
            c = start + _get_rng(rng).randint(0, size)
        return c

    # Default: XOR to flip bits while staying in printable-ish range
    if b < 32:
        return b ^ 0x1F
    return b ^ 0x7F


def type_replace(data: bytes) -> bytes:
    """Replace all bytes with different values from the same character class.

    Uses a precomputed 256-byte translate table for O(n) processing
    with no per-byte Python branching.

    Args:
        data: Input bytes to mutate.

    Returns:
        Mutated bytes with each byte replaced within its class.
    """
    return data.translate(_TYPE_REPLACE_TABLE)


# ---------------------------------------------------------------------------
# Duplicate elimination helpers (ported from AFL++ afl-fuzz-one.c)
# ---------------------------------------------------------------------------


def could_be_bitflip(xor_val: int) -> bool:
    """Check if an XOR difference could be produced by a bitflip stage.

    Deterministic bitflip stages flip 1, 2, or 4 contiguous bits, or
    flip whole bytes (XOR 0xFF). If xor_val matches one of these patterns,
    a later arithmetic or interesting-value stage would produce a duplicate.

    Args:
        xor_val: XOR between old and new byte/word value.

    Returns:
        True if the difference is already covered by bitflip stages.
    """
    if not xor_val:
        return True

    # Find position of lowest set bit
    sh = 0
    v = xor_val
    while not (v & 1):
        sh += 1
        v >>= 1

    # 1-, 2-, and 4-bit patterns are covered anywhere
    if v in (1, 3, 15):
        return True

    # 8-, 16-, 32-bit patterns only at byte boundaries
    if sh & 7:
        return False

    return v in (0xFF, 0xFFFF, 0xFFFFFFFF)


def could_be_arith(old_val: int, new_val: int, blen: int) -> bool:
    """Check if a value change could be produced by an arithmetic stage.

    Arithmetic stages add/subtract small values (1..ARITH_MAX) to individual
    bytes, words, or dwords. This checks if the old->new difference at any
    byte/word/dword boundary is within ARITH_MAX.

    Args:
        old_val: Original value (u32).
        new_val: New value (u32).
        blen: Byte length of the value (1, 2, or 4).

    Returns:
        True if the difference is already covered by arithmetic stages.
    """
    if old_val == new_val:
        return True

    # Check single-byte adjustments
    diffs = 0
    ov = nv = 0
    for i in range(blen):
        a = (old_val >> (8 * i)) & 0xFF
        b = (new_val >> (8 * i)) & 0xFF
        if a != b:
            diffs += 1
            ov, nv = a, b

    if diffs == 1 and (((ov - nv) & 0xFF) <= ARITH_MAX or ((nv - ov) & 0xFF) <= ARITH_MAX):
        return True

    if blen == 1:
        return False

    # Check two-byte (word) adjustments
    diffs = 0
    for i in range(blen // 2):
        a = (old_val >> (16 * i)) & 0xFFFF
        b = (new_val >> (16 * i)) & 0xFFFF
        if a != b:
            diffs += 1
            ov, nv = a, b

    if diffs == 1:
        # Little-endian check
        if ((ov - nv) & 0xFFFF) <= ARITH_MAX or ((nv - ov) & 0xFFFF) <= ARITH_MAX:
            return True
        # Big-endian check (byte-swap)
        ov_be = ((ov & 0xFF) << 8) | ((ov >> 8) & 0xFF)
        nv_be = ((nv & 0xFF) << 8) | ((nv >> 8) & 0xFF)
        if ((ov_be - nv_be) & 0xFFFF) <= ARITH_MAX or ((nv_be - ov_be) & 0xFFFF) <= ARITH_MAX:
            return True

    # Check dword adjustments
    if blen != 4:
        return False
    return ((old_val - new_val) & 0xFFFFFFFF) <= ARITH_MAX or (
        (new_val - old_val) & 0xFFFFFFFF
    ) <= ARITH_MAX


def could_be_interest(old_val: int, new_val: int, blen: int, check_le: bool = True) -> bool:
    """Check if a value change could be produced by an interesting-value stage.

    Interesting-value stages replace bytes/words/dwords with specific
    boundary values (-128, 0, 1, 127, 255, 32767, etc.). This checks
    if old_val with one such replacement at any position yields new_val.

    Args:
        old_val: Original value (u32).
        new_val: New value (u32).
        blen: Byte length (1, 2, or 4).
        check_le: Also check LE word insertions before BE attempts.

    Returns:
        True if the difference is already covered by interesting-value stages.
    """
    if old_val == new_val:
        return True

    # Check single-byte insertions
    for i in range(blen):
        for j in range(len(INTERESTING_8)):
            tval = (old_val & ~(0xFF << (8 * i))) | ((INTERESTING_8[j] & 0xFF) << (8 * i))
            if new_val == tval:
                return True

    if blen == 2 and not check_le:
        return False

    # Check two-byte (word) insertions
    for i in range(blen - 1):
        for j in range(len(INTERESTING_16)):
            tval = (old_val & ~(0xFFFF << (8 * i))) | ((INTERESTING_16[j] & 0xFFFF) << (8 * i))
            if new_val == tval:
                return True
            if blen > 2:
                # Big-endian variant
                swapped = ((INTERESTING_16[j] & 0xFF) << 8) | ((INTERESTING_16[j] >> 8) & 0xFF)
                tval = (old_val & ~(0xFFFF << (8 * i))) | (swapped << (8 * i))
                if new_val == tval:
                    return True

    if blen == 4 and check_le:
        for j in range(len(INTERESTING_32)):
            if new_val == (INTERESTING_32[j] & 0xFFFFFFFF):
                return True

    return False


# ---------------------------------------------------------------------------
# Supplementary mutations (from AFL++ afl-mutations.h)
# ---------------------------------------------------------------------------


def ascii_num_replace(data: bytes, rng=None) -> bytes:
    """Replace a whole multi-digit ASCII number with a random numeric value.

    Ported from go-fuzz case 15.  Finds runs of digits (optionally prefixed
    with '-') of length >= 2 and replaces the whole token with a random
    value chosen from: small int [0, 999], big int, big-int squared, or
    negative big int.  If the original token was negative, the replacement
    is also mostly negative.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)

    # Collect candidate number spans.
    numbers: list[tuple[int, int]] = []
    start = -1
    for i, v in enumerate(data):
        digit = 0x30 <= v <= 0x39
        if digit:
            if start == -1:
                start = i
        else:
            if start != -1 and i - start > 1:
                numbers.append((start, i))
            start = -1
    if start != -1 and len(data) - start > 1:
        numbers.append((start, len(data)))

    if not numbers:
        return data

    r = rng.choice(numbers)
    neg = data[r[0]] == 45
    raw = data[r[0] + 1 : r[1]] if neg else data[r[0] : r[1]]

    # Only replace when the span is all digits (no embedded '-' mid-token).
    if any(not (0x30 <= b <= 0x39) for b in raw):
        return data

    strategy = rng.randint(0, 3)
    if strategy == 0:
        v = rng.randint(0, 999)
    elif strategy == 1:
        v = rng.randint(0, (1 << 30) - 1)
    elif strategy == 2:
        v = rng.randint(0, (1 << 30) - 1) ** 2
    else:
        v = -rng.randint(0, (1 << 30) - 1)

    if neg:
        v = -v
    repl = str(v).encode("ascii")
    prefix = data[: r[0]]
    suffix = data[r[1] :]
    return prefix + repl + suffix


def insert_ascii_num(data: bytes, max_len: int = 65536, rng=None) -> bytes:
    """Insert an ASCII number string at a random position.

    Like ascii_num_replace but inserts rather than overwrites.
    Useful for fuzzing fields that accept numeric values.

    Args:
        data: Input bytes.
        max_len: Maximum output length.

    Returns:
        Bytes with an ASCII number inserted.
    """
    if len(data) >= max_len:
        return data

    idx = _get_rng(rng).randint(0, len(data))
    num = _get_rng(rng).randint(0, 99999)
    num_str = str(num).encode("ascii")
    result = data[:idx] + num_str + data[idx:]
    return result[:max_len]


def byte_shuffle(data: bytes, rng=None) -> bytes:
    """Shuffle a random subset of bytes in the input.

    Optimized: shuffle only a random portion instead of the entire buffer.

    Args:
        data: Input bytes.

    Returns:
        Partially shuffled bytes.
    """
    if len(data) <= 1:
        return data
    r = _get_rng(rng)
    result = bytearray(data)
    # Shuffle only a random 20-50% subset
    n = max(2, len(result) // r.randint(2, 5))
    start = r.randint(0, max(0, len(result) - n))
    # bytearray slicing returns a *copy*, so shuffling result[start:start+n]
    # in place discarded the result and left `result` untouched (no-op).
    # Shuffle the copy, then assign it back into the buffer.
    sub = result[start : start + n]
    r.shuffle(sub)
    result[start : start + n] = sub
    return bytes(result)


def byte_delete(data: bytes, rng=None) -> bytes:
    """Delete a single random byte from the input.

    Args:
        data: Input bytes.

    Returns:
        Bytes with one byte removed, or original if too short.
    """
    if len(data) <= 1:
        return data

    idx = _get_rng(rng).randint(0, len(data) - 1)
    return data[:idx] + data[idx + 1 :]


def byte_insert(data: bytes, max_len: int = 65536, rng=None) -> bytes:
    """Insert a single random byte at a random position.

    Args:
        data: Input bytes.
        max_len: Maximum output length.

    Returns:
        Bytes with one random byte inserted.
    """
    if len(data) >= max_len:
        return data

    idx = _get_rng(rng).randint(0, len(data))
    val = _get_rng(rng).randint(0, 255)
    return data[:idx] + bytes([val]) + data[idx:]


def splice_diff_located(a: bytes, b: bytes, rng=None) -> bytes:
    """Splice two inputs at optimal cut points found via diff locating.

    Unlike random splice, this finds the first and last differing bytes
    between a and b, then picks cut points only within that range.
    This produces more meaningful hybrids.

    Ported from AFL's locate_diffs + splice logic.

    Args:
        a: First input (base).
        b: Second input (donor).

    Returns:
        Spliced bytes, or a unchanged if inputs are too short or identical.
    """
    if len(a) < 2 or len(b) < 2:
        return a

    min_len = min(len(a), len(b))

    # Find first and last differing positions
    first_diff = -1
    last_diff = -1
    for i in range(min_len):
        if a[i] != b[i]:
            if first_diff == -1:
                first_diff = i
            last_diff = i

    if first_diff == -1:
        # Identical up to min_len — just do random splice
        cut_a = _get_rng(rng).randint(1, len(a) - 1)
        cut_b = _get_rng(rng).randint(1, len(b) - 1)
        return a[:cut_a] + b[cut_b:]

    # Pick cut points within the diff range
    cut_a = _get_rng(rng).randint(first_diff, last_diff)
    cut_b = _get_rng(rng).randint(first_diff, min(last_diff, len(b) - 1))

    return a[:cut_a] + b[cut_b:]


# ---------------------------------------------------------------------------
# Block transposition mutations
# ---------------------------------------------------------------------------


def transpose_bytes(data: bytes, width: int, rng=None) -> bytes:
    """Permute bytes within a randomly-selected aligned block of *width* bytes.

    For width=2: swaps the two bytes. For width=4 or 8: applies a random
    permutation of all bytes in the block. Preserves input length.

    Args:
        data: Input bytes.
        width: Block width in bytes (2, 4, or 8).

    Returns:
        Bytes with one block's bytes transposed.
    """
    if len(data) < width:
        return data
    max_start = len(data) - width
    start = (_get_rng(rng).randint(0, max_start) // width) * width
    result = bytearray(data)
    # Shuffle in-place on a memoryview slice
    mv = memoryview(result)[start : start + width]
    lst = list(mv)
    _get_rng(rng).shuffle(lst)
    for i, v in enumerate(lst):
        mv[i] = v
    return bytes(result)


def bit_transpose(data: bytes, width: int, rng=None) -> bytes:
    """Permute bits within a randomly-selected block of *width* bytes.

    Optimized: swaps random bit pairs instead of full shuffle.
    Uses struct.pack_into for in-place write (avoids extra copy).

    Args:
        data: Input bytes.
        width: Block width in bytes (1, 2, 4, or 8).

    Returns:
        Bytes with one block's bits transposed.
    """
    if len(data) < width:
        return data
    max_start = len(data) - width
    start = (_get_rng(rng).randint(0, max_start) // width) * width
    val = int.from_bytes(data[start : start + width], "little")
    total_bits = 8 * width
    # Swap 2-4 random bit pairs instead of full shuffle
    n_swaps = _get_rng(rng).randint(2, min(4, total_bits // 2))
    for _ in range(n_swaps):
        i = _get_rng(rng).randint(0, total_bits - 1)
        j = _get_rng(rng).randint(0, total_bits - 1)
        if i != j:
            bi = (val >> i) & 1
            bj = (val >> j) & 1
            if bi != bj:
                val ^= (1 << i) | (1 << j)
    result = bytearray(data)
    result[start : start + width] = val.to_bytes(width, "little")
    return bytes(result)


# ---------------------------------------------------------------------------
# Rotation / shift / inversion
#
# bit_transpose permutes bits inside a word and byte_flip inverts exactly one
# byte, so before these three the operator table could toggle bits and reorder
# them but could not *translate* them: nothing moved a packed bitfield off its
# alignment, nothing dropped bits off the end of a word, and nothing inverted a
# run that was not byte-aligned.
#
# All three take their word window little-endian, matching bit_transpose above
# and the LSB-first bit numbering that _op_bit_offset_flip uses (bit offset i
# is byte i >> 3, bit i & 7), so a bit offset means the same thing everywhere
# in this codebase.
#
# Note these use only randint/random/choice: this module's default RNG is the
# stdlib `random` module (see _get_rng), which has no weighted_choice, so
# non-uniform draws are spelled as repeated entries in a candidate tuple.
# ---------------------------------------------------------------------------

# bytes.translate runs in C, so span_invert's bulk middle section costs one
# pass rather than a Python-level XOR per byte.
_INVERT_TABLE = bytes(b ^ 0xFF for b in range(256))

# Retry budget for the two operators whose output can coincide with their
# input on degenerate windows (all-0x00, all-0xFF). Retrying a few starts
# costs microseconds and stops the operator from spending a scheduler
# selection slot on a guaranteed no-op -- the failure mode that hid in
# byte_shuffle until f4835f6.
_DEGENERATE_RETRIES = 4


def bit_rotate(data: bytes, rng=None) -> bytes:
    """Cyclically rotate the bits of one word-sized window.

    Preserves popcount but shifts every field boundary inside the window,
    which is what breaks packed bitfield decoders and header flag parsing.

    Args:
        data: Input bytes.
        rng: Optional RNG; defaults to the module-level `random`.

    Returns:
        Bytes with one window rotated, or the input unchanged if every
        sampled window was rotation-invariant.
    """
    if not data:
        return data
    r = _get_rng(rng)
    width = r.choice(tuple(w for w in (1, 1, 1, 2, 2, 4, 4, 8) if w <= len(data)))
    total_bits = 8 * width
    mask = (1 << total_bits) - 1
    # Sub-byte amounts are the distinctive capability here. A rotation by a
    # multiple of 8 is a cyclic *byte* shift, which swap_bytes and
    # endianness_swap already reach, so only take the full range sometimes.
    #
    # randint rather than random() < 0.75: the exhaustive pool intercepts
    # discrete draws but classifies a continuous one as un-enumerable, which
    # put this operator outside test_no_operator_exceeds_max_len's reach.
    if total_bits == 8 or r.randint(0, 3) < 3:
        amount = r.randint(1, 7)
    else:
        amount = r.randint(1, total_bits - 1)
    left = r.randint(0, 1) == 1
    max_start = len(data) - width
    result = bytearray(data)
    for _ in range(_DEGENERATE_RETRIES):
        start = r.randint(0, max_start)
        val = int.from_bytes(data[start : start + width], "little")
        if left:
            out = ((val << amount) | (val >> (total_bits - amount))) & mask
        else:
            out = ((val >> amount) | (val << (total_bits - amount))) & mask
        if out != val:
            result[start : start + width] = out.to_bytes(width, "little")
            break
    return bytes(result)


def bit_shift(data: bytes, rng=None) -> bytes:
    """Shift one word-sized window left or right, logically or arithmetically.

    Unlike bit_rotate this is lossy: bits fall off the end and the vacated
    positions are zero-filled (or sign-filled, for the arithmetic right
    shift). Good at length and index fields that get scaled before use.

    Args:
        data: Input bytes.
        rng: Optional RNG; defaults to the module-level `random`.

    Returns:
        Bytes with one window shifted, or the input unchanged if every
        sampled window was shift-invariant.
    """
    if not data:
        return data
    r = _get_rng(rng)
    width = r.choice(tuple(w for w in (1, 1, 1, 2, 2, 4, 4, 8) if w <= len(data)))
    total_bits = 8 * width
    mask = (1 << total_bits) - 1
    # Small shifts keep most of the original value in play; a large shift just
    # zeroes the field, which random_bytes already covers.
    amount = min(r.choice((1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 12)), total_bits - 1)
    kind = r.choice(("shl", "shr", "sar"))
    max_start = len(data) - width
    result = bytearray(data)
    for _ in range(_DEGENERATE_RETRIES):
        start = r.randint(0, max_start)
        val = int.from_bytes(data[start : start + width], "little")
        if kind == "shl":
            out = (val << amount) & mask
        elif kind == "shr":
            out = val >> amount
        else:
            out = val >> amount
            if val >> (total_bits - 1):  # sign-propagate
                out |= (mask << (total_bits - amount)) & mask
        if out != val:
            result[start : start + width] = out.to_bytes(width, "little")
            break
    return bytes(result)


def span_invert(data: bytes, rng=None) -> bytes:
    """Invert a contiguous run of bits, which need not be byte-aligned.

    Generalises byte_flip, which can only invert a whole byte at a
    byte-aligned index. The span length is drawn from sub-byte runs up to the
    whole buffer, so this reaches both single-flag inversion inside a packed
    header and a full ``~buffer``.

    Args:
        data: Input bytes.
        rng: Optional RNG; defaults to the module-level `random`.

    Returns:
        Bytes with one bit run inverted. Always differs from the input --
        the XOR mask is non-zero by construction.
    """
    if not data:
        return data
    r = _get_rng(rng)
    total_bits = 8 * len(data)
    span = min(
        r.choice((1, 2, 3, 4, 8, 8, 16, 16, 32, 32, 64, 128, 256, total_bits)),
        total_bits,
    )
    start = r.randint(0, total_bits - span)
    end = start + span - 1
    first, last = start >> 3, end >> 3
    result = bytearray(data)
    if first == last:
        lo, hi = start & 7, end & 7
        result[first] ^= ((1 << (hi - lo + 1)) - 1) << lo
        return bytes(result)
    result[first] ^= (0xFF << (start & 7)) & 0xFF
    result[last] ^= (1 << ((end & 7) + 1)) - 1
    if last > first + 1:
        result[first + 1 : last] = result[first + 1 : last].translate(_INVERT_TABLE)
    return bytes(result)


# Element widths for bit_repack, weighted toward the ones real formats use:
# 1/2/4/8/16 are PNG's sample depths and BMP's sub-byte bpp modes, 10/12 are
# the common camera-RAW and TIFF extended depths. 3/5/6 are included thinly
# because packed RGB modes (RGB565, RGB332) use them and they are the widths
# most likely to expose an off-by-one in a parser's bit accumulator.
_REPACK_WIDTHS = (1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 8, 8, 8, 10, 12, 16)

# Span sizes in bytes. Capped at 64 so the per-element Python loop stays
# cheap: a 64-byte span at 1-bit elements is already 512 iterations.
_REPACK_SPANS = (2, 4, 4, 8, 8, 16, 16, 32, 64)


def _repack_bits(span: bytes, src_w: int, dst_w: int, msb_first: bool, scale: bool) -> bytes:
    """Re-emit *span*'s ``src_w``-bit elements as ``dst_w``-bit elements.

    Split out from bit_repack so the bit-order and padding semantics can be
    tested directly with fixed parameters. bit_repack itself only samples the
    parameters; everything that can be wrong about the packing is in here.

    Any sub-element tail (``len(span) * 8 % src_w`` bits) is carried through
    verbatim -- dropping it would make the operator lossy in a way that has
    nothing to do with the repack.
    """
    span_bits = len(span) * 8
    n_el = span_bits // src_w
    rem_bits = span_bits - n_el * src_w
    out_bits = n_el * dst_w + rem_bits
    out_len = (out_bits + 7) // 8
    order = "big" if msb_first else "little"
    src_mask = (1 << src_w) - 1
    dst_mask = (1 << dst_w) - 1
    val = int.from_bytes(span, order)

    out = 0
    for i in range(n_el):
        shift = span_bits - (i + 1) * src_w if msb_first else i * src_w
        v = (val >> shift) & src_mask
        w = (v * dst_mask // src_mask) if scale else (v & dst_mask)
        if msb_first:
            out = (out << dst_w) | w
        else:
            out |= w << (i * dst_w)

    if rem_bits:
        tail_mask = (1 << rem_bits) - 1
        if msb_first:
            out = (out << rem_bits) | (val & tail_mask)
        else:
            out |= ((val >> (n_el * src_w)) & tail_mask) << (n_el * dst_w)

    if msb_first:
        # Left-align so the stream starts at the top bit of the first byte,
        # where an MSB-first reader expects it. Without this the whole stream
        # sits right-aligned and every element reads shifted. The
        # little-endian case needs no padding: the spare high bits are
        # already past the end of the stream.
        out <<= out_len * 8 - out_bits

    return out.to_bytes(out_len, order)


def bit_repack(data: bytes, rng=None, max_len: int = 65536) -> bytes:
    """Reinterpret a span as k-bit elements and re-emit them at j-bit.

    Every other operator in the bit band transforms a *fixed* window: flip,
    rotate, shift and transpose all leave the element boundary grid where they
    found it. This moves the grid itself, which is the mutation that matters
    for any format carrying sub-byte packed samples -- PNG bit depths 1/2/4,
    BMP's 1/4bpp modes, GIF's LZW code stream, camera-RAW 10/12-bit.

    The source and destination widths are always different: the identity
    width pair is unsampleable rather than checked for afterwards, since a
    repack at the same width is work that looks like work from the outside --
    the failure mode that hid in byte_shuffle until f4835f6. That is a
    narrower guarantee than "the output always differs": a uniform span
    repacks to a uniform span, and when the width change happens to round to
    the same byte length the result coincides with the input. Measured at ~2%
    on an all-zero buffer, 0% on structured input.

    Bit order within the packed stream is drawn per call: PNG, BMP and TIFF
    pack samples MSB-first, while many bit-reader implementations and RAW
    variants pack LSB-first, and a parser is usually only correct for one.

    Values map across widths either by masking (truncate, or zero-extend when
    widening) or by proportional scaling (``v * dst_max // src_max``, which is
    what a real bit-depth conversion does). Masking is more likely to land on
    a boundary value; scaling is more likely to survive a validity check.

    Args:
        data: Input bytes.
        rng: Optional RNG; defaults to the module-level `random`.
        max_len: Maximum output length. Repacking changes length by roughly
            ``dst_w / src_w``, so this is a real constraint rather than a
            formality: 4-bit to 16-bit quadruples the span.

    Returns:
        Bytes with one span repacked, or the input unchanged when no sampled
        span fit inside *max_len*.
    """
    if len(data) < 2:
        return data
    r = _get_rng(rng)
    src_w = r.choice(_REPACK_WIDTHS)
    # Draw the destination from the widths that are not the source, so the
    # identity case cannot be sampled at all.
    dst_w = r.choice(tuple(w for w in _REPACK_WIDTHS if w != src_w))
    msb_first = r.randint(0, 1) == 1
    scale = r.randint(0, 1) == 1

    # Widening blows the span up by dst_w/src_w, so a span that fits at 8->10
    # will not fit at 1->16. Sample a few rather than declining on the first
    # miss, which would make the operator quietly rarer on tight max_len.
    for _ in range(_DEGENERATE_RETRIES):
        span_len = min(r.choice(_REPACK_SPANS), len(data))
        span_bits = span_len * 8
        n_el = span_bits // src_w
        if n_el == 0:
            continue
        rem_bits = span_bits - n_el * src_w
        out_bits = n_el * dst_w + rem_bits
        out_len = (out_bits + 7) // 8
        if len(data) - span_len + out_len <= max_len:
            break
    else:
        return data

    start = r.randint(0, len(data) - span_len)
    repacked = _repack_bits(data[start : start + span_len], src_w, dst_w, msb_first, scale)
    result = data[:start] + repacked + data[start + span_len :]
    return bytes(result[:max_len])


# ---------------------------------------------------------------------------
# Radamsa-style number mutation (ported from radamsa/rad/shared.scm mutate-num)
# ---------------------------------------------------------------------------

# Interesting boundary numbers generated from powers of 2 ± 1 (Radamsa).
_RADAMSA_BOUNDARIES: list[int] = []
for _shift in (1, 7, 8, 15, 16, 31, 32):
    _x = 1 << _shift
    _RADAMSA_BOUNDARIES.append(_x - 1)
    _RADAMSA_BOUNDARIES.append(_x)
    _RADAMSA_BOUNDARIES.append(_x + 1)


def radamsa_mutate_num(val: int, rng=None) -> int:
    """Mutate a numeric value using Radamsa's mutate-num strategy.

    Randomly picks one of several transforms: increment, decrement,
    set to 0/1, replace with an interesting boundary, add/subtract an
    interesting boundary, or random scaling.

    Ported from radamsa/rad/shared.scm ``mutate-num``.
    """
    op = _get_rng(rng).randint(0, 9)
    if op == 0:
        return val + 1
    if op == 1:
        return val - 1
    if op in (2, 3):
        return 0 if op == 2 else 1
    if op in (4, 5, 6):
        return _get_rng(rng).choice(_RADAMSA_BOUNDARIES)
    if op == 7:
        return val + _get_rng(rng).choice(_RADAMSA_BOUNDARIES)
    if op == 8:
        return _get_rng(rng).choice(_RADAMSA_BOUNDARIES) - val
    # op == 9: random scaling
    n = _get_rng(rng).randint(1, 128)
    n = _log2_ceil(n)
    return val + n if _get_rng(rng).random() < 0.5 else val - n


def _log2_ceil(n: int) -> int:
    """Small helper: return ceil(log2(n)) for small n, clamped to 1+."""
    if n <= 1:
        return 1
    r = 0
    while (1 << r) < n:
        r += 1
    return r


# ---------------------------------------------------------------------------
# Funny Unicode sequences (ported from radamsa/rad/mutations.scm)
# Each entry is a raw byte sequence representing problematic Unicode.
# ---------------------------------------------------------------------------

_FUNNY_UNICODE: list[bytes] = [
    # Override / control characters
    b"\xe2\x80\xae",  # U+202E Right to Left Override
    b"\xe2\x80\xad",  # U+202D Left to Right Override
    b"\xe1\xa0\x8e",  # U+180E Mongolian Vowel Separator
    b"\xe2\x81\xa0",  # U+2060 Word Joiner
    # Reserved / non-characters
    b"\xef\xbb\xbe",  # U+FEFE reserved
    b"\xef\xbf\xbf",  # U+FFFF not a character
    b"\xe0\xbf\xad",  # U+0FED unassigned
    # Illegal surrogates
    b"\xed\xba\xad",  # U+DEAD illegal low surrogate
    b"\xed\xaa\xad",  # U+DAAD illegal high surrogate
    # Private use
    b"\xef\xa3\xbf",  # U+F8FF private use char (Apple)
    b"\xef\xbc\x8f",  # U+FF0F full width solidus
    # Mathematical
    b"\xf0\x9d\x9f\x96",  # U+1D7D6 MATHEMATICAL BOLD DIGIT EIGHT
    # IDNA deviant
    b"\xc3\x9f",  # U+00DF IDNA deviant
    # NFKC expansion bombs (expand dramatically under normalization)
    b"\xef\xb7\xba",  # U+FDFD expands by 11x (UTF-8) / 18x (UTF-16) NFKC
    b"\xce\x90",  # U+0390 expands 3x under NFD
    b"\xe1\xbe\x82",  # U+1F82 expands 4x under NFD
    b"\xef\xac\xac",  # U+FB2C expands 3x under NFC
    b"\xf0\x9d\x85\xa0",  # U+1D160 expands 3x under NFC
    # Out of range
    b"\xf4\x8f\xbf\xbe",  # illegal: beyond U+10FFFF
    # Boundary values
    b"\xef\xbf\xbf",  # U+FFFF
    b"\xf0\x90\x80\x80",  # U+10000
    # BOMs
    b"\xef\xbb\xbf",  # UTF-8 BOM
    b"\xfe\xff",  # UTF-16 BE BOM
    b"\xff\xfe",  # UTF-16 LE BOM
    # Mixed-endian nulls
    b"\x00\x00\xff\xff",  # ASCII null BE
    b"\xff\xff\x00\x00",  # ASCII null LE
    # Various magic bytes from Radamsa / Wikipedia
    b"\x2b\x2f\x76\x38",
    b"\x2b\x2f\x76\x39",
    b"\x2b\x2f\x76\x2b",
    b"\x2b\x2f\x76\x2f",
    b"\xf7\x64\x4c",
    b"\xdd\x73\x66\x73",
    b"\x0e\xfe\xff",
    b"\xfb\xee\x28",
    b"\xfb\xee\x28\xff",
    b"\x84\x31\x95\x33",
    # Whitespace confusables / specials
    b"\xc2\xa0",  # U+00A0 non-breaking space
    b"\xe1\x9a\x80",  # U+1680 Ogham space mark
    b"\xe1\xa0\x8e",  # U+180E Mongolian vowel separator
    b"\xe2\x80\x80",  # U+2000 en quad
    b"\xe2\x80\x8b",  # U+200B zero-width space
    b"\xe2\x80\x8c",  # U+200C zero-width non-joiner
    b"\xe2\x80\x8d",  # U+200D zero-width joiner
    b"\xef\xbb\xbf",  # U+FEFF zero-width no-break space (BOM)
]


# ---------------------------------------------------------------------------
# LEB128 helpers
# ---------------------------------------------------------------------------


def encode_uleb128(value: int) -> bytes:
    """Encode an unsigned integer as ULEB128.

    Args:
        value: Non-negative integer to encode.

    Returns:
        ULEB128 byte sequence.
    """
    if value < 0:
        raise ValueError("ULEB128 encodes unsigned values only")
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def leb128_encode(data: bytes, rng=None, max_len: int = 65536) -> bytes:
    """Mutate an input by rewriting or inserting a ULEB128-encoded integer.

    Prefers mutating an existing candidate integer in-place: scans for a
    1-5 byte little-endian unsigned integer and rewrites it as ULEB128.
    Falls back to inserting a random small LEB128 value when no candidate
    is found. Returns the input unchanged on empty data or when the
    mutated result would exceed ``max_len``.

    Args:
        data: Input bytes.
        rng: Optional random source.
        max_len: Maximum output length.

    Returns:
        Mutated bytes, or the original bytes if no mutation was applied.
    """
    if not data:
        return data
    r = _get_rng(rng)
    result = bytearray(data)

    # Candidate widths in bytes, smallest first.
    widths = [1, 2, 3, 4, 5]
    for width in widths:
        if len(result) < width:
            continue
        idx = r.randint(0, len(result) - width)
        value = int.from_bytes(result[idx : idx + width], "little")
        encoded = encode_uleb128(value)
        if not encoded:
            continue
        new_len = width - 1 + len(encoded)
        if len(result) - new_len + 1 > max_len:
            continue
        result[idx : idx + width - 1] = encoded + b"\x00" * (width - 1 - len(encoded))
        return bytes(result[:max_len])

    # Fallback: insert a random small LEB128 at a random position.
    value = r.randint(0, 255)
    encoded = encode_uleb128(value)
    if len(result) + len(encoded) > max_len:
        return data
    pos = r.randint(0, len(result))
    return result[:pos] + encoded + result[pos:max_len]


# ---------------------------------------------------------------------------
# Versifier: text-structure learner/generator (ported from go-fuzz)
# ---------------------------------------------------------------------------


class _VerseNode:
    def Visit(self, f):
        pass

    def Generate(self, v, buf):
        pass


class _WsNode(_VerseNode):
    def __init__(self, samples):
        self._samples = list(set(samples))

    def Generate(self, v, buf):
        if v._rand.random() != 0 and self._samples:
            buf += v._rand.choice(self._samples)
        else:
            for _ in range(v._rand.randint(0, 3)):
                buf += v._rand.choice([b" ", b"\t"])


class _AlphaNumNode(_VerseNode):
    def __init__(self, samples):
        self._samples = list(set(samples))

    def Generate(self, v, buf):
        if v._rand.random() != 0 and self._samples:
            buf += v._rand.choice(self._samples)
        else:
            length = [v._rand.randint(0, 3), v._rand.randint(0, 19), v._rand.randint(0, 99)][
                v._rand.randint(0, 2)
            ]
            for _ in range(length):
                kind = v._rand.randint(0, 3)
                if kind == 0:
                    buf += b"_"
                elif kind == 1:
                    buf += bytes([0x30 + v._rand.randint(0, 9)])
                elif kind == 2:
                    buf += bytes([0x61 + v._rand.randint(0, 25)])
                else:
                    buf += bytes([0x41 + v._rand.randint(0, 25)])


class _NumNode(_VerseNode):
    def __init__(self, samples, hex=False):
        self._samples = list(set(samples))
        self._hex = hex

    def Generate(self, v, buf):
        if v._rand.random() == 0 and self._samples:
            buf += v._rand.choice(self._samples)
            return
        base = [8, 10, 16][v._rand.randint(0, 2)]
        length = [v._rand.randint(0, 3), v._rand.randint(0, 15), v._rand.randint(0, 39)][
            v._rand.randint(0, 2)
        ]
        num = bytearray()
        for _ in range(length):
            if base == 8:
                num += bytes([0x30 + v._rand.randint(0, 7)])
            elif base == 10:
                num += bytes([0x30 + v._rand.randint(0, 9)])
            else:
                kind = v._rand.randint(0, 2)
                if kind == 0:
                    num += bytes([0x30 + v._rand.randint(0, 9)])
                elif kind == 1:
                    num += bytes([0x61 + v._rand.randint(0, 5)])
                else:
                    num += bytes([0x41 + v._rand.randint(0, 5)])
        if base == 8:
            buf += b"0" + num
        elif base == 16:
            buf += b"0x" + num
        else:
            buf += num
        if v._rand.random() == 0:
            buf += b"-"


class _ControlNode(_VerseNode):
    def __init__(self, ch):
        self._ch = bytes([ch])

    def Generate(self, v, buf):
        if v._rand.randint(0, 9) != 0:
            buf += self._ch
        else:
            for _ in range(10):
                b = v._rand.randint(0, 127)
                if 0x30 <= b <= 0x39 or 0x61 <= b <= 0x7A or 0x41 <= b <= 0x5A:
                    continue
                buf += bytes([b])
                break


_BRACKETS = {
    ord("<"): ord(">"),
    ord("["): ord("]"),
    ord("("): ord(")"),
    ord("{"): ord("}"),
    ord("'"): ord("'"),
    ord('"'): ord('"'),
    ord("`"): ord("`"),
}


class _BracketNode(_VerseNode):
    def __init__(self, open_ch, close_ch, inner):
        self._open = bytes([open_ch])
        self._close = bytes([close_ch])
        self._inner = inner

    def Generate(self, v, buf):
        if v._rand.randint(0, 9) != 0:
            buf += self._open
            self._inner.Generate(v, buf)
            buf += self._close
        else:
            brk = [b"<", b"[", b"(", b"{", b"'", b'"', b"`"]
            open_b = v._rand.choice(brk)
            close_b = bytes([_BRACKETS[open_b[0]]])
            if v._rand.randint(0, 4) == 0:
                close_b = bytes([_BRACKETS[v._rand.choice(brk)[0]]])
            buf += open_b
            self._inner.Generate(v, buf)
            buf += close_b


class _KeyValNode(_VerseNode):
    def __init__(self, key, value):
        self._key = key
        self._value = value

    def Generate(self, v, buf):
        self._key.Generate(v, buf)
        buf += bytes([0x3A if v._rand.random() < 0.5 else 0x3D])
        self._value.Generate(v, buf)


class _ListNode(_VerseNode):
    def __init__(self, delim, blocks):
        self._delim = bytes([delim])
        self._blocks = blocks

    def Generate(self, v, buf):
        blocks = list(self._blocks)
        if v._rand.randint(0, 4) == 0:
            blocks = []
            while v._rand.randint(0, 2) != 0:
                blocks.append(v._rand.choice(self._blocks))
        for i, b in enumerate(blocks):
            if i != 0:
                buf += self._delim
            b.Generate(v, buf)


class _LineNode(_VerseNode):
    def __init__(self, crlf, inner):
        self._crlf = crlf
        self._inner = inner

    def Generate(self, v, buf):
        self._inner.Generate(v, buf)
        if self._crlf:
            buf += b"\r\n"
        else:
            buf += b"\n"


class _BlockNode(_VerseNode):
    def __init__(self, nodes):
        self._nodes = nodes

    def Generate(self, v, buf):
        nodes = list(self._nodes)
        if v._rand.randint(0, 9) == 0:
            while len(nodes) > 0 and v._rand.randint(0, 1) == 0:
                idx = v._rand.randint(0, len(nodes) - 1)
                nodes = nodes[:idx] + nodes[idx + 1 :]
        if v._rand.randint(0, 9) == 0:
            while len(nodes) > 0 and v._rand.randint(0, 1) == 0:
                idx = v._rand.randint(0, len(nodes) - 1)
                nodes = nodes[:idx] + [None] + nodes[idx:]
        if v._rand.randint(0, 9) == 0:
            while len(nodes) > 0 and v._rand.randint(0, 1) == 0:
                idx1 = v._rand.randint(0, len(nodes) - 1)
                idx2 = v._rand.randint(0, len(nodes) - 1)
                nodes[idx1], nodes[idx2] = nodes[idx2], nodes[idx1]
        for n in nodes:
            if n is None:
                continue
            if v._rand.randint(0, 19) == 0:
                continue
            n.Generate(v, buf)


class Verse:
    def __init__(self, rng):
        self._blocks = []
        self._all_nodes = []
        self._rand = rng

    def add_block(self, block):
        self._blocks.append(block)

    def Rhyme(self):
        buf = bytearray()
        if not self._blocks:
            return buf
        block = self._blocks[self._rand.randint(0, len(self._blocks) - 1)]
        block.Generate(self, buf)
        return bytes(buf)


def _is_hex(s):
    if not s:
        return False
    for c in s:
        if 0x30 <= c <= 0x39 or 0x61 <= c <= 0x66 or 0x41 <= c <= 0x46:
            continue
        return False
    return True


def _is_dec(s):
    if not s:
        return False
    for c in s:
        if 0x30 <= c <= 0x39:
            continue
        return False
    return True


def _build_verse(data, rng):
    # Same bytearray hazard as extract_corpus_literals: _tokenize slices
    # `data`, and downstream those slices land in sets and dict keys, which
    # raises TypeError for bytearray. Callers pass corpus entries directly.
    data = bytes(data)
    printable = sum(1 for b in data if 0x20 <= b < 0x7F)
    if printable < len(data) * 9 // 10:
        return None
    v = Verse(rng)
    nodes = _tokenize(data)
    nodes = _structure(nodes)
    v.add_block(_BlockNode(nodes))
    return v


def _tokenize(data):
    nodes = []
    i = 0
    state = 0  # 0=control, 1=ws, 2=alpha, 3=num
    start = 0
    while i < len(data):
        b = data[i]
        is_alpha = (0x61 <= b <= 0x7A) or (0x41 <= b <= 0x5A) or b == 0x5F
        is_digit = 0x30 <= b <= 0x39
        is_ws = b in (0x20, 0x09)
        if is_alpha:
            if state == 0:
                start = i
                state = 2
            elif state == 1:
                nodes.append(_WsNode([data[start:i]]))
                start = i
                state = 2
            elif state == 3:
                state = 2
        elif is_digit:
            if state == 0:
                start = i
                state = 3
            elif state == 1:
                nodes.append(_WsNode([data[start:i]]))
                start = i
                state = 3
            elif state == 2:
                pass
        elif is_ws:
            if state == 0:
                start = i
                state = 1
            elif state == 2:
                nodes.append(_AlphaNumNode([data[start:i]]))
                start = i
                state = 1
            elif state == 3:
                nodes.append(_NumNode([data[start:i]], hex=False))
                start = i
                state = 1
        else:
            if state == 1:
                nodes.append(_WsNode([data[start:i]]))
            elif state == 2:
                nodes.append(_AlphaNumNode([data[start:i]]))
            elif state == 3:
                nodes.append(_NumNode([data[start:i]], hex=False))
            state = 0
            nodes.append(_ControlNode(b))
        i += 1
    if state == 2:
        nodes.append(_AlphaNumNode([data[start:]]))
    elif state == 3:
        nodes.append(_NumNode([data[start:]], hex=False))
    return nodes


def _structure(nodes):
    nodes = _extract_numbers(nodes)
    nodes = _structure_brackets(nodes)
    nodes = _structure_keyvalue(nodes)
    nodes = _structure_lists(nodes)
    nodes = _structure_lines(nodes)
    return nodes


def _extract_numbers(nodes):
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(nodes):
            n = nodes[i]
            if isinstance(n, _AlphaNumNode) and len(n._samples) == 1:
                v = n._samples[0]
                if len(v) >= 3 and v[0:2] == b"0x" and _is_hex(v[2:]):
                    nodes[i] = _NumNode([v], hex=True)
                    changed = True
                    i += 1
                    continue
                e = v.find(b"e")
                if e != -1 and _is_dec(v[:e]) and _is_dec(v[e + 1 :]):
                    nodes[i] = _NumNode([v], hex=False)
                    changed = True
                    i += 1
                    continue
            if (
                isinstance(n, _ControlNode)
                and n._ch == 45
                and i + 1 < len(nodes)
                and isinstance(nodes[i + 1], _NumNode)
            ):
                num = nodes[i + 1]
                prev = nodes[i - 1] if i > 0 else None
                if (
                    not isinstance(prev, _AlphaNumNode)
                    or len(prev._samples[0]) <= 1
                    or prev._samples[0][-1:] != b"e"
                ):
                    num._samples = [b"-" + num._samples[0]]
                    nodes = nodes[:i] + nodes[i + 1 :]
                    changed = True
                    i += 1
                    continue
            i += 1
    return nodes


def _structure_brackets(nodes):
    stk = []
    i = 0
    while i < len(nodes):
        n = nodes[i]
        if isinstance(n, _ControlNode) and n._ch in _BRACKETS:
            stk.append((n._ch, _BRACKETS[n._ch], i))
        elif isinstance(n, _ControlNode):
            close = n._ch
            for si in range(len(stk) - 1, -1, -1):
                if stk[si][1] == close:
                    open_ch, _, pos = stk[si]
                    inner = _BlockNode(nodes[pos + 1 : i])
                    nodes[pos] = _BracketNode(open_ch, stk[si][1], inner)
                    nodes = nodes[: pos + 1] + nodes[i + 1 :]
                    i = pos
                    stk = stk[:si]
                    break
        i += 1
    return nodes


def _structure_keyvalue(nodes):
    delims = {0x3A, 0x3D}
    for n in nodes:
        if isinstance(n, _BracketNode):
            n._inner = _BlockNode(_structure_keyvalue(n._inner._nodes))
    i = 0
    while i < len(nodes):
        n = nodes[i]
        if (
            isinstance(n, _ControlNode)
            and n._ch in delims
            and i > 0
            and i < len(nodes) - 1
            and isinstance(nodes[i - 1], _AlphaNumNode)
            and isinstance(nodes[i + 1], _AlphaNumNode)
        ):
            nodes[i] = _KeyValNode(nodes[i - 1], nodes[i + 1])
            nodes = nodes[: i - 1] + nodes[i:]
            i += 1
            continue
        i += 1
    return nodes


def _structure_lists(nodes):
    delims = {0x2C, 0x3B}
    for n in nodes:
        if isinstance(n, _BracketNode):
            n._inner = _BlockNode(_structure_lists(n._inner._nodes))
    i = len(nodes) - 1
    while i >= 0:
        n = nodes[i]
        if isinstance(n, _ControlNode) and n._ch in delims:
            left = i - 1
            right = i + 1
            left_tokens = set()
            right_tokens = set()
            while True:
                left_done = left < 0
                right_done = right >= len(nodes)
                if left_done and right_done:
                    break
                if not left_done:
                    ctrl = nodes[left]
                    if isinstance(ctrl, _ControlNode):
                        if ctrl._ch == n._ch:
                            left_done = True
                        else:
                            left_tokens.add(ctrl._ch)
                    if isinstance(ctrl, _BracketNode):
                        left_tokens.add(ctrl._open[0])
                        left_tokens.add(ctrl._close[0])
                if not right_done:
                    ctrl = nodes[right]
                    if isinstance(ctrl, _ControlNode):
                        if ctrl._ch == n._ch:
                            right_done = True
                        else:
                            right_tokens.add(ctrl._ch)
                    if isinstance(ctrl, _BracketNode):
                        right_tokens.add(ctrl._open[0])
                        right_tokens.add(ctrl._close[0])
                if left_tokens == right_tokens:
                    break
                left -= 1
                right += 1
            # Simple list: collect elements between matching delimiters
            blocks = []
            j = left + 1
            while j < right:
                k = j
                while k < right and not (
                    isinstance(nodes[k], _ControlNode) and nodes[k]._ch == n._ch
                ):
                    k += 1
                blocks.append(_BlockNode(nodes[j:k]))
                j = k + 1
            if len(blocks) >= 2:
                nodes = nodes[: left + 1] + [_ListNode(n._ch, blocks)] + nodes[right:]
                i = left + 1
                continue
        i -= 1
    return nodes


def _structure_lines(nodes):
    res = []
    i = 0
    while i < len(nodes):
        if isinstance(nodes[i], _BracketNode):
            nodes[i]._inner = _BlockNode(_structure_lines(nodes[i]._inner._nodes))
            i += 1
            continue
        if isinstance(nodes[i], _ControlNode) and nodes[i]._ch == ord("\n"):
            crlf = (
                i > 0 and isinstance(nodes[i - 1], _ControlNode) and nodes[i - 1]._ch == ord("\r")
            )
            if crlf:
                res.append(_LineNode(True, _BlockNode(nodes[: i - 1])))
                nodes = nodes[i + 1 :]
            else:
                res.append(_LineNode(False, _BlockNode(nodes[:i])))
                nodes = nodes[i + 1 :]
            i = 0
            continue
        i += 1
    if nodes:
        res.extend(nodes)
    return res
