"""Mutation operators and dictionary handling."""

import random
import re

INTERESTING_8 = [
    -128,  # Overflow signed 8-bit when decremented
    -1,
    0,
    1,
    16,  # One-off with common buffer size
    32,  # One-off with common buffer size
    64,  # One-off with common buffer size
    100,  # One-off with common buffer size
    127,  # Overflow signed 8-bit when incremented
]

INTERESTING_16 = [
    -32768,  # Overflow signed 16-bit when decremented
    -129,  # Overflow signed 8-bit
    128,  # Overflow signed 8-bit
    255,  # Overflow unsigned 8-bit when incremented
    256,  # Overflow unsigned 8-bit
    512,  # One-off with common buffer size
    1000,  # One-off with common buffer size
    1024,  # One-off with common buffer size
    4096,  # One-off with common buffer size
    32767,  # Overflow signed 16-bit when incremented
]

INTERESTING_32 = [
    -2147483648,  # Overflow signed 32-bit when decremented
    -100663046,  # Large negative number (endian-agnostic)
    -32769,  # Overflow signed 16-bit
    32768,  # Overflow signed 16-bit
    65535,  # Overflow unsigned 16-bit when incremented
    65536,  # Overflow unsigned 16-bit
    100663045,  # Large positive number (endian-agnostic)
    2139095040,  # Float infinity
    2147483647,  # Overflow signed 32-bit when incremented
]

INTERESTING_UNSIGNED_8 = [
    0, 1, 2, 3, 4, 5,  # Small values — trigger len < nlen underflows
    0xFE, 0xFF,  # Near unsigned 8-bit max
]

INTERESTING_UNSIGNED_16 = [
    0, 1, 2, 3, 4, 5,
    0xFFFE, 0xFFFF,  # Unsigned 16-bit max
    0x7FFE, 0x7FFF,  # Near signed 16-bit max
]

INTERESTING_UNSIGNED_32 = [
    0, 1, 2, 3, 4, 5,
    0xFFFFFFFE, 0xFFFFFFFF,  # Unsigned 32-bit max (SIZE_MAX on 32-bit)
    0x7FFFFFFE, 0x7FFFFFFF,  # Near signed 32-bit max
    0x100, 0x400, 0x1000,  # Common buffer boundaries
]

LENGTH_BOUNDARIES = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024, 4096]

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
    "simd_boundary",
    "regex_bomb",
    "clone_fixed",
    "overwrite_copy",
    "overwrite_fixed",
    "redqueen_xform",
    "skipdet_probe",
    "auto_extras",
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
]


def splice(a: bytes, b: bytes) -> bytes:
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
    cut_a = random.randint(1, len(a) - 1)
    cut_b = random.randint(1, len(b) - 1)
    return a[:cut_a] + b[cut_b:]


def crossover(a: bytes, b: bytes) -> bytes:
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
    cut1 = random.randint(1, len(a) - 3)
    cut2 = random.randint(cut1 + 1, len(a) - 1)
    seg_len = cut2 - cut1
    b_start = random.randint(0, max(0, len(b) - seg_len))
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


_HEX_ESCAPE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")


def parse_dict_line(line: str) -> bytes | None:
    """Parse a single dictionary line.

    Handles ``NAME=value`` format and ``\\x??`` hex escapes (like AFL).
    Literal backslash-x followed by exactly two hex digits is decoded;
    everything else is encoded as raw UTF-8.

    Args:
        line: Raw line from dictionary file.

    Returns:
        Parsed token bytes, or None if line is empty/comment.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("=", 1)
    token = parts[-1] if len(parts) == 2 else line
    result = bytearray()
    last = 0
    for m in _HEX_ESCAPE_RE.finditer(token):
        result.extend(token[last : m.start()].encode("utf-8"))
        result.append(int(m.group(1), 16))
        last = m.end()
    result.extend(token[last:].encode("utf-8"))
    return bytes(result)


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


def type_replace_byte(b: int) -> int:
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
    rng = _in_class(b)
    if rng:
        start, end = rng
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
            c = start + random.randint(0, size)
        return c

    # Default: XOR to flip bits while staying in printable-ish range
    if b < 32:
        return b ^ 0x1F
    return b ^ 0x7F


def type_replace(data: bytes) -> bytes:
    """Replace all bytes with different values from the same character class.

    Optimized: inlined logic with precomputed swap map, no per-byte function calls.

    Args:
        data: Input bytes to mutate.

    Returns:
        Mutated bytes with each byte replaced within its class.
    """
    result = bytearray(data)
    for i in range(len(result)):
        b = result[i]
        if b in _SWAP_MAP:
            result[i] = _SWAP_MAP[b]
        elif 0x30 <= b <= 0x39:  # digit
            result[i] = 0x30 + (b * 7 + 3) % 10
        elif 0x41 <= b <= 0x5A:  # upper
            result[i] = 0x41 + (b * 13 + 5) % 26
        elif 0x61 <= b <= 0x7A:  # lower
            result[i] = 0x61 + (b * 17 + 7) % 26
        elif b < 32:
            result[i] = b ^ 0x1F
        else:
            result[i] = b ^ 0x7F
    return bytes(result)


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

    if v in (0xFF, 0xFFFF, 0xFFFFFFFF):
        return True

    return False


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

    if diffs == 1:
        if ((ov - nv) & 0xFF) <= ARITH_MAX or ((nv - ov) & 0xFF) <= ARITH_MAX:
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
    if blen == 4:
        if ((old_val - new_val) & 0xFFFFFFFF) <= ARITH_MAX or (
            (new_val - old_val) & 0xFFFFFFFF
        ) <= ARITH_MAX:
            return True

    return False


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


def ascii_num_replace(data: bytes) -> bytes:
    """Replace a random position with an ASCII number string.

    Picks a random position and replaces a short segment with a random
    ASCII decimal number (0-99999). Useful for fuzzing numeric fields
    in text-based formats.

    Args:
        data: Input bytes.

    Returns:
        Mutated bytes with an ASCII number inserted.
    """
    if not data:
        return data

    result = bytearray(data)
    idx = random.randint(0, len(result) - 1)

    # Generate a random number as ASCII digits
    num = random.randint(0, 99999)
    num_str = str(num).encode("ascii")

    # Replace at position (truncate if near end)
    end = min(idx + len(num_str), len(result))
    result[idx:end] = num_str[: end - idx]

    return bytes(result)


def insert_ascii_num(data: bytes, max_len: int = 65536) -> bytes:
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

    idx = random.randint(0, len(data))
    num = random.randint(0, 99999)
    num_str = str(num).encode("ascii")
    result = data[:idx] + num_str + data[idx:]
    return result[:max_len]


def byte_shuffle(data: bytes) -> bytes:
    """Shuffle a random subset of bytes in the input.

    Optimized: shuffle only a random portion instead of the entire buffer.

    Args:
        data: Input bytes.

    Returns:
        Partially shuffled bytes.
    """
    if len(data) <= 1:
        return data
    result = bytearray(data)
    # Shuffle only a random 20-50% subset
    n = max(2, len(result) // random.randint(2, 5))
    start = random.randint(0, max(0, len(result) - n))
    random.shuffle(result[start:start + n])
    return bytes(result)


def byte_delete(data: bytes) -> bytes:
    """Delete a single random byte from the input.

    Args:
        data: Input bytes.

    Returns:
        Bytes with one byte removed, or original if too short.
    """
    if len(data) <= 1:
        return data

    idx = random.randint(0, len(data) - 1)
    return data[:idx] + data[idx + 1 :]


def byte_insert(data: bytes, max_len: int = 65536) -> bytes:
    """Insert a single random byte at a random position.

    Args:
        data: Input bytes.
        max_len: Maximum output length.

    Returns:
        Bytes with one random byte inserted.
    """
    if len(data) >= max_len:
        return data

    idx = random.randint(0, len(data))
    val = random.randint(0, 255)
    return data[:idx] + bytes([val]) + data[idx:]


def splice_diff_located(a: bytes, b: bytes) -> bytes:
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
        cut_a = random.randint(1, len(a) - 1)
        cut_b = random.randint(1, len(b) - 1)
        return a[:cut_a] + b[cut_b:]

    # Pick cut points within the diff range
    cut_a = random.randint(first_diff, last_diff)
    cut_b = random.randint(first_diff, min(last_diff, len(b) - 1))

    return a[:cut_a] + b[cut_b:]


# ---------------------------------------------------------------------------
# Block transposition mutations
# ---------------------------------------------------------------------------


def transpose_bytes(data: bytes, width: int) -> bytes:
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
    start = (random.randint(0, max_start) // width) * width
    block = bytearray(data[start : start + width])
    random.shuffle(block)
    result = bytearray(data)
    result[start : start + width] = block
    return bytes(result)


def bit_transpose(data: bytes, width: int) -> bytes:
    """Permute bits within a randomly-selected block of *width* bytes.

    Optimized: swaps random bit pairs instead of full shuffle.

    Args:
        data: Input bytes.
        width: Block width in bytes (1, 2, 4, or 8).

    Returns:
        Bytes with one block's bits transposed.
    """
    if len(data) < width:
        return data
    max_start = len(data) - width
    start = (random.randint(0, max_start) // width) * width
    val = int.from_bytes(data[start : start + width], "little")
    total_bits = 8 * width
    # Swap 2-4 random bit pairs instead of full shuffle
    n_swaps = random.randint(2, min(4, total_bits // 2))
    for _ in range(n_swaps):
        i = random.randint(0, total_bits - 1)
        j = random.randint(0, total_bits - 1)
        if i != j:
            bi = (val >> i) & 1
            bj = (val >> j) & 1
            if bi != bj:
                val ^= (1 << i) | (1 << j)
    result = bytearray(data)
    result[start : start + width] = val.to_bytes(width, "little")
    return bytes(result)
