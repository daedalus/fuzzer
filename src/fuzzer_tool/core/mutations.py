"""Mutation operators and dictionary handling."""

import random
import re

INTERESTING_8 = [
    -128,   # Overflow signed 8-bit when decremented
    -1,
    0,
    1,
    16,     # One-off with common buffer size
    32,     # One-off with common buffer size
    64,     # One-off with common buffer size
    100,    # One-off with common buffer size
    127,    # Overflow signed 8-bit when incremented
]

INTERESTING_16 = [
    -32768,   # Overflow signed 16-bit when decremented
    -129,     # Overflow signed 8-bit
    128,      # Overflow signed 8-bit
    255,      # Overflow unsigned 8-bit when incremented
    256,      # Overflow unsigned 8-bit
    512,      # One-off with common buffer size
    1000,     # One-off with common buffer size
    1024,     # One-off with common buffer size
    4096,     # One-off with common buffer size
    32767,    # Overflow signed 16-bit when incremented
]

INTERESTING_32 = [
    -2147483648,  # Overflow signed 32-bit when decremented
    -100663046,   # Large negative number (endian-agnostic)
    -32769,       # Overflow signed 16-bit
    32768,        # Overflow signed 16-bit
    65535,        # Overflow unsigned 16-bit when incremented
    65536,        # Overflow unsigned 16-bit
    100663045,    # Large positive number (endian-agnostic)
    2139095040,   # Float infinity
    2147483647,   # Overflow signed 32-bit when incremented
]

ARITHMETIC_DELTAS = [1, 2, 4, 8, 16, 32, 64, 128]

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
    "swap_regions",
    "swap_bytes",
    "crc_fix",
    "endianness_swap",
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
