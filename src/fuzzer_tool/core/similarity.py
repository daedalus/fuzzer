"""Levenshtein and Hamming distance for seed/crash similarity.

Pure-Python implementations with no external dependencies.

Use cases in this fuzzer:
  - Hamming: fast byte-level seed dedup (equal-length inputs)
  - Levenshtein: crash signature clustering, stack trace similarity
  - Both: fuzzy corpus dedup, mutation novelty detection
"""

import re


def hamming_distance(a: bytes, b: bytes) -> int:
    """Hamming distance between two equal-length byte sequences.

    Counts positions where bytes differ. Raises ValueError if lengths differ
    (caller should pad or use Levenshtein for unequal lengths).

    Args:
        a: First byte sequence.
        b: Second byte sequence (must be same length as a).

    Returns:
        Number of differing byte positions.

    Raises:
        ValueError: If a and b have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"Hamming distance requires equal lengths: got {len(a)} and {len(b)}")
    return sum(x != y for x, y in zip(a, b, strict=True))


def hamming_similarity(a: bytes, b: bytes) -> float:
    """Normalized Hamming similarity in [0.0, 1.0].

    1.0 = identical, 0.0 = all bytes differ. For unequal lengths, returns
    0.0 (caller should use Levenshtein instead).

    Args:
        a: First byte sequence.
        b: Second byte sequence.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dist = sum(x != y for x, y in zip(a, b, strict=True))
    return 1.0 - dist / len(a)


def hamming_distance_padded(a: bytes, b: bytes) -> int:
    """Hamming distance with zero-padding for unequal lengths.

    The shorter sequence is conceptually right-padded with zeros.

    Args:
        a: First byte sequence.
        b: Second byte sequence.

    Returns:
        Number of differing byte positions (shorter is zero-padded).
    """
    max_len = max(len(a), len(b))
    a_padded = a + b"\x00" * (max_len - len(a))
    b_padded = b + b"\x00" * (max_len - len(b))
    return sum(x != y for x, y in zip(a_padded, b_padded, strict=True))


def levenshtein_distance(a: bytes, b: bytes) -> int:
    """Levenshtein edit distance between two byte sequences.

    Uses the standard two-row DP algorithm. O(len(a) * len(b)) time,
    O(min(len(a), len(b))) space.

    Args:
        a: First byte sequence.
        b: Second byte sequence.

    Returns:
        Minimum number of insertions, deletions, or substitutions.
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Optimize: ensure a is the shorter sequence for space
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    curr = [0] * (len(a) + 1)

    for j in range(1, len(b) + 1):
        curr[0] = j
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(
                prev[i] + 1,  # deletion
                curr[i - 1] + 1,  # insertion
                prev[i - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[len(a)]


def levenshtein_similarity(a: bytes, b: bytes) -> float:
    """Normalized Levenshtein similarity in [0.0, 1.0].

    1.0 = identical, 0.0 = completely different. Normalized by the
    length of the longer sequence.

    Args:
        a: First byte sequence.
        b: Second byte sequence.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - dist / max_len if max_len > 0 else 1.0


def stack_trace_similarity(frames_a: list[str], frames_b: list[str]) -> float:
    """Levenshtein-based similarity between two stack traces.

    Joins frame names into a single string and computes Levenshtein
    similarity. This groups crashes that hit the same code paths with
    minor variations (e.g. different inlined frames, different addresses).

    Args:
        frames_a: Stack frame function names from crash A.
        frames_b: Stack frame function names from crash B.

    Returns:
        Similarity in [0.0, 1.0].
    """
    joined_a = "@".join(frames_a[:8])
    joined_b = "@".join(frames_b[:8])
    return levenshtein_similarity(joined_a.encode(), joined_b.encode())


# Strip addresses and numbers from stack frames for coarser grouping
_ADDR_RE = re.compile(r"0x[0-9a-f]+")
_NUM_RE = re.compile(r"\b\d+\b")


def normalize_frame(frame: str) -> str:
    """Normalize a stack frame by stripping addresses and numbers.

    Converts ``parse+0x1234`` to ``parse+``, ``func.c:42`` to ``func.c:``,
    etc. This makes Levenshtein comparison more meaningful for grouping
    crashes with the same root cause but different instruction offsets.
    """
    s = _ADDR_RE.sub("", frame)
    s = _NUM_RE.sub("", s)
    return s.strip()


def crash_signature_similarity(sig_a: str, sig_b: str) -> float:
    """Levenshtein-based similarity between two crash signatures.

    Strips addresses and numeric offsets before comparing, so crashes
    at the same function with different instruction offsets are grouped.

    Args:
        sig_a: Crash signature string (e.g. "ASAN:heap-buffer-overflow@parse@main").
        sig_b: Crash signature string.

    Returns:
        Similarity in [0.0, 1.0].
    """
    norm_a = normalize_frame(sig_a).encode()
    norm_b = normalize_frame(sig_b).encode()
    return levenshtein_similarity(norm_a, norm_b)


def levenshtein_align(a: bytes, b: bytes) -> list[tuple[str, int, bytes]]:
    """Compute Levenshtein alignment as an edit script.

    Returns a list of (op, offset, data) tuples:
      ("match", pos, b"")     -- a[pos] matched b[pos]
      ("replace", pos, byte)  -- a[pos] replaced with byte
      ("insert", pos, byte)   -- byte inserted before a[pos]
      ("delete", pos, b"")    -- a[pos] deleted

    Uses the full DP table for traceback. O(n*m) time, O(n*m) space.

    Args:
        a: Original byte sequence.
        b: Target byte sequence.

    Returns:
        Edit script as list of (op, offset, data) tuples.
    """
    if a == b:
        return [("match", i, b"") for i in range(len(a))]

    n, m = len(a), len(b)

    # Build full DP table for traceback
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # delete from a
                dp[i][j - 1] + 1,  # insert into a
                dp[i - 1][j - 1] + cost,  # replace or match
            )

    # Traceback
    ops: list[tuple[str, int, bytes]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", i - 1, b""))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("replace", i - 1, bytes([b[j - 1]])))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("insert", i, bytes([b[j - 1]])))
            j -= 1
        elif i > 0:
            ops.append(("delete", i - 1, b""))
            i -= 1
        else:
            break

    ops.reverse()
    return ops


def edit_script_summary(a: bytes, b: bytes) -> str:
    """Human-readable summary of the edit distance between two byte sequences.

    Returns a concise description like "3 substitutions, 1 insertion at offset 42"
    or "identical".

    Args:
        a: Original byte sequence.
        b: Target byte sequence.

    Returns:
        Human-readable edit description.
    """
    if a == b:
        return "identical"

    script = levenshtein_align(a, b)

    replaces = []
    inserts = []
    deletes = []
    for op, pos, data in script:
        if op == "replace":
            replaces.append(pos)
        elif op == "insert":
            inserts.append((pos, data))
        elif op == "delete":
            deletes.append(pos)

    parts = []
    if replaces:
        if len(replaces) <= 3:
            offsets = ", ".join(f"0x{o:02x}" for o in replaces)
            parts.append(f"{len(replaces)} substitution(s) at offset [{offsets}]")
        else:
            parts.append(f"{len(replaces)} substitutions")
    if inserts:
        if len(inserts) <= 3:
            offsets = ", ".join(f"0x{pos:02x}" for pos, _ in inserts)
            parts.append(f"{len(inserts)} insertion(s) at offset [{offsets}]")
        else:
            parts.append(f"{len(inserts)} insertions")
    if deletes:
        if len(deletes) <= 3:
            offsets = ", ".join(f"0x{o:02x}" for o in deletes)
            parts.append(f"{len(deletes)} deletion(s) at offset [{offsets}]")
        else:
            parts.append(f"{len(deletes)} deletions")

    return "; ".join(parts) if parts else "no edit ops"


def levenshtein_diff_offsets(a: bytes, b: bytes, max_ops: int = 30) -> list[int]:
    """Compute Levenshtein-aligned diff offsets between two byte sequences.

    Unlike the naive positional diff (which misaligns after insertions/deletions),
    this produces the actual edit positions by running Levenshtein alignment.
    Returns a list of byte positions where the sequences differ, in order.

    Args:
        a: Original byte sequence.
        b: Target byte sequence.
        max_ops: Maximum number of offsets to return.

    Returns:
        List of byte positions where edits occurred.
    """
    if a == b:
        return []

    script = levenshtein_align(a, b)
    offsets = []
    for op, pos, _data in script:
        if op != "match":
            offsets.append(pos)
            if len(offsets) >= max_ops:
                break
    return offsets


def _levenshtein_tokens(a: list[str], b: list[str]) -> int:
    """Levenshtein distance on token sequences (list of strings).

    O(len(a) * len(b)) time, O(min(len(a), len(b))) space.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    curr = [0] * (len(a) + 1)

    for j in range(1, len(b) + 1):
        curr[0] = j
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(
                prev[i] + 1,
                curr[i - 1] + 1,
                prev[i - 1] + cost,
            )
        prev, curr = curr, prev

    return prev[len(a)]


def frame_sequence_similarity(frames_a: list[str], frames_b: list[str]) -> float:
    """Levenshtein similarity on frame sequences (order-aware, token-level).

    Unlike Jaccard on frame sets (which discards call order), this
    correctly distinguishes A->B->C from C->B->A while still tolerating
    one extra inlined frame (a single token insertion gives small edit
    distance, not the byte-level explosion that joined-string Levenshtein produces).

    Args:
        frames_a: Stack frame names from crash A (in call order).
        frames_b: Stack frame names from crash B (in call order).

    Returns:
        Similarity in [0.0, 1.0].
    """
    norm_a = [normalize_frame(f) for f in frames_a[:8]]
    norm_b = [normalize_frame(f) for f in frames_b[:8]]

    if not norm_a and not norm_b:
        return 1.0
    dist = _levenshtein_tokens(norm_a, norm_b)
    max_len = max(len(norm_a), len(norm_b))
    return 1.0 - dist / max_len if max_len > 0 else 1.0


def find_nearest_bytes(
    target: bytes,
    candidates: list[bytes],
    max_check: int = 100,
) -> tuple[int, float]:
    """Find the candidate most similar to target using Hamming + Levenshtein.

    For equal-length candidates, uses Hamming distance (fast).
    For unequal lengths, uses Levenshtein similarity.

    Args:
        target: The byte sequence to match.
        candidates: List of candidate byte sequences.
        max_check: Maximum number of candidates to check.

    Returns:
        Tuple of (best_index, similarity). best_index=-1 if no candidates.
    """
    if not candidates:
        return -1, 0.0

    best_idx = 0
    best_sim = 0.0

    for idx, cand in enumerate(candidates[:max_check]):
        if len(target) == len(cand):
            sim = hamming_similarity(target, cand)
        else:
            sim = levenshtein_similarity(target, cand)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    return best_idx, best_sim
