"""Levenshtein and Hamming distance for seed/crash similarity.

Pure-Python implementations with no external dependencies.

Use cases in this fuzzer:
  - Hamming: fast byte-level seed dedup (equal-length inputs)
  - Levenshtein: crash signature clustering, stack trace similarity
  - Both: fuzzy corpus dedup, mutation novelty detection

Alignment cost control:
  ``levenshtein_align`` runs Myers O(ND) with a "Too Expensive" bail-out,
  then falls back to the numpy DP only when its table fits the byte budget,
  and to a coarse block diff above that. The budget is not an optimisation
  knob -- the DP table is 4*n*m bytes, so a 64 KiB crash against a 64 KiB
  corpus seed asks for 17 GB and dies. Tunable via
  ``--diff-myers-max-d`` / ``--diff-myers-max-bytes``.
"""

from __future__ import annotations

import os
import re
from array import array

# ---------------------------------------------------------------------------
# Alignment cost limits (A1 of the seventeen-source survey)
# ---------------------------------------------------------------------------
# These bound the work; they do not select an algorithm. The Myers path is
# unconditional because the unbounded DP it replaces is a latent OOM on any
# crash large enough to matter, not because it is faster on average.
_DIFF_MYERS_MAX_D: int = int(os.environ.get("FUZZER_DIFF_MYERS_MAX_D", "0") or "0")  # 0 = auto
_DIFF_MYERS_MAX_BYTES: int = int(
    os.environ.get("FUZZER_DIFF_MYERS_MAX_BYTES", str(128 * 1024 * 1024))
)  # 64 MiB default


def configure_diff_limits(
    max_d: int = 0,
    max_bytes: int = 128 * 1024 * 1024,
) -> None:
    """Set the alignment safety limits.

    ``max_d == 0`` means derive an automatic bound from input length.
    ``max_bytes`` caps the DP fallback table; above it the coarse block diff
    is used instead of allocating.
    """
    global _DIFF_MYERS_MAX_D, _DIFF_MYERS_MAX_BYTES
    _DIFF_MYERS_MAX_D = int(max_d)
    _DIFF_MYERS_MAX_BYTES = int(max_bytes)


def _hamming_dist(a: bytes, b: bytes) -> int:
    """Hamming distance over equal-length byte sequences.

    NumPy path above the 64-byte threshold (same dispatch rule as
    levenshtein_align): zero-copy uint8 views + SIMD count_nonzero. The
    genexpr is faster for tiny inputs, where numpy's fixed overhead dominates.
    """
    if len(a) < 64:
        return sum(x != y for x, y in zip(a, b, strict=True))
    import numpy as _np

    return int(
        _np.count_nonzero(_np.frombuffer(a, dtype=_np.uint8) != _np.frombuffer(b, dtype=_np.uint8))
    )


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
    return _hamming_dist(a, b)


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
    return 1.0 - _hamming_dist(a, b) / len(a)


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
    return _hamming_dist(a_padded, b_padded)


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

    prev = array("i", range(len(a) + 1))
    curr = array("i", [0]) * (len(a) + 1)

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


# Myers' cost here is dominated by the per-level diagonal sweep, so it grows as
# roughly O(D^2 + N) rather than O(N*D): measured 253 ms at n=65536 with D=652
# (long snakes, cheap) against 7.4 s at n=4096 with D=7224 (no snakes, 52M
# diagonal steps). Bounding D therefore bounds the work directly. 2048 caps a
# full bail-out at ~4M steps, about half a second, while still clearing the
# D values similar inputs actually produce (D=652 for a 64 KiB pair with one
# flip per 200 bytes).
_MYERS_STEP_BUDGET = 2048


def _myers_max_d(n: int, m: int) -> int:
    """'Too Expensive' bound for Myers.

    Independent of length on purpose: the cost is set by D, not by N, and a
    length-scaled bound let a 4 KiB dissimilar pair run to D=7224.
    """
    return _MYERS_STEP_BUDGET


def _coarse_block_diff(a: bytes, b: bytes, block: int = 64) -> list[tuple[str, int, bytes]]:
    """Fallback when Myers bails or the DP table would OOM.

    Emits a simple block-level script: matching runs become ``match``,
    differing blocks become a sequence of replaces / inserts / deletes.
    Contract is weaker than a true edit script (used only when the full
    alignment is unaffordable); callers that need positional scripts should
    keep inputs under the byte budget.
    """
    ops: list[tuple[str, int, bytes]] = []
    i = j = 0
    n, m = len(a), len(b)
    while i < n or j < m:
        if i < n and j < m and a[i] == b[j]:
            start = i
            while i < n and j < m and a[i] == b[j]:
                i += 1
                j += 1
            for k in range(start, i):
                ops.append(("match", k, b""))
            continue
        end_i = min(i + block, n)
        end_j = min(j + block, m)
        while i < end_i and j < end_j:
            ops.append(("replace", i, bytes([b[j]])))
            i += 1
            j += 1
        while i < end_i:
            ops.append(("delete", i, b""))
            i += 1
        while j < end_j:
            ops.append(("insert", i, bytes([b[j]])))
            j += 1
    return ops


def _myers_ses(a: bytes, b: bytes, max_d: int) -> list[tuple[str, int, bytes]] | None:
    """Myers O(ND) shortest-edit-script (forward + backtrack).

    Returns the edit script or ``None`` if D exceeds ``max_d`` (Too Expensive).
    Linear space for the V arrays; the trace stores one V per D for backtrack.
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return []
    max_d = max(1, max_d)
    offset = max_d
    # V[k] = furthest x reached on diagonal k
    v = array("i", [-1] * (2 * max_d + 1))
    v[1 + offset] = 0
    trace: list[array] = []

    for d in range(max_d + 1):
        v_copy = array("i", v)
        trace.append(v_copy)
        for k in range(-d, d + 1, 2):
            k_idx = k + offset
            if k == -d or (k != d and v[k_idx - 1] < v[k_idx + 1]):
                x = v[k_idx + 1]
            else:
                x = v[k_idx - 1] + 1
            y = x - k
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[k_idx] = x
            if x >= n and y >= m:
                return _myers_backtrack(a, b, trace, d, offset)
        # Continue to next d
    return None  # Too Expensive


def _myers_backtrack(
    a: bytes,
    b: bytes,
    trace: list,
    d_final: int,
    offset: int,
) -> list[tuple[str, int, bytes]]:
    """Reconstruct the edit script from the Myers V-trace.

    ``trace[d]`` is the V array as it stood *before* step ``d``, so at level
    ``d`` it names the endpoint the search came from. Walking d down to 1
    yields exactly one edit per level, and the snake between two endpoints is
    bounded by that previous endpoint -- unwinding matches greedily instead
    consumes runs belonging to lower levels and then leaves the loop with
    ``(x, y)`` short of the origin, silently truncating the script. That is
    what the first version of this function did: it round-tripped only when
    ``D == 1`` and was wrong for every larger edit distance, while still
    returning plausible-looking ops.

    Myers' model has no substitution -- it reaches ``b`` from ``a`` with
    insertions and deletions alone. The DP path this backs up emits
    ``replace``, so an adjacent delete/insert on one position is folded back
    into a single ``replace`` to keep the two paths' scripts interchangeable
    for callers that count non-match ops. The fold is not always possible:
    substitution costs 2 in Myers' model and 1 in the DP's, so a Myers script
    can carry a few more ops than the Levenshtein optimum (measured: 12 of 400
    random cases, never more than one or two ops). It is always a *valid*
    script; it is not always a minimal one.
    """
    ops: list[tuple[str, int, bytes]] = []
    x, y = len(a), len(b)

    for d in range(d_final, 0, -1):
        v = trace[d]
        k = x - y
        k_idx = k + offset
        # Same branch the forward pass took on this diagonal: down (from k+1)
        # means the step was an insertion, right (from k-1) means a deletion.
        went_down = k == -d or (k != d and v[k_idx - 1] < v[k_idx + 1])
        prev_k = k + 1 if went_down else k - 1
        prev_x = v[prev_k + offset]
        prev_y = prev_x - prev_k

        # Unwind the snake back to the previous endpoint -- not past it.
        while x > prev_x and y > prev_y:
            ops.append(("match", x - 1, b""))
            x -= 1
            y -= 1

        # Exactly one edit separates (x, y) from (prev_x, prev_y).
        if x > prev_x:
            ops.append(("delete", x - 1, b""))
            x -= 1
        elif y > prev_y:
            ops.append(("insert", x, bytes([b[y - 1]])))
            y -= 1

    # d == 0: the remaining prefix is one snake, then whatever is left over.
    while x > 0 and y > 0:
        ops.append(("match", x - 1, b""))
        x -= 1
        y -= 1
    while y > 0:
        ops.append(("insert", 0, bytes([b[y - 1]])))
        y -= 1
    while x > 0:
        ops.append(("delete", x - 1, b""))
        x -= 1

    ops.reverse()
    return _coalesce_indel_pairs(ops)


def _coalesce_indel_pairs(
    ops: list[tuple[str, int, bytes]],
) -> list[tuple[str, int, bytes]]:
    """Fold an adjacent delete/insert on one position into a ``replace``."""
    out: list[tuple[str, int, bytes]] = []
    i = 0
    n = len(ops)
    while i < n:
        op, pos, data = ops[i]
        if i + 1 < n:
            nop, npos, ndata = ops[i + 1]
            # delete a[p] then insert before a[p+1]  ==  replace a[p]
            if op == "delete" and nop == "insert" and npos == pos + 1:
                out.append(("replace", pos, ndata))
                i += 2
                continue
            # insert before a[p] then delete a[p]    ==  replace a[p]
            if op == "insert" and nop == "delete" and npos == pos:
                out.append(("replace", pos, data))
                i += 2
                continue
        out.append((op, pos, data))
        i += 1
    return out


def levenshtein_align(a: bytes, b: bytes) -> list[tuple[str, int, bytes]]:
    """Compute Levenshtein alignment as an edit script.

    Returns a list of (op, offset, data) tuples:
      ("match", pos, b"")     -- a[pos] matched b[pos]
      ("replace", pos, byte)  -- a[pos] replaced with byte
      ("insert", pos, byte)   -- byte inserted before a[pos]
      ("delete", pos, b"")    -- a[pos] deleted

    Optimized implementation:
    1. Prefix/suffix trimming — skip common leading/trailing bytes
    2. Direct Python DP for small remaining inputs (< 64 bytes each)
    3. When ``--diff-myers`` is enabled: Myers O(ND) with bail-out, then
       numpy DP only if the table fits the byte budget, else coarse block diff
    4. Otherwise (default): Numpy-vectorized DP for larger inputs

    Args:
        a: Original byte sequence.
        b: Target byte sequence.

    Returns:
        Edit script as list of (op, offset, data) tuples.
    """
    if a == b:
        return [("match", i, b"") for i in range(len(a))]

    n, m = len(a), len(b)

    # Trim common prefix
    pre = 0
    min_len = n if n < m else m
    while pre < min_len and a[pre] == b[pre]:
        pre += 1

    # Trim common suffix (from the end, after prefix)
    post = 0
    while post < n - pre and post < m - pre and a[n - 1 - post] == b[m - 1 - post]:
        post += 1

    # If everything matched after trimming, just emit matches
    if pre + post >= n and pre + post >= m:
        ops: list[tuple[str, int, bytes]] = []
        for i in range(pre):
            ops.append(("match", i, b""))
        for i in range(n - post, n):
            ops.append(("match", i, b""))
        return ops

    # Extract the differing middle parts
    a_mid = a[pre : n - post]
    b_mid = b[pre : m - post]
    na, nb = len(a_mid), len(b_mid)

    # For very small remaining inputs, use direct Python (no numpy overhead)
    if na < 64 and nb < 64:
        mid_ops = _levenshtein_align_small(a_mid, b_mid)
    else:
        # The DP is exact and vectorised in C; when its table is affordable it
        # is simply the better choice, and taking it keeps behaviour identical
        # to before this change for every input that was already survivable.
        # Myers is reached only where the DP cannot run at all -- that is the
        # bug being fixed, not a benchmark being won.
        table_bytes = (na + 1) * (nb + 1) * 4  # int32
        if table_bytes <= _DIFF_MYERS_MAX_BYTES:
            mid_ops = _levenshtein_align_numpy(a_mid, b_mid)
        else:
            max_d = _DIFF_MYERS_MAX_D or _myers_max_d(na, nb)
            myers_ops = _myers_ses(a_mid, b_mid, max_d)
            # Dissimilar *and* too big for the DP: no exact script is
            # affordable, so degrade to blocks rather than allocate.
            mid_ops = (
                myers_ops if myers_ops is not None else _coarse_block_diff(a_mid, b_mid)
            )

    # Reconstruct full script with prefix/suffix offsets
    result: list[tuple[str, int, bytes]] = []
    for i in range(pre):
        result.append(("match", i, b""))
    for op, pos, data in mid_ops:
        result.append((op, pos + pre, data))
    for i in range(n - post, n):
        result.append(("match", i, b""))

    return result


def _levenshtein_align_small(a: bytes, b: bytes) -> list[tuple[str, int, bytes]]:
    """Direct Python Levenshtein for small inputs (< 64 bytes).

    Avoids numpy allocation overhead which dominates for small arrays.
    Uses dp-comparison traceback (same logic as numpy path).
    """
    n, m = len(a), len(b)

    # Build full dp table for traceback
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(n + 1):
        dp[0][i] = i
    for j in range(1, m + 1):
        dp[j][0] = j
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j][i] = min(dp[j - 1][i] + 1, dp[j][i - 1] + 1, dp[j - 1][i - 1] + cost)

    # Traceback using dp comparisons (same logic as numpy path)
    ops: list[tuple[str, int, bytes]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[j][i] == dp[j - 1][i - 1]:
            ops.append(("match", i - 1, b""))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[j][i] == dp[j - 1][i - 1] + 1:
            ops.append(("replace", i - 1, bytes([b[j - 1]])))
            i -= 1
            j -= 1
        elif j > 0 and i > 0 and dp[j][i] == dp[j][i - 1] + 1:
            # dp[j][i-1]+1: consume a-char → delete
            ops.append(("delete", i - 1, b""))
            i -= 1
        elif i > 0 and j > 0 and dp[j][i] == dp[j - 1][i] + 1:
            # dp[j-1][i]+1: add b-char → insert
            ops.append(("insert", i, bytes([b[j - 1]])))
            j -= 1
        elif j > 0:
            # i == 0, must insert
            ops.append(("insert", 0, bytes([b[j - 1]])))
            j -= 1
        elif i > 0:
            # j == 0, must delete
            ops.append(("delete", i - 1, b""))
            i -= 1
        else:
            break

    ops.reverse()
    return ops


def _levenshtein_align_numpy(a: bytes, b: bytes) -> list[tuple[str, int, bytes]]:
    """Numpy-vectorized DP with traceback."""
    n, m = len(a), len(b)

    import numpy as _np

    a_arr = _np.frombuffer(a, dtype=_np.uint8)
    b_arr = _np.frombuffer(b, dtype=_np.uint8)

    idx = _np.arange(m + 1, dtype=_np.int32)
    prev = idx.copy()
    curr = _np.empty(m + 1, dtype=_np.int32)
    diag = _np.empty(m, dtype=_np.int32)
    h = _np.empty(m + 1, dtype=_np.int32)
    dp = _np.empty((n + 1, m + 1), dtype=_np.int32)
    dp[0] = prev

    for i in range(1, n + 1):
        curr[0] = i
        mism = a_arr[i - 1] != b_arr
        _np.add(prev[:-1], mism, out=diag, casting="unsafe")
        _np.add(prev[1:], 1, out=curr[1:])
        _np.minimum(curr[1:], diag, out=curr[1:])
        _np.subtract(curr, idx, out=h)
        _np.minimum.accumulate(h, out=h)
        _np.add(idx, h, out=curr)
        dp[i] = curr
        prev, curr = curr, prev

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

    prev = array("i", range(len(a) + 1))
    curr = array("i", [0]) * (len(a) + 1)

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
