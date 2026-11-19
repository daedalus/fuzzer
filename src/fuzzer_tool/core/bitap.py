"""Bitap near-miss repair: a token off by <= k bytes is overwritten with the exact token.

Aho-Corasick finds exact tokens only. A seed carrying ``IHxR`` where the
parser compares against ``IHDR`` fails that compare, and no exact scan sees
it. This finds the near miss and writes the exact token back::

    seed    .. 00 0d  I  H  x  R  00 ..      1 substitution from "IHDR"
    repair  .. 00 0d  I  H  D  R  00 ..

Bitap (Wu-Manber, k substitutions) keeps k+1 bit vectors over pattern
positions and advances them one text byte at a time. The shipped scan is the
same recurrence transposed: one mismatch counter per text position, advanced
one *pattern* byte at a time with numpy. 9-byte token, k=2, identical hits:

    text      scalar bitap   numpy
    256 B         140 us     32 us
    4 KiB        2400 us     45 us
    64 KiB      38810 us    142 us

A pigeonhole filter goes first: k errors leave one of k+1 token pieces
exact, so C ``bytes.find`` yields the candidates. Interleaved min-of-9 vs
the scan alone: random input 64 B-1 KiB 4-15x faster, 32-byte tokens up to
16x at any size. Short pieces past PIGEON_MAX_LEN lose (find rescans) and
keep the scan. Zero-filled input with a run piece (``00 00``) goes to the
scan; the gate costs +0.5 us there (<= 4.5%, inside run-to-run noise).

The scalar form is the test oracle (``tests/test_bitap.py``).
"""

from __future__ import annotations

import functools

import numpy as np

# Substitutions tolerated: one per MIN_REPAIR_LEN token bytes, at most MAX_ERRORS.
# A 4-byte token at k=1 still needs 3 matching bytes (chance ~1/2^22 per offset).
MAX_ERRORS = 2
MIN_REPAIR_LEN = 4
# Tokens scanned per call, from a random start: bounds the per-mutation cost.
MAX_REPAIR_PROBES = 8
# Pigeonhole filter: used up to this input size, or past it with pieces this
# long; dense input (more candidates than this) falls back to the scan.
PIGEON_MAX_LEN = 2048
PIGEON_MIN_PIECE = 8
PIGEON_MAX_CANDIDATES = 4
# Tokens whose pieces are cached (dictionary tokens repeat across calls).
PIECE_CACHE = 1024


def max_errors(m: int) -> int:
    """Substitutions allowed for an ``m``-byte token: 4..7 -> 1, 8+ -> 2."""
    return min(MAX_ERRORS, m // MIN_REPAIR_LEN)


def near_matches(data: bytes, token: bytes, k: int) -> np.ndarray:
    """Start offsets where ``token`` occurs with 1..k substitutions (exact excluded)."""
    m = len(token)
    if m == 0 or len(data) < m:
        return np.empty(0, np.intp)

    if len(data) <= PIGEON_MAX_LEN or m // (k + 1) >= PIGEON_MIN_PIECE:
        pieces = _pieces(token, k)
        hits = None if pieces is None else _pigeon(data, token, k, pieces)
        if hits is not None:
            return hits
    return _scan(data, token, k)


@functools.lru_cache(maxsize=PIECE_CACHE)
def _pieces(token: bytes, k: int) -> tuple[tuple[int, bytes], ...] | None:
    """k+1 (offset, piece) cuts of ``token``; None when one is a single-byte run.

    A run (``00 00``) matches everywhere in zero-filled input: candidates
    would only be thrown away.
    """
    size = len(token) // (k + 1)
    if size == 0:
        return None
    cuts = tuple(
        (j * size, token[j * size : len(token) if j == k else (j + 1) * size]) for j in range(k + 1)
    )
    if any(p.count(p[:1]) == len(p) for _, p in cuts):
        return None
    return cuts


def _pigeon(data: bytes, token: bytes, k: int, pieces) -> np.ndarray | None:
    """Filter-then-verify; None past PIGEON_MAX_CANDIDATES (dense: scan instead).

    k substitutions hit at most k of the k+1 pieces, so one matches exactly
    and C ``bytes.find`` locates every candidate start.
    """
    m = len(token)
    last = len(data) - m
    cands: set[int] = set()
    for lo, piece in pieces:
        i = data.find(piece)
        while i >= 0:
            if 0 <= i - lo <= last:
                cands.add(i - lo)
                if len(cands) > PIGEON_MAX_CANDIDATES:
                    return None
            i = data.find(piece, i + 1)

    hits = []
    for st in sorted(cands):
        miss = sum(a != b for a, b in zip(data[st : st + m], token, strict=True))
        if 0 < miss <= k:
            hits.append(st)
    return np.array(hits, np.intp)


def _scan(data: bytes, token: bytes, k: int) -> np.ndarray:
    """Every offset: mismatch counters, one numpy pass per token byte."""
    m = len(token)
    n = len(data) - m + 1
    text = np.frombuffer(data, np.uint8)
    miss = np.zeros(n, np.uint8 if m < 256 else np.uint32)
    for j, c in enumerate(token):
        miss += text[j : j + n] != c
    return np.flatnonzero((miss > 0) & (miss <= k))


def token_repair(data: bytes, tokens, start: int, *, rng) -> bytes | None:
    """Overwrite a near miss of one of ``tokens`` with the exact token.

    Probes up to MAX_REPAIR_PROBES tokens from a random index (wrapping); the
    first with a near miss wins. The hit at/after ``start`` is taken, else the
    first one.

    Args:
        data: Input bytes.
        tokens: Literals (dictionary / cmplog tokens; str or bytes).
        start: Planned mutation site.
        rng: RNG (one ``randint``).

    Returns:
        Repaired bytes (same length), or None when no probed token is near.
    """
    n = len(tokens)
    if not n:
        return None
    first = rng.randint(0, n - 1)
    for i in range(min(n, MAX_REPAIR_PROBES)):
        tok = tokens[(first + i) % n]
        if isinstance(tok, str):
            tok = tok.encode()
        if len(tok) < MIN_REPAIR_LEN:
            continue

        hits = near_matches(data, tok, max_errors(len(tok)))
        if not hits.size:
            continue

        at = int(np.searchsorted(hits, start))
        pos = int(hits[at] if at < hits.size else hits[0])
        return data[:pos] + tok + data[pos + len(tok) :]
    return None
