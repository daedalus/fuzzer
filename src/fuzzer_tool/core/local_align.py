"""Crash localization: Smith-Waterman-Gotoh local alignment, crash vs parent.

Levenshtein/Myers (``core/similarity.py``) give a global edit script; a
positional diff flags every byte after an insert. Local alignment with
affine gaps keeps an 8-byte insert one span and leaves the shifted tail
matched::

    parent  AAAA........BBBB          trim common prefix / suffix (numpy)
    crash   AAAA..INS.....BBBB   ──►  SW-Gotoh on the middle (row DP, numpy)
                  ^^^                 traceback ─► CIGAR, edit spans, deletions

Row DP, vectorized over the crash axis. The horizontal gap state E needs
the current row, so it is a prefix max (``np.maximum.accumulate``)::

    E[j] = GAP_OPEN + (j-1)*GAP_EXTEND + max_{k<j} (H0[k] - k*GAP_EXTEND)

One uint8 traceback cell per DP cell; a middle over ``MAX_CELLS`` falls
back to ``Fit.COARSE`` (the whole differing middle is one edit span).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Scores: +2/-3 with an affine gap -5 (first byte), -1 (each further byte).
MATCH = 2
MISMATCH = -3
GAP_OPEN = -5
GAP_EXTEND = -1

# Traceback cells: 4 MiB of uint8 (~2048 x 2048 middle), Hard Rule 54.
MAX_CELLS = 1 << 22

_NEG = -(1 << 30)

# Traceback byte: bits 0-1 = H source, bit 2 = E extends, bit 3 = F extends.
_H_ZERO = 0
_H_DIAG = 1
_H_GAP_E = 2
_H_GAP_F = 3
_H_MASK = 3
_E_SHIFT = 2
_F_SHIFT = 3
_E_EXT = 1 << _E_SHIFT
_F_EXT = 1 << _F_SHIFT

Span = tuple[int, int]


class Fit(Enum):
    """How the middle was aligned."""

    EXACT = "exact"  # full SW-Gotoh DP + traceback
    COARSE = "coarse"  # over MAX_CELLS: whole differing middle is one edit


class _Gap(Enum):
    """Traceback state: the DP matrix being walked."""

    H = "h"  # match / mismatch / local start
    E = "e"  # crash byte inserted
    F = "f"  # parent byte deleted


@dataclass(frozen=True, slots=True)
class Alignment:
    """Local alignment of a crash input against its parent.

    Attributes:
        score: Alignment score of ``cigar``.
        parent_span: Aligned region in the parent, half-open.
        crash_span: Aligned region in the crash input, half-open.
        cigar: Run-length ops over the regions: M match, X mismatch,
            I crash-only byte, D parent-only byte. E.g. ``"40M8I12M1X3M"``.
        edits: Merged half-open crash spans that are new or changed
            (mismatches, inserts, unaligned crash bytes).
        deletions: Crash offsets where parent bytes were dropped.
        fit: ``Fit.EXACT`` or ``Fit.COARSE``.
    """

    score: int
    parent_span: Span
    crash_span: Span
    cigar: str
    edits: tuple[Span, ...]
    deletions: tuple[int, ...]
    fit: Fit


def local_align(parent: bytes, crash: bytes) -> Alignment:
    """Locate the bytes of *crash* that differ from *parent*, indel-robust."""
    pre, suf = _affixes(parent, crash)
    a = parent[pre : len(parent) - suf]
    b = crash[pre : len(crash) - suf]

    if (len(a) + 1) * (len(b) + 1) > MAX_CELLS:
        return _coarse(a, b, pre)

    return _exact(a, b, pre, suf)


def span_text(al: Alignment) -> str:
    """One line: ``edits 0x64-0x6c, 0xfa; deleted @0x198``."""
    spans = ", ".join(f"0x{s:x}" if e - s == 1 else f"0x{s:x}-0x{e:x}" for s, e in al.edits)
    parts = [f"edits {spans}" if spans else "no edits"]
    if al.deletions:
        parts.append("deleted @" + ", @".join(f"0x{d:x}" for d in al.deletions))
    if al.fit is Fit.COARSE:
        parts.append("coarse")
    return "; ".join(parts)


def _affixes(a: bytes, b: bytes) -> tuple[int, int]:
    """Common prefix and (non-overlapping) common suffix lengths."""
    n = min(len(a), len(b))
    if n == 0:
        return 0, 0

    av = np.frombuffer(a, np.uint8)
    bv = np.frombuffer(b, np.uint8)
    diff = av[:n] != bv[:n]
    pre = int(diff.argmax()) if diff.any() else n

    rest = n - pre
    if rest == 0:
        return pre, 0
    tail = av[len(a) - rest :][::-1] != bv[len(b) - rest :][::-1]
    suf = int(tail.argmax()) if tail.any() else rest
    return pre, suf


def _coarse(a: bytes, b: bytes, pre: int) -> Alignment:
    """Over the cell cap: the whole differing middle is the edit."""
    edits = ((pre, pre + len(b)),) if b else ()
    dels = (pre,) if a and not b else ()
    cigar = f"{pre}M" if pre else ""
    return Alignment(MATCH * pre, (0, pre), (0, pre), cigar, edits, dels, Fit.COARSE)


def _exact(a: bytes, b: bytes, pre: int, suf: int) -> Alignment:
    """SW-Gotoh on the middle, then glue the matched affixes back on."""
    ptr, best, i1, j1 = _fill(a, b)
    ops, i0, j0 = _trace(ptr, a, b, i1, j1)

    # No local hit: anchor the empty core on the longer affix.
    if not ops and suf > pre:
        i0 = i1 = len(a)
        j0 = j1 = len(b)

    lead = pre if (i0, j0) == (0, 0) else 0
    trail = suf if (i1, j1) == (len(a), len(b)) else 0
    full = "M" * lead + "".join(ops) + "M" * trail

    edits, dels = _edits(ops, a, b, (i0, j0, i1, j1))
    return Alignment(
        score=best + MATCH * (lead + trail),
        parent_span=(pre + i0 - lead, pre + i1 + trail),
        crash_span=(pre + j0 - lead, pre + j1 + trail),
        cigar=_rle(full),
        edits=tuple((s + pre, e + pre) for s, e in edits),
        deletions=tuple(d + pre for d in dels),
        fit=Fit.EXACT,
    )


def _fill(a: bytes, b: bytes) -> tuple[np.ndarray, int, int, int]:
    """Gotoh DP; returns (traceback, best score, best row, best column)."""
    n, m = len(a), len(b)
    ptr = np.zeros((n + 1, m + 1), np.uint8)
    if n == 0 or m == 0:
        return ptr, 0, 0, 0

    # Substitution profile: one int8 row per parent byte (<= MAX_CELLS bytes).
    av = np.frombuffer(a, np.uint8)
    bv = np.frombuffer(b, np.uint8)
    prof = np.where(av[:, None] == bv[None, :], np.int8(MATCH), np.int8(MISMATCH))

    # Row buffers, reused in place: no per-row allocation.
    ramp = np.arange(m, dtype=np.int32) * GAP_EXTEND  # k * GAP_EXTEND, k < m
    h_prev = np.zeros(m + 1, np.int32)
    h_row = np.zeros(m + 1, np.int32)
    f = np.full(m, _NEG, np.int32)
    f_open, diag, h0, e, h, tmp = (np.empty(m, np.int32) for _ in range(6))
    f_bit = np.empty(m, bool)
    e_bit = np.zeros(m, bool)
    best_row = np.zeros(n + 1, np.int32)
    best_col = np.zeros(n + 1, np.int64)

    for i in range(1, n + 1):
        # F (vertical gap: parent byte deleted) from the row above.
        np.add(h_prev[1:], GAP_OPEN, out=f_open)
        f += GAP_EXTEND
        np.greater(f, f_open, out=f_bit)
        np.maximum(f, f_open, out=f)

        # H0: best of diagonal, F and the local-start floor 0.
        np.add(h_prev[:-1], prof[i - 1], out=diag)
        np.maximum(diag, f, out=h0)
        np.maximum(h0, 0, out=h0)

        # E (horizontal gap: crash byte inserted) as a prefix max over H0.
        h_row[1:] = h0
        np.subtract(h_row[:-1], ramp, out=tmp)
        np.maximum.accumulate(tmp, out=e)
        e += ramp
        e += GAP_OPEN
        np.maximum(h0, e, out=h)

        # Traceback byte: H source, then the E / F extend bits.
        row = ptr[i, 1:]
        row[:] = _H_GAP_F
        row[diag >= f] = _H_DIAG
        row[h0 == 0] = _H_ZERO
        row[e > h0] = _H_GAP_E
        np.greater(e[:-1] + GAP_EXTEND, h[:-1] + GAP_OPEN, out=e_bit[1:])
        row |= e_bit.view(np.uint8) << _E_SHIFT
        row |= f_bit.view(np.uint8) << _F_SHIFT

        col = int(h.argmax())
        best_col[i] = col
        best_row[i] = h[col]
        h_row[1:] = h
        h_prev, h_row = h_row, h_prev

    # First row holding the max, first column in it.
    bi = int(best_row.argmax())
    best = int(best_row[bi])
    if best == 0:
        return ptr, 0, 0, 0
    return ptr, best, bi, int(best_col[bi]) + 1


def _trace(ptr: np.ndarray, a: bytes, b: bytes, i: int, j: int) -> tuple[list[str], int, int]:
    """Walk back from (i, j) to the local start; ops in forward order."""
    ops: list[str] = []
    state = _Gap.H
    while True:
        cell = int(ptr[i, j])
        if state is _Gap.E:
            ops.append("I")
            state = _Gap.E if cell & _E_EXT else _Gap.H
            j -= 1
            continue
        if state is _Gap.F:
            ops.append("D")
            state = _Gap.F if cell & _F_EXT else _Gap.H
            i -= 1
            continue

        src = cell & _H_MASK
        if src == _H_ZERO:
            break
        if src == _H_GAP_E:
            state = _Gap.E
            continue
        if src == _H_GAP_F:
            state = _Gap.F
            continue
        ops.append("M" if a[i - 1] == b[j - 1] else "X")
        i, j = i - 1, j - 1

    ops.reverse()
    return ops, i, j


def _edits(
    ops: list[str], a: bytes, b: bytes, box: tuple[int, int, int, int]
) -> tuple[list[Span], list[int]]:
    """Crash-side edit spans and deletion offsets of the middle."""
    i0, j0, i1, j1 = box
    marks: list[int] = list(range(j0)) + list(range(j1, len(b)))  # unaligned crash bytes
    dels: list[int] = []

    # Parent flank with no crash flank opposite: a pure deletion.
    if i0 and not j0:
        dels.append(0)
    if i1 < len(a) and j1 == len(b):
        dels.append(j1)

    # Inside the core: X/I mark the crash byte, a D run marks one offset.
    j, last = j0, ""
    for op in ops:
        if op == "D":
            if last != "D":
                dels.append(j)
            last = op
            continue
        if op != "M":
            marks.append(j)
        j += 1
        last = op

    return _merge(sorted(marks)), sorted(set(dels))


def _merge(marks: list[int]) -> list[Span]:
    """Sorted offsets → merged half-open spans: [3,4,5,9] → [(3,6),(9,10)]."""
    spans: list[Span] = []
    for k in marks:
        if spans and spans[-1][1] == k:
            spans[-1] = (spans[-1][0], k + 1)
            continue
        spans.append((k, k + 1))
    return spans


def _rle(ops: str) -> str:
    """``"MMMII"`` → ``"3M2I"``."""
    out: list[str] = []
    k, n = 0, len(ops)
    while k < n:
        run = k
        while run < n and ops[run] == ops[k]:
            run += 1
        out.append(f"{run - k}{ops[k]}")
        k = run
    return "".join(out)
