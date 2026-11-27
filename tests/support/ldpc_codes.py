"""Alist-text fixtures for ldpc_fix tests (rows are 0-based variable lists)."""

from __future__ import annotations

from enum import Enum

# (7,4) Hamming: column j+1 written in binary down the 3 checks.
HAMMING_ROWS = ((0, 2, 4, 6), (1, 2, 5, 6), (3, 4, 5, 6))
HAMMING_N = 7

# Hamming plus row0 ^ row1: rank 3 of 4 checks; checks 0, 1, 3 are dependent.
RANK_DEF_ROWS = (*HAMMING_ROWS, (0, 1, 4, 5))

# Two copies of one check: neither can fail alone.
TWIN_ROWS = ((0, 1), (0, 1))
TWIN_N = 4

GALLAGER_N = 20
GALLAGER_WC = 3
GALLAGER_WR = 4
# Band b > 0 relabels variables j -> (a*j + c) mod n; n must be coprime to 3 and 7.
_GALLAGER_PERMS = ((1, 0), (3, 1), (7, 5))


class Pad(Enum):
    """Whether column/row lists are zero-padded to the max weight."""

    NONE = 0
    ZEROS = 1


def gallager_rows(n: int = GALLAGER_N, wr: int = GALLAGER_WR) -> tuple[tuple[int, ...], ...]:
    """Regular (wc=3, wr) Gallager H: three bands of n // wr disjoint checks."""
    rows = []
    for a, c in _GALLAGER_PERMS[:GALLAGER_WC]:
        for i in range(n // wr):
            rows.append(tuple(sorted((a * (wr * i + t) + c) % n for t in range(wr))))
    return tuple(rows)


def alist_text(n: int, rows, pad: Pad = Pad.ZEROS) -> str:
    """MacKay alist for the H whose check r covers the 0-based variables rows[r]."""
    cols: list[list[int]] = [[] for _ in range(n)]
    for r, row in enumerate(rows):
        for j in row:
            cols[j].append(r + 1)

    max_c = max(len(c) for c in cols)
    max_r = max(len(r) for r in rows)

    def line(vals, width):
        fill = [0] * (width - len(vals)) if pad is Pad.ZEROS else []
        return " ".join(str(v) for v in [*vals, *fill])

    out = [f"{n} {len(rows)}", f"{max_c} {max_r}"]
    out.append(" ".join(str(len(c)) for c in cols))
    out.append(" ".join(str(len(r)) for r in rows))
    out += [line(c, max_c) for c in cols]
    out += [line([j + 1 for j in r], max_r) for r in rows]
    return "\n".join(out) + "\n"
