"""LDPC parity repair and error injection (``--ldpc-alist``).

A target that decodes LDPC-protected data (DVB-S2, Wi-Fi, 5G NR, storage
controllers) either rejects a mutated word at the parity check or decodes
it back with belief propagation / bit flipping, so plain bit flips rarely
reach the code behind the decoder.  LDPC streams carry no magic, so this
operator is gated on a parity-check matrix H given as a MacKay alist file.

The buffer is a bitstream (MSB first in each byte) of back-to-back n-bit
codewords; trailing bits that do not fill a word are left alone::

    bit 0                 n                2n           8*len
    | codeword 0          | codeword 1     | ... | tail |

On one codeword this does one of (``Mode``):

* REPAIR: flip 1..MAX_DATA_EDITS information bits, re-encode the parity so
  H.c = 0 — the edit reaches the parser behind the decoder;
* CORRECTABLE: re-encode, flip 1..t_guess bits.  LDPC has no hard
  correction radius; t_guess = (min column weight + 1) // 2 (half the
  Tanner bound on dmin) aims at the BP / bit-flipping convergence path;
* UNCORRECTABLE: re-encode, flip every variable of one check (a
  stopping-set-like cluster) — the decoder's failure / max-iteration path;
* SYNDROME: re-encode, flip parity bits so exactly one check fails —
  early-termination logic.  If H's rank deficiency makes every single-check
  syndrome unreachable, one variable is flipped instead.

The encoder is systematic: GF(2) Gauss-Jordan on H picks pivot columns
(parity positions) and free columns (information positions), so it works
for rank-deficient H (k = n - rank).  Row masks put variable j at bit
n-1-j, so a mask reads left to right like the stream.  Length never
changes; if an edit would echo the input, the re-encoded word is emitted.

RNG draw order: codeword index, mode, then
REPAIR: edit count, one index per edit (sparse Fisher-Yates over info bits);
CORRECTABLE: flip count, one index per flip (over all n bits);
UNCORRECTABLE: live-check index; SYNDROME: feasible-check index, or a
variable index when no check is feasible.  Stdlib only (Hard Rule 51).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

#: Bound on n and m: the dense RREF encoder costs rank * n bits.
MAX_N = 8192
MAX_DATA_EDITS = 4
_MAX_CORRECTABLE = 8
_BYTE = 8
_HEADER_TOKENS = 4


class Mode(IntEnum):
    """LDPC edit; values are the RNG draw."""

    REPAIR = 0
    CORRECTABLE = 1
    UNCORRECTABLE = 2
    SYNDROME = 3


@dataclass(frozen=True)
class LdpcCode:
    """Parity-check matrix plus a precomputed systematic encoder.

    Attributes:
        n: Codeword bits.
        m: Checks (rows of H).
        k: Information bits, n - rank.
        rank: GF(2) rank of H.
        rows: Check r as a mask, variable j at bit n-1-j.
        pivots: Parity positions, one per RREF row.
        free: Information positions, ascending.
        feasible: Checks that some error pattern makes fail alone.
        live: Checks with at least one variable.
        t_guess: CORRECTABLE flip cap.
    """

    n: int
    m: int
    k: int
    rank: int
    rows: tuple[int, ...]
    pivots: tuple[int, ...]
    free: tuple[int, ...]
    feasible: tuple[int, ...]
    live: tuple[int, ...]
    t_guess: int
    _basis: tuple[int, ...] = field(repr=False)
    _trans: tuple[int, ...] = field(repr=False)
    _pivot_mask: int = field(repr=False)


# --- construction -------------------------------------------------------------


def _eliminate(rows: list[int]) -> tuple[list[int], list[int], int]:
    """Incremental Gauss-Jordan: (RREF rows, transforms, dependent-check mask).

    Invariant: each basis row holds exactly one pivot bit (its leading bit),
    so reducing a new row XORs one basis row per pivot bit it carries.  A
    transform bit r means original check r went into that row.
    """
    basis: list[int] = []
    trans: list[int] = []
    where: dict[int, int] = {}
    pivot_mask = deps = 0

    for r, row in enumerate(rows):
        t = 1 << r
        hit = row & pivot_mask
        while hit:
            lb = hit.bit_length() - 1
            hit ^= 1 << lb
            i = where[lb]
            row ^= basis[i]
            t ^= trans[i]

        # Dependent check: every combination it is in pins its syndrome bit.
        if not row:
            deps |= t
            continue

        lb = row.bit_length() - 1
        bit = 1 << lb
        for i, b in enumerate(basis):
            if b & bit:
                basis[i] = b ^ row
                trans[i] ^= t
        where[lb] = len(basis)
        basis.append(row)
        trans.append(t)
        pivot_mask |= bit
    return basis, trans, deps


def _t_guess(n: int, lists: list[list[int]]) -> int:
    """Half the Tanner bound dmin >= wc_min + 1, in 1.._MAX_CORRECTABLE."""
    weights = [0] * n
    for lst in lists:
        for j in lst:
            weights[j] += 1
    wc = min((w for w in weights if w), default=1)
    return max(1, min(_MAX_CORRECTABLE, (wc + 1) // 2))


def _build(n: int, lists: list[list[int]]) -> LdpcCode:
    """LdpcCode from 0-based variable lists, one per check."""
    rows = [sum(1 << (n - 1 - j) for j in lst) for lst in lists]
    basis, trans, deps = _eliminate(rows)
    pivots = tuple(n - b.bit_length() for b in basis)
    pivot_set = set(pivots)
    m = len(rows)
    return LdpcCode(
        n=n,
        m=m,
        k=n - len(basis),
        rank=len(basis),
        rows=tuple(rows),
        pivots=pivots,
        free=tuple(j for j in range(n) if j not in pivot_set),
        feasible=tuple(r for r in range(m) if not (deps >> r) & 1),
        live=tuple(r for r in range(m) if rows[r]),
        t_guess=_t_guess(n, lists),
        _basis=tuple(basis),
        _trans=tuple(trans),
        _pivot_mask=sum(1 << (n - 1 - p) for p in pivots),
    )


# --- alist parsing ------------------------------------------------------------


def _ints(text: str) -> list[int]:
    """All whitespace-separated tokens as ints; ValueError names the bad one."""
    out = []
    for tok in text.split():
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise ValueError(f"alist: token {tok[:20]!r} is not an integer") from exc
    return out


def _header(toks: list[int]) -> tuple[int, int, list[int], list[int]]:
    """(n, m, column weights, row weights) after validating the header."""
    if len(toks) < _HEADER_TOKENS:
        raise ValueError("alist: truncated header")
    n, m, max_c, max_r = toks[:_HEADER_TOKENS]
    if not (1 <= n <= MAX_N and 1 <= m <= MAX_N):
        raise ValueError(f"alist: need 1 <= n, m <= {MAX_N}, got n={n} m={m}")
    if len(toks) < _HEADER_TOKENS + n + m:
        raise ValueError("alist: truncated weight lines")

    colw = toks[_HEADER_TOKENS : _HEADER_TOKENS + n]
    roww = toks[_HEADER_TOKENS + n : _HEADER_TOKENS + n + m]
    if any(not 0 <= w <= m for w in colw) or any(not 0 <= w <= n for w in roww):
        raise ValueError("alist: weight out of range")
    if max(colw) != max_c or max(roww) != max_r:
        raise ValueError(f"alist: max weights {max_c} {max_r} disagree with the weight lists")
    if sum(colw) != sum(roww):
        raise ValueError("alist: column and row weights have different sums")
    return n, m, colw, roww


def _chunks(body: list[int], weights: list[int], width: int) -> list[list[int]]:
    """Split *body* into one list per weight: fixed *width* if padded, else exact."""
    out, pos = [], 0
    for w in weights:
        step = width if width else w
        out.append(body[pos : pos + step])
        pos += step
    return out


def _lists(body: list[int], weights: list[int], limit: int, what: str) -> list[list[int]]:
    """0-based index lists; ValueError on padding, range, weight or duplicates."""
    out = []
    for i, (lst, w) in enumerate(zip(body, weights, strict=True)):
        vals = lst[:w]
        if any(lst[w:]):
            raise ValueError(f"alist: {what} {i + 1} has {w} entries declared, more given")
        if len(vals) != w or any(not 1 <= v <= limit for v in vals):
            raise ValueError(f"alist: {what} {i + 1} index out of range 1..{limit}")
        if len(set(vals)) != w:
            raise ValueError(f"alist: {what} {i + 1} repeats an index")
        out.append([v - 1 for v in vals])
    return out


def parse_alist(text: str) -> LdpcCode:
    """Parse MacKay alist *text*; ValueError on any inconsistency.

    Lists may be zero-padded to the max weight or not, but not mixed.
    """
    toks = _ints(text)
    n, m, colw, roww = _header(toks)
    body = toks[_HEADER_TOKENS + n + m :]
    max_c, max_r = toks[2], toks[3]

    # Padded and exact layouts are told apart by the token count.
    if len(body) == n * max_c + m * max_r:
        widths = (max_c, max_r)
    elif len(body) == sum(colw) + sum(roww):
        widths = (0, 0)
    else:
        raise ValueError("alist: column/row list token count matches neither layout")

    split = n * widths[0] if widths[0] else sum(colw)
    cols = _lists(_chunks(body[:split], colw, widths[0]), colw, m, "column")
    rows = _lists(_chunks(body[split:], roww, widths[1]), roww, n, "row")

    from_cols = {(j, r) for j, lst in enumerate(cols) for r in lst}
    from_rows = {(j, r) for r, lst in enumerate(rows) for j in lst}
    if from_cols != from_rows:
        raise ValueError("alist: column and row lists describe different matrices")
    return _build(n, rows)


def load_alist(path: str) -> LdpcCode:
    """Read and parse an alist file; ValueError on bad content or encoding."""
    with open(path, encoding="utf-8") as fh:
        return parse_alist(fh.read())


def check_params(code: LdpcCode) -> None:
    """ValueError unless 0 < rank and 0 < k and n <= MAX_N."""
    if code.n > MAX_N:
        raise ValueError(f"LDPC n={code.n} exceeds {MAX_N}")
    if code.rank == 0:
        raise ValueError("LDPC H has rank 0: no parity to repair")
    if code.k == 0:
        raise ValueError(f"LDPC H has full rank {code.n}: no information bits (k = 0)")


# --- codeword ops -------------------------------------------------------------


def _encode(code: LdpcCode, word: int) -> int:
    """Keep *word*'s information bits, recompute its parity bits."""
    info = word & ~code._pivot_mask
    out = info
    shift = code.n - 1
    for row, p in zip(code._basis, code.pivots, strict=True):
        if (row & info).bit_count() & 1:
            out |= 1 << (shift - p)
    return out


def _pick(rng, n: int, count: int) -> list[int]:
    """*count* distinct indices in range(n), sparse Fisher-Yates (O(count))."""
    swap: dict[int, int] = {}
    out = []
    for i in range(count):
        j = rng.randint(i, n - 1)
        out.append(swap.get(j, j))
        swap[j] = swap.get(i, i)
    return out


def _bits_mask(n: int, positions) -> int:
    """Mask with variable j at bit n-1-j for each j in *positions*."""
    return sum(1 << (n - 1 - j) for j in positions)


def _error(code: LdpcCode, mode: Mode, rng) -> int:
    """Error pattern XORed onto the re-encoded word for injection modes."""
    n = code.n
    if mode is Mode.CORRECTABLE:
        count = rng.randint(1, min(code.t_guess, n))
        return _bits_mask(n, _pick(rng, n, count))

    if mode is Mode.UNCORRECTABLE:
        return code.rows[code.live[rng.randint(0, len(code.live) - 1)]]

    # SYNDROME: H.e = u_r via the RREF pivots whose transform uses check r.
    if not code.feasible:
        return _bits_mask(n, (rng.randint(0, n - 1),))
    r = code.feasible[rng.randint(0, len(code.feasible) - 1)]
    piv = (p for t, p in zip(code._trans, code.pivots, strict=True) if (t >> r) & 1)
    return _bits_mask(n, piv)


def _mutate_word(code: LdpcCode, word: int, rng) -> int:
    """One Mode edit on an n-bit word; never returns *word* itself."""
    mode = Mode(rng.randint(Mode.REPAIR, Mode.SYNDROME))
    if mode is Mode.REPAIR:
        count = rng.randint(1, min(MAX_DATA_EDITS, code.k))
        picks = _pick(rng, code.k, count)
        return _encode(code, word ^ _bits_mask(code.n, (code.free[i] for i in picks)))

    base = _encode(code, word)
    out = base ^ _error(code, mode, rng)
    return base if out == word else out


def ldpc_fix(data: bytes, rng, code: LdpcCode) -> bytes | None:
    """Mutate one LDPC codeword of *data*; None if no full codeword fits.

    Args:
        data: Bitstream of consecutive n-bit codewords, MSB first per byte.
        rng: ``RandPool`` (or ScriptedRng in tests).
        code: Parity-check matrix from ``load_alist`` (``check_params``-valid).
    """
    n = code.n
    count = len(data) * _BYTE // n
    if count == 0 or code.k == 0:
        return None

    # Slice only the bytes spanning the chosen codeword.
    start = rng.randint(0, count - 1) * n
    lo, hi = start // _BYTE, (start + n + _BYTE - 1) // _BYTE
    chunk = int.from_bytes(data[lo:hi], "big")
    shift = (hi - lo) * _BYTE - (start - lo * _BYTE) - n
    word = (chunk >> shift) & ((1 << n) - 1)

    new = _mutate_word(code, word, rng)
    chunk ^= (word ^ new) << shift
    return data[:lo] + chunk.to_bytes(hi - lo, "big") + data[hi:]
