"""Operators from J. Arndt, *Matters Computational* (the FXT book).

Each keeps one invariant a parser checks, so the edit survives the first
validation and reaches the code behind it. Reimplemented from the book's
descriptions (FXT is GPL); no FXT code.

    op               chapter   invariant kept
    ---------------  --------  ---------------------------------------------
    rational_pair    37.3      value (unreduced/reduced) or a convergent
    sum_shift        7, 16     sum of two fields (composition step)
    multiset_next    13.2      byte multiset of a span (sum, xor, histogram)
    haar_coef        24, 27    every wavelet coefficient but one
    revbin_permute   2.1       element multiset; an involution
    hilbert_patch    1.31      locality: touched cells form a 2D path
    hadamard_oa      19.1      written word is a Hadamard row (OA strength 2)
    fib_encode       14.2      Fibonacci universal code of each byte

All are length-preserving and return *data* unchanged when it cannot hold
their construction.
"""

from __future__ import annotations

from functools import cache, lru_cache
from math import gcd

from fuzzer_tool.core.mutations.structured import _edit_codebits, _pad_random, _region, _splice

ENDIANS = ("little", "big")


# ── rational_pair (37.3) ─────────────────────────────────────────────────
# Two adjacent fields read as num/den: time_base, sample aspect ratio,
# frame rate. Convergents and mediants are the values a rational-reduction
# routine (av_reduce) produces and must round-trip.
RAT_WIDTHS = (1, 2, 4)
RAT_ORDERS = ("num_first", "den_first")
RAT_KINDS = ("unreduced", "reduced", "zero_den", "unit_den", "extreme", "convergent", "mediant")
RAT_FACTOR = 2


def _cf_convergents(num: int, den: int) -> list[tuple[int, int]]:
    """Convergents ``p/q`` of ``num/den`` (den > 0), last one equals it reduced."""
    out = []
    p0, q0, p1, q1 = 0, 1, 1, 0
    while den:
        a, rem = divmod(num, den)
        p0, q0, p1, q1 = p1, q1, a * p1 + p0, a * q1 + q0
        out.append((p1, q1))
        num, den = den, rem
    return out


def _rat_unreduced(num, den, _bits, _rng):
    if num == den == 0:
        return None
    return num * RAT_FACTOR, den * RAT_FACTOR


def _rat_reduced(num, den, _bits, _rng):
    g = gcd(num, den)
    return (num // g, den // g) if g > 1 else None


def _rat_zero_den(num, den, _bits, _rng):
    return (num, 0) if den else None


def _rat_unit_den(num, den, _bits, _rng):
    return (num, 1) if den != 1 else None


def _rat_extreme(num, den, bits, _rng):
    pair = ((1 << bits) - 1, 1)
    return pair if pair != (num, den) else None


def _rat_convergent(num, den, _bits, rng):
    if not den:
        return None
    conv = _cf_convergents(num, den)
    if len(conv) < 2:
        return None
    return conv[rng.randint(0, len(conv) - 2)]


def _rat_mediant(num, den, _bits, _rng):
    # Mediant of num/den and its previous convergent (1/0 before the first):
    # the next Stern-Brocot node between them.
    if not den:
        return None
    conv = _cf_convergents(num, den)
    p, q = conv[-2] if len(conv) > 1 else (1, 0)
    rn, rd = conv[-1]
    return rn + p, rd + q


_RAT_FN = {
    "unreduced": _rat_unreduced,
    "reduced": _rat_reduced,
    "zero_den": _rat_zero_den,
    "unit_den": _rat_unit_den,
    "extreme": _rat_extreme,
    "convergent": _rat_convergent,
    "mediant": _rat_mediant,
}


def rational_pair(data: bytes, rng) -> bytes:
    """Rewrite two adjacent fields read as a fraction ``num/den``.

    Kinds (``RAT_KINDS``): same value with a common factor, reduced form,
    ``den = 0``, ``den = 1``, ``UMAX/1``, a proper convergent of the
    continued fraction, or the mediant with the previous convergent. A kind
    that does not apply or does not fit the width returns *data*.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    widths = [w for w in RAT_WIDTHS if 2 * w <= len(data)]
    if not widths:
        return data

    width = rng.choice(widths)
    offset = rng.randint(0, len(data) - 2 * width)
    endian = rng.choice(ENDIANS)
    order = rng.choice(RAT_ORDERS)
    kind = rng.choice(RAT_KINDS)

    # e.g. width 2 little, num_first: 03 00 | 05 00 is 3/5.
    first = int.from_bytes(data[offset : offset + width], endian)
    second = int.from_bytes(data[offset + width : offset + 2 * width], endian)
    num, den = (first, second) if order == RAT_ORDERS[0] else (second, first)

    bits = 8 * width
    pair = _RAT_FN[kind](num, den, bits, rng)
    if pair is None or max(pair) >> bits:
        return data

    num, den = pair
    a, b = (num, den) if order == RAT_ORDERS[0] else (den, num)
    return _splice(data, offset, a.to_bytes(width, endian) + b.to_bytes(width, endian))


# ── sum_shift (7, 16) ────────────────────────────────────────────────────
# One step between neighbouring compositions of a total: k units leave one
# field for another. Chunk sizes that must add up to a length, counts that
# must total N: every other operator breaks the total.
SUM_WIDTHS = (1, 2, 4)
SUM_DIRS = ("a_to_b", "b_to_a")
SUM_MAX_GAP = 4


def sum_shift(data: bytes, rng) -> bytes:
    """Move ``k`` units between two fields ``gap`` fields apart; their sum is kept.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    widths = [w for w in SUM_WIDTHS if 2 * w <= len(data)]
    if not widths:
        return data

    width = rng.choice(widths)
    gap = rng.randint(1, min(SUM_MAX_GAP, len(data) // width - 1))
    offset = rng.randint(0, len(data) - (gap + 1) * width)
    endian = rng.choice(ENDIANS)
    direction = rng.choice(SUM_DIRS)

    # e.g. width 1, gap 1: [10, 20] -> a_to_b 3 -> [7, 23].
    pos_a, pos_b = offset, offset + gap * width
    a = int.from_bytes(data[pos_a : pos_a + width], endian)
    b = int.from_bytes(data[pos_b : pos_b + width], endian)
    umax = (1 << (8 * width)) - 1
    src, dst = (a, b) if direction == SUM_DIRS[0] else (b, a)
    room = min(src, umax - dst)
    if not room:
        return data

    k = rng.randint(1, room)
    src, dst = src - k, dst + k
    a, b = (src, dst) if direction == SUM_DIRS[0] else (dst, src)
    out = _splice(data, pos_a, a.to_bytes(width, endian))
    return _splice(out, pos_b, b.to_bytes(width, endian))


# ── multiset_next (13.2) ─────────────────────────────────────────────────
# Lexicographic successor/predecessor among the distinct arrangements of a
# span: the smallest edit that keeps the byte multiset, so additive, xor and
# histogram checks still pass. byte_shuffle is random and cannot step.
PERM_DIRS = ("next", "prev")
PERM_MAX_SPAN = 16


def _lex_step(seq: bytearray, step_up: bool) -> None:
    """In-place next (or previous) multiset permutation; wraps at the end."""
    n = len(seq)

    def before(x, y):
        return x < y if step_up else x > y

    i = n - 2
    while i >= 0 and not before(seq[i], seq[i + 1]):
        i -= 1
    if i >= 0:
        j = n - 1
        while not before(seq[i], seq[j]):
            j -= 1
        seq[i], seq[j] = seq[j], seq[i]
    seq[i + 1 :] = seq[i + 1 :][::-1]


def multiset_next(data: bytes, rng) -> bytes:
    """Step a span to its lexicographic next or previous arrangement.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data

    length = rng.randint(2, min(PERM_MAX_SPAN, len(data)))
    offset = rng.randint(0, len(data) - length)
    direction = rng.choice(PERM_DIRS)

    # e.g. next: 01 03 02 -> 02 01 03; prev of the first wraps to the last.
    span = bytearray(data[offset : offset + length])
    _lex_step(span, direction == PERM_DIRS[0])
    return _splice(data, offset, bytes(span))


# ── haar_coef (24.3, 27.2) ───────────────────────────────────────────────
# Integer S-transform (lifting Haar): reversible on integers, so editing one
# coefficient and inverting changes the span by exactly that coefficient's
# basis function: a smooth, multi-byte step, not byte noise.
HAAR_MAX_LEVELS = 6
HAAR_DELTAS = (-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32)


def haar_forward(x: list[int]) -> list[int]:
    """S-transform of a power-of-two list: ``[s, d_coarse, ..., d_fine]``."""
    x, details = list(x), []
    while len(x) > 1:
        evens, odds = x[0::2], x[1::2]
        details = [a - b for a, b in zip(evens, odds, strict=True)] + details
        x = [(a + b) >> 1 for a, b in zip(evens, odds, strict=True)]
    return x + details


def haar_inverse(c: list[int]) -> list[int]:
    """Inverse of ``haar_forward``: ``a = s + ((d + 1) >> 1)``, ``b = a - d``."""
    x, pos = c[:1], 1
    while pos < len(c):
        d = c[pos : 2 * pos]
        nxt = []
        for s, di in zip(x, d, strict=True):
            a = s + ((di + 1) >> 1)
            nxt += (a, a - di)
        x, pos = nxt, 2 * pos
    return x


def haar_coef(data: bytes, rng) -> bytes:
    """Nudge one S-transform coefficient of a ``2^k``-byte span, clamp to bytes.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data

    levels = rng.randint(1, min(HAAR_MAX_LEVELS, len(data).bit_length() - 1))
    n = 1 << levels
    offset = rng.randint(0, len(data) - n)
    index = rng.randint(0, n - 1)
    delta = rng.choice(HAAR_DELTAS)

    coef = haar_forward(list(data[offset : offset + n]))
    coef[index] += delta
    span = bytes(min(255, max(0, v)) for v in haar_inverse(coef))
    return _splice(data, offset, span)


# ── revbin_permute (2.1) ─────────────────────────────────────────────────
# Bit-reversal permutation of 2^k elements: the FFT input order. A reader
# that assumes natural order (or applies revbin twice) meets the other one.
REVBIN_WIDTHS = (1, 2, 4)
REVBIN_MIN_K = 2
REVBIN_MAX_K = 6


@lru_cache(maxsize=REVBIN_MAX_K + 1)
def _revbin_table(k: int) -> tuple[int, ...]:
    return tuple(int(format(i, f"0{k}b")[::-1], 2) for i in range(1 << k))


def revbin_permute(data: bytes, rng) -> bytes:
    """Reorder ``2^k`` elements of width 1/2/4 by bit-reversed index.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    widths = [w for w in REVBIN_WIDTHS if (1 << REVBIN_MIN_K) * w <= len(data)]
    if not widths:
        return data

    width = rng.choice(widths)
    max_k = min(REVBIN_MAX_K, (len(data) // width).bit_length() - 1)
    k = rng.randint(REVBIN_MIN_K, max_k)
    offset = rng.randint(0, len(data) - (width << k))

    # e.g. k = 3, width 1: 0 1 2 3 4 5 6 7 -> 0 4 2 6 1 5 3 7.
    block = data[offset : offset + (width << k)]
    elems = [block[i * width : (i + 1) * width] for i in range(1 << k)]
    return _splice(data, offset, b"".join(elems[j] for j in _revbin_table(k)))


# ── hilbert_patch (1.31) ─────────────────────────────────────────────────
# Region read as an n x n grid with an unknown row stride; a run of
# consecutive Hilbert-order cells is a compact 2D blob, the kind of local
# change an image decoder's prediction and block code see as one feature.
HILBERT_MAX_K = 5
HILBERT_MAX_PATCH = 16
HILBERT_DELTAS = (1, 2, 4, 8, 16, 32, 64, 128, 0xFF, 0xFE, 0xFC, 0xF8)


def hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    """Cell ``(x, y)`` at Hilbert index ``d`` of an ``n x n`` grid (n a power of two)."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d >> 1)
        ry = 1 & (d ^ rx)
        if not ry:
            if rx:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


def hilbert_patch(data: bytes, rng) -> bytes:
    """Add one delta to a run of consecutive Hilbert cells of a 2D region.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    max_k = min(HILBERT_MAX_K, (len(data).bit_length() - 1) // 2)
    if max_k < 1:
        return data

    k = rng.randint(1, max_k)
    n = 1 << k
    stride = rng.randint(n, len(data) // n)
    offset = rng.randint(0, len(data) - ((n - 1) * stride + n))
    m = rng.randint(2, min(n * n, HILBERT_MAX_PATCH))
    d0 = rng.randint(0, n * n - m)
    delta = rng.choice(HILBERT_DELTAS)

    out = bytearray(data)
    for d in range(d0, d0 + m):
        x, y = hilbert_d2xy(n, d)
        pos = offset + y * stride + x
        out[pos] = (out[pos] + delta) & 0xFF
    return bytes(out)


# ── hadamard_oa (19.1) ───────────────────────────────────────────────────
# Rows of the order-16 Hadamard matrix (signed m-sequence, x^4 + x + 1) as
# 16-bit words, bit j set where the entry is -1. Columns 1..15 form a
# strength-2 orthogonal array: over the rows, every pair of flag bits takes
# 00/01/10/11 equally often; 16 writes cover all pairwise flag interactions.
OA_ORDER = 16
OA_MODES = ("set", "xor")


def _hadamard_rows() -> tuple[int, ...]:
    seq = [1, 0, 0, 0]
    while len(seq) < OA_ORDER - 1:
        seq.append(seq[-3] ^ seq[-4])
    rows = [0]
    for i in range(OA_ORDER - 1):
        rows.append(sum(seq[(i + j) % (OA_ORDER - 1)] << (j + 1) for j in range(OA_ORDER - 1)))
    return tuple(rows)


HADAMARD_ROWS = _hadamard_rows()


def hadamard_oa(data: bytes, rng) -> bytes:
    """Write (or XOR) one Hadamard row into a 16-bit flag word.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data

    row = HADAMARD_ROWS[rng.randint(0, OA_ORDER - 1)]
    offset = rng.randint(0, len(data) - 2)
    endian = rng.choice(ENDIANS)
    mode = rng.choice(OA_MODES)

    word = row
    if mode == OA_MODES[1]:
        word ^= int.from_bytes(data[offset : offset + 2], endian)
    return _splice(data, offset, word.to_bytes(2, endian))


# ── fib_encode (14.2) ────────────────────────────────────────────────────
# Fibonacci universal code: Zeckendorf digits (no two adjacent ones), least
# significant first, then a 1, so every codeword ends in "11". Sibling of
# elias_gamma/elias_delta: encode each byte+1, edit the bit stream, decode.
_FIB = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)


@cache
def fib_codeword(n: int) -> tuple[int, ...]:
    """Fibonacci code of ``1 <= n <= _FIB_MAX`` as a tuple of bits."""
    digits = [0] * len(_FIB)
    for i in range(len(_FIB) - 1, -1, -1):
        if _FIB[i] <= n:
            digits[i] = 1
            n -= _FIB[i]
    top = max(i for i, v in enumerate(digits) if v)
    return (*digits[: top + 1], 1)


@cache
def _fib_table() -> tuple[bytes, ...]:
    """Codeword bytes (one 0/1 byte per bit) of each byte value + 1."""
    return tuple(bytes(fib_codeword(v + 1)) for v in range(256))


def fib_bits(block: bytes) -> bytearray:
    """Concatenated codewords of each byte+1."""
    return bytearray(b"".join(map(_fib_table().__getitem__, block)))


# Largest value with Zeckendorf digits inside _FIB: every other term from the
# top, 233 + 89 + 34 + 13 + 5 + 2 = 376.
_FIB_MAX = sum(_FIB[-1::-2])


@cache
def _fib_values() -> dict[bytes, int]:
    """Digit run -> value for every run ``fib_decode`` can see within ``_FIB``.

    Runs end in a 1 and hold no two adjacent ones: each is the Zeckendorf form
    of exactly one ``n <= _FIB_MAX`` (e.g. ``00 01`` is 2).
    """
    return {bytes(fib_codeword(n)[:-1]): n for n in range(1, _FIB_MAX + 1)}


def fib_decode(bits: bytearray, length: int) -> tuple[bytearray, int]:
    """Decode up to *length* codewords -> (restored, bytes decoded).

    A codeword ends at the first ``11``: its digits are everything up to and
    including the first of the two ones (digits past ``_FIB`` count 0).
    """
    restored = bytearray(length)
    count = pos = 0
    while count < length:
        end = bits.find(b"\x01\x01", pos)
        if end < 0:
            break
        digits = bytes(bits[pos : end + 1])
        value = _fib_values().get(digits)
        if value is None:
            value = sum(f for f, b in zip(_FIB, digits, strict=False) if b)
        restored[count] = (value - 1) & 0xFF
        count += 1
        pos = end + 2
    return restored, count


def fib_encode(data: bytes, rng) -> bytes:
    """Fibonacci-code a region, edit the bit stream, decode back.

    Args:
        data: Input bytes.
        rng: Draw source, required (``RandPool`` API; tests inject ``ScriptedRng``).

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data

    offset, length = _region(len(data), rng, min_len=2)
    if length < 2:
        return data

    bits = fib_bits(data[offset : offset + length])
    _edit_codebits(bits, length, rng)
    restored, count = fib_decode(bits, length)
    _pad_random(restored, count, rng)
    return _splice(data, offset, bytes(restored))
