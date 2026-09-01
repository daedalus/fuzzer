"""Structured-regularity mutation operators (the dieharder-inverse family).

Every test in the diehard/dieharder battery defines a statistic ``S`` over a
byte stream together with its distribution under the uniform null.  The test
asks "is ``S(stream)`` improbably far from the mean?".  These operators ask
the opposite question and answer it constructively: they *build* buffers
whose statistic sits in the far tail of that null distribution.

That inversion is useful for fuzzing because the far tail is exactly where
random havoc never lands, and because the regularities the battery was
designed to detect line up with real parser and algorithm fast paths:

===========================  ==========================================
diehard/dieharder test       constructive inverse implemented here
===========================  ==========================================
marsaglia_tsang_gcd          :func:`fibonacci_pairs`
dab_filltree, dab_filltree2  :func:`monotone_fill`
diehard_opso/oqso/dna,       :func:`de_bruijn_fill` (saturate)
  diehard_bitstream          :func:`kmer_starve` (starve)
diehard_rank_32x32, _6x8     :func:`rank_deficient`
diehard_operm5,              :func:`perm_lock`
  rgb_permutations
rgb_lagged_sums              :func:`lag_correlate`
dab_dct                      :func:`spectral_peak`
diehard_birthdays,           :func:`birthday_collide`
  dab_birthdays1
rgb_persist                  :func:`invariant_break`
diehard_parking_lot,         :func:`degenerate_geometry`
  rgb_minimum_distance
diehard_squeeze              :func:`float_squeeze`
diehard_count_1s_byte/stream :func:`popcount_lock`
===========================  ==========================================

The detectors for several of these already live in
:mod:`fuzzer_tool.core.randomness` (``kmer_occupancy``, ``_batch_gf2_rank``,
``birthday_spacings``, ``permutation_test``, ``corpus_invariants``); the
functions here are their constructive duals, and the tests assert the round
trip wherever a detector exists.

Note on provenance: dieharder itself is GPL-2 and this module contains none
of its code.  The constructions are derived from the public test
*descriptions* (Marsaglia's ``tests.txt``, the dieharder manual) and from the
underlying combinatorics, which is why the parameters are chosen for
adversarial effect rather than to reproduce dieharder's sampling.

Every operator is length-preserving: each overwrites a bounded region of the
input in place and returns a buffer of the same size.  That keeps them cheap,
keeps ``max_len`` a non-issue, and lets them compose with the length-changing
operators instead of competing with them.

Like the rest of :mod:`fuzzer_tool.core.mutations`, these functions use only
the API shared by ``RandPool`` and stdlib ``random`` (``randint``, ``choice``,
``random``, ``sample``, ``randbytes``) so they stay usable with either; the
weighted draws are expressed as pre-expanded tuples rather than by calling
``RandPool.weighted_choice``, which stdlib ``random`` does not have.
"""

import math
import struct
from functools import lru_cache

from fuzzer_tool.core.mutations.generic import _get_rng

# Largest region any single operator will rewrite. Operators that scribble
# over an entire multi-megabyte seed destroy the structure the corpus spent
# CPU discovering, and cost proportionally more per call for no extra signal
# -- the regularity only has to run long enough for the target's loop to
# notice it.
MAX_REGION = 4096

# Word widths, pre-weighted toward the sizes real formats use for counts and
# offsets. ``_WIDTHS_WORD`` drops the 1-byte case for operators whose
# construction is meaningless on single bytes.
_WIDTHS = (1, 2, 2, 2, 4, 4, 4, 4, 8, 8)
_WIDTHS_WORD = (2, 2, 2, 4, 4, 4, 4, 8, 8)

_STRUCT_FMT = {
    (1, False): "<B",
    (1, True): ">B",
    (2, False): "<H",
    (2, True): ">H",
    (4, False): "<I",
    (4, True): ">I",
    (8, False): "<Q",
    (8, True): ">Q",
}


def _region(data_len: int, rng, min_len: int = 1, align: int = 1) -> tuple[int, int]:
    """Pick a random ``(offset, length)`` window to overwrite.

    Args:
        data_len: Length of the buffer being mutated.
        rng: RandPool or stdlib random.
        min_len: Smallest window the caller's construction needs.
        align: Snap the offset down to a multiple of this so the caller's
            fixed-width words line up with how a reader slices them. Without
            it an unaligned run reads as ordinary noise at the reader's own
            stride, silently defeating the construction.

    Returns:
        ``(offset, length)`` with ``offset + length <= data_len``, or
        ``(0, 0)`` when the buffer is shorter than *min_len*.
    """
    if data_len < min_len or min_len < 1:
        return 0, 0
    span = min(MAX_REGION, data_len)
    if span < min_len:
        return 0, 0
    length = rng.randint(min_len, span)
    offset = rng.randint(0, data_len - length)
    if align > 1:
        offset -= offset % align
        length = min(length, data_len - offset)
        if length < min_len:
            return 0, 0
    return offset, length


def _pack_words(values, width: int, big_endian: bool) -> bytes:
    """Pack integers into fixed-width words, masking each to the width."""
    pack = struct.Struct(_STRUCT_FMT[(width, big_endian)]).pack
    mask = (1 << (width * 8)) - 1
    return b"".join(pack(v & mask) for v in values)


def _splice(data: bytes, offset: int, block: bytes) -> bytes:
    """Overwrite ``data[offset:offset+len(block)]`` with *block*."""
    if not block:
        return data
    out = bytearray(data)
    end = min(offset + len(block), len(out))
    out[offset:end] = block[: end - offset]
    return bytes(out)


def _map_table(alphabet: bytes) -> bytes:
    """Build a 256-entry ``bytes.translate`` table onto *alphabet*.

    Lets an operator draw a whole region with one ``randbytes`` call plus a
    C-level translate, instead of one Python-level RNG call per byte. The
    modulo bias across 256 is irrelevant here -- the point is which alphabet
    the bytes come from, not that they are uniform within it.
    """
    n = len(alphabet)
    return bytes(alphabet[i % n] for i in range(256))


# ── 1. marsaglia_tsang_gcd inverse ─────────────────────────────────────


def _fibonacci_table() -> list[int]:
    """Fibonacci numbers up to just past 2^64; index 0 holds F(1) == 1."""
    fib = [1, 1]
    limit = 1 << 64
    while fib[-1] < limit:
        fib.append(fib[-1] + fib[-2])
    return fib


_FIB = _fibonacci_table()

# Largest index whose successor still fits each word width, precomputed so
# the operator does not rescan the table on every call.
_FIB_TOP = {
    width: max(1, max(i for i, v in enumerate(_FIB) if v < (1 << (width * 8))) - 1)
    for width in (2, 4, 8)
}


def fibonacci_pairs(data: bytes, rng=None) -> bytes:
    """Overwrite a region with consecutive Fibonacci pairs (gcd worst case).

    The Euclidean algorithm's iteration count is maximised, over operands
    below a given bound, exactly by consecutive Fibonacci numbers -- Lame's
    theorem, and the reason dieharder's GCD test sees a right-tail excess in
    the step count ``k`` when a generator emits them.  Emitting them
    deliberately drives any ``gcd``/``av_reduce``-style reduction to its worst
    case, which is where rational normalisers, aspect-ratio and timebase code,
    and bignum fast paths tend to break.

    With probability 1/2 a multiplier ``m`` is applied so the pair has gcd
    ``m`` rather than 1: the step count stays maximal but the result is
    non-trivial, exercising the "common factor found" branch as well.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    width = rng.choice(_WIDTHS_WORD)
    pair_size = width * 2
    offset, length = _region(len(data), rng, min_len=pair_size, align=width)
    if length < pair_size:
        return data
    big_endian = rng.random() < 0.5

    top = _FIB_TOP[width]
    # Back off a few indices at random so the operator explores a range of
    # step counts rather than only the single maximum.
    start = max(1, top - rng.randint(0, min(8, top - 1)))

    multiplier = 1
    if rng.random() < 0.5:
        headroom = (1 << (width * 8)) // max(1, _FIB[start + 1])
        if headroom > 1:
            multiplier = rng.randint(2, min(headroom, 256))

    values = []
    for _ in range(length // pair_size):
        values.append(_FIB[start] * multiplier)
        values.append(_FIB[start + 1] * multiplier)
    return _splice(data, offset, _pack_words(values, width, big_endian))


# ── 2. dab_filltree inverse ────────────────────────────────────────────

_STRIDES = (1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 7, 7, 7, 256, 256)


def monotone_fill(data: bytes, rng=None) -> bytes:
    """Overwrite a region with a strictly monotone run of fixed-width words.

    dab_filltree measures how many words a fixed-depth binary tree accepts
    before one cannot be inserted; a uniform stream balances the tree.  A
    monotone run degenerates it into a linked list, so any parser that feeds
    parsed records into a BST, an interval map or a sorted index -- symbol
    tables, ZIP central directories, font cmaps, key maps with a tree
    fallback -- walks its worst-case insertion path.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    width = rng.choice(_WIDTHS)
    offset, length = _region(len(data), rng, min_len=width * 2, align=width)
    if length < width * 2:
        return data
    big_endian = rng.random() < 0.5

    n = length // width
    ceiling = 1 << (width * 8)
    stride = rng.choice(_STRIDES)
    span = stride * (n - 1)
    if rng.random() < 0.5:
        start = min(ceiling - 1, span)
        values = [start - i * stride for i in range(n)]
    else:
        start = rng.randint(0, ceiling - 1 - span) if span < ceiling else 0
        values = [start + i * stride for i in range(n)]
    return _splice(data, offset, _pack_words(values, width, big_endian))


# ── 3. diehard_opso / oqso / dna / bitstream inverse ───────────────────


def _de_bruijn_symbols(k: int, n: int) -> list[int]:
    """Symbols of a de Bruijn sequence B(k, n) via the iterative FKM scheme.

    Every one of the ``k**n`` words of length *n* over a *k*-symbol alphabet
    occurs exactly once as a cyclic substring of the result.  Written
    iteratively rather than with the usual recursive ``db()`` helper, per the
    repo's preference for non-recursive formulations.
    """
    a = [0] * (n + 1)
    seq: list[int] = []
    i = 1
    while i > 0:
        if n % i == 0:
            seq.extend(a[1 : i + 1])
        for j in range(i + 1, n + 1):
            a[j] = a[j - i]
        i = n
        while i > 0 and a[i] == k - 1:
            i -= 1
        if i > 0:
            a[i] += 1
    return seq


@lru_cache(maxsize=32)
def de_bruijn_bytes(k: int, n: int) -> bytes:
    """Cached de Bruijn sequence B(k, n) rendered as bytes (``k <= 256``).

    Returns an immutable ``bytes`` precisely so the cache cannot be mutated
    through: caching a mutable sequence here would corrupt every later caller
    the first time an operator wrote through the result.
    """
    step = 256 // k if k < 256 else 1
    return bytes((s * step) & 0xFF for s in _de_bruijn_symbols(k, n))


# (alphabet size, word length), smallest output first. k=2 with a long word
# is the byte-level analogue of diehard_bitstream; k=256, n=2 is OPSO's
# two-letter-word saturation carried out to completion.
_DE_BRUIJN_SHAPES = ((2, 8), (4, 4), (16, 2), (4, 6), (16, 3), (256, 2))


def de_bruijn_fill(data: bytes, rng=None) -> bytes:
    """Overwrite the buffer with a de Bruijn sequence (k-mer saturation).

    OPSO, OQSO, DNA and BITSTREAM all count *missing* k-letter words in an
    overlapping window; a uniform stream leaves a predictable number unseen.
    A de Bruijn sequence leaves exactly zero unseen, the extreme opposite
    tail.  For a table-driven lexer or DFA that means every reachable state
    transition of order ``n`` is taken, in the shortest possible number of
    bytes -- maximum branch diversity per byte of input.

    The sequence tiles the *whole* buffer: the missing-k-mer count is a
    property of the buffer a reader samples, not of one region inside it, so
    a partial fill lets the surrounding noise dominate the statistic it is
    supposed to move.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    if len(data) < 16:
        return data
    shapes = [(k, n) for k, n in _DE_BRUIJN_SHAPES if k**n <= len(data)]
    k, n = rng.choice(shapes) if shapes else (2, 4)
    seq = de_bruijn_bytes(k, n)
    # The sequence is cyclic, so tiling keeps every window boundary a
    # legitimate de Bruijn window.
    reps = -(-len(data) // len(seq))
    return (seq * reps)[: len(data)]


def kmer_starve(data: bytes, rng=None) -> bytes:
    """Overwrite a region using a 2-4 symbol alphabet (k-mer starvation).

    The opposite tail of the same statistic: instead of hitting every k-mer,
    hit almost none.  A stream drawn from a tiny alphabet locks a state
    machine into one region of its transition table and holds it there, which
    is how the long-run states get reached at all -- random input leaves them
    after a few bytes.  Symbols are drawn from the buffer's own bytes most of
    the time so the result stays plausible to a format check.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=8)
    if length < 8:
        return data
    n_symbols = rng.randint(2, 4)
    if data and rng.random() < 0.7:
        alphabet = bytes(data[rng.randint(0, len(data) - 1)] for _ in range(n_symbols))
    else:
        alphabet = bytes(rng.randint(0, 255) for _ in range(n_symbols))
    block = rng.randbytes(length).translate(_map_table(alphabet))
    return _splice(data, offset, block)


def _pack_bits_msb(symbols: list[int]) -> bytes:
    """Pack a sequence of 0/1 symbols one-per-*bit*, MSB-first.

    ``de_bruijn_bytes`` spends a whole byte per symbol and reaches full
    range only for its largest alphabet (k=256); for k=2 that scheme writes
    ``0x00``/``0x80`` and leaves the low 7 bits of every byte constant, so
    only the byte-aligned windows of the resulting stream are actually
    exhaustive. This packs tight instead, so the guarantee holds at every
    bit offset. ``len(symbols)`` must be a multiple of 8 for the result to
    round-trip cleanly; every caller here draws it from ``2**n`` with
    ``n >= 3``, so that always holds.
    """
    n = len(symbols)
    out = bytearray(n // 8)
    for i, s in enumerate(symbols):
        if s:
            out[i >> 3] |= 0x80 >> (i & 7)
    return bytes(out)


@lru_cache(maxsize=16)
def de_bruijn_bits(n: int) -> bytes:
    """Binary de Bruijn sequence B(2, n), packed one symbol per bit.

    Every one of the ``2**n`` possible n-bit windows occurs exactly once as
    a cyclic substring, at *every* bit offset -- not just the byte-aligned
    ones ``de_bruijn_bytes`` guarantees. That is the property a bit-level
    accumulator needs: an Exp-Golomb / CABAC-adjacent H.264 RBSP reader, a
    protobuf varint's continuation-bit chain, or any hand-rolled packed
    bitfield struct pulls its next few bits from wherever the previous field
    left off, which is essentially never byte-aligned after the first field.
    Packing bit-tight also buys density for free: the same n-bit window
    space that costs ``2**n`` bytes in ``de_bruijn_bytes`` costs ``2**n``
    *bits* here, so a buffer 8x smaller reaches the same order of coverage.
    """
    n = max(n, 3)  # below 3, 2**n is not byte-aligned once packed
    return _pack_bits_msb(_de_bruijn_symbols(2, n))


# Bit-window widths, smallest period first. All base 2 (this variant only
# saturates a binary alphabet -- the byte-aligned k=4/16/256 shapes live in
# de_bruijn_fill instead, since a wide alphabet packed to bit-granularity
# stops being a *bitfield* saturator and just becomes de_bruijn_bytes again).
_DE_BRUIJN_BIT_WORDS = (4, 6, 8, 10, 12, 14, 16, 18, 20)


def kmer_saturate_bits(data: bytes, rng=None) -> bytes:
    """Overwrite the buffer with a bit-packed binary de Bruijn sequence.

    The bit-granularity dual of :func:`de_bruijn_fill`: same idea (hit every
    k-mer in the far tail of the OPSO/OQSO/DNA/BITSTREAM occupancy count),
    but built so the exhaustive coverage survives arbitrary bit offsets
    instead of only byte-aligned ones. Byte-aligned saturation drives a
    byte-at-a-time lexer or DFA into every reachable state; this drives a
    *bit*-at-a-time accumulator there instead, which is the read pattern
    bitfield-parsing code actually uses (H.264 RBSP Exp-Golomb fields,
    protobuf varint continuation bits, packed struct bitfields) and which
    byte-aligned saturation quietly fails to reach whenever a field's start
    offset isn't a multiple of 8.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    if len(data) < 16:
        return data
    bit_budget = len(data) * 8
    words = [n for n in _DE_BRUIJN_BIT_WORDS if (1 << n) <= bit_budget]
    n = rng.choice(words) if words else 4
    seq = de_bruijn_bits(n)
    # Cyclic, same as de_bruijn_fill: tiling keeps every window boundary,
    # including the wrap from the last bit back to the first, a legitimate
    # de Bruijn window rather than a truncation artifact.
    reps = -(-len(data) // len(seq))
    return (seq * reps)[: len(data)]


# ── 4. diehard_rank_32x32 / rank_6x8 inverse ───────────────────────────

# (rows, cols, bytes per row) -- 32x32 packs one big-endian u32 per row like
# the original test; 6x8 packs one byte per row.
_RANK_SHAPES = ((32, 32, 4), (6, 8, 1))


def rank_deficient(data: bytes, rng=None) -> bytes:
    """Overwrite a region with rank-deficient GF(2) matrices.

    The binary rank tests build bit matrices from the stream and chi-square
    the rank histogram; a uniform stream is almost always full rank or one
    short.  Building each row as an XOR combination of a deliberately small
    basis forces low rank -- the far left tail.  Singular GF(2) matrices are
    the input that erasure/Reed-Solomon and LDPC decoders, GF(2) checksum
    code, and linear-algebra fast paths handle on their rarely-exercised
    "not invertible" branch.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    rows, cols, row_bytes = rng.choice(_RANK_SHAPES)
    block_size = rows * row_bytes
    offset, length = _region(len(data), rng, min_len=block_size, align=block_size)
    if length < block_size:
        return data

    col_mask = (1 << cols) - 1
    out = bytearray()
    for _ in range(length // block_size):
        # Rank is at most len(basis). Drawing it in the low half leaves room
        # for the severely degenerate cases -- rank 1 makes every row equal.
        rank = rng.randint(1, max(1, rows // 2))
        basis = [rng.randint(0, col_mask) for _ in range(rank)]
        for _r in range(rows):
            coeffs = rng.randint(0, (1 << rank) - 1)
            word = 0
            for b in range(rank):
                if coeffs >> b & 1:
                    word ^= basis[b]
            out += word.to_bytes(row_bytes, "big")
    return _splice(data, offset, bytes(out))


# ── 5. diehard_operm5 / rgb_permutations inverse ───────────────────────

_PERM_MODES = ("ascending", "descending", "organ_pipe", "equal", "interleave")


def _sorted_shape(n: int, mode: str) -> list[int]:
    """Adversarial orderings of ``1..n`` for comparison sorts."""
    base = list(range(1, n + 1))
    if mode == "ascending":
        return base
    if mode == "descending":
        return base[::-1]
    if mode == "organ_pipe":
        return base[: (n + 1) // 2] + base[: n // 2][::-1]
    if mode == "equal":
        return [1] * n
    # "interleave": alternate the low and high halves, so the first, middle
    # and last elements straddle both -- the shape median-of-3 pivot
    # selection handles worst.
    half = n // 2
    out: list[int] = []
    for i in range(half):
        out.append(base[i])
        out.append(base[half + i])
    if n % 2:
        out.append(base[-1])
    return out


def perm_lock(data: bytes, rng=None) -> bytes:
    """Overwrite a region with an ordering-degenerate word sequence.

    OPERM5 counts which of the 120 orderings each overlapping five-word
    window falls into, and rgb_permutations generalises that to ``k``; a
    uniform stream spreads across all cells.  These shapes collapse the
    histogram onto one or two.  Sorted, reverse-sorted, all-equal and
    organ-pipe sequences are also the classic quadratic inputs for comparison
    sorts, so this reaches the O(n^2) path of any sort over parsed records.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    width = rng.choice(_WIDTHS)
    offset, length = _region(len(data), rng, min_len=width * 4, align=width)
    if length < width * 4:
        return data
    big_endian = rng.random() < 0.5
    n = length // width
    values = _sorted_shape(n, rng.choice(_PERM_MODES))
    return _splice(data, offset, _pack_words(values, width, big_endian))


# ── 5b. single n-cycle / all-fixed-points permutation ──────────────────

_CYCLE_MODES = ("single_cycle", "fixed_points")


def cycle_lock(data: bytes, rng=None) -> bytes:
    """Overwrite a region with an index permutation at a traversal extreme.

    ``perm_lock`` targets comparison-sort orderings; this targets index-chase
    depth. Interpreting each word as a 0-based next-index pointer, a random
    permutation has expected cycle length O(log n) (mean ~log n, longest
    cycle concentrated well below n). Two shapes sit at the opposite ends of
    that distribution:

    - ``single_cycle``: one n-cycle (``i -> (i+1) mod n``) — the worst case
      for any bounded pointer-chase, since following it visits every slot
      before repeating.
    - ``fixed_points``: the identity (``i -> i``) — the other extreme,
      every chase terminates in one step.

    Both are degenerate under the same permutation-cycle statistic and both
    are adversarial for code that walks an index chain expecting short
    cycles: hash open-addressing probe sequences, a linked list stored as
    array indices, jump tables, and union-find parent arrays.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    width = rng.choice(_WIDTHS)
    offset, length = _region(len(data), rng, min_len=width * 4, align=width)
    if length < width * 4:
        return data
    big_endian = rng.random() < 0.5
    n = length // width
    if rng.choice(_CYCLE_MODES) == "single_cycle":
        values = [(i + 1) % n for i in range(n)]
    else:
        values = list(range(n))
    return _splice(data, offset, _pack_words(values, width, big_endian))


# ── 6. rgb_lagged_sums inverse ─────────────────────────────────────────

# Lags worth forcing: powers of two (alignment and stride readers), small
# primes (defeat naive stride detection), and common window sizes.
_LAGS = (1, 2, 3, 4, 5, 7, 8, 16, 17, 32, 64, 128, 255, 256)


def lag_correlate(data: bytes, rng=None) -> bytes:
    """Make a region exactly periodic at a chosen lag.

    rgb_lagged_sums correlates the stream against itself at lag ``L`` and
    expects no signal.  Forcing ``buf[i] == buf[i - L]`` produces the maximum
    possible signal, and with it the maximum possible LZ77 match length: a
    match finder sees one enormous back-reference, which is the shape of a
    decompression bomb and of the pathological cases in delta filters and RLE
    decoders.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    lag = rng.choice(_LAGS)
    offset, length = _region(len(data), rng, min_len=lag * 2)
    if length < lag * 2:
        return data
    out = bytearray(data)
    period = bytes(out[offset : offset + lag])
    reps = -(-length // lag)
    out[offset : offset + length] = (period * reps)[:length]
    return bytes(out)


# ── 7. dab_dct inverse ─────────────────────────────────────────────────

_SPECTRAL_MODES = ("cosine", "dc", "nyquist", "impulse", "max_ac")
_DC_LEVELS = (0x00, 0x01, 0x7F, 0x80, 0xFF)


def spectral_peak(data: bytes, rng=None) -> bytes:
    """Overwrite a region with a spectrally degenerate signal.

    dab_dct transforms blocks of the stream and checks that the position of
    the largest coefficient is uniform.  Each mode here pins that position: a
    pure cosine puts all energy in one bin, a constant block puts it all in
    DC, Nyquist alternation puts it in the last bin, and an impulse spreads
    it perfectly flat.  For a DCT-based codec these are the blocks whose
    inverse transform reaches the saturation and clamping arithmetic --
    ``max_ac`` in particular is the standard IDCT overflow probe.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=8)
    if length < 8:
        return data
    mode = rng.choice(_SPECTRAL_MODES)

    if mode == "dc":
        block = bytes([rng.choice(_DC_LEVELS)]) * length
    elif mode == "nyquist":
        pair = (0x00, 0xFF) if rng.random() < 0.5 else (0x80, 0x7F)
        block = bytes(pair[i & 1] for i in range(length))
    elif mode == "impulse":
        arr = bytearray(length)
        arr[rng.randint(0, length - 1)] = 0xFF
        block = bytes(arr)
    elif mode == "max_ac":
        # Full-scale extremes alternating at the transform's block period:
        # the worst case for an 8-point IDCT's intermediate range.
        block = bytes(0xFF if (i >> 3) & 1 else 0x00 for i in range(length))
    else:
        # A cosine sampled at an exact bin of an 8-point transform, so all
        # energy lands in a single coefficient.
        bin_index = rng.randint(1, 7)
        block = bytes(
            int(127.5 + 127.0 * math.cos(math.pi * bin_index * (i + 0.5) / 8.0)) & 0xFF
            for i in range(length)
        )
    return _splice(data, offset, block)


# ── 8. diehard_birthdays / dab_birthdays1 inverse ──────────────────────

# Powers of two dominate: a progression with a power-of-two common difference
# collides under any bucket count that is itself a power of two, which is the
# common case for hash tables.
_BIRTHDAY_DELTAS = (
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    16,
    16,
    16,
    16,
    256,
    256,
    256,
    256,
    4096,
    4096,
    4096,
    65536,
    65536,
)


def birthday_collide(data: bytes, rng=None) -> bytes:
    """Overwrite a region with words whose birthday spacings all coincide.

    The birthday test sorts the sampled words and checks that the *spacings*
    between them look Poisson -- duplicate spacings should be rare.  An
    arithmetic progression makes every spacing identical, the
    maximum-duplication tail.  Downstream that is the classic hash-flooding
    shape, so hash tables, dedup logic and bloom filters degrade toward their
    linear-probe worst case.

    With probability 1/4 the progression collapses to literal repeats of one
    word, the degenerate limit at spacing zero.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    width = rng.choice(_WIDTHS_WORD)
    offset, length = _region(len(data), rng, min_len=width * 4, align=width)
    if length < width * 4:
        return data
    big_endian = rng.random() < 0.5
    n = length // width
    base = rng.randint(0, (1 << (width * 8)) - 1)
    if rng.random() < 0.25:
        values = [base] * n
    else:
        delta = rng.choice(_BIRTHDAY_DELTAS)
        values = [base + i * delta for i in range(n)]
    return _splice(data, offset, _pack_words(values, width, big_endian))


# ── 9. rgb_persist inverse ─────────────────────────────────────────────

# Header fields are far more often length- or version-checked than compared
# for inequality, so bias toward the values those checks turn on.
_HEADER_VALUES = (
    0x00,
    0x00,
    0x00,
    0x01,
    0x01,
    0x01,
    0x7F,
    0x7F,
    0x7F,
    0x80,
    0x80,
    0x80,
    0xFF,
    0xFF,
    0xFF,
    0xFF,
)


def invariant_break(data: bytes, invariants, rng=None) -> bytes:
    """Randomise exactly the bytes the corpus never varies.

    rgb_persist reports the bits of a generator's output that never change.
    Applied to a corpus rather than an RNG, the same measurement finds the
    offsets every accepted input agrees on: magic numbers, version fields,
    fixed-width headers, structural constants.  Those are precisely the
    offsets ordinary mutation must leave alone to keep an input parseable,
    and therefore the ones whose validation code is least explored.

    This operator inverts the usual protection: it freezes the variable bytes
    and scribbles only on the invariant ones.

    Args:
        data: Input bytes.
        invariants: A ``CorpusInvariants`` from
            :func:`fuzzer_tool.core.randomness.corpus_invariants`, or None.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    if not data or invariants is None:
        return data
    n = len(data)
    # Fully-locked offsets are the obvious targets; the partially-locked ones
    # are the more informative, because a 0xF0 mask on a length field means
    # the corpus never drove the field past its low nibble. Writing only the
    # locked bits leaves the varying bits alone, which is what keeps the rest
    # of the record self-consistent while the untested range gets probed.
    sites = [(o, 0xFF) for o in invariants.fixed_offsets if o < n]
    sites += [(o, m) for o, m in invariants.partial_offsets if o < n]
    if not sites:
        return data
    n_hits = min(len(sites), rng.randint(1, 8))
    out = bytearray(data)
    for idx, mask in rng.sample(sites, n_hits):
        value = rng.choice(_HEADER_VALUES) if rng.random() < 0.75 else rng.randint(0, 255)
        out[idx] = (out[idx] & ~mask & 0xFF) | (value & mask)
    return bytes(out)


# ── 10. diehard_parking_lot / rgb_minimum_distance inverse ─────────────

_GEOMETRY_MODES = ("coincident", "collinear", "origin")
# Coordinate widths and dimensionalities, as module constants rather than
# inline literals so a test can pin one and read the result back at a
# known alignment.
_GEOMETRY_WIDTHS = (2, 4, 8)
_GEOMETRY_DIMS = (2, 3)


def degenerate_geometry(data: bytes, rng=None) -> bytes:
    """Overwrite a region with coincident or collinear coordinate tuples.

    The parking-lot and minimum-distance tests both measure how close the
    closest sampled pair of points gets; a uniform stream keeps them apart.
    Driving the minimum distance to zero (coincident points), or putting
    every point on one line, is the degenerate input for hull, triangulation,
    collision and area code -- the branches that divide by a distance, a
    determinant or a cross product, and yield a NaN, an infinity or a
    division by zero when it vanishes.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    width = rng.choice(_GEOMETRY_WIDTHS)
    dims = rng.choice(_GEOMETRY_DIMS)
    point_size = width * dims
    offset, length = _region(len(data), rng, min_len=point_size * 2, align=point_size)
    if length < point_size * 2:
        return data
    big_endian = rng.random() < 0.5
    mode = rng.choice(_GEOMETRY_MODES)

    if mode == "origin":
        base = [0] * dims
        step = [0] * dims
    else:
        base = [rng.randint(0, (1 << (width * 8)) - 1) for _ in range(dims)]
        step = [0] * dims if mode == "coincident" else [rng.randint(1, 16)] * dims

    values: list[int] = []
    for i in range(length // point_size):
        values.extend(base[d] + step[d] * i for d in range(dims))
    return _splice(data, offset, _pack_words(values, width, big_endian))


# ── 11. diehard_squeeze inverse ────────────────────────────────────────

# IEEE-754 bit patterns that break iterative numeric loops. Each entry pairs
# the float64 and float32 encodings of the *same* value class, so the width
# choice picks a semantically matching pattern rather than a truncation.
_FLOAT_PATTERNS = (
    (0x3FEFFFFFFFFFFFFF, 0x3F7FFFFF),  # largest value below 1.0
    (0x3FF0000000000001, 0x3F800001),  # smallest value above 1.0
    (0x0000000000000001, 0x00000001),  # smallest denormal
    (0x000FFFFFFFFFFFFF, 0x007FFFFF),  # largest denormal
    (0x7FEFFFFFFFFFFFFF, 0x7F7FFFFF),  # largest finite
    (0x7FF0000000000000, 0x7F800000),  # +inf
    (0xFFF0000000000000, 0xFF800000),  # -inf
    (0x7FF8000000000000, 0x7FC00000),  # quiet NaN
    (0x7FF0000000000001, 0x7F800001),  # signalling NaN
    (0x8000000000000000, 0x80000000),  # negative zero
)

_FLOAT_WIDTHS = (4, 8)


def float_squeeze(data: bytes, rng=None) -> bytes:
    """Overwrite a region with pathological IEEE-754 values.

    The squeeze test counts iterations of ``k = ceil(k * U)`` until ``k``
    reaches 1, with ``U`` floated from the stream.  Its tails are the values
    that never terminate -- ``U`` indistinguishable from 1.0 -- and the ones
    that terminate at once.  Generalised into a mutation, that is the set of
    float bit patterns that break convergence outright: one ulp from 1.0,
    denormals, infinities, NaN payloads.  Any target that parses floats, or
    reinterprets attacker bytes as floats, has a loop or a comparison
    somewhere that these do not terminate.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    width = rng.choice(_FLOAT_WIDTHS)
    offset, length = _region(len(data), rng, min_len=width, align=width)
    if length < width:
        return data
    big_endian = rng.random() < 0.5
    idx = 1 if width == 4 else 0
    values = [rng.choice(_FLOAT_PATTERNS)[idx] for _ in range(length // width)]
    return _splice(data, offset, _pack_words(values, width, big_endian))


# ── 12. diehard_count_1s inverse ───────────────────────────────────────

# 0 and 8 are the all-zero/all-ones degenerate cases; 4 is the largest weight
# class and the one a weight-based validator is most likely to accept.
_POPCOUNT_WEIGHTS = (0, 0, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 7, 7, 8, 8, 8)


@lru_cache(maxsize=16)
def _popcount_table(weight: int) -> bytes:
    """256-entry translate table onto the bytes of Hamming weight *weight*."""
    return _map_table(bytes(b for b in range(256) if b.bit_count() == weight))


def popcount_lock(data: bytes, rng=None) -> bytes:
    """Overwrite a region with bytes of a single Hamming weight.

    diehard_count_1s maps each byte to one of five letters by population
    count and checks the resulting word frequencies; a uniform stream gives
    the binomial spread.  Pinning every byte to one weight collapses that to
    a single letter.  Fixed-weight byte streams are the natural input for
    bit-packed formats, for the validity classes of UTF-8 and Base64 (which
    are themselves popcount-delimited), for ECC and constant-weight codes,
    and for SIMD popcount fast paths whose scalar tail is rarely reached.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=4)
    if length < 4:
        return data
    weight = rng.choice(_POPCOUNT_WEIGHTS)
    block = rng.randbytes(length).translate(_popcount_table(weight))
    return _splice(data, offset, block)
