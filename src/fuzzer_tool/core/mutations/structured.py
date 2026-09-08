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
from collections import Counter
from functools import lru_cache

from fuzzer_tool.core.debruijn_cache import fingerprint as _db_fingerprint
from fuzzer_tool.core.debruijn_cache import load as _db_cache_load
from fuzzer_tool.core.debruijn_cache import store as _db_cache_store
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


def _region(
    data_len: int, rng, min_len: int = 1, align: int = 1, max_len: int | None = None
) -> tuple[int, int]:
    """Pick a random ``(offset, length)`` window to overwrite.

    Args:
        data_len: Length of the buffer being mutated.
        rng: RandPool or stdlib random.
        min_len: Smallest window the caller's construction needs.
        align: Snap the offset down to a multiple of this so the caller's
            fixed-width words line up with how a reader slices them. Without
            it an unaligned run reads as ordinary noise at the reader's own
            stride, silently defeating the construction.
        max_len: When set, the returned length is rounded down to a multiple
            of this value.  This lets callers enforce width-specific
            invariants (e.g. bit-plane interleaving requires multiples of 8)
            without altering the global ``_region`` contract for everyone
            else.

    Returns:
        ``(offset, length)`` with ``offset + length <= data_len``, or
        ``(0, 0)`` when the buffer is shorter than *min_len*.
    """
    if data_len < min_len or min_len < 1:
        return 0, 0
    span = min(MAX_REGION, data_len)
    if span < min_len:
        return 0, 0
    hi = span if max_len is None else span - (span % max_len)
    length = rng.randint(min_len, hi)
    if max_len is not None:
        length -= length % max_len
        if length < min_len:
            return 0, 0
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


# Fingerprint of the construction algorithm, folded into every disk-cache
# filename (see core/debruijn_cache.py, handover 10f): computed once at
# import time since _de_bruijn_symbols is fixed for the life of the
# process, rather than re-hashing its source on every cache access.
_DB_FINGERPRINT = _db_fingerprint(_de_bruijn_symbols)


@lru_cache(maxsize=32)
def de_bruijn_bytes(k: int, n: int) -> bytes:
    """Cached de Bruijn sequence B(k, n) rendered as bytes (``k <= 256``).

    Returns an immutable ``bytes`` precisely so the cache cannot be mutated
    through: caching a mutable sequence here would corrupt every later caller
    the first time an operator wrote through the result.

    Checks the on-disk cache (shared across processes on the same machine)
    before falling back to construction, and populates it on a miss --
    the in-memory ``@lru_cache`` above this still avoids the disk round
    trip for repeat calls within one process.
    """
    cache_key = f"k{k}_n{n}"
    cached = _db_cache_load("bytes", cache_key, _DB_FINGERPRINT)
    if cached is not None:
        return cached
    step = 256 // k if k < 256 else 1
    data = bytes((s * step) & 0xFF for s in _de_bruijn_symbols(k, n))
    _db_cache_store("bytes", cache_key, _DB_FINGERPRINT, data)
    return data


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

    Same on-disk cache as :func:`de_bruijn_bytes` (see handover 10f),
    keyed by ``n`` alone since the alphabet is fixed at k=2 here.
    """
    n = max(n, 3)  # below 3, 2**n is not byte-aligned once packed
    cache_key = f"n{n}"
    cached = _db_cache_load("bits", cache_key, _DB_FINGERPRINT)
    if cached is not None:
        return cached
    data = _pack_bits_msb(_de_bruijn_symbols(2, n))
    _db_cache_store("bits", cache_key, _DB_FINGERPRINT, data)
    return data


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


# ── 13. MTF / BWT / RLE round-trip transforms ────────────────────────────
#
# These are not diehard/dieharder inverses; they are reversible byte-domain
# transforms that produce mutation shapes no existing operator can reach:
#
#   * mtf  — edits near frequent/recent symbols in the original byte stream.
#   * bwt  — edits in BWT space scatter across every position sharing the
#            same following context in the original.
#   * rle  — edits run lengths/values, changing repetition structure.
#
# All three are length-preserving: the mutate-then-invert path always returns
# a buffer of exactly len(data) bytes, or declines on inputs too short to
# transform meaningfully.


def _mtf_encode(data: bytes, alphabet: bytearray) -> bytes:
    """Encode *data* with Move-To-Front, mutating *alphabet* in place."""
    out = bytearray(len(data))
    for i, b in enumerate(data):
        idx = alphabet.index(b)
        out[i] = idx & 0xFF
        del alphabet[idx]
        alphabet.insert(0, b)
    return bytearray(out)


def _mtf_decode(encoded: bytes, alphabet: bytearray) -> bytes:
    """Decode MTF-encoded *encoded* using a fresh copy of *alphabet*."""
    syms = list(alphabet)
    out = bytearray(len(encoded))
    for i, idx in enumerate(encoded):
        out[i] = syms[idx]
        del syms[idx]
        syms.insert(0, out[i])
    return bytes(out)


def mtf(data: bytes, rng=None) -> bytes:
    """Move-To-Front encode, edit MTF indices, decode back.

    A one-byte edit in MTF-index space becomes a value-correlated edit in the
    original: small indices correspond to frequent/recent symbols, so editing
    them nudges common bytes rather than rare ones.  That is the opposite of
    uniform random overwrite and exercises fast paths keyed on byte-value
    distribution.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    alphabet = bytearray(range(256))
    encoded = _mtf_encode(data, alphabet)
    # Mutate 1–3 low indices (frequent symbols) in the MTF domain.
    n_mut = rng.randint(1, min(3, len(encoded)))
    for _ in range(n_mut):
        pos = rng.randint(0, len(encoded) - 1)
        # Weight toward small indices: frequent symbols sit near 0 after MTF.
        encoded[pos] = rng.randint(0, min(15, len(encoded) - 1)) & 0xFF
    alphabet = bytearray(range(256))
    return _mtf_decode(encoded, alphabet)


def _bwt(data: bytes) -> tuple[bytes, int]:
    """Burrows–Wheeler transform: returns (bwt_data, primary_key)."""
    n = len(data)
    if n <= 1:
        return data, 0
    rotations = [data[i:] + data[:i] for i in range(n)]
    rotations.sort()
    primary = rotations.index(data)
    return bytes(row[-1] for row in rotations), primary


def _bwt_inverse(bwt_data: bytes, primary: int) -> bytes:
    """Inverse Burrows–Wheeler transform."""
    n = len(bwt_data)
    if n <= 1:
        return bwt_data
    table = sorted((bwt_data[i], i) for i in range(n))
    # First column F and rank lookup for the LF-mapping.
    F = [table[i][0] for i in range(n)]
    rows_f: dict[int, list[int]] = {}
    for i, c in enumerate(F):
        rows_f.setdefault(c, []).append(i)
    idx = primary
    out = bytearray(n)
    for i in range(n):
        c = bwt_data[idx]
        out[n - 1 - i] = c
        rank = sum(1 for j in range(idx + 1) if bwt_data[j] == c)
        idx = rows_f[c][rank - 1]
    return bytes(out)


def bwt(data: bytes, rng=None) -> bytes:
    """BWT + MTF round-trip: transform, edit in BWT+MTF space, invert back.

    BWT groups bytes by following context; a one-byte edit in BWT space
    scatters across every original position sharing that context.  MTF on top
    converts the BWT output to small indices for frequent symbols, so the
    mutation lands on structurally important bytes rather than noise.

    The primary-key byte needed for inverse BWT is preserved unchanged; it is
    not part of the MTF domain and is never mutated.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    block_size = rng.choice((8, 16, 32, 64, 128, 256))
    block_size = min(block_size, len(data))
    offset = rng.randint(0, len(data) - block_size)
    block = data[offset : offset + block_size]
    bwt_data, primary = _bwt(block)
    alphabet = bytearray(range(256))
    encoded = _mtf_encode(bwt_data, alphabet)
    n_mut = rng.randint(1, min(3, len(encoded)))
    for _ in range(n_mut):
        pos = rng.randint(0, len(encoded) - 1)
        encoded[pos] = rng.randint(0, min(15, len(encoded) - 1)) & 0xFF
    alphabet = bytearray(range(256))
    decoded = _mtf_decode(encoded, alphabet)
    restored = _bwt_inverse(decoded, primary)
    return _splice(data, offset, restored)


def rle(data: bytes, rng=None) -> bytes:
    """Run-length encode, edit runs, decode back.

    Editing run lengths changes repetition structure in the original: merging
    adjacent runs creates longer uniform stretches, splitting creates shorter
    ones, and changing run values swaps which byte repeats.  These are the
    mutations a runs-test detector is most sensitive to and that RLE-compressed
    format parsers (BMP, TIFF, PCX) process along their fast paths.

    Length is preserved: run-length edits that would grow or shrink the total
    are compensated by adjusting a neighbouring run in the opposite direction.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(data):
        b = data[i]
        j = i + 1
        while j < len(data) and data[j] == b:
            j += 1
        runs.append((b, j - i))
        i = j
    if len(runs) < 2:
        return data
    n_mut = rng.randint(1, min(3, len(runs)))
    for _ in range(n_mut):
        pos = rng.randint(0, len(runs) - 1)
        if rng.random() < 0.6:
            # Mutate run value.
            runs[pos] = (rng.randint(0, 255), runs[pos][1])
        else:
            # Mutate run length, compensating to preserve total length.
            new_len = rng.randint(1, max(2, runs[pos][1] * 3))
            delta = new_len - runs[pos][1]
            if delta > 0 and len(runs) >= 2:
                other = rng.randint(0, len(runs) - 1)
                while other == pos:
                    other = rng.randint(0, len(runs) - 1)
                new_other = max(1, runs[other][1] - delta)
                delta -= runs[other][1] - new_other
                runs[other] = (runs[other][0], new_other)
                if delta > 0:
                    runs[pos] = (runs[pos][0], runs[pos][1] + delta)
            else:
                runs[pos] = (runs[pos][0], max(1, runs[pos][1] + delta))
    out = bytearray()
    for b, length in runs:
        out.extend([b] * length)
    if len(out) != len(data):
        return data
    return _splice(data, 0, bytes(out))


def delta_encode(data: bytes, rng=None) -> bytes:
    """First-difference encode a region, edit deltas, decode back.

    Delta encoding converts each byte to the difference from its predecessor.
    For correlated data (adjacent bytes differ by small amounts), deltas
    concentrate into a narrow range; a mutation in delta space becomes a
    smooth local perturbation in the original.  For uncorrelated data the
    deltas spread across the full byte range, so the mutation looks like
    ordinary noise.

    This is the inverse of a statistical test that rejects streams with
    extreme first-difference variance; fuzzing the delta domain exercises
    parser fast-paths keyed on sequential smoothness.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=2)
    if length < 2:
        return data
    block = data[offset : offset + length]
    deltas = bytearray(length)
    deltas[0] = block[0]
    for i in range(1, length):
        deltas[i] = (block[i] - block[i - 1]) & 0xFF
    n_mut = rng.randint(1, min(4, length))
    for _ in range(n_mut):
        pos = rng.randint(0, length - 1)
        delta = rng.randint(-16, 16)
        deltas[pos] = (deltas[pos] + delta) & 0xFF
    restored = bytearray(length)
    restored[0] = deltas[0]
    for i in range(1, length):
        restored[i] = (restored[i - 1] + deltas[i]) & 0xFF
    return _splice(data, offset, bytes(restored))


def delta_sigma(data: bytes, rng=None) -> bytes:
    """Predictive first-order delta-sigma modulate a region, edit, demodulate.

    Each byte is encoded as the prediction error from a running accumulated
    sum rather than from the immediate predecessor.  For smooth input the
    errors concentrate near zero; for abrupt changes they grow.  Editing the
    error stream changes the cumulative offset of the reconstructed tail, so
    a single edit propagates through many subsequent values.

    Targets parsers that accumulate running checksums, running totals, or
    stateful decoders where a small perturbation cascades through many
    subsequent decoded values.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=2)
    if length < 2:
        return data
    block = data[offset : offset + length]
    modulated = bytearray(length)
    acc = block[0]
    modulated[0] = block[0]
    for i in range(1, length):
        modulated[i] = (block[i] - acc) & 0xFF
        acc = (acc + modulated[i]) & 0xFF
    n_mut = rng.randint(1, min(4, length))
    for _ in range(n_mut):
        pos = rng.randint(0, length - 1)
        delta = rng.randint(-32, 32)
        modulated[pos] = (modulated[pos] + delta) & 0xFF
    restored = bytearray(length)
    acc = modulated[0]
    restored[0] = acc
    for i in range(1, length):
        acc = (acc + modulated[i]) & 0xFF
        restored[i] = acc
    return _splice(data, offset, bytes(restored))


def bitcast_float(data: bytes, rng=None) -> bytes:
    """Reinterpret a region as float/double, mutate, write back as bytes.

    Targets parsers that decode floating-point fields: image pixel formats,
    scientific-data containers, audio sample headers, 3D mesh normals/UVs,
    and SIMD-friendly binary formats where a NaN or infinity triggers a
    missing-error-handling path.

    The mutation space is the IEEE 754 value domain, not the byte domain:
    a single edit can produce NaN, +/-inf, subnormal, or a huge magnitude
    that overflows a parser's fixed-point fallback.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    width = rng.choice((4, 8))
    if len(data) < width:
        return data
    offset = rng.randint(0, len(data) - width)
    endian = "<" if rng.random() < 0.5 else ">"
    fmt = f"{endian}f" if width == 4 else f"{endian}d"
    try:
        value = struct.unpack(fmt, data[offset : offset + width])[0]
    except struct.error:
        return data
    strategy = rng.randint(0, 4)
    if strategy == 0:
        value = 0.0
    elif strategy == 1:
        value = float("inf")
    elif strategy == 2:
        value = float("-inf")
    elif strategy == 3:
        value = float("nan")
    else:
        value = value * 1e3 if value != 0.0 else 1e38
    try:
        packed = struct.pack(fmt, value)
    except (struct.error, OverflowError):
        return data
    out = bytearray(data)
    out[offset : offset + width] = packed
    return bytes(out)


def bitcast_int32(data: bytes, rng=None) -> bytes:
    """Reinterpret a region as signed/unsigned int, mutate, write back.

    Targets integer-overflow and truncation bugs in parsers that read
    size/count/offset fields: a 4-byte length read as int32 with value
    0x7FFFFFFF becomes 0xFFFFFFFF when sign-extended to int64, or a
    2-byte count of 0xFFFF becomes 0 after a `short + 1` increment.

    The mutation overwrites with values that are canonical overflow
    triggers: 0, -1, MAX_SIGNED, MAX_UNSIGNED, and a few shifted
    variants that survive one sanitizer pass.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    width = rng.choice((2, 4, 8))
    if len(data) < width:
        return data
    offset = rng.randint(0, len(data) - width)
    endian = "<" if rng.random() < 0.5 else ">"
    if width == 2:
        signed_fmt = f"{endian}h"
    elif width == 4:
        signed_fmt = f"{endian}i"
    else:
        signed_fmt = f"{endian}q"
    try:
        signed_val = struct.unpack(signed_fmt, data[offset : offset + width])[0]
    except struct.error:
        return data
    max_signed = (1 << (width * 8 - 1)) - 1
    max_unsigned = (1 << (width * 8)) - 1
    strategy = rng.randint(0, 5)
    if strategy == 0:
        new_val = 0
    elif strategy == 1:
        new_val = -1
    elif strategy == 2:
        new_val = max_signed
    elif strategy == 3:
        new_val = max_unsigned
    elif strategy == 4:
        new_val = signed_val + max_signed
    else:
        new_val = signed_val - max_signed - 1
    try:
        packed = struct.pack(signed_fmt, new_val & max_unsigned)
    except (struct.error, OverflowError):
        return data
    out = bytearray(data)
    out[offset : offset + width] = packed
    return bytes(out)


def size_field_overflow(data: bytes, rng=None) -> bytes:
    """Overwrite plausible size/count fields with overflow trigger values.

    Scans the buffer for values that look like declared-size or count
    fields — positions where a 2/4/8-byte integer is followed by at
    least that many bytes of payload — and overwrites them with the
    canonical overflow triggers: 0, -1, MAX_SIGNED, MAX_UNSIGNED.

    Targets format parsers that compute memory allocation from a
    size field without upper-bounds checking.  Classic failure mode:
    `malloc(size_field)` with `size_field = 0xFFFFFFFF` on a 32-bit
    target, or `size_field = -1` interpreted as a huge unsigned count.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    width = rng.choice((2, 4))
    if len(data) < width + 1:
        return data
    max_offset = len(data) - width
    # Prefer offsets where the current value looks like a plausible size
    # field: non-negative, not obviously ASCII, and the buffer extends past it.
    candidates = []
    for off in range(0, max_offset, max(1, width)):
        raw = data[off : off + width]
        if len(raw) < width:
            continue
        if all(32 <= b < 127 for b in raw):
            continue  # Looks like text, not a size field.
        candidates.append(off)
    if not candidates:
        offset = rng.randint(0, max_offset)
    else:
        offset = candidates[rng.randint(0, len(candidates) - 1)]
    endian = "<" if rng.random() < 0.5 else ">"
    fmt = f"{endian}H" if width == 2 else f"{endian}I"
    max_signed = (1 << (width * 8 - 1)) - 1
    max_unsigned = (1 << (width * 8)) - 1
    triggers = (0, -1, max_signed, max_unsigned)
    new_val = triggers[rng.randint(0, len(triggers) - 1)]
    try:
        packed = struct.pack(fmt, new_val & max_unsigned)
    except (struct.error, OverflowError):
        return data
    out = bytearray(data)
    out[offset : offset + width] = packed
    return bytes(out)


def bpe(data: bytes, rng=None) -> bytes:
    """Byte-pair encode a region, edit token stream, expand back.

    BPE finds the most frequent adjacent byte pairs in the input and merges
    them into single tokens.  The token stream is shorter than the original
    byte stream; mutations in token space change multiple adjacent bytes at
    once, probing parser fast-paths keyed on repeated multi-byte sequences.

    The merge is deterministic and data-local: no external vocabulary is
    needed, and the same input always produces the same merge set.  A
    mutation in token space expands back to a byte stream of exactly the
    original length because each token expands to 1 or 2 bytes and the
    merge/unmerge counts balance.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=4)
    if length < 4:
        return data
    block = data[offset : offset + length]
    # Count all adjacent byte pairs.
    pair_counts: Counter[tuple[int, int]] = Counter()
    for i in range(len(block) - 1):
        pair_counts[(block[i], block[i + 1])] += 1
    if not pair_counts:
        return data
    # Pick top-N pairs greedily, descending by count.
    sorted_pairs = [p for p, _ in sorted(pair_counts.items(), key=lambda x: -x[1])]
    n_merge = rng.randint(1, min(5, len(sorted_pairs)))
    merge_set = set(sorted_pairs[:n_merge])
    # Greedy left-to-right merge pass.
    tokens: list[tuple[int, int]] = []  # (lo, hi); hi=-1 for unmerged singletons.
    i = 0
    while i < len(block):
        if i + 1 < len(block) and (block[i], block[i + 1]) in merge_set:
            tokens.append((block[i], block[i + 1]))
            i += 2
        else:
            tokens.append((block[i], -1))
            i += 1
    # Mutate the token stream.
    n_mut = rng.randint(1, max(2, len(tokens) // 2))
    for _ in range(n_mut):
        pos = rng.randint(0, len(tokens) - 1)
        action = rng.randint(0, 2)
        if action == 0:
            # Swap token with neighbour.
            neighbour = rng.randint(0, len(tokens) - 1)
            while neighbour == pos:
                neighbour = rng.randint(0, len(tokens) - 1)
            tokens[pos], tokens[neighbour] = tokens[neighbour], tokens[pos]
        elif action == 1:
            # Replace with random single-byte token.
            tokens[pos] = (rng.randint(0, 255), -1)
        else:
            # Delete token and reinsert at random position.
            tok = tokens.pop(pos)
            ins = rng.randint(0, len(tokens))
            tokens.insert(ins, tok)
    # Expand tokens back to bytes; length is preserved because each token
    # expands to exactly 1 or 2 bytes and the number of tokens equals the
    # number of merge pairs subtracted from the original length.
    restored = bytearray(length)
    j = 0
    for lo, hi in tokens:
        restored[j] = lo
        j += 1
        if hi != -1:
            restored[j] = hi
            j += 1
    return _splice(data, offset, bytes(restored[:length]))


def golomb(data: bytes, rng=None) -> bytes:
    """Golomb/Rice code a region, edit codewords, decode back.

    Golomb coding is the optimal prefix code for geometric-distribution
    integers.  Many binary formats encode run lengths, coefficient counts,
    and delta values with Golomb/Rice codes because they concentrate
    small integers into very short bit sequences.

    This operator edits the codeword stream in place, preserving the
    codebook parameters.  A single edit can turn a short codeword into
    a long one or vice versa, probing parser decoders that assume
    codeword lengths stay within a valid range.

    Uses Rice parameter k=4 as a reasonable default; the codebook is
    fixed for the duration of the mutation so the decoder sees a
    consistent stream.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=2)
    if length < 2:
        return data
    block = bytearray(data[offset : offset + length])
    k = rng.choice((2, 3, 4, 5, 6))
    mask = (1 << k) - 1
    # Encode to Golomb-Rice bitstream (LSB-first quotient, then remainder).
    bits = bytearray()
    for b in block:
        q = b >> k
        r = b & mask
        # quotient in unary: q ones followed by a zero.
        bits.extend([1] * q + [0])
        # remainder in k bits, LSB-first.
        for i in range(k):
            bits.append((r >> i) & 1)
    if not bits:
        return data
    # Mutate the bitstream: flip/toggle/insert/delete bits.
    n_mut = rng.randint(1, min(8, len(bits) // 2 + 1))
    for _ in range(n_mut):
        action = rng.randint(0, 3)
        pos = rng.randint(0, len(bits) - 1)
        if action == 0:
            bits[pos] ^= 1
        elif action == 1 and len(bits) < length * 16:
            bits.insert(pos, rng.randint(0, 1))
        elif action == 2 and len(bits) > length // 2:
            del bits[pos]
        elif action == 3:
            bits[pos] = 1 if bits[pos] == 0 else 0
    # Decode back: read unary quotient then k-bit remainder.
    restored = bytearray(length)
    ridx = 0
    bidx = 0
    while ridx < length and bidx < len(bits):
        q = 0
        while bidx < len(bits) and bits[bidx] == 1:
            q += 1
            bidx += 1
        if bidx >= len(bits):
            break
        bidx += 1  # skip the terminating 0.
        r = 0
        for i in range(k):
            if bidx + i < len(bits):
                r |= bits[bidx + i] << i
        restored[ridx] = ((q << k) | r) & 0xFF
        ridx += 1
        bidx += k
    # Pad any remaining bytes with random values to keep length fixed.
    while ridx < length:
        restored[ridx] = rng.randint(0, 255)
        ridx += 1
    return _splice(data, offset, bytes(restored[:length]))


def endian_convert(data: bytes, rng=None) -> bytes:
    """Flip endianness of integer fields in a region.

    Many binary formats store multi-byte integers in a fixed endianness,
    but mixed-endian protocols, network-byte-order fields adjacent to
    native-order fields, and misparsed headers all create situations
    where the parser's byte-swap assumption is wrong.  Flipping
    endianness of a region probes exactly those paths.

    The operator scans for runs of 2, 4, or 8-byte integers in a
    randomly-selected region and reverses each word in place.  The
    width is uniform across the region so adjacent fields stay
    aligned to the chosen stride.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=2)
    if length < 2:
        return data
    width = rng.choice((2, 4, 8))
    # Align offset to word boundary.
    if offset % width != 0:
        offset += width - (offset % width)
        length = max(0, data[offset : offset + length].__len__())
    if length < width:
        return data
    n_words = length // width
    if n_words == 0:
        return data
    block = bytearray(data[offset : offset + n_words * width])
    # Reverse bytes within each word.
    for i in range(0, len(block), width):
        block[i : i + width] = block[i : i + width][::-1]
    return _splice(data, offset, bytes(block))


def count_overflow(data: bytes, rng=None) -> bytes:
    """Overwrite count/n fields with integer-overflow trigger values.

    Like :func:`size_field_overflow`, but targets count fields rather
    than declared-size fields.  Count fields control how many elements
    a parser iterates over, so setting them to -1, MAX_SIGNED, or
    MAX_UNSIGNED triggers out-of-bounds reads, infinite loops, or
    allocation miscalculations.

    The scanner looks for 2/4-byte integers at aligned offsets whose
    value is small enough to plausibly be a count (not already at
    overflow range) and overwrites one with a trigger.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    width = rng.choice((2, 4))
    if len(data) < width + 1:
        return data
    max_offset = len(data) - width
    candidates = []
    for off in range(0, max_offset, max(1, width)):
        raw = data[off : off + width]
        if len(raw) < width:
            continue
        if all(32 <= b < 127 for b in raw):
            continue
        val = int.from_bytes(raw, "little")
        # Prefer fields whose current value is in a plausible count range.
        if 0 <= val < min(1024, (1 << (width * 8 - 1)) - 1):
            candidates.append(off)
    if not candidates:
        offset = rng.randint(0, max_offset)
    else:
        offset = candidates[rng.randint(0, len(candidates) - 1)]
    endian = "<" if rng.random() < 0.5 else ">"
    fmt = f"{endian}H" if width == 2 else f"{endian}I"
    max_signed = (1 << (width * 8 - 1)) - 1
    max_unsigned = (1 << (width * 8)) - 1
    triggers = (-1, max_signed, max_unsigned)
    new_val = triggers[rng.randint(0, len(triggers) - 1)]
    try:
        packed = struct.pack(fmt, new_val & max_unsigned)
    except (struct.error, OverflowError):
        return data
    out = bytearray(data)
    out[offset : offset + width] = packed
    return bytes(out)


def zero_run_amplify(data: bytes, rng=None) -> bytes:
    """Extend zero runs in a region.

    Zero runs trigger fast paths in image/video codecs, compression
    routines, and memset-optimized memory functions.  Amplifying a zero
    run probes those fast paths for missing length/width checks.

    The operator finds a zero run in a randomly-selected region and
    extends it by converting neighbouring non-zero bytes to zero,
    compensated by converting an equal number of zero bytes at the
    opposite end of the region back to random values so the total
    length is preserved.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=4)
    if length < 4:
        return data
    block = bytearray(data[offset : offset + length])
    # Find the longest contiguous zero run.
    best_start, best_len = 0, 0
    run_start, run_len = 0, 0
    for i in range(len(block)):
        if block[i] == 0:
            if run_len == 0:
                run_start = i
            run_len += 1
        else:
            if run_len > best_len:
                best_start, best_len = run_start, run_len
            run_len = 0
    if run_len > best_len:
        best_start, best_len = run_start, run_len
    if best_len < 1:
        return data
    # Extend the run by up to min(best_len, available non-zero neighbours).
    extend = rng.randint(1, min(best_len, len(block) - best_len))
    # Grow from both ends of the run.
    left = min(extend, best_start)
    right = min(extend - left, len(block) - best_start - best_len)
    for i in range(best_start - left, best_start):
        block[i] = 0
    for i in range(best_start + best_len, best_start + best_len + right):
        block[i] = 0
    # Compensate: overwrite 'extend' zero bytes at the region edges with
    # random values so the zero-count change is localised to the run.
    edges = []
    for i in range(len(block)):
        if block[i] == 0 and not (best_start - left <= i < best_start + best_len + right):
            edges.append(i)
    for i in rng.sample(edges, min(extend, len(edges))):
        block[i] = rng.randint(1, 255)
    return _splice(data, offset, bytes(block))


def zero_run_suppress(data: bytes, rng=None) -> bytes:
    """Break zero runs by injecting non-zero bytes.

    The inverse of :func:`zero_run_amplify`: finds a zero run and
    breaks it by overwriting bytes within the run with non-zero values.
    Targets parsers that special-case contiguous-zero regions — a
    broken zero run can change the execution path from a fast memset
    clone to a general-purpose copy loop.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=4)
    if length < 4:
        return data
    block = bytearray(data[offset : offset + length])
    # Find all zero runs longer than 1 byte.
    runs = []
    run_start, run_len = 0, 0
    for i in range(len(block)):
        if block[i] == 0:
            if run_len == 0:
                run_start = i
            run_len += 1
        else:
            if run_len > 1:
                runs.append((run_start, run_len))
            run_len = 0
    if run_len > 1:
        runs.append((run_start, run_len))
    if not runs:
        return data
    start, run_len = runs[rng.randint(0, len(runs) - 1)]
    # Break the run by overwriting 1..min(3, run_len-1) bytes.
    n_break = rng.randint(1, min(3, run_len - 1))
    positions = rng.sample(range(start, start + run_len), n_break)
    for pos in positions:
        block[pos] = rng.randint(1, 255)
    return _splice(data, offset, bytes(block))


def type_promote(data: bytes, rng=None) -> bytes:
    """Simulate a buggy integer promotion at a plausible field.

    Type promotion bugs arise when a parser reads a small integer and
    widens it without sign-extension or overflow checking: a 2-byte
    unsigned count of 0xFFFF becomes 0 after `short + 1`, or a 1-byte
    signed value of -1 becomes 0xFFFFFFFF when zero-extended to 32 bits.

    This operator picks a 1/2/4-byte field, reads its current value,
    and writes back a value that simulates a common promotion bug:
    sign-extension of an unsigned value, zero-extension of a signed
    value, or overflow past the field width.  The field width stays
    the same so the result is always length-preserving.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    width = rng.choice((1, 2, 4))
    if len(data) < width + 1:
        return data
    max_offset = len(data) - width
    offset = rng.randint(0, max_offset)
    raw = data[offset : offset + width]
    val = int.from_bytes(raw, "little", signed=False)
    max_val = (1 << (width * 8)) - 1
    max_signed = (1 << (width * 8 - 1)) - 1
    strategy = rng.randint(0, 3)
    if strategy == 0:
        # Zero-extend a signed-looking value: clear the high bit.
        new_val = val & max_signed if val & (1 << (width * 8 - 1)) else val | (1 << (width * 8 - 1))
    elif strategy == 1:
        # Sign-extend an unsigned-looking value: set/copy high bit.
        new_val = (
            val | ((1 << (width * 8 - 1)) - 1) if val > max_signed else val | (1 << (width * 8 - 1))
        )
    elif strategy == 2:
        # Overflow past width: add/subtract a large offset.
        new_val = (val + (1 << (width * 8 - 1))) & max_val
    else:
        # Truncate-as-promote: write max_signed or 0 regardless of input.
        new_val = rng.choice((0, max_signed, max_val))
    endian = "<" if rng.random() < 0.5 else ">"
    fmt = f"{endian}H" if width == 2 else f"{endian}I"
    if width == 1:
        packed = bytes([new_val & 0xFF])
    else:
        try:
            packed = struct.pack(fmt, new_val & max_val)
        except (struct.error, OverflowError):
            return data
    out = bytearray(data)
    out[offset : offset + width] = packed
    return bytes(out)


def length_miscalculate(data: bytes, rng=None) -> bytes:
    """Overwrite length/size fields with values that contradict actual payload.

    Many format parsers compute a checksum, allocate a buffer, or enter a
    loop based on a declared length field without verifying that the
    declared length matches the actual payload on disk.  Setting the
    length to 0, -1, or a value larger than the real payload triggers
    allocation failures, infinite loops, or out-of-bounds reads.

    The operator scans for 2/4-byte fields at aligned offsets whose
    current value is within a plausible range and overwrites one with
    a contradictory length.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    width = rng.choice((2, 4))
    if len(data) < width + 1:
        return data
    max_offset = len(data) - width
    candidates = []
    for off in range(0, max_offset, max(1, width)):
        raw = data[off : off + width]
        if len(raw) < width:
            continue
        if all(32 <= b < 127 for b in raw):
            continue
        val = int.from_bytes(raw, "little")
        # Prefer fields whose value is in a plausible length range.
        max_plausible = min(len(data) - off, 1 << (width * 8 - 1))
        if 0 <= val <= max_plausible:
            candidates.append(off)
    if not candidates:
        offset = rng.randint(0, max_offset)
    else:
        offset = candidates[rng.randint(0, len(candidates) - 1)]
    endian = "<" if rng.random() < 0.5 else ">"
    fmt = f"{endian}H" if width == 2 else f"{endian}I"
    max_unsigned = (1 << (width * 8)) - 1
    # Contradictory lengths: 0, -1, MAX, or larger than actual remaining payload.
    actual_remainder = len(data) - offset - width
    choices = (0, -1, max_unsigned, actual_remainder + rng.randint(1, 256))
    new_val = choices[rng.randint(0, len(choices) - 1)]
    try:
        packed = struct.pack(fmt, new_val & max_unsigned)
    except (struct.error, OverflowError):
        return data
    out = bytearray(data)
    out[offset : offset + width] = packed
    return bytes(out)


def elias_gamma(data: bytes, rng=None) -> bytes:
    """Elias gamma encode a region, edit codewords, decode back.

    Elias gamma coding is a universal code for positive integers: the
    codeword for integer N consists of floor(log2 N) zero bits followed
    by the (log2 N + 1)-bit binary representation of N.  Small integers
    get short codewords; large integers get long ones.

    Many binary formats use gamma or similar universal codes for
    variable-length integer fields (Google Protocol Buffers, MPEG,
    FLAC).  Editing the codeword stream probes the decoder's length
    parsing and variable-length integer reconstruction.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=2)
    if length < 2:
        return data
    block = data[offset : offset + length]
    # Encode each byte as Elias gamma (1-indexed: byte+1).
    bits = bytearray()
    for b in block:
        n = b + 1
        log2 = n.bit_length() - 1
        bits.extend([0] * log2)
        bits.extend(int(x) for x in format(n, f"0{log2 + 1}b"))
    if not bits:
        return data
    # Mutate the bitstream.
    n_mut = rng.randint(1, min(6, len(bits) // 2 + 1))
    for _ in range(n_mut):
        pos = rng.randint(0, len(bits) - 1)
        if rng.random() < 0.5 and len(bits) < length * 16:
            bits.insert(pos, rng.randint(0, 1))
        elif len(bits) > length // 2:
            del bits[pos]
        else:
            bits[pos] ^= 1
    # Decode back.
    restored = bytearray(length)
    ridx = 0
    bidx = 0
    while ridx < length and bidx < len(bits):
        # Count leading zeros.
        log2 = 0
        while bidx < len(bits) and bits[bidx] == 0:
            log2 += 1
            bidx += 1
        if bidx >= len(bits) or log2 == 0:
            break
        total_bits = log2 + 1
        if bidx + total_bits > len(bits):
            break
        val = 0
        for i in range(total_bits):
            val = (val << 1) | bits[bidx + i]
        restored[ridx] = (val - 1) & 0xFF
        ridx += 1
        bidx += total_bits
    while ridx < length:
        restored[ridx] = rng.randint(0, 255)
        ridx += 1
    return _splice(data, offset, bytes(restored[:length]))


def elias_delta(data: bytes, rng=None) -> bytes:
    """Elias delta encode a region, edit codewords, decode back.

    Elias delta coding is a universal code for positive integers that
    prefixes each codeword with the Elias gamma code of floor(log2 N) + 1,
    followed by the binary representation of N without the leading 1.
    Delta coding is shorter than gamma for large N and is used in
    variable-length integer schemes (e.g. Google Protocol Buffers'
    varint, MPEG-4).

    Editing the delta codeword stream probes parsers that decode
    variable-length integers with assumptions about maximum codeword
    length or minimum value.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=2)
    if length < 2:
        return data
    block = data[offset : offset + length]
    # Encode each byte+1 with Elias delta.
    bits = bytearray()
    for b in block:
        n = b + 1
        log2 = n.bit_length() - 1
        # Gamma code of log2 + 1.
        gamma = log2 + 1
        g_log2 = gamma.bit_length() - 1
        bits.extend([0] * g_log2)
        bits.extend(int(x) for x in format(gamma, f"0{g_log2 + 1}b"))
        # Remainder: binary of n without leading 1, length = log2 bits.
        if log2 > 0:
            bits.extend(int(x) for x in format(n & ((1 << log2) - 1), f"0{log2}b"))
    if not bits:
        return data
    # Mutate.
    n_mut = rng.randint(1, min(6, len(bits) // 2 + 1))
    for _ in range(n_mut):
        pos = rng.randint(0, len(bits) - 1)
        if rng.random() < 0.5 and len(bits) < length * 16:
            bits.insert(pos, rng.randint(0, 1))
        elif len(bits) > length // 2:
            del bits[pos]
        else:
            bits[pos] ^= 1
    # Decode.
    restored = bytearray(length)
    ridx = 0
    bidx = 0
    while ridx < length and bidx < len(bits):
        # Read gamma-coded length prefix.
        g_log2 = 0
        while bidx < len(bits) and bits[bidx] == 0:
            g_log2 += 1
            bidx += 1
        if bidx >= len(bits) or g_log2 == 0:
            break
        gamma_bits = g_log2 + 1
        if bidx + gamma_bits > len(bits):
            break
        gamma = 0
        for i in range(gamma_bits):
            gamma = (gamma << 1) | bits[bidx + i]
        bidx += gamma_bits
        log2 = gamma - 1
        if log2 < 0:
            break
        total_bits = log2
        if bidx + total_bits > len(bits):
            break
        val = 1 << log2
        for i in range(total_bits):
            val = (val << 1) | bits[bidx + i]
        restored[ridx] = (val - 1) & 0xFF
        ridx += 1
        bidx += total_bits
    while ridx < length:
        restored[ridx] = rng.randint(0, 255)
        ridx += 1
    return _splice(data, offset, bytes(restored[:length]))


def simd_shuffle(data: bytes, rng=None) -> bytes:
    """Shuffle bytes within SIMD-width windows in a region.

    SIMD instruction sets (SSE/AVX/NEON) process data in fixed-width
    lanes of 16, 32, or 64 bytes.  Parsers that accelerate media or
    crypto routines with SIMD often assume data is aligned to the lane
    width and that adjacent lanes are independent.  Shuffling bytes
    within each lane probes those assumptions: a lane that crosses a
    field boundary can expose unvectorized fallback paths or
    out-of-bounds lane loads.

    The operator selects a random lane width, picks a region aligned
    to that width, and permutes bytes within each lane independently.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 16:
        return data
    rng = _get_rng(rng)
    width = rng.choice((16, 32, 64))
    offset, length = _region(len(data), rng, min_len=width)
    if length < width:
        return data
    aligned_offset = offset + ((width - offset % width) % width)
    aligned_end = offset + length - ((offset + length - aligned_offset) % width)
    if aligned_end <= aligned_offset:
        return data
    block = bytearray(data[aligned_offset:aligned_end])
    for i in range(0, len(block), width):
        lane = block[i : i + width]
        rng.shuffle(lane)
        block[i : i + width] = lane
    return _splice(data, aligned_offset, bytes(block))


def bit_interleave(data: bytes, rng=None) -> bytes:
    """Interleave or deinterleave bit planes in a region.

    Bit-plane interleaving reorders bytes by extracting one bit from
    each of 8 consecutive bytes to form a new byte.  The result spreads
    the bits of each original byte across 8 different output bytes.
    Deinterleaving reverses the process.

    This mutation is meaningful for formats that store pixels, audio
    samples, or packed fields as bit planes rather than as packed bytes:
    BMP, GIF, TIFF, and some RAW camera formats all use interleaved bit
    layouts.  A parser that assumes the wrong plane order reads garbage
    or crashes on the bit extraction fast path.

    The operator operates on a 64-byte aligned window; shorter inputs
    fall through to the identity.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 64:
        return data
    rng = _get_rng(rng)
    # Bit-plane interleave needs regions that are a multiple of 64 bytes
    # (8 groups of 8 bytes each).  Round the length to that granularity.
    offset, length = _region(len(data), rng, min_len=64, max_len=64)
    if length < 64:
        return data
    block = bytearray(data[offset : offset + length])
    # Each 64-byte chunk: 8 groups of 8 bytes.  Extract one bit from
    # each byte at position j (MSB-first) to form each destination byte,
    # then shuffle the 64 destination bytes and reverse.
    for chunk_start in range(0, length, 64):
        grp = block[chunk_start : chunk_start + 64]
        dst = bytearray(64)
        for j in range(8):
            for g in range(8):
                dst[j * 8 + g] = (grp[g * 8 + j] >> (7 - j)) & 1
        rng.shuffle(dst)
        out = bytearray(64)
        for g in range(8):
            b = 0
            for j in range(8):
                b |= dst[g * 8 + j] << (7 - j)
            out[g] = b
        block[chunk_start : chunk_start + 64] = out
    return _splice(data, offset, bytes(block))


def gray_code(data: bytes, rng=None) -> bytes:
    """Mutate bytes in Gray-coded space.

    Gray code is a binary encoding where adjacent values differ by
    exactly one bit.  Flipping bits in Gray-coded space produces
    mutations that differ from the original by a Hamming distance of
    exactly 1, 2, or 3 bits per byte — a different bias from ordinary
    bit-flip operators, which tend to cluster in the low bits of the
    byte.

    This is useful for probing counter/register fields, CRC checksums,
    and other values where single-bit transitions are the expected
    update pattern.  A parser that assumes Gray-coded monotonicity can
    miss the mutation; one that assumes natural binary order reads a
    very different value.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 2:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=2)
    if length < 2:
        return data
    block = bytearray(data[offset : offset + length])
    gray = [b ^ (b >> 1) for b in block]
    n_flip = rng.randint(1, min(3, length))
    for _ in range(n_flip):
        pos = rng.randint(0, length - 1)
        bit = rng.randint(0, 7)
        gray[pos] ^= 1 << bit
    restored = bytearray(length)
    for i, g in enumerate(gray):
        b = g
        k = g >> 1
        while k:
            b ^= k
            k >>= 1
        restored[i] = b & 0xFF
    return _splice(data, offset, bytes(restored))


def lz_dict_mutate(data: bytes, rng=None) -> bytes:
    """Mutate an LZ77-style dictionary/literal pair in a region.

    LZ77-family compressors (DEFLATE, LZ4, Zstandard, XZ) encode data
    as a mix of literal bytes and back-reference tokens of the form
    (distance, length).  The distance selects an earlier position in
    the sliding window; the length selects how many bytes to copy from
    there.

    This operator simulates an LZ77 token stream over the input and
    mutates the distance/length pairs.  A distance mutation changes
    which historical bytes get copied; a length mutation changes how
    many bytes get copied.  Both probe decompressor fast paths that
    assume distance/length pairs are valid and within bounds.

    The mutation is applied to a randomly-selected region; the result
    is always the same length as the input.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 8:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=8)
    if length < 8:
        return data
    block = bytearray(data[offset : offset + length])
    tokens = []
    i = 0
    min_match = 3
    max_match = min(64, length - 1)
    while i < len(block):
        best_dist, best_len = 0, 0
        search_start = max(0, i - 32768)
        for j in range(search_start, i):
            match_len = 0
            while (
                match_len < max_match
                and i + match_len < len(block)
                and block[j + match_len] == block[i + match_len]
            ):
                match_len += 1
            if match_len > best_len:
                best_dist = i - j
                best_len = match_len
        if best_len >= min_match:
            tokens.append(("ref", best_dist, best_len))
            i += best_len
        else:
            tokens.append(("lit", block[i]))
            i += 1
    if not tokens:
        return data
    n_mut = rng.randint(1, min(4, len(tokens)))
    for _ in range(n_mut):
        pos = rng.randint(0, len(tokens) - 1)
        tok = tokens[pos]
        if tok[0] == "lit":
            dist = rng.randint(1, min(32768, pos))
            run_len = rng.randint(min_match, min(max_match, len(block) - pos))
            tokens[pos] = ("ref", dist, run_len)
        else:
            action = rng.randint(0, 2)
            if action == 0:
                new_dist = max(1, min(32768, tok[1] + rng.randint(-16, 16)))
                tokens[pos] = ("ref", new_dist, tok[2])
            elif action == 1:
                new_len = max(min_match, min(max_match, tok[2] + rng.randint(-2, 2)))
                tokens[pos] = ("ref", tok[1], new_len)
            else:
                lit_pos = min(pos, len(block) - 1)
                tokens[pos] = ("lit", block[lit_pos])
    restored = bytearray(length)
    ridx = 0
    for tok in tokens:
        if tok[0] == "lit":
            if ridx < length:
                restored[ridx] = tok[1]
                ridx += 1
        else:
            _, dist, run_len = tok
            for _ in range(run_len):
                if ridx >= length:
                    break
                src_idx = ridx - dist
                if 0 <= src_idx < length:
                    restored[ridx] = restored[src_idx]
                else:
                    restored[ridx] = rng.randint(0, 255)
                ridx += 1
    while ridx < length:
        restored[ridx] = rng.randint(0, 255)
        ridx += 1
    return _splice(data, offset, bytes(restored[:length]))


def huffman_tree_mutate(data: bytes, rng=None) -> bytes:
    """Mutate the Huffman codebook implied by a region, re-encode.

    Huffman coding assigns short bit patterns to frequent symbols and
    long ones to rare symbols.  The assignment is determined by a tree
    structure: swapping two sibling leaves changes which symbol gets
    the short code without violating the prefix-free property.

    This operator builds an approximate Huffman tree for the region,
    swaps two leaf assignments, and re-encodes the region with the
    mutated tree.  The result is a valid Huffman stream that decodes
    to different bytes — probing the decoder's tree traversal and
    symbol reconstruction paths.

    Args:
        data: Input bytes.
        rng: RandPool or stdlib random.

    Returns:
        Mutated bytes, the same length as *data*.
    """
    if len(data) < 4:
        return data
    rng = _get_rng(rng)
    offset, length = _region(len(data), rng, min_len=4)
    if length < 4:
        return data
    block = data[offset : offset + length]
    freq = [0] * 256
    for b in block:
        freq[b] += 1
    symbols = sorted(range(256), key=lambda s: (freq[s], s))
    codes = [0] * 256
    lengths = [0] * 256
    code = 0
    prev_len = 0
    for rank, sym in enumerate(symbols):
        if freq[sym] == 0:
            continue
        bit_len = max(1, rank.bit_length())
        if rank > 0 and bit_len > prev_len:
            code <<= bit_len - prev_len
        codes[sym] = code
        lengths[sym] = bit_len
        code += 1
        prev_len = bit_len
    non_zero = [s for s in symbols if freq[s] > 0]
    if len(non_zero) < 2:
        return data
    a, b = rng.sample(non_zero, 2)
    codes[a], codes[b] = codes[b], codes[a]
    lengths[a], lengths[b] = lengths[b], lengths[a]
    bits = bytearray()
    for b in block:
        sym_code = codes[b]
        sym_len = lengths[b]
        for i in range(sym_len - 1, -1, -1):
            bits.append((sym_code >> i) & 1)
    decode_table = {}
    for sym in range(256):
        if lengths[sym] == 0:
            continue
        decode_table[(codes[sym], lengths[sym])] = sym
    restored = bytearray(length)
    ridx = 0
    bidx = 0
    while ridx < length and bidx < len(bits):
        for bit_len in range(1, 17):
            if bidx + bit_len > len(bits):
                break
            prefix = 0
            for i in range(bit_len):
                prefix = (prefix << 1) | bits[bidx + i]
            key = (prefix, bit_len)
            if key in decode_table:
                restored[ridx] = decode_table[key]
                ridx += 1
                bidx += bit_len
                break
        else:
            break
    while ridx < length:
        restored[ridx] = rng.randint(0, 255)
        ridx += 1
    return _splice(data, offset, bytes(restored[:length]))
