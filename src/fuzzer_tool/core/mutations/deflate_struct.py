"""Structural mutation of the DEFLATE (RFC 1951) bitstream itself.

Two positions already exist and this module is the gap between them:
``mutations/zlib.py``/``gzip.py`` corrupt the *compressed bytes* in place,
which reliably breaks the stream so the target's inflate call aborts before
the payload parser ever runs; ``mutations/recompress.py`` inflates, mutates
the *plaintext*, and re-deflates, which never touches the compressed
representation at all. Neither reaches the DEFLATE decoder's own structural
error paths -- the block-type field, the dynamic Huffman header
(HLIT/HDIST/HCLEN), the code-length alphabet, or a back-reference
(length, distance) pair with an out-of-window distance.

This module parses a raw DEFLATE stream into its literal structure (block
headers, canonical Huffman code-length arrays, and the literal/match token
stream), applies one mutation to that structure, and re-serializes it. Unlike
``recompress.py`` this only preserves *plaintext* semantics when the
mutation happens to keep the stream decodable -- several of the mutations
below are deliberately chosen to produce a well-formed-until-suddenly-not
bitstream, which is exactly the shape real DEFLATE decoder bugs are found by.

Self-validating oracle: after re-serializing, the module attempts its own
bounded inflate of the mutated stream via the stdlib ``zlib`` decoder (the
same decoder the rest of the tree already trusts as ground truth). When that
succeeds, the wrapper checksum (Adler-32 for zlib, CRC-32+ISIZE for gzip) is
recomputed from the real resulting plaintext, so a structurally-valid but
semantically different mutation still reaches the target's payload parser
instead of being rejected on a stale checksum. When it fails, the checksum is
irrelevant -- the target's own inflate will fail at the same structural fault
before it ever checks one.

Only the zlib and gzip containers are handled, matching ``recompress.py``'s
scope note: no zstd, it is not vendored.

Every code path here is bounded (block count, token count, huffman code
length) so a pathological or already-corrupt stream costs a fixed amount of
work instead of spinning or raising past the caller.
"""

from __future__ import annotations

import binascii
import heapq
import random
import struct
import zlib

# ── Bounds ──────────────────────────────────────────────────────────────
_MAX_COMPRESSED_IN = 1 << 20  # 1 MiB, matches recompress.py
_MAX_BLOCKS = 256
_MAX_TOKENS = 1 << 16  # 65536 literal/match tokens across all blocks
_MAX_STORED_LEN = 1 << 20
_MAX_HUFFMAN_CODE_LEN = 15  # RFC 1951 hard limit; also our decode bail-out
_MAX_SELF_INFLATE = 1 << 22  # bound on the self-validating trial inflate

# ── RFC 1951 tables ───────────────────────────────────────────────────────

# Order in which code-length-alphabet lengths are transmitted (3.2.7).
CLCL_ORDER = (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)

# Length codes 257..285 -> (base length, extra bits) (3.2.5).
_LENGTH_BASE = (
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    15,
    17,
    19,
    23,
    27,
    31,
    35,
    43,
    51,
    59,
    67,
    83,
    99,
    115,
    131,
    163,
    195,
    227,
    258,
)
_LENGTH_EXTRA = (
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    0,
)
# Distance codes 0..29 -> (base distance, extra bits) (3.2.5).
_DIST_BASE = (
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    25,
    33,
    49,
    65,
    97,
    129,
    193,
    257,
    385,
    513,
    769,
    1025,
    1537,
    2049,
    3073,
    4097,
    6145,
    8193,
    12289,
    16385,
    24577,
)
_DIST_EXTRA = (
    0,
    0,
    0,
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
    12,
    13,
    13,
)

FIXED_LITLEN_LENGTHS = tuple(
    8 if i < 144 else 9 if i < 256 else 7 if i < 280 else 8 for i in range(288)
)
FIXED_DIST_LENGTHS = (5,) * 30


def _length_to_symbol(length: int) -> tuple[int, int, int]:
    """length -> (symbol 257..285, extra_value, extra_bits)."""
    for sym in range(28, -1, -1):
        if length >= _LENGTH_BASE[sym]:
            return 257 + sym, length - _LENGTH_BASE[sym], _LENGTH_EXTRA[sym]
    raise ValueError(f"length {length} out of range")


def _dist_to_symbol(dist: int) -> tuple[int, int, int]:
    """distance -> (symbol 0..29, extra_value, extra_bits)."""
    for sym in range(29, -1, -1):
        if dist >= _DIST_BASE[sym]:
            return sym, dist - _DIST_BASE[sym], _DIST_EXTRA[sym]
    raise ValueError(f"distance {dist} out of range")


class DeflateError(ValueError):
    """Raised (and always caught) when a stream can't be parsed as DEFLATE."""


# ── Bit-level I/O ─────────────────────────────────────────────────────────
#
# RFC 1951 3.1.1: "packed into bytes starting with the least-significant bit
# of the byte" for every ordinary field (BFINAL, BTYPE, HLIT/HDIST/HCLEN,
# code-length values, extra bits) -- so BitReader.read()/BitWriter.write()
# assemble/emit LSB-first. Huffman codes are the one exception: "packed
# starting with the most-significant bit of the code", which is why they go
# through read_huffman()/write_huffman() instead.


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.bitpos = 0
        self.nbits = len(data) * 8

    def read_bit(self) -> int:
        if self.bitpos >= self.nbits:
            raise DeflateError("unexpected end of stream")
        byte = self.data[self.bitpos >> 3]
        bit = (byte >> (self.bitpos & 7)) & 1
        self.bitpos += 1
        return bit

    def read(self, n: int) -> int:
        value = 0
        for i in range(n):
            value |= self.read_bit() << i
        return value

    def align(self) -> None:
        self.bitpos = (self.bitpos + 7) & ~7

    def read_bytes(self, n: int) -> bytes:
        # Caller must be byte-aligned (true after a stored-block align()).
        start = self.bitpos >> 3
        end = start + n
        if end > len(self.data):
            raise DeflateError("stored block runs past end of stream")
        self.bitpos = end * 8
        return self.data[start:end]

    def decode_huffman(self, table: dict[tuple[int, int], int]) -> int:
        code = 0
        for length in range(1, _MAX_HUFFMAN_CODE_LEN + 1):
            code = (code << 1) | self.read_bit()
            sym = table.get((length, code))
            if sym is not None:
                return sym
        raise DeflateError("no matching huffman code")


class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.cur = 0
        self.nbits = 0  # bits filled in self.cur, 0..7

    def write_bit(self, bit: int) -> None:
        self.cur |= (bit & 1) << self.nbits
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.cur)
            self.cur = 0
            self.nbits = 0

    def write(self, value: int, n: int) -> None:
        for i in range(n):
            self.write_bit((value >> i) & 1)

    def write_huffman(self, code: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            self.write_bit((code >> i) & 1)

    def align(self) -> None:
        if self.nbits:
            self.buf.append(self.cur)
            self.cur = 0
            self.nbits = 0

    def write_bytes(self, data: bytes) -> None:
        self.buf.extend(data)

    def getvalue(self) -> bytes:
        self.align()
        return bytes(self.buf)


# ── Canonical Huffman code construction ───────────────────────────────────


def build_canonical_codes(lengths: list[int]) -> dict[int, int]:
    """RFC 1951 3.2.2: symbol -> code, from a code-length array indexed by
    symbol (0 = symbol unused)."""
    max_len = max(lengths, default=0)
    if max_len == 0:
        return {}
    bl_count = [0] * (max_len + 1)
    for length in lengths:
        if length:
            bl_count[length] += 1
    code = 0
    next_code = [0] * (max_len + 1)
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    codes = {}
    for symbol, length in enumerate(lengths):
        if length:
            codes[symbol] = next_code[length]
            next_code[length] += 1
    return codes


def build_decode_table(lengths: list[int]) -> dict[tuple[int, int], int]:
    return {
        (length, code): symbol
        for symbol, code in build_canonical_codes(lengths).items()
        for length in (lengths[symbol],)
    }


def _huffman_lengths(symbols: list[int]) -> dict[int, int]:
    """Near-balanced code lengths for an equal-weight symbol set.

    Only ever called on the code-length alphabet (<=16 distinct values,
    since this module never emits RFC 1951's 16/17/18 repeat codes -- see
    the module docstring), so the standard combine-two-smallest merge stays
    shallow (depth ~ ceil(log2(n)) <= 4) and easily inside the alphabet's
    own 7-bit code-length limit. Fuzzing doesn't need optimal code lengths,
    only a valid (Kraft-equality) assignment, which this always produces.
    """
    symbols = sorted(set(symbols))
    if not symbols:
        return {}
    if len(symbols) == 1:
        return {symbols[0]: 1}
    heap = [(1, i, [s]) for i, s in enumerate(symbols)]
    heapq.heapify(heap)
    depth = dict.fromkeys(symbols, 0)
    counter = len(symbols)
    while len(heap) > 1:
        f1, _, g1 = heapq.heappop(heap)
        f2, _, g2 = heapq.heappop(heap)
        for s in g1:
            depth[s] += 1
        for s in g2:
            depth[s] += 1
        heapq.heappush(heap, (f1 + f2, counter, g1 + g2))
        counter += 1
    return depth


_FIXED_LITLEN_CODES = build_canonical_codes(list(FIXED_LITLEN_LENGTHS))
_FIXED_DIST_CODES = build_canonical_codes(list(FIXED_DIST_LENGTHS))
_FIXED_LITLEN_DECODE = build_decode_table(list(FIXED_LITLEN_LENGTHS))
_FIXED_DIST_DECODE = build_decode_table(list(FIXED_DIST_LENGTHS))


# ── Parsing ────────────────────────────────────────────────────────────────


def _decode_tokens(reader: BitReader, litlen_decode, dist_decode, litlen_lengths):
    tokens = []
    while True:
        if len(tokens) > _MAX_TOKENS:
            raise DeflateError("too many tokens (bomb guard)")
        sym = reader.decode_huffman(litlen_decode)
        if sym < 256:
            tokens.append(("lit", sym))
        elif sym == 256:
            return tokens
        elif sym <= 285:
            length = _LENGTH_BASE[sym - 257] + reader.read(_LENGTH_EXTRA[sym - 257])
            if dist_decode is None:
                raise DeflateError("back-reference with no distance alphabet")
            dsym = reader.decode_huffman(dist_decode)
            if dsym > 29:
                raise DeflateError("invalid distance symbol")
            distance = _DIST_BASE[dsym] + reader.read(_DIST_EXTRA[dsym])
            tokens.append(("match", length, distance))
        else:
            raise DeflateError("invalid literal/length symbol")


def parse_deflate(data: bytes) -> list[dict]:
    """Parse a raw (headerless) DEFLATE stream into a list of block dicts.

    Raises DeflateError on anything that doesn't parse cleanly -- callers
    treat that as "not usable", the same contract ``recompress.py``'s
    ``_inflate`` uses (return None, fall through to another operator).
    """
    reader = BitReader(data)
    blocks = []
    while True:
        if len(blocks) >= _MAX_BLOCKS:
            raise DeflateError("too many blocks (bomb guard)")
        bfinal = reader.read(1)
        btype = reader.read(2)
        if btype == 0:
            reader.align()
            length = reader.read(16)
            nlength = reader.read(16)
            if length > _MAX_STORED_LEN:
                raise DeflateError("stored block too large")
            stored = reader.read_bytes(length)
            blocks.append(
                {
                    "bfinal": bfinal,
                    "btype": 0,
                    "stored_bytes": stored,
                    "nlen_ok": (~length & 0xFFFF) == nlength,
                }
            )
        elif btype == 1:
            tokens = _decode_tokens(
                reader, _FIXED_LITLEN_DECODE, _FIXED_DIST_DECODE, FIXED_LITLEN_LENGTHS
            )
            blocks.append({"bfinal": bfinal, "btype": 1, "tokens": tokens})
        elif btype == 2:
            hlit = reader.read(5) + 257
            hdist = reader.read(5) + 1
            hclen = reader.read(4) + 4
            clcl = [0] * 19
            for i in range(hclen):
                clcl[CLCL_ORDER[i]] = reader.read(3)
            cl_decode = build_decode_table(clcl)
            if not cl_decode:
                raise DeflateError("empty code-length alphabet")
            lengths: list[int] = []
            prev = 0
            target = hlit + hdist
            while len(lengths) < target:
                sym = reader.decode_huffman(cl_decode)
                if sym <= 15:
                    lengths.append(sym)
                    prev = sym
                elif sym == 16:
                    if not lengths:
                        raise DeflateError("repeat-previous with nothing to repeat")
                    lengths.extend([prev] * (reader.read(2) + 3))
                elif sym == 17:
                    lengths.extend([0] * (reader.read(3) + 3))
                elif sym == 18:
                    lengths.extend([0] * (reader.read(7) + 11))
                else:
                    raise DeflateError("invalid code-length symbol")
            if len(lengths) != target:
                raise DeflateError("code-length run overshot HLIT+HDIST")
            litlen_lengths = lengths[:hlit]
            dist_lengths = lengths[hlit:]
            litlen_decode = build_decode_table(litlen_lengths)
            dist_decode = build_decode_table(dist_lengths) if any(dist_lengths) else None
            tokens = _decode_tokens(reader, litlen_decode, dist_decode, litlen_lengths)
            blocks.append(
                {
                    "bfinal": bfinal,
                    "btype": 2,
                    "hlit": hlit,
                    "hdist": hdist,
                    "litlen_lengths": litlen_lengths,
                    "dist_lengths": dist_lengths,
                    "tokens": tokens,
                }
            )
        else:
            raise DeflateError("reserved block type (11)")
        if bfinal:
            return blocks


# ── Serialization ──────────────────────────────────────────────────────────


def _serialize_tokens(
    writer: BitWriter, tokens, litlen_codes, litlen_lengths, dist_codes, dist_lengths
):
    for tok in tokens:
        if tok[0] == "lit":
            b = tok[1]
            writer.write_huffman(litlen_codes[b], litlen_lengths[b])
        else:
            _, length, distance = tok
            lsym, lextra, lextra_bits = _length_to_symbol(length)
            writer.write_huffman(litlen_codes[lsym], litlen_lengths[lsym])
            if lextra_bits:
                writer.write(lextra, lextra_bits)
            dsym, dextra, dextra_bits = _dist_to_symbol(distance)
            writer.write_huffman(dist_codes[dsym], dist_lengths[dsym])
            if dextra_bits:
                writer.write(dextra, dextra_bits)
    writer.write_huffman(litlen_codes[256], litlen_lengths[256])


def _write_dynamic_block(
    writer: BitWriter,
    header_litlen_lengths: list[int],
    header_dist_lengths: list[int],
    encode_litlen_lengths: list[int],
    encode_dist_lengths: list[int],
    tokens,
) -> None:
    """Write a BTYPE=2 block whose *declared* header can differ from the
    lengths actually used to encode the token stream.

    The normal (non-mutated) path calls this with header arrays identical to
    the encode arrays -- a real encoder never has reason to lie about its
    own table. ``permute_symbol_lengths`` is the one caller that passes
    different arrays on purpose: the header the decoder builds its table
    from doesn't match the table the data was actually written with.
    """
    writer.write(len(header_litlen_lengths) - 257, 5)
    writer.write(len(header_dist_lengths) - 1, 5)
    used = sorted(set(header_litlen_lengths) | set(header_dist_lengths)) or [0]
    cl_lengths_map = _huffman_lengths(used)
    clcl19 = [0] * 19
    for sym, length in cl_lengths_map.items():
        clcl19[sym] = length
    writer.write(19 - 4, 4)
    for i in range(19):
        writer.write(clcl19[CLCL_ORDER[i]], 3)
    cl_codes = build_canonical_codes(clcl19)
    for value in header_litlen_lengths + header_dist_lengths:
        writer.write_huffman(cl_codes[value], clcl19[value])
    litlen_codes = build_canonical_codes(encode_litlen_lengths)
    dist_codes = build_canonical_codes(encode_dist_lengths) if any(encode_dist_lengths) else {}
    _serialize_tokens(
        writer, tokens, litlen_codes, encode_litlen_lengths, dist_codes, encode_dist_lengths
    )


def serialize_deflate(blocks: list[dict]) -> bytes:
    """Inverse of :func:`parse_deflate`.

    Dynamic-Huffman blocks always re-derive a *fresh* code-length alphabet
    (via ``_huffman_lengths``) rather than replaying the stored HCLEN/CLCL --
    this module never emits RFC 1951's 16/17/18 run-length codes, only plain
    0..15 values one per array entry, which is spec-legal (repeat codes are
    an optimization, not a requirement) and sidesteps re-deriving a run-length
    encoding for a possibly-mutated length array.
    """
    writer = BitWriter()
    for block in blocks:
        writer.write(block["bfinal"], 1)
        writer.write(block["btype"], 2)
        if block["btype"] == 0:
            writer.align()
            stored = block["stored_bytes"]
            length = len(stored)
            writer.write(length, 16)
            writer.write((~length) & 0xFFFF, 16)
            writer.write_bytes(stored)
        elif block["btype"] == 1:
            _serialize_tokens(
                writer,
                block["tokens"],
                _FIXED_LITLEN_CODES,
                FIXED_LITLEN_LENGTHS,
                _FIXED_DIST_CODES,
                FIXED_DIST_LENGTHS,
            )
        elif block["btype"] == 2:
            litlen_lengths = list(block["litlen_lengths"])
            dist_lengths = list(block["dist_lengths"])
            _write_dynamic_block(
                writer, litlen_lengths, dist_lengths, litlen_lengths, dist_lengths, block["tokens"]
            )
        else:
            raise DeflateError(f"unsupported btype {block['btype']}")
    return writer.getvalue()


def _serialize_truncated_reserved(blocks: list[dict], cut: int) -> bytes:
    """Serialize blocks[:cut] normally, then emit a bare reserved-type
    (BTYPE=11) header for blocks[cut] and stop.

    RFC 1951 3.2.3 leaves BTYPE=11 undefined -- there is no body to write,
    only the fault a decoder must detect. A random byte flip lands on these
    2 bits with low probability (2 bits out of a whole block); this targets
    them every time.
    """
    writer = BitWriter()
    for block in blocks[:cut]:
        # Force non-final so the stream keeps going into our reserved block.
        writer.write(0, 1)
        writer.write(block["btype"], 2)
        if block["btype"] == 0:
            writer.align()
            stored = block["stored_bytes"]
            length = len(stored)
            writer.write(length, 16)
            writer.write((~length) & 0xFFFF, 16)
            writer.write_bytes(stored)
        elif block["btype"] == 1:
            _serialize_tokens(
                writer,
                block["tokens"],
                _FIXED_LITLEN_CODES,
                FIXED_LITLEN_LENGTHS,
                _FIXED_DIST_CODES,
                FIXED_DIST_LENGTHS,
            )
        else:
            litlen_lengths = list(block["litlen_lengths"])
            dist_lengths = list(block["dist_lengths"])
            _write_dynamic_block(
                writer, litlen_lengths, dist_lengths, litlen_lengths, dist_lengths, block["tokens"]
            )
    tail_bfinal = blocks[cut]["bfinal"] if cut < len(blocks) else 1
    writer.write(tail_bfinal, 1)
    writer.write(3, 2)  # BTYPE=11, reserved/undefined
    return writer.getvalue()


# ── Mutations over the parsed structure ────────────────────────────────────
#
# Each mutation takes the parsed block list and an RNG, and returns a *new*
# byte string (not a block list) -- some of them (the reserved-type one)
# don't produce a re-parseable structure at all, so a shared "mutate then
# serialize_deflate()" pipeline can't cover every case uniformly.


def mutate_final_flag(blocks: list[dict], rng) -> bytes:
    """Flip BFINAL on one block.

    On a non-last block: the stream truncates early (everything after is
    simply never reached). On the last block: the decoder finishes this
    block, believes another follows, and reads a block header from
    whatever comes after the stream -- normally nothing, so it fails at
    the boundary check instead of mid-block.
    """
    blocks = [dict(b) for b in blocks]
    i = rng.randrange(len(blocks))
    blocks[i]["bfinal"] ^= 1
    if i != len(blocks) - 1:
        # Keep exactly one BFINAL=1 elsewhere so encodable inputs with a
        # flipped *non-final* flag still terminate somewhere, exercising
        # "stream ends earlier/later than it should" rather than "no
        # terminator at all", which is already covered by flipping the
        # last block's flag.
        for j, b in enumerate(blocks):
            if j != i:
                b["bfinal"] = 1 if j == len(blocks) - 1 else 0
    return serialize_deflate(blocks)


def mutate_reserved_block_type(blocks: list[dict], rng) -> bytes:
    """Truncate at a random block and replace it with a bare BTYPE=11 header."""
    cut = rng.randrange(len(blocks))
    return _serialize_truncated_reserved(blocks, cut)


def shift_hlit_hdist_boundary(blocks: list[dict], rng) -> bytes:
    """Move a few code-length entries across the HLIT/HDIST split.

    The combined length array is unchanged in total size and content, only
    the split point moves, so the *codes* built from each half are
    completely different -- literal/length symbols reappear as distance
    symbols and vice versa -- while the *token stream itself is untouched*.
    This targets exactly the seam between the two alphabets that only a
    dynamic-Huffman block has, which a byte-level flip essentially never
    isolates on its own.
    """
    candidates = [i for i, b in enumerate(blocks) if b["btype"] == 2]
    if not candidates:
        raise DeflateError("no dynamic-huffman block to mutate")
    blocks = [dict(b) for b in blocks]
    i = rng.choice(candidates)
    block = dict(blocks[i])
    litlen = list(block["litlen_lengths"])
    dist = list(block["dist_lengths"])
    total = len(litlen) + len(dist)
    shift = rng.choice([-3, -2, -1, 1, 2, 3])
    new_hlit = len(litlen) + shift
    # HLIT must leave room for symbols 0..256 (>=257) and at least one
    # distance-alphabet entry.
    new_hlit = max(257, min(total - 1, new_hlit))
    combined = litlen + dist
    block["litlen_lengths"] = combined[:new_hlit]
    block["dist_lengths"] = combined[new_hlit:]
    blocks[i] = block
    return serialize_deflate(blocks)


def permute_symbol_lengths(blocks: list[dict], rng) -> bytes:
    """Declare a swapped code-length table while encoding with the real one.

    Swapping two entries in a code-length array preserves the exact
    multiset of lengths, so the swapped array is just as Kraft-valid as the
    original -- a decoder builds a perfectly good, fully-decodable Huffman
    table from it. The trick is *which* table: the transmitted header here
    is the swapped array, but the token stream is encoded with the
    *original* table, so the two are silently different. A rebuilding
    encoder (this module, on its own output) would stay self-consistent
    and this would be a no-op -- the point is that a real target decodes
    with the table the header claims, so codewords now land on the wrong
    symbols. Almost every codeword differs in bit-length from what the
    swapped table expects, so this desyncs bit alignment for the rest of
    the block; whether that surfaces as "decodes to something else" or
    "invalid huffman code" depends on where in the token stream the first
    difference falls.
    """
    candidates = [i for i, b in enumerate(blocks) if b["btype"] == 2]
    if not candidates:
        raise DeflateError("no dynamic-huffman block to mutate")
    target = rng.choice(candidates)
    litlen = list(blocks[target]["litlen_lengths"])
    nonzero = [idx for idx, length in enumerate(litlen) if length and idx != 256]
    if len(nonzero) < 2:
        raise DeflateError("not enough distinct symbols to permute")
    a, b = rng.sample(nonzero, 2)
    header_litlen = list(litlen)
    header_litlen[a], header_litlen[b] = header_litlen[b], header_litlen[a]

    writer = BitWriter()
    for i, block in enumerate(blocks):
        writer.write(block["bfinal"], 1)
        writer.write(block["btype"], 2)
        if block["btype"] == 0:
            writer.align()
            stored = block["stored_bytes"]
            length = len(stored)
            writer.write(length, 16)
            writer.write((~length) & 0xFFFF, 16)
            writer.write_bytes(stored)
        elif block["btype"] == 1:
            _serialize_tokens(
                writer,
                block["tokens"],
                _FIXED_LITLEN_CODES,
                FIXED_LITLEN_LENGTHS,
                _FIXED_DIST_CODES,
                FIXED_DIST_LENGTHS,
            )
        else:
            dist_lengths = list(block["dist_lengths"])
            if i == target:
                _write_dynamic_block(
                    writer, header_litlen, dist_lengths, litlen, dist_lengths, block["tokens"]
                )
            else:
                same = list(block["litlen_lengths"])
                _write_dynamic_block(
                    writer, same, dist_lengths, same, dist_lengths, block["tokens"]
                )
    return writer.getvalue()


def mutate_backref(blocks: list[dict], rng) -> bytes:
    """Push one back-reference (length, distance) pair to an adversarial edge.

    Three variants, chosen at random:
      - max overlap: distance=1, i.e. "repeat the last byte `length` times"
        -- legal DEFLATE, but a decoder implemented with a bulk ``memcpy``
        instead of a byte-at-a-time copy corrupts the overlapping region.
      - out-of-window: distance larger than the plaintext produced so far
        *in this parse*, which is a real decoder needing to read before
        the start of its own output buffer.
      - max length: push length to 258, DEFLATE's own ceiling, cheap edge
        coverage for off-by-one bounds checks.
    """
    blocks = [dict(b) for b in blocks]
    match_sites = [
        (bi, ti)
        for bi, b in enumerate(blocks)
        for ti, t in enumerate(b.get("tokens", ()))
        if t[0] == "match"
    ]
    if not match_sites:
        raise DeflateError("no back-reference to mutate")
    bi, ti = rng.choice(match_sites)
    block = dict(blocks[bi])
    tokens = list(block["tokens"])
    _, length, distance = tokens[ti]

    produced = 0
    for b in blocks[:bi]:
        if b["btype"] == 0:
            produced += len(b["stored_bytes"])
        else:
            for t in b["tokens"]:
                produced += 1 if t[0] == "lit" else t[1]
    for t in tokens[:ti]:
        produced += 1 if t[0] == "lit" else t[1]

    variant = rng.choice(("overlap", "out_of_window", "max_length"))
    if variant == "overlap":
        distance = 1
    elif variant == "out_of_window":
        distance = min(32768, produced + rng.randint(1, 4096))
        distance = max(distance, 1)
    else:
        length = 258

    tokens[ti] = ("match", length, distance)
    block["tokens"] = tokens
    blocks[bi] = block
    return serialize_deflate(blocks)


_MUTATORS = (
    mutate_final_flag,
    mutate_reserved_block_type,
    shift_hlit_hdist_boundary,
    permute_symbol_lengths,
    mutate_backref,
)


# ── Container handling (zlib / gzip) ────────────────────────────────────────


def _split_zlib(data: bytes) -> tuple[bytes, bytes] | None:
    """(header+trailer info, raw deflate payload) for a zlib stream, or None."""
    if len(data) < 6 or (data[0] & 0x0F) != 8 or ((data[0] << 8) | data[1]) % 31 != 0:
        return None
    has_dict = bool(data[1] & 0x20)
    start = 6 if has_dict else 2  # FDICT adds a 4-byte DICTID we don't handle
    if has_dict:
        return None
    if len(data) < start + 4:
        return None
    return data[:start], data[start:-4]


def _split_gzip(data: bytes) -> tuple[bytes, bytes] | None:
    if len(data) < 18 or data[:3] != b"\x1f\x8b\x08":
        return None
    flg = data[3]
    pos = 10
    if flg & 0x04:  # FEXTRA
        if pos + 2 > len(data):
            return None
        xlen = struct.unpack_from("<H", data, pos)[0]
        pos += 2 + xlen
    if flg & 0x08:  # FNAME
        end = data.find(b"\x00", pos)
        if end == -1:
            return None
        pos = end + 1
    if flg & 0x10:  # FCOMMENT
        end = data.find(b"\x00", pos)
        if end == -1:
            return None
        pos = end + 1
    if flg & 0x02:  # FHCRC
        pos += 2
    if pos + 8 > len(data):
        return None
    return data[:pos], data[pos:-8]


def _rebuild_zlib(header: bytes, deflate_payload: bytes) -> bytes:
    """Recompute Adler-32 from a real trial-inflate of the mutated stream.

    If the mutated payload doesn't inflate cleanly the checksum can't be
    (and doesn't need to be) meaningful: the target's own zlib will fail at
    the same structural fault before it ever reaches the checksum check.
    """
    try:
        plain = zlib.decompressobj(-15).decompress(deflate_payload, _MAX_SELF_INFLATE)
        adler = struct.pack(">I", zlib.adler32(plain) & 0xFFFFFFFF)
    except zlib.error:
        adler = b"\x00\x00\x00\x00"
    return header + deflate_payload + adler


def _rebuild_gzip(header: bytes, deflate_payload: bytes) -> bytes:
    try:
        plain = zlib.decompressobj(-15).decompress(deflate_payload, _MAX_SELF_INFLATE)
        trailer = struct.pack("<II", binascii.crc32(plain) & 0xFFFFFFFF, len(plain) & 0xFFFFFFFF)
    except zlib.error:
        trailer = b"\x00\x00\x00\x00\x00\x00\x00\x00"
    return header + deflate_payload + trailer


def mutate_deflate_structure(data: bytes, max_len: int = 4096, rng=None) -> bytes | None:
    """Parse a zlib or gzip stream's DEFLATE structure, apply one structural
    mutation, and re-emit a valid container around it.

    Returns None when *data* isn't an invertible zlib/gzip+DEFLATE stream so
    callers can fall through to another operator, matching
    ``recompress.py``'s contract.
    """
    r = rng or random
    if len(data) > _MAX_COMPRESSED_IN:
        return None
    split = _split_zlib(data)
    rebuild = _rebuild_zlib
    if split is None:
        split = _split_gzip(data)
        rebuild = _rebuild_gzip
    if split is None:
        return None
    header, payload = split
    try:
        blocks = parse_deflate(payload)
    except DeflateError:
        return None

    mutator = r.choice(_MUTATORS)
    try:
        mutated_payload = mutator(blocks, r)
    except (DeflateError, KeyError, IndexError, ValueError):
        # A mutation can produce a token/symbol combination that its own
        # rebuilt alphabet no longer covers (e.g. shifting the HLIT/HDIST
        # boundary can strand a token referencing a now-absent distance
        # symbol) -- that is a legitimate "this particular mutation isn't
        # applicable to this stream" outcome, not a bug in the mutator.
        return None

    out = rebuild(header, mutated_payload)
    if len(out) > max_len:
        return None
    return out
