"""Structure-aware native FLAC metadata-block mutator.

Native FLAC layout (libavformat/flacdec.c, libavcodec/flac.c):

  "fLaC" magic (4 bytes)
  then a sequence of metadata blocks, each:
    [1 byte:  is_last (bit 0x80) | block_type (bits 0x7F)]
    [3 bytes: length, big-endian]
    [length bytes: block payload]
  followed, once a block with is_last set has been read, by raw frame
  data this module never touches.

No operator here reaches any of this before now -- ``operator_registry``
has no ``flac_*`` entry at all, unlike every other format in the
``--minimal`` FFmpeg vendor scope (mov/matroska/wav/aiff/flac/mp3/ogg).

Invariants ``flac_read_header`` (flacdec.c:59-197) enforces, and that
this module's mutators are built to violate one at a time:

- STREAMINFO (block_type 0) must be the *first* block and occur exactly
  once (flacdec.c:120: "STREAMINFO can only occur once").
- STREAMINFO's length must be exactly ``FLAC_STREAMINFO_SIZE`` (34
  bytes); anything else is rejected (flacdec.c:124).
- ``is_last`` is the only thing separating "one more metadata block
  follows" from "frame data starts here" -- clearing it on the true
  last block sends the parser to read a metadata block header from
  what is actually a FLAC frame sync code.
- block_type 127 is reserved and never explicitly handled by any of the
  type-specific branches (flacdec.c:93-197), so it falls through to
  "skip declared length" with no structural validation at all.

Every mutator here is bounded (block count, declared length clamps) so
a pathological or already-truncated stream costs a fixed amount of
work instead of raising past the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool

FLAC_STREAMINFO_SIZE = 34
FLAC_MAGIC = b"fLaC"

BLOCK_TYPE_STREAMINFO = 0
BLOCK_TYPE_PADDING = 1
BLOCK_TYPE_APPLICATION = 2
BLOCK_TYPE_SEEKTABLE = 3
BLOCK_TYPE_VORBIS_COMMENT = 4
BLOCK_TYPE_CUESHEET = 5
BLOCK_TYPE_PICTURE = 6
BLOCK_TYPE_RESERVED = 127

_MAX_BLOCKS = 4096  # matches the other structural mutators' bounded-work discipline
_MAX_DECLARED_LENGTH = 0xFFFFFF  # 24-bit length field ceiling


@dataclass
class FlacBlock:
    """A single FLAC metadata block.

    ``declared_length``, when not None, overrides ``len(data)`` in the
    emitted 3-byte length field -- the lying-length case
    (``_mutate_declared_length``) that a byte count derived purely from
    ``len(data)`` could never express.
    """

    is_last: bool
    block_type: int
    data: bytes
    declared_length: int | None = None


def parse_flac(data: bytes) -> tuple[list[FlacBlock], bytes] | None:
    """Parse a native FLAC stream into (metadata_blocks, trailing_bytes).

    Returns None unless *data* starts with the "fLaC" magic and at
    least one syntactically well-formed metadata block follows. The
    loop stops at the first block with ``is_last`` set (or when a
    declared length would run past the buffer), exactly mirroring
    ``flac_read_header``'s own read loop.
    """
    if len(data) < 4 or data[:4] != FLAC_MAGIC:
        return None
    pos = 4
    blocks: list[FlacBlock] = []
    is_last = False
    while pos + 4 <= len(data) and not is_last and len(blocks) < _MAX_BLOCKS:
        header = data[pos]
        is_last = bool(header & 0x80)
        block_type = header & 0x7F
        length = int.from_bytes(data[pos + 1 : pos + 4], "big")
        block_start = pos + 4
        block_end = block_start + length
        if block_end > len(data):
            break
        blocks.append(
            FlacBlock(is_last=is_last, block_type=block_type, data=data[block_start:block_end])
        )
        pos = block_end
    if not blocks:
        return None
    return blocks, data[pos:]


def serialize_flac(blocks: list[FlacBlock], trailer: bytes) -> bytes:
    """Serialize metadata blocks (in their current order/flags) plus the trailer.

    Deliberately does *not* recompute ``is_last`` from position: a
    mutator that clears the true last block's flag, or sets it on an
    earlier block, needs that to survive serialization unchanged.
    """
    out = bytearray(FLAC_MAGIC)
    for block in blocks:
        length = len(block.data) if block.declared_length is None else block.declared_length
        length = max(0, min(length, _MAX_DECLARED_LENGTH))
        header = (0x80 if block.is_last else 0x00) | (block.block_type & 0x7F)
        out.append(header)
        out.extend(length.to_bytes(3, "big"))
        out.extend(block.data)
    out.extend(trailer)
    return bytes(out)


class FlacMutator:
    """Structure-aware native-FLAC metadata-block mutator."""

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        self._rng = rng or self._rng
        parsed = parse_flac(data)
        if parsed is None:
            return self._generate_random_flac(max_len=max_len, rng=self._rng)
        blocks, trailer = parsed

        op = self._rng.randint(0, 7)
        mutators = [
            self._mutate_clear_last_flag,
            self._mutate_duplicate_streaminfo,
            self._mutate_declared_length,
            self._mutate_block_type,
            self._mutate_streaminfo_field,
            self._mutate_delete_block,
            self._mutate_duplicate_block,
            self._mutate_shuffle_blocks,
        ]
        blocks = mutators[op](blocks, max_len)
        return serialize_flac(blocks, trailer)[:max_len]

    def _mutate_clear_last_flag(self, blocks: list[FlacBlock], max_len: int) -> list[FlacBlock]:
        """Clear ``is_last`` on the true last block.

        The next 4 bytes the parser reads are then whatever byte
        actually starts the frame data (a FLAC frame sync code, 0xFFF8
        or similar) reinterpreted as [is_last|block_type][length:u24],
        which almost always yields a bogus declared length pointed at
        real frame bytes.
        """
        blocks = list(blocks)
        if blocks:
            last = blocks[-1]
            blocks[-1] = FlacBlock(False, last.block_type, last.data, last.declared_length)
        return blocks

    def _mutate_duplicate_streaminfo(
        self, blocks: list[FlacBlock], max_len: int
    ) -> list[FlacBlock]:
        """Insert a second STREAMINFO block, violating flacdec.c's "only once"."""
        blocks = list(blocks)
        streaminfos = [b for b in blocks if b.block_type == BLOCK_TYPE_STREAMINFO]
        template = streaminfos[0] if streaminfos else blocks[0]
        dup_data = (
            template.data
            if len(template.data) == FLAC_STREAMINFO_SIZE
            else (template.data + b"\x00" * FLAC_STREAMINFO_SIZE)[:FLAC_STREAMINFO_SIZE]
        )
        dup = FlacBlock(is_last=False, block_type=BLOCK_TYPE_STREAMINFO, data=dup_data)
        # Insert somewhere after the first block -- a duplicate *first*
        # block is just the ordinary, already-legal case.
        idx = self._rng.randint(1, len(blocks)) if len(blocks) > 1 else 1
        blocks.insert(min(idx, len(blocks)), dup)
        if blocks:
            last = blocks[-1]
            blocks[-1] = FlacBlock(True, last.block_type, last.data, last.declared_length)
            for b in blocks[:-1]:
                b.is_last = False
        return blocks

    def _mutate_declared_length(self, blocks: list[FlacBlock], max_len: int) -> list[FlacBlock]:
        """Make a block's 3-byte length field lie about its real payload size.

        Shrinking strands real payload bytes to be reinterpreted as the
        next block's header; growing walks the "next block" pointer
        into what is actually later blocks' bytes (or the frame data).
        """
        blocks = list(blocks)
        idx = self._rng.randint(0, len(blocks) - 1)
        target = blocks[idx]
        real = len(target.data)
        declared = self._rng.choice(
            [0, 1, max(0, real - 1), real + 1, real + 64, _MAX_DECLARED_LENGTH]
        )
        blocks[idx] = FlacBlock(target.is_last, target.block_type, target.data, declared)
        return blocks

    def _mutate_block_type(self, blocks: list[FlacBlock], max_len: int) -> list[FlacBlock]:
        """Set a block's type to something else, including the reserved 127."""
        blocks = list(blocks)
        idx = self._rng.randint(0, len(blocks) - 1)
        target = blocks[idx]
        candidates = [
            t
            for t in (
                BLOCK_TYPE_STREAMINFO,
                BLOCK_TYPE_PADDING,
                BLOCK_TYPE_APPLICATION,
                BLOCK_TYPE_SEEKTABLE,
                BLOCK_TYPE_VORBIS_COMMENT,
                BLOCK_TYPE_CUESHEET,
                BLOCK_TYPE_PICTURE,
                BLOCK_TYPE_RESERVED,
            )
            if t != target.block_type
        ]
        blocks[idx] = FlacBlock(
            target.is_last, self._rng.choice(candidates), target.data, target.declared_length
        )
        return blocks

    def _mutate_streaminfo_field(self, blocks: list[FlacBlock], max_len: int) -> list[FlacBlock]:
        """Corrupt the bit-packed sample-rate/channels/bps/total-samples region.

        STREAMINFO layout: [min/max blocksize: u16 u16][min/max
        framesize: u24 u24][20 bits sample_rate | 3 bits channels-1 | 5
        bits bps-1 | 36 bits total_samples][16 bytes MD5]. flacdec.c
        reads total_samples as ``(AV_RB64(extradata+13) >> 24) &
        ((1<<36)-1)``, i.e. the packed region spans payload bytes
        10..20. Randomizing it, or forcing it to all-ones (the maximum
        36-bit total_samples value), hits that unpack directly instead
        of a blind whole-buffer flip finding it by chance.
        """
        blocks = list(blocks)
        streaminfos = [i for i, b in enumerate(blocks) if b.block_type == BLOCK_TYPE_STREAMINFO]
        if not streaminfos:
            return blocks
        idx = self._rng.choice(streaminfos)
        target = blocks[idx]
        raw = bytearray(target.data)
        if len(raw) < 18:
            return blocks
        if self._rng.random() < 0.5:
            raw[10:18] = b"\xff" * 8  # max sample_rate/channels/bps/total_samples
        else:
            for i in range(10, 18):
                raw[i] = self._rng.randint(0, 0xFF)
        blocks[idx] = FlacBlock(
            target.is_last, target.block_type, bytes(raw), target.declared_length
        )
        return blocks

    def _mutate_delete_block(self, blocks: list[FlacBlock], max_len: int) -> list[FlacBlock]:
        """Delete a non-STREAMINFO block, shifting every later offset."""
        blocks = list(blocks)
        candidates = [i for i, b in enumerate(blocks) if b.block_type != BLOCK_TYPE_STREAMINFO]
        if not candidates or len(blocks) <= 1:
            return blocks
        del blocks[self._rng.choice(candidates)]
        if blocks:
            last = blocks[-1]
            blocks[-1] = FlacBlock(True, last.block_type, last.data, last.declared_length)
        return blocks

    def _mutate_duplicate_block(self, blocks: list[FlacBlock], max_len: int) -> list[FlacBlock]:
        """Duplicate a random block in place."""
        blocks = list(blocks)
        idx = self._rng.randint(0, len(blocks) - 1)
        target = blocks[idx]
        dup = FlacBlock(False, target.block_type, target.data, target.declared_length)
        blocks.insert(idx + 1, dup)
        last = blocks[-1]
        blocks[-1] = FlacBlock(True, last.block_type, last.data, last.declared_length)
        for b in blocks[:-1]:
            b.is_last = False
        return blocks

    def _mutate_shuffle_blocks(self, blocks: list[FlacBlock], max_len: int) -> list[FlacBlock]:
        """Reorder the non-STREAMINFO blocks (STREAMINFO must stay first)."""
        blocks = list(blocks)
        if len(blocks) < 3:
            return blocks
        head, rest = blocks[0], blocks[1:]
        self._rng.shuffle(rest)
        new_blocks = [head] + rest
        for b in new_blocks[:-1]:
            b.is_last = False
        last = new_blocks[-1]
        new_blocks[-1] = FlacBlock(True, last.block_type, last.data, last.declared_length)
        return new_blocks

    def _generate_random_flac(self, _blocks=None, max_len: int = 65536, rng=None) -> bytes:
        """Generate a minimal, valid native FLAC file."""
        # An int in the first slot is a max_len passed positionally -- same
        # overload convention every other mutator's generator here follows.
        if isinstance(_blocks, int):
            max_len = _blocks
        self._rng = rng or self._rng
        r = self._rng

        min_blocksize = 4096
        max_blocksize = 4096
        min_framesize = 0
        max_framesize = 0
        sample_rate = 44100
        channels = 2
        bps = 16
        total_samples = r.randint(0, 1_000_000)

        packed = (sample_rate << 44) | ((channels - 1) << 41) | ((bps - 1) << 36) | total_samples
        streaminfo = (
            min_blocksize.to_bytes(2, "big")
            + max_blocksize.to_bytes(2, "big")
            + min_framesize.to_bytes(3, "big")
            + max_framesize.to_bytes(3, "big")
            + packed.to_bytes(8, "big")
            + bytes(16)  # MD5, zeroed
        )
        blocks = [FlacBlock(is_last=True, block_type=BLOCK_TYPE_STREAMINFO, data=streaminfo)]
        trailer = bytes(r.randint(0, 0xFF) for _ in range(r.randint(0, 32)))
        return serialize_flac(blocks, trailer)[:max_len]
