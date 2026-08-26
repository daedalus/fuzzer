"""Structure-aware Ogg container mutator.

Ogg is page-based, wrapping Vorbis/Theora/Opus/FLAC-in-Ogg — all of
which reach libavformat's `oggdec.c` and go through this same page
framing regardless of which codec the payload carries. None of the
existing format operators touch it.

Page layout ("OggS", all multi-byte fields little-endian):

  capture_pattern(4="OggS") version(1) header_type_flag(1)
  granule_position(8) bitstream_serial_number(4) page_sequence_number(4)
  CRC_checksum(4) page_segments(1) segment_table(page_segments bytes)
  data(sum(segment_table) bytes)

header_type_flag bit 0x01 = continued packet, 0x02 = beginning-of-stream,
0x04 = end-of-stream. serial_number is how the demuxer tells multiple
logical bitstreams apart inside one physical file (e.g. audio + video
tracks, or chained streams) — corrupting it targets stream-demux
confusion specifically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

OGG_MIN_PAGE = 27  # header without segment table


@dataclass
class OggPage:
    version: int
    header_type: int
    granule_position: int
    serial_number: int
    sequence_number: int
    checksum: int
    segment_table: bytes
    data: bytes

    def to_bytes(self) -> bytes:
        return (
            b"OggS"
            + bytes([self.version & 0xFF, self.header_type & 0xFF])
            + (self.granule_position & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
            + (self.serial_number & 0xFFFFFFFF).to_bytes(4, "little")
            + (self.sequence_number & 0xFFFFFFFF).to_bytes(4, "little")
            + (self.checksum & 0xFFFFFFFF).to_bytes(4, "little")
            + bytes([len(self.segment_table) & 0xFF])
            + self.segment_table
            + self.data
        )


def parse_ogg_pages(data: bytes) -> list[OggPage] | None:
    if len(data) < OGG_MIN_PAGE or data[:4] != b"OggS":
        return None

    pos = 0
    n = len(data)
    pages: list[OggPage] = []
    while pos + OGG_MIN_PAGE <= n and data[pos : pos + 4] == b"OggS":
        version = data[pos + 4]
        header_type = data[pos + 5]
        granule = int.from_bytes(data[pos + 6 : pos + 14], "little")
        serial = int.from_bytes(data[pos + 14 : pos + 18], "little")
        seq = int.from_bytes(data[pos + 18 : pos + 22], "little")
        checksum = int.from_bytes(data[pos + 22 : pos + 26], "little")
        page_segments = data[pos + 26]
        seg_start = pos + 27
        seg_end = seg_start + page_segments
        if seg_end > n:
            break
        seg_table = data[seg_start:seg_end]
        data_len = sum(seg_table)
        data_start = seg_end
        data_end = min(data_start + data_len, n)
        page_data = data[data_start:data_end]
        pages.append(OggPage(version, header_type, granule, serial, seq, checksum, seg_table, page_data))
        pos = data_end

    return pages if pages else None


def serialize_ogg_pages(pages: list[OggPage]) -> bytes:
    return b"".join(p.to_bytes() for p in pages)


class OggMutator:
    """Structure-aware Ogg page-level mutator."""

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        pages = parse_ogg_pages(data)
        if not pages:
            return self._generate_random_ogg(max_len=max_len, rng=rng)

        op = rng.randint(0, 7)
        mutators = [
            self._mutate_header_type,
            self._mutate_granule_position,
            self._mutate_serial_number,
            self._mutate_sequence_number,
            self._mutate_segment_table,
            self._mutate_checksum,
            self._duplicate_page,
            self._delete_page,
        ]
        mutators[op](pages, rng)
        return serialize_ogg_pages(pages)[:max_len]

    def _mutate_header_type(self, pages: list[OggPage], rng) -> None:
        """Flip continuation/BOS/EOS bits — tests packet-reassembly and
        stream-boundary handling independent of payload content."""
        target = rng.choice(pages)
        target.header_type ^= 1 << rng.randint(0, 2)

    def _mutate_granule_position(self, pages: list[OggPage], rng) -> None:
        target = rng.choice(pages)
        target.granule_position = rng.choice([0, 0xFFFFFFFFFFFFFFFF, rng.getrandbits(64)])

    def _mutate_serial_number(self, pages: list[OggPage], rng) -> None:
        """Remap a page to a different logical-stream serial — tests
        multiplexed-stream demux confusion."""
        target = rng.choice(pages)
        others = [p.serial_number for p in pages if p.serial_number != target.serial_number]
        target.serial_number = rng.choice(others) if others else rng.getrandbits(32)

    def _mutate_sequence_number(self, pages: list[OggPage], rng) -> None:
        target = rng.choice(pages)
        target.sequence_number = rng.choice([0, 0xFFFFFFFF, rng.getrandbits(32)])

    def _mutate_segment_table(self, pages: list[OggPage], rng) -> None:
        """Corrupt one lacing-value byte in the segment table without
        touching `data` — creates a declared-vs-actual data-length
        mismatch, the Ogg-specific analogue of a box/chunk size mismatch."""
        target = rng.choice(pages)
        if not target.segment_table:
            return
        seg = bytearray(target.segment_table)
        idx = rng.randint(0, len(seg) - 1)
        seg[idx] = rng.randint(0, 255)
        target.segment_table = bytes(seg)

    def _mutate_checksum(self, pages: list[OggPage], rng) -> None:
        target = rng.choice(pages)
        target.checksum = rng.getrandbits(32)

    def _duplicate_page(self, pages: list[OggPage], rng) -> None:
        idx = rng.randint(0, len(pages) - 1)
        orig = pages[idx]
        pages.insert(
            idx + 1,
            OggPage(orig.version, orig.header_type, orig.granule_position, orig.serial_number,
                    orig.sequence_number, orig.checksum, orig.segment_table, orig.data),
        )

    def _delete_page(self, pages: list[OggPage], rng) -> None:
        if len(pages) > 1:
            pages.pop(rng.randint(0, len(pages) - 1))

    def _generate_random_ogg(self, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        # Minimal single BOS page carrying an OpusHead-shaped identification
        # packet — small, single-segment, plausible for a demuxer to probe.
        opus_head = b"OpusHead" + bytes([1, 2, 0, 0]) + (48000).to_bytes(4, "little") + bytes(3)
        page = OggPage(
            version=0,
            header_type=0x02,  # BOS
            granule_position=0,
            serial_number=rng.getrandbits(32),
            sequence_number=0,
            checksum=0,
            segment_table=bytes([len(opus_head)]),
            data=opus_head,
        )
        return serialize_ogg_pages([page])[:max_len]
