"""Structure-aware PGS (Presentation Graphics Subtitle) segment mutator.

PGS segments have a simple fixed-header structure:
  [0x50 0x47]  — PG magic (2 bytes)
  [pts: u32]   — 4 bytes LE
  [dts: u32]   — 4 bytes LE
  [seg_type: u8] — segment type
  [seg_size: u16] — segment size (big-endian)
  [payload]    — seg_size bytes

Segment types: 0x14 (PCS), 0x15 (WDS), 0x16 (PDS), 0x17 (ODS), 0x80 (END).

A real bug was already found in FFmpeg's PGS decode path, making this
the area with the highest confirmed hit rate in the fuzz target.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.generic import _swap_pair

import random
import struct
from dataclasses import dataclass, field

PG_MAGIC = b"\x50\x47"

# Segment types
SEG_PCS = 0x14  # Presentation Composition Segment
SEG_WDS = 0x15  # Window Definition Segment
SEG_PDS = 0x16  # Palette Definition Segment
SEG_ODS = 0x17  # Object Definition Segment
SEG_END = 0x80  # End of Display Set Segment

SEGMENT_TYPES = [SEG_PCS, SEG_WDS, SEG_PDS, SEG_ODS, SEG_END]
NON_CRITICAL_TYPES = [SEG_PCS, SEG_WDS, SEG_PDS, SEG_ODS]  # END cannot be deleted

# Headers sizes
PGS_HEADER_SIZE = 13  # 2 (magic) + 4 (pts) + 4 (dts) + 1 (seg_type) + 2 (seg_size)


@dataclass
class PgsSegment:
    """A single PGS segment."""

    pts: int
    dts: int
    seg_type: int
    seg_size: int
    payload: bytes = field(default_factory=bytes)


def parse_pgs_segments(data: bytes) -> list[PgsSegment] | None:
    """Parse a stream of PGS segments.

    Scans for PG magic (\\x50\\x47) and extracts all complete segments.
    Returns None if no PG magic is found.
    """
    if len(data) < PGS_HEADER_SIZE or not data.startswith(PG_MAGIC):
        return None

    segments: list[PgsSegment] = []
    pos = 0

    while pos + PGS_HEADER_SIZE <= len(data):
        # Check for PG magic at current position
        if data[pos : pos + 2] != PG_MAGIC:
            break

        pts = struct.unpack_from("<I", data, pos + 2)[0]
        dts = struct.unpack_from("<I", data, pos + 6)[0]
        seg_type = data[pos + 10]
        seg_size = struct.unpack_from(">H", data, pos + 11)[0]

        if pos + PGS_HEADER_SIZE + seg_size > len(data):
            seg_size = len(data) - (pos + PGS_HEADER_SIZE)
            if seg_size < 0:
                break

        payload = data[pos + PGS_HEADER_SIZE : pos + PGS_HEADER_SIZE + seg_size]

        segments.append(
            PgsSegment(pts=pts, dts=dts, seg_type=seg_type, seg_size=seg_size, payload=payload)
        )
        pos += PGS_HEADER_SIZE + seg_size

    return segments if segments else None


def serialize_pgs_segments(segments: list[PgsSegment]) -> bytes:
    """Serialize PGS segments back to bytes using stored seg_size (may differ from payload)."""
    buf = bytearray()
    for seg in segments:
        buf.extend(PG_MAGIC)
        buf.extend(struct.pack("<I", seg.pts))
        buf.extend(struct.pack("<I", seg.dts))
        buf.append(seg.seg_type)
        buf.extend(struct.pack(">H", seg.seg_size))
        buf.extend(seg.payload)
    return bytes(buf)


class PgsMutator:
    """Structure-aware PGS segment mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        self._rng = rng or random
        segments = parse_pgs_segments(data)
        if segments is None:
            return self._generate_random_pgs(max_len=max_len, rng=self._rng)

        if not segments:
            return self._generate_random_pgs(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 8)
        mutators = [
            self._mutate_pts,
            self._mutate_dts,
            self._mutate_segment_type,
            self._mutate_payload,
            self._duplicate_segment,
            self._delete_segment,
            self._reorder_segments,
            self._mutate_segment_size,
            self._generate_random_pgs,
        ]
        result = mutators[op](segments, max_len)
        if isinstance(result, list):
            return serialize_pgs_segments(result)[:max_len]
        return result[:max_len]

    def _mutate_pts(self, segments: list[PgsSegment], max_len: int) -> list[PgsSegment]:
        """Corrupt PTS of a random segment."""
        seg = self._rng.choice(segments)
        seg.pts = self._rng.choice([0, 1, 0xFFFFFFFF, self._rng.randint(0, 0xFFFFFFFF)])
        return segments

    def _mutate_dts(self, segments: list[PgsSegment], max_len: int) -> list[PgsSegment]:
        """Corrupt DTS of a random segment."""
        seg = self._rng.choice(segments)
        seg.dts = self._rng.choice([0, 1, 0xFFFFFFFF, self._rng.randint(0, 0xFFFFFFFF)])
        return segments

    def _mutate_segment_type(self, segments: list[PgsSegment], max_len: int) -> list[PgsSegment]:
        """Replace a segment type with an unexpected value."""
        seg = self._rng.choice(segments)
        weird_types = [0x00, 0x01, 0x12, 0x13, 0x18, 0x81, 0xFF, 0xFE]
        seg.seg_type = self._rng.choice(weird_types)
        return segments

    def _mutate_payload(self, segments: list[PgsSegment], max_len: int) -> list[PgsSegment]:
        """Flip random bytes in a segment's payload."""
        for seg in segments:
            if seg.payload:
                data = bytearray(seg.payload)
                for _ in range(self._rng.randint(1, min(4, len(data)))):
                    idx = self._rng.randint(0, len(data) - 1)
                    data[idx] ^= 1 << self._rng.randint(0, 7)
                seg.payload = bytes(data)
                seg.seg_size = len(seg.payload)
        return segments

    def _duplicate_segment(self, segments: list[PgsSegment], max_len: int) -> list[PgsSegment]:
        """Clone a random segment and insert it after the original."""
        seg = self._rng.choice(segments)
        dup = PgsSegment(
            pts=seg.pts,
            dts=seg.dts,
            seg_type=seg.seg_type,
            seg_size=seg.seg_size,
            payload=seg.payload,
        )
        idx = segments.index(seg) + 1
        segments.insert(idx, dup)
        return segments

    def _delete_segment(self, segments: list[PgsSegment], max_len: int) -> list[PgsSegment]:
        """Delete a non-critical segment (not the END marker)."""
        deletable = [(i, s) for i, s in enumerate(segments) if s.seg_type != SEG_END]
        if deletable:
            idx, _seg = self._rng.choice(deletable)
            segments.pop(idx)
        return segments

    def _reorder_segments(self, segments: list[PgsSegment], max_len: int) -> list[PgsSegment]:
        """Swap two random segments."""
        if (pair := _swap_pair(len(segments), self._rng)) is not None:
            i, j = pair
            segments[i], segments[j] = segments[j], segments[i]
        return segments

    def _mutate_segment_size(self, segments: list[PgsSegment], max_len: int) -> list[PgsSegment]:
        """Corrupt the size field independently of the payload length.

        Creates a mismatch between declared seg_size and actual payload
        to probe bounds-checking in the demuxer.
        """
        seg = self._rng.choice(segments)
        extreme_sizes = [
            0,
            1,
            0xFFFF,
            len(seg.payload),
            len(seg.payload) + 1,
            max(0, len(seg.payload) - 1),
        ]
        seg.seg_size = self._rng.choice(extreme_sizes)
        return segments

    def _generate_random_pgs(self, _segments=None, max_len: int = 65536, rng=None) -> bytes:
        """Generate a minimal random PGS stream."""
        # An int in the first slot is a max_len passed positionally. Without
        # this the cap lands in the vestigial placeholder and is dropped, and
        # the generator silently falls back to its own default -- the same
        # overload bmp/gzip/jpeg/zlib already handle and document.
        if isinstance(_segments, int):
            max_len = _segments
        self._rng = rng or self._rng
        segments: list[PgsSegment] = []

        # Always include PCS, ODS, and END for a minimally valid display set
        pcs_payload = bytes(self._rng.randint(0, 255) for _ in range(self._rng.randint(4, 20)))
        segments.append(
            PgsSegment(
                pts=self._rng.randint(0, 0xFFFFFFFF),
                dts=0xFFFFFFFF,
                seg_type=SEG_PCS,
                seg_size=len(pcs_payload),
                payload=pcs_payload,
            )
        )

        # Optionally add WDS, PDS
        extra_types = self._rng.sample([SEG_WDS, SEG_PDS], self._rng.randint(0, 2))
        for st in extra_types:
            pl = bytes(self._rng.randint(0, 255) for _ in range(self._rng.randint(2, 16)))
            segments.append(
                PgsSegment(
                    pts=self._rng.randint(0, 0xFFFFFFFF),
                    dts=0xFFFFFFFF,
                    seg_type=st,
                    seg_size=len(pl),
                    payload=pl,
                )
            )

        ods_payload = bytes(self._rng.randint(0, 255) for _ in range(self._rng.randint(4, 32)))
        segments.append(
            PgsSegment(
                pts=self._rng.randint(0, 0xFFFFFFFF),
                dts=0xFFFFFFFF,
                seg_type=SEG_ODS,
                seg_size=len(ods_payload),
                payload=ods_payload,
            )
        )

        segments.append(
            PgsSegment(pts=0, dts=0xFFFFFFFF, seg_type=SEG_END, seg_size=0, payload=b"")
        )

        result = serialize_pgs_segments(segments)
        return result[:max_len]
