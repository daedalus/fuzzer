"""Structure-aware ISO-BMFF (MP4/MOV) box mutator.

ISO-BMFF box structure:
  [size: u32 BE]  — total box size including header
  [type: 4 bytes]  — fourcc (ftyp, moov, trak, mdia, minf, stbl, stsd, hdlr, ...)
  [payload: size - 8 bytes]
    — Container boxes: payload contains sub-boxes
    — Leaf boxes: payload is data

This is the most common container format FFmpeg handles. It is also
deeply nested (moov → trak → mdia → minf → stbl → stsd), which is
exactly the kind of length-field-heavy structure that plain byte
mutation struggles with at nesting depth.

Stream-type/codec-ID field mutation is embedded here as sub-operations
that target hdlr handler_type and stsd codec fourcc fields.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from fuzzer_tool.core.mutations.generic import _swap_pair
from fuzzer_tool.core.rand_pool import RandPool

# Box types that contain sub-boxes
CONTAINER_TYPES = {
    b"moov",
    b"trak",
    b"mdia",
    b"minf",
    b"stbl",
    b"moof",
    b"traf",
    b"mvex",
    b"edts",
    b"dinf",
    b"udta",
    b"meta",
    b"skip",
    b"free",
    b"mdra",
    b"nmhd",
    b"smhd",
    b"vmhd",
    b"hmhd",
    b"sthd",
    b"meco",
    b"strk",
    b"ipro",
    b"sinf",
    b"schi",
    b"mfra",
    b"tfra",
    b"mfro",
    b"pdin",
}

# Box types whose payloads are full boxes (with version/flags prefix)
FULL_BOX_TYPES = {
    b"stsd",
    b"stts",
    b"stsc",
    b"stsz",
    b"stco",
    b"co64",
    b"stss",
    b"ctts",
    b"hdlr",
    b"mdhd",
    b"mvhd",
    b"tkhd",
    b"vmhd",
    b"smhd",
    b"nmhd",
    b"dref",
    b"elst",
    b"minf",
}

# Stream-type handler fourcc values
HANDLER_TYPES = [b"vide", b"soun", b"subt", b"meta", b"hint", b"auxv"]

# Codec fourcc values that can be swapped into stsd entries
STSD_CODECS = [
    b"avc1",
    b"mp4a",
    b"pgss",
    b"subp",
    b"tx3g",
    b"hvc1",
    b"hev1",
    b"hev2",
    b"mp4v",
    b"s263",
    b"av01",
    b"vp09",
    b"theo",
    b"vorb",
    b"Opus",
    b"FLAC",
    b"ac-3",
    b"eac3",
    b"dtsc",
    b"dtsh",
    b"dtse",
    b"dtsl",
    b"mlpa",
    b"lpcm",
    b"sowt",
    b"samr",
    b"sawb",
]


# Edge values for sample-table fields: zero, one, sign boundary, all-ones.
EDGE_U32 = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
EDGE_U64 = (0, 1, 0xFFFFFFFF, 0x1_0000_0000, 0x7FFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF)
COUNT_WIDTH = 4


@dataclass(frozen=True)
class _TableLayout:
    """Where a sample table keeps its count and the shape of one entry.

    Offsets are into the FullBox payload (version/flags first).
    """

    count_off: int
    entries_off: int
    field_widths: tuple[int, ...]

    @property
    def entry_len(self) -> int:
        return sum(self.field_widths)


# ISO/IEC 14496-12 8.6.1.2 (stts), 8.7.4 (stsc), 8.7.3 (stsz), 8.7.5
# (stco/co64), 8.6.6 (elst). stsz: sample_size precedes sample_count.
_TABLE_LAYOUTS = {
    b"stts": _TableLayout(4, 8, (4, 4)),  # sample_count, sample_delta
    b"stsc": _TableLayout(4, 8, (4, 4, 4)),  # first_chunk, per_chunk, desc_index
    b"stsz": _TableLayout(8, 12, (4,)),  # entry_size (when sample_size == 0)
    b"stco": _TableLayout(4, 8, (4,)),  # chunk_offset
    b"co64": _TableLayout(4, 8, (8,)),
    b"elst": _TableLayout(4, 8, (4, 4, 4)),  # duration, media_time, rate (v0)
}
_ELST_V1 = _TableLayout(4, 8, (8, 8, 4))
SAMPLE_TABLE_TYPES = frozenset(_TABLE_LAYOUTS)


def _table_layout(box: Box) -> _TableLayout:
    if box.box_type == b"elst" and box.data[:1] == b"\x01":
        return _ELST_V1
    return _TABLE_LAYOUTS[box.box_type]


def _find_types(boxes: list[Box], types: frozenset[bytes]) -> list[Box]:
    """All boxes in the tree whose type is in *types*, depth-first."""
    found: list[Box] = []
    stack = list(reversed(boxes))
    while stack:
        box = stack.pop()
        if box.box_type in types:
            found.append(box)
        stack.extend(reversed(box.children))
    return found


# AVCC/HVCC: NAL units in mdat carry a big-endian length prefix whose width
# is lengthSizeMinusOne + 1 (ISO/IEC 14496-15 5.3.3.1 / 8.3.3.1).
DEFAULT_NAL_LENGTH_SIZE = 4
_NAL_CONFIG_SIZE_AT = {b"avcC": 4, b"hvcC": 21}  # config byte holding the field
_LENGTH_SIZE_MASK = 0x03
FOURCC_LEN = 4


def avcc_length_size(boxes: list[Box]) -> int:
    """NAL length-prefix width from the first avcC/hvcC in any stsd, default 4.

    stsd payloads aren't parsed into boxes, so the config is found by fourcc.
    """
    for stsd in _find_types(boxes, frozenset({b"stsd"})):
        for tag, at in _NAL_CONFIG_SIZE_AT.items():
            i = stsd.data.find(tag)
            j = i + FOURCC_LEN + at
            if i != -1 and j < len(stsd.data):
                return (stsd.data[j] & _LENGTH_SIZE_MASK) + 1
    return DEFAULT_NAL_LENGTH_SIZE


def _nal_length_sites(data: bytes, width: int) -> list[tuple[int, int]]:
    """(offset, length) of each consistent length prefix from the start of *data*.

    Stops at the first zero or overrunning length: past it the bytes are
    not (or no longer) a length-prefixed NAL run (e.g. interleaved audio).
    """
    sites: list[tuple[int, int]] = []
    pos, n = 0, len(data)
    while pos + width <= n:
        length = int.from_bytes(data[pos : pos + width], "big")
        if length == 0 or pos + width + length > n:
            break
        sites.append((pos, length))
        pos += width + length
    return sites


@dataclass
class Box:
    """A single ISO-BMFF box.

    Container boxes hold sub-boxes in *children* (not *data*).
    Leaf boxes hold their raw payload in *data*.
    """

    box_type: bytes
    size_orig: int  # original declared size
    data: bytes = field(default_factory=bytes)
    children: list[Box] = field(default_factory=list)

    @property
    def is_container(self) -> bool:
        return self.box_type in CONTAINER_TYPES


def parse_boxes(data: bytes, offset: int = 0, parent_end: int = 0) -> list[Box] | None:
    """Parse ISO-BMFF boxes from *data* at *offset*.

    Returns list of boxes, or None if parsing fails at the first box.
    Returns empty list for data that doesn't look like ISOBMFF.
    """
    if len(data) < 8:
        return None

    boxes: list[Box] = []
    pos = offset
    end = parent_end if parent_end > 0 else len(data)

    while pos + 8 <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        box_type = data[pos + 4 : pos + 8]

        # Large size (size == 1): next 8 bytes are the real size
        if size == 1:
            if pos + 16 > end:
                break
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            box_type = data[pos + 4 : pos + 8]
            # Extended size: actual data starts after 16 bytes of header
            payload_start = pos + 16
            if payload_start > end:
                break
        else:
            if size < 8:
                # Box size less than header — invalid but continue scanning
                size = 8
            payload_start = pos + 8

        if payload_start > end:
            break

        # Clamp declared size to available data
        actual_end = min(pos + size, end) if size > 0 else end
        payload_end = actual_end

        # Check if container (either by known type or has sub-box structure)
        children: list[Box] = []
        if box_type in CONTAINER_TYPES:
            child_boxes = _parse_children(data, payload_start, payload_end)
            if child_boxes is not None:
                children = child_boxes

        payload = data[payload_start:payload_end] if not children else b""

        boxes.append(Box(box_type=box_type, size_orig=size, data=payload, children=children))

        if size == 0:
            break
        pos = actual_end

    return boxes if boxes else None


def _parse_children(data: bytes, start: int, end: int) -> list[Box] | None:
    """Parse sub-boxes within a container payload range."""
    result: list[Box] = []
    pos = start
    while pos + 8 <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        if pos + size > end:
            break
        if size < 8:
            pos += 1
            continue
        box_type = data[pos + 4 : pos + 8]
        payload_start = pos + 8
        payload_end = pos + size

        children: list[Box] = []
        if box_type in CONTAINER_TYPES:
            child_boxes = _parse_children(data, payload_start, payload_end)
            if child_boxes is not None:
                children = child_boxes

        payload = data[payload_start:payload_end] if not children else b""
        result.append(Box(box_type=box_type, size_orig=size, data=payload, children=children))
        pos = payload_end
    return result if result else None


def serialize_boxes(boxes: list[Box]) -> bytes:
    """Serialize boxes back to bytes using stored size_orig (may differ from payload).

    Supports extended-size (64-bit) boxes when size_orig exceeds u32 range.
    """
    buf = bytearray()
    for box in boxes:
        payload = serialize_boxes(box.children) if box.children else bytearray(box.data)
        if box.size_orig > 0xFFFFFFFF:
            # Extended size (64-bit)
            buf.extend(b"\x00\x00\x00\x01")
            buf.extend(struct.pack(">Q", box.size_orig))
            buf.extend(box.box_type)
        else:
            buf.extend(struct.pack(">I", box.size_orig))
            buf.extend(box.box_type)
        buf.extend(payload)
    return bytes(buf)


def _find_hdlr(boxes: list[Box]) -> list[tuple[list[Box], int]]:
    """Find all hdlr boxes in the box tree.

    Returns list of (parent_children_list, index) tuples.
    """
    found: list[tuple[list[Box], int]] = []
    for i, box in enumerate(boxes):
        if box.box_type == b"hdlr":
            found.append((boxes, i))
        if box.children:
            found.extend(_find_hdlr(box.children))
    return found


def _find_stsd(boxes: list[Box]) -> list[tuple[list[Box], int]]:
    """Find all stsd boxes in the box tree."""
    found: list[tuple[list[Box], int]] = []
    for i, box in enumerate(boxes):
        if box.box_type == b"stsd":
            found.append((boxes, i))
        if box.children:
            found.extend(_find_stsd(box.children))
    return found


def _find_ctts(boxes: list[Box]) -> list[Box]:
    """Find all ctts boxes in the box tree."""
    found: list[Box] = []
    for box in boxes:
        if box.box_type == b"ctts":
            found.append(box)
        if box.children:
            found.extend(_find_ctts(box.children))
    return found


def _find_moov(boxes: list[Box]) -> list[tuple[list[Box], int]]:
    """Find all moov boxes in the box tree.

    Returns list of (parent_children_list, index) tuples, matching
    ``_find_hdlr``/``_find_stsd``.
    """
    found: list[tuple[list[Box], int]] = []
    for i, box in enumerate(boxes):
        if box.box_type == b"moov":
            found.append((boxes, i))
        if box.children:
            found.extend(_find_moov(box.children))
    return found


def _mutate_hdlr_handler_type(box: Box, rng: Any) -> None:
    """Replace the handler_type field in a hdlr box payload.

    hdlr payload: [predefined: u32][handler_type: 4 bytes][reserved: 3*u32][name: str]
    Handler type at offset 4-7.
    """
    data = bytearray(box.data)
    if len(data) >= 8:
        current = bytes(data[4:8])
        new_type = rng.choice(
            [t for t in HANDLER_TYPES if t != current] + [b"\xff\xff\xff\xff", b"\x00\x00\x00\x00"]
        )
        data[4:8] = new_type
        box.data = bytes(data)


def _mutate_stsd_codec(box: Box, rng: Any) -> None:
    """Corrupt the first sample entry's codec fourcc in an stsd box.

    stsd payload: [version: u8][flags: u24][entry_count: u32][entries...]
    Each sample entry: [entry_size: u32][codec_fourcc: 4 bytes][...]

    Codec fourcc at offset 8+4 = 12 (after stsd header).
    """
    data = bytearray(box.data)
    if len(data) >= 16:
        current = bytes(data[12:16])
        new_codec = rng.choice(
            [c for c in STSD_CODECS if c != current] + [b"\xff\xff\xff\xff", b"\x00\x00\x00\x00"]
        )
        data[12:16] = new_codec
        box.data = bytes(data)


class IsobmffMutator:
    """Structure-aware ISO-BMFF box mutator."""

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        self._rng = rng or self._rng
        boxes = parse_boxes(data)
        if boxes is None or not boxes:
            return self._generate_random_isobmff(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 15)
        mutators = [
            self._mutate_box_type,
            self._mutate_box_size,
            self._mutate_ftyp,
            self._mutate_hdlr,
            self._mutate_stsd,
            self._mutate_ctts,
            self._mutate_ctts_sample_count,
            self._mutate_ctts_sample_offset,
            self._swap_boxes,
            self._delete_box,
            self._duplicate_box,
            self._truncate_box,
            self._mutate_free_hoov_confusion,
            self._generate_random_isobmff,
            self._mutate_sample_table,
            self._mutate_nal_length,
        ]
        result = mutators[op](boxes, max_len)
        if isinstance(result, list):
            return serialize_boxes(result)[:max_len]
        return result[:max_len]

    def _mutate_box_type(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Corrupt a random box type (fourcc)."""
        target = self._rng.choice(boxes)
        weird_fourccs = [b"\x00\x00\x00\x00", b"\xff\xff\xff\xff", b"    ", b"xxxx"]
        target.box_type = self._rng.choice(weird_fourccs)
        return boxes

    def _mutate_box_size(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Corrupt the size field of a random box."""
        target = self._rng.choice(boxes)

        target.size_orig = self._rng.choice(
            [0, 1, 8, max_len, 0xFFFFFFFF, self._rng.randint(0, max_len)]
        )
        # Recompute/truncate/pad payload from new size.
        # Clamp payload_len to max_len to avoid OOM on extreme size values
        # (e.g. 0xFFFFFFFF which would allocate ~4 GB).
        raw = target.data
        payload_len = max(0, min(target.size_orig - 8, max_len))
        if payload_len >= len(raw):
            target.data = raw + b"\x00" * (payload_len - len(raw))
        else:
            target.data = raw[:payload_len]
        return boxes

    def _mutate_ftyp(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Corrupt the ftyp compatible-brands list."""
        for box in boxes:
            if box.box_type == b"ftyp":
                data = bytearray(box.data)
                if len(data) >= 8:
                    # Corrupt major brand (offset 0-3) or a compatible brand
                    pos = (
                        0
                        if self._rng.randint(0, 1) == 0
                        else self._rng.randint(8, max(8, len(data) - 4))
                    )
                    data[pos : pos + 4] = self._rng.choice(
                        [b"xxxx", b"\xff\xff\xff\xff", b"\x00\x00\x00\x00", b"????"]
                    )
                box.data = bytes(data)
        return boxes

    def _mutate_free_hoov_confusion(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Relabel a moov box as free/hoov (mov.c's moov-in-free-atom quirk).

        mov_read_default (mov.c:10158-10173) special-cases a top-level
        'free' or 'hoov' box: it peeks 4 bytes into the payload, and if
        those 4 bytes read 'mvhd' or 'cmov' it reclassifies the whole box
        as 'moov' and parses it as one. A moov box's first child is
        ordinarily mvhd, whose own [size][type] header already puts the
        literal bytes 'mvhd' at payload offset 4:8 -- exactly where the
        check peeks. Relabeling the outer fourcc with the children left
        untouched reaches that reclassification path directly. This is
        fourcc *aliasing*, not corruption -- the one case here where
        renaming a box to something wrong makes it parse as something
        else *more* fully, not less.
        """
        locations = _find_moov(boxes)
        if not locations:
            return boxes
        parent, idx = self._rng.choice(locations)
        parent[idx].box_type = self._rng.choice([b"free", b"hoov"])
        return boxes

    def _mutate_hdlr(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Mutate handler_type fields in hdlr boxes — stream-type mutation."""
        hdlr_locations = _find_hdlr(boxes)
        if hdlr_locations:
            parent, idx = self._rng.choice(hdlr_locations)
            _mutate_hdlr_handler_type(parent[idx], self._rng)
        return boxes

    def _mutate_stsd(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Mutate codec fourcc in stsd sample description entries."""
        stsd_locations = _find_stsd(boxes)
        if stsd_locations:
            parent, idx = self._rng.choice(stsd_locations)
            _mutate_stsd_codec(parent[idx], self._rng)
        return boxes

    def _mutate_ctts(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Mutate ctts sample count/offset fields (CVE-2022-2566).

        ctts is a FullBox: [version:u32][flags:u24][entry_count:u32]
        then entry_count * {sample_count:u32}[sample_offset:s32}.
        The vulnerability: sum of ctts_data[i].count overflows
        sample_offsets_count → av_calloc too small → OOB writes.
        """
        ctts_locations = []
        for box in boxes:
            if box.box_type == b"ctts":
                ctts_locations.append(box)
            if box.children:
                ctts_locations.extend(_find_ctts(box.children))
        if ctts_locations:
            target = self._rng.choice(ctts_locations)
            if len(target.data) >= 8:
                data = bytearray(target.data)
                # FullBox header: version (4) + entry_count (4)
                entry_count = struct.unpack_from(">I", data, 4)[0]
                if entry_count > 0:
                    # Corrupt entry_count to trigger overflow
                    if self._rng.randint(0, 1) == 0:
                        new_count = 0xFFFFFFFF
                        struct.pack_into(">I", data, 4, new_count)
                    else:
                        # Corrupt a sample_count field in first entry
                        if len(data) >= 12:
                            new_sample_count = self._rng.choice([0, 1, 0xFFFFFFFF, 0x7FFFFFFF])
                            struct.pack_into(">I", data, 8, new_sample_count)
                else:
                    # No entries yet - corrupt the entry_count itself
                    struct.pack_into(
                        ">I", data, 4, self._rng.choice([0xFFFFFFFF, 0x7FFFFFFF, 0x80000000])
                    )
                target.data = bytes(data)
        return boxes

    def _mutate_ctts_sample_count(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Mutate ctts sample_count fields."""
        ctts_locations = []
        for box in boxes:
            if box.box_type == b"ctts":
                ctts_locations.append(box)
            if box.children:
                ctts_locations.extend(_find_ctts(box.children))
        if ctts_locations:
            target = self._rng.choice(ctts_locations)
            if len(target.data) >= 12:
                data = bytearray(target.data)
                # First entry: sample_count at offset 8
                new_sample_count = self._rng.choice([0, 1, 0xFFFFFFFF, 0x7FFFFFFF])
                struct.pack_into(">I", data, 8, new_sample_count)
                target.data = bytes(data)
        return boxes

    def _mutate_ctts_sample_offset(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Mutate ctts sample_offset fields (s32)."""
        ctts_locations = []
        for box in boxes:
            if box.box_type == b"ctts":
                ctts_locations.append(box)
            if box.children:
                ctts_locations.extend(_find_ctts(box.children))
        if ctts_locations:
            target = self._rng.choice(ctts_locations)
            if len(target.data) >= 12:
                data = bytearray(target.data)
                # First entry: sample_offset at offset 12
                new_sample_offset = self._rng.choice(
                    [-1, -2, -100, 1, 2, 100, 0x7FFFFFFF, -0x80000000]
                )
                struct.pack_into(">i", data, 12, new_sample_offset)
                target.data = bytes(data)
        return boxes

    def _mutate_sample_table(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Set a sample-table count or entry field to an edge value.

        stts/stsc/stsz/stco/co64/elst counts size FFmpeg's index arrays
        and their entries index into them (mov.c `mov_build_index`): a
        count past the entries, `first_chunk = 0`, or a chunk offset past
        EOF are classic OOB and overflow triggers. Entries are indexed by
        the bytes present, never by the declared count.
        """
        tables = [b for b in _find_types(boxes, SAMPLE_TABLE_TYPES) if len(b.data) >= 8]
        if not tables:
            return boxes

        box = self._rng.choice(tables)
        layout = _table_layout(box)
        data = bytearray(box.data)
        present = (len(data) - layout.entries_off) // layout.entry_len

        # Count field: edge values plus "one past the entries present".
        if present <= 0 or self._rng.randint(0, 1) == 0:
            off, width = layout.count_off, COUNT_WIDTH
            value = self._rng.choice((*EDGE_U32, present + 1))
        else:
            entry = self._rng.randint(0, present - 1)
            idx = self._rng.randint(0, len(layout.field_widths) - 1)
            width = layout.field_widths[idx]
            off = layout.entries_off + entry * layout.entry_len + sum(layout.field_widths[:idx])
            value = self._rng.choice(EDGE_U64 if width == 8 else EDGE_U32)

        if off + width > len(data):
            return boxes
        data[off : off + width] = value.to_bytes(width, "big")
        box.data = bytes(data)
        return boxes

    def _mutate_nal_length(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Set an AVCC/HVCC NAL length prefix inside mdat to an edge value.

        FFmpeg's h264/hevc parsers split mdat samples on these prefixes
        (`ff_h2645_packet_split`, `is_nalff`); a length past the sample, a
        zero, or one byte off desyncs the split. Width comes from avcC/hvcC.
        """
        width = avcc_length_size(boxes)
        mdats = [
            (b, sites)
            for b in _find_types(boxes, frozenset({b"mdat"}))
            if (sites := _nal_length_sites(b.data, width))
        ]
        if not mdats:
            return boxes

        box, sites = self._rng.choice(mdats)
        off, actual = self._rng.choice(sites)
        mask = (1 << (8 * width)) - 1
        value = self._rng.choice((0, 1, actual - 1, actual + 1, *EDGE_U32[-1:])) & mask
        box.data = box.data[:off] + value.to_bytes(width, "big") + box.data[off + width :]
        return boxes

    def _swap_boxes(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Swap two sibling boxes.

        ``max_len`` is unused, as in ``_delete_box`` and ``_duplicate_box``:
        these three return a box list, and ``mutate`` caps after serializing
        it. The parameter is present because the dispatch list passes it.
        """
        if (pair := _swap_pair(len(boxes), self._rng)) is not None:
            i, j = pair
            boxes[i], boxes[j] = boxes[j], boxes[i]
        return boxes

    def _delete_box(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Delete a random box (not ftyp or root moov)."""
        deletable = [(i, b) for i, b in enumerate(boxes) if b.box_type not in (b"ftyp", b"moov")]
        if deletable:
            idx, _box = self._rng.choice(deletable)
            boxes.pop(idx)
        return boxes

    def _duplicate_box(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Clone a random box after the original."""
        if boxes:
            idx = self._rng.randint(0, len(boxes) - 1)
            orig = boxes[idx]
            dup = Box(
                box_type=orig.box_type,
                size_orig=orig.size_orig,
                data=orig.data[:],
                children=list(orig.children),
            )
            boxes.insert(idx + 1, dup)
        return boxes

    def _truncate_box(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Truncate a box's payload."""
        if boxes:
            target = self._rng.choice(boxes)
            if target.data and len(target.data) > 4:
                target.data = target.data[: self._rng.randint(1, max(2, len(target.data) // 2))]
        return boxes

    def _generate_random_isobmff(self, _boxes=None, max_len: int = 65536, rng=None) -> bytes:
        """Generate a minimal random ISOBMFF file."""
        # An int in the first slot is a max_len passed positionally. Without
        # this the cap lands in the vestigial placeholder and is dropped, and
        # the generator silently falls back to its own default -- the same
        # overload bmp/gzip/jpeg/zlib already handle and document.
        if isinstance(_boxes, int):
            max_len = _boxes
        self._rng = rng or self._rng

        # ftyp box
        major_brand = b"isom"
        minor_version = struct.pack(">I", self._rng.randint(0, 0xFFFFFFFF))
        compatible_brands = b"isommp42"
        ftyp_data = major_brand + minor_version + compatible_brands
        ftyp = Box(box_type=b"ftyp", size_orig=8 + len(ftyp_data), data=ftyp_data)

        # Minimal moov with one trak → mdia → hdlr
        hdlr_data = (
            struct.pack(">I", 0) + b"vide" + struct.pack(">III", 0, 0, 0) + b"VideoHandler\x00"
        )
        hdlr = Box(box_type=b"hdlr", size_orig=8 + len(hdlr_data), data=hdlr_data)

        mdhd_data = struct.pack(">I", 0) + struct.pack(">I", self._rng.randint(0, 0xFFFFFFFF))
        mdhd = Box(box_type=b"mdhd", size_orig=8 + len(mdhd_data), data=mdhd_data)

        mdia = Box(box_type=b"mdia", size_orig=0, children=[mdhd, hdlr])

        tkhd_data = struct.pack(">I", 0) + struct.pack(">I", 1)  # version=0, track_ID=1
        tkhd = Box(box_type=b"tkhd", size_orig=8 + len(tkhd_data), data=tkhd_data)

        trak = Box(box_type=b"trak", size_orig=0, children=[tkhd, mdia])

        mvhd_data = struct.pack(
            ">II", 0, self._rng.randint(0, 0xFFFFFFFF)
        )  # version+flags, timescale
        mvhd = Box(box_type=b"mvhd", size_orig=8 + len(mvhd_data), data=mvhd_data)

        moov = Box(box_type=b"moov", size_orig=0, children=[mvhd, trak])

        result = serialize_boxes([ftyp, moov])
        return result[:max_len]
