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

import random
import struct
from dataclasses import dataclass, field

# Box types that contain sub-boxes
CONTAINER_TYPES = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"moof", b"traf",
    b"mvex", b"edts", b"dinf", b"udta", b"meta", b"skip", b"free",
    b"mdra", b"nmhd", b"smhd", b"vmhd", b"hmhd", b"sthd", b"meco",
    b"strk", b"ipro", b"sinf", b"schi", b"mfra", b"tfra", b"mfro",
    b"pdin",
}

# Box types whose payloads are full boxes (with version/flags prefix)
FULL_BOX_TYPES = {
    b"stsd", b"stts", b"stsc", b"stsz", b"stco", b"co64", b"stss",
    b"ctts", b"hdlr", b"mdhd", b"mvhd", b"tkhd", b"vmhd", b"smhd",
    b"nmhd", b"dref", b"elst", b"minf",
}

# Stream-type handler fourcc values
HANDLER_TYPES = [b"vide", b"soun", b"subt", b"meta", b"hint", b"auxv"]

# Codec fourcc values that can be swapped into stsd entries
STSD_CODECS = [
    b"avc1", b"mp4a", b"pgss", b"subp", b"tx3g", b"hvc1", b"hev1",
    b"hev2", b"mp4v", b"s263", b"av01", b"vp09", b"theo", b"vorb",
    b"Opus", b"FLAC", b"ac-3", b"eac3", b"dtsc", b"dtsh", b"dtse",
    b"dtsl", b"mlpa", b"lpcm", b"sowt", b"samr", b"sawb",
]


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


def _mutate_hdlr_handler_type(box: Box, rng: random.Random) -> None:
    """Replace the handler_type field in a hdlr box payload.

    hdlr payload: [predefined: u32][handler_type: 4 bytes][reserved: 3*u32][name: str]
    Handler type at offset 4-7.
    """
    data = bytearray(box.data)
    if len(data) >= 8:
        current = bytes(data[4:8])
        new_type = rng.choice([t for t in HANDLER_TYPES if t != current] + [b"\xff\xff\xff\xff", b"\x00\x00\x00\x00"])
        data[4:8] = new_type
        box.data = bytes(data)


def _mutate_stsd_codec(box: Box, rng: random.Random) -> None:
    """Corrupt the first sample entry's codec fourcc in an stsd box.

    stsd payload: [version: u8][flags: u24][entry_count: u32][entries...]
    Each sample entry: [entry_size: u32][codec_fourcc: 4 bytes][...]

    Codec fourcc at offset 8+4 = 12 (after stsd header).
    """
    data = bytearray(box.data)
    if len(data) >= 16:
        current = bytes(data[12:16])
        new_codec = rng.choice([c for c in STSD_CODECS if c != current] + [b"\xff\xff\xff\xff", b"\x00\x00\x00\x00"])
        data[12:16] = new_codec
        box.data = bytes(data)


class IsobmffMutator:
    """Structure-aware ISO-BMFF box mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        self._rng = rng or random
        boxes = parse_boxes(data)
        if boxes is None or not boxes:
            return self._generate_random_isobmff(max_len, rng=self._rng)

        op = self._rng.randint(0, 9)
        mutators = [
            self._mutate_box_type,
            self._mutate_box_size,
            self._mutate_ftyp,
            self._mutate_hdlr,
            self._mutate_stsd,
            self._swap_boxes,
            self._delete_box,
            self._duplicate_box,
            self._truncate_box,
            self._generate_random_isobmff,
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

        target.size_orig = self._rng.choice([0, 1, 8, max_len, 0xFFFFFFFF, self._rng.randint(0, max_len)])
        # Recompute/truncate/pad payload from new size
        raw = target.data
        payload_len = max(0, target.size_orig - 8)
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
                    pos = 0 if self._rng.random() < 0.5 else self._rng.randint(8, max(8, len(data) - 4))
                    data[pos : pos + 4] = self._rng.choice([b"xxxx", b"\xff\xff\xff\xff", b"\x00\x00\x00\x00", b"????"])
                box.data = bytes(data)
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

    def _swap_boxes(self, boxes: list[Box], max_len: int) -> list[Box]:
        """Swap two sibling boxes."""
        if len(boxes) >= 2:
            i, j = self._rng.sample(list(range(len(boxes))), 2)
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
            dup = Box(box_type=orig.box_type, size_orig=orig.size_orig, data=orig.data[:], children=list(orig.children))
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
        self._rng = rng or self._rng

        # ftyp box
        major_brand = b"isom"
        minor_version = struct.pack(">I", self._rng.randint(0, 0xFFFFFFFF))
        compatible_brands = b"isommp42"
        ftyp_data = major_brand + minor_version + compatible_brands
        ftyp = Box(box_type=b"ftyp", size_orig=8 + len(ftyp_data), data=ftyp_data)

        # Minimal moov with one trak → mdia → hdlr
        hdlr_data = struct.pack(">I", 0) + b"vide" + struct.pack(">III", 0, 0, 0) + b"VideoHandler\x00"
        hdlr = Box(box_type=b"hdlr", size_orig=8 + len(hdlr_data), data=hdlr_data)

        mdhd_data = struct.pack(">I", 0) + struct.pack(">I", self._rng.randint(0, 0xFFFFFFFF))
        mdhd = Box(box_type=b"mdhd", size_orig=8 + len(mdhd_data), data=mdhd_data)

        mdia = Box(box_type=b"mdia", size_orig=0, children=[mdhd, hdlr])

        tkhd_data = struct.pack(">I", 0) + struct.pack(">I", 1)  # version=0, track_ID=1
        tkhd = Box(box_type=b"tkhd", size_orig=8 + len(tkhd_data), data=tkhd_data)

        trak = Box(box_type=b"trak", size_orig=0, children=[tkhd, mdia])

        mvhd_data = struct.pack(">II", 0, self._rng.randint(0, 0xFFFFFFFF))  # version+flags, timescale
        mvhd = Box(box_type=b"mvhd", size_orig=8 + len(mvhd_data), data=mvhd_data)

        moov = Box(box_type=b"moov", size_orig=0, children=[mvhd, trak])

        result = serialize_boxes([ftyp, moov])
        return result[:max_len]
