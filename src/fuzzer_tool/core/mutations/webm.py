"""Structure-aware WebM (Matroska/EBML) mutator.

EBML element structure:
  [id: vint, 1-4 bytes][size: vint, 1-8 bytes][payload]
  VINT: first byte's leading zero bits encode the length; the marker
        bit is stripped and remaining bits are the value.
        An all-ones payload (for its length) means "unknown size".

Container elements recurse into children; leaf elements keep raw data.
An untouched element round-trips byte-identically: the size vint is
re-emitted verbatim unless the payload length actually changed, in which
case it is re-encoded at minimal length (parents recompute bottom-up in
one recursive pass).
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.generic import _swap_pair

import random
import struct
from dataclasses import dataclass, field

# Matroska/WebM container element IDs (recurse into children)
CONTAINER_IDS = {
    0x1A45DFA3,  # EBML header
    0x18538067,  # Segment
    0x1549A966,  # Info
    0x1654AE6B,  # Tracks
    0xAE,  # TrackEntry
    0x1F43B675,  # Cluster
    0x114D9B74,  # SeekHead
    0x4DBB,  # Seek
    0x1043A770,  # Chapters
    0x45B9,  # EditionEntry
    0xB6,  # ChapterAtom
    0x1941A469,  # Attachments
    0x61A7,  # AttachedFile
    0x1C53BB6B,  # Cues
    0xBB,  # CuePoint
    0xB7,  # CueTrackPositions
    0x1254C367,  # Tags
    0x7373,  # Tag
    0x63C0,  # Target
    0x67C8,  # SimpleTag
}

# Known element IDs that can be swapped (id -> canonical vint raw bytes)
KNOWN_IDS = {
    0x1A45DFA3: b"\x1a\x45\xdf\xa3",  # EBML
    0x18538067: b"\x18\x53\x80\x67",  # Segment
    0x1549A966: b"\x15\x49\xa9\x66",  # Info
    0x1654AE6B: b"\x16\x54\xae\x6b",  # Tracks
    0xAE: b"\xae",  # TrackEntry
    0x1F43B675: b"\x1f\x43\xb6\x75",  # Cluster
    0x2AD7B1: b"\x2a\xd7\xb1",  # TimecodeScale
    0x4489: b"\x44\x89",  # Duration
    0x86: b"\x86",  # CodecID
    0x63A2: b"\x63\xa2",  # CodecPrivate
    0xE0: b"\xe0",  # Video
    0xE1: b"\xe1",  # Audio
    0xA3: b"\xa3",  # SimpleBlock
    0xE7: b"\xe7",  # Timecode
    0x83: b"\x83",  # TrackType
    0xD7: b"\xd7",  # TrackNumber
    0x4282: b"\x42\x82",  # DocType
    0x4286: b"\x42\x86",  # EBMLVersion
    0x42F7: b"\x42\xf7",  # EBMLReadVersion
    0x42F2: b"\x42\xf2",  # EBMLMaxIDLength
    0x42F3: b"\x42\xf3",  # EBMLMaxSizeLength
    0x4287: b"\x42\x87",  # DocTypeVersion
    0x4285: b"\x42\x85",  # DocTypeReadVersion
    0x73C5: b"\x73\xc5",  # TrackUID
    0xB0: b"\xb0",  # PixelWidth
    0xBA: b"\xba",  # PixelHeight
}

# Codec IDs for codec_swap (CodecID element data)
CODEC_IDS = [b"V_VP8", b"V_VP9", b"V_AV1", b"V_MPEG4/ISO/AVC", b"A_OPUS", b"A_VORBIS", b"A_FLAC"]


@dataclass
class Element:
    """A single EBML element."""

    elem_id: int
    id_raw: bytes  # raw id vint bytes (emitted verbatim)
    size_raw: bytes  # raw size vint bytes (emitted verbatim unless re-encoded)
    size_val: int  # declared size; -1 means unknown-size (all-ones vint)
    data: bytes = field(default_factory=bytes)  # leaf payload
    children: list[Element] = field(default_factory=list)  # container children
    size_reencoded: bool = False  # set by size_vint_rewrite: force re-encode


def _read_vint(data: bytes, pos: int, max_len: int = 8) -> tuple[int, bytes, int] | None:
    """Read an EBML vint at *pos*.

    Returns (value, raw_bytes, new_pos). The returned value is
    marker-inclusive (the ID convention: element IDs are stored with
    their marker bit set). value is -1 for the unknown-size all-ones
    pattern. Returns None if invalid/truncated.
    """
    if pos >= len(data):
        return None
    first = data[pos]
    if first == 0:
        return None
    mask = 0x80
    length = 1
    while mask and not (first & mask):
        mask >>= 1
        length += 1
    if length > max_len or pos + length > len(data):
        return None
    raw = data[pos : pos + length]
    val = int.from_bytes(raw, "big")
    if val == (1 << (8 * length)) - 1:
        val = -1
    return val, raw, pos + length


def _encode_size_vint(value: int) -> bytes:
    """Encode a size value as a minimal-length vint."""
    for length in range(1, 9):
        cap = (1 << (7 * length)) - 1
        if value < cap:
            return (value | (1 << (7 * length))).to_bytes(length, "big")
    # Value too large for 8 bytes — emit unknown-size marker
    return b"\xff" * 8


def _parse_element(data: bytes, pos: int) -> tuple[Element | None, int]:
    """Parse one element starting at *pos*. Returns (element, new_pos)."""
    v = _read_vint(data, pos, max_len=4)
    if v is None:
        return None, pos
    elem_id, id_raw, pos = v
    v2 = _read_vint(data, pos, max_len=8)
    if v2 is None:
        return None, pos
    size_raw_val, size_raw, pos = v2
    # Strip the marker bit: size values use the payload bits only
    length = len(size_raw)
    size_val = -1 if size_raw_val == -1 else (size_raw_val & ((1 << (7 + 8 * (length - 1))) - 1))

    if size_val == -1:
        body = data[pos:]
        body_end = len(data)
    else:
        if pos + size_val > len(data):
            return None, pos
        body = data[pos : pos + size_val]
        body_end = pos + size_val

    children: list[Element] = []
    if elem_id in CONTAINER_IDS:
        # Try to parse children; if the body does not fully parse, keep it
        # as a leaf so no bytes are lost on round-trip.
        cpos = 0
        parsed: list[Element] = []
        ok = True
        while cpos < len(body):
            c, new_pos = _parse_element(body, cpos)
            if c is None:
                ok = False
                break
            parsed.append(c)
            cpos = new_pos
        if ok and cpos == len(body):
            children = parsed
            body = b""

    return (
        Element(
            elem_id=elem_id,
            id_raw=id_raw,
            size_raw=size_raw,
            size_val=size_val,
            data=body,
            children=children,
        ),
        body_end,
    )


def parse_webm(data: bytes) -> list[Element] | None:
    """Parse a WebM file into a list of top-level elements.

    Returns None unless the top level starts with an EBML header
    element (1A45DFA3) followed by a Segment (18538067).
    """
    if len(data) < 4:
        return None
    elem, pos = _parse_element(data, 0)
    if elem is None or elem.elem_id != 0x1A45DFA3:
        return None
    seg, _pos2 = _parse_element(data, pos)
    if seg is None or seg.elem_id != 0x18538067:
        return None
    return [elem, seg]


def _serialize_element(el: Element) -> bytes:
    payload = b"".join(_serialize_element(c) for c in el.children) if el.children else el.data
    if el.size_reencoded:
        # Deliberate mismatch: write the stored size_val (or unknown marker)
        if el.size_val == -1:
            return el.id_raw + b"\xff" * 8 + payload
        return el.id_raw + _encode_size_vint(el.size_val) + payload
    if el.size_val == -1:
        # Unknown-size element: emit size vint verbatim
        return el.id_raw + el.size_raw + payload
    if len(payload) == el.size_val:
        return el.id_raw + el.size_raw + payload
    return el.id_raw + _encode_size_vint(len(payload)) + payload


def serialize_webm(elements: list[Element]) -> bytes:
    """Serialize elements back to bytes (bottom-up size recompute)."""
    return b"".join(_serialize_element(el) for el in elements)


def _find_all(elements: list[Element], elem_id: int) -> list[Element]:
    found: list[Element] = []
    for el in elements:
        if el.elem_id == elem_id:
            found.append(el)
        found.extend(_find_all(el.children, elem_id))
    return found


def _find_leaves(elements: list[Element]) -> list[Element]:
    leaves: list[Element] = []
    for el in elements:
        if el.children:
            leaves.extend(_find_leaves(el.children))
        else:
            leaves.append(el)
    return leaves


class WebmMutator:
    """Structure-aware WebM mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = rng or random
        elements = parse_webm(data)
        if elements is None:
            return self._generate_random_webm(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 11)
        mutators = [
            self._swap_id,
            self._rewrite_size_vint,
            self._swap_codec,
            self._nest_unnest,
            self._mutate_leaf_data,
            self._duplicate_element,
            self._delete_element,
            self._swap_elements,
            self._mutate_timecode_scale,
            self._mutate_duration,
            self._truncate_element,
            self._generate_random_webm,
        ]
        result = mutators[op](elements, max_len)
        if isinstance(result, list):
            return serialize_webm(result)[:max_len]
        return result[:max_len]

    def _swap_id(self, elements: list[Element], max_len: int) -> list[Element]:
        """Replace an element's ID with another known ID."""
        target = self._rng.choice(elements)
        new_id = self._rng.choice(list(KNOWN_IDS.keys()))
        target.elem_id = new_id
        target.id_raw = KNOWN_IDS[new_id]
        return elements

    def _rewrite_size_vint(self, elements: list[Element], max_len: int) -> list[Element]:
        """Rewrite a random element's declared size (shrink/grow/unknown)."""
        target = self._rng.choice(elements)
        payload_len = len(target.data) if not target.children else 0
        target.size_val = self._rng.choice(
            [0, 1, payload_len, max_len, 0xFFFFFFFF, -1, self._rng.randint(0, max(1, max_len))]
        )
        target.size_reencoded = True
        return elements

    def _swap_codec(self, elements: list[Element], max_len: int) -> list[Element]:
        """Replace a CodecID leaf's data with another codec ID."""
        codec = _find_all(elements, 0x86)
        if not codec:
            return elements
        target = self._rng.choice(codec)
        current = target.data
        options = [c for c in CODEC_IDS if c != current]
        target.data = self._rng.choice(options)
        return elements

    def _nest_unnest(self, elements: list[Element], max_len: int) -> list[Element]:
        """Nest a leaf into a new container, or unnest a container's children."""
        if self._rng.random() < 0.5:
            leaves = _find_leaves(elements)
            if not leaves:
                return elements
            target = self._rng.choice(leaves)
            # Find the parent list containing the target
            parent = _find_parent_list(elements, target)
            if parent is None:
                return elements
            idx = parent.index(target)
            wrapper = Element(
                elem_id=0x1549A966,  # Info
                id_raw=KNOWN_IDS[0x1549A966],
                size_raw=_encode_size_vint(0),
                size_val=0,
                children=[target],
            )
            parent[idx] = wrapper
        else:
            targets = [el for el in elements if el.children]
            if not targets:
                return elements
            target = self._rng.choice(targets)
            idx = elements.index(target)
            elements[idx : idx + 1] = target.children
        return elements

    def _mutate_leaf_data(self, elements: list[Element], max_len: int) -> list[Element]:
        leaves = _find_leaves(elements)
        if not leaves:
            return elements
        target = self._rng.choice(leaves)
        if target.data:
            raw = bytearray(target.data)
            mode = self._rng.randint(0, 2)
            if mode == 0:
                for _ in range(self._rng.randint(1, min(8, len(raw)))):
                    raw[self._rng.randint(0, len(raw) - 1)] = self._rng.randint(0, 0xFF)
            elif mode == 1:
                cut = self._rng.randint(1, max(1, len(raw) // 2))
                raw = raw[:cut]
            else:
                raw.extend(self._rng.randbytes(self._rng.randint(0, 8)))
            target.data = bytes(raw)
        return elements

    def _duplicate_element(self, elements: list[Element], max_len: int) -> list[Element]:
        if elements:
            idx = self._rng.randint(0, len(elements) - 1)
            orig = elements[idx]
            dup = Element(
                elem_id=orig.elem_id,
                id_raw=orig.id_raw,
                size_raw=orig.size_raw,
                size_val=orig.size_val,
                data=orig.data[:],
                children=list(orig.children),
                size_reencoded=orig.size_reencoded,
            )
            elements.insert(idx + 1, dup)
        return elements

    def _delete_element(self, elements: list[Element], max_len: int) -> list[Element]:
        if len(elements) > 1:
            elements.pop(self._rng.randint(0, len(elements) - 1))
        return elements

    def _swap_elements(self, elements: list[Element], max_len: int) -> list[Element]:
        if (pair := _swap_pair(len(elements), self._rng)) is not None:
            i, j = pair
            elements[i], elements[j] = elements[j], elements[i]
        return elements

    def _mutate_timecode_scale(self, elements: list[Element], max_len: int) -> list[Element]:
        scales = _find_all(elements, 0x2AD7B1)
        if not scales:
            return elements
        target = self._rng.choice(scales)
        value = self._rng.choice(
            [1, 1000, 10000, 1000000, 0xFFFFFFFF, self._rng.randint(0, 0xFFFFFFFF)]
        )
        target.data = _encode_size_vint(value)
        return elements

    def _mutate_duration(self, elements: list[Element], max_len: int) -> list[Element]:
        durations = _find_all(elements, 0x4489)
        if not durations:
            return elements
        target = self._rng.choice(durations)
        if len(target.data) >= 8:
            value = struct.unpack("<d", target.data[:8])[0]
            new_value = self._rng.choice(
                [0.0, 1.0, -1.0, float("inf"), float("nan"), value * 2.0, value / 2.0]
            )
            target.data = struct.pack("<d", new_value) + target.data[8:]
        else:
            target.data = struct.pack("<d", self._rng.random() * 100)
        return elements

    def _truncate_element(self, elements: list[Element], max_len: int) -> list[Element]:
        if elements:
            target = self._rng.choice(elements)
            if target.children:
                target.children = target.children[: self._rng.randint(0, len(target.children))]
            elif len(target.data) > 1:
                target.data = target.data[: self._rng.randint(0, len(target.data))]
        return elements

    def _generate_random_webm(self, _elements=None, max_len: int = 4096, rng=None) -> bytes:
        """Generate a minimal random WebM file."""
        # An int in the first slot is a max_len passed positionally. Without
        # this the cap lands in the vestigial placeholder and is dropped, and
        # the generator silently falls back to its own default -- the same
        # overload bmp/gzip/jpeg/zlib already handle and document.
        if isinstance(_elements, int):
            max_len = _elements
        self._rng = rng or self._rng

        def leaf(elem_id: int, data: bytes) -> Element:
            return Element(
                elem_id=elem_id,
                id_raw=KNOWN_IDS[elem_id],
                size_raw=_encode_size_vint(len(data)),
                size_val=len(data),
                data=data,
            )

        def container(elem_id: int, children: list[Element]) -> Element:
            payload = b"".join(_serialize_element(c) for c in children)
            return Element(
                elem_id=elem_id,
                id_raw=KNOWN_IDS[elem_id],
                size_raw=_encode_size_vint(len(payload)),
                size_val=len(payload),
                children=children,
            )

        ebml = container(
            0x1A45DFA3,
            [
                leaf(0x4286, b"\x01"),
                leaf(0x42F7, b"\x01"),
                leaf(0x42F2, b"\x04"),
                leaf(0x42F3, b"\x08"),
                leaf(0x4282, b"webm"),
                leaf(0x4287, b"\x02"),
                leaf(0x4285, b"\x02"),
            ],
        )

        info = container(
            0x1549A966,
            [
                leaf(0x2AD7B1, _encode_size_vint(1000000)),
                leaf(0x4489, struct.pack("<d", self._rng.random() * 10)),
            ],
        )

        video = container(0xE0, [leaf(0xB0, b"\x81"), leaf(0xBA, b"\x81")])
        track = container(
            0xAE,
            [
                leaf(0xD7, b"\x01"),
                leaf(0x73C5, b"\x01"),
                leaf(0x83, b"\x01"),
                leaf(0x86, b"V_VP8"),
                video,
            ],
        )
        tracks = container(0x1654AE6B, [track])

        cluster = container(0x1F43B675, [leaf(0xE7, b"\x81")])
        segment = container(0x18538067, [info, tracks, cluster])

        return serialize_webm([ebml, segment])[:max_len]


def _find_parent_list(elements: list[Element], target: Element) -> list[Element] | None:
    """Find the child list containing *target* (depth-first)."""
    for el in elements:
        if target in el.children:
            return el.children
        parent = _find_parent_list(el.children, target)
        if parent is not None:
            return parent
    return None
