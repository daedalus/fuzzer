"""Structure-aware AVIF (HEIF/ISO-BMFF still image) mutator.

AVIF file layout:
  [ftyp]  major_brand="avif"/"avis", compatible_brands include avif/mif1/miaf
  [meta]  FullBox(version, flags) containing, as plain sub-boxes:
            hdlr  - handler_type "pict"
            pitm  - FullBox: primary_item_ID
            iloc  - FullBox: item location table (item_ID -> data extents)
            iinf  - FullBox: entry_count, then "infe" FullBox entries
                    (item_ID, item_protection_index, item_type e.g. "av01")
            iref  - item reference box (auxl/cdsc/... entries)
            iprp  - item properties:
                      ipco - property container: ispe (width,height), av1C
                             (AV1 codec config), pixi (bits per channel),
                             colr, auxC, ...
                      ipma - FullBox: item_ID -> [(essential, property_index)]
  [mdat]  - raw AV1 OBU payload referenced by iloc extents

``meta`` is a FullBox: its payload starts with a 4-byte version/flags field
*before* its children boxes. The generic ``isobmff`` mutator does not model
this (it treats every container's payload as a flat run of child boxes), so
reusing it here would parse ``meta``'s first child four bytes short. This
module keeps ``isobmff.Box``/``serialize_boxes`` for the plain (non-full)
container levels -- top level, iprp, ipco -- and handles the FullBox header
of meta/pitm/iloc/iinf/infe/ipma itself.

Field-level mutations of iloc extents, infe item types, ipma associations,
ispe dimensions, and av1C config bytes are applied by locating each field's
byte offset within the relevant box's raw payload and overwriting in place,
the same convention ``bmp.py``/``webp.py`` use for fixed-layout sub-fields --
full re-parsing round-trip fidelity is not needed for a corruption probe.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.generic import _swap_pair

import random
import struct
from dataclasses import dataclass, field
from typing import Any

from fuzzer_tool.core.mutations.isobmff import Box, serialize_boxes

# ftyp brands that mark a file as (an image derived from) AVIF
AVIF_BRANDS = (b"avif", b"avis")
COMPATIBLE_BRAND_POOL = [b"avif", b"avis", b"mif1", b"miaf", b"MA1B", b"MA1A", b"msf1"]

# infe item types that can be swapped in (av01 is the AV1 image item; the
# rest probe type-confusion in the item-info / item-reference walkers)
ITEM_TYPES = [b"av01", b"grid", b"Exif", b"mime", b"hvc1", b"jpeg", b"uri "]

WEIRD_FOURCCS = [b"\x00\x00\x00\x00", b"\xff\xff\xff\xff", b"    ", b"xxxx", b"????"]

# Interesting values for u16/u32 numeric fields (item IDs, dimensions, ...)
INT_VALUES = [0, 1, 2, 0x7FFF, 0xFFFF, 0xFFFFFFFF]

CONTAINER_TYPES = {b"iprp", b"ipco"}


def _parse_boxes(data: bytes, start: int, end: int) -> list[Box] | None:
    """Parse a flat run of plain (non-FullBox) boxes in ``data[start:end]``.

    Recurses into ``CONTAINER_TYPES`` (``iprp``, ``ipco``) only -- every
    other box type (including FullBoxes like ``iloc``/``iinf``/``ipma``) is
    kept as an opaque leaf and interpreted on demand by the field-offset
    helpers below.
    """
    boxes: list[Box] = []
    pos = start
    while pos + 8 <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        if size < 8 or pos + size > end:
            break
        box_type = data[pos + 4 : pos + 8]
        payload_start = pos + 8
        payload_end = pos + size
        children: list[Box] = []
        if box_type in CONTAINER_TYPES:
            sub = _parse_boxes(data, payload_start, payload_end)
            if sub:
                children = sub
        payload = data[payload_start:payload_end] if not children else b""
        boxes.append(Box(box_type=box_type, size_orig=size, data=payload, children=children))
        pos = payload_end
    return boxes if boxes else None


@dataclass
class AvifDoc:
    """A parsed AVIF file: the top-level box list plus meta's contents."""

    top_boxes: list[Box]
    meta_idx: int
    meta_version_flags: bytes
    meta_children: list[Box]
    mdat_idx: int | None = field(default=None)


def _find_ftyp_brands(data: bytes) -> tuple[bytes, bytes] | None:
    """Return (major_brand, compatible_brands_blob) from a leading ftyp box."""
    if len(data) < 16 or data[4:8] != b"ftyp":
        return None
    size = struct.unpack_from(">I", data, 0)[0]
    if size < 16 or size > len(data):
        return None
    major_brand = data[8:12]
    compatible = data[16:size]
    return major_brand, compatible


def sniff_avif(data: bytes) -> bool:
    """Cheap format check: leading ftyp box declaring an AVIF-family brand."""
    brands = _find_ftyp_brands(data)
    if brands is None:
        return False
    major, compatible = brands
    if major in AVIF_BRANDS:
        return True
    return any(compatible[i : i + 4] in AVIF_BRANDS for i in range(0, len(compatible) - 3, 4))


def parse_avif(data: bytes) -> AvifDoc | None:
    """Parse an AVIF file into an :class:`AvifDoc`, or None if unparseable."""
    if not sniff_avif(data):
        return None

    top = _parse_boxes(data, 0, len(data))
    if not top:
        return None

    meta_idx = next((i for i, b in enumerate(top) if b.box_type == b"meta"), None)
    if meta_idx is None:
        return None
    meta_box = top[meta_idx]
    if len(meta_box.data) < 4:
        return None

    version_flags = meta_box.data[:4]
    children = _parse_boxes(meta_box.data, 4, len(meta_box.data))
    if children is None:
        return None

    mdat_idx = next((i for i, b in enumerate(top) if b.box_type == b"mdat"), None)
    return AvifDoc(
        top_boxes=top,
        meta_idx=meta_idx,
        meta_version_flags=version_flags,
        meta_children=children,
        mdat_idx=mdat_idx,
    )


def serialize_avif(doc: AvifDoc) -> bytes:
    """Serialize an :class:`AvifDoc` back to bytes.

    Every box's own ``size_orig`` is written verbatim (never recomputed) --
    matching the ``isobmff``/``webp`` convention that a declared-size-vs-
    payload mismatch is a deliberate corruption probe, not a bug. Only the
    synthetic ``meta`` box header, which this module builds fresh from
    ``meta_children`` on every call, gets a recomputed size.
    """
    out = bytearray()
    for i, box in enumerate(doc.top_boxes):
        if i == doc.meta_idx:
            payload = doc.meta_version_flags + serialize_boxes(doc.meta_children)
            out += struct.pack(">I", 8 + len(payload)) + b"meta" + payload
        else:
            payload = serialize_boxes(box.children) if box.children else bytes(box.data)
            out += struct.pack(">I", box.size_orig & 0xFFFFFFFF) + box.box_type + payload
    return bytes(out)


def _find_leaf(boxes: list[Box], *fourccs: bytes) -> list[tuple[list[Box], int]]:
    """Find every leaf box of one of *fourccs*, anywhere in the tree."""
    found: list[tuple[list[Box], int]] = []
    for i, box in enumerate(boxes):
        if box.box_type in fourccs:
            found.append((boxes, i))
        if box.children:
            found.extend(_find_leaf(box.children, *fourccs))
    return found


# ── FullBox field-offset helpers ────────────────────────────────────────
#
# Each of these locates a fixed-layout field inside a FullBox payload and
# returns None when the payload is too short / malformed to touch safely.


def _iloc_item_offsets(data: bytes) -> list[dict[str, Any]]:
    """Byte offsets of each iloc item's id/extent fields.

    ``version`` selects item_ID (16 vs 32 bit) and extent index (present
    from version 1) widths; the nibble byte at offset 4 selects
    offset/length/base_offset field widths. Returns [] on anything that
    does not parse as a plausible iloc payload -- callers must treat that
    as "nothing to mutate", not an error.
    """
    if len(data) < 6:
        return []
    version = data[0]
    offset_size = data[4] >> 4
    length_size = data[4] & 0x0F
    base_offset_size = data[5] >> 4
    index_size = data[5] & 0x0F if version in (1, 2) else 0

    pos = 6
    if version < 2:
        if pos + 2 > len(data):
            return []
        item_count = struct.unpack_from(">H", data, pos)[0]
        pos += 2
    else:
        if pos + 4 > len(data):
            return []
        item_count = struct.unpack_from(">I", data, pos)[0]
        pos += 4

    items: list[dict[str, Any]] = []
    for _ in range(min(item_count, 4096)):
        item_id_off = pos
        item_id_size = 2 if version < 2 else 4
        pos += item_id_size
        if version in (1, 2):
            pos += 2  # construction_method (2 reserved bits + 14 + method, kept opaque)
        pos += 2  # data_reference_index
        pos += base_offset_size
        if pos + 2 > len(data):
            return items
        extent_count = struct.unpack_from(">H", data, pos)[0]
        pos += 2
        extents = []
        for _ in range(min(extent_count, 4096)):
            if version in (1, 2):
                pos += index_size
            offset_off = pos
            pos += offset_size
            length_off = pos
            pos += length_size
            if pos > len(data):
                return items
            extents.append(
                {
                    "offset_off": offset_off,
                    "offset_size": offset_size,
                    "length_off": length_off,
                    "length_size": length_size,
                }
            )
        items.append({"item_id_off": item_id_off, "item_id_size": item_id_size, "extents": extents})
    return items


def _write_be(data: bytearray, off: int, size: int, value: int) -> None:
    if size not in (1, 2, 3, 4, 8):
        return
    value &= (1 << (size * 8)) - 1
    if size == 3:
        data[off : off + 3] = struct.pack(">I", value)[1:]
    else:
        fmt = {1: ">B", 2: ">H", 4: ">I", 8: ">Q"}[size]
        struct.pack_into(fmt, data, off, value)


def _infe_entries(iinf_data: bytes) -> list[tuple[int, int]]:
    """Return (start, end) offsets of each child box inside an iinf payload."""
    if len(iinf_data) < 6:
        return []
    version = iinf_data[0]
    pos = 6 if version == 0 else 8
    entries = []
    while pos + 8 <= len(iinf_data):
        size = struct.unpack_from(">I", iinf_data, pos)[0]
        if size < 8 or pos + size > len(iinf_data):
            break
        entries.append((pos, pos + size))
        pos += size
    return entries


class AvifMutator:
    """Structure-aware AVIF mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        self._rng = rng or random
        doc = parse_avif(data)
        if doc is None:
            return self._generate_random_avif(max_len=max_len, rng=self._rng)

        mutators = [
            self._mutate_ftyp_brand,
            self._mutate_ispe,
            self._mutate_av1c,
            self._mutate_pixi,
            self._mutate_pitm,
            self._mutate_iloc_extent,
            self._mutate_infe_item_type,
            self._mutate_iinf_entry_count,
            self._mutate_ipma,
            self._mutate_hdlr_type,
            self._mutate_meta_version_flags,
            self._mutate_mdat_obu,
            self._swap_meta_children,
            self._delete_meta_child,
            self._duplicate_meta_child,
            self._mutate_box_size,
        ]
        op = self._rng.randint(0, len(mutators) - 1)
        result = mutators[op](doc, max_len)
        return serialize_avif(result)[:max_len]

    # ── mutations ────────────────────────────────────────────────────

    def _mutate_ftyp_brand(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        ftyp = next((b for b in doc.top_boxes if b.box_type == b"ftyp"), None)
        if ftyp is None or len(ftyp.data) < 8:
            return doc
        raw = bytearray(ftyp.data)
        if self._rng.random() < 0.5:
            raw[0:4] = self._rng.choice([b for b in AVIF_BRANDS] + WEIRD_FOURCCS)
        else:
            if len(raw) > 8:
                stop = max(9, len(raw) - 3)
                width = max(1, (stop - 8 + 3) // 4)  # ceil((stop-8)/4) steps of 4, starting at 8
                pos = 8 + self._rng.randrange(width) * 4
            else:
                pos = 8
            if pos + 4 <= len(raw):
                raw[pos : pos + 4] = self._rng.choice(COMPATIBLE_BRAND_POOL + WEIRD_FOURCCS)
            else:
                raw.extend(self._rng.choice(COMPATIBLE_BRAND_POOL))
        ftyp.data = bytes(raw)
        return doc

    def _mutate_ispe(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Corrupt ispe's declared width/height (FullBox: u32 width, u32 height)."""
        found = _find_leaf(doc.meta_children, b"ispe")
        if not found:
            return doc
        parent, idx = self._rng.choice(found)
        box = parent[idx]
        raw = bytearray(box.data)
        if len(raw) < 12:
            return doc
        off = self._rng.choice([4, 8])
        _write_be(raw, off, 4, self._rng.choice(INT_VALUES + [self._rng.randint(0, 0xFFFFFFFF)]))
        box.data = bytes(raw)
        return doc

    def _mutate_av1c(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Corrupt AV1CodecConfigurationRecord bytes (av1C box payload).

        Layout: marker/version(1) | seq_profile(3b)+seq_level_idx_0(5b)(1) |
        seq_tier_0/high_bitdepth/twelve_bit/monochrome/chroma_subsampling(1) |
        chroma_sample_position(2b)+reserved(6b)(1) | initial_presentation...
        """
        found = _find_leaf(doc.meta_children, b"av1C")
        if not found:
            return doc
        parent, idx = self._rng.choice(found)
        box = parent[idx]
        raw = bytearray(box.data)
        if len(raw) < 4:
            box.data = bytes([0x81, self._rng.randint(0, 0xFF), self._rng.randint(0, 0xFF), 0])
            return doc
        pos = self._rng.randint(0, min(3, len(raw) - 1))
        if self._rng.random() < 0.5:
            raw[pos] ^= 1 << self._rng.randint(0, 7)
        else:
            raw[pos] = self._rng.choice([0x00, 0x7F, 0x80, 0xFF])
        box.data = bytes(raw)
        return doc

    def _mutate_pixi(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Corrupt pixi's channel count or a per-channel bit depth."""
        found = _find_leaf(doc.meta_children, b"pixi")
        if not found:
            return doc
        parent, idx = self._rng.choice(found)
        box = parent[idx]
        raw = bytearray(box.data)
        if len(raw) < 5:
            return doc
        if self._rng.random() < 0.5:
            raw[4] = self._rng.choice([0, 1, 2, 3, 8, 255])  # num_channels
        else:
            ch_off = 5 + self._rng.randint(0, max(0, len(raw) - 6))
            if ch_off < len(raw):
                raw[ch_off] = self._rng.choice([0, 1, 8, 16, 32, 255])
        box.data = bytes(raw)
        return doc

    def _mutate_pitm(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Corrupt the primary_item_ID (u16 if version==0, else u32)."""
        found = _find_leaf(doc.meta_children, b"pitm")
        if not found:
            return doc
        _parent, idx = found[0]
        box = _parent[idx]
        raw = bytearray(box.data)
        if len(raw) < 4:
            return doc
        version = raw[0]
        size = 2 if version == 0 else 4
        if len(raw) < 4 + size:
            return doc
        _write_be(raw, 4, size, self._rng.choice(INT_VALUES))
        box.data = bytes(raw)
        return doc

    def _mutate_iloc_extent(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Corrupt an item's extent offset/length, or its item_ID."""
        found = _find_leaf(doc.meta_children, b"iloc")
        if not found:
            return doc
        _parent, idx = found[0]
        box = _parent[idx]
        items = _iloc_item_offsets(box.data)
        if not items:
            return doc
        raw = bytearray(box.data)
        item = self._rng.choice(items)
        if item["extents"] and self._rng.random() < 0.7:
            extent = self._rng.choice(item["extents"])
            field_name = self._rng.choice(["offset", "length"])
            off, size = extent[f"{field_name}_off"], extent[f"{field_name}_size"]
            if size:
                value = self._rng.choice(
                    [0, 1, len(raw), max_len, self._rng.randint(0, max_len * 4)]
                )
                _write_be(raw, off, size, value)
        elif item["item_id_size"]:
            _write_be(raw, item["item_id_off"], item["item_id_size"], self._rng.choice(INT_VALUES))
        box.data = bytes(raw)
        return doc

    def _mutate_infe_item_type(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Swap an infe entry's item_type fourcc (version 2/3 only)."""
        found = _find_leaf(doc.meta_children, b"iinf")
        if not found:
            return doc
        _parent, idx = found[0]
        box = _parent[idx]
        entries = _infe_entries(box.data)
        if not entries:
            return doc
        raw = bytearray(box.data)
        start, end = self._rng.choice(entries)
        # infe box: [size:4][b"infe"][version+flags:4][item_ID][protection_idx:2][item_type:4]
        if end - start < 20:
            return doc
        version = raw[start + 8]
        id_size = 2 if version < 3 else 4
        type_off = start + 8 + 4 + id_size + 2
        if type_off + 4 <= end:
            current = bytes(raw[type_off : type_off + 4])
            raw[type_off : type_off + 4] = self._rng.choice(
                [t for t in ITEM_TYPES if t != current] + WEIRD_FOURCCS
            )
        box.data = bytes(raw)
        return doc

    def _mutate_iinf_entry_count(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Corrupt iinf's declared entry_count without adding/removing infe
        entries -- a declared-count-vs-actual-entries mismatch, the same
        probe class as the box-size mutator one level up."""
        found = _find_leaf(doc.meta_children, b"iinf")
        if not found:
            return doc
        _parent, idx = found[0]
        box = _parent[idx]
        raw = bytearray(box.data)
        # iinf FullBox: [version+flags:4][entry_count: u16 if version==0
        # else u32], then the infe entries themselves.
        if len(raw) < 6:
            return doc
        version = raw[0]
        size = 2 if version == 0 else 4
        if len(raw) < 4 + size:
            return doc
        _write_be(raw, 4, size, self._rng.choice(INT_VALUES))
        box.data = bytes(raw)
        return doc

    def _mutate_hdlr_type(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Swap hdlr's handler_type away from "pict".

        An AVIF ``meta`` box is only an image collection because its handler
        says so; every other handler sends the reader down a different
        item-walker, which is exactly the type-confusion path worth probing.
        """
        found = _find_leaf(doc.meta_children, b"hdlr")
        if not found:
            return doc
        _parent, idx = found[0]
        box = _parent[idx]
        raw = bytearray(box.data)
        # hdlr FullBox: [version+flags:4][pre_defined:4][handler_type:4]
        if len(raw) < 12:
            return doc
        raw[8:12] = self._rng.choice([b"pict", b"vide", b"soun", b"meta", b"auxv"] + WEIRD_FOURCCS)
        box.data = bytes(raw)
        return doc

    def _mutate_meta_version_flags(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Corrupt meta's own FullBox version/flags word.

        A non-zero version is undefined for ``meta``; readers that check it
        bail early, and readers that do not go on to parse the following
        bytes as a child box header regardless.
        """
        raw = bytearray(doc.meta_version_flags)
        if len(raw) < 4:
            return doc
        if self._rng.random() < 0.5:
            raw[0] = self._rng.choice([0, 1, 2, 0x7F, 0xFF])
        else:
            _write_be(raw, 1, 3, self._rng.choice(INT_VALUES))
        doc.meta_version_flags = bytes(raw)
        return doc

    def _mutate_mdat_obu(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Mutate the AV1 payload in mdat at OBU granularity.

        The mdat payload is the only part of an AVIF file the AV1 decoder
        itself parses -- every other mutation here stops at the container
        walker. An OBU is ``[header byte][leb128 size (if has_size_field)]
        [payload]``; the header byte carries forbidden(1) | type(4) |
        extension_flag(1) | has_size_field(1) | reserved(1).
        """
        if doc.mdat_idx is None:
            return doc
        box = doc.top_boxes[doc.mdat_idx]
        raw = bytearray(box.data)
        if not raw:
            return doc
        choice = self._rng.randint(0, 2)
        if choice == 0:
            # Retype the leading OBU (1=sequence header, 6=frame, 15=padding).
            obu_type = self._rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 15, 0])
            raw[0] = (raw[0] & 0x81) | ((obu_type & 0x0F) << 3)
        elif choice == 1 and len(raw) >= 2:
            # Corrupt the leb128 size field: an oversized or continuation-
            # heavy encoding is the classic "size runs past the buffer" probe.
            raw[1] = self._rng.choice([0x00, 0x01, 0x7F, 0x80, 0xFF])
        else:
            pos = self._rng.randint(0, len(raw) - 1)
            raw[pos] ^= 1 << self._rng.randint(0, 7)
        box.data = bytes(raw)
        return doc

    def _mutate_ipma(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Flip a byte in ipma's association table (item_ID/index/essential bit)."""
        found = _find_leaf(doc.meta_children, b"ipma")
        if not found:
            return doc
        _parent, idx = found[0]
        box = _parent[idx]
        raw = bytearray(box.data)
        if len(raw) < 9:
            return doc
        pos = self._rng.randint(8, len(raw) - 1)
        raw[pos] = self._rng.choice([0x00, 0xFF, raw[pos] ^ (1 << self._rng.randint(0, 7))])
        box.data = bytes(raw)
        return doc

    def _mutate_box_size(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        """Corrupt a random meta-child box's declared size (mismatch probe)."""
        if not doc.meta_children:
            return doc
        target = self._rng.choice(doc.meta_children)
        target.size_orig = self._rng.choice(
            [0, 1, 8, max_len, 0xFFFFFFFF, self._rng.randint(0, max_len)]
        )
        return doc

    def _swap_meta_children(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        if (pair := _swap_pair(len(doc.meta_children), self._rng)) is not None:
            i, j = pair
            doc.meta_children[i], doc.meta_children[j] = (
                doc.meta_children[j],
                doc.meta_children[i],
            )
        return doc

    def _delete_meta_child(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        if len(doc.meta_children) > 1:
            doc.meta_children.pop(self._rng.randint(0, len(doc.meta_children) - 1))
        return doc

    def _duplicate_meta_child(self, doc: AvifDoc, max_len: int) -> AvifDoc:
        if doc.meta_children:
            i = self._rng.randint(0, len(doc.meta_children) - 1)
            orig = doc.meta_children[i]
            dup = Box(
                box_type=orig.box_type,
                size_orig=orig.size_orig,
                data=orig.data[:],
                children=list(orig.children),
            )
            doc.meta_children.insert(i + 1, dup)
        return doc

    # ── generation ───────────────────────────────────────────────────

    def _generate_random_avif(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate a minimal AVIF file: ftyp + meta(hdlr/pitm/iloc/iinf/iprp) + mdat."""
        self._rng = rng or self._rng or random
        r = self._rng

        ftyp_payload = b"avif" + struct.pack(">I", 0) + b"avifmif1miaf"
        ftyp = Box(b"ftyp", 8 + len(ftyp_payload), ftyp_payload)

        av1_payload = bytes([r.randint(0, 255) for _ in range(r.randint(4, 32))])
        mdat = Box(b"mdat", 8 + len(av1_payload), av1_payload)

        hdlr_data = (
            struct.pack(">I", 0)
            + struct.pack(">I", 0)
            + b"pict"
            + struct.pack(">III", 0, 0, 0)
            + b"\x00"
        )
        hdlr = Box(b"hdlr", 8 + len(hdlr_data), hdlr_data)

        pitm_data = struct.pack(">I", 0) + struct.pack(">H", 1)
        pitm = Box(b"pitm", 8 + len(pitm_data), pitm_data)

        # iloc v0: 1 item, construction_method-free, offset/length sizes = 4
        iloc_data = bytearray()
        iloc_data += struct.pack(">I", 0)  # version 0, flags 0
        iloc_data += bytes([0x44, 0x00])  # offset_size=4, length_size=4 | base_offset_size=0
        iloc_data += struct.pack(">H", 1)  # item_count
        iloc_data += struct.pack(">H", 1)  # item_ID
        iloc_data += struct.pack(">H", 0)  # data_reference_index
        iloc_data += struct.pack(">H", 1)  # extent_count
        iloc_data += struct.pack(">I", 0)  # extent_offset (into mdat payload)
        iloc_data += struct.pack(">I", len(av1_payload))  # extent_length
        iloc = Box(b"iloc", 8 + len(iloc_data), bytes(iloc_data))

        infe_data = (
            struct.pack(">I", 0x02000000)  # version 2, flags 0
            + struct.pack(">H", 1)  # item_ID
            + struct.pack(">H", 0)  # item_protection_index
            + b"av01"  # item_type
            + b"\x00"  # item_name
        )
        infe = Box(b"infe", 8 + len(infe_data), infe_data)
        # iinf is a FullBox *and* a container: its payload is
        # version/flags + entry_count followed by the infe boxes. ``Box``
        # cannot hold both a payload prefix and children (serialize_boxes
        # drops ``data`` whenever ``children`` is non-empty), and
        # ``_parse_boxes`` keeps iinf as an opaque leaf on the way back in,
        # so build it as a leaf whose data already contains the serialized
        # entries -- which is exactly what ``_infe_entries`` walks.
        iinf_data = struct.pack(">I", 0) + struct.pack(">H", 1) + serialize_boxes([infe])
        iinf = Box(b"iinf", 8 + len(iinf_data), iinf_data)

        ispe_data = struct.pack(">I", 0) + struct.pack(
            ">II", r.randint(1, 4096), r.randint(1, 4096)
        )
        ispe = Box(b"ispe", 8 + len(ispe_data), ispe_data)

        av1c_data = bytes([0x81, r.randint(0, 0x1F), 0x00, 0x00])
        av1c = Box(b"av1C", 8 + len(av1c_data), av1c_data)

        # Container sizes must be real: serialize_boxes writes size_orig
        # verbatim, and a declared size < 8 makes _parse_boxes stop at that
        # box, silently truncating everything after it on the way back in.
        ipco = Box(b"ipco", 8 + ispe.size_orig + av1c.size_orig, children=[ispe, av1c])

        ipma_data = (
            struct.pack(">I", 0)  # version 0, flags 0
            + struct.pack(">I", 1)  # entry_count
            + struct.pack(">H", 1)  # item_ID
            + bytes([2])  # association_count
            + bytes([0x81, 0x82])  # essential+index pairs (1-based property_index)
        )
        ipma = Box(b"ipma", 8 + len(ipma_data), ipma_data)

        iprp = Box(b"iprp", 8 + ipco.size_orig + ipma.size_orig, children=[ipco, ipma])

        meta_children = [hdlr, pitm, iloc, iinf, iprp]
        meta_version_flags = struct.pack(">I", 0)
        doc = AvifDoc(
            top_boxes=[ftyp, Box(b"meta", 0, b""), mdat],
            meta_idx=1,
            meta_version_flags=meta_version_flags,
            meta_children=meta_children,
            mdat_idx=2,
        )
        return serialize_avif(doc)[:max_len]
