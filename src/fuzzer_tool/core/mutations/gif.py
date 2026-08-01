"""Structure-aware GIF mutator.

GIF structure:
  [header: 6 bytes]       — "GIF87a" or "GIF89a"
  [LSD: 7 bytes]          — u16le width, u16le height, packed flags,
                            bg color index, pixel aspect ratio
  [GCT: 3*2^((flags&7)+1) bytes] — global color table (if flags & 0x80)
  [blocks...]
    — 0x2C image descriptor: 9 bytes (u16le left/top/width/height + packed)
      + LZW min code size (1 byte) + sub-blocks + 0x00 terminator
    — 0x21 extension: label byte + sub-blocks + 0x00 terminator
    — 0x3B trailer
  Sub-block: [len: u8][len data bytes]

Serialization concatenates node.raw verbatim, so an untouched parse
round-trips byte-identically. Image/extension nodes rebuild their raw
bytes from their parsed fields only when a mutation changes them.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field

MAGICS = (b"GIF87a", b"GIF89a")

# Extension labels that are common in the wild
EXTENSION_LABELS = [0x01, 0xF9, 0xFE, 0xFF, 0x00, 0x21]

# Interesting width/height values for dimension mutation
DIM_VALUES = [0, 1, 2, 3, 0xFF, 0x100, 0xFFFF, 0x7FFF]

# Interesting packed-flags values for the logical screen descriptor
LSD_FLAG_VALUES = [0x00, 0x01, 0x07, 0x08, 0x10, 0x70, 0x80, 0xF0, 0xFF]

# Interesting LZW min-code-size values
MINCODE_VALUES = [0, 1, 2, 7, 8, 9, 12, 13, 0xFF]

# Interesting sub-block lengths
SUBBLOCK_LEN_VALUES = [0, 1, 2, 0x40, 0x80, 0xFE, 0xFF]


@dataclass
class GifNode:
    """A single GIF block or region.

    kinds:
      header    — 6-byte magic
      lsd       — logical screen descriptor (7 bytes)
      gct       — global color table
      image     — 0x2C block (descriptor + min code size + sub-blocks + terminator)
      extension — 0x21 block (label + sub-blocks + terminator)
      trailer   — 0x3B terminator byte
      raw       — stash of bytes that did not fit any known block
    """

    kind: str
    raw: bytes
    fields: dict = field(default_factory=dict)


def _parse_subblocks(data: bytes, pos: int) -> tuple[list[bytes], int] | tuple[None, None]:
    """Parse a sub-block run starting at *pos*.

    Returns (list of raw sub-blocks including length byte, new pos),
    or (None, None) if the stream is truncated before a 0x00 terminator.
    """
    subs: list[bytes] = []
    while pos < len(data):
        ln = data[pos]
        if ln == 0:
            return subs, pos + 1
        if pos + 1 + ln > len(data):
            return None, None
        subs.append(data[pos : pos + 1 + ln])
        pos += 1 + ln
    return None, None


def parse_gif(data: bytes) -> list[GifNode] | None:
    """Parse a GIF into a list of nodes.

    Returns None if the data does not start with a GIF magic.
    Truncated trailing blocks are stashed as raw nodes.
    """
    if len(data) < 13 or data[:6] not in MAGICS:
        return None

    nodes: list[GifNode] = [GifNode("header", data[:6])]

    lsd = data[6:13]
    flags = lsd[4]
    nodes.append(
        GifNode(
            "lsd",
            lsd,
            {
                "width": struct.unpack_from("<H", lsd, 0)[0],
                "height": struct.unpack_from("<H", lsd, 2)[0],
                "flags": flags,
                "bg": lsd[5],
                "aspect": lsd[6],
            },
        )
    )

    pos = 13
    if flags & 0x80:
        gct_size = 3 * (1 << ((flags & 0x07) + 1))
        if pos + gct_size > len(data):
            nodes.append(GifNode("raw", data[pos:]))
            return nodes
        gct = data[pos : pos + gct_size]
        nodes.append(GifNode("gct", gct, {"data": gct}))
        pos += gct_size

    while pos < len(data):
        b = data[pos]
        if b == 0x2C:
            # Image descriptor: 0x2C + 9 descriptor bytes + min code size.
            # data[pos+10] (min code size) requires pos+11 <= len(data).
            if pos + 11 > len(data):
                nodes.append(GifNode("raw", data[pos:]))
                break
            desc = data[pos + 1 : pos + 10]
            mincode = data[pos + 10]
            subblocks, pos2 = _parse_subblocks(data, pos + 11)
            if pos2 is None:
                nodes.append(GifNode("raw", data[pos:]))
                break
            nodes.append(
                GifNode(
                    "image",
                    data[pos:pos2],
                    {
                        "left": struct.unpack_from("<H", desc, 0)[0],
                        "top": struct.unpack_from("<H", desc, 2)[0],
                        "width": struct.unpack_from("<H", desc, 4)[0],
                        "height": struct.unpack_from("<H", desc, 6)[0],
                        "packed": desc[8],
                        "mincode": mincode,
                        "subblocks": subblocks,
                    },
                )
            )
            pos = pos2
        elif b == 0x21:
            # Extension: 0x21 + label + sub-blocks
            if pos + 1 >= len(data):
                nodes.append(GifNode("raw", data[pos:]))
                break
            label = data[pos + 1]
            subblocks, pos2 = _parse_subblocks(data, pos + 2)
            if pos2 is None:
                nodes.append(GifNode("raw", data[pos:]))
                break
            nodes.append(
                GifNode(
                    "extension",
                    data[pos:pos2],
                    {"label": label, "subblocks": subblocks},
                )
            )
            pos = pos2
        elif b == 0x3B:
            nodes.append(GifNode("trailer", b"\x3b"))
            pos += 1
            if pos < len(data):
                nodes.append(GifNode("raw", data[pos:]))
            break
        else:
            # Unknown byte: stash one byte and resync
            nodes.append(GifNode("raw", data[pos : pos + 1]))
            pos += 1

    return nodes


def _rebuild_image(node: GifNode) -> None:
    """Rebuild node.raw from the parsed image fields."""
    f = node.fields
    desc = struct.pack(
        "<HHHHB",
        f["left"] & 0xFFFF,
        f["top"] & 0xFFFF,
        f["width"] & 0xFFFF,
        f["height"] & 0xFFFF,
        f["packed"] & 0xFF,
    )
    node.raw = b"\x2c" + desc + bytes([f["mincode"] & 0xFF]) + b"".join(f["subblocks"]) + b"\x00"


def _rebuild_extension(node: GifNode) -> None:
    """Rebuild node.raw from the parsed extension fields."""
    f = node.fields
    node.raw = b"\x21" + bytes([f["label"] & 0xFF]) + b"".join(f["subblocks"]) + b"\x00"


def serialize_gif(nodes: list[GifNode]) -> bytes:
    """Serialize nodes back to bytes by concatenating their raw bytes."""
    return b"".join(n.raw for n in nodes)


def _find_first(nodes: list[GifNode], *kinds: str) -> GifNode | None:
    for n in nodes:
        if n.kind in kinds:
            return n
    return None


class GifMutator:
    """Structure-aware GIF mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = rng or random
        nodes = parse_gif(data)
        if nodes is None:
            return self._generate_random_gif(max_len, rng=self._rng)

        op = self._rng.randint(0, 10)
        mutators = [
            self._mutate_lsd_dim,
            self._mutate_lsd_flags,
            self._mutate_gct,
            self._mutate_image_desc,
            self._mutate_mincode,
            self._rewrite_subblock_len,
            self._insert_subblock,
            self._delete_subblock,
            self._swap_extension_label,
            self._duplicate_block,
            self._truncate_block,
        ]
        result = mutators[op](nodes, max_len)
        if isinstance(result, list):
            return serialize_gif(result)[:max_len]
        return result[:max_len]

    def _mutate_lsd_dim(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        lsd = _find_first(nodes, "lsd")
        if lsd is None:
            return nodes
        f = lsd.fields
        if self._rng.random() < 0.5:
            f["width"] = self._rng.choice(DIM_VALUES + [self._rng.randint(0, max_len)])
        else:
            f["height"] = self._rng.choice(DIM_VALUES + [self._rng.randint(0, max_len)])
        raw = bytearray(lsd.raw)
        struct.pack_into("<H", raw, 0, f["width"] & 0xFFFF)
        struct.pack_into("<H", raw, 2, f["height"] & 0xFFFF)
        lsd.raw = bytes(raw)
        return nodes

    def _mutate_lsd_flags(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        lsd = _find_first(nodes, "lsd")
        if lsd is None:
            return nodes
        new_flags = self._rng.choice(LSD_FLAG_VALUES + [self._rng.randint(0, 0xFF)])
        lsd.fields["flags"] = new_flags
        raw = bytearray(lsd.raw)
        raw[4] = new_flags & 0xFF
        lsd.raw = bytes(raw)
        return nodes

    def _mutate_gct(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        gct = _find_first(nodes, "gct")
        if gct is None or not gct.raw:
            return nodes
        raw = bytearray(gct.raw)
        for _ in range(self._rng.randint(1, min(8, len(raw)))):
            idx = self._rng.randint(0, len(raw) - 1)
            raw[idx] = self._rng.randint(0, 0xFF)
        gct.raw = bytes(raw)
        gct.fields["data"] = gct.raw
        return nodes

    def _mutate_image_desc(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        image = _find_first(nodes, "image")
        if image is None:
            return nodes
        f = image.fields
        key = self._rng.choice(["left", "top", "width", "height", "packed"])
        if key == "packed":
            f[key] = self._rng.choice(LSD_FLAG_VALUES + [self._rng.randint(0, 0xFF)])
        else:
            f[key] = self._rng.choice(DIM_VALUES + [self._rng.randint(0, max_len)])
        _rebuild_image(image)
        return nodes

    def _mutate_mincode(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        image = _find_first(nodes, "image")
        if image is None:
            return nodes
        image.fields["mincode"] = self._rng.choice(MINCODE_VALUES)
        _rebuild_image(image)
        return nodes

    def _rewrite_subblock_len(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        blk = _find_first(nodes, "image", "extension")
        if blk is None or not blk.fields.get("subblocks"):
            return nodes
        subs = blk.fields["subblocks"]
        idx = self._rng.randint(0, len(subs) - 1)
        old = subs[idx]
        new_len = self._rng.choice(SUBBLOCK_LEN_VALUES + [self._rng.randint(0, 0xFF)])
        # Shrink or pad the payload to the new declared length
        payload = old[1:]
        if new_len >= len(payload):
            payload = payload + b"\x00" * (new_len - len(payload))
        else:
            payload = payload[:new_len]
        subs[idx] = bytes([new_len & 0xFF]) + payload
        if blk.kind == "image":
            _rebuild_image(blk)
        else:
            _rebuild_extension(blk)
        return nodes

    def _insert_subblock(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        blk = _find_first(nodes, "image", "extension")
        if blk is None:
            return nodes
        subs = blk.fields["subblocks"]
        ln = self._rng.choice([1, 2, 4, 8, 16])
        payload = self._rng.randbytes(ln)
        subs.insert(self._rng.randint(0, len(subs)), bytes([ln & 0xFF]) + payload)
        if blk.kind == "image":
            _rebuild_image(blk)
        else:
            _rebuild_extension(blk)
        return nodes

    def _delete_subblock(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        blk = _find_first(nodes, "image", "extension")
        if blk is None or not blk.fields.get("subblocks"):
            return nodes
        subs = blk.fields["subblocks"]
        subs.pop(self._rng.randint(0, len(subs) - 1))
        if blk.kind == "image":
            _rebuild_image(blk)
        else:
            _rebuild_extension(blk)
        return nodes

    def _swap_extension_label(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        ext = _find_first(nodes, "extension")
        if ext is None:
            return nodes
        ext.fields["label"] = self._rng.choice(EXTENSION_LABELS + [self._rng.randint(0, 0xFF)])
        _rebuild_extension(ext)
        return nodes

    def _duplicate_block(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        dupable = [i for i, n in enumerate(nodes) if n.kind in ("gct", "image", "extension")]
        if not dupable:
            return nodes
        idx = self._rng.choice(dupable)
        orig = nodes[idx]
        dup = GifNode(
            orig.kind,
            orig.raw,
            {k: (list(v) if isinstance(v, list) else v) for k, v in orig.fields.items()},
        )
        nodes.insert(idx + 1, dup)
        return nodes

    def _truncate_block(self, nodes: list[GifNode], max_len: int) -> list[GifNode]:
        truncable = [i for i, n in enumerate(nodes) if n.kind in ("gct", "image", "extension")]
        if not truncable:
            return nodes
        idx = self._rng.choice(truncable)
        node = nodes[idx]
        if len(node.raw) > 3:
            cut = self._rng.randint(1, len(node.raw) // 2)
            node.raw = node.raw[:cut]
            if node.kind == "gct":
                node.fields["data"] = node.raw
            elif node.kind == "image":
                node.fields["subblocks"] = []
                node.fields["mincode"] = 0
            else:
                node.fields["subblocks"] = []
        return nodes

    def _generate_random_gif(self, _nodes=None, max_len: int = 4096, rng=None) -> bytes:
        """Generate a minimal random GIF."""
        self._rng = rng or self._rng
        w = self._rng.randint(1, 32)
        h = self._rng.randint(1, 32)
        flags = 0x80 | self._rng.randint(0, 7)  # GCT present
        lsd = struct.pack("<HHBBBB", w, h, flags, 0, 0, 0)
        gct = self._rng.randbytes(3 * (1 << ((flags & 0x07) + 1)))
        # One image block with a small LZW stream (min code size + sub-block)
        mincode = self._rng.randint(2, 8)
        lzw_payload = self._rng.randbytes(self._rng.randint(0, 16))
        image_desc = struct.pack("<HHHHB", 0, 0, w, h, 0)
        subblock = bytes([len(lzw_payload) & 0xFF]) + lzw_payload
        image = b"\x2c" + image_desc + bytes([mincode]) + subblock + b"\x00"
        result = b"GIF89a" + lsd + gct + image + b"\x3b"
        return result[:max_len]
