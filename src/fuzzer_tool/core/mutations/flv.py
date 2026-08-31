"""Structure-aware FLV (Flash Video) mutator.

FLV layout:

  Header: "FLV" + version(1) + flags(1) + header_size(4 BE, usually 9)
  Then repeated: PreviousTagSize(4 BE) + Tag
    Tag: tag_type(1) + data_size(3 BE) + timestamp(3 BE) + timestamp_ext(1)
         + stream_id(3 BE, always 0) + data(data_size bytes)
  Trailing PreviousTagSize(4 BE) after the last tag.

tag_type is 8 (audio), 9 (video), or 18 (script/AMF metadata) — the
demuxer branches on it to decide how to interpret `data`, so corrupting
it while leaving payload bytes alone is the cheapest type-confusion
trigger this mutator has. `data_size` is an explicit length field (like
ISOBMFF's box size, unlike TS's implicit-in-alignment framing), so a
size/data mismatch is a first-class mutation here too.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TAG_AUDIO = 8
TAG_VIDEO = 9
TAG_SCRIPT = 18
KNOWN_TAG_TYPES = [TAG_AUDIO, TAG_VIDEO, TAG_SCRIPT]


@dataclass
class FlvTag:
    prev_tag_size: int  # PreviousTagSize preceding this tag
    tag_type: int
    data_size: int  # declared 3-byte size; may not equal len(data)
    timestamp: int  # combined 24-bit + 8-bit extension, as one int
    stream_id: int
    data: bytes

    def to_bytes(self) -> bytes:
        ts24 = self.timestamp & 0xFFFFFF
        ts_ext = (self.timestamp >> 24) & 0xFF
        return (
            self.prev_tag_size.to_bytes(4, "big")
            + bytes([self.tag_type & 0xFF])
            + (self.data_size & 0xFFFFFF).to_bytes(3, "big")
            + ts24.to_bytes(3, "big")
            + bytes([ts_ext])
            + (self.stream_id & 0xFFFFFF).to_bytes(3, "big")
            + self.data
        )


def parse_flv(data: bytes) -> tuple[bytes, list[FlvTag], int | None] | None:
    """Returns (header_bytes, tags, trailing_prev_tag_size) or None."""
    if len(data) < 13 or data[:3] != b"FLV":
        return None
    header_size = int.from_bytes(data[5:9], "big")
    if header_size < 9 or header_size > len(data):
        return None
    header = data[:header_size]

    pos = header_size
    n = len(data)
    tags: list[FlvTag] = []
    while pos + 4 <= n:
        prev_tag_size = int.from_bytes(data[pos : pos + 4], "big")
        pos += 4
        if pos + 11 > n:
            trailing = prev_tag_size if not tags else None
            return (header, tags, trailing) if tags else None
        tag_type = data[pos]
        data_size = int.from_bytes(data[pos + 1 : pos + 4], "big")
        ts24 = int.from_bytes(data[pos + 4 : pos + 7], "big")
        ts_ext = data[pos + 7]
        stream_id = int.from_bytes(data[pos + 8 : pos + 11], "big")
        body_start = pos + 11
        body_end = min(body_start + data_size, n)
        tag_data = data[body_start:body_end]
        tags.append(
            FlvTag(prev_tag_size, tag_type, data_size, (ts_ext << 24) | ts24, stream_id, tag_data)
        )
        pos = body_end

    if not tags:
        return None
    trailing = int.from_bytes(data[pos : pos + 4], "big") if pos + 4 <= n else None
    return header, tags, trailing


def serialize_flv(header: bytes, tags: list[FlvTag], trailing: int | None) -> bytes:
    out = header + b"".join(t.to_bytes() for t in tags)
    if trailing is None:
        last = tags[-1]
        trailing = 11 + len(last.data)
    out += trailing.to_bytes(4, "big")
    return out


class FlvMutator:
    """Structure-aware FLV tag-level mutator."""

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        parsed = parse_flv(data)
        if not parsed:
            return self._generate_random_flv(max_len=max_len, rng=rng)
        header, tags, trailing = parsed

        op = rng.randint(0, 7)
        if op == 7:
            out = bytearray(header)
            out[5:9] = rng.choice([9, 0, 0xFFFFFFFF, len(header) + 1]).to_bytes(4, "big")
            return bytes(out + b"".join(t.to_bytes() for t in tags))[:max_len]

        mutators = [
            self._mutate_tag_type,
            self._mutate_data_size,
            self._mutate_timestamp,
            self._mutate_prev_tag_size,
            self._mutate_stream_id,
            self._duplicate_tag,
            self._delete_tag,
        ]
        mutators[op](tags, rng)
        return serialize_flv(header, tags, trailing)[:max_len]

    def _mutate_tag_type(self, tags: list[FlvTag], rng) -> None:
        target = rng.choice(tags)
        target.tag_type = rng.choice(KNOWN_TAG_TYPES + [0, 1, 255])

    def _mutate_data_size(self, tags: list[FlvTag], rng) -> None:
        """Corrupt declared data_size independent of the real data length."""
        target = rng.choice(tags)
        target.data_size = rng.choice(
            [
                0,
                0xFFFFFF,
                len(target.data) + 1,
                max(0, len(target.data) - 1),
                rng.randint(0, 0xFFFFFF),
            ]
        )

    def _mutate_timestamp(self, tags: list[FlvTag], rng) -> None:
        target = rng.choice(tags)
        target.timestamp = rng.choice([0, 0xFFFFFFFF, rng.randint(0, 0xFFFFFFFF)])

    def _mutate_prev_tag_size(self, tags: list[FlvTag], rng) -> None:
        """Corrupt a PreviousTagSize — used for reverse-seek; a mismatch
        against the real preceding tag's length tests that path directly."""
        target = rng.choice(tags)
        target.prev_tag_size = rng.choice([0, 0xFFFFFFFF, rng.randint(0, 0xFFFFFFFF)])

    def _mutate_stream_id(self, tags: list[FlvTag], rng) -> None:
        """stream_id is spec-mandated 0; a nonzero value tests that assumption."""
        target = rng.choice(tags)
        target.stream_id = rng.randint(1, 0xFFFFFF)

    def _duplicate_tag(self, tags: list[FlvTag], rng) -> None:
        idx = rng.randint(0, len(tags) - 1)
        orig = tags[idx]
        tags.insert(
            idx + 1,
            FlvTag(
                orig.prev_tag_size,
                orig.tag_type,
                orig.data_size,
                orig.timestamp,
                orig.stream_id,
                orig.data,
            ),
        )

    def _delete_tag(self, tags: list[FlvTag], rng) -> None:
        if len(tags) > 1:
            tags.pop(rng.randint(0, len(tags) - 1))

    def _generate_random_flv(self, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        header = b"FLV" + bytes([1, 0x05]) + (9).to_bytes(4, "big")
        payload = bytes([0x17, 0x00, 0x00, 0x00, 0x00]) + bytes(
            rng.randint(0, 255) for _ in range(64)
        )
        tag = FlvTag(
            prev_tag_size=0,
            tag_type=TAG_VIDEO,
            data_size=len(payload),
            timestamp=0,
            stream_id=0,
            data=payload,
        )
        return serialize_flv(header, [tag], None)[:max_len]
