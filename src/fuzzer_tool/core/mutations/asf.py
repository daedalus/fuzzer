"""Structure-aware ASF (Advanced Systems Format / WMV/WMA) mutator.

ASF is a sequence of GUID-tagged objects:

  object_guid(16) + object_size(8 LE, includes this header) + data

A conforming file starts with the Header Object, followed by the Data
Object and (optionally) an Index Object. This mutator parses that
top-level object sequence and treats each object's `data` as an opaque
blob — it does not recurse into the Header Object's own nested
sub-objects (File Properties, Stream Properties, ...), except for one
targeted, best-effort mutation on the leading `NumberOfHeaderObjects`
count field, which is the field `asfdec.c` uses to decide how many
nested objects to walk. Not recursing into the nested tree trades away
independently corrupting individual sub-objects (matching the scope
decision in nal.py, which similarly doesn't descend into an ISOBMFF
box's internal structure) for a mutator that can't misparse its own
structure back into an invalid file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

OBJECT_HEADER_LEN = 24  # guid(16) + size(8)

HEADER_OBJECT_GUID = bytes.fromhex("3026B2758E66CF11A6D900AA0062CE6C")
DATA_OBJECT_GUID = bytes.fromhex("3626B2758E66CF11A6D900AA0062CE6C")
FILE_PROPERTIES_GUID = bytes.fromhex("A1DCAB8C47A9CF118EE400C00C205365")
STREAM_PROPERTIES_GUID = bytes.fromhex("9107DCB7B7A9CF118EE600C00C205365")
HEADER_EXTENSION_GUID = bytes.fromhex("B503BFF5EA9F0F4385BB40CC50D30B5A")
OTHER_KNOWN_GUIDS = [
    FILE_PROPERTIES_GUID,
    STREAM_PROPERTIES_GUID,
    HEADER_EXTENSION_GUID,
    DATA_OBJECT_GUID,
]


@dataclass
class AsfObject:
    guid: bytes  # 16 bytes
    declared_size: int  # 8-byte LE size field; may not equal 24+len(data)
    data: bytes

    def to_bytes(self) -> bytes:
        return (
            self.guid + (self.declared_size & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little") + self.data
        )


def _parse_object_sequence(buf: bytes) -> list[AsfObject]:
    pos = 0
    n = len(buf)
    out: list[AsfObject] = []
    while pos + OBJECT_HEADER_LEN <= n:
        guid = buf[pos : pos + 16]
        size = int.from_bytes(buf[pos + 16 : pos + 24], "little")
        body_start = pos + OBJECT_HEADER_LEN
        avail = n - body_start
        body_len = max(0, min(size - OBJECT_HEADER_LEN if size >= OBJECT_HEADER_LEN else 0, avail))
        body = buf[body_start : body_start + body_len]
        out.append(AsfObject(guid, size, body))
        pos = body_start + body_len
        if body_len == 0:
            break
    return out


def parse_asf_objects(data: bytes) -> list[AsfObject] | None:
    """Top-level object sequence. Requires the file to open with the
    Header Object GUID and contain at least a Header + Data object."""
    if len(data) < OBJECT_HEADER_LEN or data[:16] != HEADER_OBJECT_GUID:
        return None
    objs = _parse_object_sequence(data)
    return objs if len(objs) >= 2 else None


def serialize_asf_objects(objs: list[AsfObject]) -> bytes:
    return b"".join(o.to_bytes() for o in objs)


class AsfMutator:
    """Structure-aware ASF top-level object mutator."""

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        objs = parse_asf_objects(data)
        if not objs:
            return self._generate_random_asf(max_len=max_len, rng=rng)

        op = rng.randint(0, 6)
        mutators = [
            self._mutate_object_size,
            self._mutate_object_guid,
            self._mutate_header_object_count,
            self._flip_header_body_bit,
            self._duplicate_object,
            self._delete_object,
            self._reorder_objects,
        ]
        mutators[op](objs, rng)
        return serialize_asf_objects(objs)[:max_len]

    def _mutate_object_size(self, objs: list[AsfObject], rng) -> None:
        """Corrupt a declared object size independent of its real data
        length — the field asfdec.c trusts to skip/bound each object."""
        target = rng.choice(objs)
        real = OBJECT_HEADER_LEN + len(target.data)
        target.declared_size = rng.choice(
            [0, 0xFFFFFFFFFFFFFFFF, real + 1, max(0, real - 1), rng.randint(0, 0xFFFFFFFF)]
        )

    def _mutate_object_guid(self, objs: list[AsfObject], rng) -> None:
        """Relabel an object's type GUID — tests type-dispatch confusion."""
        target = rng.choice(objs)
        target.guid = rng.choice(
            OTHER_KNOWN_GUIDS + [bytes(rng.randint(0, 255) for _ in range(16))]
        )

    def _mutate_header_object_count(self, objs: list[AsfObject], rng) -> None:
        """Patch the Header Object's leading NumberOfHeaderObjects (u32 LE)
        against its real nested-object count — the count the demuxer
        actually iterates by, independent of what's really there."""
        header = objs[0] if objs and objs[0].guid == HEADER_OBJECT_GUID else None
        if header is None or len(header.data) < 4:
            self._mutate_object_size(objs, rng)
            return
        body = bytearray(header.data)
        bogus = rng.choice([0, 0xFFFFFFFF, rng.randint(0, 0xFFFFFFFF)])
        body[0:4] = bogus.to_bytes(4, "little")
        header.data = bytes(body)

    def _flip_header_body_bit(self, objs: list[AsfObject], rng) -> None:
        """Generic bit flip somewhere inside the Header Object's nested
        sub-object region, since that region isn't parsed structurally."""
        header = objs[0]
        if not header.data:
            return
        body = bytearray(header.data)
        idx = rng.randint(0, len(body) - 1)
        body[idx] ^= 1 << rng.randint(0, 7)
        header.data = bytes(body)

    def _duplicate_object(self, objs: list[AsfObject], rng) -> None:
        idx = rng.randint(0, len(objs) - 1)
        orig = objs[idx]
        objs.insert(idx + 1, AsfObject(orig.guid, orig.declared_size, orig.data))

    def _delete_object(self, objs: list[AsfObject], rng) -> None:
        if len(objs) > 2:
            # Never drop below 2 so a Header+Data skeleton always survives.
            idx = rng.randint(1, len(objs) - 1)
            objs.pop(idx)

    def _reorder_objects(self, objs: list[AsfObject], rng) -> None:
        if len(objs) >= 2:
            i, j = rng.sample(range(len(objs)), 2)
            objs[i], objs[j] = objs[j], objs[i]

    def _generate_random_asf(self, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        # File Properties Object as the sole header sub-object.
        file_props_body = bytes(rng.randint(0, 255) for _ in range(80))
        file_props = AsfObject(
            FILE_PROPERTIES_GUID, OBJECT_HEADER_LEN + len(file_props_body), file_props_body
        )

        num_header_objects = 1
        reserved = bytes([1, 2])
        header_body = num_header_objects.to_bytes(4, "little") + reserved + file_props.to_bytes()
        header_obj = AsfObject(
            HEADER_OBJECT_GUID, OBJECT_HEADER_LEN + len(header_body), header_body
        )

        data_body = bytes(16) + bytes(
            rng.randint(0, 255) for _ in range(32)
        )  # file_id placeholder + padding
        data_obj = AsfObject(DATA_OBJECT_GUID, OBJECT_HEADER_LEN + len(data_body), data_body)

        return serialize_asf_objects([header_obj, data_obj])[:max_len]
