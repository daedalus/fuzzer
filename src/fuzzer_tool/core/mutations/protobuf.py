"""Structure-aware protobuf (wire format) mutator.

Protobuf wire format:
  [tag: varint] = field_num << 3 | wire_type
  wire 0: varint payload
  wire 1: 8-byte payload
  wire 2: length-delimited ([uvarint len][payload])
  wire 5: 4-byte payload
  wire 3/4: group start/end (nested fields)

Parse produces a flat field list plus interstitial raw bytes between
fields and a trailing raw remainder (for mid-stream junk). An untouched
message round-trips byte-identically as long as tags and length fields
were canonically encoded (non-canonical varints are normalized).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

MAX_FIELD_NUM = (1 << 29) - 1


class ProtoFields(list):
    """A list of Field with parse metadata.

    raw_between[i] holds the raw bytes that appeared before field i
    (usually empty); the final element holds the trailing remainder.
    """

    def __init__(self, iterable=()):
        super().__init__(iterable)
        self.raw_between: list[bytes] = []


@dataclass
class Field:
    """A single protobuf field."""

    field_num: int
    wire_type: int
    raw_payload: bytes  # payload bytes, stored verbatim
    children: list[Field] | None = None  # nested fields for group (wire 3)


def _decode_varint(data: bytes, pos: int) -> tuple[int | None, int]:
    """Decode a base-128 varint at *pos*. Returns (value, new_pos)."""
    result = 0
    shift = 0
    for _ in range(10):
        if pos >= len(data):
            return None, pos
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
    return None, pos


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _tag_ok(tag: int) -> bool:
    field_num = tag >> 3
    wire = tag & 7
    return field_num != 0 and field_num <= MAX_FIELD_NUM and wire <= 5


def _parse_fields(
    data: bytes, pos: int, end: int, depth: int = 0, stop_at_end: bool = False
) -> tuple[list[Field], list[bytes]] | None:
    """Parse fields from data[pos:end].

    Returns (fields, raw_between) or None if the first tag is undecodable.
    raw_between[i] holds the raw bytes between field i-1 and field i
    (the final entry holds the trailing remainder). When *stop_at_end*
    is set, a wire-4 end-group tag ends parsing cleanly (the caller
    consumed it).
    """
    if depth > 16:
        return None, []
    fields: list[Field] = []
    raw_between: list[bytes] = []
    gap_start = pos

    while pos < end:
        start = pos
        tag, pos = _decode_varint(data, pos)
        if tag is None or not _tag_ok(tag):
            raw_between.append(data[gap_start:end])
            return fields, raw_between
        raw_between.append(data[gap_start:start])
        field_num = tag >> 3
        wire = tag & 7

        if wire == 0:
            value, new_pos = _decode_varint(data, pos)
            if value is None:
                raw_between.append(data[gap_start:end])
                return fields, raw_between
            fields.append(Field(field_num, wire, data[pos:new_pos]))
            pos = new_pos
        elif wire == 1:
            if pos + 8 > end:
                raw_between.append(data[gap_start:end])
                return fields, raw_between
            fields.append(Field(field_num, wire, data[pos : pos + 8]))
            pos += 8
        elif wire == 5:
            if pos + 4 > end:
                raw_between.append(data[gap_start:end])
                return fields, raw_between
            fields.append(Field(field_num, wire, data[pos : pos + 4]))
            pos += 4
        elif wire == 2:
            length, new_pos = _decode_varint(data, pos)
            if length is None:
                raw_between.append(data[gap_start:end])
                return fields, raw_between
            pos = new_pos
            if pos + length > end:
                # Truncated payload: stash everything from the gap start
                raw_between.append(data[gap_start:end])
                return fields, raw_between
            fields.append(Field(field_num, wire, data[pos : pos + length]))
            pos += length
        elif wire == 3:
            # Group: find the matching end-group tag, then parse nested fields
            end_tag = (field_num << 3) | 4
            scan = pos
            end_pos = -1
            while scan < end:
                t, new_pos = _decode_varint(data, scan)
                if t is None:
                    break
                if t == end_tag:
                    end_pos = new_pos
                    break
                scan = new_pos
            if end_pos < 0:
                # No matching end tag: stash the rest
                raw_between.append(data[gap_start:end])
                return fields, raw_between
            inner, inner_between = _parse_fields(data, pos, end_pos, depth + 1, stop_at_end=True)
            if inner is None:
                raw_between.append(data[gap_start:end])
                return fields, raw_between
            fields.append(Field(field_num, wire, b"", children=inner))
            pos = end_pos
        elif wire == 4:
            # End-group tag: stop parsing (caller consumed it when
            # stop_at_end, otherwise it is mid-stream junk).
            if stop_at_end:
                return fields, raw_between
            raw_between.append(data[gap_start:end])
            return fields, raw_between

        gap_start = pos

    return fields, raw_between


def parse_protobuf(data: bytes) -> ProtoFields | None:
    """Parse a protobuf message into a ProtoFields list.

    Returns None unless the first tag decodes to a valid
    (field_num, wire_type) pair.
    """
    if not data:
        return None
    tag, _pos = _decode_varint(data, 0)
    if tag is None or not _tag_ok(tag):
        return None

    fields, raw_between = _parse_fields(data, 0, len(data))
    if fields is None:
        return None
    result = ProtoFields(fields)
    result.raw_between = raw_between
    return result


def serialize_protobuf(fields: ProtoFields) -> bytes:
    """Serialize fields back to bytes (canonical tag and length varints)."""
    raw_between = getattr(fields, "raw_between", None)
    if raw_between is None:
        raw_between = [b""] * (len(fields) + 1)

    out = bytearray()
    for i, f in enumerate(fields):
        if i < len(raw_between):
            out.extend(raw_between[i])
        tag = _encode_varint((f.field_num << 3) | f.wire_type)
        out.extend(tag)
        if f.wire_type == 2:
            out.extend(_encode_varint(len(f.raw_payload)))
            out.extend(f.raw_payload)
        elif f.wire_type == 3:
            if f.children is not None:
                child_fields = ProtoFields(f.children)
                child_fields.raw_between = [b""] * (len(f.children) + 1)
                out.extend(serialize_protobuf(child_fields))
            out.extend(_encode_varint((f.field_num << 3) | 4))
        else:
            out.extend(f.raw_payload)
    if len(raw_between) > len(fields):
        out.extend(raw_between[-1])
    return bytes(out)


def _varint_value(payload: bytes) -> int | None:
    value, pos = _decode_varint(payload, 0)
    if value is None or pos != len(payload):
        return None
    return value


def _mutate_varint_payload(field: Field, rng: random.Random) -> None:
    """Replace a wire-0 payload with a canonically-encoded mutated value."""
    value = _varint_value(field.raw_payload)
    if value is None:
        return
    options = [
        0,
        1,
        2,
        0x7F,
        0x80,
        0xFFFF,
        0xFFFFFFFF,
        (1 << 63) - 1,
        value + 1,
        value * 2,
        value ^ 0xFFFFFFFF,
    ]
    new_value = rng.choice(options)
    field.raw_payload = _encode_varint(new_value & ((1 << 64) - 1))


# Interesting field numbers for tag renumbering
RENUMBER_VALUES = [1, 2, 15, 16, 100, MAX_FIELD_NUM]

# Wire types that can be swapped in
WIRE_VALUES = [0, 1, 2, 5]


class ProtobufMutator:
    """Structure-aware protobuf mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = rng or random
        fields = parse_protobuf(data)
        if fields is None:
            return self._generate_random_protobuf(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 11)
        mutators = [
            self._renumber_tag,
            self._change_wire_type,
            self._mutate_varint_value,
            self._flip_varint_bytes,
            self._rewrite_length,
            self._delete_field,
            self._duplicate_field,
            self._swap_fields,
            self._mutate_payload_bytes,
            self._splice_fields,
            self._truncate_message,
            self._generate_random_protobuf,
        ]
        result = mutators[op](fields, max_len)
        if isinstance(result, ProtoFields):
            return serialize_protobuf(result)[:max_len]
        return result[:max_len]

    def _renumber_tag(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        if fields:
            target = self._rng.choice(fields)
            target.field_num = self._rng.choice(
                RENUMBER_VALUES + [self._rng.randint(1, MAX_FIELD_NUM)]
            )
        return fields

    def _change_wire_type(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        if fields:
            target = self._rng.choice(fields)
            current = target.wire_type
            options = [w for w in WIRE_VALUES if w != current]
            target.wire_type = self._rng.choice(options)
        return fields

    def _mutate_varint_value(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        targets = [f for f in fields if f.wire_type == 0]
        if targets:
            _mutate_varint_payload(self._rng.choice(targets), self._rng)
        return fields

    def _flip_varint_bytes(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        targets = [f for f in fields if f.wire_type == 0 and f.raw_payload]
        if targets:
            target = self._rng.choice(targets)
            raw = bytearray(target.raw_payload)
            idx = self._rng.randint(0, len(raw) - 1)
            raw[idx] ^= 1 << self._rng.randint(0, 6)
            target.raw_payload = bytes(raw)
        return fields

    def _rewrite_length(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        targets = [f for f in fields if f.wire_type == 2]
        if not targets:
            return fields
        target = self._rng.choice(targets)
        new_len = self._rng.randint(0, min(len(target.raw_payload) + 16, max_len))
        if new_len <= len(target.raw_payload):
            target.raw_payload = target.raw_payload[:new_len]
        else:
            target.raw_payload = target.raw_payload + b"\x00" * (new_len - len(target.raw_payload))
        return fields

    def _delete_field(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        if len(fields) > 1:
            idx = self._rng.randint(0, len(fields) - 1)
            del fields[idx]
            del fields.raw_between[idx]
        return fields

    def _duplicate_field(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        if fields:
            idx = self._rng.randint(0, len(fields) - 1)
            orig = fields[idx]
            dup = Field(
                field_num=orig.field_num,
                wire_type=orig.wire_type,
                raw_payload=orig.raw_payload[:],
                children=list(orig.children) if orig.children else None,
            )
            fields.insert(idx + 1, dup)
            fields.raw_between.insert(idx + 1, b"")
        return fields

    def _swap_fields(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        if len(fields) >= 2:
            i, j = self._rng.sample(list(range(len(fields))), 2)
            fields[i], fields[j] = fields[j], fields[i]
        return fields

    def _mutate_payload_bytes(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        candidates = [f for f in fields if f.raw_payload and f.wire_type in (0, 1, 2, 5)]
        if candidates:
            target = self._rng.choice(candidates)
            raw = bytearray(target.raw_payload)
            for _ in range(self._rng.randint(1, min(8, len(raw)))):
                raw[self._rng.randint(0, len(raw) - 1)] = self._rng.randint(0, 0xFF)
            target.raw_payload = bytes(raw)
        return fields

    def _splice_fields(self, fields: ProtoFields, max_len: int) -> ProtoFields:
        """Append a slice of one field's payload into another."""
        targets = [f for f in fields if f.wire_type == 2 and f.raw_payload]
        if len(targets) >= 2:
            dst = self._rng.choice(targets)
            src = self._rng.choice([t for t in targets if t is not dst])
            chunk = src.raw_payload[: self._rng.randint(1, len(src.raw_payload) + 1)]
            dst.raw_payload = dst.raw_payload[: len(dst.raw_payload) // 2] + chunk
        return fields

    def _truncate_message(self, fields: ProtoFields, max_len: int) -> bytes:
        """Truncate the serialized message at a random byte."""
        raw = bytearray(serialize_protobuf(fields))
        cut = self._rng.randint(0, len(raw))
        return bytes(raw[:cut])

    def _generate_random_protobuf(self, _fields=None, max_len: int = 4096, rng=None) -> bytes:
        """Generate a random protobuf message."""
        # An int in the first slot is a max_len passed positionally. Without
        # this the cap lands in the vestigial placeholder and is dropped, and
        # the generator silently falls back to its own default -- the same
        # overload bmp/gzip/jpeg/zlib already handle and document.
        if isinstance(_fields, int):
            max_len = _fields
        self._rng = rng or self._rng
        fields: list[Field] = []
        for _ in range(self._rng.randint(1, 6)):
            field_num = self._rng.randint(1, 15)
            wire = self._rng.choice([0, 2, 5])
            if wire == 0:
                payload = _encode_varint(self._rng.randint(0, (1 << self._rng.randint(0, 32)) - 1))
            elif wire == 2:
                payload = self._rng.randbytes(self._rng.randint(0, 16))
            else:
                payload = self._rng.randbytes(4)
            fields.append(Field(field_num, wire, payload))
        result = ProtoFields(fields)
        result.raw_between = [b""] * (len(fields) + 1)
        message = serialize_protobuf(result)
        if self._rng.random() < 0.3:
            message += self._rng.randbytes(self._rng.randint(0, 8))
        return message[:max_len]
