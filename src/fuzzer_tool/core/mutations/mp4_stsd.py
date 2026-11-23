"""stsd sample-description mutator (MP4/MOV).

``mp4_codec_cfg_mutate`` edits avcC/hvcC/esds and ``isobmff`` swaps the first
entry's fourcc; nothing touched the entry table itself:

  stsd  [ver/flags u32][entry_count u32] entry*
  entry [size u32][fourcc][6 B reserved][data_ref u16] fixed fields, child boxes
          visual: 78 B fixed        audio: 28 B (+16 QT v1, +36 QT v2)

Ops: COUNT (entry_count edge), FIELD (fixed-field edge: width, channels,
samplerate, QT packet sizes), DUP/DROP/SWAP (entries), CHILD_DUP/DROP/ADD
(boxes inside one entry). Repair draws keep the count or the entry size
valid half the time and stale the other half, so both a consistent table
and a lying one reach the parser. The stsd box itself and its ancestors are
always resized through ``mp4_path.splice``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

from fuzzer_tool.core.mutations.mp4_codec_cfg import W1_EDGES, W2_EDGES, W4_EDGES
from fuzzer_tool.core.mutations.mp4_path import (
    AUDIO_ENTRY_LEN,
    BOX_HDR,
    QT_SOUND_EXTRA,
    QT_VERSION_OFF,
    U32_MAX,
    VISUAL_ENTRY_LEN,
    find_boxes,
    splice,
)
from fuzzer_tool.core.rand_pool import RandPool

STSD = frozenset({b"stsd"})
STSD_HEAD = 8  # version/flags + entry_count
COUNT_OFF = 4
MAX_ENTRIES = 64
CHILD_MAX = 64  # bytes of random payload in an added box
BIG_COUNT = 0x7FFFFFFF

VISUAL_TYPES = frozenset(
    {b"avc1", b"avc3", b"hvc1", b"hev1", b"encv", b"mp4v", b"vp08", b"vp09", b"av01"}
    | {b"dvh1", b"dvhe", b"s263", b"jpeg"}
)
AUDIO_TYPES = frozenset(
    {b"mp4a", b"enca", b"ac-3", b"ec-3", b"alac", b"samr", b"sawb", b"Opus", b"fLaC"}
    | {b"sowt", b"twos", b"lpcm", b"ulaw", b"alaw"}
)
CHILD_TYPES = (
    b"pasp", b"colr", b"btrt", b"clap", b"fiel", b"chnl", b"wave", b"esds",
    b"avcC", b"hvcC", b"dac3", b"dec3", b"srat", b"glbl",
)  # fmt: skip


class Op(IntEnum):
    """stsd edit; values are the RNG draw."""

    COUNT = 0
    FIELD = 1
    DUP = 2
    DROP = 3
    SWAP = 4
    CHILD_DUP = 5
    CHILD_DROP = 6
    CHILD_ADD = 7


class Repair(IntEnum):
    """Whether a derived field is rewritten to match; values are the draw."""

    KEEP = 0
    FIX = 1


class Kind(IntEnum):
    VISUAL = 0
    AUDIO = 1
    OTHER = 2


class Field(NamedTuple):
    """Fixed field at *off* bytes after the entry's box header."""

    name: str
    off: int
    width: int


VISUAL_FIELDS = (
    Field("width", 24, 2),
    Field("height", 26, 2),
    Field("hres", 28, 4),
    Field("vres", 32, 4),
    Field("frames", 40, 2),
    Field("cname_len", 42, 1),
    Field("depth", 74, 2),
    Field("predef", 76, 2),
)
AUDIO_FIELDS = (
    Field("qt_version", 8, 2),
    Field("qt_rev", 10, 2),
    Field("channels", 16, 2),
    Field("samplesize", 18, 2),
    Field("compid", 20, 2),
    Field("pktsize", 22, 2),
    Field("samplerate", 24, 4),
)
QT_V1_FIELDS = (
    Field("samples_per_pkt", 28, 4),
    Field("bytes_per_pkt", 32, 4),
    Field("bytes_per_frame", 36, 4),
    Field("bytes_per_sample", 40, 4),
)
QT_V2_FIELDS = (
    Field("num_channels", 40, 4),
    Field("const_bytes_pkt", 56, 4),
    Field("const_lpcm_pkt", 60, 4),
)
QT_FIELDS = {1: QT_V1_FIELDS, 2: QT_V2_FIELDS}
EDGES = {1: W1_EDGES, 2: W2_EDGES, 4: W4_EDGES}


@dataclass
class Stsd:
    """Parsed stsd payload. *entries* come from the bytes, not from *count*."""

    head: bytes  # version/flags
    count: int
    entries: list[bytes]
    tail: bytes


def count_edges(n: int) -> tuple[int, ...]:
    """entry_count values around the real count *n*."""
    return (0, 1, n + 1, max(n - 1, 0), n * 2, BIG_COUNT, U32_MAX)


def parse_stsd(payload: bytes) -> Stsd | None:
    """Split a stsd payload; stops at the first lying entry size."""
    if len(payload) < STSD_HEAD:
        return None

    count = struct.unpack_from(">I", payload, COUNT_OFF)[0]
    entries: list[bytes] = []
    pos = STSD_HEAD
    while pos + BOX_HDR <= len(payload) and len(entries) < MAX_ENTRIES:
        size = struct.unpack_from(">I", payload, pos)[0]
        if size < BOX_HDR or pos + size > len(payload):
            break
        entries.append(payload[pos : pos + size])
        pos += size
    return Stsd(payload[:COUNT_OFF], count, entries, payload[pos:])


def _render(st: Stsd) -> bytes:
    return st.head + struct.pack(">I", st.count) + b"".join(st.entries) + st.tail


def _kind(entry: bytes) -> Kind:
    typ = entry[4:8]
    if typ in VISUAL_TYPES:
        return Kind.VISUAL
    return Kind.AUDIO if typ in AUDIO_TYPES else Kind.OTHER


def _qt_version(entry: bytes) -> int:
    off = BOX_HDR + QT_VERSION_OFF
    if len(entry) < off + 2:
        return 0
    return struct.unpack_from(">H", entry, off)[0]


def _fixed_len(entry: bytes, kind: Kind) -> int:
    if kind is Kind.VISUAL:
        return VISUAL_ENTRY_LEN
    return AUDIO_ENTRY_LEN + QT_SOUND_EXTRA.get(_qt_version(entry), 0)


def _fields(entry: bytes) -> tuple[Field, ...]:
    """Editable fixed fields that fit inside *entry*."""
    kind = _kind(entry)
    if kind is Kind.OTHER:
        return ()
    if kind is Kind.VISUAL:
        pool = VISUAL_FIELDS
    else:
        pool = AUDIO_FIELDS + QT_FIELDS.get(_qt_version(entry), ())
    return tuple(f for f in pool if BOX_HDR + f.off + f.width <= len(entry))


def _kids(entry: bytes) -> tuple[int, list[bytes], bytes] | None:
    """(fixed-part length, child boxes, trailing bytes); None if opaque."""
    kind = _kind(entry)
    head = BOX_HDR + (0 if kind is Kind.OTHER else _fixed_len(entry, kind))
    if kind is Kind.OTHER or head > len(entry):
        return None

    kids: list[bytes] = []
    pos = head
    while pos + BOX_HDR <= len(entry):
        size = struct.unpack_from(">I", entry, pos)[0]
        if size < BOX_HDR or pos + size > len(entry):
            break
        kids.append(entry[pos : pos + size])
        pos += size
    return head, kids, entry[pos:]


def _rebuild(entry: bytes, head: int, kids: list[bytes], tail: bytes, rng: RandPool) -> bytes:
    """Entry from parts; size field fixed or left stale per the repair draw."""
    body = entry[:head] + b"".join(kids) + tail
    fixed = Repair(rng.randint(0, 1)) is Repair.FIX
    size = len(body) if fixed else struct.unpack_from(">I", entry)[0]
    return struct.pack(">I", size & U32_MAX) + body[4:]


def _op_count(st: Stsd, rng: RandPool) -> bool:
    st.count = rng.choice(count_edges(len(st.entries)))
    return True


def _op_field(st: Stsd, rng: RandPool) -> bool:
    if not st.entries:
        return False

    i = rng.randint(0, len(st.entries) - 1)
    fields = _fields(st.entries[i])
    if not fields:
        return False

    fld = fields[rng.randint(0, len(fields) - 1)]
    value = rng.choice(EDGES[fld.width])
    entry = bytearray(st.entries[i])
    at = BOX_HDR + fld.off
    entry[at : at + fld.width] = (value & ((1 << (8 * fld.width)) - 1)).to_bytes(fld.width, "big")
    st.entries[i] = bytes(entry)
    return True


def _recount(st: Stsd, rng: RandPool) -> None:
    if Repair(rng.randint(0, 1)) is Repair.FIX:
        st.count = len(st.entries)


def _op_dup(st: Stsd, rng: RandPool) -> bool:
    if not st.entries or len(st.entries) >= MAX_ENTRIES:
        return False

    i = rng.randint(0, len(st.entries) - 1)
    st.entries.insert(i + 1, st.entries[i])
    _recount(st, rng)
    return True


def _op_drop(st: Stsd, rng: RandPool) -> bool:
    if not st.entries:
        return False

    del st.entries[rng.randint(0, len(st.entries) - 1)]
    _recount(st, rng)
    return True


def _op_swap(st: Stsd, rng: RandPool) -> bool:
    if len(st.entries) < 2:
        return False

    i = rng.randint(0, len(st.entries) - 1)
    j = rng.randint(0, len(st.entries) - 1)
    if i == j:
        return False
    st.entries[i], st.entries[j] = st.entries[j], st.entries[i]
    return True


def _pick_kids(st: Stsd, rng: RandPool):
    """Draw an entry; return (index, parts) or None when opaque."""
    if not st.entries:
        return None

    i = rng.randint(0, len(st.entries) - 1)
    parts = _kids(st.entries[i])
    return None if parts is None else (i, parts)


def _op_child_dup(st: Stsd, rng: RandPool) -> bool:
    picked = _pick_kids(st, rng)
    if picked is None or not picked[1][1]:
        return False

    i, (head, kids, tail) = picked
    k = rng.randint(0, len(kids) - 1)
    kids.insert(k + 1, kids[k])
    st.entries[i] = _rebuild(st.entries[i], head, kids, tail, rng)
    return True


def _op_child_drop(st: Stsd, rng: RandPool) -> bool:
    picked = _pick_kids(st, rng)
    if picked is None or not picked[1][1]:
        return False

    i, (head, kids, tail) = picked
    del kids[rng.randint(0, len(kids) - 1)]
    st.entries[i] = _rebuild(st.entries[i], head, kids, tail, rng)
    return True


def _op_child_add(st: Stsd, rng: RandPool) -> bool:
    picked = _pick_kids(st, rng)
    if picked is None:
        return False

    i, (head, kids, tail) = picked
    typ = rng.choice(CHILD_TYPES)
    blob = rng.randbytes(rng.randint(0, CHILD_MAX))
    kids.insert(rng.randint(0, len(kids)), struct.pack(">I", BOX_HDR + len(blob)) + typ + blob)
    st.entries[i] = _rebuild(st.entries[i], head, kids, tail, rng)
    return True


_OPS = {
    Op.COUNT: _op_count,
    Op.FIELD: _op_field,
    Op.DUP: _op_dup,
    Op.DROP: _op_drop,
    Op.SWAP: _op_swap,
    Op.CHILD_DUP: _op_child_dup,
    Op.CHILD_DROP: _op_child_drop,
    Op.CHILD_ADD: _op_child_add,
}


def mutate_mp4_stsd(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Edit one stsd table or entry; None if declined.

    Draw order: stsd index, op, then the op's draws (entry, field/child,
    value, repair).
    """
    hits = find_boxes(data, STSD)
    if not hits:
        return None

    hit = hits[rng.randint(0, len(hits) - 1)]
    st = parse_stsd(data[hit.start + BOX_HDR : hit.start + hit.size])
    if st is None:
        return None

    op = Op(rng.randint(0, len(Op) - 1))
    if not _OPS[op](st, rng):
        return None

    out = splice(data, hit, _render(st))
    return None if out is None else out[:max_len]
