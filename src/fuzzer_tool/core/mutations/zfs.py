"""ZFS uberblock / XDR nvlist mutator with embedded label-checksum repair.

A vdev label (256 KiB, 4 copies) holds the pool config and the uberblock ring::

    label + 0     8K blank | 8K boot header
    label + 16K   vdev_phys: XDR nvlist ......................... | zio_eck_t
    label + 128K  uberblock ring, slots of 2^max(ashift,10) bytes, each:
                  magic 0x00bab10c | version | txg | guid_sum | timestamp
                  | rootbp (blkptr_t, 128 B) | sw_version | mmp magic/delay/
                  config | checkpoint_txg ....................... | zio_eck_t

    zio_eck_t (last 40 bytes): zec_magic 0x0210da7ab10c7a11 | 4 x u64 SHA-256

The label checksum is SHA-256 over the whole region with the checksum words
set to the verifier ``{vdev offset, 0, 0, 0}``; the digest is stored as four
big-endian-read words in the writer's byte order (that of ``zec_magic``).
Inputs are usually fragments, so slots are found by magic (either byte
order) at 1 KiB-aligned offsets, the verifier offset is the region's offset
in the buffer (buffer = vdev from byte 0), and nvlists are found at offset 0
or at label+16K.

Edits target import/zdb/vdev_label parsers: version gates (5000 =
feature flags), txg regressions picking a stale uberblock, DVA vdev/offset/
asize and blk_prop fields (lsize/psize/compress/checksum/type/level/embedded)
steering block reads, nvpair encode sizes / name lengths / data types /
nelem / string lengths overrunning the stream, and a removed terminator.
Every edit is in place; a sealed region is resealed so the edit gets past
``zio_checksum_error``, except ECK_BREAK, which corrupts one checksum bit.

Draw order: op (``choice`` over the live ops, in ``Op`` order), target
index (``randint``), field selector (``randint``, only when the op has
several fields), value (``choice``); ECK_BREAK draws region, then bit.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import IntEnum

from fuzzer_tool.core.rand_pool import RandPool

UB_MAGIC = 0x00BAB10C
ZEC_MAGIC = 0x0210DA7AB10C7A11
MMP_MAGIC = 0xA11CEA11
SLOT_MIN = 1024
SLOT_SIZES = (1024, 2048, 4096, 8192)
ECK_LEN = 40
CKSUM_BITS = 256
LABEL_SIZE = 256 * 1024
PHYS_OFF = 16 * 1024
PHYS_SIZE = 112 * 1024
MAX_SLOTS = 512
MAX_PAIRS = 256
MAX_DEPTH = 8

_MAGIC_FMT = {
    struct.pack("<Q", UB_MAGIC): "<",
    struct.pack(">Q", UB_MAGIC): ">",
}
_ZEC_FMT = {
    struct.pack("<Q", ZEC_MAGIC): "<",
    struct.pack(">Q", ZEC_MAGIC): ">",
}

# uberblock_t / blkptr_t field offsets.
UB_VERSION_OFF = 8
UB_TXG_OFF = 16
UB_GUID_OFF = 24
UB_TIME_OFF = 32
UB_ROOTBP = 40
BP_PROP_OFF = UB_ROOTBP + 48
BP_BIRTH_OFFS = (UB_ROOTBP + 72, UB_ROOTBP + 80, UB_ROOTBP + 88)  # phys_birth, birth, fill
UB_SW_VERSION_OFF = 168
UB_MMP_OFFS = (176, 184, 192)  # mmp magic, delay, config
UB_CHECKPOINT_OFF = 200
DVA_LEN = 16
N_DVAS = 3

U64_MAX = (1 << 64) - 1
U64_EDGES = (0, 1, 0xFFFFFFFF, 1 << 32, (1 << 63) - 1, 1 << 63, U64_MAX)
# SPA versions: none, first, last numbered (28), gap, feature flags (5000).
UB_VERSIONS = (0, 1, 28, 29, 4999, 5000, 5001, 1 << 63, U64_MAX)

# (word, shift, width) bitfields: DVA word 0/1, then blk_prop.
DVA_FIELDS = (
    (0, 32, 32),
    (0, 24, 8),
    (0, 0, 24),
    (1, 0, 63),
    (1, 63, 1),
)  # vdev grid asize off gang
PROP_FIELDS = (
    (0, 16),  # lsize (sectors - 1)
    (16, 16),  # psize
    (32, 7),  # compression
    (39, 1),  # embedded
    (40, 8),  # checksum type
    (48, 8),  # object type
    (56, 5),  # level
    (61, 1),  # encrypted
    (62, 1),  # dedup
    (63, 1),  # byteorder
)

# XDR nvlist.
NV_ENCODE_XDR = 1
NV_HEADER_LEN = 4
NVL_HEADER_LEN = 8  # nvl_version, nvl_nvflag
PAIR_MIN = 20  # encode/decode size, name len, type, nelem
DT_STRING = 9
DT_NVLIST = 19
U32_MAX = 0xFFFFFFFF
NV_SIZE_EDGES = (0, 4, 8, PAIR_MIN - 1, 0x7FFFFFFF, 0x80000000, U32_MAX)
NV_LEN_EDGES = (0, 1, 0xFFFF, 0x7FFFFFFF, 0x80000000, U32_MAX)
# data_type_t: unknown, boolean, uint64, string, uint64/string arrays,
# nvlist, nvlist array, double, first undefined, int32 max, -1.
NV_TYPES = (0, 1, 8, 9, 16, 17, 19, 20, 27, 28, 0x7FFFFFFF, U32_MAX)
NV_NELEM_EDGES = (0, 2, 0x10000, 0x7FFFFFFF, 0x80000000, U32_MAX)
NV_TERM_EDGES = (PAIR_MIN, 0x7FFFFFFF, U32_MAX)


class Op(IntEnum):
    """ZFS edit; ``choice`` over the live subset, in this order."""

    UB_VERSION = 0
    UB_TXG = 1
    UB_GUID = 2
    UB_TIME = 3
    UB_MMP = 4
    BP_DVA = 5
    BP_PROP = 6
    BP_BIRTH = 7
    NV_SIZE = 8
    NV_NAME_LEN = 9
    NV_TYPE = 10
    NV_NELEM = 11
    NV_STRLEN = 12
    NV_TERM = 13
    ECK_BREAK = 14


@dataclass(frozen=True)
class Slot:
    """One uberblock slot: offset, byte order, size, has a zio_eck_t trailer."""

    off: int
    fmt: str
    size: int
    sealed: bool


@dataclass(frozen=True)
class Pair:
    """One nvpair header: field offsets inside the buffer."""

    off: int
    name: bytes
    type_off: int
    value_off: int
    dtype: int


@dataclass(frozen=True)
class Region:
    """A checksummed region and the verifier offset its SHA-256 is keyed on."""

    off: int
    size: int
    fmt: str


@dataclass
class _Scan:
    slots: list[Slot] = field(default_factory=list)
    pairs: list[Pair] = field(default_factory=list)
    terms: list[int] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)

    @property
    def strings(self) -> list[Pair]:
        return [p for p in self.pairs if p.dtype == DT_STRING]


def _trailer(data: bytes, off: int, fmt: str) -> tuple[int, bool]:
    """Smallest slot size whose last 40 bytes start with zec_magic in *fmt*."""
    n = len(data)
    for size in SLOT_SIZES:
        end = off + size
        if end > n:
            break
        if _ZEC_FMT.get(data[end - ECK_LEN : end - ECK_LEN + 8]) == fmt:
            return size, True
    return SLOT_MIN, False


def find_slots(data: bytes) -> list[Slot]:
    """Uberblock slots at 1 KiB-aligned offsets, at most MAX_SLOTS."""
    slots: list[Slot] = []
    last = len(data) - SLOT_MIN
    for off in range(0, last + 1, SLOT_MIN):
        fmt = _MAGIC_FMT.get(data[off : off + 8])
        if fmt is None:
            continue

        size, sealed = _trailer(data, off, fmt)
        slots.append(Slot(off, fmt, size, sealed))
        if len(slots) >= MAX_SLOTS:
            break
    return slots


def _is_nvlist(data: bytes, p: int) -> bool:
    """XDR stream header, nvl_version 0 and a small nvflag at *p*."""
    head = data[p : p + NV_HEADER_LEN + NVL_HEADER_LEN]
    if len(head) < NV_HEADER_LEN + NVL_HEADER_LEN:
        return False
    return (
        head[0] == NV_ENCODE_XDR
        and head[1] <= 1
        and head[2:8] == bytes(6)
        and head[8:11] == bytes(3)
        and head[11] <= 3
    )


def _nvlist_starts(data: bytes) -> list[int]:
    """Offset 0 and every label+16K that hold an nvlist header."""
    cands = [0, *range(PHYS_OFF, len(data), LABEL_SIZE)]
    return [p for p in cands if _is_nvlist(data, p)]


def _pair_at(data: bytes, pos: int, esz: int) -> Pair | None:
    """Parse one nvpair header whose encode size is *esz*; None if malformed."""
    if esz < PAIR_MIN or pos + esz > len(data):
        return None

    name_len = struct.unpack_from(">I", data, pos + 8)[0]
    type_off = pos + 12 + ((name_len + 3) & ~3)
    if type_off + 8 > pos + esz:
        return None

    name = bytes(data[pos + 12 : pos + 12 + name_len])
    dtype = struct.unpack_from(">I", data, type_off)[0]
    return Pair(pos, name, type_off, type_off + 8, dtype)


def _walk(data: bytes, start: int, scan: _Scan) -> None:
    """Iterative nvpair walk; an NVLIST pair is followed by its embedded list.

    Encode sizes exclude embedded lists, so after an NVLIST pair the walk
    descends (skipping the nested version/flag) and a zero terminator pops.
    """
    n = len(data)
    pos = start + NV_HEADER_LEN + NVL_HEADER_LEN
    depth = 0
    while len(scan.pairs) < MAX_PAIRS and pos + 8 <= n:
        esz, dsz = struct.unpack_from(">ii", data, pos)
        if esz == 0 and dsz == 0:
            scan.terms.append(pos)
            pos += 8
            if depth == 0:
                return
            depth -= 1
            continue

        pair = _pair_at(data, pos, esz)
        if pair is None:
            return
        scan.pairs.append(pair)
        pos += esz

        if pair.dtype == DT_NVLIST and depth < MAX_DEPTH:
            pos += NVL_HEADER_LEN
            depth += 1


def _phys_region(data: bytes, p: int) -> Region | None:
    """vdev_phys region holding the nvlist at *p*, when its trailer is present."""
    if p % LABEL_SIZE != PHYS_OFF:
        return None
    end = p + PHYS_SIZE
    fmt = _ZEC_FMT.get(data[end - ECK_LEN : end - ECK_LEN + 8])
    return None if fmt is None else Region(p, PHYS_SIZE, fmt)


def _scan(data: bytes) -> _Scan:
    scan = _Scan(slots=find_slots(data))
    scan.regions = [Region(s.off, s.size, s.fmt) for s in scan.slots if s.sealed]
    for p in _nvlist_starts(data):
        _walk(data, p, scan)
        region = _phys_region(data, p)
        if region is not None:
            scan.regions.append(region)
    return scan


def nv_pairs(data: bytes) -> list[Pair]:
    """Every nvpair found (depth-first), for tests and diagnostics."""
    return _scan(data).pairs


def sniff_zfs(data: bytes) -> bool:
    """An uberblock magic at a 1 KiB boundary, or an XDR nvlist header."""
    if any(_MAGIC_FMT.get(data[off : off + 8]) for off in range(0, len(data), SLOT_MIN)):
        return True
    return bool(_nvlist_starts(data))


def _seal(buf: bytearray, reg: Region) -> None:
    """Rewrite the trailer checksum: SHA-256 keyed on the verifier offset."""
    words = reg.off + reg.size - 32
    struct.pack_into(reg.fmt + "4Q", buf, words, reg.off, 0, 0, 0)
    digest = hashlib.sha256(memoryview(buf)[reg.off : reg.off + reg.size]).digest()
    struct.pack_into(reg.fmt + "4Q", buf, words, *struct.unpack(">4Q", digest))


def _reseal(buf: bytearray, scan: _Scan, at: int) -> None:
    for reg in scan.regions:
        if reg.off <= at < reg.off + reg.size:
            _seal(buf, reg)


def _put(buf: bytearray, off: int, code: str, value: int) -> None:
    """Store *value*; flip its low bit if it equals what is already there."""
    if struct.unpack_from(code, buf, off)[0] == value:
        value ^= 1
    struct.pack_into(code, buf, off, value)


def _put_bits(buf: bytearray, off: int, fmt: str, bits: tuple[int, int], value: int) -> None:
    """Set a (shift, width) bitfield of the u64 at *off*; always changes it."""
    shift, width = bits
    mask = ((1 << width) - 1) << shift
    word = struct.unpack_from(fmt + "Q", buf, off)[0]
    new = (word & ~mask) | ((value << shift) & mask)
    if new == word:
        new ^= 1 << shift
    struct.pack_into(fmt + "Q", buf, off, new)


def _width_edges(width: int) -> tuple[int, ...]:
    top = (1 << width) - 1
    return (0, 1, top >> 1, top)


# u64 field groups: offsets inside the slot, value table.
_U64_GROUPS = {
    Op.UB_VERSION: ((UB_VERSION_OFF,), UB_VERSIONS),
    Op.UB_GUID: ((UB_GUID_OFF,), U64_EDGES),
    Op.UB_TIME: ((UB_TIME_OFF, UB_SW_VERSION_OFF, UB_CHECKPOINT_OFF), U64_EDGES),
    Op.UB_MMP: (UB_MMP_OFFS, (*U64_EDGES, MMP_MAGIC)),
    Op.BP_BIRTH: (BP_BIRTH_OFFS, U64_EDGES),
}


class _Editor:
    """One edit on *buf*; each handler returns the offset it touched."""

    def __init__(self, buf: bytearray, scan: _Scan, rng: RandPool) -> None:
        self.buf = buf
        self.scan = scan
        self.rng = rng

    def _pick(self, items: list):
        return items[self.rng.randint(0, len(items) - 1)]

    def u64_group(self, op: Op) -> int:
        slot = self._pick(self.scan.slots)
        offs, values = _U64_GROUPS[op]
        rel = offs[self.rng.randint(0, len(offs) - 1)] if len(offs) > 1 else offs[0]
        off = slot.off + rel
        _put(self.buf, off, slot.fmt + "Q", self.rng.choice(values))
        return off

    def txg(self, _op: Op) -> int:
        """Edge txg, or txg - 1 so an older uberblock looks newest."""
        slot = self._pick(self.scan.slots)
        off = slot.off + UB_TXG_OFF
        cur = struct.unpack_from(slot.fmt + "Q", self.buf, off)[0]
        _put(self.buf, off, slot.fmt + "Q", self.rng.choice((*U64_EDGES, (cur - 1) & U64_MAX)))
        return off

    def dva(self, _op: Op) -> int:
        slot = self._pick(self.scan.slots)
        base = slot.off + UB_ROOTBP + DVA_LEN * self.rng.randint(0, N_DVAS - 1)
        word, shift, width = DVA_FIELDS[self.rng.randint(0, len(DVA_FIELDS) - 1)]
        off = base + 8 * word
        _put_bits(self.buf, off, slot.fmt, (shift, width), self.rng.choice(_width_edges(width)))
        return off

    def prop(self, _op: Op) -> int:
        slot = self._pick(self.scan.slots)
        bits = PROP_FIELDS[self.rng.randint(0, len(PROP_FIELDS) - 1)]
        off = slot.off + BP_PROP_OFF
        _put_bits(self.buf, off, slot.fmt, bits, self.rng.choice(_width_edges(bits[1])))
        return off

    def nv_size(self, _op: Op) -> int:
        """Encode (0) or decode (1) size of one pair."""
        off = self._pick(self.scan.pairs).off + 4 * self.rng.randint(0, 1)
        _put(self.buf, off, ">I", self.rng.choice(NV_SIZE_EDGES))
        return off

    def nv_name_len(self, _op: Op) -> int:
        off = self._pick(self.scan.pairs).off + 8
        _put(self.buf, off, ">I", self.rng.choice(NV_LEN_EDGES))
        return off

    def nv_type(self, _op: Op) -> int:
        off = self._pick(self.scan.pairs).type_off
        _put(self.buf, off, ">I", self.rng.choice(NV_TYPES))
        return off

    def nv_nelem(self, _op: Op) -> int:
        off = self._pick(self.scan.pairs).type_off + 4
        _put(self.buf, off, ">I", self.rng.choice(NV_NELEM_EDGES))
        return off

    def nv_strlen(self, _op: Op) -> int:
        off = self._pick(self.scan.strings).value_off
        _put(self.buf, off, ">I", self.rng.choice(NV_LEN_EDGES))
        return off

    def nv_term(self, _op: Op) -> int:
        """Overwrite a zero terminator's encode size: the list runs on."""
        off = self._pick(self.scan.terms)
        _put(self.buf, off, ">I", self.rng.choice(NV_TERM_EDGES))
        return off

    def eck_break(self, _op: Op) -> None:
        """Reseal, then flip one checksum bit (torn / stale label path)."""
        reg = self._pick(self.scan.regions)
        _seal(self.buf, reg)
        bit = self.rng.randint(0, CKSUM_BITS - 1)
        self.buf[reg.off + reg.size - 32 + bit // 8] ^= 1 << (bit % 8)


_HANDLERS = {
    **{op: _Editor.u64_group for op in _U64_GROUPS},
    Op.UB_TXG: _Editor.txg,
    Op.BP_DVA: _Editor.dva,
    Op.BP_PROP: _Editor.prop,
    Op.NV_SIZE: _Editor.nv_size,
    Op.NV_NAME_LEN: _Editor.nv_name_len,
    Op.NV_TYPE: _Editor.nv_type,
    Op.NV_NELEM: _Editor.nv_nelem,
    Op.NV_STRLEN: _Editor.nv_strlen,
    Op.NV_TERM: _Editor.nv_term,
    Op.ECK_BREAK: _Editor.eck_break,
}
_UB_OPS = [op for op in Op if op <= Op.BP_BIRTH]
_NV_OPS = [Op.NV_SIZE, Op.NV_NAME_LEN, Op.NV_TYPE, Op.NV_NELEM]


def _live_ops(scan: _Scan) -> list[Op]:
    """Ops whose target structure exists, in ``Op`` order."""
    pool: list[Op] = []
    if scan.slots:
        pool += _UB_OPS
    if scan.pairs:
        pool += _NV_OPS
    if scan.strings:
        pool.append(Op.NV_STRLEN)
    if scan.terms:
        pool.append(Op.NV_TERM)
    if scan.regions:
        pool.append(Op.ECK_BREAK)
    return pool


def mutate_zfs(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One uberblock / blkptr / nvlist edit, resealed; None if no ZFS structure."""
    scan = _scan(data)
    pool = _live_ops(scan)
    if not pool:
        return None

    buf = bytearray(data)
    op = rng.choice(pool)
    at = _HANDLERS[op](_Editor(buf, scan, rng), op)
    if at is not None:
        _reseal(buf, scan, at)
    return bytes(buf[:max_len])
