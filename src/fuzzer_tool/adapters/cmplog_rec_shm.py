"""Cmplog CMP records in a SysV segment (``$_CMPLOG_RECORDS_SHM``).

The shim used to hex-format every record onto a FIFO and Python parsed the
text back. With this segment attached it stores operands as binary; Python
reads them through numpy views. DIV/GEP records stay on the text stream.

    header   80 B   magic | gen | int_sites | int_hits | byte_slots |
                    rec_slots | rec_used | byte_used |
                    site_dropped | hit_dropped | rec_dropped | byte_dropped
    sites    {pc u64, gen u32, count u32, width u8, pad[7]}[int_sites]
    recs     {a u64, b u64, site u32, pad u32}[rec_slots]   (trace-cmp, write order)
    bytes    {a[64], b[64], n u32, result i32, k u8, pad[3]}[byte_slots]   (libc)

The site table is an index keyed by (pc, width), sized per target from its
trace-cmp call-site count; it holds each site's pc, width and how many
records it wrote this generation (at most ``int_hits``). Records share one
array, so memory is ``sites x 24 B + recs x 24 B``, not sites x hits.

A drain reads what this generation wrote, then bumps ``gen``: a site whose
stamp differs is free, so the reset is O(1). Mirrors ``struct
__afl_rec_hdr`` in ``afl_shim.c``; a shim that predates it never writes the
magic, and the text stream keeps carrying CMP records.
"""

from __future__ import annotations

import ctypes
import weakref
from typing import NamedTuple

import numpy as np

from fuzzer_tool.adapters import libc_shm

CMPLOG_REC_ENV = "_CMPLOG_RECORDS_SHM"

# Must equal __AFL_REC_MAGIC / CMPLOG_MAX_OPERAND in afl_shim.c.
REC_MAGIC = int.from_bytes(b"CMPREC02", "little")
REC_OPERAND = 64

# Site table: default when the call-site count is unknown, floor and cap.
INT_SITES = 1 << 14
MIN_SITES = 1 << 12
MAX_SITES = 1 << 20
# Open addressing with 8 probes: size the table to at most half full.
_LOAD_FACTOR_INV = 2

# Records per site per drain. Only keeps one hot site from filling the
# shared array (costs no memory). 32 (AFL++) refused 944k records / 5k
# FFmpeg execs that the text path, with no per-site limit, keeps.
INT_HITS = 1024
REC_SLOTS = 1 << 18  # trace-cmp records per drain (6 MiB)
BYTE_SLOTS = 4096  # libc records per drain (560 KiB)

_U64_BYTES = 8

_HEADER = np.dtype(
    [
        ("magic", "<u8"),
        ("gen", "<u4"),
        ("int_sites", "<u4"),
        ("int_hits", "<u4"),
        ("byte_slots", "<u4"),
        ("rec_slots", "<u4"),
        ("rec_used", "<u4"),
        ("byte_used", "<u4"),
        ("pad", "<u4"),
        ("site_dropped", "<u8"),
        ("hit_dropped", "<u8"),
        ("rec_dropped", "<u8"),
        ("byte_dropped", "<u8"),
        ("pad2", "<u8"),
    ]
)
_SITE = np.dtype(
    [("pc", "<u8"), ("gen", "<u4"), ("count", "<u4"), ("width", "u1"), ("pad", "u1", (7,))]
)
_REC = np.dtype([("a", "<u8"), ("b", "<u8"), ("site", "<u4"), ("pad", "<u4")])
_BYTES = np.dtype(
    [
        ("a", "u1", (REC_OPERAND,)),
        ("b", "u1", (REC_OPERAND,)),
        ("n", "<u4"),
        ("result", "<i4"),
        ("k", "u1"),
        ("pad", "u1", (3,)),
    ]
)
assert _HEADER.itemsize == 80 and _SITE.itemsize == 24 and _REC.itemsize == 24
assert _BYTES.itemsize == 140


class RecDrops(NamedTuple):
    """Records the table refused, by cause."""

    site: int  # site table probe run full
    hit: int  # site already wrote int_hits records this drain
    rec: int  # record array full
    byte: int  # libc record array full


def site_slots_for(call_sites: int | None) -> int:
    """Site table size for a target with *call_sites* trace-cmp call sites.

    Power of two, at most half full when every site fires, in
    [MIN_SITES, MAX_SITES]; INT_SITES when the count is unknown.
    Example: FFmpeg 132,959 sites -> 524,288 slots (12 MiB).
    """
    if call_sites is None:
        return INT_SITES
    want = max(call_sites * _LOAD_FACTOR_INV, MIN_SITES)
    return min(1 << (want - 1).bit_length(), MAX_SITES)


def _release(addr: int, shm_id: int) -> None:
    libc_shm.shmdt(addr)
    libc_shm.shmctl_rmid(shm_id)


def _layout(int_sites: int, rec_slots: int, byte_slots: int) -> tuple[int, int, int, int]:
    """Offsets of sites/recs/bytes and the total size."""
    sites = _HEADER.itemsize
    recs = sites + _SITE.itemsize * int_sites
    byts = recs + _REC.itemsize * rec_slots
    return sites, recs, byts, byts + _BYTES.itemsize * byte_slots


class CmpRecordShm:
    """One record segment: create, drain into cond tuples, release."""

    def __init__(
        self,
        int_sites: int = INT_SITES,
        int_hits: int = INT_HITS,
        rec_slots: int = REC_SLOTS,
        byte_slots: int = BYTE_SLOTS,
    ):
        if int_sites < 2 or int_sites & (int_sites - 1):
            raise ValueError("int_sites must be a power of two >= 2")
        if min(int_hits, rec_slots, byte_slots) < 1:
            raise ValueError("int_hits, rec_slots and byte_slots must be >= 1")

        off_sites, off_recs, off_bytes, size = _layout(int_sites, rec_slots, byte_slots)
        shm_id = libc_shm.shmget(size)
        if shm_id is None:
            raise OSError("shmget failed for cmplog records")

        addr = libc_shm.shmat(shm_id)
        if addr is None:
            libc_shm.shmctl_rmid(shm_id)
            raise OSError("shmat failed for cmplog records")

        self._id = shm_id
        self._finalizer = weakref.finalize(self, _release, addr, shm_id)
        buf = (ctypes.c_char * size).from_address(addr)
        self._hdr = np.frombuffer(buf, _HEADER, 1, 0)
        self._sites = np.frombuffer(buf, _SITE, int_sites, off_sites)
        self._recs = np.frombuffer(buf, _REC, rec_slots, off_recs)
        self._bytes = np.frombuffer(buf, _BYTES, byte_slots, off_bytes)
        self.int_sites, self._nrecs, self._nbytes = int_sites, rec_slots, byte_slots
        self.size = size

        hdr = self._hdr
        hdr["gen"] = 1
        hdr["int_sites"] = int_sites
        hdr["int_hits"] = int_hits
        hdr["rec_slots"] = rec_slots
        hdr["byte_slots"] = byte_slots

    @property
    def env_id(self) -> str:
        return str(self._id)

    @property
    def attached(self) -> bool:
        """True once a shim attached and writes records here."""
        return int(self._hdr["magic"][0]) == REC_MAGIC

    def drain(self) -> list[tuple]:
        """This generation's ``(op_a, op_b, width, result, pc)``, distinct, then reset.

        Order: trace-cmp records in write order, then libc records in write order.
        """
        hdr = self._hdr
        out = self._int_conds(int(hdr["rec_used"][0])) + self._byte_conds(int(hdr["byte_used"][0]))

        hdr["rec_used"] = 0
        hdr["byte_used"] = 0
        gen = (int(hdr["gen"][0]) + 1) & 0xFFFFFFFF
        hdr["gen"] = gen or 1
        return list(dict.fromkeys(out))

    def _int_conds(self, used: int) -> list[tuple]:
        """The first *used* trace-cmp records, pc and width from their site."""
        used = min(used, self._nrecs)
        if not used:
            return []

        recs = self._recs[:used]
        sites = self._sites[np.minimum(recs["site"], self.int_sites - 1)]
        a, b = recs["a"], recs["b"]
        result = (a > b).astype(np.int64) - (a < b)

        abuf = a.astype("<u8").tobytes()
        bbuf = b.astype("<u8").tobytes()
        out = []
        for i, (w, r, pc) in enumerate(
            zip(sites["width"].tolist(), result.tolist(), sites["pc"].tolist(), strict=True)
        ):
            o = i * _U64_BYTES
            out.append((abuf[o : o + w], bbuf[o : o + w], w, r, pc))
        return out

    def _byte_conds(self, used: int) -> list[tuple]:
        """Libc records (no pc): operands truncated to ``k`` bytes."""
        used = min(used, self._nbytes)
        if not used:
            return []

        recs = self._bytes[:used]
        abuf = recs["a"].tobytes()
        bbuf = recs["b"].tobytes()
        out = []
        for i, (k, n, r) in enumerate(
            zip(recs["k"].tolist(), recs["n"].tolist(), recs["result"].tolist(), strict=True)
        ):
            o = i * REC_OPERAND
            out.append((abuf[o : o + k], bbuf[o : o + k], n, r, None))
        return out

    def take_dropped(self) -> RecDrops:
        """Records refused since the last call, by cause; zeroes the counters."""
        hdr = self._hdr
        fields = ("site_dropped", "hit_dropped", "rec_dropped", "byte_dropped")
        drops = RecDrops(*(int(hdr[f][0]) for f in fields))
        for f in fields:
            hdr[f] = 0
        return drops

    def close(self) -> None:
        self._finalizer()
