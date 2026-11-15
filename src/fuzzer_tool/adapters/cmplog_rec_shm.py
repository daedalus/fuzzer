"""Cmplog CMP records in a SysV segment (``$_CMPLOG_RECORDS_SHM``).

The shim used to hex-format every record onto a FIFO and Python parsed the
text back. With this segment attached it stores operands as binary; Python
reads them through numpy views. DIV/GEP records stay on the text stream.

    header      64 B   magic | gen | int_sites | int_hits | byte_slots |
                       int_used | byte_used | int_dropped | byte_dropped
    touched     u32[int_sites]           site slots in first-touch order
    sites       {pc u64, gen u32, count u32, width u8, pad[7]}[int_sites]
    ints        {a u64, b u64}[int_sites * int_hits]   (trace-cmp records)
    bytes       {a[64], b[64], n u32, result i32, k u8, pad[3]}[byte_slots]
                                                       (libc records)

A drain reads what this generation wrote, then bumps ``gen``: a site whose
stamp differs is free, so the reset is O(1). Mirrors ``struct
__afl_rec_hdr`` in ``afl_shim.c``; a shim that predates it never writes the
magic, and the text stream keeps carrying CMP records.

Memory: defaults 16384 sites x 32 records (8 MiB) + 4096 byte records
(560 KiB), one shared copy.
"""

from __future__ import annotations

import ctypes
import weakref

import numpy as np

from fuzzer_tool.adapters import libc_shm

CMPLOG_REC_ENV = "_CMPLOG_RECORDS_SHM"

# Must equal __AFL_REC_MAGIC / CMPLOG_MAX_OPERAND in afl_shim.c.
REC_MAGIC = int.from_bytes(b"CMPREC01", "little")
REC_OPERAND = 64

INT_SITES = 1 << 14
INT_HITS = 32
BYTE_SLOTS = 4096

_U64_BYTES = 8

_HEADER = np.dtype(
    [
        ("magic", "<u8"),
        ("gen", "<u4"),
        ("int_sites", "<u4"),
        ("int_hits", "<u4"),
        ("byte_slots", "<u4"),
        ("int_used", "<u4"),
        ("byte_used", "<u4"),
        ("int_dropped", "<u8"),
        ("byte_dropped", "<u8"),
        ("pad", "<u8", (2,)),
    ]
)
_SITE = np.dtype(
    [("pc", "<u8"), ("gen", "<u4"), ("count", "<u4"), ("width", "u1"), ("pad", "u1", (7,))]
)
_INT = np.dtype([("a", "<u8"), ("b", "<u8")])
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
assert _HEADER.itemsize == 64 and _SITE.itemsize == 24 and _BYTES.itemsize == 140


def _release(addr: int, shm_id: int) -> None:
    libc_shm.shmdt(addr)
    libc_shm.shmctl_rmid(shm_id)


def _layout(int_sites: int, int_hits: int, byte_slots: int) -> tuple[int, int, int, int, int]:
    """Offsets of touched/sites/ints/bytes and the total size."""
    touched = _HEADER.itemsize
    sites = touched + 4 * int_sites
    ints = sites + _SITE.itemsize * int_sites
    byts = ints + _INT.itemsize * int_sites * int_hits
    return touched, sites, ints, byts, byts + _BYTES.itemsize * byte_slots


class CmpRecordShm:
    """One record segment: create, drain into cond tuples, release."""

    def __init__(
        self, int_sites: int = INT_SITES, int_hits: int = INT_HITS, byte_slots: int = BYTE_SLOTS
    ):
        if int_sites < 2 or int_sites & (int_sites - 1):
            raise ValueError("int_sites must be a power of two >= 2")
        if int_hits < 1 or byte_slots < 1:
            raise ValueError("int_hits and byte_slots must be >= 1")

        off_touched, off_sites, off_ints, off_bytes, size = _layout(int_sites, int_hits, byte_slots)
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
        self._touched = np.frombuffer(buf, np.uint32, int_sites, off_touched)
        self._sites = np.frombuffer(buf, _SITE, int_sites, off_sites)
        self._ints = np.frombuffer(buf, _INT, int_sites * int_hits, off_ints)
        self._bytes = np.frombuffer(buf, _BYTES, byte_slots, off_bytes)
        self._nsites, self._nhits, self._nbytes = int_sites, int_hits, byte_slots
        self.size = size

        hdr = self._hdr
        hdr["gen"] = 1
        hdr["int_sites"] = int_sites
        hdr["int_hits"] = int_hits
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

        Order: trace-cmp sites in first-touch order (each site's records in
        write order), then libc records in write order.
        """
        hdr = self._hdr
        out = self._int_conds(int(hdr["int_used"][0])) + self._byte_conds(int(hdr["byte_used"][0]))

        hdr["int_used"] = 0
        hdr["byte_used"] = 0
        gen = (int(hdr["gen"][0]) + 1) & 0xFFFFFFFF
        hdr["gen"] = gen or 1
        return list(dict.fromkeys(out))

    def _int_conds(self, used: int) -> list[tuple]:
        """Trace-cmp records of the first *used* touched sites."""
        used = min(used, self._nsites)
        if not used:
            return []

        slots = self._touched[:used].astype(np.int64)
        sites = self._sites[slots]
        counts = np.minimum(sites["count"], self._nhits).astype(np.int64)
        total = int(counts.sum())
        if not total:
            return []

        # Flat entry index per record: site base + rank within its site.
        starts = np.repeat(slots * self._nhits, counts)
        rank = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        recs = self._ints[starts + rank]
        a, b = recs["a"], recs["b"]
        result = (a > b).astype(np.int64) - (a < b)

        abuf = a.astype("<u8").tobytes()
        bbuf = b.astype("<u8").tobytes()
        widths = np.repeat(sites["width"], counts).tolist()
        pcs = np.repeat(sites["pc"], counts).tolist()
        out = []
        for i, (w, r, pc) in enumerate(zip(widths, result.tolist(), pcs, strict=True)):
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

    def take_dropped(self) -> tuple[int, int]:
        """``(int, byte)`` records refused since the last call; zeroes them."""
        hdr = self._hdr
        dropped = int(hdr["int_dropped"][0]), int(hdr["byte_dropped"][0])
        hdr["int_dropped"] = 0
        hdr["byte_dropped"] = 0
        return dropped

    def close(self) -> None:
        self._finalizer()
