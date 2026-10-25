"""Cmplog counters in a SysV segment (``$_CMPLOG_COUNTS_SHM``).

The shim used to format ``CNT``/``CNS`` text per exec and Python parsed it
back. Here the shim's counter arrays live in this segment; Python reads
them through numpy views and zeroes what it read (the same delta contract
as the text dumps).

    offset 0     magic           u64  written by the shim once attached
           8     site_dropped    u64  insertions the site table refused
           16    fired[27]       u64  per-callback fire counts
           232   hit[27]         u64  per-callback satisfied counts
           448   sites[4096]     {pc u64, fired u64, hit u64, id u32, pad u32}

Mirrors ``struct __afl_cmp_shm`` in ``afl_shim.c``. A shim that predates
this never writes the magic, so callers fall back to the text files.
"""

from __future__ import annotations

import ctypes
import weakref

import numpy as np

from fuzzer_tool.adapters import libc_shm

CMP_SHM_ENV = "_CMPLOG_COUNTS_SHM"

# Index-matched to __afl_cmp_names in afl_shim.c.
CMP_NAMES = (
    "memcmp", "strcmp", "strncmp", "strcasecmp",
    "strncasecmp", "bcmp", "memchr", "memrchr",
    "memmem", "strstr", "strcasestr", "strpbrk",
    "strspn", "strcspn", "wmemcmp", "wcsncmp",
    "wcscmp", "wcscasecmp", "trace_cmp1", "trace_cmp2",
    "trace_cmp4", "trace_cmp8", "trace_const_cmp1", "trace_const_cmp2",
    "trace_const_cmp4", "trace_const_cmp8", "trace_switch",
)  # fmt: skip

# Must equal __AFL_CMP_SITE_SLOTS / __AFL_CMP_SHM_MAGIC in afl_shim.c.
SITE_SLOTS = 4096
CMP_SHM_MAGIC = int.from_bytes(b"CMPSHM01", "little")

_HEADER = np.dtype([("magic", "<u8"), ("site_dropped", "<u8")])
_COUNTS = np.dtype((np.uint64, len(CMP_NAMES)))
_SITE = np.dtype([("pc", "<u8"), ("fired", "<u8"), ("hit", "<u8"), ("id", "<u4"), ("pad", "<u4")])
_LAYOUT = np.dtype(
    [
        ("header", _HEADER),
        ("fired", _COUNTS),
        ("hit", _COUNTS),
        ("sites", _SITE, (SITE_SLOTS,)),
    ]
)


def _release(addr: int, shm_id: int) -> None:
    libc_shm.shmdt(addr)
    libc_shm.shmctl_rmid(shm_id)


class CmpCounterShm:
    """One counters segment: create, expose views, drain, release."""

    def __init__(self):
        shm_id = libc_shm.shmget(_LAYOUT.itemsize)
        if shm_id is None:
            raise OSError("shmget failed for cmplog counters")

        addr = libc_shm.shmat(shm_id)
        if addr is None:
            libc_shm.shmctl_rmid(shm_id)
            raise OSError("shmat failed for cmplog counters")

        self._id = shm_id
        self._finalizer = weakref.finalize(self, _release, addr, shm_id)
        buf = (ctypes.c_char * _LAYOUT.itemsize).from_address(addr)
        seg = np.frombuffer(buf, dtype=_LAYOUT, count=1)
        self.header = seg["header"]
        self.fired = seg["fired"][0]
        self.hit = seg["hit"][0]
        self.sites = seg["sites"][0]

    @property
    def env_id(self) -> str:
        return str(self._id)

    @property
    def attached(self) -> bool:
        """True once a shim attached and took over counting."""
        return int(self.header["magic"][0]) == CMP_SHM_MAGIC

    def mark_attached(self) -> None:
        """What the shim writes on attach (tests drive the segment by hand)."""
        self.header["magic"] = CMP_SHM_MAGIC

    def drain_counts(self) -> list[tuple[str, int, int]]:
        """``(callback, fired, hit)`` for every callback that fired; zeroes them."""
        idx = np.flatnonzero(self.fired)
        fired = self.fired[idx].tolist()
        hit = self.hit[idx].tolist()
        self.fired[idx] = 0
        self.hit[idx] = 0
        return [(CMP_NAMES[i], f, h) for i, f, h in zip(idx.tolist(), fired, hit, strict=True)]

    def drain_sites(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(slots, fired, hit)`` of sites that fired; zeroes counters, keeps keys."""
        sites = self.sites
        slots = np.flatnonzero(sites["fired"])
        fired = sites["fired"][slots].astype(np.int64)
        hit = sites["hit"][slots].astype(np.int64)
        sites["fired"][slots] = 0
        sites["hit"][slots] = 0
        return slots, fired, hit

    def site_key(self, slot: int) -> tuple[str, int]:
        """``(callback, pc)`` of a slot; stable once the shim inserted it."""
        e = self.sites[slot]
        cb = int(e["id"])
        name = CMP_NAMES[cb] if cb < len(CMP_NAMES) else "?"
        return name, int(e["pc"])

    def take_dropped(self) -> int:
        n = int(self.header["site_dropped"][0])
        self.header["site_dropped"] = 0
        return n

    def close(self) -> None:
        self._finalizer()
