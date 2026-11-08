"""pc-table / control-flow tables the shim dumps to ``$__AFL_SANCOV_OUT``.

Mirrors the dump in ``afl_shim.c`` (``__sanitizer_cov_pcs_init`` /
``__sanitizer_cov_cfs_init``). One file, a run of records::

    [u32 magic][u32 kind][u32 first guard][u32 pad][u64 bias][u64 count][count x u64]

    PCS  words = (pc, flags) pairs, one per block; flags bit 0 = function entry
    CFS  words = per block: pc, successors..., 0, callees... (-1 indirect), 0

Values are absolute; ``bias`` is the module load bias (ELF vaddr = value -
bias). ``first`` is the global guard index of pc-table entry 0, so function
entries index straight into the block bitmap.

The shim creates the file with O_EXCL: this class only reserves the path.
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
import weakref
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

SANCOV_ENV = "__AFL_SANCOV_OUT"
SANCOV_MAGIC = 0x564F4353  # "SCOV" LE
NO_FIRST = 0xFFFFFFFF  # pc-table not paired with a guard region
PC_FLAG_ENTRY = 1  # pc-table flag: block is a function entry

_HEAD = struct.Struct("<IIIIQQ")  # magic kind first pad bias count
_WORD = 8
_DUMP_NAME = "sancov.bin"


class SancovKind(IntEnum):
    """Record kind, as ``enum __afl_sancov_kind`` in the shim."""

    PCS = 1
    CFS = 2


@dataclass(frozen=True, slots=True)
class SancovRecord:
    """One module's table."""

    kind: SancovKind
    first: int
    bias: int
    words: np.ndarray  # uint64, absolute values

    def pcs_vaddrs(self) -> np.ndarray:
        """Block PCs as ELF vaddrs (PCS records)."""
        return self.words[0::2] - np.uint64(self.bias)

    def pcs_flags(self) -> np.ndarray:
        return self.words[1::2]

    def entry_indices(self) -> np.ndarray:
        """Global guard indices of function entries; empty when unpaired."""
        if self.kind != SancovKind.PCS or self.first == NO_FIRST:
            return np.zeros(0, dtype=np.uint32)

        hits = np.flatnonzero(self.pcs_flags() & np.uint64(PC_FLAG_ENTRY))
        return (hits + self.first).astype(np.uint32)


def parse_sancov(buf: bytes) -> list[SancovRecord]:
    """Records in *buf*; stops at the first torn or foreign record.

    Unknown kinds are skipped by their count, so a newer shim's record
    does not end the parse.
    """
    out: list[SancovRecord] = []
    kinds = {k.value for k in SancovKind}
    n = len(buf)
    pos = 0

    while pos + _HEAD.size <= n:
        magic, kind, first, _pad, bias, count = _HEAD.unpack_from(buf, pos)
        body = pos + _HEAD.size
        end = body + count * _WORD
        if magic != SANCOV_MAGIC or end > n:
            break

        pos = end
        if kind not in kinds:
            continue

        words = np.frombuffer(buf, dtype="<u8", count=count, offset=body)
        out.append(SancovRecord(SancovKind(kind), first, bias, words))

    return out


class SancovTables:
    """Reserves the dump path and re-parses the file when it grows."""

    def __init__(self) -> None:
        self._dir = tempfile.mkdtemp(prefix="fuzzer_sancov_")
        self.path = os.path.join(self._dir, _DUMP_NAME)
        self._size = -1
        self._records: list[SancovRecord] = []
        self._finalizer = weakref.finalize(self, shutil.rmtree, self._dir)

    def records(self) -> list[SancovRecord]:
        """Current records; a cheap stat when the file is unchanged."""
        try:
            size = os.stat(self.path).st_size
        except FileNotFoundError:
            return self._records

        if size == self._size:
            return self._records

        with open(self.path, "rb") as fh:
            self._records = parse_sancov(fh.read())
        self._size = size
        return self._records

    def entry_indices(self) -> np.ndarray:
        """Guard indices of every paired function entry."""
        parts = [r.entry_indices() for r in self.records() if r.kind == SancovKind.PCS]
        if not parts:
            return np.zeros(0, dtype=np.uint32)
        return np.concatenate(parts)

    def cleanup(self) -> None:
        self._finalizer()
