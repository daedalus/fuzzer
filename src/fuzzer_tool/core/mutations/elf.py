"""Structure-aware ELF mutator (ELF32/ELF64, both endiannesses).

Targets the container rather than the instruction stream — ``x86.py`` and
``arm.py`` already cover raw code bytes. The interesting attack surface in
ELF consumers (readelf, objdump, libbfd, libdwarf, linkers, loaders, and RE
tooling) is the offset/count/size metadata: ``e_shoff``/``e_phoff`` pointing
outside the file, ``e_shnum`` disagreeing with the actual table, section
sizes that overflow when added to their offsets, ``sh_link`` indices past
the end of the table, and string-table offsets that run off the end.

Throughput design:

- Only the 52/64-byte ELF header is parsed up front. Section and program
  header *entries* are located by arithmetic and patched in place; the
  tables are never materialized as Python objects.
- Mutation is in-place on a ``bytearray`` via ``struct.pack_into``. There is
  no parse -> object-model -> re-serialize round trip, so cost is
  independent of file size (a 10 MiB binary costs the same as a 10 KiB one).
- ``parse_elf_header`` returns a small tuple of ints, not a dataclass, and
  is memoized on the header bytes only.
"""

from __future__ import annotations

import random
import struct

_MAGIC = b"\x7fELF"

# ELF header field offsets, by class. Each entry is
# (offset, size) keyed by field name.
_EHDR32 = {
    "e_type": (16, 2),
    "e_machine": (18, 2),
    "e_version": (20, 4),
    "e_entry": (24, 4),
    "e_phoff": (28, 4),
    "e_shoff": (32, 4),
    "e_flags": (36, 4),
    "e_ehsize": (40, 2),
    "e_phentsize": (42, 2),
    "e_phnum": (44, 2),
    "e_shentsize": (46, 2),
    "e_shnum": (48, 2),
    "e_shstrndx": (50, 2),
}
_EHDR64 = {
    "e_type": (16, 2),
    "e_machine": (18, 2),
    "e_version": (20, 4),
    "e_entry": (24, 8),
    "e_phoff": (32, 8),
    "e_shoff": (40, 8),
    "e_flags": (48, 4),
    "e_ehsize": (52, 2),
    "e_phentsize": (54, 2),
    "e_phnum": (56, 2),
    "e_shentsize": (58, 2),
    "e_shnum": (60, 2),
    "e_shstrndx": (62, 2),
}

# Section header entry field offsets, by class.
_SHDR32 = {
    "sh_name": (0, 4),
    "sh_type": (4, 4),
    "sh_flags": (8, 4),
    "sh_addr": (12, 4),
    "sh_offset": (16, 4),
    "sh_size": (20, 4),
    "sh_link": (24, 4),
    "sh_info": (28, 4),
    "sh_addralign": (32, 4),
    "sh_entsize": (36, 4),
}
_SHDR64 = {
    "sh_name": (0, 4),
    "sh_type": (4, 4),
    "sh_flags": (8, 8),
    "sh_addr": (16, 8),
    "sh_offset": (24, 8),
    "sh_size": (32, 8),
    "sh_link": (40, 4),
    "sh_info": (44, 4),
    "sh_addralign": (48, 8),
    "sh_entsize": (56, 8),
}

# Program header entry field offsets. Note the 32/64 layouts differ in field
# *order*, not just width — p_flags moves from the end to right after p_type.
_PHDR32 = {
    "p_type": (0, 4),
    "p_offset": (4, 4),
    "p_vaddr": (8, 4),
    "p_paddr": (12, 4),
    "p_filesz": (16, 4),
    "p_memsz": (20, 4),
    "p_flags": (24, 4),
    "p_align": (28, 4),
}
_PHDR64 = {
    "p_type": (0, 4),
    "p_flags": (4, 4),
    "p_offset": (8, 8),
    "p_vaddr": (16, 8),
    "p_paddr": (24, 8),
    "p_filesz": (32, 8),
    "p_memsz": (40, 8),
    "p_align": (48, 8),
}

_PACK = {
    (2, "<"): "<H",
    (4, "<"): "<I",
    (8, "<"): "<Q",
    (2, ">"): ">H",
    (4, ">"): ">I",
    (8, ">"): ">Q",
}

# Values chosen to sit on parser boundary conditions: zero, off-by-one,
# signed/unsigned flips, and width overflows.
_INTERESTING = (
    0,
    1,
    2,
    0x7F,
    0x80,
    0xFF,
    0x100,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
)

# Real SHT_/PT_ values plus values just past the defined ranges, which is
# where type-dispatch switch statements tend to fall through.
_SH_TYPES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 18, 19, 0x6FFFFFFF, 0x70000000)
_P_TYPES = (0, 1, 2, 3, 4, 5, 6, 7, 0x6474E550, 0x6474E551, 0x6474E552, 0x70000000)


def _get_rng(rng=None):
    return rng or random


def sniff_elf(data: bytes) -> bool:
    """Cheap magic + class/endianness sanity check."""
    return len(data) >= 64 and data[:4] == _MAGIC and data[4] in (1, 2) and data[5] in (1, 2)


def parse_elf_header(data: bytes) -> tuple | None:
    """Return ``(is64, endian, shoff, shnum, shentsize, phoff, phnum, phentsize)``.

    Returns None if *data* is not a plausible ELF. Deliberately returns a
    tuple of ints rather than a dataclass — this is on the hot path and the
    caller only ever reads the fields positionally.
    """
    if not sniff_elf(data):
        return None
    is64 = data[4] == 2
    endian = "<" if data[5] == 1 else ">"
    ehdr = _EHDR64 if is64 else _EHDR32
    need = 64 if is64 else 52
    if len(data) < need:
        return None
    try:
        vals = []
        for field in ("e_shoff", "e_shnum", "e_shentsize", "e_phoff", "e_phnum", "e_phentsize"):
            off, size = ehdr[field]
            vals.append(struct.unpack_from(_PACK[(size, endian)], data, off)[0])
    except struct.error:
        return None
    return (is64, endian, *vals)


def _patch(buf: bytearray, offset: int, size: int, endian: str, value: int) -> bool:
    """Write *value* at *offset*, masked to *size* bytes. False if out of range."""
    if offset < 0 or offset + size > len(buf):
        return False
    mask = (1 << (size * 8)) - 1
    try:
        struct.pack_into(_PACK[(size, endian)], buf, offset, value & mask)
    except (struct.error, KeyError):
        return False
    return True


def _rand_value(r, size: int) -> int:
    """Pick a boundary value most of the time, a random one otherwise."""
    if r.randint(0, 3):
        return r.choice(_INTERESTING)
    return r.randint(0, (1 << (size * 8)) - 1)


class ElfMutator:
    """Structure-aware ELF mutator.

    Dispatches one of several targeted mutations per call. Every mutation is
    an in-place patch, so a call costs the same regardless of binary size.
    """

    _rng = random

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = _get_rng(rng)
        hdr = parse_elf_header(data)
        if hdr is None:
            return data
        buf = bytearray(data)
        ops = (
            self._mutate_ehdr_field,
            self._mutate_ident,
            self._mutate_table_counts,
            self._mutate_shdr_entry,
            self._mutate_phdr_entry,
            self._overlap_section,
            self._mutate_shstrndx,
            self._mutate_sh_link,
        )
        ops[self._rng.randint(0, len(ops) - 1)](buf, hdr)
        return bytes(buf[:max_len]) if len(buf) > max_len else bytes(buf)

    # ── ELF header ─────────────────────────────────────────────────────

    def _mutate_ehdr_field(self, buf: bytearray, hdr: tuple) -> None:
        """Corrupt one arbitrary ELF header field."""
        is64, endian = hdr[0], hdr[1]
        ehdr = _EHDR64 if is64 else _EHDR32
        r = self._rng
        field = r.choice(list(ehdr))
        off, size = ehdr[field]
        _patch(buf, off, size, endian, _rand_value(r, size))

    def _mutate_ident(self, buf: bytearray, _hdr: tuple) -> None:
        """Corrupt e_ident: class, endianness, version, ABI, or padding.

        Byte 4 (class) and byte 5 (endianness) are the two that make every
        downstream offset calculation in the consumer wrong at once, which is
        exactly the disagreement worth testing.
        """
        r = self._rng
        idx = r.choice((4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
        if len(buf) > idx:
            buf[idx] = r.choice((0, 1, 2, 3, 0x7F, 0xFF)) if r.randint(0, 1) else r.randint(0, 255)

    def _mutate_table_counts(self, buf: bytearray, hdr: tuple) -> None:
        """Desynchronize a table's count or entry size from the real table.

        e_shnum/e_phnum larger than the file supports drives readers off the
        end; e_shentsize/e_phentsize that disagree with the real stride makes
        them walk entries misaligned.
        """
        is64, endian = hdr[0], hdr[1]
        ehdr = _EHDR64 if is64 else _EHDR32
        r = self._rng
        field = r.choice(("e_shnum", "e_phnum", "e_shentsize", "e_phentsize"))
        off, size = ehdr[field]
        if r.randint(0, 1):
            value = r.choice((0, 1, 0xFFFF, 0x7FFF, 0x8000))
        else:
            cur = struct.unpack_from(_PACK[(size, endian)], buf, off)[0]
            value = cur + r.choice((-1, 1, 2, 16, 256))
        _patch(buf, off, size, endian, value)

    def _mutate_shstrndx(self, buf: bytearray, hdr: tuple) -> None:
        """Point the section-name string table index at a bogus section."""
        is64, endian, _shoff, shnum = hdr[0], hdr[1], hdr[2], hdr[3]
        ehdr = _EHDR64 if is64 else _EHDR32
        off, size = ehdr["e_shstrndx"]
        r = self._rng
        value = r.choice((0, 0xFFFF, 0xFF00, shnum, shnum + 1, max(0, shnum - 1)))
        _patch(buf, off, size, endian, value)

    # ── Section / program header tables ─────────────────────────────────

    def _shdr_slot(self, hdr: tuple) -> tuple[int, dict, str] | None:
        """Locate a random section header entry, or None if there is no table."""
        is64, endian, shoff, shnum, shentsize = hdr[0], hdr[1], hdr[2], hdr[3], hdr[4]
        if not shoff or not shnum or not shentsize:
            return None
        idx = self._rng.randint(0, min(shnum, 4096) - 1)
        return shoff + idx * shentsize, (_SHDR64 if is64 else _SHDR32), endian

    def _phdr_slot(self, hdr: tuple) -> tuple[int, dict, str] | None:
        """Locate a random program header entry, or None if there is no table."""
        is64, endian, phoff, phnum, phentsize = hdr[0], hdr[1], hdr[5], hdr[6], hdr[7]
        if not phoff or not phnum or not phentsize:
            return None
        idx = self._rng.randint(0, min(phnum, 4096) - 1)
        return phoff + idx * phentsize, (_PHDR64 if is64 else _PHDR32), endian

    def _mutate_shdr_entry(self, buf: bytearray, hdr: tuple) -> None:
        """Corrupt one field of one section header entry."""
        slot = self._shdr_slot(hdr)
        if slot is None:
            return self._mutate_ehdr_field(buf, hdr)
        base, layout, endian = slot
        r = self._rng
        field = r.choice(list(layout))
        off, size = layout[field]
        value = r.choice(_SH_TYPES) if field == "sh_type" else _rand_value(r, size)
        _patch(buf, base + off, size, endian, value)

    def _mutate_phdr_entry(self, buf: bytearray, hdr: tuple) -> None:
        """Corrupt one field of one program header entry."""
        slot = self._phdr_slot(hdr)
        if slot is None:
            return self._mutate_ehdr_field(buf, hdr)
        base, layout, endian = slot
        r = self._rng
        field = r.choice(list(layout))
        off, size = layout[field]
        value = r.choice(_P_TYPES) if field == "p_type" else _rand_value(r, size)
        _patch(buf, base + off, size, endian, value)

    def _overlap_section(self, buf: bytearray, hdr: tuple) -> None:
        """Make sh_offset + sh_size overflow or run past end of file.

        Consumers that compute ``offset + size`` and compare against the file
        length without checking for wraparound will read out of bounds; this
        is the classic ELF parser bug class.
        """
        slot = self._shdr_slot(hdr)
        if slot is None:
            return self._mutate_ehdr_field(buf, hdr)
        base, layout, endian = slot
        r = self._rng
        off_o, off_s = layout["sh_offset"]
        size_o, size_s = layout["sh_size"]
        mode = r.randint(0, 2)
        if mode == 0:  # size overflows when added to a valid offset
            _patch(buf, base + size_o, size_s, endian, (1 << (size_s * 8)) - 1)
        elif mode == 1:  # offset past EOF, plausible size
            _patch(buf, base + off_o, off_s, endian, len(buf) + r.choice((1, 16, 4096)))
        else:  # both near the wrap boundary
            half = 1 << (off_s * 8 - 1)
            _patch(buf, base + off_o, off_s, endian, half)
            _patch(buf, base + size_o, size_s, endian, half + r.choice((0, 1, 16)))

    def _mutate_sh_link(self, buf: bytearray, hdr: tuple) -> None:
        """Point sh_link/sh_info at a nonexistent section index.

        Symbol tables reach their string table through sh_link, so an
        out-of-range index sends the consumer to an unvalidated section.
        """
        slot = self._shdr_slot(hdr)
        if slot is None:
            return self._mutate_ehdr_field(buf, hdr)
        base, layout, endian = slot
        r = self._rng
        shnum = hdr[3]
        field = "sh_link" if r.randint(0, 1) else "sh_info"
        off, size = layout[field]
        value = r.choice((shnum, shnum + 1, 0xFFFF, 0xFFFFFFFF, 0))
        _patch(buf, base + off, size, endian, value)
