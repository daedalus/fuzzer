"""Shared ELF parsing utilities for sancov counter discovery.

Consolidates the duplicated ELF parsing logic from shim_factory.py
and fuzzer.py (PtraceCoverage). The embedded _PERSISTENT_LOADER script
in persistent_loader.py retains its own copy since it runs in a
separate Python process.
"""

import logging
import struct

log = logging.getLogger(__name__)


def parse_sancov_offsets(target: str) -> tuple[int, int] | None:
    """Parse ELF to find __start/__stop___sancov_cntrs virtual addresses.

    Args:
        target: Path to ELF binary (shared library or executable).

    Returns:
        Tuple of (start_addr, stop_addr) if found, None otherwise.
    """
    try:
        with open(target, "rb") as f:
            elf = f.read()
        if len(elf) < 64 or elf[:4] != b"\x7fELF":
            return None
        if elf[4] != 2 or elf[5] != 1:  # ELF64, little-endian
            return None
        e_shoff = struct.unpack_from("<Q", elf, 40)[0]
        e_shnum = struct.unpack_from("<H", elf, 60)[0]
        e_shentsize = struct.unpack_from("<H", elf, 58)[0]
        e_shstrndx = struct.unpack_from("<H", elf, 62)[0]
        if e_shnum == 0 or e_shstrndx >= e_shnum:
            return None
        shstr_off = e_shoff + e_shstrndx * e_shentsize
        shstr_offset = struct.unpack_from("<Q", elf, shstr_off + 24)[0]
        symtab_sec = strtab_sec = None
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
            sh_name_idx = struct.unpack_from("<I", elf, sh)[0]
            name = elf[shstr_offset + sh_name_idx : shstr_offset + sh_name_idx + 32].split(b"\x00")[
                0
            ]
            if sh_type == 2:
                symtab_sec = sh
            elif sh_type == 3 and name == b".strtab":
                strtab_sec = sh
        if symtab_sec is None or strtab_sec is None:
            return None
        sym_offset = struct.unpack_from("<Q", elf, symtab_sec + 24)[0]
        sym_size = struct.unpack_from("<Q", elf, symtab_sec + 32)[0]
        sym_entsize = struct.unpack_from("<Q", elf, symtab_sec + 56)[0]
        if sym_entsize == 0:
            return None
        sym_count = sym_size // sym_entsize
        strtab_offset = struct.unpack_from("<Q", elf, strtab_sec + 24)[0]
        start_addr = stop_addr = None
        for i in range(min(sym_count, 10000)):
            sym = sym_offset + i * sym_entsize
            st_value = struct.unpack_from("<Q", elf, sym + 8)[0]
            st_name_idx = struct.unpack_from("<I", elf, sym)[0]
            name = (
                elf[strtab_offset + st_name_idx : strtab_offset + st_name_idx + 64]
                .split(b"\x00")[0]
                .decode(errors="replace")
            )
            if name == "__start___sancov_cntrs" and st_value > 0:
                start_addr = st_value
            elif name == "__stop___sancov_cntrs" and st_value > 0:
                stop_addr = st_value
        if start_addr is not None and stop_addr is not None:
            return (start_addr, stop_addr)
    except Exception as e:
        log.debug("ELF parse failed: %s", e)
    return None


def find_load_segment(elf_data: bytes, vaddr: int) -> tuple[int, int, int] | None:
    """Find the LOAD segment containing vaddr.

    Args:
        elf_data: Raw ELF file contents.
        vaddr: Virtual address to search for.

    Returns:
        Tuple of (segment_vaddr, filesz, memsz) if found, None otherwise.
    """
    if len(elf_data) < 64 or elf_data[:4] != b"\x7fELF":
        return None
    e_phoff = struct.unpack_from("<Q", elf_data, 32)[0]
    e_phentsize = struct.unpack_from("<H", elf_data, 54)[0]
    e_phnum = struct.unpack_from("<H", elf_data, 56)[0]
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from("<I", elf_data, off)[0]
        if p_type == 1:  # PT_LOAD
            p_vaddr = struct.unpack_from("<Q", elf_data, off + 16)[0]
            p_filesz = struct.unpack_from("<Q", elf_data, off + 32)[0]
            p_memsz = struct.unpack_from("<Q", elf_data, off + 40)[0]
            if p_vaddr <= vaddr < p_vaddr + p_memsz:
                return (p_vaddr, p_filesz, p_memsz)
    return None
