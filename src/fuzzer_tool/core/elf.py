"""Shared ELF parsing utilities for sancov counter discovery and analysis.

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


def branch_density(target: str) -> float | None:
    """Compute branch density (conditional branches per KB) of a binary.

    Disassembles the .text section and counts conditional jump instructions
    (Jcc family). Tries Capstone first, falls back to objdump.

    Returns branches per KB of code, or None if analysis fails.

    This is a static metric that predicts fuzzing difficulty:
    - High density → more decision points per KB → harder to saturate
    - Useful for sizing edge bitmaps, estimating saturation, ranking targets

    Args:
        target: Path to ELF binary.

    Returns:
        Branches per KB (float), or None on failure.
    """
    result = _branch_density_capstone(target)
    if result is not None:
        return result
    return _branch_density_objdump(target)


def _branch_density_capstone(target: str) -> float | None:
    """Branch density via Capstone disassembly (preferred)."""
    try:
        from capstone import CS_ARCH_X86, CS_MODE_64, Cs
        from capstone.x86_const import X86_GRP_JUMP
    except ImportError:
        return None

    try:
        with open(target, "rb") as f:
            elf = f.read()
    except OSError:
        return None

    if len(elf) < 64 or elf[:4] != b"\x7fELF":
        return None
    if elf[4] != 2 or elf[5] != 1:
        return None

    # Find .text section
    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]
    if e_shnum == 0 or e_shstrndx >= e_shnum:
        return None

    shstr_off = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from("<Q", elf, shstr_off + 24)[0]

    text_data = None
    text_vaddr = 0
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        if sh + e_shentsize > len(elf):
            break
        sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
        sh_name_idx = struct.unpack_from("<I", elf, sh)[0]
        name = elf[shstr_offset + sh_name_idx : shstr_offset + sh_name_idx + 32].split(b"\x00")[0]
        if sh_type == 1 and name == b".text":
            sh_offset = struct.unpack_from("<Q", elf, sh + 24)[0]
            sh_size = struct.unpack_from("<Q", elf, sh + 32)[0]
            text_vaddr = struct.unpack_from("<Q", elf, sh + 16)[0]
            text_data = elf[sh_offset : sh_offset + sh_size]
            break

    if text_data is None or len(text_data) == 0:
        return None

    # Disassemble and count conditional branches
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    cond_branches = 0
    for insn in md.disasm(text_data, text_vaddr):
        if X86_GRP_JUMP in insn.groups:
            is_long_jcc = (insn.bytes[0] == 0x0F and len(insn.bytes) >= 2
                           and (insn.bytes[1] & 0xF0) == 0x80)
            is_short_jcc = insn.bytes[0] in range(0x70, 0x80)
            if is_long_jcc or is_short_jcc:
                cond_branches += 1

    return (cond_branches / len(text_data)) * 1024


def _branch_density_objdump(target: str) -> float | None:
    """Branch density via objdump (fallback when Capstone unavailable)."""
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["objdump", "-d", "--no-show-raw-insn", "-j", ".text", target],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    output = result.stdout.decode(errors="replace")

    # Count conditional jumps: je, jne, jg, jl, ja, jb, jge, jle, etc.
    cond_pattern = re.compile(
        r"\t(je|jne|jg|jl|ja|jb|jge|jle|jae|jbe|jz|jnz|js|jns|jo|jno|jp|jnp"
        r"|loop|loope|loopne|loopnz|loopz)\b"
    )
    cond_branches = len(cond_pattern.findall(output))

    # Get .text size from readelf
    # readelf -S --wide format (fixed columns):
    #   [Nr] Name  Type  Addr  Off  Size  ES  Flg ...
    # Size is column 5 (0-indexed), Addr is column 3
    try:
        result = subprocess.run(
            ["readelf", "-S", "--wide", target],
            capture_output=True, timeout=10,
        )
        for line in result.stdout.decode(errors="replace").splitlines():
            if ".text" in line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        size = int(parts[5], 16)
                        if size > 0:
                            return (cond_branches / size) * 1024
                    except ValueError:
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _text_size(target: str) -> int | None:
    """Get .text section size in bytes from ELF binary."""
    try:
        with open(target, "rb") as f:
            elf = f.read()
    except OSError:
        return None

    if len(elf) < 64 or elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        return None

    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]
    if e_shnum == 0 or e_shstrndx >= e_shnum:
        return None

    shstr_off = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from("<Q", elf, shstr_off + 24)[0]

    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        if sh + e_shentsize > len(elf):
            break
        sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
        sh_name_idx = struct.unpack_from("<I", elf, sh)[0]
        name = elf[shstr_offset + sh_name_idx : shstr_offset + sh_name_idx + 32].split(b"\x00")[0]
        if sh_type == 1 and name == b".text":
            return struct.unpack_from("<Q", elf, sh + 32)[0]
    return None


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def estimate_map_size(target: str) -> int:
    """Estimate optimal AFL_MAP_SIZE from branch density and .text size.

    Formula:
        estimated_edges = branch_density (branches/KB) × .text_size (KB)
        map_size = next_power_of_2(estimated_edges × 2)  # 2x headroom

    Clamped to [4096, 1048576] (AFL's practical range).

    Args:
        target: Path to ELF binary.

    Returns:
        Recommended map size (int), defaults to 65536 on failure.
    """
    DEFAULT = 65536

    bd = branch_density(target)
    ts = _text_size(target)
    if bd is None or ts is None or ts == 0:
        return DEFAULT

    # Estimated edge count: density (per KB) × size (KB)
    estimated_edges = bd * (ts / 1024)

    # 2x headroom for hash collisions and edge aliasing
    map_size = _next_power_of_2(int(estimated_edges * 2))

    # Clamp to AFL's practical range
    return max(4096, min(1048576, map_size))
