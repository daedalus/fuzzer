"""Shared ELF parsing utilities for sancov counter discovery and analysis.

Consolidates the duplicated ELF parsing logic from shim_factory.py
and fuzzer.py (PtraceCoverage). The embedded _PERSISTENT_LOADER script
in persistent_loader.py retains its own copy since it runs in a
separate Python process.
"""

import logging
import struct
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ── Minimal pure-Python x86-64 decoder (no capstone dependency) ─────────
# Handles only the instruction patterns needed by extract_div_constants:
# DIV/IDIV, MOV r,imm, XOR r,r, CMP r,imm / CMP r,r, LEA, and control flow.

# Instruction type IDs (arbitrary constants, only used internally)
_INS_MOV = 1
_INS_MOVABS = 2
_INS_XOR = 3
_INS_CMP = 4
_INS_LEA = 5
_INS_DIV = 6
_INS_IDIV = 7
_INS_CALL = 8
_INS_JMP = 9
_INS_RET = 10
_INS_JCC = 11
_INS_OTHER = 99

# Operand types
_OP_REG = 1
_OP_IMM = 2
_OP_MEM = 3

# Control-flow groups
_GRP_CALL = 1
_GRP_JUMP = 2
_GRP_RET = 3

# x86-64 register names by 3-bit encoding (extended by REX.B to 4-bit)
_REG_NAMES = [
    "rax",
    "rcx",
    "rdx",
    "rbx",
    "rsp",
    "rbp",
    "rsi",
    "rdi",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
]


@dataclass
class _Operand:
    """Decoded x86-64 operand."""

    type: int  # _OP_REG, _OP_IMM, _OP_MEM
    reg: int = 0  # register encoding (0-15)
    imm: int = 0  # immediate value
    size: int = 4  # operand size in bytes
    # Memory operand fields
    base: int = -1
    index: int = -1
    scale: int = 1
    disp: int = 0


@dataclass
class _DisasmInsn:
    """Decoded x86-64 instruction (capstone-compatible interface)."""

    address: int = 0
    length: int = 0
    insn_id: int = _INS_OTHER
    bytes: bytes = b""
    operands: list = field(default_factory=list)
    groups: set = field(default_factory=set)
    op_str: str = ""
    _regs_read: set = field(default_factory=set)
    _regs_write: set = field(default_factory=set)

    def regs_access(self):
        return self._regs_read, self._regs_write


def _reg_base_pure(reg_id: int) -> str | None:
    """Derive canonical base name for x86 register (no capstone needed).

    All widths of the same register (al/ax/eax/rax) map to the same name.
    """
    if 0 <= reg_id < len(_REG_NAMES):
        return _REG_NAMES[reg_id]
    return None


def _decode_x86_64(text: bytes, base_addr: int):
    """Minimal x86-64 instruction decoder — yields _DisasmInsn objects.

    Handles only the patterns needed by extract_div_constants.
    Unrecognized opcodes are yielded as _INS_OTHER with length=1.
    """
    pc = 0
    n = len(text)

    while pc < n:
        start = pc
        addr = base_addr + pc

        # ── REX prefix (optional) ──
        rex = 0
        rex_w = False
        rex_r = False
        rex_b = False
        if pc < n and 0x40 <= text[pc] <= 0x4F:
            rex = text[pc]
            rex_w = bool(rex & 0x08)
            rex_r = bool(rex & 0x04)
            rex_b = bool(rex & 0x01)
            pc += 1

        if pc >= n:
            yield _DisasmInsn(
                address=addr, length=1, insn_id=_INS_OTHER, bytes=text[start : start + 1]
            )
            return

        opbyte = text[pc]
        pc += 1

        # ── Helper: decode ModR/M + SIB + displacement ──
        def _decode_modrm(_rex_r=rex_r, _rex_b=rex_b):
            """Returns (mod, reg, rm, bytes_consumed, has_sib, sib_byte)."""
            nonlocal pc
            if pc >= n:
                return None
            mrm = text[pc]
            pc += 1
            mod = (mrm >> 6) & 3
            reg = ((mrm >> 3) & 7) | (0x8 if _rex_r else 0)
            rm_raw = mrm & 7
            rm = rm_raw | (0x8 if _rex_b else 0)

            has_sib = False
            sib_byte = 0
            # SIB present when mod≠11 and rm_raw==4 (RSP-based)
            if mod != 3 and rm_raw == 4 and pc < n:
                sib_byte = text[pc]
                pc += 1
                has_sib = True

            # Displacement
            if mod == 1:
                if pc < n:
                    pc += 1  # disp8
            elif mod == 2:
                if pc + 4 <= n:
                    pc += 4  # disp32
            elif mod == 0:
                if has_sib:
                    base_raw = sib_byte & 7
                    if base_raw == 5 and pc + 4 <= n:  # [disp32 + index*scale]
                        pc += 4
                elif rm_raw == 5:  # [disp32]
                    if pc + 4 <= n:
                        pc += 4

            return mod, reg, rm, has_sib, sib_byte

        insn = _DisasmInsn(address=addr, bytes=text[start:pc])

        # ── Single-byte opcodes ──

        # MOV r32, imm32  (B8+rd)
        if 0xB8 <= opbyte <= 0xBF:
            rd = (opbyte - 0xB8) | (0x8 if rex_b else 0)
            if rex_w:
                # MOV r64, imm64
                if pc + 8 <= n:
                    imm = struct.unpack_from("<q", text, pc)[0]
                    pc += 8
                else:
                    imm = 0
                insn.insn_id = _INS_MOV
                insn.operands = [_Operand(_OP_REG, rd, size=8), _Operand(_OP_IMM, imm=imm, size=8)]
                insn._regs_write = {rd}
            else:
                # MOV r32, imm32
                if pc + 4 <= n:
                    imm = struct.unpack_from("<i", text, pc)[0]
                    pc += 4
                else:
                    imm = 0
                insn.insn_id = _INS_MOV
                insn.operands = [_Operand(_OP_REG, rd, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
                insn._regs_write = {rd}
            insn.length = pc - start
            insn.bytes = text[start:pc]
            yield insn
            continue

        # RET
        if opbyte in (0xC3, 0xCB):
            insn.insn_id = _INS_RET
            insn.groups = {_GRP_RET}
            insn.length = pc - start
            yield insn
            continue

        # JMP rel8
        if opbyte == 0xEB:
            if pc < n:
                off = struct.unpack_from("<b", text, pc)[0]
                pc += 1
            else:
                off = 0
            insn.insn_id = _INS_JMP
            insn.groups = {_GRP_JUMP}
            insn.op_str = f"0x{addr + pc + off:x}"
            insn.length = pc - start
            yield insn
            continue

        # JMP rel32
        if opbyte == 0xE9:
            if pc + 4 <= n:
                off = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                off = 0
            insn.insn_id = _INS_JMP
            insn.groups = {_GRP_JUMP}
            insn.op_str = f"0x{addr + pc + off:x}"
            insn.length = pc - start
            yield insn
            continue

        # CALL rel32
        if opbyte == 0xE8:
            if pc + 4 <= n:
                off = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                off = 0
            insn.insn_id = _INS_CALL
            insn.groups = {_GRP_CALL}
            insn.op_str = f"0x{addr + pc + off:x}"
            insn.length = pc - start
            yield insn
            continue

        # Jcc rel8 (70-7F)
        if 0x70 <= opbyte <= 0x7F:
            if pc < n:
                off = struct.unpack_from("<b", text, pc)[0]
                pc += 1
            else:
                off = 0
            insn.insn_id = _INS_JCC
            insn.groups = {_GRP_JUMP}
            insn.op_str = f"0x{addr + pc + off:x}"
            insn.length = pc - start
            yield insn
            continue

        # ── Two-byte opcodes (0x0F prefix) ──
        if opbyte == 0x0F and pc < n:
            op2 = text[pc]
            pc += 1

            # Jcc rel32 (0F 80-8F)
            if 0x80 <= op2 <= 0x8F:
                if pc + 4 <= n:
                    off = struct.unpack_from("<i", text, pc)[0]
                    pc += 4
                else:
                    off = 0
                insn.insn_id = _INS_JCC
                insn.groups = {_GRP_JUMP}
                insn.op_str = f"0x{addr + pc + off:x}"
                insn.length = pc - start
                yield insn
                continue

            # Unknown two-byte opcode — skip
            insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # ── Multi-byte opcodes with ModR/M ──

        # F7 — GRP3 (DIV/IDIV/NOT/NEG/MUL/IMUL)
        if opbyte == 0xF7:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg_ext, rm, has_sib, sib_byte = result
            if mod == 3:  # register operand
                if reg_ext & 7 == 6:  # DIV
                    insn.insn_id = _INS_DIV
                    insn.operands = [_Operand(_OP_REG, rm, size=4)]
                    insn._regs_read = {0, 2, rm}  # EAX, EDX, rm
                    insn._regs_write = {0, 2}  # EAX, EDX
                elif reg_ext & 7 == 7:  # IDIV
                    insn.insn_id = _INS_IDIV
                    insn.operands = [_Operand(_OP_REG, rm, size=4)]
                    insn._regs_read = {0, 2, rm}
                    insn._regs_write = {0, 2}
                else:
                    insn.insn_id = _INS_OTHER
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 81 — GRP1 r/m, imm32
        if opbyte == 0x81:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg_ext, rm, has_sib, sib_byte = result
            # Read imm32
            if pc + 4 <= n:
                imm = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                imm = 0
            if mod == 3 and (reg_ext & 7) == 7:  # CMP r/m32, imm32
                insn.insn_id = _INS_CMP
                insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
                insn._regs_read = {rm}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 83 — GRP1 r/m, imm8 (sign-extended)
        if opbyte == 0x83:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg_ext, rm, has_sib, sib_byte = result
            # Read imm8 (sign-extended to 32-bit)
            if pc < n:
                imm = struct.unpack_from("<b", text, pc)[0]
                pc += 1
            else:
                imm = 0
            if mod == 3 and (reg_ext & 7) == 7:  # CMP r/m32, imm8
                insn.insn_id = _INS_CMP
                insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
                insn._regs_read = {rm}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 39 / 3B — CMP r/m, r / CMP r, r/m
        if opbyte in (0x39, 0x3B):
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:  # register-register
                insn.insn_id = _INS_CMP
                if opbyte == 0x39:
                    # CMP r/m32, r32 — reg is source, rm is destination (read)
                    insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
                else:
                    # CMP r32, r/m32 — reg is destination (read), rm is source (read)
                    insn.operands = [_Operand(_OP_REG, reg, size=4), _Operand(_OP_REG, rm, size=4)]
                insn._regs_read = {reg, rm}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 33 / 31 — XOR r, r/m / XOR r/m, r
        if opbyte in (0x33, 0x31):
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:  # register-register
                insn.insn_id = _INS_XOR
                if opbyte == 0x33:
                    # XOR r32, r/m32 — reg is destination (written)
                    insn.operands = [_Operand(_OP_REG, reg, size=4), _Operand(_OP_REG, rm, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {reg}
                else:
                    # XOR r/m32, r32 — rm is destination (written)
                    insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {rm}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # C7 /0 — MOV r/m32, imm32
        if opbyte == 0xC7:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg_ext, rm, has_sib, sib_byte = result
            if pc + 4 <= n:
                imm = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                imm = 0
            if mod == 3 and (reg_ext & 7) == 0:  # MOV r/m32, imm32
                insn.insn_id = _INS_MOV
                insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
                insn._regs_write = {rm}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 8D — LEA r, m
        if opbyte == 0x8D:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod != 3:  # memory operand
                insn.insn_id = _INS_LEA
                mem_op = _Operand(_OP_MEM, size=8)
                mem_op.base = rm if mod != 3 else -1
                insn.operands = [_Operand(_OP_REG, reg, size=8), mem_op]
                insn._regs_write = {reg}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # FF — various (CALL r/m, JMP r/m, etc.)
        if opbyte == 0xFF:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg_ext, rm, has_sib, sib_byte = result
            ext = reg_ext & 7
            if ext == 2:  # CALL r/m
                insn.insn_id = _INS_CALL
                insn.groups = {_GRP_CALL}
                if mod == 3:
                    insn.operands = [_Operand(_OP_REG, rm, size=8)]
                    insn._regs_read = {rm}
            elif ext == 4:  # JMP r/m
                insn.insn_id = _INS_JMP
                insn.groups = {_GRP_JUMP}
                if mod == 3:
                    insn.operands = [_Operand(_OP_REG, rm, size=8)]
                    insn._regs_read = {rm}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # Unrecognized — skip 1 byte
        insn.insn_id = _INS_OTHER
        insn.length = 1
        yield insn


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
    md.detail = True
    cond_branches = 0
    for insn in md.disasm(text_data, text_vaddr):
        if X86_GRP_JUMP in insn.groups:
            is_long_jcc = (
                insn.bytes[0] == 0x0F and len(insn.bytes) >= 2 and (insn.bytes[1] & 0xF0) == 0x80
            )
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
            capture_output=True,
            timeout=30,
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
            capture_output=True,
            timeout=10,
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


def extract_capstone_constants(target: str) -> list[bytes]:
    """Extract compile-time constants from disassembly via Capstone.

    Disassembles .text and collects immediate operands from comparison,
    move, and test instructions (CMP, MOV, TEST, AND, OR, XOR, SUB,
    ADD with immediate). Also extracts SIMD pattern-match constants
    from PCMP{E,I}STR{I,M} and PABSB/W/D instructions.

    These constants are compile-time magic bytes, pattern strings, and
    boundary values that the code compares against — exactly what the
    fuzzer's dictionary should contain. Unlike .rodata string extraction,
    this catches:

      - Inlined memcmp constants folded into integer immediates
        (e.g. ``cmp rax, 0x0A1A0A0D0A474E89`` → "\\x89PNG\\r\\n\\x1a\\n")
      - SIMD comparison vectors (e.g. PCMPEQB with constant operand)
      - Bitmask / flag values used in test/and/or instructions

    Returns:
        List of unique byte values (deduplicated, truncated to 256 entries).
    """
    try:
        from capstone import CS_ARCH_X86, CS_MODE_64, Cs
        from capstone.x86_const import (
            X86_INS_ADD,
            X86_INS_AND,
            X86_INS_CMP,
            X86_INS_CMPXCHG,
            X86_INS_OR,
            X86_INS_SUB,
            X86_INS_TEST,
            X86_INS_XOR,
            X86_OP_IMM,
        )
    except ImportError:
        return []

    try:
        with open(target, "rb") as f:
            elf = f.read()
    except OSError:
        return []

    if len(elf) < 64 or elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        return []

    # Find .text section
    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]
    if e_shnum == 0 or e_shstrndx >= e_shnum:
        return []

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
        return []

    # Instructions whose immediate operands are likely comparison constants.
    # MOV is excluded because most immediates it loads are addresses/offsets.
    TARGET_INSNS = {
        X86_INS_CMP,
        X86_INS_TEST,
        X86_INS_AND,
        X86_INS_OR,
        X86_INS_XOR,
        X86_INS_SUB,
        X86_INS_ADD,
        X86_INS_CMPXCHG,
    }

    constants: set[bytes] = set()

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True

    for insn in md.disasm(text_data, text_vaddr):
        has_imm = False
        imm_value = 0
        imm_size = 0

        for op in insn.operands:
            if op.type == X86_OP_IMM:
                has_imm = True
                imm_value = op.imm
                # Determine size from the operand's access size
                # Capstone provides op.size in bytes for some operands
                imm_size = getattr(op, "size", 0) or _guess_imm_width(imm_value)
                break

        if not has_imm:
            continue

        # Skip small/noise immediates
        if imm_size <= 0 or imm_size > 8:
            continue

        # Filter out uninteresting values
        if _is_noise_immediate(imm_value, imm_size):
            continue

        if insn.id in TARGET_INSNS:
            # Pack as little-endian bytes of the operand width
            unsigned = imm_value & ((1 << (imm_size * 8)) - 1)
            packed = unsigned.to_bytes(imm_size, "little")
            if len(packed) >= 2:  # skip single-byte constants (too noisy)
                _maybe_add_constant(constants, packed)

            # Also add sub-words (2-byte and 4-byte slices) for patterns
            # that contain embedded ASCII
            if len(packed) > 4:
                _maybe_add_constant(constants, packed[:4])
                _maybe_add_constant(constants, packed[4:])
            if len(packed) > 2:
                _maybe_add_constant(constants, packed[:2])
                _maybe_add_constant(constants, packed[2:4] if len(packed) >= 4 else b"")

    # Cap at 256 entries to bound dictionary size
    result = list(constants)[:256]
    if result:
        log.info("Capstone constants: extracted %d values from %s", len(result), target)
    return result


def _guess_imm_width(value: int) -> int:
    """Guess the byte width of an immediate from its value range."""
    if value < 0:
        value = -value
    if value <= 0xFF:
        return 1
    if value <= 0xFFFF:
        return 2
    if value <= 0xFFFFFFFF:
        return 4
    return 8


def _is_noise_immediate(value: int, size: int) -> bool:
    """Return True if *value* is likely uninteresting.

    Filters obvious noise: zero, tiny counters, small negatives.
    Conservative by design — false negatives (keeping noise) are bounded by
    the 256-entry cap and do less harm than false positives (discarding
    legitimate comparison constants like 0xFFFF0000 or 0x400000).
    """
    if value == 0:
        return True
    unsigned = value & ((1 << (size * 8)) - 1)
    # Small positive counters/indices
    if 0 < unsigned < 128:
        return True
    # Small negative in two's complement (multi-byte only — single-byte
    # values 128-255 include legit constants like 0x89 PNG, 0xFF JPEG)
    if size > 1:
        upper = 1 << (size * 8)
        if upper - 128 <= unsigned < upper:
            return True  # -1 through -128
    # User-space addresses on 64-bit (conservative: page-aligned AND high bit set)
    return size == 8 and unsigned > 0x7FFFFFFFFFFF and unsigned % 0x1000 == 0


def _maybe_add_constant(constants: set[bytes], data: bytes):
    """Add *data* to *constants* if it looks like a useful dictionary token."""
    if not data or len(data) < 2:
        return
    # Skip all-zeros, all-ones, all-0xFF
    if data == b"\x00" * len(data):
        return
    if data == b"\xff" * len(data):
        return
    if data == b"\x01" * len(data):
        return
    # Skip if already present
    if data in constants:
        return
    constants.add(data)


def estimate_map_size(target: str) -> int:
    """Estimate optimal number of hash table entries from sancov guard count
    or branch density.  Returns the number of entries (AFL_MAP_SIZE convention).
    Multiply by 8 to get SHM bytes.

    Priority:
    1. If sancov counter section exists (Clang -fsanitize-coverage), use
       guard count directly — this is the exact number of instrumented edges.
    2. Fall back to branch_density × .text_size estimation.

    Args:
        target: Path to ELF binary.

    Returns:
        Recommended number of entries (int), defaults to 8192 on failure.
    """
    DEFAULT = 8192

    # Try sancov guard count first — most accurate for instrumented binaries
    offsets = parse_sancov_offsets(target)
    if offsets:
        start, stop = offsets
        if stop > start:
            # Each guard is a uint32_t; guards are 4 bytes apart
            guard_count = (stop - start) // 4
            if guard_count > 0:
                map_size = _next_power_of_2(guard_count)
                return max(DEFAULT, min(131072, map_size))

    # Fall back to branch density estimation
    bd = branch_density(target)
    ts = _text_size(target)
    if bd is None or ts is None or ts == 0:
        return DEFAULT

    branches = bd * (ts / 1024)
    estimated_edges = int(branches * 2)  # 2 edges per branch
    map_size = _next_power_of_2(max(estimated_edges, DEFAULT))
    return min(131072, map_size)


def _reg_base(md, reg_id: int) -> str | None:
    """Derive a canonical base name for an x86 register.

    All variants of the same physical register (al/ax/eax/rax, r8b/r8w/r8d/r8)
    map to the same base ('ax', 'r8', etc.).  Used for alias detection in
    the backward-scan heuristic.
    """
    name = md.reg_name(reg_id)
    if not name:
        return None
    name = name.lower()
    # Extended registers r8-r15 in various widths
    if len(name) >= 2 and name[0] == "r":
        try:
            _ = int(name[1])  # is it r8, r9, ... r15?
            # Strip trailing width suffix: r8b → r8, r8w → r8, r8d → r8
            if name[-1] in ("b", "w", "d") and len(name) >= 3:
                return name[:-1]
            return name  # already the base (r8, r9, ..., r15)
        except ValueError:
            pass
    # Strip 'e' or 'r' prefix: eax → ax, rax → ax
    if name[0] in ("e", "r") and len(name) > 2:
        return name[1:]
    # Low-byte registers: al, bl, cl, dl → ax, bx, cx, dx
    if len(name) == 2 and name[1] == "l" and name[0] in "abcd":
        return name[0] + "x"
    # High-byte registers: ah, bh, ch, dh → ax, bx, cx, dx
    if len(name) == 2 and name[1] == "h" and name[0] in "abcd":
        return name[0] + "x"
    return name


def _extract_imm(insn) -> int | None:
    """If *insn* loads a constant into a register, return the constant.

    Handles ``mov reg, imm``, ``movabs reg, imm``, and simple
    ``lea reg, [disp]`` (no base/index).
    Works with both _DisasmInsn (pure decoder) and capstone insn objects.
    """
    # Pure decoder path
    if isinstance(insn, _DisasmInsn):
        if (
            insn.insn_id == _INS_MOV
            and len(insn.operands) >= 2
            and insn.operands[1].type == _OP_IMM
        ):
            return insn.operands[1].imm
        if insn.insn_id == _INS_LEA and len(insn.operands) >= 2:
            mem = insn.operands[1]
            if mem.type == _OP_MEM and mem.base < 0 and mem.index < 0 and mem.disp != 0:
                return mem.disp
        return None

    # Capstone path
    try:
        from capstone.x86_const import (
            X86_INS_LEA,
            X86_INS_MOV,
            X86_INS_MOVABS,
            X86_OP_IMM,
        )
    except ImportError:
        return None

    if insn.id == X86_INS_MOV and len(insn.operands) >= 2 and insn.operands[1].type == X86_OP_IMM:
        return insn.operands[1].imm

    if (
        insn.id == X86_INS_MOVABS
        and len(insn.operands) >= 2
        and insn.operands[1].type == X86_OP_IMM
    ):
        return insn.operands[1].imm

    if insn.id == X86_INS_LEA and len(insn.operands) >= 2:
        mem = insn.operands[1].mem
        if mem.base == 0 and mem.index == 0 and mem.disp != 0:
            return mem.disp

    return None


def _is_ctrl_flow(insn) -> bool:
    """Return True if *insn* changes control flow (call, jmp, ret, jcc).

    Works with both _DisasmInsn (pure decoder) and capstone insn objects.
    """
    # Pure decoder path
    if isinstance(insn, _DisasmInsn):
        return bool(insn.groups & {_GRP_CALL, _GRP_JUMP, _GRP_RET})

    # Capstone path
    try:
        from capstone.x86_const import (
            X86_GRP_CALL,
            X86_GRP_JUMP,
            X86_GRP_RET,
        )
    except ImportError:
        return False
    return any(g in (X86_GRP_CALL, X86_GRP_JUMP, X86_GRP_RET) for g in insn.groups)


def extract_div_constants(target: str) -> tuple[dict[int, int], set[int]]:
    """Find DIV/IDIV instructions and extract divisor and modulus constants.

    Two extraction methods:

    1. **Backward divisor extraction** — determines the divisor for a DIV
       instruction by scanning backward up to 50 instructions for a ``mov``
       that loads a constant into the DIV's operand register (handles both
       immediate and register operands).

    2. **Forward modulus extraction** — after a DIV places the remainder in
       EDX, subsequent ``cmp edx, …`` instructions are mapped to the same
       divisor.  This lets trace-mode constraint solving recognise which
       comparison is checking ``x % N == expected``.

    Returns:
        ``(div_map, weak_mod_pcs)`` where:
        - ``div_map`` maps a PC (DIV or CMP address) to a known constant divisor.
        - ``weak_mod_pcs`` contains CMP addresses that reference the DIV
          remainder but whose divisor could NOT be determined statically
          (variable divisor at runtime).  The solver can still try the
          heuristic common-divisor set for these.
    """
    try:
        with open(target, "rb") as f:
            elf = f.read()
    except OSError:
        return {}, set()

    if len(elf) < 64 or elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        return {}, set()

    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]
    if e_shnum == 0 or e_shstrndx >= e_shnum:
        return {}, set()

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

    if not text_data:
        return {}, set()

    # Try pure decoder first (no external dependencies)
    try:
        result = _extract_div_pure(text_data, text_vaddr)
        if result != ({}, set()):
            return result
    except Exception:
        pass

    # Fall back to capstone if available
    try:
        return _extract_div_capstone(text_data, text_vaddr)
    except Exception:
        return {}, set()


def _extract_div_pure(text_data: bytes, text_vaddr: int) -> tuple[dict[int, int], set[int]]:
    """extract_div_constants using the pure-Python decoder (no capstone)."""
    # Register alias map — in our pure encoding, each register ID IS its own alias
    # (all widths of the same register map to the same ID 0-15)
    reg_alias: dict[int, set[int]] = {i: {i} for i in range(16)}

    # DX family: register 2 (rdx/edx/dx/dl all map to 2)
    _dx_family: set[int] = {2}

    MAX_BACKWARD = 50
    recent: list[tuple] = []

    div_map: dict[int, int] = {}
    _known_divs: dict[int, int] = {}
    weak_mod_pcs: set[int] = set()

    for insn in _decode_x86_64(text_data, text_vaddr):
        regs_read, regs_write = insn.regs_access()
        recent.append((insn, set(regs_write)))
        if len(recent) > MAX_BACKWARD:
            recent.pop(0)

        # ── DIV/IDIV detection ──
        if insn.insn_id in (_INS_DIV, _INS_IDIV):
            op = insn.operands[0] if insn.operands else None
            if op is None:
                continue

            divisor: int | None = None

            # Method 1: immediate operand
            if op.type == _OP_IMM:
                if 0 < op.imm <= 0xFFFFFFFF:
                    divisor = op.imm

            # Method 2: register operand with backward scan
            elif op.type == _OP_REG:
                div_reg = op.reg
                div_reg_family = reg_alias.get(div_reg, {div_reg})
                for prev_insn, prev_writes in reversed(recent[:-1]):
                    if _is_ctrl_flow(prev_insn):
                        break
                    if prev_writes & div_reg_family:
                        candidate = _extract_imm(prev_insn)
                        if candidate is not None and 0 < candidate <= 0xFFFFFFFF:
                            divisor = candidate
                        break

            if divisor is not None:
                div_map[insn.address] = divisor
                _known_divs[insn.address] = divisor
            continue

        # ── Forward modulus extraction ──
        if (
            insn.insn_id == _INS_CMP
            and len(insn.operands) >= 2
            and any(op.type == _OP_REG and op.reg in _dx_family for op in insn.operands)
        ):
            for prev_insn, _ in reversed(recent[:-1]):
                if prev_insn.insn_id in (_INS_DIV, _INS_IDIV):
                    d = _known_divs.get(prev_insn.address)
                    if d is not None:
                        div_map[insn.address] = d
                    else:
                        weak_mod_pcs.add(insn.address)
                    break

    if div_map or weak_mod_pcs:
        log.info(
            "elf: found %d DIV/IDIV mappings, %d weak modulus PCs (pure decoder)",
            len(div_map),
            len(weak_mod_pcs),
        )
    return div_map, weak_mod_pcs


def _extract_div_capstone(text_data: bytes, text_vaddr: int) -> tuple[dict[int, int], set[int]]:
    """extract_div_constants using capstone disassembly (fallback)."""
    from capstone import CS_ARCH_X86, CS_MODE_64, Cs
    from capstone.x86_const import (
        X86_INS_CMP,
        X86_INS_DIV,
        X86_INS_IDIV,
        X86_OP_IMM,
        X86_OP_REG,
    )

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True

    # Build register alias map
    reg_alias: dict[int, set[int]] = {}
    group_to_ids: dict[str, set[int]] = {}
    for insn in md.disasm(text_data, text_vaddr):
        for op in insn.operands:
            if op.type == X86_OP_REG:
                base = _reg_base(md, op.reg)
                if base:
                    group_to_ids.setdefault(base, set()).add(op.reg)
        break
    for rid in range(1, 200):
        name = md.reg_name(rid)
        if name:
            base = _reg_base(md, rid)
            if base:
                group_to_ids.setdefault(base, set()).add(rid)
    for _base, members in group_to_ids.items():
        for rid in members:
            reg_alias[rid] = members

    # DX family
    _dx_family: set[int] = set()
    for rid in range(1, 200):
        b = _reg_base(md, rid)
        if b == "dx":
            _dx_family.add(rid)

    MAX_BACKWARD = 50
    recent: list[tuple] = []

    div_map: dict[int, int] = {}
    _known_divs: dict[int, int] = {}
    weak_mod_pcs: set[int] = set()

    for insn in md.disasm(text_data, text_vaddr):
        regs_read, regs_write = insn.regs_access()
        recent.append((insn, set(regs_write)))
        if len(recent) > MAX_BACKWARD:
            recent.pop(0)

        if insn.id in (X86_INS_DIV, X86_INS_IDIV):
            op = insn.operands[0] if insn.operands else None
            if op is None:
                continue
            divisor: int | None = None
            if op.type == X86_OP_IMM:
                if 0 < op.imm <= 0xFFFFFFFF:
                    divisor = op.imm
            elif op.type == X86_OP_REG:
                div_reg = op.reg
                div_reg_family = reg_alias.get(div_reg, {div_reg})
                for prev_insn, prev_writes in reversed(recent[:-1]):
                    if _is_ctrl_flow(prev_insn):
                        break
                    if prev_writes & div_reg_family:
                        candidate = _extract_imm(prev_insn)
                        if candidate is not None and 0 < candidate <= 0xFFFFFFFF:
                            divisor = candidate
                        break
            if divisor is not None:
                div_map[insn.address] = divisor
                _known_divs[insn.address] = divisor
            continue

        if (
            insn.id == X86_INS_CMP
            and len(insn.operands) >= 2
            and any(op.type == X86_OP_REG and op.reg in _dx_family for op in insn.operands)
        ):
            for prev_insn, _ in reversed(recent[:-1]):
                if prev_insn.id in (X86_INS_DIV, X86_INS_IDIV):
                    d = _known_divs.get(prev_insn.address)
                    if d is not None:
                        div_map[insn.address] = d
                    else:
                        weak_mod_pcs.add(insn.address)
                    break

    if div_map or weak_mod_pcs:
        log.info(
            "elf: found %d DIV/IDIV mappings, %d weak modulus PCs (capstone)",
            len(div_map),
            len(weak_mod_pcs),
        )
    return div_map, weak_mod_pcs
