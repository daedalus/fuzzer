"""Shared ELF parsing utilities for sancov counter discovery and analysis.

Includes a pure-Python x86-64 instruction decoder that replaces the
optional Capstone dependency for all static analysis tasks:
branch density, constant extraction, DIV detection, and ctrl-flow analysis.

Consolidates the duplicated ELF parsing logic from shim_factory.py
and fuzzer.py (PtraceCoverage). The embedded _PERSISTENT_LOADER script
in persistent_subprocess.py retains its own copy since it runs in a
separate Python process.
"""

import logging
import os
import struct
from dataclasses import dataclass, field
from typing import NamedTuple

log = logging.getLogger(__name__)


# ── Pure-Python x86-64 decoder (no external dependencies) ────────────────
# Handles instruction patterns needed by the fuzzer's static analysis:
# arithmetic (ADD/SUB/AND/OR/XOR/CMP/TEST), moves (MOV/LEA), division
# (DIV/IDIV), control flow (CALL/JMP/JCC/RET), and CMPXCHG.
# Unrecognized opcodes are yielded as _INS_OTHER with length=1.

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
_INS_TEST = 12
_INS_AND = 13
_INS_OR = 14
_INS_SUB = 15
_INS_ADD = 16
_INS_CMPXCHG = 17
_INS_OTHER = 99

# Operand types
_OP_REG = 1
_OP_IMM = 2
_OP_MEM = 3

# Control-flow groups
_GRP_CALL = 1
_GRP_JUMP = 2
_GRP_RET = 3
_GRP_INT = 4

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

    @property
    def size(self):
        """Compatibility alias — capstone uses ``.size``, we use ``.length``."""
        return self.length

    def regs_access(self):
        return self._regs_read, self._regs_write


def _reg_base_pure(reg_id: int) -> str | None:
    """Derive canonical base name for x86 register (no capstone needed).

    All widths of the same register (al/ax/eax/rax) map to the same name.
    """
    if 0 <= reg_id < len(_REG_NAMES):
        return _REG_NAMES[reg_id]
    return None


# Legacy prefixes skipped before REX/opcode (REP, REPNE, opsize, addrsize)
_LEGACY_PREFIXES = (0xF3, 0xF2, 0x66, 0x67)

# REX prefix bits
_REX_W = 0x08
_REX_R = 0x04
_REX_B = 0x01

# Register-register form kinds (see _RR_SPEC)
_RR_ALU = 0  # reads both, writes dst
_RR_CMP = 1  # reads both, flags only
_RR_MOV = 2  # reads src, writes dst


def _imm32(text: bytes, n: int, pc: int) -> tuple[int, int]:
    """Signed imm32 at pc → (imm, new_pc); (0, pc) when truncated."""
    if pc + 4 <= n:
        return struct.unpack_from("<i", text, pc)[0], pc + 4
    return 0, pc


def _imm8s(text: bytes, n: int, pc: int) -> tuple[int, int]:
    """Signed imm8 at pc → (imm, new_pc); (0, pc) when truncated."""
    if pc < n:
        return struct.unpack_from("<b", text, pc)[0], pc + 1
    return 0, pc


def _skip_disp(n: int, pc: int, mod: int, rm_raw: int, sib_base: int) -> int:
    """Skip the ModR/M displacement; bounds-checked so truncation never overruns.

    mod 1 → disp8; mod 2 → disp32; mod 0 → disp32 only for [disp32]
    (rm=5, no SIB) or SIB base=5.
    """
    if mod == 1:
        return pc + 1 if pc < n else pc
    wide = mod == 2 or (mod == 0 and (sib_base if sib_base >= 0 else rm_raw) == 5)
    if wide and pc + 4 <= n:
        return pc + 4
    return pc


def _modrm(text: bytes, n: int, pc: int, rex: int):
    """Decode ModR/M + SIB + disp → (new_pc, mod, reg, rm), or None at EOF."""
    if pc >= n:
        return None
    mrm = text[pc]
    pc += 1
    mod = (mrm >> 6) & 3
    reg = ((mrm >> 3) & 7) | ((rex & _REX_R) << 1)
    rm_raw = mrm & 7
    rm = rm_raw | ((rex & _REX_B) << 3)

    # SIB present when mod≠11 and rm_raw==4 (RSP-based); -1 = no SIB
    sib_base = -1
    if mod != 3 and rm_raw == 4 and pc < n:
        sib_base = text[pc] & 7
        pc += 1

    return _skip_disp(n, pc, mod, rm_raw, sib_base), mod, reg, rm


# Opcode handlers: each fills `insn` and returns the new pc. The decoder
# loop sets insn.length afterwards. Unhandled forms leave _INS_OTHER.


def _x_mov_imm(text, n, pc, start, op, rex, insn):
    """B8+rd — MOV r32, imm32 / MOV r64, imm64 (REX.W)."""
    rd = (op - 0xB8) | ((rex & _REX_B) << 3)
    size = 4
    if rex & _REX_W:
        size = 8
        imm = 0
        if pc + 8 <= n:
            imm = struct.unpack_from("<q", text, pc)[0]
            pc += 8
    else:
        imm, pc = _imm32(text, n, pc)

    insn.insn_id = _INS_MOV
    insn.operands = [_Operand(_OP_REG, rd, size=size), _Operand(_OP_IMM, imm=imm, size=size)]
    insn._regs_write = {rd}
    insn.bytes = text[start:pc]
    return pc


def _x_ret(text, n, pc, start, op, rex, insn):
    """C3/CB — RET."""
    insn.insn_id = _INS_RET
    insn.groups = {_GRP_RET}
    return pc


def _rel_branch(text, n, pc, start, insn, spec):
    """Relative branch: spec = (disp width, insn_id, group); op_str = target."""
    width, insn_id, grp = spec
    off, pc = _imm8s(text, n, pc) if width == 1 else _imm32(text, n, pc)
    insn.insn_id = insn_id
    insn.groups = {grp}
    insn.op_str = f"0x{insn.address + (pc - start) + off:x}"
    return pc


_JCC32 = (4, _INS_JCC, _GRP_JUMP)

# opcode → relative-branch spec (EB/E9 JMP, E8 CALL, 70-7F Jcc rel8)
_BRANCH_SPEC = {
    0xEB: (1, _INS_JMP, _GRP_JUMP),
    0xE9: (4, _INS_JMP, _GRP_JUMP),
    0xE8: (4, _INS_CALL, _GRP_CALL),
    **{op: (1, _INS_JCC, _GRP_JUMP) for op in range(0x70, 0x80)},
}


def _x_branch(text, n, pc, start, op, rex, insn):
    """JMP/CALL/Jcc with a relative displacement."""
    return _rel_branch(text, n, pc, start, insn, _BRANCH_SPEC[op])


def _x_int(text, n, pc, start, op, rex, insn):
    """CD ib — INT imm8."""
    if pc < n:
        pc += 1
    insn.groups = {_GRP_INT}
    return pc


# opcode → (insn_id, writes EAX) for accumulator imm32 forms
_ACC_SPEC = {
    0x05: (_INS_ADD, True),
    0x0D: (_INS_OR, True),
    0x25: (_INS_AND, True),
    0x2D: (_INS_SUB, True),
    0xA9: (_INS_TEST, False),
}


def _x_acc_imm(text, n, pc, start, op, rex, insn):
    """ADD/OR/AND/SUB/TEST EAX, imm32."""
    insn_id, writes = _ACC_SPEC[op]
    imm, pc = _imm32(text, n, pc)
    insn.insn_id = insn_id
    insn.operands = [_Operand(_OP_REG, 0, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
    insn._regs_read = {0}
    if writes:
        insn._regs_write = {0}
    return pc


def _x_two_byte(text, n, pc, start, op, rex, insn):
    """0F xx — Jcc rel32, NOP/CET (ModRM), CMPXCHG; others skipped."""
    if pc >= n:
        return pc
    op2 = text[pc]
    pc += 1

    if 0x80 <= op2 <= 0x8F:
        return _rel_branch(text, n, pc, start, insn, _JCC32)

    if op2 not in (0x1E, 0x1F, 0xB1):
        return pc

    m = _modrm(text, n, pc, rex)
    if m is None:
        return pc
    pc, mod, reg, rm = m

    # CMPXCHG r/m32, r32 (0F B1) — register form only
    if op2 == 0xB1 and mod == 3:
        insn.insn_id = _INS_CMPXCHG
        insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
        insn._regs_read = {reg, rm}
        insn._regs_write = {rm}
    return pc


def _grp3_test(text, n, pc, insn, mod, rm, size):
    """F6/F7 /0 — TEST r/m, imm (imm8 or imm32)."""
    if size == 4:
        imm, pc = _imm32(text, n, pc)
    else:
        imm = text[pc] if pc < n else 0
        pc += 1
    insn.insn_id = _INS_TEST
    if mod == 3:
        insn.operands = [_Operand(_OP_REG, rm, size=size), _Operand(_OP_IMM, imm=imm, size=size)]
    insn._regs_read = {rm}
    return pc


def _grp3_div(insn, ext, mod, rm, size):
    """F6/F7 /6 /7 — DIV / IDIV; implicit EAX:EDX in and out."""
    insn.insn_id = _INS_IDIV if ext == 7 else _INS_DIV
    if mod == 3:  # register divisor
        insn.operands = [_Operand(_OP_REG, rm, size=size)]
        insn._regs_read = {0, 2, rm}
    else:  # memory divisor
        insn.operands = [_Operand(_OP_MEM, size=size)]
        insn._regs_read = {0, 2}
    insn._regs_write = {0, 2}


def _x_grp3(text, n, pc, start, op, rex, insn):
    """F6/F7 — GRP3 (TEST/DIV/IDIV decoded; NOT/NEG/MUL/IMUL → other)."""
    m = _modrm(text, n, pc, rex)
    if m is None:
        return pc
    pc, mod, reg_ext, rm = m

    size = 4 if op == 0xF7 else 1
    ext = reg_ext & 7
    if ext == 0:
        return _grp3_test(text, n, pc, insn, mod, rm, size)
    if ext in (6, 7):
        _grp3_div(insn, ext, mod, rm, size)
    return pc


# GRP1 /ext → (insn_id, writes r/m)
_GRP1_OPS = {
    0: (_INS_ADD, True),
    1: (_INS_OR, True),
    4: (_INS_AND, True),
    5: (_INS_SUB, True),
    7: (_INS_CMP, False),
}


def _x_grp1(text, n, pc, start, op, rex, insn):
    """81 /ext imm32, 83 /ext imm8 — register forms of ADD/OR/AND/SUB/CMP."""
    m = _modrm(text, n, pc, rex)
    if m is None:
        return pc
    pc, mod, reg_ext, rm = m

    imm, pc = _imm32(text, n, pc) if op == 0x81 else _imm8s(text, n, pc)
    spec = _GRP1_OPS.get(reg_ext & 7)
    if mod != 3 or spec is None:
        return pc

    insn.insn_id, writes = spec
    insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
    insn._regs_read = {rm}
    if writes:
        insn._regs_write = {rm}
    return pc


# opcode → (insn_id, r/m is first operand, kind) for register-register forms
_RR_SPEC = {
    0x01: (_INS_ADD, True, _RR_ALU),
    0x03: (_INS_ADD, False, _RR_ALU),
    0x09: (_INS_OR, True, _RR_ALU),
    0x0B: (_INS_OR, False, _RR_ALU),
    0x21: (_INS_AND, True, _RR_ALU),
    0x23: (_INS_AND, False, _RR_ALU),
    0x29: (_INS_SUB, True, _RR_ALU),
    0x2B: (_INS_SUB, False, _RR_ALU),
    0x31: (_INS_XOR, True, _RR_ALU),
    0x33: (_INS_XOR, False, _RR_ALU),
    0x39: (_INS_CMP, True, _RR_CMP),
    0x3B: (_INS_CMP, False, _RR_CMP),
    0x85: (_INS_TEST, True, _RR_CMP),
    0x89: (_INS_MOV, True, _RR_MOV),
    0x8B: (_INS_MOV, False, _RR_MOV),
}


def _x_reg_reg(text, n, pc, start, op, rex, insn):
    """ALU/CMP/TEST/MOV r/m32, r32 and r32, r/m32 — register form only."""
    m = _modrm(text, n, pc, rex)
    if m is None:
        return pc
    pc, mod, reg, rm = m
    if mod != 3:
        return pc

    insn_id, rm_first, kind = _RR_SPEC[op]
    dst, src = (rm, reg) if rm_first else (reg, rm)
    insn.insn_id = insn_id
    insn.operands = [_Operand(_OP_REG, dst, size=4), _Operand(_OP_REG, src, size=4)]
    if kind == _RR_MOV:
        insn._regs_read = {src}
        insn._regs_write = {dst}
        return pc

    insn._regs_read = {reg, rm}
    if kind == _RR_ALU:
        insn._regs_write = {dst}
    return pc


def _x_mov_rm_imm(text, n, pc, start, op, rex, insn):
    """C7 /0 — MOV r/m32, imm32 (register form only)."""
    m = _modrm(text, n, pc, rex)
    if m is None:
        return pc
    pc, mod, reg_ext, rm = m

    imm, pc = _imm32(text, n, pc)
    if mod == 3 and (reg_ext & 7) == 0:
        insn.insn_id = _INS_MOV
        insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
        insn._regs_write = {rm}
    return pc


def _x_lea(text, n, pc, start, op, rex, insn):
    """8D — LEA r, m (memory form only)."""
    m = _modrm(text, n, pc, rex)
    if m is None:
        return pc
    pc, mod, reg, rm = m
    if mod == 3:
        return pc

    insn.insn_id = _INS_LEA
    mem_op = _Operand(_OP_MEM, size=8)
    mem_op.base = rm
    insn.operands = [_Operand(_OP_REG, reg, size=8), mem_op]
    insn._regs_write = {reg}
    return pc


# FF /ext → (insn_id, group) for indirect CALL / JMP
_FF_OPS = {2: (_INS_CALL, _GRP_CALL), 4: (_INS_JMP, _GRP_JUMP)}


def _x_ff(text, n, pc, start, op, rex, insn):
    """FF /2 CALL r/m, FF /4 JMP r/m."""
    m = _modrm(text, n, pc, rex)
    if m is None:
        return pc
    pc, mod, reg_ext, rm = m

    spec = _FF_OPS.get(reg_ext & 7)
    if spec is None:
        return pc
    insn.insn_id, grp = spec
    insn.groups = {grp}
    if mod == 3:
        insn.operands = [_Operand(_OP_REG, rm, size=8)]
        insn._regs_read = {rm}
    return pc


def _build_x86_dispatch() -> list:
    """256-entry opcode → handler table; None = unrecognized (_INS_OTHER)."""
    table: list = [None] * 256
    groups = (
        (range(0xB8, 0xC0), _x_mov_imm),
        ((0xC3, 0xCB), _x_ret),
        (_BRANCH_SPEC, _x_branch),
        ((0xCD,), _x_int),
        (_ACC_SPEC, _x_acc_imm),
        ((0x0F,), _x_two_byte),
        ((0xF6, 0xF7), _x_grp3),
        ((0x81, 0x83), _x_grp1),
        (_RR_SPEC, _x_reg_reg),
        ((0xC7,), _x_mov_rm_imm),
        ((0x8D,), _x_lea),
        ((0xFF,), _x_ff),
    )
    for opcodes, handler in groups:
        for op in opcodes:
            table[op] = handler
    return table


_X86_DISPATCH = _build_x86_dispatch()


def _decode_x86_64(text: bytes, base_addr: int):
    """Pure-Python x86-64 instruction decoder — yields _DisasmInsn objects.

    Handles: MOV, LEA, CMP, TEST, ADD/SUB/AND/OR/XOR, CMPXCHG, DIV/IDIV,
    CALL/JMP/JCC/RET, INT. Unrecognized opcodes are yielded as _INS_OTHER
    with length=1.
    """
    pc = 0
    n = len(text)
    dispatch = _X86_DISPATCH

    while pc < n:
        start = pc

        # ── Legacy prefixes (F3, F2, 66, 67) ──
        while pc < n and text[pc] in _LEGACY_PREFIXES:
            pc += 1

        # ── REX prefix (optional) ──
        rex = 0
        if pc < n and 0x40 <= text[pc] <= 0x4F:
            rex = text[pc]
            pc += 1

        if pc >= n:
            yield _DisasmInsn(
                address=base_addr + start,
                length=1,
                insn_id=_INS_OTHER,
                bytes=text[start : start + 1],
            )
            return

        opbyte = text[pc]
        pc += 1

        # Unrecognized — consume what was actually read (incl. any
        # prefixes).  Hardcoding 1 here misreports REX/legacy-prefixed
        # unknowns (e.g. "41 57" push r15) as length 1, which shifts
        # every subsequent block boundary in CFG analysis.
        insn = _DisasmInsn(address=base_addr + start, bytes=text[start:pc])
        handler = dispatch[opbyte]
        if handler is not None:
            pc = handler(text, n, pc, start, opbyte, rex, insn)
        insn.length = pc - start
        yield insn


def _elf64_le(elf: bytes) -> bool:
    """True for a full-header ELF64 little-endian image."""
    return len(elf) >= 64 and elf[:4] == b"\x7fELF" and elf[4] == 2 and elf[5] == 1


def _sym_sections(elf: bytes) -> tuple | None:
    """Section-header offsets of (.symtab, .strtab, .dynsym, .dynstr).

    Missing sections are None; returns None when the section table is
    empty or e_shstrndx is out of range.
    """
    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]
    if e_shnum == 0 or e_shstrndx >= e_shnum:
        return None
    shstr_off = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from("<Q", elf, shstr_off + 24)[0]
    symtab_sec = strtab_sec = dynsym_sec = dynstr_sec = None
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
        sh_name_idx = struct.unpack_from("<I", elf, sh)[0]
        name = elf[shstr_offset + sh_name_idx : shstr_offset + sh_name_idx + 32].split(b"\x00")[0]
        if sh_type == 2:
            symtab_sec = sh
        elif sh_type == 3 and name == b".strtab":
            strtab_sec = sh
        elif sh_type == 11:
            dynsym_sec = sh
        elif sh_type == 3 and name == b".dynstr":
            dynstr_sec = sh
    return symtab_sec, strtab_sec, dynsym_sec, dynstr_sec


def _read_sym_names(elf: bytes, sym_sec: int | None, str_sec: int | None) -> list[str]:
    """Every symbol name in one symbol table; [] when either section is missing."""
    if sym_sec is None or str_sec is None:
        return []
    sym_offset = struct.unpack_from("<Q", elf, sym_sec + 24)[0]
    sym_size = struct.unpack_from("<Q", elf, sym_sec + 32)[0]
    sym_entsize = struct.unpack_from("<Q", elf, sym_sec + 56)[0]
    if sym_entsize == 0:
        return []
    strtab_offset = struct.unpack_from("<Q", elf, str_sec + 24)[0]
    names: list[str] = []
    for i in range(sym_size // sym_entsize):
        sym = sym_offset + i * sym_entsize
        st_name_idx = struct.unpack_from("<I", elf, sym)[0]
        names.append(
            elf[strtab_offset + st_name_idx : strtab_offset + st_name_idx + 64]
            .split(b"\x00")[0]
            .decode(errors="replace")
        )
    return names


def _symbol_names(target: str) -> list[str]:
    """Return every name in the ELF .symtab and .dynsym. Empty list when unreadable.

    Shared by parse_sancov_offsets() and detect_ctx_bits(); both need a
    symbol-name scan and neither needs anything else from the ELF.
    """
    with open(target, "rb") as f:
        elf = f.read()
    if not _elf64_le(elf):
        return []
    secs = _sym_sections(elf)
    if secs is None:
        return []
    symtab_sec, strtab_sec, dynsym_sec, dynstr_sec = secs

    static_names = _read_sym_names(elf, symtab_sec, strtab_sec)
    dynamic_names = _read_sym_names(elf, dynsym_sec, dynstr_sec)
    # Deduplicate while preserving order
    return list(dict.fromkeys(static_names + dynamic_names))


def _bounds_in_symtab(elf: bytes, symtab_sec: int, strtab_sec: int, start_sym: str, stop_sym: str):
    """(start, stop) st_values of the two named symbols; None unless both are > 0."""
    sym_offset = struct.unpack_from("<Q", elf, symtab_sec + 24)[0]
    sym_size = struct.unpack_from("<Q", elf, symtab_sec + 32)[0]
    sym_entsize = struct.unpack_from("<Q", elf, symtab_sec + 56)[0]
    if sym_entsize == 0:
        return None
    sym_count = sym_size // sym_entsize
    strtab_offset = struct.unpack_from("<Q", elf, strtab_sec + 24)[0]
    start_addr = stop_addr = None
    for i in range(sym_count):
        sym = sym_offset + i * sym_entsize
        st_value = struct.unpack_from("<Q", elf, sym + 8)[0]
        st_name_idx = struct.unpack_from("<I", elf, sym)[0]
        name = (
            elf[strtab_offset + st_name_idx : strtab_offset + st_name_idx + 64]
            .split(b"\x00")[0]
            .decode(errors="replace")
        )
        if name == start_sym and st_value > 0:
            start_addr = st_value
        elif name == stop_sym and st_value > 0:
            stop_addr = st_value
    if start_addr is not None and stop_addr is not None:
        return (start_addr, stop_addr)
    return None


def _sancov_section_bounds(target: str, section: str) -> tuple[int, int] | None:
    """Virtual addresses of `__start___sancov_<section>` / `__stop___...`.

    `section` is the suffix, without the leading `__sancov_`: "cntrs" for
    -fsanitize-coverage=inline-8bit-counters, "guards" for trace-pc-guard.
    They are different sections with different element widths, and a binary
    may carry either, both, or neither.

    Args:
        target: Path to ELF binary (shared library or executable).
        section: Section-name suffix, e.g. "cntrs" or "guards".

    Returns:
        Tuple of (start_addr, stop_addr) if found, None otherwise.
    """
    start_sym = f"__start___sancov_{section}"
    stop_sym = f"__stop___sancov_{section}"
    try:
        with open(target, "rb") as f:
            elf = f.read()
        if not _elf64_le(elf):
            return None
        secs = _sym_sections(elf)
        if secs is None or secs[0] is None or secs[1] is None:
            return None
        return _bounds_in_symtab(elf, secs[0], secs[1], start_sym, stop_sym)
    except Exception as e:
        log.debug("ELF parse failed: %s", e)
    return None


def _has_static_symtab(target: str) -> bool:
    """True when *target* carries a non-empty SHT_SYMTAB section.

    ``_symbol_names`` merges ``.symtab`` and ``.dynsym``, so it cannot answer
    this: a stripped shared object still reports thousands of dynamic names.
    The ``__start___sancov_*`` bounds are static-only, so only the static
    table decides whether their absence means anything.
    """
    try:
        with open(target, "rb") as f:
            elf = f.read()
    except OSError as e:
        log.debug("ELF read failed for %s: %s", target, e)
        return False
    if len(elf) < 64 or elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        return False
    try:
        e_shoff = struct.unpack_from("<Q", elf, 40)[0]
        e_shnum = struct.unpack_from("<H", elf, 60)[0]
        e_shentsize = struct.unpack_from("<H", elf, 58)[0]
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            if struct.unpack_from("<I", elf, sh + 4)[0] == 2:  # SHT_SYMTAB
                return struct.unpack_from("<Q", elf, sh + 32)[0] > 0  # sh_size
    except struct.error as e:
        log.debug("ELF parse failed for %s: %s", target, e)
    return False


def sancov_guard_status(target: str) -> str:
    """Classify *target*'s compiler-inserted edge coverage as present/absent/unknown.

    ``afl_instrumentation_status`` answers a different question: it looks for
    ``__afl_area``/``__afl_map_shm``/``__sanitizer_cov``, which are the shim's
    own definitions and are present in every target, because the shim is
    ``-include``'d into all of them.  A binary with no instrumented call sites
    at all therefore reports "present" there.  Measured: a default
    ``tools/build_targets.sh`` run produced 20 binaries, 4 of which carried
    any instrumentation, and all 20 were classified "present".

    What actually produces edges is the guard array
    ``-fsanitize-coverage=trace-pc-guard`` emits, which is what
    ``verify_sancov`` in the build script greps for.  This is the same check,
    available before a campaign rather than only at build time, and reading
    the ELF directly rather than shelling out to ``readelf``.

    ``inline-8bit-counters`` and ``inline-bool-flag`` count too: different
    sections and element widths, but a target carrying either is likewise
    compiler-instrumented.

    The third state matters for the same reason it does in
    ``afl_instrumentation_status``: both section-bound symbols live in the
    static symbol table, so a stripped-but-instrumented target is
    indistinguishable from an uninstrumented one here.  Say "unknown" rather
    than raise a false alarm.

    Args:
        target: Path to an ELF executable or shared object.

    Returns:
        One of ``"present"``, ``"absent"``, ``"unknown"``.
    """
    if not _has_static_symtab(target):
        # Stripped, or unreadable. Both bound symbols live in .symtab, so
        # there is nothing here to distinguish "no instrumentation" from
        # "symbol table removed". A false alarm on a stripped-but-working
        # target is the fastest way to teach someone to ignore the warning.
        return "unknown"
    for section in ("guards", "cntrs", "bools"):
        bounds = _sancov_section_bounds(target, section)
        if bounds is not None and bounds[1] > bounds[0]:
            return "present"
    return "absent"


def _gnu_build_id_note(elf: bytes, pos: int, note_end: int) -> bytes | None:
    """Scan one PT_NOTE range [pos, note_end) for the GNU build-id descriptor."""
    # Note triplets are 4-aligned: namesz, descsz, type, name,
    # desc. Malformed lengths must not loop forever — bound each
    # step to leave at least the 12-byte header.
    while pos + 12 <= note_end:
        namesz = struct.unpack_from("<I", elf, pos)[0]
        descsz = struct.unpack_from("<I", elf, pos + 4)[0]
        ntype = struct.unpack_from("<I", elf, pos + 8)[0]
        name_off = pos + 12
        desc_off = name_off + (namesz + 3) & ~3
        next_pos = desc_off + (descsz + 3) & ~3
        if namesz == 0 or descsz <= 0 or desc_off + descsz > note_end or next_pos <= pos:
            break
        if ntype == 3 and elf[name_off : name_off + namesz] == b"GNU\x00":
            return bytes(elf[desc_off : desc_off + descsz])
        pos = next_pos
    return None


def build_id(target: str) -> bytes | None:
    """NT_GNU_BUILD_ID note contents, or None when absent/unparseable.

    Walks PT_NOTE segments for a note with name "GNU" and type 3
    (ELF_NOTE_GNU_BUILD_ID); the descriptor is 16-20 bytes (md5/sha1
    style) on every mainstream toolchain. Used by core/cfg_cache.py as
    the binary-identity component of its cache key.
    """
    try:
        with open(target, "rb") as f:
            elf = f.read()
        if not _elf64_le(elf):
            return None
        e_phoff = struct.unpack_from("<Q", elf, 32)[0]
        e_phentsize = struct.unpack_from("<H", elf, 54)[0]
        e_phnum = struct.unpack_from("<H", elf, 56)[0]
        if e_phentsize < 56 or e_phoff + e_phnum * e_phentsize > len(elf):
            return None
        for i in range(e_phnum):
            ph = e_phoff + i * e_phentsize
            p_type = struct.unpack_from("<I", elf, ph)[0]
            if p_type != 4:  # PT_NOTE
                continue
            p_offset = struct.unpack_from("<Q", elf, ph + 8)[0]
            p_filesz = struct.unpack_from("<Q", elf, ph + 32)[0]
            desc = _gnu_build_id_note(elf, p_offset, min(p_offset + p_filesz, len(elf)))
            if desc is not None:
                return desc
    except OSError as e:
        log.debug("build_id read failed: %s", e)
    return None


def parse_sancov_offsets(target: str) -> tuple[int, int] | None:
    """Parse ELF to find __start/__stop___sancov_cntrs virtual addresses.

    This is the *8-bit counters* section (-fsanitize-coverage=
    inline-8bit-counters), one byte per instrumented block, which is what
    `shim_factory` sizes its direct-mode bitmap from. Targets in this tree
    are built with trace-pc-guard instead — see parse_sancov_guard_count().

    Args:
        target: Path to ELF binary (shared library or executable).

    Returns:
        Tuple of (start_addr, stop_addr) if found, None otherwise.
    """
    return _sancov_section_bounds(target, "cntrs")


def parse_sancov_guard_count(target: str) -> int | None:
    """Exact instrumented block count from the `__sancov_guards` section.

    `-fsanitize-coverage=trace-pc-guard` emits one uint32 guard per basic
    block into `__sancov_guards`, so the block count is the section length
    over 4. This is the section every target in this tree actually carries;
    `parse_sancov_offsets` reads `__sancov_cntrs`, which trace-pc-guard
    builds do not emit at all.

    Args:
        target: Path to ELF binary (shared library or executable).

    Returns:
        Block count, or None when the binary is not trace-pc-guard
        instrumented (or the section is empty).
    """
    bounds = _sancov_section_bounds(target, "guards")
    if bounds is None:
        return None
    start, stop = bounds
    if stop <= start:
        return None
    return (stop - start) // 4


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


# The fields this module reads out of an Elf64_Shdr live at sh_name 0,
# sh_type 4, sh_addr 16, sh_offset 24, sh_size 32 -- so an entry shorter
# than 40 bytes cannot hold them.  A real ELF64 always uses 64.
_ELF64_SHDR_MIN = 40


def _find_text_section(elf: bytes) -> tuple[bytes, int, int] | None:
    """Locate ``.text`` in a 64-bit little-endian ELF image.

    Returns ``(text_data, sh_addr, sh_size)``, or ``None`` when the image is
    not a usable ELF64 or its section-header table does not fit inside the
    buffer.

    Every offset derived from the header is attacker-controlled -- the target
    path is a command-line argument, and a corrupt or hostile binary must make
    the fuzzer decline to analyse it, not abort out of startup.  The four
    callers each inlined this prologue with no bounds check on ``e_shoff``, so
    a crafted value reached ``struct.unpack_from("<Q", elf, shstr_off + 24)``
    and raised ``struct.error`` before any target ran (finding #23; it also
    violates the repo's own bounds-check rule).  The per-entry
    ``sh + e_shentsize > len(elf)`` guard the loops did have is not enough on
    its own: it says nothing about where the table starts, and it passes for a
    small ``e_shentsize`` while ``sh + 32`` still reads past the buffer.

    Sharing one prologue is the other half of the fix.  Four copies of the
    same parse is exactly the "fixed classes recur in sibling files" pattern
    the bug report calls out, and the next bounds bug found here would
    otherwise have to be fixed four times again.
    """
    n = len(elf)
    if n < 64 or elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        return None

    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]

    if e_shnum == 0 or e_shstrndx >= e_shnum:
        return None
    if e_shentsize < _ELF64_SHDR_MIN:
        return None
    # The whole section-header table must lie inside the buffer.  Phrased as
    # a subtraction so a 64-bit e_shoff cannot overflow the comparison.
    if e_shoff == 0 or e_shoff > n or e_shnum * e_shentsize > n - e_shoff:
        return None

    shstr_off = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from("<Q", elf, shstr_off + 24)[0]

    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
        if sh_type != 1:  # SHT_PROGBITS
            continue
        sh_name_idx = struct.unpack_from("<I", elf, sh)[0]
        name_at = min(shstr_offset + sh_name_idx, n)
        # Slicing clamps, so an out-of-range name reads as empty and simply
        # fails the comparison below -- same outcome as before, without the
        # unpack that used to precede it.
        if elf[name_at : name_at + 32].split(b"\x00")[0] != b".text":
            continue
        sh_addr = struct.unpack_from("<Q", elf, sh + 16)[0]
        sh_offset = struct.unpack_from("<Q", elf, sh + 24)[0]
        sh_size = struct.unpack_from("<Q", elf, sh + 32)[0]
        # sh_size is returned unclamped for _text_size(), which reports the
        # declared size; text_data is the slice, which clamps on its own.
        return elf[sh_offset : sh_offset + sh_size], sh_addr, sh_size
    return None


def _read_target_elf(target: str) -> bytes | None:
    """Read a target binary, returning None instead of raising on I/O error."""
    try:
        with open(target, "rb") as f:
            return f.read()
    except OSError:
        return None


# ── Section-header table scan (shared, bounds-checked) ──────────────────

# Section names that hold interesting word constants, mirroring honggfuzz's
# arch_isInterestingSection (rodata, data, data.rel.ro, and the suffixed
# variants; .text excluded).
_SECT_WORD_EXACT = (b".rodata", b".data", b".data.rel.ro")
_SECT_WORD_PREFIXES = (b".rodata.", b".data.rel.ro.")

# A .rodata.data-probing cap guards against a huge or degenerate section
# flooding the dictionary: collect at most 8192 raw words, dedupe, then
# emit at most 1024 tokens.
_DATA_WORD_COLLECT_CAP = 8192
_DATA_WORD_RESULT_CAP = 1024


def _iter_sections(elf: bytes):
    """Yield ``(name, sh_type, sh_addr, sh_offset, sh_size)`` for every
    SHT_PROGBITS section with a non-empty name.

    Bounds-checked against the same hostile-header regime as
    ``_find_text_section``: a corrupt or malicious binary must be declined,
    never crash the prologue. The section-header table and the shstrtab
    offset are validated before any unpack.
    """
    n = len(elf)
    if n < 64 or elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        return

    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]

    if e_shnum == 0 or e_shstrndx >= e_shnum:
        return
    if e_shentsize < _ELF64_SHDR_MIN:
        return
    # The whole section-header table must lie inside the buffer.  Phrased as
    # a subtraction so a 64-bit e_shoff cannot overflow the comparison.
    if e_shoff == 0 or e_shoff > n or e_shnum * e_shentsize > n - e_shoff:
        return

    shstr_off = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from("<Q", elf, shstr_off + 24)[0]

    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
        if sh_type != 1:  # SHT_PROGBITS
            continue
        sh_name_idx = struct.unpack_from("<I", elf, sh)[0]
        name_at = min(shstr_offset + sh_name_idx, n)
        # Slicing clamps, so an out-of-range name reads as empty.
        name = elf[name_at : name_at + 128].split(b"\x00")[0]
        if not name:
            continue
        sh_addr = struct.unpack_from("<Q", elf, sh + 16)[0]
        sh_offset = struct.unpack_from("<Q", elf, sh + 24)[0]
        sh_size = struct.unpack_from("<Q", elf, sh + 32)[0]
        yield name, sh_type, sh_addr, sh_offset, sh_size


def _find_text_section(elf: bytes) -> tuple[bytes, int, int] | None:
    """Locate ``.text`` in a 64-bit little-endian ELF image.

    Returns ``(text_data, sh_addr, sh_size)``, or ``None`` when the image is
    not a usable ELF64 or its section-header table does not fit inside the
    buffer.

    Every offset derived from the header is attacker-controlled -- the target
    path is a command-line argument, and a corrupt or hostile binary must make
    the fuzzer decline to analyse it, not abort out of startup.  The four
    callers each inlined this prologue with no bounds check on ``e_shoff``, so
    a crafted value reached ``struct.unpack_from("<Q", elf, shstr_off + 24)``
    and raised ``struct.error`` before any target ran (finding #23; it also
    violates the repo's own bounds-check rule).  The per-entry
    ``sh + e_shentsize > len(elf)`` guard the loops did have is not enough on
    its own: it says nothing about where the table starts, and it passes for a
    small ``e_shentsize`` while ``sh + 32`` still reads past the buffer.

    Sharing one prologue is the other half of the fix.  Four copies of the
    same parse is exactly the "fixed classes recur in sibling files" pattern
    the bug report calls out, and the next bounds bug found here would
    otherwise have to be fixed four times again.
    """
    for name, _t, sh_addr, sh_offset, sh_size in _iter_sections(elf):
        if name == b".text":
            return elf[sh_offset : sh_offset + sh_size], sh_addr, sh_size
    return None


def _word_is_noise(value: int, width: int) -> bool:
    """True when a data word is not worth a dictionary token.

    A 64-bit window that straddles noise u32 words (counters, -1 halves)
    yields a padded garbage token; keep only words whose two 32-bit halves
    are individually interesting.
    """
    if width == 8:
        lo, hi = value & 0xFFFFFFFF, value >> 32
        if lo == 0 or hi == 0:
            return True
        return _is_noise_immediate(lo, 4) or _is_noise_immediate(hi, 4)
    return value == 0 or _is_noise_immediate(value, width)


def _collect_words(section: bytes, size: int, words: set[int]) -> None:
    """Add every aligned non-noise u64 then u32 of *section* to *words* (capped)."""
    for width in (8, 4):  # u64 first, then u32, like honggfuzz
        n_words = size // width
        for i in range(min(n_words, _DATA_WORD_COLLECT_CAP)):
            value = struct.unpack_from("<Q" if width == 8 else "<I", section, i * width)[0]
            if _word_is_noise(value, width):
                continue
            words.add(value)
        if len(words) >= _DATA_WORD_COLLECT_CAP:
            break


def extract_data_word_constants(target: str) -> list[bytes]:
    """Extract literal word constants from rodata/.data as little-endian bytes.

    Mirrors honggfuzz's ``arch_elfCollectRoValues`` ro32/ro64 channel: every
    aligned 4- and 8-byte word in the interesting data sections is collected,
    deduplicated, and packed into dictionary tokens. These catch comparison
    constants (file magics, checksums, boundary values) that live in data
    rather than as immediates in ``.text``.

    Noise words (0, counters, -1, page-aligned addresses) are excluded through
    the same ``_is_noise_immediate`` filter the disassembly extractor uses.

    Returns:
        List of unique little-endian byte words (capped), or [] on failure.
    """
    elf = _read_target_elf(target)
    if elf is None:
        return []

    def _is_interesting(name: bytes) -> bool:
        if name in _SECT_WORD_EXACT:
            return True
        return any(name.startswith(p) for p in _SECT_WORD_PREFIXES)

    words: set[int] = set()
    for _name, _t, _addr, offset, size in _iter_sections(elf):
        if not _is_interesting(_name):
            continue
        if size <= 0 or offset > len(elf):
            continue
        _collect_words(elf[offset : offset + size], size, words)

    if not words:
        return []

    # Pack each word as its minimal little-endian representation, at least 4
    # bytes wide (mirroring the honggfuzz ro32/ro64 token, deduped by value).
    result = sorted(v.to_bytes(max(4, (v.bit_length() + 7) // 8), "little") for v in words)[
        :_DATA_WORD_RESULT_CAP
    ]
    log.info("Data-word constants: extracted %d values from %s", len(result), target)
    return result


def branch_density(target: str) -> float | None:
    """Compute branch density (conditional branches per KB) of a binary.

    Disassembles the .text section and counts conditional jump instructions
    (Jcc family) using the pure-Python decoder.

    Returns branches per KB of code, or None if analysis fails.

    This is a static metric that predicts fuzzing difficulty:
    - High density → more decision points per KB → harder to saturate
    - Useful for sizing edge bitmaps, estimating saturation, ranking targets

    Args:
        target: Path to ELF binary.

    Returns:
        Branches per KB (float), or None on failure.
    """
    elf = _read_target_elf(target)
    if elf is None:
        return None

    found = _find_text_section(elf)
    if found is None:
        return None
    text_data, text_vaddr, _ = found
    if not text_data:
        return None

    # Disassemble and count conditional branches
    return _branch_density_pure(text_data, text_vaddr)


def _branch_density_pure(text_data: bytes, text_vaddr: int) -> float | None:
    """Branch density via pure-Python decoder."""
    cond_branches = 0
    for insn in _decode_x86_64(text_data, text_vaddr):
        if insn.insn_id == _INS_JCC:
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
    elf = _read_target_elf(target)
    if elf is None:
        return None

    found = _find_text_section(elf)
    if found is None:
        return None
    return found[2]


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


def _first_imm(insn: _DisasmInsn) -> tuple[int, int] | None:
    """(value, byte width) of the first immediate operand, or None."""
    for op in insn.operands:
        if op.type == _OP_IMM:
            return op.imm, op.size or _guess_imm_width(op.imm)
    return None


def _add_imm_constants(constants: set[bytes], imm_value: int, imm_size: int) -> None:
    """Pack an immediate little-endian at its width, plus 2/4-byte sub-words."""
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


def extract_constants_pure(target: str) -> list[bytes]:
    """Extract compile-time constants from disassembly using pure-Python decoder.

    Disassembles .text and collects immediate operands from comparison,
    move, and test instructions (CMP, MOV, TEST, AND, OR, XOR, SUB,
    ADD with immediate).

    These constants are compile-time magic bytes, pattern strings, and
    boundary values that the code compares against — exactly what the
    fuzzer's dictionary should contain. Unlike .rodata string extraction,
    this catches:

      - Inlined memcmp constants folded into integer immediates
        (e.g. ``cmp rax, 0x0A1A0A0D0A474E89`` → "\\x89PNG\\r\\n\\x1a\\n")
      - Bitmask / flag values used in test/and/or instructions

    Returns:
        List of unique byte values (deduplicated, truncated to 256 entries).
    """
    elf = _read_target_elf(target)
    if elf is None:
        return []

    found = _find_text_section(elf)
    if found is None:
        return []
    text_data, text_vaddr, _ = found
    if not text_data:
        return []

    # Instructions whose immediate operands are likely comparison constants.
    # MOV is excluded because most immediates it loads are addresses/offsets.
    TARGET_IDS = {
        _INS_CMP,
        _INS_TEST,
        _INS_AND,
        _INS_OR,
        _INS_XOR,
        _INS_SUB,
        _INS_ADD,
        _INS_CMPXCHG,
    }

    constants: set[bytes] = set()

    for insn in _decode_x86_64(text_data, text_vaddr):
        imm = _first_imm(insn)
        if imm is None:
            continue
        imm_value, imm_size = imm

        # Skip small/noise immediates
        if imm_size <= 0 or imm_size > 8:
            continue

        # Filter out uninteresting values
        if _is_noise_immediate(imm_value, imm_size):
            continue

        if insn.insn_id in TARGET_IDS:
            _add_imm_constants(constants, imm_value, imm_size)

    # Cap at 256 entries to bound dictionary size
    result = list(constants)[:256]
    if result:
        log.info("Disassembly constants: extracted %d values from %s", len(result), target)
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


# ── Coverage map sizing ─────────────────────────────────────────────────

MAP_SIZE_DEFAULT = 8192

# Distinct edges per instrumented basic block. trace_pc_guard fires once per
# block, so guard_count counts BLOCKS, while the table is keyed on edges
# (prev_loc ^ cur_loc). A block with two successors contributes two edges;
# 2.0 is the usual CFG rule of thumb. Sizing straight from guard_count --
# which is what this function used to do -- is therefore already a factor of
# two short before any headroom is added.
EDGES_PER_BLOCK = 2.0

# Open addressing with linear probing degrades sharply as it fills: measured
# average probes per insertion at map_size entries were 1.5 at load 0.49,
# 2.8 at 0.79, 12.1 at 0.95 and 57.2 at 1.00, and every probe is a random
# access paid on every EDGE EXECUTION, not once per unique edge. Target
# load 0.5.
TARGET_LOAD_FACTOR = 0.5

# Upper bound on entries, and the reason for it.
#
# ShmCoverage.reset_edge_map() bumps a generation counter instead of
# memsetting the whole table, so table size no longer carries a per-exec
# clear tax. The old measurements below are kept for historical context;
# the cap is now driven by probe cost and memory, not reset overhead.
#
# Measured on this machine (old memset-based reset):
#
#     8,192 entries   0.1 MiB     3.8 us
#    65,536 entries   0.5 MiB    21.3 us
#   131,072 entries   1.0 MiB    84.6 us
#   262,144 entries   2.0 MiB    86.9 us
# 1,048,576 entries   8.0 MiB   352.8 us
#
# At a 100 us target execution the 1 MiB clear was already comparable to the
# run itself. So the old 131072 cap was not arbitrary after all -- it sits
# near where the reset cost stops being negligible, and simply raising it
# trades probe cost for memset cost without measuring which one dominates.
#
# 262144 is the honest compromise: it doubles the headroom for one large
# target at ~the same reset cost as 131072 (86.9 us vs 84.6 us, the two
# straddle a cache-hierarchy step), and stops well short of the cliff.
#
# Lifting this properly means removing the memset from the hot path --
# generation-tagged entries would make reset O(1) and let the cap follow
# instrumentation size instead of clear bandwidth. Until then, a target that
# wants more must say so via AFL_MAP_SIZE_MAX and accept the reset cost.
MAP_SIZE_MAX = 262144


def _map_size_max() -> int:
    """Entry cap, overridable via AFL_MAP_SIZE_MAX for targets that need it."""
    raw = os.environ.get("AFL_MAP_SIZE_MAX")
    if not raw:
        return MAP_SIZE_MAX
    try:
        v = int(raw)
    except ValueError:
        log.warning("AFL_MAP_SIZE_MAX=%r is not an integer; ignoring", raw)
        return MAP_SIZE_MAX
    if v < MAP_SIZE_DEFAULT:
        log.warning("AFL_MAP_SIZE_MAX=%d below minimum %d; ignoring", v, MAP_SIZE_DEFAULT)
        return MAP_SIZE_DEFAULT
    return _next_power_of_2(v)


def detect_ctx_bits(target: str) -> int | None:
    """Read __AFL_CTX_BITS out of a target's symbol table.

    afl_shim.c emits a marker symbol whose NAME carries the value
    (``__afl_ctx_bits_8``), so this needs only a symbol scan -- no section
    contents, no running process. That matters because the map has to be
    sized before the target has ever been executed.

    Returns:
        The context width, 0 for a context-free build, or None when the
        marker is absent -- an older shim, or a binary this shim never
        touched. None is deliberately distinct from 0: it means "unknown",
        not "context-free".
    """
    try:
        names = _symbol_names(target)
    except Exception as e:  # noqa: BLE001
        log.debug("ctx-bits detection failed for %s: %s", target, e)
        return None
    best = None
    for name in names:
        if name.startswith("__afl_ctx_bits_"):
            suffix = name[len("__afl_ctx_bits_") :]
            if suffix.isdigit():
                # A .so may link several instrumented TUs; they are built
                # under one contract, but take the widest if they disagree
                # so the map is sized for the worst case rather than the
                # first symbol encountered.
                v = int(suffix)
                best = v if best is None else max(best, v)
    return best


def detect_ngram_k(target: str) -> int:
    """Read __AFL_NGRAM_K out of a target's symbol table.

    Same marker-name scan as `detect_ctx_bits`: a `.so` may link several
    instrumented TUs, and on disagreement the maximum wins so the map is
    sized for the widest-instrumented TU rather than the first symbol
    seen.

    Unlike context bits there is no unknowable case: absence of the
    marker means an older shim, whose behaviour IS k=2 (the compatibility
    default), so 2 is returned instead of None.
    """
    try:
        names = _symbol_names(target)
    except Exception as e:  # noqa: BLE001
        log.debug("ngram-k detection failed for %s: %s", target, e)
        return 2
    best = 2
    for name in names:
        if name.startswith("__afl_ngram_k_"):
            suffix = name[len("__afl_ngram_k_") :]
            if suffix.isdigit():
                best = max(best, int(suffix))
    return best


def detect_ctx_relative_capable(target: str) -> bool | None:
    """Say whether *target*'s shim can hash caller context ASLR-invariantly.

    The mode itself is chosen at runtime by the target, from
    ``FUZZER_KEEP_ASLR`` in its environment; what this reads is whether the
    shim compiled into the binary knows about that switch at all. A target
    built before the shim grew base-relative context ignores the variable and
    hashes raw return addresses, so with ASLR on it reports a different edge
    set in every process (F1, docs/handover/
    handover_edge_id_axis_2026-09-18.md) -- which is indistinguishable from a
    target with endless new coverage, and was until this marker existed
    indistinguishable from a current build by anything short of executing it
    three times and comparing edge sets.

    Same symbol-name scan as `detect_ctx_bits` and `detect_shm_layout`, and
    like `detect_shm_layout` a safety check rather than a sizing input.

    Returns:
        True when the marker is present; False when the symbol table was read
        and the marker is absent -- an older shim, or a binary this shim
        never touched; None when no names could be read at all (stripped,
        unreadable, not an ELF64), which is not evidence either way.

        The distinction matters at the call site: False is worth warning
        about, None is not. False is only meaningful once `detect_ctx_bits`
        has established there is a shim and it hashes context -- without
        that, "no marker" says nothing, because an uninstrumented binary has
        no marker either.
    """
    try:
        names = _symbol_names(target)
    except Exception as e:  # noqa: BLE001
        log.debug("ctx-relative detection failed for %s: %s", target, e)
        return None
    if not names:
        return None
    return "__afl_ctx_relative_capable" in names


def detect_edge_id_scheme(target: str) -> int | None:
    """Say which edge-id function *target*'s shim uses.

    The shim stopped merging distinct edges with the hashed-location scheme (hashed guard and
    manual locations, zero remapped instead of ``edge_id |= 1``); every edge
    got a new id. State persisted under one scheme cannot be resumed under
    the other, and nothing short of this marker distinguishes the two before
    the target runs.

    Returns:
        2 when ``__afl_edge_ids_v2`` is present; 1 when the symbol table was
        read and the marker is absent (an older shim, or no shim -- ptrace
        targets hash addresses and never had either scheme, so they stay at
        1 on both sides of a resume); None when no names could be read,
        which the contract check treats as "unknown, do not compare".
    """
    try:
        names = _symbol_names(target)
    except Exception as e:  # noqa: BLE001
        log.debug("edge-id scheme detection failed for %s: %s", target, e)
        return None
    if not names:
        return None
    return 2 if "__afl_edge_ids_v2" in names else 1


def detect_scoped_crash_handler(target: str) -> bool | None:
    """Say whether *target*'s shim scopes its crash handler to the guard.

    Older shims jumped to ``__afl_guarded_call``'s sigjmp_buf on every
    crash signal, guarded or not: one-shot and forkserver crashes surfaced
    as SIGSEGV whatever the real signal, ASAN's SEGV report was lost, and a
    ctypes host died on its own broken pipe. ``__afl_scoped_crash_handler``
    marks a shim that hands unguarded signals back.

    Returns:
        True when the marker is present; False when the symbol table was
        read and it is absent (older shim, or no shim -- the caller decides
        whether a shim is there at all); None when no names could be read.
    """
    try:
        names = _symbol_names(target)
    except Exception as e:  # noqa: BLE001
        log.debug("crash-handler scope detection failed for %s: %s", target, e)
        return None
    if not names:
        return None
    return "__afl_scoped_crash_handler" in names


#: Segment layout the Python side is built for. Must equal
#: __AFL_SHM_LAYOUT in adapters/afl_shim.c.
SHM_LAYOUT_CURRENT = 3


def detect_elf_type(target: str) -> int | None:
    """Return the ELF e_type of *target* or None.

    e_type values: ET_EXEC=2 (position-dependent executable),
    ET_DYN=3 (shared object / PIE). Used to distinguish a PIE
    executable from a shared library so that callers that rely on
    ``ctypes.CDLL`` can refuse PIE targets with a clear error
    instead of the cryptic OS OSError.
    """
    try:
        with open(target, "rb") as f:
            head = f.read(64)
        if len(head) < 64 or head[:4] != b"\x7fELF":
            return None
        if head[4] != 2:  # EI_CLASS: ELFCLASS64 only
            return None
        return struct.unpack_from("<H", head, 16)[0]
    except OSError:
        log.debug("elf type detection failed for %s", target)
        return None


_ET_EXEC = 2
_ET_DYN = 3
_PT_INTERP = 3
_E_PHOFF = 32  # u64
_E_PHENTSIZE = 54  # u16, followed by e_phnum u16


def is_elf_executable(target: str) -> bool:
    """True when *target* is an ELF executable, PIE or not.

    ET_EXEC is always one. ET_DYN is either a PIE executable or a shared
    object; only the executable asks for a loader (PT_INTERP). Neither
    kind of executable can be dlopen'd, so in-process modes must not take it.
    """
    e_type = detect_elf_type(target)
    if e_type == _ET_EXEC:
        return True
    if e_type != _ET_DYN:
        return False

    # Read only the program header table: targets run to 100+ MB (ffmpeg).
    try:
        with open(target, "rb") as f:
            head = f.read(64)
            (phoff,) = struct.unpack_from("<Q", head, _E_PHOFF)
            phentsize, phnum = struct.unpack_from("<HH", head, _E_PHENTSIZE)
            f.seek(phoff)
            phdrs = f.read(phentsize * phnum)
    except OSError:
        return False

    n = len(phdrs)
    for off in range(0, phentsize * phnum, phentsize or 1):
        if off + 4 > n:
            return False
        if struct.unpack_from("<I", phdrs, off)[0] == _PT_INTERP:
            return True
    return False


def detect_shm_layout(target: str) -> int:
    """Read __AFL_SHM_LAYOUT out of a target's symbol table.

    Same marker-name scan as `detect_ctx_bits` and `detect_ngram_k`, and a
    safety check rather than a sizing input.

    The layouts disagree about where the edge table starts and what the word
    at offset 4 means, so running a stale prebuilt target against the
    current fuzzer does not degrade coverage, it corrupts it: the target
    writes entries at its offset and we read at ours, so every edge id read
    back is a splice of two adjacent entries and our own header words are
    read as edges. That produces a plausible-looking stream of
    never-before-seen edge ids -- a corpus that grows on garbage, which is
    worse than a run that reports nothing. Nothing downstream can detect it,
    which is why it is caught statically.

    Absence of the marker means layout 1: every shim built before the marker
    existed produced that, and none of them exported anything to say so.
    """
    try:
        names = _symbol_names(target)
    except Exception as e:  # noqa: BLE001
        # Unknown, not stale. Reporting a mismatch for every unreadable or
        # non-ELF path would abort runs that are perfectly fine.
        log.debug("shm-layout detection failed for %s: %s", target, e)
        return SHM_LAYOUT_CURRENT
    best = 1
    for name in names:
        if name.startswith("__afl_shm_layout_"):
            suffix = name[len("__afl_shm_layout_") :]
            if suffix.isdigit():
                best = max(best, int(suffix))
    return best


def detect_cmplog_functions(target: str) -> tuple[str, ...]:
    """Read supported cmplog interceptors from the target's exported symbols.

    When the shim is compiled with ``__AFL_CMPLOG=1`` each interceptor is
    also exported as ``afl_cmp_<name>``.  This function scans for those
    markers and returns the underlying function names, so the fuzzer
    banner reflects the real interceptor set instead of a hardcoded list.
    """
    prefix = "afl_cmp_"
    try:
        names = _symbol_names(target)
    except Exception as e:  # noqa: BLE001
        log.debug("cmplog-function detection failed for %s: %s", target, e)
        return ()
    return tuple(name[len(prefix) :] for name in names if name.startswith(prefix))


def ctx_inflation_factor(ctx_bits: int | None) -> float:
    """How much context-sensitivity multiplies the distinct-edge count.

    The true factor is the target's call-graph fan-in, which no static
    analysis here can predict -- 2**ctx_bits is only the ceiling, and a real
    target sits far below it (most edges have exactly one caller). Sizing for
    the ceiling would demand gigabytes for ctx_bits=8.

    So this returns a deliberately modest estimate and leans on the runtime
    drop counter (ShmCoverage.read_dropped_edges) to correct it: guessing low
    and resizing on evidence beats guessing high and paying the reset cost on
    every execution forever. sqrt of the ceiling, clamped, is a heuristic
    with no measurement behind it -- it is a starting point that the feedback
    loop is expected to fix, not a prediction.
    """
    if not ctx_bits:
        return 1.0
    return min(2.0 ** (ctx_bits / 2.0), 16.0)


def ngram_inflation_factor(k: int | None) -> float:
    """How much n-gram history multiplies the distinct-edge count.

    A k-gram hashes the current block with its k−1 predecessors, so live
    cardinality grows toward E^(k−1) for a target with E reachable hops.
    As with context bits, sizing for the ceiling would be absurd; this
    heuristic (quadratic in k−1, capped at 32) only has to start the
    runtime drop-counter feedback loop in the right neighbourhood.
    """
    if not k or k <= 2:
        return 1.0
    return min(float((k - 1) ** 2), 32.0)


def _size_from_blocks(block_count: int, ctx_bits: int | None, ngram_k: int = 2) -> int:
    """Entries needed for ``block_count`` instrumented blocks."""
    edges = (
        block_count
        * EDGES_PER_BLOCK
        * ctx_inflation_factor(ctx_bits)
        * ngram_inflation_factor(ngram_k)
    )
    needed = int(edges / TARGET_LOAD_FACTOR)
    return max(MAP_SIZE_DEFAULT, min(_map_size_max(), _next_power_of_2(needed)))


class MapSizeEstimate(NamedTuple):
    """Where a map size came from, not just what it was.

    `source` is the tier that produced `blocks`:

    - ``"sancov_guards"`` — exact, ``__sancov_guards`` (trace-pc-guard).
    - ``"sancov_cntrs"``  — exact, ``__sancov_cntrs`` (inline-8bit-counters).
    - ``"sancov_bools"``  — exact, ``__sancov_bools`` (inline-bool-flag).
    - ``"profile"``       — TargetProfile.total_branches.
    - ``"branch_density"`` — disassembly estimate. Approximate.
    - ``"default"``       — nothing worked; MAP_SIZE_DEFAULT.

    The first three are measurements and the rest are guesses, and the gap
    between them is wide: on this tree's targets, branch density ran 4-16x
    above the true guard count. A caller that cannot tell which it got
    cannot tell a sized map from a guessed one -- which is exactly how
    `parse_sancov_offsets` reading the wrong section stayed invisible for
    the whole life of this function. See
    docs/learnings/2026-08-14-sancov-guards-vs-cntrs.md.
    """

    entries: int
    blocks: int
    source: str
    ctx_bits: int
    ngram_k: int
    capped: bool

    @property
    def exact(self) -> bool:
        """True when `blocks` was read out of the binary, not estimated."""
        return self.source in ("sancov_guards", "sancov_cntrs", "sancov_bools")


def estimate_map_size_detail(target: str, profile: object | None = None) -> MapSizeEstimate:
    """`estimate_map_size()`, with the provenance of the answer attached.

    Args:
        target: Path to ELF binary.
        profile: Optional TargetProfile with precomputed static analysis.

    Returns:
        MapSizeEstimate. `entries` is what estimate_map_size() returns.
    """
    ctx_bits = detect_ctx_bits(target) or 0
    ngram_k = detect_ngram_k(target)

    blocks = 0
    source = "default"

    # 1. sancov, exact block count for instrumented binaries. trace-pc-guard
    #    (__sancov_guards) first: it is what build_targets.sh emits.
    #    inline-8bit-counters (__sancov_cntrs) after, for externally built
    #    targets -- one *byte* per block there, not one uint32.
    #    inline-bool-flag (__sancov_bools) last, also one byte per block.
    guards = parse_sancov_guard_count(target)
    if guards:
        blocks, source = guards, "sancov_guards"
    else:
        offsets = parse_sancov_offsets(target)
        if offsets and offsets[1] > offsets[0]:
            blocks, source = offsets[1] - offsets[0], "sancov_cntrs"
    if not blocks:
        bools = _sancov_section_bounds(target, "bools")
        if bools and bools[1] > bools[0]:
            blocks, source = bools[1] - bools[0], "sancov_bools"

    # 2. Cached profile data — avoids a full-text disassembly.
    #    total_branches is a branch count, and _size_from_blocks applies
    #    EDGES_PER_BLOCK, so pass it through as the block-equivalent.
    if not blocks and profile is not None:
        ts = getattr(profile, "text_size", 0)
        tb = getattr(profile, "total_branches", 0)
        if isinstance(ts, int) and isinstance(tb, int) and ts > 0 and tb > 0:
            blocks, source = tb, "profile"

    # 3. Branch density estimation (full-text disassembly)
    if not blocks:
        bd = branch_density(target)
        ts_opt = _text_size(target)
        if bd is not None and ts_opt:
            blocks, source = int(bd * (ts_opt / 1024)), "branch_density"

    if not blocks:
        return MapSizeEstimate(
            entries=MAP_SIZE_DEFAULT,
            blocks=0,
            source="default",
            ctx_bits=ctx_bits,
            ngram_k=ngram_k,
            capped=False,
        )

    entries = _size_from_blocks(blocks, ctx_bits, ngram_k)
    return MapSizeEstimate(
        entries=entries,
        blocks=blocks,
        source=source,
        ctx_bits=ctx_bits,
        ngram_k=ngram_k,
        capped=entries >= _map_size_max(),
    )


def estimate_map_size(target: str, profile: object | None = None) -> int:
    """Size the coverage hash table, in entries (AFL_MAP_SIZE convention).

    Multiply by 8 for SHM bytes.

    Sizing accounts for three things the previous version did not:

    1. **Edges, not blocks.** trace_pc_guard fires per basic BLOCK, but the
       table is keyed on edges, so guard_count is scaled by EDGES_PER_BLOCK.
    2. **Load factor.** This is open addressing with linear probing, not
       AFL's direct-indexed bitmap. Filling it to 1.0 does not merely
       collide, it makes every edge execution walk the table and then drop
       the edge. Sized for TARGET_LOAD_FACTOR.
    3. **Context sensitivity.** A -D__AFL_CTX_SENSITIVE=1 build multiplies
       distinct edge IDs by call-graph fan-in. detect_ctx_bits() reads the
       width straight out of the binary, so a CTX target no longer gets
       silently sized as if it were context-free.

    Together (1) and (2) mean a context-free target now asks for roughly
    4x guard_count where it previously asked for next_pow2(guard_count) --
    i.e. it was under-sized by about 4x, before any context inflation.

    The result is capped (see MAP_SIZE_MAX): the table is memset before
    every execution, so size is a per-exec cost, and past a point a bigger
    map loses more to clearing than it saves in probes. When the cap binds,
    the target may still saturate -- that is what the shim's drop counter is
    for. Check ShmCoverage.read_dropped_edges() rather than assuming the
    static estimate held.

    Priority, logged at INFO on every call so the tier is visible in a run
    log rather than inferred from the number:

    1. sancov guard count, when either sancov section is present (exact).
    2. TargetProfile.total_branches, avoiding a redundant disassembly.
    3. branch_density x .text_size estimation.

    Use estimate_map_size_detail() when the caller needs the tier.

    Args:
        target: Path to ELF binary.
        profile: Optional TargetProfile with precomputed static analysis.

    Returns:
        Number of entries; MAP_SIZE_DEFAULT on failure.
    """
    est = estimate_map_size_detail(target, profile)

    if est.ctx_bits:
        log.info(
            "%s: context-sensitive coverage (__AFL_CTX_BITS=%d), "
            "sizing map with a %.1fx inflation allowance",
            target,
            est.ctx_bits,
            ctx_inflation_factor(est.ctx_bits),
        )

    if est.source == "default":
        log.info(
            "%s: map sized at %d entries (default -- no sancov section, no "
            "profile, and .text could not be disassembled)",
            target,
            est.entries,
        )
    else:
        log.info(
            "%s: map sized at %d entries from %d blocks (%s, %s)%s",
            target,
            est.entries,
            est.blocks,
            est.source,
            "exact" if est.exact else "ESTIMATED",
            " -- AT CAP, check read_dropped_edges()" if est.capped else "",
        )
    if not est.exact and est.source != "default":
        log.info(
            "%s: no sancov section found, so this size is an estimate. Build "
            "with -fsanitize-coverage=trace-pc-guard (tools/build_targets.sh "
            "--clang-scov) for an exact count.",
            target,
        )

    return est.entries


def _extract_imm(insn) -> int | None:
    """If *insn* loads a constant into a register, return the constant.

    Handles ``mov reg, imm``, ``movabs reg, imm``, and simple
    ``lea reg, [disp]`` (no base/index).
    Works with _DisasmInsn from the pure-Python decoder.
    """
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

    return None


def _is_ctrl_flow(insn) -> bool:
    """Return True if *insn* changes control flow (call, jmp, ret, jcc).

    Works with _DisasmInsn from the pure-Python decoder.
    """
    if isinstance(insn, _DisasmInsn):
        return bool(insn.groups & {_GRP_CALL, _GRP_JUMP, _GRP_RET})
    return False


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
    elf = _read_target_elf(target)
    if elf is None:
        return {}, set()

    found = _find_text_section(elf)
    if found is None:
        return {}, set()
    text_data, text_vaddr, _ = found
    if not text_data:
        return {}, set()

    try:
        return _extract_div_pure(text_data, text_vaddr)
    except Exception:
        return {}, set()


def _track_rem_regs(insn: _DisasmInsn, regs_write, rem_regs: set[int], dx_family: set[int]):
    """Propagate the set of registers holding a DIV remainder across *insn*."""
    # 1) Remove registers overwritten by this instruction (preserve EDX)
    for r in regs_write:
        if r not in dx_family:
            rem_regs.discard(r)
    # 2) MOV dest, src where src carries the remainder → track dest too
    if insn.insn_id == _INS_MOV and len(insn.operands) == 2:
        d, s = insn.operands[0], insn.operands[1]
        if d.type == _OP_REG and s.type == _OP_REG and s.reg in rem_regs:
            rem_regs.add(d.reg)
    # 3) DIV/IDIV puts the remainder in EDX
    if insn.insn_id in (_INS_DIV, _INS_IDIV):
        return set(dx_family)
    return rem_regs


def _div_divisor(op: _Operand, recent: list[tuple], reg_alias: dict[int, set[int]]):
    """Constant divisor of a DIV/IDIV operand, or None when not static."""
    # Method 1: immediate operand
    if op.type == _OP_IMM:
        return op.imm if 0 < op.imm <= 0xFFFFFFFF else None
    if op.type != _OP_REG:
        return None

    # Method 2: register operand with backward scan to its last writer
    div_reg = op.reg
    div_reg_family = reg_alias.get(div_reg, {div_reg})
    for prev_insn, prev_writes in reversed(recent[:-1]):
        if _is_ctrl_flow(prev_insn):
            return None
        if prev_writes & div_reg_family:
            candidate = _extract_imm(prev_insn)
            if candidate is not None and 0 < candidate <= 0xFFFFFFFF:
                return candidate
            return None
    return None


def _mod_cmp_link(insn, recent, known_divs, div_map, weak_mod_pcs) -> None:
    """Map a CMP on the remainder to the nearest preceding DIV's divisor."""
    for prev_insn, _ in reversed(recent[:-1]):
        if prev_insn.insn_id not in (_INS_DIV, _INS_IDIV):
            continue
        d = known_divs.get(prev_insn.address)
        if d is not None:
            div_map[insn.address] = d
        else:
            weak_mod_pcs.add(insn.address)
        return


def _extract_div_pure(text_data: bytes, text_vaddr: int) -> tuple[dict[int, int], set[int]]:
    """extract_div_constants using the pure-Python decoder (no capstone)."""
    # Register alias map — in our pure encoding, each register ID IS its own alias
    # (all widths of the same register map to the same ID 0-15)
    reg_alias: dict[int, set[int]] = {i: {i} for i in range(16)}

    # DX family: register 2 (rdx/edx/dx/dl all map to 2)
    _dx_family: set[int] = {2}
    # Dynamic remainder register tracking — expands through MOV copies
    _rem_regs: set[int] = set()

    MAX_BACKWARD = 50
    recent: list[tuple] = []

    div_map: dict[int, int] = {}
    _known_divs: dict[int, int] = {}
    weak_mod_pcs: set[int] = set()

    for insn in _decode_x86_64(text_data, text_vaddr):
        regs_read, regs_write = insn.regs_access()

        # ── Track remainder register propagation ──
        _rem_regs = _track_rem_regs(insn, regs_write, _rem_regs, _dx_family)

        recent.append((insn, set(regs_write)))
        if len(recent) > MAX_BACKWARD:
            recent.pop(0)

        # ── DIV/IDIV detection ──
        if insn.insn_id in (_INS_DIV, _INS_IDIV):
            if not insn.operands:
                continue
            divisor = _div_divisor(insn.operands[0], recent, reg_alias)
            if divisor is not None:
                div_map[insn.address] = divisor
                _known_divs[insn.address] = divisor
            continue

        # ── Forward modulus extraction ──
        if (
            insn.insn_id == _INS_CMP
            and len(insn.operands) >= 2
            and any(op.type == _OP_REG and op.reg in _rem_regs for op in insn.operands)
        ):
            _mod_cmp_link(insn, recent, _known_divs, div_map, weak_mod_pcs)

    if div_map or weak_mod_pcs:
        log.info(
            "elf: found %d DIV/IDIV mappings, %d weak modulus PCs (pure decoder)",
            len(div_map),
            len(weak_mod_pcs),
        )
    return div_map, weak_mod_pcs
