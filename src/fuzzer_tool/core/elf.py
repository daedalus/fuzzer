"""Shared ELF parsing utilities for sancov counter discovery and analysis.

Includes a pure-Python x86-64 instruction decoder that replaces the
optional Capstone dependency for all static analysis tasks:
branch density, constant extraction, DIV detection, and ctrl-flow analysis.

Consolidates the duplicated ELF parsing logic from shim_factory.py
and fuzzer.py (PtraceCoverage). The embedded _PERSISTENT_LOADER script
in persistent_loader.py retains its own copy since it runs in a
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


def _decode_x86_64(text: bytes, base_addr: int):
    """Pure-Python x86-64 instruction decoder — yields _DisasmInsn objects.

    Handles: MOV, LEA, CMP, TEST, ADD/SUB/AND/OR/XOR, CMPXCHG, DIV/IDIV,
    CALL/JMP/JCC/RET, INT. Unrecognized opcodes are yielded as _INS_OTHER
    with length=1.
    """
    pc = 0
    n = len(text)

    while pc < n:
        start = pc
        addr = base_addr + pc

        # ── Legacy prefixes (F3, F2, 66, 67) ──
        while pc < n and text[pc] in (0xF3, 0xF2, 0x66, 0x67):
            pc += 1

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
            insn.length = pc - start
            insn.op_str = f"0x{addr + insn.length + off:x}"
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
            insn.length = pc - start
            insn.op_str = f"0x{addr + insn.length + off:x}"
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
            insn.length = pc - start
            insn.op_str = f"0x{addr + insn.length + off:x}"
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
            insn.length = pc - start
            insn.op_str = f"0x{addr + insn.length + off:x}"
            yield insn
            continue

        # INT imm8 (0xCD)
        if opbyte == 0xCD:
            if pc < n:
                pc += 1  # consume imm8
            insn.groups = {_GRP_INT}
            insn.length = pc - start
            yield insn
            continue

        # ── Accumulator-specific immediate forms ──
        # ADD EAX, imm32 (0x05)
        if opbyte == 0x05:
            if pc + 4 <= n:
                imm = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                imm = 0
            insn.insn_id = _INS_ADD
            insn.operands = [_Operand(_OP_REG, 0, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
            insn._regs_read = {0}
            insn._regs_write = {0}
            insn.length = pc - start
            yield insn
            continue

        # OR EAX, imm32 (0x0D)
        if opbyte == 0x0D:
            if pc + 4 <= n:
                imm = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                imm = 0
            insn.insn_id = _INS_OR
            insn.operands = [_Operand(_OP_REG, 0, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
            insn._regs_read = {0}
            insn._regs_write = {0}
            insn.length = pc - start
            yield insn
            continue

        # AND EAX, imm32 (0x25)
        if opbyte == 0x25:
            if pc + 4 <= n:
                imm = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                imm = 0
            insn.insn_id = _INS_AND
            insn.operands = [_Operand(_OP_REG, 0, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
            insn._regs_read = {0}
            insn._regs_write = {0}
            insn.length = pc - start
            yield insn
            continue

        # SUB EAX, imm32 (0x2D)
        if opbyte == 0x2D:
            if pc + 4 <= n:
                imm = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                imm = 0
            insn.insn_id = _INS_SUB
            insn.operands = [_Operand(_OP_REG, 0, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
            insn._regs_read = {0}
            insn._regs_write = {0}
            insn.length = pc - start
            yield insn
            continue

        # TEST EAX, imm32 (0xA9)
        if opbyte == 0xA9:
            if pc + 4 <= n:
                imm = struct.unpack_from("<i", text, pc)[0]
                pc += 4
            else:
                imm = 0
            insn.insn_id = _INS_TEST
            insn.operands = [_Operand(_OP_REG, 0, size=4), _Operand(_OP_IMM, imm=imm, size=4)]
            insn._regs_read = {0}
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
                insn.length = pc - start
                insn.op_str = f"0x{addr + insn.length + off:x}"
                yield insn
                continue

            # NOP / CET (0F 1E, 0F 1F) — always has ModRM
            if op2 in (0x1E, 0x1F):
                _decode_modrm()  # consume ModRM + optional SIB+disp
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue

            # CMPXCHG r/m32, r32 (0F B1)
            if op2 == 0xB1:
                result = _decode_modrm()
                if result is not None:
                    mod, reg, rm, has_sib, sib_byte = result
                    if mod == 3:
                        insn.insn_id = _INS_CMPXCHG
                        insn.operands = [
                            _Operand(_OP_REG, rm, size=4),
                            _Operand(_OP_REG, reg, size=4),
                        ]
                        insn._regs_read = {reg, rm}
                        insn._regs_write = {rm}
                    insn.length = pc - start
                    yield insn
                    continue
                # fall through on decode failure

            # Unknown two-byte opcode — skip
            insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # ── Multi-byte opcodes with ModR/M ──

        # F7 — GRP3 (TEST/DIV/IDIV/NOT/NEG/MUL/IMUL)
        if opbyte == 0xF7:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg_ext, rm, has_sib, sib_byte = result
            ext = reg_ext & 7
            if ext == 0:  # TEST r/m32, imm32
                if pc + 4 <= n:
                    imm = struct.unpack_from("<i", text, pc)[0]
                    pc += 4
                else:
                    imm = 0
                insn.insn_id = _INS_TEST
                if mod == 3:
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                insn._regs_read = {rm}
            elif ext in (6, 7):  # DIV (6) / IDIV (7) — register OR memory operand
                is_idiv = ext == 7
                insn.insn_id = _INS_IDIV if is_idiv else _INS_DIV
                if mod == 3:  # register operand
                    insn.operands = [_Operand(_OP_REG, rm, size=4)]
                    insn._regs_read = {0, 2, rm}  # EAX, EDX, divisor reg
                else:  # memory operand — divisor not a register
                    insn.operands = [_Operand(_OP_MEM, size=4)]
                    insn._regs_read = {0, 2}  # EAX, EDX only
                insn._regs_write = {0, 2}  # EAX, EDX (quotient, remainder)
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # F6 — GRP3 byte-size (TEST r/m8 / DIV / IDIV / NOT / NEG / MUL / IMUL)
        if opbyte == 0xF6:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg_ext, rm, has_sib, sib_byte = result
            ext = reg_ext & 7
            if ext == 0:  # TEST r/m8, imm8
                imm = text[pc] if pc < n else 0
                pc += 1
                insn.insn_id = _INS_TEST
                if mod == 3:
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=1),
                        _Operand(_OP_IMM, imm=imm, size=1),
                    ]
                insn._regs_read = {rm}
            elif ext in (6, 7):  # DIV (6) / IDIV (7) — register OR memory operand
                is_idiv = ext == 7
                insn.insn_id = _INS_IDIV if is_idiv else _INS_DIV
                if mod == 3:  # register operand
                    insn.operands = [_Operand(_OP_REG, rm, size=1)]
                    insn._regs_read = {0, 2, rm}  # AL/AX, DX, divisor reg
                else:  # memory operand — divisor not a register
                    insn.operands = [_Operand(_OP_MEM, size=1)]
                    insn._regs_read = {0, 2}  # AL/AX, DX only
                insn._regs_write = {0, 2}  # quotient/remainder
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
            if mod == 3:  # register operand
                ext = reg_ext & 7
                if ext == 0:  # ADD r/m32, imm32
                    insn.insn_id = _INS_ADD
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                    insn._regs_write = {rm}
                elif ext == 1:  # OR r/m32, imm32
                    insn.insn_id = _INS_OR
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                    insn._regs_write = {rm}
                elif ext == 4:  # AND r/m32, imm32
                    insn.insn_id = _INS_AND
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                    insn._regs_write = {rm}
                elif ext == 5:  # SUB r/m32, imm32
                    insn.insn_id = _INS_SUB
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                    insn._regs_write = {rm}
                elif ext == 7:  # CMP r/m32, imm32
                    insn.insn_id = _INS_CMP
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                else:
                    insn.insn_id = _INS_OTHER
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
            if mod == 3:  # register operand
                ext = reg_ext & 7
                if ext == 0:  # ADD r/m32, imm8
                    insn.insn_id = _INS_ADD
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                    insn._regs_write = {rm}
                elif ext == 1:  # OR r/m32, imm8
                    insn.insn_id = _INS_OR
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                    insn._regs_write = {rm}
                elif ext == 4:  # AND r/m32, imm8
                    insn.insn_id = _INS_AND
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                    insn._regs_write = {rm}
                elif ext == 5:  # SUB r/m32, imm8
                    insn.insn_id = _INS_SUB
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                    insn._regs_write = {rm}
                elif ext == 7:  # CMP r/m32, imm8
                    insn.insn_id = _INS_CMP
                    insn.operands = [
                        _Operand(_OP_REG, rm, size=4),
                        _Operand(_OP_IMM, imm=imm, size=4),
                    ]
                    insn._regs_read = {rm}
                else:
                    insn.insn_id = _INS_OTHER
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

        # 01 / 03 — ADD r/m32, r32 / ADD r32, r/m32
        if opbyte in (0x01, 0x03):
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:
                insn.insn_id = _INS_ADD
                if opbyte == 0x01:
                    # ADD r/m32, r32
                    insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {rm}
                else:
                    # ADD r32, r/m32
                    insn.operands = [_Operand(_OP_REG, reg, size=4), _Operand(_OP_REG, rm, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {reg}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 09 / 0B — OR r/m32, r32 / OR r32, r/m32
        if opbyte in (0x09, 0x0B):
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:
                insn.insn_id = _INS_OR
                if opbyte == 0x09:
                    insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {rm}
                else:
                    insn.operands = [_Operand(_OP_REG, reg, size=4), _Operand(_OP_REG, rm, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {reg}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 21 / 23 — AND r/m32, r32 / AND r32, r/m32
        if opbyte in (0x21, 0x23):
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:
                insn.insn_id = _INS_AND
                if opbyte == 0x21:
                    insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {rm}
                else:
                    insn.operands = [_Operand(_OP_REG, reg, size=4), _Operand(_OP_REG, rm, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {reg}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 29 / 2B — SUB r/m32, r32 / SUB r32, r/m32
        if opbyte in (0x29, 0x2B):
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:
                insn.insn_id = _INS_SUB
                if opbyte == 0x29:
                    insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {rm}
                else:
                    insn.operands = [_Operand(_OP_REG, reg, size=4), _Operand(_OP_REG, rm, size=4)]
                    insn._regs_read = {reg, rm}
                    insn._regs_write = {reg}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 85 — TEST r/m32, r32 (like AND, but read-only — flags only)
        if opbyte == 0x85:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:
                insn.insn_id = _INS_TEST
                insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
                insn._regs_read = {reg, rm}
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

        # 89 — MOV r/m32, r32 (register-to-register copy)
        if opbyte == 0x89:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:  # MOV rm32, r32
                insn.insn_id = _INS_MOV
                insn.operands = [_Operand(_OP_REG, rm, size=4), _Operand(_OP_REG, reg, size=4)]
                insn._regs_read = {reg}
                insn._regs_write = {rm}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # 8B — MOV r32, r/m32 (register-to-register copy)
        if opbyte == 0x8B:
            result = _decode_modrm()
            if result is None:
                insn.insn_id = _INS_OTHER
                insn.length = pc - start
                yield insn
                continue
            mod, reg, rm, has_sib, sib_byte = result
            if mod == 3:  # MOV reg32, rm32
                insn.insn_id = _INS_MOV
                insn.operands = [_Operand(_OP_REG, reg, size=4), _Operand(_OP_REG, rm, size=4)]
                insn._regs_read = {rm}
                insn._regs_write = {reg}
            else:
                insn.insn_id = _INS_OTHER
            insn.length = pc - start
            yield insn
            continue

        # Unrecognized — consume what was actually read (incl. any
        # prefixes).  Hardcoding 1 here misreports REX/legacy-prefixed
        # unknowns (e.g. "41 57" push r15) as length 1, which shifts
        # every subsequent block boundary in CFG analysis.
        insn.insn_id = _INS_OTHER
        insn.length = pc - start
        yield insn


def _symbol_names(target: str) -> list[str]:
    """Return every name in the ELF .symtab. Empty list when unreadable.

    Shared by parse_sancov_offsets() and detect_ctx_bits(); both need a
    symbol-name scan and neither needs anything else from the ELF.
    """
    out: list[str] = []
    with open(target, "rb") as f:
        elf = f.read()
    if len(elf) < 64 or elf[:4] != b"\x7fELF":
        return out
    if elf[4] != 2 or elf[5] != 1:  # ELF64, little-endian
        return out
    e_shoff = struct.unpack_from("<Q", elf, 40)[0]
    e_shnum = struct.unpack_from("<H", elf, 60)[0]
    e_shentsize = struct.unpack_from("<H", elf, 58)[0]
    e_shstrndx = struct.unpack_from("<H", elf, 62)[0]
    if e_shnum == 0 or e_shstrndx >= e_shnum:
        return out
    shstr_off = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from("<Q", elf, shstr_off + 24)[0]
    symtab_sec = strtab_sec = None
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_type = struct.unpack_from("<I", elf, sh + 4)[0]
        sh_name_idx = struct.unpack_from("<I", elf, sh)[0]
        name = elf[shstr_offset + sh_name_idx : shstr_offset + sh_name_idx + 32].split(b"\x00")[0]
        if sh_type == 2:
            symtab_sec = sh
        elif sh_type == 3 and name == b".strtab":
            strtab_sec = sh
    if symtab_sec is None or strtab_sec is None:
        return out
    sym_offset = struct.unpack_from("<Q", elf, symtab_sec + 24)[0]
    sym_size = struct.unpack_from("<Q", elf, symtab_sec + 32)[0]
    sym_entsize = struct.unpack_from("<Q", elf, symtab_sec + 56)[0]
    if sym_entsize == 0:
        return out
    strtab_offset = struct.unpack_from("<Q", elf, strtab_sec + 24)[0]
    for i in range(min(sym_size // sym_entsize, 20000)):
        sym = sym_offset + i * sym_entsize
        st_name_idx = struct.unpack_from("<I", elf, sym)[0]
        out.append(
            elf[strtab_offset + st_name_idx : strtab_offset + st_name_idx + 64]
            .split(b"\x00")[0]
            .decode(errors="replace")
        )
    return out


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
    except Exception as e:
        log.debug("ELF parse failed: %s", e)
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
        has_imm = False
        imm_value = 0
        imm_size = 0

        for op in insn.operands:
            if op.type == _OP_IMM:
                has_imm = True
                imm_value = op.imm
                imm_size = op.size or _guess_imm_width(imm_value)
                break

        if not has_imm:
            continue

        # Skip small/noise immediates
        if imm_size <= 0 or imm_size > 8:
            continue

        # Filter out uninteresting values
        if _is_noise_immediate(imm_value, imm_size):
            continue

        if insn.insn_id in TARGET_IDS:
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


def _size_from_blocks(block_count: int, ctx_bits: int | None) -> int:
    """Entries needed for ``block_count`` instrumented blocks."""
    edges = block_count * EDGES_PER_BLOCK * ctx_inflation_factor(ctx_bits)
    needed = int(edges / TARGET_LOAD_FACTOR)
    return max(MAP_SIZE_DEFAULT, min(_map_size_max(), _next_power_of_2(needed)))


class MapSizeEstimate(NamedTuple):
    """Where a map size came from, not just what it was.

    `source` is the tier that produced `blocks`:

    - ``"sancov_guards"`` — exact, ``__sancov_guards`` (trace-pc-guard).
    - ``"sancov_cntrs"``  — exact, ``__sancov_cntrs`` (inline-8bit-counters).
    - ``"profile"``       — TargetProfile.total_branches.
    - ``"branch_density"`` — disassembly estimate. Approximate.
    - ``"default"``       — nothing worked; MAP_SIZE_DEFAULT.

    The first two are measurements and the rest are guesses, and the gap
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
    capped: bool

    @property
    def exact(self) -> bool:
        """True when `blocks` was read out of the binary, not estimated."""
        return self.source in ("sancov_guards", "sancov_cntrs")


def estimate_map_size_detail(target: str, profile: object | None = None) -> MapSizeEstimate:
    """`estimate_map_size()`, with the provenance of the answer attached.

    Args:
        target: Path to ELF binary.
        profile: Optional TargetProfile with precomputed static analysis.

    Returns:
        MapSizeEstimate. `entries` is what estimate_map_size() returns.
    """
    ctx_bits = detect_ctx_bits(target) or 0

    blocks = 0
    source = "default"

    # 1. sancov, exact block count for instrumented binaries. trace-pc-guard
    #    (__sancov_guards) first: it is what build_targets.sh emits.
    #    inline-8bit-counters (__sancov_cntrs) after, for externally built
    #    targets -- one *byte* per block there, not one uint32.
    guards = parse_sancov_guard_count(target)
    if guards:
        blocks, source = guards, "sancov_guards"
    else:
        offsets = parse_sancov_offsets(target)
        if offsets and offsets[1] > offsets[0]:
            blocks, source = offsets[1] - offsets[0], "sancov_cntrs"

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
        return MapSizeEstimate(MAP_SIZE_DEFAULT, 0, "default", ctx_bits, False)

    entries = _size_from_blocks(blocks, ctx_bits)
    return MapSizeEstimate(entries, blocks, source, ctx_bits, entries >= _map_size_max())


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

    try:
        return _extract_div_pure(text_data, text_vaddr)
    except Exception:
        return {}, set()


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
        # 1) Remove registers overwritten by this instruction (preserve EDX)
        for r in regs_write:
            if r not in _dx_family:
                _rem_regs.discard(r)
        # 2) MOV dest, src where src carries the remainder → track dest too
        if insn.insn_id == _INS_MOV and len(insn.operands) == 2:
            d, s = insn.operands[0], insn.operands[1]
            if d.type == _OP_REG and s.type == _OP_REG and s.reg in _rem_regs:
                _rem_regs.add(d.reg)
        # 3) DIV/IDIV puts the remainder in EDX
        if insn.insn_id in (_INS_DIV, _INS_IDIV):
            _rem_regs = set(_dx_family)

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
            and any(op.type == _OP_REG and op.reg in _rem_regs for op in insn.operands)
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
