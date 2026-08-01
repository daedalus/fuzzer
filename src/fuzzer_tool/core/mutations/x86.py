"""Structure-aware x86/x86-64 instruction mutator.

Uses a compact length-only decoder (NOT elf._decode_x86_64 — that
decoder yields length=1 for unknown opcodes, which poisons boundary
alignment when splitting a byte stream into instructions). Unknown
opcodes here also fall back to length 1, but every structural mutation
re-runs the decoder on the result so cached lengths are never trusted.

Instruction structure considered:
  [legacy prefixes: 66 67 F0 F2 F3 2E 36 3E 26 64 65]
  [REX: 40-4F, once after prefixes]
  [opcode (1 or 2 bytes with 0F escape)]
  [modrm/sib/disp]  [immediate]

Immediate width follows the 66 prefix (operand16) and REX.W.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

LEGACY_PREFIXES = {0x66, 0x67, 0xF0, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65}

# rel8 conditional jumps (0x70-0x7F)
JCC_REL8 = list(range(0x70, 0x80))

# Interesting immediate values
IMM_VALUES = [0, 1, 2, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x7FFFFFFF, 0x80000000]

# Interesting displacement values
DISP_VALUES = [0, 1, 4, 8, 0xFF, 0x100, 0xFFFF, 0xFFFFFF, 0x7FFFFFFF]

# Same-length opcode swap sets (byte-for-byte)
NOP_INTS = {0x90: 0xCC, 0xCC: 0x90}  # nop <-> int3

# ModRM field values for modrm_field_flip
MODRM_MOD_VALUES = [0, 1, 2, 3]
MODRM_REG_VALUES = list(range(8))
MODRM_RM_VALUES = list(range(8))


@dataclass
class Insn:
    """A single decoded instruction."""

    raw: bytes
    group: str
    length: int
    modrm_off: int = -1  # offset of the modrm byte within raw, or -1
    imm_off: int = -1  # offset of the immediate within raw, or -1
    imm_size: int = 0
    disp_off: int = -1  # offset of the displacement within raw, or -1
    disp_size: int = 0


def _consume_modrm(data: bytes, n: int, pc: int) -> tuple[int, int, int, int] | None:
    """Consume modrm + sib + displacement at *pc*.

    Returns (new_pc, modrm_off, disp_off, disp_size), or None if truncated.
    """
    if pc >= n:
        return None
    modrm_off = pc
    mrm = data[pc]
    pc += 1
    mod = (mrm >> 6) & 3
    rm = mrm & 7
    disp_off = -1
    disp_size = 0
    if mod != 3 and rm == 4:
        # SIB byte follows
        if pc >= n:
            return None
        sib = data[pc]
        pc += 1
        if mod == 0 and (sib & 7) == 5:
            if pc + 4 > n:
                return None
            disp_off = pc
            disp_size = 4
            pc += 4
    if mod == 1:
        if pc >= n:
            return None
        disp_off = pc
        disp_size = 1
        pc += 1
    elif mod in (0, 2) and (mod == 2 or rm == 5):
        if pc + 4 > n:
            return None
        disp_off = pc
        disp_size = 4
        pc += 4
    return pc, modrm_off, disp_off, disp_size


def _decode_insns(data: bytes) -> list[Insn]:
    """Length-only linear sweep decoder."""
    insns: list[Insn] = []
    pc = 0
    n = len(data)
    while pc < n:
        start = pc
        while pc < n and data[pc] in LEGACY_PREFIXES:
            pc += 1
        operand16 = 0x66 in data[start:pc]
        rex_w = False
        if pc < n and 0x40 <= data[pc] <= 0x4F:
            rex_w = bool(data[pc] & 8)
            pc += 1
        if pc >= n:
            insns.append(Insn(raw=data[start:pc], group="trunc", length=pc - start))
            break
        op = data[pc]
        pc += 1

        modrm_off = -1
        imm_off = -1
        imm_size = 0
        disp_off = -1
        disp_size = 0
        group = "other"
        truncated = False

        def need_modrm() -> bool:
            nonlocal truncated, pc, modrm_off, disp_off, disp_size
            result = _consume_modrm(data, n, pc)
            if result is None:
                truncated = True
                return False
            pc, modrm_off, disp_off, disp_size = result
            return True

        def need_imm(size: int) -> bool:
            nonlocal truncated, pc, imm_off, imm_size
            if pc + size > n:
                truncated = True
                return False
            imm_off = pc
            imm_size = size
            pc += size
            return True

        # ── primary opcode classes ──
        if 0x00 <= op <= 0x3F:
            group = "alu"
            need_modrm()
        elif 0x40 <= op <= 0x5F:
            # push/pop reg (REX already consumed, so only 50-5F reach here
            # except when the byte was a bare inc/dec in 32-bit mode)
            group = "push" if 0x50 <= op <= 0x57 else ("pop" if 0x58 <= op <= 0x5F else "alu")
        elif op in (0x62, 0x63):
            group = "mov"
            need_modrm()
        elif op == 0x68:
            group = "push"
            need_imm(2 if operand16 else 4)
        elif op == 0x69:
            group = "imul"
            if need_modrm():
                need_imm(2 if operand16 else 4)
        elif op == 0x6A:
            group = "push"
            need_imm(1)
        elif op == 0x6B:
            group = "imul"
            if need_modrm():
                need_imm(1)
        elif 0x70 <= op <= 0x7F:
            group = "jcc"
            need_imm(1)
        elif 0x80 <= op <= 0x83:
            group = "alu"
            if need_modrm():
                need_imm(1 if op in (0x80, 0x82, 0x83) else (2 if operand16 else 4))
        elif 0x88 <= op <= 0x8B:
            group = "mov"
            need_modrm()
        elif op == 0x8F:
            group = "pop"
            need_modrm()
        elif 0x90 <= op <= 0x97:
            group = "nop" if op == 0x90 else "other"
        elif 0x98 <= op <= 0x9F:
            # cbw/cdq/cwd/cwde/callf/pushf/popf/sahf/lahf
            if op == 0x9A:
                group = "call"
                need_imm(4)  # far pointer ptr16:32
            elif op == 0x9B:
                group = "other"  # wait
            else:
                group = "other"
        elif 0xB0 <= op <= 0xB7:
            group = "mov"
            need_imm(1)
        elif 0xB8 <= op <= 0xBF:
            group = "mov"
            need_imm(8 if rex_w else (2 if operand16 else 4))
        elif op in (0xC0, 0xC1):
            group = "alu"
            if need_modrm():
                need_imm(1)
        elif op == 0xC2:
            group = "ret"
            need_imm(2)
        elif op == 0xC3:
            group = "ret"
        elif op in (0xC4, 0xC5):
            # LES/LDS (also VEX prefix start on newer CPUs)
            need_modrm()
        elif op == 0xC6:
            group = "mov"
            if need_modrm():
                need_imm(1)
        elif op == 0xC7:
            group = "mov"
            if need_modrm():
                need_imm(2 if operand16 else 4)
        elif op == 0xC8:
            if need_imm(2):
                need_imm(1)  # enter imm16, imm8
        elif op == 0xC9:
            pass
        elif op in (0xCA,):
            group = "ret"
            need_imm(2)
        elif op == 0xCB:
            group = "ret"
        elif op == 0xCC:
            group = "int3"
        elif op == 0xCD:
            group = "int"
            need_imm(1)
        elif op == 0xCE:
            group = "int"
        elif 0xD0 <= op <= 0xD3:
            group = "alu"
            need_modrm()
        elif op in (0xD6, 0xD7):
            pass  # salc/xlat
        elif 0xD8 <= op <= 0xDF:
            group = "x87"
            need_modrm()
        elif 0xE0 <= op <= 0xE3:
            group = "loop"
            need_imm(1)
        elif 0xE4 <= op <= 0xE7:
            need_imm(1)  # in/out imm8
        elif op in (0xE8, 0xE9):
            group = "call" if op == 0xE8 else "jmp"
            need_imm(2 if operand16 else 4)
        elif op == 0xEA:
            group = "jmp"
            need_imm(4)  # far jmp ptr16:32
        elif op == 0xEB:
            group = "jmp"
            need_imm(1)
        elif 0xEC <= op <= 0xEF:
            pass  # in/out dx
        elif op in (0xF4, 0xF5):
            pass  # hlt/cmc
        elif op in (0xF6, 0xF7):
            group = "alu"
            if need_modrm():
                reg = (data[modrm_off] >> 3) & 7
                if op == 0xF6 and reg == 0:
                    group = "test"
                    need_imm(1)
                elif op == 0xF7 and reg in (0, 1):
                    group = "test"
                    need_imm(2 if operand16 else 4)
        elif 0xF9 <= op <= 0xFD:
            pass  # stc/clc/sti/cld/std
        elif op in (0xFE, 0xFF):
            group = "alu"
            need_modrm()
        elif op == 0x0F:
            # two-byte escape
            if pc >= n:
                truncated = True
            else:
                op2 = data[pc]
                pc += 1
                if 0x80 <= op2 <= 0x8F:
                    group = "jcc"
                    need_imm(2 if operand16 else 4)
                elif 0x90 <= op2 <= 0x9F:
                    group = "setcc"
                    need_modrm()
                elif op2 in (0xA0, 0xA1, 0xA2, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB8):
                    pass  # push/pop fs/gs, cpuid, cmpxchg, rsm, jmpe
                elif op2 in (0xA4, 0xA5, 0xAC, 0xAD):
                    group = "alu"  # shld/shrd imm8
                    if need_modrm():
                        need_imm(1)
                elif op2 in (
                    0xA3,
                    0xAB,
                    0xAE,
                    0xAF,
                    0xB0,
                    0xB1,
                    0xB2,
                    0xB3,
                    0xB4,
                    0xB5,
                    0xB6,
                    0xB7,
                    0xB9,
                    0xBB,
                    0xBC,
                    0xBD,
                    0xBE,
                    0xBF,
                    0xC0,
                    0xC1,
                    0xC3,
                    0xC7,
                ):
                    need_modrm()
                elif op2 in (0xBA, 0xC2, 0xC4, 0xC5, 0xC6):
                    if need_modrm():
                        need_imm(1)
                else:
                    pass  # unknown 0F opcode — 2 bytes
                # TODO follow-up: 3-byte 0F 38 / 0F 3A opcode maps unsupported —
                # unknown second byte stays a 2-byte resync unit.
        else:
            pass  # unknown opcode — 1 byte, resync

        if truncated:
            pc = n
        length = pc - start
        insns.append(
            Insn(
                raw=bytes(data[start:pc]),
                group=group,
                length=length,
                modrm_off=modrm_off - start if modrm_off >= 0 else -1,
                imm_off=imm_off - start if imm_off >= 0 else -1,
                imm_size=imm_size,
                disp_off=disp_off - start if disp_off >= 0 else -1,
                disp_size=disp_size,
            )
        )
    return insns


def serialize_x86(insns: list[Insn]) -> bytes:
    """Concatenate decoded instructions back to bytes."""
    return b"".join(i.raw for i in insns)


def _pack_imm(value: int, size: int) -> bytes:
    return value.to_bytes(size, "little", signed=False)[:size]


class X86Mutator:
    """Structure-aware x86/x86-64 mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = rng or random
        insns = _decode_insns(data)
        if not insns:
            return self._generate_random_x86(max_len, rng=self._rng)

        op = self._rng.randint(0, 11)
        mutators = [
            self._opcode_class_swap,
            self._modrm_field_flip,
            self._imm_mutate,
            self._disp_mutate,
            self._prefix_toggle,
            self._nop_replace,
            self._delete_insn,
            self._duplicate_insn,
            self._swap_insns,
            self._truncate_boundary,
            self._splice,
            self._generate_random_x86,
        ]
        result = mutators[op](insns, max_len)
        if isinstance(result, list):
            return serialize_x86(result)[:max_len]
        return result[:max_len]

    def _opcode_class_swap(self, insns: list[Insn], max_len: int) -> list[Insn]:
        """Swap an opcode for a same-length class equivalent."""
        target = self._rng.choice(insns)
        raw = bytearray(target.raw)
        if not raw:
            return insns
        if target.group in ("nop", "int3"):
            raw[0] = NOP_INTS.get(raw[0], 0x90)
        elif target.group == "ret" and len(raw) == 1:
            raw[0] = 0xC3 if raw[0] == 0xCB else 0xCB
        elif target.group == "ret" and len(raw) == 3:
            raw[0] = 0xCA if raw[0] == 0xC2 else 0xC2
        elif target.group == "jcc" and target.imm_size == 1:
            # swap rel8 jcc for another rel8 jcc (same length)
            raw[0] = self._rng.choice([j for j in JCC_REL8 if j != raw[0]])
        elif target.group == "jcc" and target.imm_size in (2, 4):
            # 0F 8x rel32: swap second byte within 0x80-0x8F
            if len(raw) >= 2:
                raw[1] = self._rng.choice([b for b in range(0x80, 0x90) if b != raw[1]])
        elif target.group == "alu" and len(raw) >= 1 and 0x00 <= raw[0] <= 0x3F:
            # swap between add/or/adc/sbb/and/sub/xor/cmp families
            fam = raw[0] & 0xF8
            raw[0] = fam | ((raw[0] + 1) & 7)
        target.raw = bytes(raw)
        return insns

    def _modrm_field_flip(self, insns: list[Insn], max_len: int) -> list[Insn]:
        """Flip mod/reg/rm fields of a modrm byte."""
        targets = [i for i in insns if i.modrm_off >= 0]
        if not targets:
            return insns
        target = self._rng.choice(targets)
        raw = bytearray(target.raw)
        mrm = raw[target.modrm_off]
        part = self._rng.choice(["mod", "reg", "rm"])
        if part == "mod":
            new = self._rng.choice([m for m in MODRM_MOD_VALUES if m != ((mrm >> 6) & 3)])
            mrm = (mrm & 0x3F) | (new << 6)
        elif part == "reg":
            new = self._rng.choice([r for r in MODRM_REG_VALUES if r != ((mrm >> 3) & 7)])
            mrm = (mrm & 0xC7) | (new << 3)
        else:
            new = self._rng.choice([r for r in MODRM_RM_VALUES if r != (mrm & 7)])
            mrm = (mrm & 0xF8) | new
        raw[target.modrm_off] = mrm
        target.raw = bytes(raw)
        return insns

    def _imm_mutate(self, insns: list[Insn], max_len: int) -> list[Insn]:
        """Mutate an immediate field to an interesting value."""
        targets = [i for i in insns if i.imm_off >= 0 and i.imm_size > 0]
        if not targets:
            return insns
        target = self._rng.choice(targets)
        raw = bytearray(target.raw)
        value = self._rng.choice(
            IMM_VALUES + [self._rng.randint(0, (1 << min(32, target.imm_size * 8)) - 1)]
        )
        mask = (1 << (target.imm_size * 8)) - 1
        raw[target.imm_off : target.imm_off + target.imm_size] = _pack_imm(
            value & mask, target.imm_size
        )
        target.raw = bytes(raw)
        return insns

    def _disp_mutate(self, insns: list[Insn], max_len: int) -> list[Insn]:
        """Mutate a displacement field."""
        targets = [i for i in insns if i.disp_off >= 0 and i.disp_size > 0]
        if not targets:
            return insns
        target = self._rng.choice(targets)
        raw = bytearray(target.raw)
        value = self._rng.choice(
            DISP_VALUES + [self._rng.randint(0, (1 << min(32, target.disp_size * 8)) - 1)]
        )
        mask = (1 << (target.disp_size * 8)) - 1
        raw[target.disp_off : target.disp_off + target.disp_size] = _pack_imm(
            value & mask, target.disp_size
        )
        target.raw = bytes(raw)
        return insns

    def _prefix_toggle(self, insns: list[Insn], max_len: int) -> list[Insn]:
        """Add or remove a legacy prefix / REX.W on a random instruction."""
        target = self._rng.choice(insns)
        raw = bytearray(target.raw)
        if not raw:
            return insns
        if raw[0] in LEGACY_PREFIXES:
            raw.pop(0)
        else:
            raw.insert(0, self._rng.choice([0x66, 0x67, 0xF2, 0xF3, 0x48]))
        target.raw = bytes(raw)
        return insns

    def _nop_replace(self, insns: list[Insn], max_len: int) -> list[Insn]:
        """Replace a random instruction with a same-length NOP."""
        target = self._rng.choice(insns)
        length = target.length
        if length == 1:
            target.raw = b"\x90"
        elif length == 2:
            target.raw = b"\x66\x90"
        elif length == 3:
            target.raw = b"\x0f\x1f\x00"
        else:
            target.raw = b"\x90" * length
        target.group = "nop"
        return insns

    def _delete_insn(self, insns: list[Insn], max_len: int) -> list[Insn]:
        if len(insns) > 1:
            insns.pop(self._rng.randint(0, len(insns) - 1))
        return insns

    def _duplicate_insn(self, insns: list[Insn], max_len: int) -> list[Insn]:
        if insns:
            idx = self._rng.randint(0, len(insns) - 1)
            orig = insns[idx]
            dup = Insn(raw=orig.raw[:], group=orig.group, length=orig.length)
            insns.insert(idx + 1, dup)
        return insns

    def _swap_insns(self, insns: list[Insn], max_len: int) -> list[Insn]:
        if len(insns) >= 2:
            i, j = self._rng.sample(list(range(len(insns))), 2)
            insns[i], insns[j] = insns[j], insns[i]
        return insns

    def _truncate_boundary(self, insns: list[Insn], max_len: int) -> list[Insn]:
        """Truncate the instruction stream at a random boundary."""
        if insns:
            cut = self._rng.randint(0, len(insns) - 1)
            del insns[cut:]
        return insns

    def _splice(self, insns: list[Insn], max_len: int) -> bytes:
        """Byte-level splice of the serialized stream."""
        raw = bytearray(serialize_x86(insns))
        if len(raw) >= 4:
            a = self._rng.randint(0, len(raw) - 1)
            b = self._rng.randint(0, len(raw) - 1)
            raw[a], raw[b] = raw[b], raw[a]
        return bytes(raw)

    def _generate_random_x86(self, _insns=None, max_len: int = 4096, rng=None) -> bytes:
        """Generate a random x86 byte stream with injected NOP padding."""
        self._rng = rng or self._rng
        out = bytearray()
        for _ in range(self._rng.randint(1, 16)):
            choice = self._rng.randint(0, 3)
            if choice == 0:
                out.append(0x90)  # nop
            elif choice == 1:
                out.extend(b"\x0f\x1f\x00")  # 3-byte nop
            elif choice == 2:
                out.append(0xC3)  # ret
            else:
                out.extend(self._rng.randbytes(self._rng.randint(1, 4)))
        return bytes(out)[:max_len]
