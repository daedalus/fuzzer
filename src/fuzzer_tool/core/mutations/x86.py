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

from dataclasses import dataclass

from fuzzer_tool.core.mutations.generic import _swap_pair
from fuzzer_tool.core.rand_pool import RandPool

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


# Immediate-size kinds in _PRIMARY (positive ints are fixed byte sizes).
_IMM_Z = -1  # imm16 with a 66 prefix, else imm32
_IMM_V = -2  # imm64 with REX.W, else _IMM_Z
_IMM_ENTER = -3  # ENTER: imm16 then imm8


def _build_primary() -> list[tuple[str, bool, int]]:
    """Primary opcode table: op -> (group, has_modrm, imm_kind).

    Precedence mirrors the original if/elif chain. Note 0x0F sits in the
    0x00-0x3F ALU band, so it decodes as ALU+modrm (see _decode_0f TODO).
    """
    t = [("other", False, 0)] * 256

    def put(ops, group: str, modrm: bool = False, imm: int = 0) -> None:
        for op in ops:
            t[op] = (group, modrm, imm)

    put(range(0x00, 0x40), "alu", True)
    put(range(0x40, 0x50), "alu")  # bare inc/dec (second REX byte)
    put(range(0x50, 0x58), "push")
    put(range(0x58, 0x60), "pop")
    put((0x62, 0x63, 0x88, 0x89, 0x8A, 0x8B), "mov", True)
    put((0x68,), "push", imm=_IMM_Z)
    put((0x69,), "imul", True, _IMM_Z)
    put((0x6A,), "push", imm=1)
    put((0x6B,), "imul", True, 1)
    put(range(0x70, 0x80), "jcc", imm=1)
    put((0x80, 0x82, 0x83, 0xC0, 0xC1, 0xC6), "alu", True, 1)
    put((0x81,), "alu", True, _IMM_Z)
    put((0x8F,), "pop", True)
    put((0x90,), "nop")
    put((0x9A,), "call", imm=4)  # far pointer ptr16:32
    put(range(0xB0, 0xB8), "mov", imm=1)
    put(range(0xB8, 0xC0), "mov", imm=_IMM_V)
    put((0xC2, 0xCA), "ret", imm=2)
    put((0xC3, 0xCB), "ret")
    put((0xC4, 0xC5), "other", True)  # LES/LDS (VEX prefix on newer CPUs)
    put((0xC6,), "mov", True, 1)
    put((0xC7,), "mov", True, _IMM_Z)
    put((0xC8,), "other", imm=_IMM_ENTER)
    put((0xCC,), "int3")
    put((0xCD,), "int", imm=1)
    put((0xCE,), "int")
    put(range(0xD0, 0xD4), "alu", True)
    put(range(0xD8, 0xE0), "x87", True)
    put(range(0xE0, 0xE4), "loop", imm=1)
    put(range(0xE4, 0xE8), "other", imm=1)  # in/out imm8
    put((0xE8,), "call", imm=_IMM_Z)
    put((0xE9,), "jmp", imm=_IMM_Z)
    put((0xEA,), "jmp", imm=4)  # far jmp ptr16:32
    put((0xEB,), "jmp", imm=1)
    put((0xF6, 0xF7, 0xFE, 0xFF), "alu", True)
    return t


_PRIMARY = _build_primary()

# 0F xx second bytes taking modrm (+ imm8 for the _IMM8 set).
_0F_NO_OPERAND = frozenset((0xA0, 0xA1, 0xA2, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB8))
_0F_MODRM_IMM8 = frozenset((0xA4, 0xA5, 0xAC, 0xAD, 0xBA, 0xC2, 0xC4, 0xC5, 0xC6))
_0F_MODRM = frozenset(
    (0xA3, 0xAB, 0xAE, 0xAF, *range(0xB0, 0xB8), 0xB9, *range(0xBB, 0xC2), 0xC3, 0xC7)
)


def _decode_0f(op2: int, operand16: bool) -> tuple[str, bool, int]:
    """Two-byte (0F xx) opcode -> (group, has_modrm, imm_kind).

    TODO: unreachable — 0x0F is claimed by the ALU band in _PRIMARY
    (pre-existing precedence bug). Wire into _decode_one at op == 0x0F.
    3-byte 0F 38 / 0F 3A maps also unsupported.
    """
    if 0x80 <= op2 <= 0x8F:
        return "jcc", False, 2 if operand16 else 4
    if 0x90 <= op2 <= 0x9F:
        return "setcc", True, 0
    if op2 in _0F_NO_OPERAND:
        return "other", False, 0
    if op2 in (0xA4, 0xA5, 0xAC, 0xAD):
        return "alu", True, 1  # shld/shrd imm8
    if op2 in _0F_MODRM_IMM8:
        return "other", True, 1
    return "other", op2 in _0F_MODRM, 0


def _scan_prefixes(data: bytes, n: int, pc: int) -> tuple[int, bool, bool]:
    """Skip legacy prefixes + one REX byte -> (pc, operand16, rex_w)."""
    start = pc
    while pc < n and data[pc] in LEGACY_PREFIXES:
        pc += 1
    operand16 = 0x66 in data[start:pc]
    rex_w = False
    if pc < n and 0x40 <= data[pc] <= 0x4F:
        rex_w = bool(data[pc] & 8)
        pc += 1
    return pc, operand16, rex_w


def _consume_imm(
    data: bytes, n: int, pc: int, kind: int, operand16: bool, rex_w: bool
) -> tuple[int, int, int, bool]:
    """Consume an immediate of *kind* -> (pc, imm_off, imm_size, ok)."""
    if kind == _IMM_ENTER:
        # enter imm16, imm8: the last consumed immediate is recorded
        if pc + 2 > n:
            return pc, -1, 0, False
        if pc + 3 > n:
            return pc + 2, pc, 2, False
        return pc + 3, pc + 2, 1, True
    if kind == _IMM_V:
        size = 8 if rex_w else (2 if operand16 else 4)
    elif kind == _IMM_Z:
        size = 2 if operand16 else 4
    else:
        size = kind
    if pc + size > n:
        return pc, -1, 0, False
    return pc + size, pc, size, True


def _grp3_imm(op: int, modrm: int) -> tuple[str, int]:
    """F6/F7 group 3: only TEST (reg 0, or reg 1 for F7) carries an imm."""
    reg = (modrm >> 3) & 7
    if op == 0xF6 and reg == 0:
        return "test", 1
    if op == 0xF7 and reg in (0, 1):
        return "test", _IMM_Z
    return "alu", 0


def _decode_one(data: bytes, n: int, start: int) -> Insn:
    """Decode one instruction at *start*; a truncated one runs to *n*."""
    pc, operand16, rex_w = _scan_prefixes(data, n, start)
    if pc >= n:
        return Insn(raw=data[start:pc], group="trunc", length=pc - start)
    op = data[pc]
    pc += 1
    group, has_modrm, imm_kind = _PRIMARY[op]

    modrm_off = imm_off = disp_off = -1
    imm_size = disp_size = 0
    ok = True
    if has_modrm:
        res = _consume_modrm(data, n, pc)
        if res is None:
            ok = False
        else:
            pc, modrm_off, disp_off, disp_size = res

    if ok and (op == 0xF6 or op == 0xF7):
        group, imm_kind = _grp3_imm(op, data[modrm_off])
    if ok and imm_kind:
        pc, imm_off, imm_size, ok = _consume_imm(data, n, pc, imm_kind, operand16, rex_w)

    # Truncated operands swallow the rest of the buffer
    end = pc if ok else n
    return Insn(
        raw=bytes(data[start:end]),
        group=group,
        length=end - start,
        modrm_off=modrm_off - start if modrm_off >= 0 else -1,
        imm_off=imm_off - start if imm_off >= 0 else -1,
        imm_size=imm_size,
        disp_off=disp_off - start if disp_off >= 0 else -1,
        disp_size=disp_size,
    )


def _decode_insns(data: bytes) -> list[Insn]:
    """Length-only linear sweep decoder."""
    insns: list[Insn] = []
    pc = 0
    n = len(data)
    while pc < n:
        insn = _decode_one(data, n, pc)
        insns.append(insn)
        pc += insn.length
    return insns


def serialize_x86(insns: list[Insn]) -> bytes:
    """Concatenate decoded instructions back to bytes."""
    return b"".join(i.raw for i in insns)


def _pack_imm(value: int, size: int) -> bytes:
    return value.to_bytes(size, "little", signed=False)[:size]


def _swap_fixed_class(group: str, raw: bytearray) -> None:
    """RNG-free same-length opcode swaps (nop/int3, ret forms, ALU family)."""
    if group in ("nop", "int3"):
        raw[0] = NOP_INTS.get(raw[0], 0x90)
    elif group == "ret" and len(raw) == 1:
        raw[0] = 0xC3 if raw[0] == 0xCB else 0xCB
    elif group == "ret" and len(raw) == 3:
        raw[0] = 0xCA if raw[0] == 0xC2 else 0xC2
    elif group == "alu" and raw[0] <= 0x3F:
        # swap between add/or/adc/sbb/and/sub/xor/cmp families
        fam = raw[0] & 0xF8
        raw[0] = fam | ((raw[0] + 1) & 7)


class X86Mutator:
    """Structure-aware x86/x86-64 mutator."""

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 4096, rng=None) -> bytes:
        self._rng = rng or self._rng
        insns = _decode_insns(data)
        if not insns:
            return self._generate_random_x86(max_len=max_len, rng=self._rng)

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
        if target.group == "jcc":
            self._swap_jcc(target.imm_size, raw)
        else:
            _swap_fixed_class(target.group, raw)
        target.raw = bytes(raw)
        return insns

    def _swap_jcc(self, imm_size: int, raw: bytearray) -> None:
        """Swap a jcc for another same-width condition code."""
        if imm_size == 1:
            # swap rel8 jcc for another rel8 jcc (same length)
            raw[0] = self._rng.choice([j for j in JCC_REL8 if j != raw[0]])
        elif imm_size in (2, 4) and len(raw) >= 2:
            # 0F 8x rel32: swap second byte within 0x80-0x8F
            raw[1] = self._rng.choice([b for b in range(0x80, 0x90) if b != raw[1]])

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
        if (pair := _swap_pair(len(insns), self._rng)) is not None:
            i, j = pair
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
        # An int in the first slot is a max_len passed positionally. Without
        # this the cap lands in the vestigial placeholder and is dropped, and
        # the generator silently falls back to its own default -- the same
        # overload bmp/gzip/jpeg/zlib already handle and document.
        if isinstance(_insns, int):
            max_len = _insns
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
