"""ROP/JOP gadget detection and chain-feasibility synthesis for crash triage.

When a crash is found the next question is whether it is exploitable: does the
target's own executable code hold the gadgets an attacker would chain after
hijacking control flow? This module answers that *statically* from the target
ELF -- it never executes anything, it only reads the executable sections.

It is the static half of what ROPgadget / ropper / pwntools' ROP do: sweep the
executable sections for terminated instruction sequences (the classic
"galileo" unaligned walk-back on x86-64, 4-byte-aligned heads on aarch64),
classify each (pop-reg, syscall, stack-pivot, JOP dispatch, ...), report which
argument registers a gadget loads from the chain stack, and whether a syscall
or ret2libc chain exists. "Generation" here is a capability *skeleton* -- the
ordered gadget addresses a chain would use -- not a filled-in payload; the
analyst supplies the values.

    exec sections ─► scan() ─► [Gadget] ─► build_report() ─► GadgetReport
       │                │                                          │
       │       galileo walk-back (x86-64)          reg control + syscall
       │       aligned heads (aarch64)             + ret2libc => skeletons
       └─ plt_map() (core.elf) ─► call targets ──────────┘

The decode layers are deliberately small. x86-64 reuses ``core.elf``'s shared
decoder for general instructions and recognises only the extra ROP-relevant
opcodes (push/pop/leave/nop/syscall/int, xchg rax,rsp, add/sub rsp, mov
r64,imm64) that the shared decoder -- built for CFG/coverage, where
instruction *length* and block boundaries are what matter -- leaves as unknown
bytes. The aarch64 decoder recognises only the ROP-relevant encodings (ldp/ldr
from sp, sp arithmetic/pivot, ret/br/blr, svc, nop) and declines everything
else. Keeping that knowledge here, rather than widening the shared decoder,
protects its other callers (CFG, ptrace BB discovery, map-size estimation),
which depend on its current length output.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field

from fuzzer_tool.core.elf import (
    _GRP_CALL,
    _GRP_INT,
    _GRP_JUMP,
    _GRP_RET,
    _INS_ADD,
    _INS_AND,
    _INS_CMP,
    _INS_CMPXCHG,
    _INS_LEA,
    _INS_MOV,
    _INS_MOVABS,
    _INS_OR,
    _INS_OTHER,
    _INS_SUB,
    _INS_TEST,
    _INS_XOR,
    _REX_W,
    _decode_x86_64,
    _iter_exec_sections,
    _read_target_elf,
    _static_symbol_addr,
    machine_arch,
    plt_map,
)

# ── x86-64 encoding facts ───────────────────────────────────────────────

_RET = 0xC3  # ret (near)
_RET_IMM = 0xC2  # ret imm16 -> opcode + 2 immediate bytes
_REX_LO, _REX_HI = 0x40, 0x4F
_REX_B = 0x01  # REX.B extends the push/pop register to r8..r15

# r64 register file, indexed by the 3-bit opcode field (+8 with REX.B).
_R64 = (
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
)  # fmt: skip

# SysV syscall/call ABI, per arch: nr register and argument registers, in order.
_SYSCALL_NR_REG = {"x86_64": "rax", "aarch64": "x8"}
_ARG0_REG = {"x86_64": "rdi", "aarch64": "x0"}
_ARG_REGS = {
    "x86_64": ("rdi", "rsi", "rdx", "r10", "r8", "r9"),
    "aarch64": ("x0", "x1", "x2", "x3", "x4", "x5"),
}

# Mnemonics for the shared decoder's recognised ids, for best-effort render of
# the "generic" instructions inside a gadget (operands come from op_str).
_MNEMONIC = {
    _INS_MOV: "mov", _INS_MOVABS: "movabs", _INS_XOR: "xor", _INS_CMP: "cmp",
    _INS_LEA: "lea", _INS_TEST: "test", _INS_AND: "and", _INS_OR: "or",
    _INS_SUB: "sub", _INS_ADD: "add", _INS_CMPXCHG: "cmpxchg",
}  # fmt: skip

# Instruction groups that end a basic block: one inside a candidate gadget
# means the gadget really ends there, so that start offset is rejected.
_CTRL_GROUPS = frozenset({_GRP_CALL, _GRP_JUMP, _GRP_RET, _GRP_INT})

# Kinds that cannot sit *inside* a ret-terminated gadget.
_REJECT_KINDS = frozenset({"ctrl", "priv"})

_MAX_X86_LEN = 15  # architectural cap on one x86-64 instruction
_MAX_BACK = 10  # bytes to walk back from a ret looking for a gadget head
_MAX_INSNS = 6  # instructions before the ret (longer gadgets are rarely useful)

# Bounds (Hard Rule 54): a huge target (ffmpeg) must not blow time or memory.
_MAX_TEXT = 1 << 21  # scan at most 2 MiB per exec section
_MAX_GADGETS = 4096
_SAMPLE_CAP = 32
# scan_target sweep budget: at most 16 exec sections, 4 MiB cumulative code
# per target -- a per-crash triage aid, cached per target, stays bounded even
# on a hostile binary declaring hundreds of exec sections.
_MAX_SECTIONS = 16
_MAX_TOTAL_TEXT = 1 << 22

# ret2libc inventory: call targets a ret-chain can reach once registers are
# controlled. `system` alone drives feasibility (one argument, the command
# string); the rest are reported as available targets for the analyst.
_LIBC_SINK = "system"
_LIBC_FUNCS = (
    "system", "popen", "execl", "execlp", "execv", "execve", "execvp",
    "mprotect", "mmap", "dlopen", "dlsym", "dup2", "setuid",
)  # fmt: skip


# ── Decoded-instruction model ───────────────────────────────────────────


@dataclass(frozen=True)
class _Insn:
    """One decoded instruction inside a candidate gadget."""

    length: int
    text: str
    # pop | push | ret | leave | nop | syscall | int | ctrl | priv | generic
    # | pivot (xchg rax,rsp / mov sp,xN) | adj (add/sub sp, imm)
    # | movimm (mov r64, imm64) | jop (call/jmp r64 tail)
    kind: str
    regs: tuple[str, ...] = ()  # r64s loaded from the chain stack (pop/ldp/ldr)


def _r64(idx: int, rex: int) -> str:
    """Register name for a push/pop opcode field, honouring REX.B."""
    return _R64[idx + (8 if rex & _REX_B else 0)]


def _prim_opcode(buf: bytes, p: int, n: int, b: int, rex: int, start: int) -> _Insn | None:
    """Recognise a ROP-relevant opcode the shared decoder leaves as unknown.

    ``p`` indexes the opcode byte ``b`` (after any REX at ``start``); ``n`` is
    the buffer length. Returns None for anything not handled here, which falls
    back to the shared decoder.
    """
    length = p - start + 1

    if 0x50 <= b <= 0x57:  # push r64
        reg = _r64(b - 0x50, rex)
        return _Insn(length, f"push {reg}", "push", (reg,))

    if 0x58 <= b <= 0x5F:  # pop r64
        reg = _r64(b - 0x58, rex)
        return _Insn(length, f"pop {reg}", "pop", (reg,))

    if b == 0xC9:  # leave (mov rsp,rbp ; pop rbp)
        return _Insn(length, "leave", "leave")

    if b == 0x90 and rex == 0:  # nop
        return _Insn(length, "nop", "nop")

    if b == 0x0F and p + 1 < n and buf[p + 1] == 0x05:  # syscall
        return _Insn(length + 1, "syscall", "syscall")

    if b == 0xCD and p + 1 < n:  # int imm8 (0xcd 0x80 = int 0x80)
        return _Insn(length + 1, f"int 0x{buf[p + 1]:x}", "int")

    if b == 0xF4:  # hlt -- privileged, cannot sit inside a gadget
        return _Insn(length, "hlt", "priv")

    sp = _prim_sp(buf, p, n, b, rex, length)
    if sp is not None:
        return sp
    return _prim_movimm(buf, p, n, b, rex, length)


def _prim_sp(buf: bytes, p: int, n: int, b: int, rex: int, length: int) -> _Insn | None:
    """Stack-pointer primitves the shared decoder leaves unknown.

    - REX.W + xchg rax, rsp (48 94): the classic xchg pivot. Without REX.W
      the swap is 32-bit and truncates rsp, so it stays with the shared
      decoder -- not claimed as a pivot.
    - add/sub rsp, imm8/imm32 (48 83/81 C4/EC ..): fixed-size stack adjust;
      REX.B would remap rm=100 from rsp to r12, so it excludes those forms.
    """
    if b == 0x94 and rex & _REX_W and not rex & _REX_B:
        return _Insn(length, "xchg rax, rsp", "pivot")

    if b not in (0x81, 0x83) or not rex & _REX_W or rex & _REX_B or p + 1 >= n:
        return None
    if buf[p + 1] not in (0xC4, 0xEC):
        return None

    imm_len = 1 if b == 0x83 else 4
    if p + 2 + imm_len > n:
        return None
    raw = int.from_bytes(buf[p + 2 : p + 2 + imm_len], "little", signed=True)
    op = "add" if buf[p + 1] == 0xC4 else "sub"
    sign = "-" if raw < 0 else ""
    return _Insn(length + 1 + imm_len, f"{op} rsp, {sign}0x{abs(raw):x}", "adj")


def _prim_movimm(buf: bytes, p: int, n: int, b: int, rex: int, length: int) -> _Insn | None:
    """mov r64, imm64 (48/49 B8+r ..): constant load.

    The immediate comes from the gadget's own bytes, NOT from the chain
    stack, so the register is reported in the movimm category but never in
    reg_control.
    """
    if not 0xB8 <= b <= 0xBF or not rex & _REX_W or p + 9 > n:
        return None
    reg = _r64(b - 0xB8, rex)
    imm = int.from_bytes(buf[p + 1 : p + 9], "little")
    return _Insn(length + 8, f"mov {reg}, 0x{imm:x}", "movimm", (reg,))


def _rop_prim(buf: bytes, off: int, n: int) -> _Insn | None:
    """ROP-primitive at ``off`` (consuming one optional REX), else None."""
    p = off
    rex = 0
    b = buf[p]

    if _REX_LO <= b <= _REX_HI:
        rex = b
        p += 1
        if p >= n:
            return None
        b = buf[p]

    return _prim_opcode(buf, p, n, b, rex, off)


def _render(insn) -> str:
    """Best-effort text for a shared-decoder instruction."""
    mnem = _MNEMONIC.get(insn.insn_id, "?")
    return f"{mnem} {insn.op_str}".strip() if insn.op_str else mnem


def _fallback(buf: bytes, off: int, n: int) -> _Insn | None:
    """Decode one general instruction via the shared x86-64 decoder.

    None when the byte is not understood (so the gadget is not "clean") -- we
    only report sequences we fully decode.
    """
    try:
        insn = next(_decode_x86_64(buf[off : off + _MAX_X86_LEN], 0))
    except StopIteration:
        return None

    if insn.insn_id == _INS_OTHER:
        return None

    kind = "ctrl" if insn.groups & _CTRL_GROUPS else "generic"
    return _Insn(length=insn.length, text=_render(insn), kind=kind)


def _decode_one(buf: bytes, off: int, n: int) -> _Insn | None:
    """One instruction at ``off``: ROP primitive first, then shared decoder."""
    prim = _rop_prim(buf, off, n)
    return prim if prim is not None else _fallback(buf, off, n)


# ── Gadget model + classification ───────────────────────────────────────


@dataclass(frozen=True)
class Gadget:
    """A terminated instruction sequence found in an executable section."""

    vaddr: int  # virtual address of the gadget head
    text: str  # "insn ; insn ; ret"
    category: str  # pop | adj | movimm | syscall | pivot | jop | ret | generic
    pop_regs: tuple[str, ...] = ()  # regs a pop-gadget loads from the stack
    has_syscall: bool = False
    is_pivot: bool = False


def _is_pivot(insn: _Insn) -> bool:
    """True when ``insn`` moves the stack pointer under attacker control."""
    if insn.kind in ("leave", "pivot"):  # leave, xchg rax,rsp, mov sp,xN
        return True
    return insn.kind == "pop" and "rsp" in insn.regs  # pop rsp


# Tail kinds that end a gadget with a (near or far) return. A pop/adj/movimm
# body only counts as such when the tail returns -- a JOP dispatch tail makes
# the whole gadget a dispatcher, not a stack loader.
_RET_TAILS = ("ret", "retf")


def _ret_body(insns: list[_Insn], kinds: set[str]) -> tuple[str, tuple[str, ...]] | None:
    """(category, pop_regs) for a ret-tailed body of one uniform kind."""
    if kinds == {"pop"}:
        return "pop", tuple(r for i in insns for r in i.regs)
    if kinds == {"adj"}:
        return "adj", ()
    if kinds == {"movimm"}:
        return "movimm", ()
    return None


def _classify(insns: list[_Insn], tail: _Insn) -> tuple[str, tuple[str, ...], bool, bool]:
    """``(category, pop_regs, has_syscall, is_pivot)`` for a gadget body."""
    has_syscall = any(i.kind in ("syscall", "int") for i in insns) or tail.kind == "syscall"
    is_pivot = any(_is_pivot(i) for i in insns)

    if not insns:
        if has_syscall:  # bare svc (aarch64): the syscall site itself
            return "syscall", (), True, is_pivot
        if tail.kind == "jop":
            return "jop", (), has_syscall, is_pivot
        return "ret", (), has_syscall, is_pivot

    if tail.kind in _RET_TAILS:
        body = _ret_body(insns, {i.kind for i in insns})
        if body is not None:
            return (*body, has_syscall, is_pivot)

    if has_syscall:
        return "syscall", (), True, is_pivot

    if is_pivot:
        return "pivot", (), has_syscall, True

    if tail.kind == "jop":
        return "jop", (), has_syscall, is_pivot

    return "generic", (), has_syscall, is_pivot


def _build_gadget(head_va: int, insns: list[_Insn], tail: _Insn) -> Gadget:
    category, pop_regs, has_syscall, is_pivot = _classify(insns, tail)
    text = " ; ".join(i.text for i in insns + [tail])
    return Gadget(head_va, text, category, pop_regs, has_syscall, is_pivot)


# ── Terminators (x86-64) ──────────────────────────────────────────────
#
# A gadget ends at a terminator: the returns (ret, ret imm16, retf, retfq)
# and the register dispatches (call/jmp r64 -- JOP/COP). A call, jmp or ret
# inside a *body* is control flow and rejects the head (_REJECT_KINDS via
# the shared decoder's ctrl group).

_RETF = 0xCB  # retf
_RETFQ = b"\x48\xcb"  # REX.W + retf = retfq


def _jop_tail(buf: bytes, i: int, n: int) -> _Insn | None:
    """``call r64`` / ``jmp r64`` (FF /2 or /4, mod=11) at *i*, else None.

    Memory-indirect forms (mod != 11, e.g. ``call [rax]``) dispatch through
    memory, not through a register, so they are not JOP terminators.
    """
    j = i
    rex = 0
    b = buf[j]

    if _REX_LO <= b <= _REX_HI:  # optional REX before the FF
        rex = b
        j += 1
        if j + 1 >= n:
            return None
        b = buf[j]
    if b != 0xFF or j + 1 >= n:
        return None

    modrm = buf[j + 1]
    if 0xD0 <= modrm <= 0xD7:
        return _Insn(j - i + 2, f"call {_r64(modrm - 0xD0, rex)}", "jop")
    if 0xE0 <= modrm <= 0xE7:
        return _Insn(j - i + 2, f"jmp {_r64(modrm - 0xE0, rex)}", "jop")
    return None


def _terminator(buf: bytes, i: int, n: int) -> _Insn | None:
    """The tail instruction when a terminator starts at *i*, else None."""
    b = buf[i]
    if b == _RET:
        return _Insn(1, "ret", "ret")
    if b == _RET_IMM:  # ret imm16 -- opcode + 2 immediate bytes
        if i + 3 > n:
            return None
        imm = buf[i + 1] | (buf[i + 2] << 8)
        return _Insn(3, f"ret 0x{imm:x}", "ret")
    if b == _RETF:
        return _Insn(1, "retf", "retf")
    if b == _RETFQ[0] and buf[i : i + 2] == _RETFQ:
        return _Insn(2, "retfq", "retf")
    return _jop_tail(buf, i, n)


def _try_head(buf, n, base, start, t, tail, out, decode) -> None:
    """Emit the gadget whose head is ``start`` and whose tail starts at ``t``.

    Forward-decode from ``start`` with *decode* (arch-specific); accept only
    when the instruction boundaries land exactly on ``t`` (so the tail is
    reached, not straddled) and no rejected instruction sits inside. Keyed by
    (head, tail): on aarch64 one head can pair with two tails -- a bare
    ``svc`` and the ``svc; ret`` that continues it -- and both are gadgets.
    """
    insns: list[_Insn] = []
    off = start

    while off < t:
        ins = decode(buf, off, n)
        if ins is None or ins.kind in _REJECT_KINDS or len(insns) >= _MAX_INSNS:
            return
        off += ins.length
        insns.append(ins)

    if off != t:  # straddled the tail boundary -> misaligned start
        return

    out.setdefault((base + start, t), _build_gadget(base + start, insns, tail))


def scan(
    text: bytes, base: int, *, max_gadgets: int = _MAX_GADGETS, arch: str = "x86_64"
) -> list[Gadget]:
    """All distinct gadgets in ``text`` (loaded at ``base``), for one arch.

    x86-64 walks back from every terminator byte (the galileo unaligned
    walk-back); aarch64 scans 4-byte-aligned heads (instruction fetch is
    aligned there, so unaligned heads are not real gadgets).
    """
    if arch == "aarch64":
        return _scan_a64(text, base, max_gadgets=max_gadgets)
    return _scan_x86(text, base, max_gadgets=max_gadgets)


def _scan_x86(text: bytes, base: int, *, max_gadgets: int) -> list[Gadget]:
    buf = text[:_MAX_TEXT]
    n = len(buf)
    out: dict[tuple[int, int], Gadget] = {}

    i = 0
    while i < n:
        tail = _terminator(buf, i, n)
        if tail is not None:
            lo = max(0, i - _MAX_BACK)
            for start in range(i, lo - 1, -1):  # bare tail (start==i) first
                _try_head(buf, n, base, start, i, tail, out, _decode_one)
                if len(out) >= max_gadgets:
                    return list(out.values())

        i += 1

    return list(out.values())


# ── aarch64 decoder ───────────────────────────────────────────────────
#
# Deliberately small, mirroring the x86-64 prim layer: only the ROP-relevant
# encodings (loads from sp, sp arithmetic/pivot, ret/br/blr, svc, nop) are
# decoded; anything else declines the head rather than being guessed. All
# encodings below were verified against capstone 5.0.7.

_A64_INSN = 4  # fixed instruction width
_A64_RET_MASK = 0xFFFFFC1F
_A64_RET = 0xD65F0000  # ret xN (x30 when omitted)
_A64_BR = 0xD61F0000  # br xN
_A64_BLR = 0xD63F0000  # blr xN
_A64_SVC_MASK, _A64_SVC = 0xFFE0001F, 0xD4000001  # svc #imm16
_A64_NOP = 0xD503201F
_A64_PAIR_MASK = 0xFFC00000  # ldp/ldr/add/sub share the top 10 bits layout
_A64_LDP_SIGNED = 0xA9400000  # ldp xT1, xT2, [sp, #imm]
_A64_LDP_PRE = 0xA9C00000  # ldp xT1, xT2, [sp, #imm]!
_A64_LDP_POST = 0xA8C00000  # ldp xT1, xT2, [sp], #imm
_A64_LDR = 0xF9400000  # ldr xT, [sp, #imm]
_A64_ADD, _A64_SUB = 0x91000000, 0xD1000000  # add/sub xD, sp, #imm
_A64_SP = 31  # Rn/Rd field encoding of sp
_A64_LR = 30  # the link register a plain ret returns through


def _a64_fields(word: int) -> tuple[int, int, int]:
    """(Rt/Rd, Rn, Rt2) register fields of one aarch64 word."""
    return word & 31, (word >> 5) & 31, (word >> 10) & 31


def _a64_ldp(word: int) -> _Insn | None:
    """ldp xT1, xT2, [sp, ...] -- the aarch64 double-pop from the stack."""
    rt1, rn, rt2 = _a64_fields(word)
    if rn != _A64_SP:
        return None

    imm7 = (word >> 15) & 0x7F
    if imm7 >= 64:
        imm7 -= 128
    off = imm7 * 8
    off_s = f"-0x{-off:x}" if off < 0 else f"0x{off:x}"

    pair = word & _A64_PAIR_MASK
    if pair == _A64_LDP_SIGNED:
        text = f"ldp x{rt1}, x{rt2}, [sp, #{off_s}]"
    elif pair == _A64_LDP_PRE:
        text = f"ldp x{rt1}, x{rt2}, [sp, #{off_s}]!"
    elif pair == _A64_LDP_POST:
        text = f"ldp x{rt1}, x{rt2}, [sp], #{off_s}"
    else:
        return None
    return _Insn(_A64_INSN, text, "pop", (f"x{rt1}", f"x{rt2}"))


def _a64_ldr(word: int) -> _Insn | None:
    """ldr xT, [sp, #imm] -- single reg loaded from the stack."""
    if word & _A64_PAIR_MASK != _A64_LDR:
        return None
    rt, rn, _rt2 = _a64_fields(word)
    if rn != _A64_SP:
        return None
    off = ((word >> 10) & 0xFFF) * 8
    return _Insn(_A64_INSN, f"ldr x{rt}, [sp, #0x{off:x}]", "pop", (f"x{rt}",))


def _a64_sp_arith(word: int) -> _Insn | None:
    """add/sub with an sp result: fixed adjust, or a pivot from a register."""
    rd, rn, _ = _a64_fields(word)
    imm12 = (word >> 10) & 0xFFF
    add = (word & _A64_PAIR_MASK) == _A64_ADD

    if rd != _A64_SP:
        # add xN, sp, #imm (frame-pointer setup) decodes; nothing else does.
        if add and rn == _A64_SP:
            return _Insn(_A64_INSN, f"add x{rd}, sp, #0x{imm12:x}", "generic")
        return None
    if rn == _A64_SP:
        op = "add" if add else "sub"
        return _Insn(_A64_INSN, f"{op} sp, sp, #0x{imm12:x}", "adj")
    if add and imm12 == 0:  # mov sp, xN -- pivot under register control
        return _Insn(_A64_INSN, f"mov sp, x{rn}", "pivot")
    return None


def _a64_word(insn: bytes, off: int) -> int:
    return struct.unpack_from("<I", insn, off)[0]


def _a64_decode_one(buf: bytes, off: int, n: int) -> _Insn | None:
    """One aarch64 instruction at ``off`` (body view).

    ret/br/blr decode as 'ctrl' so a head that runs into one is rejected;
    svc stays 'syscall' (it may sit mid-gadget, like x86's syscall).
    """
    word = _a64_word(buf, off)

    if word == _A64_NOP:
        return _Insn(_A64_INSN, "nop", "nop")
    if word & _A64_RET_MASK == _A64_RET:
        rn = (word >> 5) & 31
        text = "ret" if rn == _A64_LR else f"ret x{rn}"
        return _Insn(_A64_INSN, text, "ctrl")
    if word & _A64_RET_MASK in (_A64_BR, _A64_BLR):
        op = "br" if word & _A64_RET_MASK == _A64_BR else "blr"
        return _Insn(_A64_INSN, f"{op} x{(word >> 5) & 31}", "ctrl")
    if word & _A64_SVC_MASK == _A64_SVC:
        return _Insn(_A64_INSN, f"svc #0x{(word >> 5) & 0xFFFF:x}", "syscall")

    ldp = _a64_ldp(word)
    if ldp is not None:
        return ldp
    ldr = _a64_ldr(word)
    if ldr is not None:
        return ldr
    return _a64_sp_arith(word)


def _a64_terminator(buf: bytes, i: int) -> _Insn | None:
    """The tail instruction when a terminator starts at *i*, else None.

    ret/br/blr (control flow through a register or the link register) and
    svc end aarch64 gadgets -- unlike x86, a bare svc is itself a gadget:
    libc syscall wrappers have no ret right after the svc.
    """
    word = _a64_word(buf, i)

    if word & _A64_RET_MASK == _A64_RET:
        rn = (word >> 5) & 31
        text = "ret" if rn == _A64_LR else f"ret x{rn}"
        return _Insn(_A64_INSN, text, "ret")
    if word & _A64_RET_MASK in (_A64_BR, _A64_BLR):
        op = "br" if word & _A64_RET_MASK == _A64_BR else "blr"
        return _Insn(_A64_INSN, f"{op} x{(word >> 5) & 31}", "jop")
    if word & _A64_SVC_MASK == _A64_SVC:
        return _Insn(_A64_INSN, f"svc #0x{(word >> 5) & 0xFFFF:x}", "syscall")
    return None


def _scan_a64(text: bytes, base: int, *, max_gadgets: int) -> list[Gadget]:
    buf = text[:_MAX_TEXT]
    m = len(buf) - len(buf) % _A64_INSN  # whole aligned instructions only
    out: dict[tuple[int, int], Gadget] = {}

    i = 0
    while i < m:
        tail = _a64_terminator(buf, i)
        if tail is not None:
            lo = max(0, i - _A64_INSN * _MAX_INSNS)
            for start in range(i, lo - 1, -_A64_INSN):  # bare tail first
                _try_head(buf, m, base, start, i, tail, out, _a64_decode_one)
                if len(out) >= max_gadgets:
                    return list(out.values())

        i += _A64_INSN

    return list(out.values())


# ── Report + chain synthesis ────────────────────────────────────────────


@dataclass
class GadgetReport:
    """Static exploitability summary of a target's gadget inventory."""

    total: int = 0
    truncated: bool = False  # a section clipped, or the sweep budget was hit
    arch: str = "x86_64"
    categories: dict[str, int] = field(default_factory=dict)
    reg_control: dict[str, int] = field(default_factory=dict)  # reg -> head vaddr
    syscall_sites: list[int] = field(default_factory=list)
    pivot_sites: list[int] = field(default_factory=list)
    samples: list[tuple[int, str]] = field(default_factory=list)
    chain_feasible: bool = False
    chain_skeleton: list[tuple[str, int]] = field(default_factory=list)
    chain_note: str = ""
    # Per-exec-section gadget counts (section-sweep attribution).
    sections: dict[str, int] = field(default_factory=dict)
    # Interesting call targets (PLT stubs + defined symbols), for ret2libc.
    libc_targets: dict[str, int] = field(default_factory=dict)
    chain_libc_feasible: bool = False
    chain_libc: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for the crash sidecar (addresses as 0x-hex strings)."""
        return {
            "total": self.total,
            "truncated": self.truncated,
            "arch": self.arch,
            "categories": dict(self.categories),
            "reg_control": {r: f"0x{v:x}" for r, v in self.reg_control.items()},
            "syscall_sites": [f"0x{v:x}" for v in self.syscall_sites],
            "pivot_sites": [f"0x{v:x}" for v in self.pivot_sites],
            "chain_feasible": self.chain_feasible,
            "chain_skeleton": [[role, f"0x{v:x}"] for role, v in self.chain_skeleton],
            "chain_note": self.chain_note,
            "samples": [[f"0x{v:x}", t] for v, t in self.samples],
            "sections": dict(self.sections),
            "libc_targets": {n: f"0x{v:x}" for n, v in self.libc_targets.items()},
            "chain_libc_feasible": self.chain_libc_feasible,
            "chain_libc": [[role, f"0x{v:x}"] for role, v in self.chain_libc],
        }


def _reg_control(gadgets: list[Gadget]) -> dict[str, int]:
    """Shortest ``pop ...; ret`` head address controlling each register.

    Shorter gadgets clobber fewer other registers, so they are preferred.
    """
    best: dict[str, int] = {}
    cost: dict[str, int] = {}
    for g in gadgets:
        if g.category != "pop":
            continue
        n_insns = g.text.count(";")  # pops + ret; fewer is cleaner
        for reg in g.pop_regs:
            if reg not in cost or n_insns < cost[reg]:
                best[reg] = g.vaddr
                cost[reg] = n_insns
    return best


def _pick_samples(gadgets: list[Gadget]) -> list[tuple[int, str]]:
    """A bounded, deterministic sample favouring the useful categories."""
    rank = {
        "pop": 0,
        "syscall": 1,
        "pivot": 2,
        "jop": 3,
        "adj": 4,
        "movimm": 5,
        "generic": 6,
        "ret": 7,
    }
    ordered = sorted(gadgets, key=lambda g: (rank.get(g.category, 9), g.vaddr))
    return [(g.vaddr, g.text) for g in ordered[:_SAMPLE_CAP]]


def _synthesize(reg_control: dict[str, int], syscall_sites: list[int], arch: str):
    """A syscall-chain skeleton from the controllable registers.

    Feasible when a syscall primitive exists and both the syscall number
    register (rax / x8) and the first argument (rdi / x0) are controllable.
    The skeleton lists the nr register, then every controllable arg register
    in ABI order, then the syscall site -- the ordered addresses an analyst
    fills with values.
    """
    nr_reg = _SYSCALL_NR_REG[arch]
    if not syscall_sites or nr_reg not in reg_control or _ARG0_REG[arch] not in reg_control:
        return (
            False,
            [],
            f"no syscall chain: need a syscall site plus {nr_reg} and {_ARG0_REG[arch]} control",
        )

    skeleton: list[tuple[str, int]] = [(nr_reg, reg_control[nr_reg])]
    skeleton += [(r, reg_control[r]) for r in _ARG_REGS[arch] if r in reg_control]
    skeleton.append(("syscall", syscall_sites[0]))

    controlled = ", ".join(role for role, _ in skeleton[:-1])
    note = f"syscall chain feasible: control {controlled}; {len(syscall_sites)} syscall site(s)"
    return True, skeleton, note


def _synthesize_libc(
    reg_control: dict[str, int], libc_targets: dict[str, int], arch: str
) -> tuple[bool, list[tuple[str, int]]]:
    """A ret2libc skeleton: the one-arg sink plus arg0 control.

    ``system(command)`` needs exactly one attacker-controlled argument, so
    feasibility stays honest: a sink stub plus arg0 (rdi / x0) control. Other
    targets in the inventory are named for the analyst, not claimed.
    """
    arg0 = _ARG0_REG[arch]
    sink = libc_targets.get(_LIBC_SINK)
    if sink is None or arg0 not in reg_control:
        return False, []
    return True, [(arg0, reg_control[arg0]), (_LIBC_SINK, sink)]


def build_report(
    gadgets: list[Gadget],
    *,
    truncated: bool = False,
    arch: str = "x86_64",
    sections: dict[str, int] | None = None,
    libc_targets: dict[str, int] | None = None,
) -> GadgetReport:
    """Summarize a gadget list into an exploitability triage report."""
    categories = dict(Counter(g.category for g in gadgets))
    reg_control = _reg_control(gadgets)
    # A set, not a list: on aarch64 a bare svc and its svc;ret continuation
    # are two gadgets at one head address -- one site, listed once.
    syscall_sites = sorted({g.vaddr for g in gadgets if g.has_syscall})
    pivot_sites = sorted(g.vaddr for g in gadgets if g.is_pivot)
    libc_targets = dict(libc_targets) if libc_targets else {}

    feasible, skeleton, note = _synthesize(reg_control, syscall_sites, arch)
    libc_feasible, libc_skeleton = _synthesize_libc(reg_control, libc_targets, arch)
    if libc_feasible:
        note += f"; ret2libc feasible: {_LIBC_SINK} + {_ARG0_REG[arch]} control"
    elif libc_targets:
        note += "; ret2libc targets: " + ", ".join(sorted(libc_targets))

    return GadgetReport(
        total=len(gadgets),
        truncated=truncated,
        arch=arch,
        categories=categories,
        reg_control=reg_control,
        syscall_sites=syscall_sites,
        pivot_sites=pivot_sites,
        samples=_pick_samples(gadgets),
        chain_feasible=feasible,
        chain_skeleton=skeleton,
        chain_note=note,
        sections=dict(sections) if sections else {},
        libc_targets=libc_targets,
        chain_libc_feasible=libc_feasible,
        chain_libc=libc_skeleton,
    )


def _libc_targets(elf: bytes) -> dict[str, int]:
    """Interesting ret2libc targets of *elf*: PLT stubs for the imports,
    plus defined symbols of the same names (a static binary's own code)."""
    targets = {n: a for n, a in plt_map(elf).items() if n in _LIBC_FUNCS}
    for name in _LIBC_FUNCS:
        if name in targets:
            continue
        addr = _static_symbol_addr(elf, name)
        if addr is not None:
            targets[name] = addr
    return targets


def scan_target(target: str) -> GadgetReport | None:
    """Scan a target ELF's executable sections; None when it cannot be read
    or is neither x86-64 nor aarch64.

    Every SHF_EXECINSTR PROGBITS section is swept (a stripped target's code
    section need not be named ``.text``; .plt/.plt.sec carry stubs), under
    the _MAX_SECTIONS/_MAX_TOTAL_TEXT budget -- any clip reports
    truncated=True instead of pretending completeness. Never raises on a
    hostile or truncated binary -- a crash-triage aid must decline such a
    target, not take the fuzzer down with it.
    """
    elf = _read_target_elf(target)
    if elf is None:
        return None
    arch = machine_arch(elf)
    if arch is None:
        return None

    found: list[Gadget] = []
    sections: dict[str, int] = {}
    truncated = False
    budget = _MAX_TOTAL_TEXT
    for name, sh_addr, sh_off, sh_size in _iter_exec_sections(elf):
        if len(sections) >= _MAX_SECTIONS or budget <= 0 or len(found) >= _MAX_GADGETS:
            truncated = True  # more code exists than the budget allows
            break

        data = elf[sh_off : sh_off + sh_size]
        if len(data) > _MAX_TEXT or sh_size > _MAX_TEXT:
            truncated = True
        chunk = data[:_MAX_TEXT]
        budget -= len(chunk)

        room = _MAX_GADGETS - len(found)
        got = scan(chunk, sh_addr, max_gadgets=room, arch=arch)
        if len(got) >= room:
            truncated = True  # the per-target gadget cap clipped this section
        sections[name.decode(errors="replace")] = len(got)
        found.extend(got)

    if not sections:
        return None  # no executable code to scan

    return build_report(
        found,
        truncated=truncated,
        arch=arch,
        sections=sections,
        libc_targets=_libc_targets(elf),
    )
