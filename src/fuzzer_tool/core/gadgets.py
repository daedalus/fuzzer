"""ROP gadget detection and chain-feasibility synthesis for crash triage.

When a crash is found the next question is whether it is exploitable: does the
target's own executable text hold the gadgets an attacker would chain after
hijacking control flow? This module answers that *statically* from the target
ELF -- it never executes anything, it only reads ``.text``.

It is the static half of what ROPgadget / ropper / pwntools' ROP do: sweep
``.text`` for ``ret``-terminated instruction sequences (the classic "galileo"
unaligned walk-back), classify each (pop-reg, syscall, stack-pivot, ...), and
report which argument registers a ``pop <reg>; ret`` controls and whether a
syscall primitive exists. "Generation" here is a capability *skeleton* -- the
ordered gadget addresses a syscall chain would use -- not a filled-in payload;
the analyst supplies the values.

    .text bytes ──► scan() ──► [Gadget] ──► build_report() ──► GadgetReport
                     │                                           │
            galileo walk-back                          reg control + syscall
            from every ret byte                        => chain_skeleton

The decode layer is deliberately small. It reuses ``core.elf``'s x86-64
decoder for general instructions and recognises only the extra ROP-relevant
opcodes (push/pop/leave/nop/syscall/int) that the shared decoder -- built for
CFG/coverage, where instruction *length* and block boundaries are what matter
-- leaves as unknown bytes. Keeping that knowledge here, rather than widening
the shared decoder, protects its other callers (CFG, ptrace BB discovery,
map-size estimation), which depend on its current length output.
"""

from __future__ import annotations

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
    _decode_x86_64,
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

# SysV x86-64 syscall/call ABI: nr in rax, args in these registers, in order.
_SYSCALL_NR_REG = "rax"
_ARG_REGS = ("rdi", "rsi", "rdx", "r10", "r8", "r9")

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

# Bounds (Hard Rule 54): a huge .text (ffmpeg) must not blow time or memory.
_MAX_TEXT = 1 << 21  # scan at most 2 MiB of .text
_MAX_GADGETS = 4096
_SAMPLE_CAP = 32


# ── Decoded-instruction model ───────────────────────────────────────────


@dataclass(frozen=True)
class _Insn:
    """One decoded instruction inside a candidate gadget."""

    length: int
    text: str
    kind: str  # pop | push | ret | leave | nop | syscall | int | ctrl | priv | generic
    reg: str = ""  # r64 touched, for push/pop


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
        return _Insn(length, f"push {reg}", "push", reg)

    if 0x58 <= b <= 0x5F:  # pop r64
        reg = _r64(b - 0x58, rex)
        return _Insn(length, f"pop {reg}", "pop", reg)

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

    return None


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
    """A ret-terminated instruction sequence found in ``.text``."""

    vaddr: int  # virtual address of the gadget head
    text: str  # "insn ; insn ; ret"
    category: str  # pop | syscall | pivot | ret | generic
    pop_regs: tuple[str, ...] = ()  # regs a pop-gadget loads from the stack
    has_syscall: bool = False
    is_pivot: bool = False


def _is_pivot(insn: _Insn) -> bool:
    """True when ``insn`` moves the stack pointer under attacker control."""
    return insn.kind == "leave" or (insn.kind == "pop" and insn.reg == "rsp")


def _classify(insns: list[_Insn], tail: _Insn) -> tuple[str, tuple[str, ...], bool, bool]:
    """``(category, pop_regs, has_syscall, is_pivot)`` for a gadget body."""
    has_syscall = any(i.kind in ("syscall", "int") for i in insns)
    is_pivot = any(_is_pivot(i) for i in insns)

    if not insns:
        return "ret", (), has_syscall, is_pivot

    if tail.kind == "ret" and all(i.kind == "pop" for i in insns):
        return "pop", tuple(i.reg for i in insns), has_syscall, is_pivot

    if has_syscall:
        return "syscall", (), True, is_pivot

    if is_pivot:
        return "pivot", (), has_syscall, True

    return "generic", (), has_syscall, is_pivot


def _build_gadget(head_va: int, insns: list[_Insn], tail: _Insn) -> Gadget:
    category, pop_regs, has_syscall, is_pivot = _classify(insns, tail)
    text = " ; ".join(i.text for i in insns + [tail])
    return Gadget(head_va, text, category, pop_regs, has_syscall, is_pivot)


# ── Scan (galileo unaligned walk-back) ──────────────────────────────────


def _ret_tail(b: int) -> int:
    """Encoded length of a ret opcode byte (ret=1, ret imm16=3), else 0."""
    if b == _RET:
        return 1
    if b == _RET_IMM:
        return 3
    return 0


def _tail_insn(buf: bytes, t: int, tlen: int) -> _Insn:
    if tlen == 3:  # ret imm16
        imm = buf[t + 1] | (buf[t + 2] << 8)
        return _Insn(tlen, f"ret 0x{imm:x}", "ret")
    return _Insn(tlen, "ret", "ret")


def _try_head(buf, n, base, start, t, tail, out) -> None:
    """Emit the gadget whose head is ``start`` and whose ret ends at ``t``.

    Forward-decode from ``start``; accept only when the instruction boundaries
    land exactly on ``t`` (so the ret is reached, not straddled) and no
    rejected instruction sits inside.
    """
    insns: list[_Insn] = []
    off = start

    while off < t:
        ins = _decode_one(buf, off, n)
        if ins is None or ins.kind in _REJECT_KINDS or len(insns) >= _MAX_INSNS:
            return
        off += ins.length
        insns.append(ins)

    if off != t:  # straddled the ret boundary -> misaligned start
        return

    head_va = base + start
    out.setdefault(head_va, _build_gadget(head_va, insns, tail))


def scan(text: bytes, base: int, *, max_gadgets: int = _MAX_GADGETS) -> list[Gadget]:
    """All distinct ret-terminated gadgets in ``text`` (loaded at ``base``)."""
    buf = text[:_MAX_TEXT]
    n = len(buf)
    out: dict[int, Gadget] = {}

    i = 0
    while i < n:
        tlen = _ret_tail(buf[i])
        if not tlen or i + tlen > n:
            i += 1
            continue

        tail = _tail_insn(buf, i, tlen)
        lo = max(0, i - _MAX_BACK)
        for start in range(i, lo - 1, -1):  # bare ret (start==i) first
            _try_head(buf, n, base, start, i, tail, out)
            if len(out) >= max_gadgets:
                return list(out.values())

        i += 1

    return list(out.values())


# ── Report + chain synthesis ────────────────────────────────────────────


@dataclass
class GadgetReport:
    """Static exploitability summary of a target's gadget inventory."""

    total: int = 0
    truncated: bool = False  # .text exceeded _MAX_TEXT and was clipped
    categories: dict[str, int] = field(default_factory=dict)
    reg_control: dict[str, int] = field(default_factory=dict)  # reg -> head vaddr
    syscall_sites: list[int] = field(default_factory=list)
    pivot_sites: list[int] = field(default_factory=list)
    samples: list[tuple[int, str]] = field(default_factory=list)
    chain_feasible: bool = False
    chain_skeleton: list[tuple[str, int]] = field(default_factory=list)
    chain_note: str = ""

    def to_dict(self) -> dict:
        """Serialize for the crash sidecar (addresses as 0x-hex strings)."""
        return {
            "total": self.total,
            "truncated": self.truncated,
            "categories": dict(self.categories),
            "reg_control": {r: f"0x{v:x}" for r, v in self.reg_control.items()},
            "syscall_sites": [f"0x{v:x}" for v in self.syscall_sites],
            "pivot_sites": [f"0x{v:x}" for v in self.pivot_sites],
            "chain_feasible": self.chain_feasible,
            "chain_skeleton": [[role, f"0x{v:x}"] for role, v in self.chain_skeleton],
            "chain_note": self.chain_note,
            "samples": [[f"0x{v:x}", t] for v, t in self.samples],
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
    rank = {"pop": 0, "syscall": 1, "pivot": 2, "generic": 3, "ret": 4}
    ordered = sorted(gadgets, key=lambda g: (rank.get(g.category, 9), g.vaddr))
    return [(g.vaddr, g.text) for g in ordered[:_SAMPLE_CAP]]


def _synthesize(reg_control: dict[str, int], syscall_sites: list[int]):
    """A syscall-chain skeleton from the controllable registers.

    Feasible when a syscall primitive exists and both the syscall number
    (rax) and the first argument (rdi) are controllable. The skeleton lists
    rax, then every controllable arg register in ABI order, then the syscall
    site -- the ordered addresses an analyst fills with values.
    """
    if not syscall_sites or _SYSCALL_NR_REG not in reg_control or "rdi" not in reg_control:
        return False, [], "no syscall chain: need a syscall site plus rax and rdi control"

    skeleton: list[tuple[str, int]] = [(_SYSCALL_NR_REG, reg_control[_SYSCALL_NR_REG])]
    skeleton += [(r, reg_control[r]) for r in _ARG_REGS if r in reg_control]
    skeleton.append(("syscall", syscall_sites[0]))

    controlled = ", ".join(role for role, _ in skeleton[:-1])
    note = f"syscall chain feasible: control {controlled}; {len(syscall_sites)} syscall site(s)"
    return True, skeleton, note


def build_report(gadgets: list[Gadget], *, truncated: bool = False) -> GadgetReport:
    """Summarize a gadget list into an exploitability triage report."""
    categories = dict(Counter(g.category for g in gadgets))
    reg_control = _reg_control(gadgets)
    syscall_sites = sorted(g.vaddr for g in gadgets if g.has_syscall)
    pivot_sites = sorted(g.vaddr for g in gadgets if g.is_pivot)

    feasible, skeleton, note = _synthesize(reg_control, syscall_sites)

    return GadgetReport(
        total=len(gadgets),
        truncated=truncated,
        categories=categories,
        reg_control=reg_control,
        syscall_sites=syscall_sites,
        pivot_sites=pivot_sites,
        samples=_pick_samples(gadgets),
        chain_feasible=feasible,
        chain_skeleton=skeleton,
        chain_note=note,
    )


def scan_target(target: str) -> GadgetReport | None:
    """Scan a target ELF's ``.text`` for gadgets; None when it cannot be read.

    Never raises on a hostile or truncated binary -- a crash-triage aid must
    decline such a target, not take the fuzzer down with it.
    """
    from fuzzer_tool.core.elf import _find_text_section, _read_target_elf

    elf = _read_target_elf(target)
    if elf is None:
        return None

    section = _find_text_section(elf)
    if section is None:
        return None

    text, vaddr, declared = section
    truncated = len(text) > _MAX_TEXT or declared > _MAX_TEXT
    return build_report(scan(text, vaddr), truncated=truncated)
