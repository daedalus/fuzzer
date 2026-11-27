"""ROP gadget detection / chain-feasibility synthesis (core.gadgets).

Fully deterministic: every case is hand-assembled x86-64 opcode bytes and the
expected gadgets are derived from the encoding, not echoed from the code under
test. No RNG, so no scripted-rng seam is needed here.
"""

from __future__ import annotations

import os
import struct

from fuzzer_tool.core import gadgets
from tests.test_elf import SHF_EXECINSTR, _build_elf

# Hand-assembled single-byte / short opcodes used across the cases.
POP = {"rax": b"\x58", "rdx": b"\x5a", "rsi": b"\x5e", "rdi": b"\x5f"}
POP_R15 = b"\x41\x5f"  # REX.B + pop rdi-slot -> pop r15
PUSH_RSP = b"\x54"
RET = b"\xc3"
RET_IMM = b"\xc2\x08\x00"
SYSCALL = b"\x0f\x05"
LEAVE = b"\xc9"
NOP = b"\x90"

BASE = 0x400000


def _by_va(report_gadgets):
    return {g.vaddr: g for g in report_gadgets}


def test_pop_ret_controls_reg():
    text = POP["rdi"] + RET  # 0x5f 0xc3
    gs = gadgets.scan(text, BASE)
    rep = gadgets.build_report(gs)

    # One pop;ret head at BASE, one bare ret head at BASE+1.
    assert rep.total == 2
    assert rep.reg_control == {"rdi": BASE}
    head = _by_va(gs)[BASE]
    assert head.category == "pop"
    assert head.text == "pop rdi ; ret"


def test_rex_pop_r15():
    text = POP_R15 + RET  # 0x41 0x5f 0xc3
    gs = gadgets.scan(text, BASE)
    rep = gadgets.build_report(gs)
    assert rep.reg_control.get("r15") == BASE
    assert _by_va(gs)[BASE].text == "pop r15 ; ret"


def test_two_pop_gadget_controls_both():
    text = POP["rdi"] + POP["rsi"] + RET  # 0x5f 0x5e 0xc3
    gs = gadgets.scan(text, BASE)
    rep = gadgets.build_report(gs)
    # The 2-pop head controls rdi; the inner 1-pop head (BASE+1) controls rsi.
    assert rep.reg_control["rdi"] == BASE
    assert rep.reg_control["rsi"] == BASE + 1
    assert _by_va(gs)[BASE].text == "pop rdi ; pop rsi ; ret"


def test_syscall_gadget_detected():
    text = SYSCALL + RET  # 0x0f 0x05 0xc3
    gs = gadgets.scan(text, BASE)
    rep = gadgets.build_report(gs)
    assert rep.syscall_sites == [BASE]
    assert _by_va(gs)[BASE].category == "syscall"


def test_bare_ret_is_a_gadget():
    gs = gadgets.scan(RET, BASE)
    assert len(gs) == 1
    assert gs[0].category == "ret"
    assert gs[0].text == "ret"


def test_ret_imm_tail_length():
    # pop rdi ; ret 0x8 -- the imm16 ret is 3 bytes, so the head must land on
    # the pop (misreading it as 1 byte would shift the boundary and drop it).
    text = POP["rdi"] + RET_IMM
    gs = gadgets.scan(text, BASE)
    head = _by_va(gs)[BASE]
    assert head.category == "pop"
    assert head.text == "pop rdi ; ret 0x8"


def test_chain_feasible_syscall():
    text = POP["rax"] + RET + POP["rdi"] + RET + POP["rsi"] + RET + POP["rdx"] + RET + SYSCALL + RET
    gs = gadgets.scan(text, BASE)
    rep = gadgets.build_report(gs)

    assert rep.chain_feasible is True
    roles = [role for role, _va in rep.chain_skeleton]
    # rax (syscall nr) first, then the controllable arg regs in ABI order,
    # then the syscall site.
    assert roles == ["rax", "rdi", "rsi", "rdx", "syscall"]
    # Every skeleton address is a real gadget head.
    heads = {g.vaddr for g in gs}
    assert all(va in heads for _role, va in rep.chain_skeleton)


def test_stack_pivot_detected():
    # push rsp then a mov-back is a classic pivot shape; here the simpler
    # `pop rsp ; ret` pivot (0x5c 0xc3) must be flagged.
    text = b"\x5c" + RET  # pop rsp ; ret
    gs = gadgets.scan(text, BASE)
    rep = gadgets.build_report(gs)
    assert rep.pivot_sites == [BASE]


# ── Falsification: no ret bytes => no gadgets, nothing feasible ──────────


def test_no_ret_no_gadgets():
    text = NOP * 32  # never contains 0xc3/0xc2
    gs = gadgets.scan(text, BASE)
    rep = gadgets.build_report(gs)
    assert gs == []
    assert rep.total == 0
    assert rep.reg_control == {}
    assert rep.chain_feasible is False
    assert rep.chain_skeleton == []


def test_rets_without_control_not_feasible():
    # All ret, no pop/syscall: gadgets exist but a chain cannot be built.
    text = RET * 16
    rep = gadgets.build_report(gadgets.scan(text, BASE))
    assert rep.total >= 1
    assert rep.reg_control == {}
    assert rep.chain_feasible is False


# ── Adversarial: hostile / truncated input must not raise ────────────────


def test_scan_target_missing_file_returns_none():
    assert gadgets.scan_target("/nonexistent/path/to/target") is None


def test_scan_target_non_elf(tmp_path):
    p = tmp_path / "notelf.bin"
    p.write_bytes(b"not an elf file at all" * 8)
    assert gadgets.scan_target(str(p)) is None


def test_random_bytes_do_not_raise_and_are_bounded():
    blob = os.urandom(8192)
    gs = gadgets.scan(blob, BASE, max_gadgets=64)
    assert len(gs) <= 64
    # A report over garbage must still serialize without raising.
    rep = gadgets.build_report(gs)
    assert isinstance(rep.to_dict(), dict)


def test_truncated_rex_at_end():
    # A lone REX prefix (no opcode) and a lone ret-imm opcode (no imm) must be
    # declined, not read past the buffer.
    assert gadgets.scan(b"\x41", BASE) == []  # dangling REX, no ret
    assert gadgets.scan(b"\xc2", BASE) == []  # ret imm16 needs 2 more bytes


def test_report_to_dict_roundtrips_addresses():
    text = POP["rdi"] + RET
    rep = gadgets.build_report(gadgets.scan(text, BASE))
    d = rep.to_dict()
    assert d["reg_control"]["rdi"] == f"0x{BASE:x}"
    assert d["total"] == 2


# ── x86-64 decoder additions: pivot, stack-adj, mov-imm, retf ───────────

XCHG_RAX_RSP = b"\x48\x94"  # xchg rax, rsp
XCHG_EAX_ESP = b"\x94"  # 32-bit form: zeroes rsp's top half, not a 64-bit pivot
ADD_RSP_0X20 = b"\x48\x83\xc4\x20"  # add rsp, 0x20
SUB_RSP_0X100 = b"\x48\x81\xec\x00\x01\x00\x00"  # sub rsp, 0x100
ADD_R12_0X20 = b"\x49\x83\xc4\x20"  # REX.WB -> add r12, imm8 (NOT rsp)
MOV_RAX_0X3B = b"\x48\xb8\x3b\x00\x00\x00\x00\x00\x00\x00"  # mov rax, 0x3b
MOV_R15_0X1337 = b"\x49\xbf\x37\x13\x00\x00\x00\x00\x00\x00"  # mov r15, 0x1337
RETF = b"\xcb"
RETFQ = b"\x48\xcb"
CALL_RAX = b"\xff\xd0"
JMP_R12 = b"\x41\xff\xe4"
CALL_MEM_RAX = b"\xff\x10"  # call [rax]: memory-indirect, not a register dispatch


def test_xchg_rax_rsp_is_pivot():
    gs = gadgets.scan(XCHG_RAX_RSP + RET, BASE)
    rep = gadgets.build_report(gs)
    assert rep.pivot_sites == [BASE]
    assert _by_va(gs)[BASE].category == "pivot"
    assert _by_va(gs)[BASE].text == "xchg rax, rsp ; ret"


def test_xchg_eax_esp_is_not_a_pivot():
    # FALSIFICATION: without REX.W the swap is 32-bit and truncates rsp.
    rep = gadgets.build_report(gadgets.scan(XCHG_EAX_ESP + RET, BASE))
    assert rep.pivot_sites == []


def test_add_rsp_imm8_is_adj():
    gs = gadgets.scan(ADD_RSP_0X20 + RET, BASE)
    rep = gadgets.build_report(gs)
    assert rep.categories.get("adj") == 1
    assert _by_va(gs)[BASE].text == "add rsp, 0x20 ; ret"


def test_sub_rsp_imm32_is_adj():
    gs = gadgets.scan(SUB_RSP_0X100 + RET, BASE)
    assert _by_va(gs)[BASE].text == "sub rsp, 0x100 ; ret"
    assert gadgets.build_report(gs).categories.get("adj") == 1


def test_add_r12_is_not_stack_adj():
    # FALSIFICATION: REX.B remaps rm=100 from rsp to r12.
    rep = gadgets.build_report(gadgets.scan(ADD_R12_0X20 + RET, BASE))
    assert "adj" not in rep.categories


def test_mov_imm64_is_movimm():
    gs = gadgets.scan(MOV_RAX_0X3B + RET, BASE)
    assert _by_va(gs)[BASE].category == "movimm"
    assert _by_va(gs)[BASE].text == "mov rax, 0x3b ; ret"


def test_mov_r15_imm64():
    gs = gadgets.scan(MOV_R15_0X1337 + RET, BASE)
    assert _by_va(gs)[BASE].text == "mov r15, 0x1337 ; ret"
    assert _by_va(gs)[BASE].category == "movimm"


def test_mov_imm_does_not_control_reg():
    # FALSIFICATION: the immediate lives in the gadget's own bytes, not on
    # the chain stack -- reg_control stays stack-loaded (pop) only.
    rep = gadgets.build_report(gadgets.scan(MOV_RAX_0X3B + RET, BASE))
    assert rep.reg_control == {}


def test_pop_retf_controls_reg():
    gs = gadgets.scan(POP["rdi"] + RETF, BASE)
    rep = gadgets.build_report(gs)
    assert rep.reg_control == {"rdi": BASE}
    assert _by_va(gs)[BASE].category == "pop"
    assert _by_va(gs)[BASE].text == "pop rdi ; retf"


def test_retfq_terminator():
    gs = gadgets.scan(POP["rdi"] + RETFQ, BASE)
    rep = gadgets.build_report(gs)
    assert rep.reg_control == {"rdi": BASE}
    assert _by_va(gs)[BASE].text == "pop rdi ; retfq"
    assert _by_va(gs)[BASE + 1].text == "retfq"


# ── JOP/COP: call/jmp-register-terminated gadgets ───────────────────────


def test_call_rax_jop_gadget():
    gs = gadgets.scan(POP["rax"] + CALL_RAX, BASE)
    rep = gadgets.build_report(gs)
    assert _by_va(gs)[BASE].category == "jop"
    assert _by_va(gs)[BASE].text == "pop rax ; call rax"
    # Two JOP gadgets: the bare dispatch and the pop that feeds it.
    assert rep.categories.get("jop") == 2


def test_rex_jmp_r12_jop():
    gs = gadgets.scan(JMP_R12, BASE)
    assert _by_va(gs)[BASE].category == "jop"
    assert _by_va(gs)[BASE].text == "jmp r12"


def test_call_mem_is_not_jop():
    # FALSIFICATION: FF /2 with mod != 11 calls through memory, not through
    # a register -- no JOP terminator, and no pop head through it either.
    rep = gadgets.build_report(gadgets.scan(POP["rax"] + CALL_MEM_RAX, BASE))
    assert "jop" not in rep.categories
    assert rep.reg_control == {}


def test_jop_inside_ret_body_rejected():
    # pop rdi; call rax; ret: the call is control flow inside a would-be
    # ret-terminated body, so the pop head must be declined. The call itself
    # is a JOP gadget (bare dispatch + the pop feeding it); the ret survives
    # bare.
    text = POP["rdi"] + CALL_RAX + RET
    rep = gadgets.build_report(gadgets.scan(text, BASE))
    assert rep.reg_control == {}
    assert rep.categories.get("jop") == 2
    assert rep.categories.get("ret") == 1


# ── aarch64 (second architecture) ───────────────────────────────────────
#
# Instruction words verified against capstone 5.0.7 (aarch64, LE); see
# scripts/verify_encodings.py in the task workspace for the oracle run.

A64_BASE = 0x400000

A64_RET = 0xD65F03C0  # ret
A64_NOP = 0xD503201F  # nop
A64_SVC0 = 0xD4000001  # svc #0
A64_LDP_X0_X1_POST = 0xA8C107E0  # ldp x0, x1, [sp], #0x10
A64_LDP_X29_X30_POST = 0xA8C17BFD  # ldp x29, x30, [sp], #0x10
A64_LDP_X2_X3_SIGNED = 0xA9410FE2  # ldp x2, x3, [sp, #0x10]
A64_LDP_X8_X0_SIGNED = 0xA94003E8  # ldp x8, x0, [sp]
A64_LDR_X9 = 0xF94007E9  # ldr x9, [sp, #0x8]
A64_MOV_SP_X5 = 0x910000BF  # mov sp, x5
A64_ADD_SP = 0x910043FF  # add sp, sp, #0x10
A64_BR_X4 = 0xD61F0080  # br x4
A64_BLR_X4 = 0xD63F0080  # blr x4
A64_STP = 0xA9BF7BFD  # stp x29, x30, [sp, #-0x10]!  (a store: rejected)
A64_CBZ = 0xB4000040  # cbz x0, .+8                     (a branch: rejected)


def _a64(*words):
    """Pack aarch64 instruction words (little-endian) into a code blob."""
    return b"".join(struct.pack("<I", w) for w in words)


def _a64_report(*words):
    gs = gadgets.scan(_a64(*words), A64_BASE, arch="aarch64")
    return gs, gadgets.build_report(gs, arch="aarch64")


def test_a64_ldp_post_index_controls_two_regs():
    gs, rep = _a64_report(A64_LDP_X0_X1_POST, A64_RET)
    assert rep.reg_control == {"x0": A64_BASE, "x1": A64_BASE}
    assert _by_va(gs)[A64_BASE].category == "pop"
    assert _by_va(gs)[A64_BASE].text == "ldp x0, x1, [sp], #0x10 ; ret"


def test_a64_epilogue_controls_ret_pc():
    # The canonical epilogue: x30 IS the return address, so controlling it
    # is control-flow control.
    _gs, rep = _a64_report(A64_LDP_X29_X30_POST, A64_RET)
    assert rep.reg_control == {"x29": A64_BASE, "x30": A64_BASE}


def test_a64_ldp_signed_offset():
    _gs, rep = _a64_report(A64_LDP_X2_X3_SIGNED, A64_RET)
    assert rep.reg_control == {"x2": A64_BASE, "x3": A64_BASE}


def test_a64_ldr_from_sp_controls_reg():
    gs, rep = _a64_report(A64_LDR_X9, A64_RET)
    assert rep.reg_control == {"x9": A64_BASE}
    assert _by_va(gs)[A64_BASE].text == "ldr x9, [sp, #0x8] ; ret"


def test_a64_bare_svc_is_syscall_site():
    gs, rep = _a64_report(A64_SVC0, A64_NOP)
    assert rep.syscall_sites == [A64_BASE]
    assert _by_va(gs)[A64_BASE].category == "syscall"


def test_a64_svc_inside_ret_gadget():
    gs, rep = _a64_report(A64_SVC0, A64_RET)
    assert rep.syscall_sites == [A64_BASE]
    assert _by_va(gs)[A64_BASE].category == "syscall"
    assert _by_va(gs)[A64_BASE].text == "svc #0x0 ; ret"


def test_a64_chain_feasible_svc():
    # ldp x8, x0, [sp]; ret  +  svc #0; ret: syscall nr (x8) + arg0 (x0).
    _gs, rep = _a64_report(A64_LDP_X8_X0_SIGNED, A64_RET, A64_SVC0, A64_RET)
    assert rep.chain_feasible is True
    roles = [role for role, _va in rep.chain_skeleton]
    assert roles == ["x8", "x0", "syscall"]


def test_a64_chain_infeasible_without_nr_reg():
    # FALSIFICATION: arg control (x0, x1) without the syscall-number reg.
    _gs, rep = _a64_report(A64_LDP_X0_X1_POST, A64_RET, A64_SVC0, A64_RET)
    assert rep.chain_feasible is False
    assert rep.chain_skeleton == []


def test_a64_pivot_mov_sp():
    gs, rep = _a64_report(A64_MOV_SP_X5, A64_RET)
    assert rep.pivot_sites == [A64_BASE]
    assert _by_va(gs)[A64_BASE].text == "mov sp, x5 ; ret"


def test_a64_add_sp_is_adj_not_pivot():
    # FALSIFICATION: a fixed-size stack bump moves rsp by a constant, it
    # does not load it from a register.
    _gs, rep = _a64_report(A64_ADD_SP, A64_RET)
    assert rep.pivot_sites == []
    assert rep.categories.get("adj") == 1


def test_a64_undecodable_body_rejects_head():
    # A store (stp) is outside the decoder: the head spanning it is
    # declined, the inner ldp;ret still stands.
    _gs, rep = _a64_report(A64_STP, A64_LDP_X0_X1_POST, A64_RET)
    assert rep.reg_control == {"x0": A64_BASE + 4, "x1": A64_BASE + 4}


def test_a64_branch_in_body_rejects_head():
    # FALSIFICATION: cbz is a conditional branch; no gadget may span it.
    gs, _rep = _a64_report(A64_CBZ, A64_RET)
    assert len(gs) == 1  # the bare ret only
    assert gs[0].text == "ret"


def test_a64_br_is_jop_terminator():
    gs, _rep = _a64_report(A64_LDP_X0_X1_POST, A64_BR_X4)
    assert _by_va(gs)[A64_BASE].category == "jop"
    assert _by_va(gs)[A64_BASE].text == "ldp x0, x1, [sp], #0x10 ; br x4"


def test_a64_blr_is_jop_terminator():
    gs, _rep = _a64_report(A64_BLR_X4)
    assert _by_va(gs)[A64_BASE].category == "jop"
    assert _by_va(gs)[A64_BASE].text == "blr x4"


def test_a64_unaligned_bytes_ignored():
    # A ret encoding straddling an instruction boundary must not be read as
    # a terminator: aarch64 instruction fetch is 4-byte aligned.
    assert gadgets.scan(b"\x00" + struct.pack("<I", A64_RET), A64_BASE, arch="aarch64") == []


def test_a64_libc_chain_uses_x0():
    gs = gadgets.scan(_a64(A64_LDP_X0_X1_POST, A64_RET), A64_BASE, arch="aarch64")
    rep = gadgets.build_report(gs, arch="aarch64", libc_targets={"system": 0x401020})
    assert rep.chain_libc_feasible is True
    assert rep.chain_libc == [("x0", A64_BASE), ("system", 0x401020)]


# ── ret2libc via PLT / defined symbols ──────────────────────────────────


def test_libc_chain_feasible_system():
    gs = gadgets.scan(POP["rdi"] + RET, BASE)
    rep = gadgets.build_report(gs, libc_targets={"system": 0x401020})
    assert rep.chain_libc_feasible is True
    assert rep.chain_libc == [("rdi", BASE), ("system", 0x401020)]
    assert "ret2libc" in rep.chain_note


def test_libc_chain_needs_arg0_control():
    # FALSIFICATION: system takes its command in rdi; rax control alone
    # cannot reach it.
    rep = gadgets.build_report(
        gadgets.scan(POP["rax"] + RET, BASE), libc_targets={"system": 0x401020}
    )
    assert rep.chain_libc_feasible is False
    assert rep.chain_libc == []


def test_libc_chain_needs_a_one_arg_sink():
    # FALSIFICATION: mprotect needs three args (rdi, rsi, rdx); with rdi
    # control only, no sink is claimable and none is claimed.
    rep = gadgets.build_report(
        gadgets.scan(POP["rdi"] + RET, BASE), libc_targets={"mprotect": 0x401030}
    )
    assert rep.chain_libc_feasible is False


# ── scan_target: exec-section sweep, arch detect, budget ────────────────


def _target(tmp_path, elf, name="t"):
    p = tmp_path / name
    p.write_bytes(elf)
    return str(p)


def test_scan_target_sweeps_all_exec_sections(tmp_path):
    # .text holds a bare ret; .plt holds pop rdi; ret. The sweep must
    # attribute gadgets per section and merge reg control across them.
    elf = _build_elf(
        [
            (b".text", 1, SHF_EXECINSTR, 0x401000, RET),
            (b".plt", 1, SHF_EXECINSTR, 0x402000, POP["rdi"] + RET),
        ]
    )
    rep = gadgets.scan_target(_target(tmp_path, elf))
    assert rep is not None
    assert rep.arch == "x86_64"
    assert rep.sections == {".text": 1, ".plt": 2}
    assert rep.reg_control == {"rdi": 0x402000}


def test_scan_target_renamed_text_section(tmp_path):
    # A stripped or obfuscated target's code section need not be named
    # .text; the SHF_EXECINSTR sweep still finds it.
    elf = _build_elf([(b".lol", 1, SHF_EXECINSTR, 0x401000, POP["rax"] + RET)])
    rep = gadgets.scan_target(_target(tmp_path, elf))
    assert rep is not None
    assert rep.sections == {".lol": 2}
    assert rep.reg_control == {"rax": 0x401000}


def test_scan_target_non_exec_text_declined(tmp_path):
    # FALSIFICATION: a .text-named section without SHF_EXECINSTR is data.
    elf = _build_elf([(b".text", 1, 0, 0x401000, POP["rdi"] + RET)])
    assert gadgets.scan_target(_target(tmp_path, elf)) is None


def test_scan_target_budget_marks_truncated(tmp_path, monkeypatch):
    # Hard Rule 54: the sweep is bounded. Tiny caps clip both sections and
    # the report must say so instead of pretending completeness.
    monkeypatch.setattr(gadgets, "_MAX_TEXT", 8)
    monkeypatch.setattr(gadgets, "_MAX_TOTAL_TEXT", 8)
    elf = _build_elf(
        [
            (b".text", 1, SHF_EXECINSTR, 0x401000, bytes(16)),
            (b".plt", 1, SHF_EXECINSTR, 0x402000, bytes(16)),
        ]
    )
    rep = gadgets.scan_target(_target(tmp_path, elf))
    assert rep.truncated is True
    assert list(rep.sections) == [".text"]  # budget exhausted before .plt


def test_scan_target_section_cap_marks_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(gadgets, "_MAX_SECTIONS", 1)
    elf = _build_elf(
        [
            (b".text", 1, SHF_EXECINSTR, 0x401000, RET),
            (b".plt", 1, SHF_EXECINSTR, 0x402000, RET),
        ]
    )
    rep = gadgets.scan_target(_target(tmp_path, elf))
    assert rep.truncated is True
    assert list(rep.sections) == [".text"]


def test_scan_target_detects_aarch64(tmp_path):
    elf = _build_elf(
        [(b".text", 1, SHF_EXECINSTR, 0x400000, _a64(A64_LDP_X0_X1_POST, A64_RET))],
        machine=183,
    )
    rep = gadgets.scan_target(_target(tmp_path, elf))
    assert rep is not None
    assert rep.arch == "aarch64"
    assert rep.reg_control == {"x0": 0x400000, "x1": 0x400000}


def test_scan_target_unknown_machine_declined(tmp_path):
    elf = _build_elf([(b".text", 1, SHF_EXECINSTR, 0x401000, RET)], machine=40)
    assert gadgets.scan_target(_target(tmp_path, elf)) is None


def test_scan_target_ret2libc_via_plt(tmp_path):
    # .text: pop rdi; ret. .plt slot 1 jumps through system's GOT slot via
    # .rela.plt. The ret2libc skeleton must chain rdi control to the stub.
    stub = 0x401010
    entry = b"\xff\x25" + struct.pack("<i", 0x601000 - (stub + 6)) + b"\x90" * 10
    # dynsym: mandatory NULL entry at index 0, the import at index 1.
    sym = bytearray(24)
    struct.pack_into("<I", sym, 0, 1)
    sym[4] = 0x12
    rela = bytearray(24)
    struct.pack_into("<Q", rela, 0, 0x601000)
    struct.pack_into("<Q", rela, 8, (1 << 32) | 7)  # sym 1, JUMP_SLOT
    elf = _build_elf(
        [
            (b".dynsym", 11, 0, 0x600000, b"\x00" * 24 + bytes(sym), 3, 24),
            (b".dynstr", 3, 0, 0x600100, b"\x00system\x00"),
            (b".rela.plt", 4, 0, 0x600200, bytes(rela), 2, 24),
            (b".text", 1, SHF_EXECINSTR, 0x401000, POP["rdi"] + RET),
            (b".plt", 1, SHF_EXECINSTR, 0x401000, b"\x90" * 0x10 + entry),
        ]
    )
    rep = gadgets.scan_target(_target(tmp_path, elf))
    assert rep is not None
    assert rep.libc_targets == {"system": stub}
    assert rep.chain_libc_feasible is True
    assert rep.chain_libc == [("rdi", 0x401000), ("system", stub)]
