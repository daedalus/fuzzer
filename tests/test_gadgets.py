"""ROP gadget detection / chain-feasibility synthesis (core.gadgets).

Fully deterministic: every case is hand-assembled x86-64 opcode bytes and the
expected gadgets are derived from the encoding, not echoed from the code under
test. No RNG, so no scripted-rng seam is needed here.
"""

from __future__ import annotations

import os

from fuzzer_tool.core import gadgets

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
