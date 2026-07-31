"""Unit tests for core/cfg.py — intra-procedural CFG builder.

All inputs are hand-crafted x86-64 byte sequences and all expected
structures are hand-computed literals (independent of the code under
test), per the repo rule that equivalence assertions must not validate
code against itself.
"""

from fuzzer_tool.core.cfg import build_function_cfg

BASE = 0x1000


def _starts(cfg):
    return sorted(cfg.blocks)


def _succ(cfg, start):
    return sorted(cfg.blocks[start].successors)


class TestBasicDiamond:
    """nop; je L1; nop; jmp L1; L1: nop; ret

    Blocks (hand-derived): [0x1000,0x1003) nop;je, [0x1003,0x1006)
    nop;jmp, [0x1006,0x1008) nop;ret. The je is the terminator of its
    block, not a block start.
    """

    CODE = bytes.fromhex("90 74 03 90 EB 00 90 C3")

    def test_block_starts(self):
        cfg = build_function_cfg("f", self.CODE, BASE)
        assert _starts(cfg) == [0x1000, 0x1003, 0x1006]

    def test_entry_exit(self):
        cfg = build_function_cfg("f", self.CODE, BASE)
        assert cfg.blocks[0x1000].is_entry
        # last block ends in ret
        assert cfg.blocks[0x1006].is_exit
        assert not cfg.blocks[0x1000].is_exit

    def test_successors(self):
        cfg = build_function_cfg("f", self.CODE, BASE)
        # je: taken 0x1006 + fall-through 0x1003
        assert _succ(cfg, 0x1000) == [0x1003, 0x1006]
        assert _succ(cfg, 0x1003) == [0x1006]
        assert _succ(cfg, 0x1006) == []

    def test_block_ranges(self):
        cfg = build_function_cfg("f", self.CODE, BASE)
        assert (cfg.blocks[0x1000].start, cfg.blocks[0x1000].end) == (0x1000, 0x1003)
        assert (cfg.blocks[0x1003].start, cfg.blocks[0x1003].end) == (0x1003, 0x1006)
        assert (cfg.blocks[0x1006].start, cfg.blocks[0x1006].end) == (0x1006, 0x1008)


class TestCallSites:
    """call helper (outside range); ret — callee resolved via callback."""

    CODE = bytes.fromhex("E8 05 00 00 00 C3")

    def test_direct_call_callee(self):
        resolver = {0x100A: "helper"}
        cfg = build_function_cfg(
            "caller", self.CODE, BASE, resolve_callee=lambda a: resolver.get(a)
        )
        assert _starts(cfg) == [0x1000, 0x1005]
        assert cfg.blocks[0x1000].callees == {"helper"}
        # call contributes only the fall-through edge
        assert _succ(cfg, 0x1000) == [0x1005]
        assert cfg.blocks[0x1005].is_exit

    def test_call_without_resolver_has_no_callee(self):
        cfg = build_function_cfg("caller", self.CODE, BASE)
        assert cfg.blocks[0x1000].callees == set()
        assert not cfg.blocks[0x1000].indirect_call

    def test_unresolved_direct_call_flagged(self):
        cfg = build_function_cfg("caller", self.CODE, BASE, resolve_callee=lambda a: None)
        assert cfg.blocks[0x1000].indirect_call

    def test_indirect_call_flagged(self):
        # call rax
        cfg = build_function_cfg("caller", bytes.fromhex("FF D0 C3"), BASE)
        assert cfg.blocks[0x1000].indirect_call
        assert cfg.blocks[0x1000].callees == set()
        assert _succ(cfg, 0x1000) == [0x1002]


class TestIndirectJump:
    """jmp rax; nop; ret — indirect jmp has no static successor."""

    CODE = bytes.fromhex("FF E0 90 C3")

    def test_indirect_jump(self):
        cfg = build_function_cfg("f", self.CODE, BASE)
        assert _starts(cfg) == [0x1000, 0x1002]
        assert cfg.blocks[0x1000].indirect_jump
        assert _succ(cfg, 0x1000) == []
        assert cfg.blocks[0x1002].is_exit


class TestDecoderGaps:
    """ret imm16 (C2) and loop (E0) — extra operand bytes consumed."""

    def test_ret_imm16_consumes_operand_bytes(self):
        # ret 8; ret
        cfg = build_function_cfg("f", bytes.fromhex("C2 08 00 C3"), BASE)
        assert _starts(cfg) == [0x1000, 0x1003]
        assert cfg.blocks[0x1000].is_exit
        assert (cfg.blocks[0x1000].start, cfg.blocks[0x1000].end) == (0x1000, 0x1003)

    def test_loop_self_edge(self):
        # loopnz -2 → self-loop; ret
        cfg = build_function_cfg("f", bytes.fromhex("E0 FE C3"), BASE)
        assert _starts(cfg) == [0x1000, 0x1002]
        assert _succ(cfg, 0x1000) == [0x1000, 0x1002]
        assert cfg.blocks[0x1002].is_exit


class TestJccRel32:
    """je rel32 (0F 84): taken 0x1008, fall-through 0x1006."""

    CODE = bytes.fromhex("0F 84 02 00 00 00 90 90 C3")

    def test_rel32_taken_and_fall(self):
        cfg = build_function_cfg("f", self.CODE, BASE)
        assert _starts(cfg) == [0x1000, 0x1006, 0x1008]
        assert _succ(cfg, 0x1000) == [0x1006, 0x1008]
        assert _succ(cfg, 0x1006) == [0x1008]
        assert cfg.blocks[0x1008].is_exit


class TestBlockContaining:
    CODE = bytes.fromhex("90 74 03 90 EB 00 90 C3")

    def test_containing_lookup(self):
        cfg = build_function_cfg("f", self.CODE, BASE)
        assert cfg.block_containing(0x1000).start == 0x1000
        assert cfg.block_containing(0x1002).start == 0x1000
        assert cfg.block_containing(0x1005).start == 0x1003
        assert cfg.block_containing(0x1007).start == 0x1006
        assert cfg.block_containing(BASE - 1) is None
        assert cfg.block_containing(0x1008) is None


class TestDegenerate:
    def test_empty_code(self):
        cfg = build_function_cfg("f", b"", BASE)
        assert cfg.blocks == {}

    def test_single_ret(self):
        cfg = build_function_cfg("f", b"\xc3", BASE)
        assert _starts(cfg) == [BASE]
        assert cfg.blocks[BASE].is_entry and cfg.blocks[BASE].is_exit
        assert _succ(cfg, BASE) == []

    def test_out_of_range_targets_are_dropped(self):
        # jmp far outside the function range (tail call) → no successor
        cfg = build_function_cfg("f", bytes.fromhex("E9 00 00 00 40 C3"), BASE)
        assert _starts(cfg) == [0x1000, 0x1005]
        assert _succ(cfg, 0x1000) == []
