"""Intra-procedural control-flow graph builder for AFLGo-style distance.

Builds basic-block CFGs from the pure-Python x86-64 decoder in
``fuzzer_tool.core.elf``. Each function's decoded instructions are split
into basic blocks at control-flow boundaries (conditional/unconditional
jumps, calls, returns) and branch targets. For every block we record:

  - successors: intra-procedural block-start addresses (fall-through and
    taken edges). Calls contribute only their fall-through edge; the
    callee is recorded separately via ``resolve_callee``.
  - callees: direct-call targets resolved to function names (the
    BB-callsite map, AFLGo's ``BBcalls.txt`` analog).
  - indirect_call / indirect_jump: set when a call/jump target cannot be
    resolved statically (indirect ``call r/m`` / ``jmp r/m``). These have
    no static successor — the AFLGo runtime-tracing step (CG-gap
    patching) exists precisely to recover those edges.

Decoder gaps compensated here (the decoder classifies them as
``_INS_OTHER``): ``C2/CA`` (ret imm16) and ``E0-E3`` (loop/jcxz, treated
as conditional jumps with rel8 targets). The extra operand bytes are
consumed so block ranges stay exact; the decoder's own (mis)interpreted
yields inside those bytes are skipped.
"""

import bisect
import logging
from dataclasses import dataclass, field

from fuzzer_tool.core.elf import _INS_CALL, _INS_JCC, _INS_JMP, _INS_OTHER, _INS_RET, _decode_x86_64

log = logging.getLogger(__name__)

# Cap on decoded instructions per function — bounds pathological decode
# loops and keeps the builder cheap on huge binaries.
_MAX_INSNS_PER_FUNC = 1_000_000


@dataclass
class BasicBlock:
    """A basic block within a function.

    Attributes:
        start: Address of the first instruction.
        end: Address just past the last instruction.
        successors: Start addresses of intra-procedural successor blocks.
        callees: Names of directly-called functions from this block.
        indirect_call: True if the block contains a call whose target
            cannot be resolved statically.
        indirect_jump: True if the block ends in an indirect jump (no
            static successor).
        is_entry: True for the function's entry block.
        is_exit: True if the block ends in a return.
    """

    start: int
    end: int
    successors: list[int] = field(default_factory=list)
    callees: set[str] = field(default_factory=set)
    indirect_call: bool = False
    indirect_jump: bool = False
    is_entry: bool = False
    is_exit: bool = False

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class FunctionCFG:
    """Control-flow graph of a single function."""

    name: str
    start: int
    end: int
    blocks: dict[int, BasicBlock] = field(default_factory=dict)

    def block_containing(self, addr: int) -> BasicBlock | None:
        """Return the block containing *addr*, or None."""
        starts = sorted(self.blocks)
        idx = bisect.bisect_right(starts, addr) - 1
        if idx < 0:
            return None
        blk = self.blocks[starts[idx]]
        return blk if blk.start <= addr < blk.end else None


# ── Instruction classification ────────────────────────────────────────

_FALL = 0  # ordinary instruction: single fall-through successor
_JCC = 1  # conditional jump: taken + fall-through
_JMP = 2  # unconditional jump: taken only
_CALL = 3  # call: fall-through only (callee recorded separately)
_RET = 4  # return: no successors


def _classify(insn, fixups: dict[int, tuple]):
    """Classify a decoded instruction into (kind, target, extra_len).

    *kind* is one of the _FALL/_JCC/_JMP/_CALL/_RET constants; *target*
    is the direct branch/call target address or None; *extra_len* is the
    corrected instruction length when a decoder gap (ret imm16, loop
    opcodes) needs the following bytes consumed.

    Indirect call/jump forms decode with an empty ``op_str`` and no
    target — they are reported as _CALL/_JMP with target=None.
    """
    if insn.insn_id == _INS_CALL:
        return _CALL, _target_from_opstr(insn), 0
    if insn.insn_id == _INS_JMP:
        return _JMP, _target_from_opstr(insn), 0
    if insn.insn_id == _INS_JCC:
        return _JCC, _target_from_opstr(insn), 0
    if insn.insn_id == _INS_RET:
        return _RET, None, 0

    # Decoder gaps: only compensate single-byte unknowns.
    if insn.insn_id == _INS_OTHER and insn.length == 1 and insn.address in fixups:
        return fixups[insn.address]

    return _FALL, None, 0


def _target_from_opstr(insn):
    """Parse a direct branch target from the decoder's ``op_str``."""
    if not insn.op_str:
        return None
    try:
        return int(insn.op_str, 16)
    except ValueError:
        return None


def build_function_cfg(
    name: str,
    code: bytes,
    base_addr: int,
    resolve_callee=None,
) -> FunctionCFG:
    """Build the CFG of one function.

    Args:
        name: Function name (used as the CFG's identity).
        code: Raw instruction bytes of the function (from the ELF text
            segment slice).
        base_addr: Virtual address of *code*[0].
        resolve_callee: Optional callable mapping a call-target address
            to a function name (used to populate block ``callees``).
            PLT-stub resolution lives in the caller (TargetDistance owns
            the symbol table).

    Returns:
        The FunctionCFG (possibly with no blocks for degenerate input).
    """
    func_end = base_addr + len(code)
    cfg = FunctionCFG(name=name, start=base_addr, end=func_end)

    # Pre-scan for decoder-gap fixups: address -> (kind, target, extra_len)
    fixups: dict[int, tuple] = {}
    for off in range(len(code)):
        b = code[off]
        addr = base_addr + off
        if b in (0xC2, 0xCA):  # ret imm16 (3 bytes: opcode + imm16)
            fixups[addr] = (_RET, None, 3)
        elif 0xE0 <= b <= 0xE3:  # loop/jcxz: jcc rel8 (2 bytes)
            if off + 1 < len(code):
                disp = code[off + 1]
                if disp >= 0x80:
                    disp -= 0x100
                target = addr + 2 + disp
            else:
                target = addr
            fixups[addr] = (_JCC, target, 2)

    # Decode, applying fixups and dropping bytes consumed by them.
    insns = []  # (address, insn, kind, target)
    skip_until = -1
    for insn in _decode_x86_64(code, base_addr):
        if insn.address < skip_until or len(insns) >= _MAX_INSNS_PER_FUNC:
            continue
        kind, target, extra_len = _classify(insn, fixups)
        if extra_len:
            insn.length = extra_len
            skip_until = insn.address + extra_len
        insns.append((insn.address, insn, kind, target))

    if not insns:
        return cfg

    # Leaders: function entry, after every terminator, and taken targets.
    leaders = {base_addr}
    for addr, insn, kind, target in insns:
        if kind in (_JCC, _JMP, _CALL, _RET):
            leaders.add(addr + insn.length)
        if kind in (_JCC, _JMP) and target is not None and base_addr <= target < func_end:
            leaders.add(target)

    # Group instructions into blocks.
    blocks: dict[int, BasicBlock] = {}
    cur_start = base_addr
    cur_insns: list = []
    for addr, insn, kind, target in insns:
        if addr in leaders and addr != base_addr and cur_insns:
            blocks[cur_start] = _close_block(
                cur_start, cur_insns, base_addr, func_end, resolve_callee
            )
            cur_start = addr
            cur_insns = []
        cur_insns.append((addr, insn, kind, target))
    if cur_insns:
        blocks[cur_start] = _close_block(cur_start, cur_insns, base_addr, func_end, resolve_callee)

    cfg.blocks = blocks
    # Successors must be actual block starts: decoder desync gaps can
    # leave a fall-through address that was never grouped into a block.
    for blk in blocks.values():
        blk.successors = [s for s in blk.successors if s in blocks]
    if base_addr in blocks:
        blocks[base_addr].is_entry = True
    return cfg


def _close_block(start, insns, base_addr, func_end, resolve_callee) -> BasicBlock:
    """Build a BasicBlock from its instruction list (terminator last)."""
    last_addr, last_insn, kind, target = insns[-1]
    blk = BasicBlock(start=start, end=last_addr + last_insn.length)

    # Callsites: every direct call inside the block names its callee.
    # A call whose target cannot be resolved (indirect form, or a direct
    # target outside the known function set) is flagged so callers can
    # treat it as a CG gap for runtime-edge patching.
    for k, tgt in ((i[2], i[3]) for i in insns):
        if k == _CALL:
            if tgt is not None and resolve_callee is not None:
                callee = resolve_callee(tgt)
                if callee:
                    blk.callees.add(callee)
                else:
                    blk.indirect_call = True
            elif tgt is None:
                blk.indirect_call = True

    def _intra(addr: int) -> bool:
        return base_addr <= addr < func_end

    if kind == _RET:
        blk.is_exit = True
        return blk

    fall = last_addr + last_insn.length
    if kind == _JCC:
        if target is not None and _intra(target):
            blk.successors.append(target)
        if _intra(fall):
            blk.successors.append(fall)
    elif kind == _JMP:
        if target is None:
            blk.indirect_jump = True
        elif _intra(target):
            blk.successors.append(target)
    elif kind in (_CALL, _FALL):
        if _intra(fall):
            blk.successors.append(fall)
    return blk
