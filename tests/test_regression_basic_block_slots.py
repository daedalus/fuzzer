"""Regression: BasicBlock carried a 296 B __dict__ per instance.

ffmpeg's ICFG build decodes 2.74M blocks at once. Slots cut the object from
352 B to 96 B; a shared empty ``callees`` saves 216 B more per call-free block.
"""

import io
import pickle
from types import SimpleNamespace

from fuzzer_tool.core import cfg_cache
from fuzzer_tool.core.cfg import _CALL, _FALL, _RET, BasicBlock, FunctionCFG, _close_block

_INSN = SimpleNamespace(length=5)
_CALLEES = {0x100: "a", 0x200: "b"}


def test_block_has_no_dict():
    """Falsification: no per-instance __dict__."""
    assert not hasattr(BasicBlock(start=0x10, end=0x20), "__dict__")


def test_cache_round_trip():
    """Adversarial: the restricted cache unpickler still loads slotted blocks."""
    blk = BasicBlock(0x10, 0x20, [0x20], {"callee"}, indirect_call=True, is_exit=True)
    cfgs = {"f": FunctionCFG("f", 0x10, 0x30, {0x10: blk})}

    data = pickle.dumps(cfgs, protocol=pickle.HIGHEST_PROTOCOL)
    loaded = cfg_cache._CfgUnpickler(io.BytesIO(data)).load()

    assert loaded == cfgs
    assert loaded["f"].blocks[0x10].callees == {"callee"}


def test_callee_free_blocks_share_empty():
    """Falsification: no per-block empty set (216 B x 2.74M on ffmpeg)."""
    assert BasicBlock(0x10, 0x20).callees is BasicBlock(0x30, 0x40).callees


def test_callees_stay_per_block():
    """Adversarial: a calling block owns its set; the shared default stays empty."""
    calls = [(0x10, _INSN, _CALL, 0x100), (0x15, _INSN, _CALL, 0x200), (0x1A, _INSN, _RET, None)]
    plain = [(0x30, _INSN, _FALL, None), (0x35, _INSN, _RET, None)]

    caller = _close_block(0x10, calls, 0x10, 0x40, _CALLEES.get)
    other = _close_block(0x30, plain, 0x10, 0x40, _CALLEES.get)

    assert caller.callees == set(_CALLEES.values())
    assert other.callees == frozenset()
    assert BasicBlock(0x50, 0x60).callees == frozenset()
