"""Tests for core/mutations/tlul.tlul_mutate (TL-UL bus instruction streams)."""

import pytest

from fuzzer_tool.core.mutations.tlul import (
    ADDR_DELTAS,
    ADDR_EDGES,
    DATA_EDGES,
    MODES,
    WAIT_RUNS,
    frames,
    sniff_tlul,
    tlul_mutate,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_MAX = 4096
_U32_MASK = 0xFFFFFFFF

# Independent encoders: hw-fuzzing tlul_fuzz_instr.py defaults
# (constant opcodes wait=1 read=2 write=3, variable frames, little-endian).
WAIT = b"\x01"


def _rd(addr):
    return b"\x02" + addr.to_bytes(4, "little")


def _wr(addr, data):
    return b"\x03" + addr.to_bytes(4, "little") + data.to_bytes(4, "little")


def _mode(name):
    return [m.__name__ for m in MODES].index(name)


_SEQ = _wr(0x10, 7) + WAIT + _rd(0x14)


# ── Parsing ──


def test_frames_split_variable_instructions():
    assert list(frames(_SEQ)) == [0, 9, 10]


# Falsification: a truncated or unknown frame is not a stream.
@pytest.mark.parametrize("bad", [b"", _rd(4)[:-1], b"\x00", b"\x04" + bytes(8), _SEQ + b"\x03"])
def test_regression_frames_reject_malformed(bad):
    assert frames(bad) is None


def test_sniff_needs_a_bus_op():
    assert sniff_tlul(_SEQ) is True
    assert sniff_tlul(WAIT * 16) is False
    assert sniff_tlul(bytes(range(256))) is False
    assert sniff_tlul(_SEQ[:-1]) is False


def test_sniff_is_bounded_and_mutate_declines_bad_tail():
    blob = _wr(0, 1) * 64 + b"\xee" * 16
    assert sniff_tlul(blob) is True
    assert tlul_mutate(blob, ScriptedRng(choice_idxs=[0]), _MAX) is None


# ── Modes: exact outputs under a scripted RNG ──


def test_insert_instr_reuses_known_address():
    # boundary 1, opcode write, address pool = known (0x10, 0x14) + edges.
    rng = ScriptedRng(choice_idxs=[_mode("insert_instr"), 1, 2, 1, DATA_EDGES.index(_U32_MASK)])
    out = tlul_mutate(_SEQ, rng, _MAX)
    assert out == _wr(0x10, 7) + _wr(0x14, _U32_MASK) + WAIT + _rd(0x14)


def test_insert_instr_wait():
    rng = ScriptedRng(choice_idxs=[_mode("insert_instr"), 0, 0])
    assert tlul_mutate(_SEQ, rng, _MAX) == WAIT + _SEQ


def test_delete_instr():
    rng = ScriptedRng(choice_idxs=[_mode("delete_instr"), 1])
    assert tlul_mutate(_SEQ, rng, _MAX) == _wr(0x10, 7) + _rd(0x14)


def test_dup_instr_repeats_write():
    rng = ScriptedRng(choice_idxs=[_mode("dup_instr"), 0])
    assert tlul_mutate(_SEQ, rng, _MAX) == _wr(0x10, 7) * 2 + WAIT + _rd(0x14)


def test_swap_instr():
    rng = ScriptedRng(choice_idxs=[_mode("swap_instr"), 1])
    assert tlul_mutate(_SEQ, rng, _MAX) == _wr(0x10, 7) + _rd(0x14) + WAIT


def test_addr_swap_to_edge():
    edge = ADDR_EDGES[-1]
    rng = ScriptedRng(choice_idxs=[_mode("addr_swap"), 1, 2 + ADDR_EDGES.index(edge)])
    assert tlul_mutate(_SEQ, rng, _MAX) == _wr(0x10, 7) + WAIT + _rd(edge)


def test_addr_step_wraps_and_misaligns():
    seq = _rd(0) + _rd(0x10)
    rng = ScriptedRng(choice_idxs=[_mode("addr_step"), 0, ADDR_DELTAS.index(-4)])
    assert tlul_mutate(seq, rng, _MAX) == _rd(0xFFFFFFFC) + _rd(0x10)

    rng = ScriptedRng(choice_idxs=[_mode("addr_step"), 1, ADDR_DELTAS.index(1)])
    assert tlul_mutate(seq, rng, _MAX) == _rd(0) + _rd(0x11)


def test_data_edge():
    rng = ScriptedRng(choice_idxs=[_mode("data_edge"), 0, DATA_EDGES.index(0x80000000)])
    assert tlul_mutate(_SEQ, rng, _MAX) == _wr(0x10, 0x80000000) + WAIT + _rd(0x14)


def test_data_flip():
    rng = ScriptedRng(choice_idxs=[_mode("data_flip"), 0], randints=[31])
    assert tlul_mutate(_SEQ, rng, _MAX) == _wr(0x10, 7 ^ (1 << 31)) + WAIT + _rd(0x14)


def test_wait_run_inserts_clock_periods():
    run = WAIT_RUNS[0]
    rng = ScriptedRng(choice_idxs=[_mode("wait_run"), 1, 0])
    assert tlul_mutate(_SEQ, rng, _MAX) == _wr(0x10, 7) + WAIT * run + WAIT + _rd(0x14)


def test_rw_flip_both_directions():
    rng = ScriptedRng(choice_idxs=[_mode("rw_flip"), 0])
    assert tlul_mutate(_SEQ, rng, _MAX) == _rd(0x10) + WAIT + _rd(0x14)

    rng = ScriptedRng(choice_idxs=[_mode("rw_flip"), 1, DATA_EDGES.index(1)])
    assert tlul_mutate(_SEQ, rng, _MAX) == _wr(0x10, 7) + WAIT + _wr(0x14, 1)


# ── Declines ──


def test_bus_modes_decline_on_wait_only_stream():
    for name in ("addr_swap", "addr_step", "data_edge", "data_flip", "rw_flip"):
        assert tlul_mutate(WAIT * 4, ScriptedRng(choice_idxs=[_mode(name)]), _MAX) is None, name


def test_delete_and_swap_decline_on_single_instr():
    for name in ("delete_instr", "swap_instr"):
        assert tlul_mutate(_rd(0), ScriptedRng(choice_idxs=[_mode(name)]), _MAX) is None, name


def test_over_budget_declines():
    rng = ScriptedRng(choice_idxs=[_mode("dup_instr"), 0])
    assert tlul_mutate(_SEQ, rng, len(_SEQ)) is None


def test_noop_declines():
    # Swapping two identical frames changes nothing.
    rng = ScriptedRng(choice_idxs=[_mode("swap_instr"), 0])
    assert tlul_mutate(WAIT * 2, rng, _MAX) is None


# Adversarial: whatever the draws, output stays a parseable stream in budget.
@pytest.mark.parametrize("max_len", [1, 5, 9, 32, 4096])
def test_adversarial_output_stays_valid_stream(max_len):
    streams = [
        _SEQ,
        _rd(0xFFFFFFFF),
        _wr(0xFFFFFFFC, _U32_MASK) * 3,
        WAIT * 9 + _wr(0, 0),
        (_rd(4) + _wr(8, 0) + WAIT) * 40,
    ]
    for seed in range(40):
        rng = RandPool(seed=seed)
        for s in streams:
            out = tlul_mutate(s, rng, max_len)
            assert out is None or (len(out) <= max_len and frames(out) is not None)
