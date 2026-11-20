"""Tests for the shared elementary-stream helpers (`mutations/frame_stream.py`)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations.frame_stream import FrameOp, get_bits, reframe, set_bits

from .support.scripted_rng import ScriptedRng

FRAMES = [b"a", b"bb", b"ccc"]


class TestBits:
    def test_get_bits_msb_first(self):
        # 0b1011_0110: bits 1..3 are 011.
        assert get_bits(b"\xb6", 1, 3) == 0b011

    def test_set_bits_across_byte_boundary(self):
        buf = bytearray(2)
        set_bits(buf, 6, 4, 0b1111)
        assert buf == bytearray([0b0000_0011, 0b1100_0000])

    def test_set_bits_masks_value_and_keeps_neighbours(self):
        buf = bytearray(b"\xff\xff")
        set_bits(buf, 4, 4, 0x10)  # only the low 4 bits of the value fit
        assert buf == bytearray(b"\xf0\xff")

    @pytest.mark.parametrize(("pos", "width"), [(0, 16), (3, 9), (15, 1)])
    def test_round_trip(self, pos, width):
        buf = bytearray(b"\x5a\xa5")
        value = (1 << width) - 2
        set_bits(buf, pos, width, value)
        assert get_bits(buf, pos, width) == value


class TestReframe:
    def test_dup(self):
        assert reframe(FRAMES, FrameOp.DUP, ScriptedRng(randints=[1])) == [
            b"a",
            b"bb",
            b"bb",
            b"ccc",
        ]

    def test_drop(self):
        assert reframe(FRAMES, FrameOp.DROP, ScriptedRng(randints=[0])) == [b"bb", b"ccc"]

    def test_swap_skips_self(self):
        # j draw 1 maps past i=1 to 2: never a self-swap.
        assert reframe(FRAMES, FrameOp.SWAP, ScriptedRng(randints=[1, 1])) == [
            b"a",
            b"ccc",
            b"bb",
        ]

    def test_input_list_untouched(self):
        frames = list(FRAMES)
        reframe(frames, FrameOp.SWAP, ScriptedRng(randints=[0, 0]))
        assert frames == FRAMES

    # Falsification: a lone frame cannot be dropped or swapped.
    @pytest.mark.parametrize("op", [FrameOp.DROP, FrameOp.SWAP])
    def test_single_frame_declines(self, op):
        assert reframe([b"x"], op, ScriptedRng(randints=[0, 0])) is None

    # Adversarial: no frames at all, no draws consumed.
    @pytest.mark.parametrize("op", list(FrameOp))
    def test_empty_declines_without_drawing(self, op):
        assert reframe([], op, ScriptedRng()) is None
