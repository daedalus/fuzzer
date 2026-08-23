"""Tests for the bit rotation / shift / inversion operators.

Before these three the operator table could toggle bits (bit_flip,
bit_offset_span) and reorder them (bit_transpose_*) but could not translate
them: nothing moved a packed bitfield off its alignment, nothing dropped bits
off the end of a word, and nothing inverted a non-byte-aligned run.

The properties pinned here are the ones that distinguish each operator from
its neighbours -- rotation preserves popcount, shift does not, span_invert
always changes the buffer -- plus the invariants every operator in this
codebase must hold: length preservation where claimed, reproducibility under a
seed, and a non-trivial change rate so the operator cannot silently become the
no-op that hid in byte_shuffle until f4835f6.
"""

import random

from fuzzer_tool.core.mutations import MUTATIONS, bit_rotate, bit_shift, span_invert
from fuzzer_tool.core.operator_registry import REGISTRY

_OPS = (bit_rotate, bit_shift, span_invert)
_NAMES = ("bit_rotate", "bit_shift", "span_invert")


def _popcount(b: bytes) -> int:
    return sum(bin(x).count("1") for x in b)


class TestSharedInvariants:
    """Properties all three must hold, whatever they do internally."""

    def test_length_preserved(self):
        data = bytes(range(64))
        for op in _OPS:
            for i in range(100):
                assert len(op(data, random.Random(i))) == len(data), op.__name__

    def test_deterministic_under_seed(self):
        data = bytes(range(64))
        for op in _OPS:
            assert op(data, random.Random(7)) == op(data, random.Random(7)), op.__name__

    def test_empty_input_returned_unchanged(self):
        for op in _OPS:
            assert op(b"", random.Random(1)) == b"", op.__name__

    def test_single_byte_input_is_handled(self):
        """A 1-byte buffer is a legitimate 8-bit window, not a decline."""
        for op in _OPS:
            out = op(b"\x5a", random.Random(3))
            assert len(out) == 1, op.__name__

    def test_short_buffers_never_raise(self):
        """Word-width selection must clamp to what the buffer can hold."""
        for op in _OPS:
            for n in range(1, 10):
                for i in range(20):
                    op(bytes(range(n)), random.Random(i))

    def test_change_rate_is_high(self):
        """The no-op guard. A structured buffer should almost always move."""
        data = bytes(range(64))
        for op, name in zip(_OPS, _NAMES, strict=True):
            changed = sum(op(data, random.Random(i)) != data for i in range(200))
            assert changed >= 180, f"{name} changed only {changed}/200"


class TestBitRotate:
    def test_preserves_popcount(self):
        """Rotation is a permutation of bits: popcount is invariant.

        This is what separates it from bit_shift, which is lossy.
        """
        for i in range(200):
            data = random.Random(i).randbytes(16)
            assert _popcount(bit_rotate(data, random.Random(i))) == _popcount(data), i

    def test_reaches_sub_byte_rotations(self):
        """Rotation is *not* a byte permutation -- that is the whole point.

        A sub-byte rotation changes byte values, so it reaches states that
        swap_bytes and endianness_swap cannot.

        A rotation by a multiple of 8 only permutes whole bytes inside the
        window, leaving the buffer's byte multiset intact. A sub-byte rotation
        changes the byte values themselves. Measured rate is ~98%; the floor
        sits well above the ~53% a byte-aligned-only implementation reaches
        via the 8-bit-window path, which is the regression this pins.
        """
        data = bytes(range(64))
        differing = sum(
            sorted(bit_rotate(data, random.Random(i))) != sorted(data) for i in range(200)
        )
        assert differing >= 170, (
            f"only {differing}/200 rotations left byte boundaries; "
            "sub-byte amounts may have been lost"
        )

    def test_uniform_window_is_rotation_invariant(self):
        """All-zero input cannot be rotated into anything else.

        Documented rather than fixed: it is inherent to rotation, not a bug.
        The retry loop only re-samples the offset, which cannot help when
        every window is identical.
        """
        data = bytes(64)
        assert all(bit_rotate(data, random.Random(i)) == data for i in range(50))


class TestBitShift:
    def test_is_lossy(self):
        """Unlike rotation, shifting must sometimes destroy set bits."""
        data = bytes(range(1, 65))
        lost = sum(
            _popcount(bit_shift(data, random.Random(i))) != _popcount(data) for i in range(200)
        )
        assert lost >= 100, "shift never changed popcount; it may be rotating"

    def test_arithmetic_right_shift_sign_propagates(self):
        """A negative word stays negative under the sar variant.

        An all-ones window is the discriminating input: shl and shr both
        clear bits at one end and so must return something different, which
        leaves sar as the only variant that can hand back the input intact.
        Seeing the input in the output set therefore proves sign-fill
        happened; never seeing it would mean sar is zero-filling like shr.
        """
        data = b"\xff" * 16
        outs = {bit_shift(data, random.Random(i)) for i in range(300)}
        assert data in outs, "sar must sign-fill: an all-ones word shifts to itself"

    def test_zero_input_stays_zero(self):
        """Shifting zero yields zero under every variant."""
        data = bytes(32)
        assert all(bit_shift(data, random.Random(i)) == data for i in range(50))


class TestSpanInvert:
    def test_always_changes_the_buffer(self):
        """The XOR mask is non-zero by construction, so there is no no-op path.

        Holds even for degenerate input where rotate and shift cannot move.
        """
        for data in (bytes(64), b"\xff" * 64, bytes(range(64))):
            assert all(span_invert(data, random.Random(i)) != data for i in range(200))

    def test_is_an_involution_under_a_repeated_seed(self):
        """Inverting the same span twice restores the input.

        Both calls draw span and start from the same seed and the same buffer
        length, so the second call targets the same run. Genuine round-trip:
        an off-by-one in either partial-byte edge mask makes the two masks
        differ and the input fails to come back.
        """
        data = bytes(range(64))
        for i in range(100):
            once = span_invert(data, random.Random(i))
            assert span_invert(once, random.Random(i)) == data, i

    def test_mask_is_contiguous_in_bit_offset_space(self):
        """XOR delta must be one unbroken run of set bits, LSB-first per byte.

        Bit offset i is byte i >> 3, bit i & 7 -- the numbering
        _op_bit_offset_flip uses. A span that wrapped or split would show as
        two runs here.
        """
        data = bytes(64)
        for i in range(200):
            out = span_invert(data, random.Random(i))
            bits = [(out[o >> 3] >> (o & 7)) & 1 for o in range(8 * len(out))]
            runs = sum(1 for a, b in zip([0] + bits, bits, strict=False) if b and not a)
            assert runs == 1, f"seed {i}: mask had {runs} runs, expected 1"

    def test_reaches_sub_byte_and_whole_buffer_spans(self):
        """byte_flip only inverts one aligned byte; this must do strictly more."""
        data = bytes(64)
        widths = {_popcount(span_invert(data, random.Random(i))) for i in range(400)}
        assert min(widths) < 8, "no sub-byte span observed"
        assert max(widths) == 8 * len(data), "whole-buffer inversion unreachable"


class TestRegistryWiring:
    """The operators must be reachable by the scheduler, not just importable."""

    def test_registered_in_the_bit_band(self):
        for name in _NAMES:
            assert name in REGISTRY.names(), f"{name} not registered"
            assert REGISTRY.category_of(name) == "bit", f"{name} not in the bit band"

    def test_present_in_the_legacy_mutation_list(self):
        for name in _NAMES:
            assert name in MUTATIONS, f"{name} missing from MUTATIONS"

    def test_unconditionally_available(self):
        """No availability predicate: like bit_flip, these work on any input."""
        for name in _NAMES:
            assert REGISTRY._ops[name].available is None, f"{name} gained a gate unexpectedly"
