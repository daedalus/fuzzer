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

from fuzzer_tool.core.mutations import (
    MUTATIONS,
    bit_repack,
    bit_rotate,
    bit_shift,
    span_invert,
)
from fuzzer_tool.core.mutations.generic import _REPACK_WIDTHS, _repack_bits
from fuzzer_tool.core.operator_registry import REGISTRY, format_gate_matches

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


class TestBitRepack:
    """bit_repack moves the element boundary grid rather than the bits in it.

    The properties worth pinning are the ones that make it not-a-no-op by
    construction (widths always differ), not-a-truncator (max_len is honoured
    as a real constraint, not a trailing clamp), and not-ungated (it is
    mechanically applicable everywhere and meaningful only on packed formats).
    """

    def test_length_scales_with_the_width_ratio(self):
        """Widening grows the buffer, narrowing shrinks it.

        The observable signature of an actual repack: an operator that only
        permuted bits could not change length at all.
        """
        data = bytes(range(32))
        lens = {len(bit_repack(data, random.Random(i))) for i in range(400)}
        assert min(lens) < len(data), "no narrowing repack observed"
        assert max(lens) > len(data), "no widening repack observed"

    def test_never_an_identity_repack_on_structured_input(self):
        """src_w == dst_w is unsampleable, so real data always moves.

        This is the byte_shuffle lesson applied in advance: the degenerate
        *width pair* is excluded by construction rather than caught after the
        fact. Note that is a narrower claim than "the output always differs" --
        see the uniform-input case below.
        """
        for data in (bytes(range(32)), b"\xff" * 32, bytes(range(64))):
            unchanged = sum(bit_repack(data, random.Random(i)) == data for i in range(300))
            assert unchanged == 0, f"{unchanged}/300 repacks were no-ops"

    def test_uniform_input_can_repack_to_itself(self):
        """Repacking an all-zero span yields an all-zero span.

        Documented rather than fixed, like bit_rotate on a uniform window:
        when the width change also happens to round to the same byte length,
        the result coincides with the input. Measured at ~2% on an all-zero
        buffer and 0% on structured input, so it costs a negligible share of
        selections and is inherent to the transform, not a defect.
        """
        data = bytes(32)
        unchanged = sum(bit_repack(data, random.Random(i)) == data for i in range(300))
        assert unchanged < 30, f"{unchanged}/300 is too high to be the rounding coincidence"

    def test_respects_max_len_at_every_cap(self):
        """Mirrors test_no_operator_exceeds_max_len, which cannot reach this op.

        The exhaustive pool classifies bit_repack over_budget rather than
        enumerated, so the suite-wide max_len invariant silently skips it.
        A length-changing operator is exactly the wrong one to leave uncovered.
        """
        for max_len, seed in ((8, bytes(range(8))), (4, b"abcd"), (2, b"ab"), (1, b"a")):
            worst = max(
                len(bit_repack(seed, random.Random(i), max_len=max_len)) for i in range(500)
            )
            assert worst <= max_len, f"max_len={max_len}: emitted {worst} bytes"

    def test_declines_rather_than_truncating_when_nothing_fits(self):
        """A cap too tight for any span returns the input, not a cut-down one.

        Truncating a repacked stream mid-element yields a buffer that is not
        a valid packing at either width -- worse than declining, and it would
        show up as a healthy change% while being pure noise.
        """
        seed = bytes(range(8))
        outs = {bit_repack(seed, random.Random(i), max_len=len(seed)) for i in range(300)}
        assert all(len(o) <= len(seed) for o in outs)
        assert seed in outs, "no sampled span declined; the cap is not being honoured by decline"

    def test_deterministic_under_seed(self):
        data = bytes(range(32))
        assert bit_repack(data, random.Random(11)) == bit_repack(data, random.Random(11))

    def test_short_input_declines(self):
        for n in (0, 1):
            seed = bytes(range(n))
            assert bit_repack(seed, random.Random(1)) == seed

    def test_reaches_both_bit_orders(self):
        """MSB-first and LSB-first packings must both actually be sampled.

        A parser is usually correct for only one order, so hardwiring either
        halves the operator's reach. Checked by enumerating the complete
        MSB-only reachable set for a small seed and showing that live sampling
        escapes it -- a distinct-output count cannot tell the two apart,
        because one order alone still produces plenty of distinct outputs.
        """
        seed = bytes(range(4))
        msb_only = set()
        for src_w in set(_REPACK_WIDTHS):
            for dst_w in set(_REPACK_WIDTHS) - {src_w}:
                for span_len in range(1, len(seed) + 1):
                    for start in range(len(seed) - span_len + 1):
                        for scale in (False, True):
                            packed = _repack_bits(
                                seed[start : start + span_len], src_w, dst_w, True, scale
                            )
                            msb_only.add(seed[:start] + packed + seed[start + span_len :])
        live = {bit_repack(seed, random.Random(i), max_len=4096) for i in range(600)}
        assert live - msb_only, "every sampled output was MSB-first; LSB-first is unreachable"


class TestBitRepackGating:
    """The op is sniffer-gated, which the no-op sweep cannot tell you."""

    def test_is_gated_not_unconditional(self):
        assert REGISTRY._ops["bit_repack"].available is not None, (
            "bit_repack must stay gated: it changes bytes on any input, so an "
            "ungated version scores ~100% change% while producing garbage for "
            "targets that do not parse packed samples"
        )

    def test_sniffer_matches_packed_sample_formats(self):
        for magic in (
            b"\x89PNG\r\n\x1a\n" + bytes(32),
            b"BM" + bytes(32),
            b"GIF89a" + bytes(32),
            b"II*\x00" + bytes(32),
            b"MM\x00*" + bytes(32),
        ):
            assert format_gate_matches("bit_repack", magic) is True, magic[:8]

    def test_sniffer_rejects_unrelated_input(self):
        for data in (b"\xff\xd8\xff\xe0" + bytes(32), b'{"a": 1}', bytes(32), b"\x7fELF"):
            assert format_gate_matches("bit_repack", data) is False, data[:8]

    def test_registered_in_the_bit_band(self):
        assert "bit_repack" in REGISTRY.names()
        assert REGISTRY.category_of("bit_repack") == "bit"
        assert "bit_repack" in MUTATIONS


class TestRepackBits:
    """Direct tests of the packing core, with parameters fixed.

    bit_repack only samples parameters; everything that can be wrong about
    the packing itself lives in _repack_bits, and sampling makes those
    properties nearly impossible to pin from the outside.
    """

    def test_msb_stream_is_left_aligned(self):
        """An MSB-first stream must start at the top bit of the first byte.

        4 bytes of 8-bit elements re-emitted at 6 bits is 24 bits of payload
        in 3 bytes, so nothing is left over and the first element lands in
        the top 6 bits. Right-aligning the stream instead shifts every
        element and an MSB-first reader decodes garbage.
        """
        span = b"\xff\x00\xff\x00"
        out = _repack_bits(span, 8, 6, msb_first=True, scale=False)
        assert out[0] >> 2 == 0b111111, f"first element not at the top bit: {out[0]:08b}"

    def test_msb_padding_goes_in_the_low_bits(self):
        """With a partial final byte, the pad must be at the end of the stream.

        2 bytes of 8-bit elements at 6 bits is 12 payload bits in 2 bytes, so
        4 pad bits exist. They belong after the payload, not before it.
        """
        out = _repack_bits(b"\xff\xff", 8, 6, msb_first=True, scale=False)
        assert out[0] == 0xFF, f"payload not left-aligned: {out.hex()}"
        assert out[1] & 0x0F == 0, f"pad bits not in the low nibble: {out.hex()}"

    def test_lsb_stream_starts_at_the_low_bit(self):
        out = _repack_bits(b"\xff\x00", 8, 4, msb_first=False, scale=False)
        assert out[0] & 0x0F == 0x0F, f"first element not at the low bit: {out.hex()}"

    def test_bit_orders_disagree(self):
        """The two orders must be genuinely different transforms."""
        span = bytes(range(4))
        for src_w, dst_w in ((8, 4), (4, 8), (8, 12), (2, 8)):
            a = _repack_bits(span, src_w, dst_w, True, False)
            b = _repack_bits(span, src_w, dst_w, False, False)
            assert a != b, f"{src_w}->{dst_w} identical under both bit orders"

    def test_widening_then_narrowing_round_trips_when_masking(self):
        """Mask-mode widening is lossless, so the inverse restores the span.

        The strongest correctness statement available for the element walk:
        an off-by-one in the shift, the tail carry, or the padding breaks the
        round-trip. Only holds for widths that divide the span evenly, so no
        sub-element tail is involved.
        """
        span = bytes(range(8))
        for msb in (True, False):
            wide = _repack_bits(span, 4, 8, msb, scale=False)
            back = _repack_bits(wide, 8, 4, msb, scale=False)
            assert back == span, f"msb_first={msb}: {back.hex()} != {span.hex()}"

    def test_scale_mode_maps_full_range_to_full_range(self):
        """Proportional scaling sends an all-ones element to all-ones.

        This is what a real bit-depth conversion does -- PNG expanding 4-bit
        to 8-bit sends 0xF to 0xFF, not to 0x0F.
        """
        out = _repack_bits(b"\xff", 4, 8, msb_first=True, scale=True)
        assert out == b"\xff\xff", out.hex()

    def test_mask_mode_zero_extends_instead(self):
        """Masking sends 0xF to 0x0F -- the boundary-hunting variant."""
        out = _repack_bits(b"\xff", 4, 8, msb_first=True, scale=False)
        assert out == b"\x0f\x0f", out.hex()

    def test_sub_element_tail_is_carried(self):
        """3 bytes at 5-bit elements leaves 4 tail bits that must survive.

        Dropping the tail would make the operator lossy for reasons unrelated
        to the repack, and would shorten output unpredictably.
        """
        span = b"\xab\xcd\xef"
        for src_w, dst_w in ((5, 7), (3, 5), (7, 3)):
            n_el = len(span) * 8 // src_w
            rem = len(span) * 8 - n_el * src_w
            expect = (n_el * dst_w + rem + 7) // 8
            assert len(_repack_bits(span, src_w, dst_w, True, False)) == expect

    def test_sub_element_tail_content_survives(self):
        """The tail bits must arrive intact, not merely be counted.

        Length alone does not pin this: shifting the accumulator by rem_bits
        without OR-ing the tail in reserves exactly the right space and
        zero-fills it, so an output of the correct size can still have lost
        the data. Checked by reading the tail back out of the stream.
        """
        span = b"\xab\xcd\xef"
        src_w, dst_w = 5, 7
        span_bits = len(span) * 8
        n_el = span_bits // src_w
        rem = span_bits - n_el * src_w
        tail_mask = (1 << rem) - 1
        assert rem, "pick widths that actually leave a tail"

        out = _repack_bits(span, src_w, dst_w, True, False)
        expect = int.from_bytes(span, "big") & tail_mask
        got = int.from_bytes(out, "big") >> (len(out) * 8 - (n_el * dst_w + rem)) & tail_mask
        assert got == expect, f"msb tail lost: {got:#x} != {expect:#x}"

        out = _repack_bits(span, src_w, dst_w, False, False)
        expect = (int.from_bytes(span, "little") >> (n_el * src_w)) & tail_mask
        got = (int.from_bytes(out, "little") >> (n_el * dst_w)) & tail_mask
        assert got == expect, f"lsb tail lost: {got:#x} != {expect:#x}"
