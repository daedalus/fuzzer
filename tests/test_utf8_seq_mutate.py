"""Tests for the sequence-aware UTF-8 operator (`utf8_seq_mutate`).

The two existing UTF-8 operators only *add* bytes: `utf8_widen` rewrites an
ASCII byte as a 2-byte overlong, `utf8_insert` splices a curated blob in at a
random offset. Neither decodes the buffer, so no operator in the table ever
mutates a multi-byte sequence that is already there. This one does, which is
why every assertion below is about an existing sequence rather than an
insertion point.
"""

import pytest

from fuzzer_tool.core.mutations import utf8
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.operators import OperatorEngine

from .support.operator_env import make_minimal_fuzzer
from .support.scripted_rng import ScriptedRng

# "A" then U+00E9 (2 bytes), U+6771 (3 bytes), U+1F600 (4 bytes), then "Z".
MIXED = "A\u00e9\u6771\U0001f600Z".encode()
E_ACUTE_AT = 1
CJK_AT = 3
EMOJI_AT = 6


def _mode(fn) -> int:
    """Index of *fn* in the mode table, for scripting `choice()`."""
    return utf8.MODES.index(fn)


# ── the unvalidated encoder ──────────────────────────────────────────


class TestEncode:
    @pytest.mark.parametrize("cp", [0x00, 0x41, 0x7F, 0x80, 0xE9, 0x7FF, 0x800, 0x6771, 0xFFFF])
    def test_minimal_width_matches_the_reference_encoder(self, cp):
        """Control: at minimal width the encoder must agree with CPython."""
        expected = chr(cp).encode()
        assert utf8.encode_width(cp, len(expected)) == expected

    def test_astral_minimal_width_matches_the_reference_encoder(self):
        assert utf8.encode_width(0x1F600, 4) == "\U0001f600".encode()

    def test_wider_than_minimal_is_an_overlong_of_the_same_code_point(self):
        # 'A' at width 2/3/4 is exactly the classic overlong ladder.
        assert utf8.encode_width(0x41, 2) == b"\xc1\x81"
        assert utf8.encode_width(0x41, 3) == b"\xe0\x81\x81"
        assert utf8.encode_width(0x41, 4) == b"\xf0\x80\x81\x81"

    def test_reaches_the_retired_five_and_six_byte_forms(self):
        """RFC 3629 removed these; decoders that still accept them are the point."""
        assert utf8.encode_width(0x10FFFF, 5) == b"\xf8\x84\x8f\xbf\xbf"
        assert utf8.encode_width(0x10FFFF, 6) == b"\xfc\x80\x84\x8f\xbf\xbf"

    def test_encodes_the_first_code_point_past_the_unicode_range(self):
        assert utf8.encode_width(0x110000, 4) == b"\xf4\x90\x80\x80"

    def test_encodes_a_lone_surrogate(self):
        assert utf8.encode_width(0xD800, 3) == b"\xed\xa0\x80"


# ── sequence location ────────────────────────────────────────────────


class TestFindSeq:
    def test_finds_the_sequence_covering_the_drawn_offset(self):
        # Offset 4 is the middle continuation byte of the 3-byte U+6771.
        assert utf8.find_seq(MIXED, 4) == (CJK_AT, 3)

    def test_snaps_forward_when_the_offset_is_ascii(self):
        assert utf8.find_seq(MIXED, 0) == (E_ACUTE_AT, 2)

    def test_declines_on_a_buffer_with_no_multi_byte_sequence(self):
        assert utf8.find_seq(b"plain ascii only", 0) is None

    def test_declines_past_the_search_window(self):
        far = b"a" * (utf8.SEQ_SEARCH_WINDOW + 8) + "\u00e9".encode()
        assert utf8.find_seq(far, 0) is None
        assert utf8.find_seq(far, len(far) - 4) is not None

    def test_skips_a_truncated_sequence(self):
        """A lead byte with its continuation bytes missing has no code point."""
        assert utf8.find_seq(b"\xe6\x9d" + b"ascii", 0) is None

    def test_skips_a_sequence_that_is_already_overlong(self):
        # 0xE0 0x81 0x81 is 'A' spelled in three bytes.
        assert utf8.find_seq(b"\xe0\x81\x81 tail", 0) is None

    def test_skips_a_sequence_that_is_already_a_surrogate(self):
        assert utf8.find_seq(b"\xed\xa0\x80 tail", 0) is None

    def test_walks_past_a_bad_candidate_to_a_good_one(self):
        # A truncated 3-byte lead, then a well-formed 2-byte sequence.
        data = b"\xe6\x9d" + "\u00e9".encode()
        assert utf8.find_seq(data, 0) == (2, 2)


# ── one test per mode, exact output, scripted draws ──────────────────


class TestModes:
    def test_truncate_drops_trailing_continuation_bytes(self):
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_truncate)], randints=[2])
        out = utf8.seq_mutate(MIXED, EMOJI_AT, rng=rng, max_len=64)
        assert out == MIXED[:EMOJI_AT] + b"\xf0\x9f" + MIXED[EMOJI_AT + 4 :]

    def test_orphan_drops_the_lead_byte(self):
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_orphan)])
        out = utf8.seq_mutate(MIXED, CJK_AT, rng=rng, max_len=64)
        assert out == MIXED[:CJK_AT] + MIXED[CJK_AT + 1 : CJK_AT + 3] + MIXED[CJK_AT + 3 :]

    def test_widen_re_encodes_the_same_code_point_wider(self):
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_widen)], randints=[4])
        out = utf8.seq_mutate(MIXED, E_ACUTE_AT, rng=rng, max_len=64)
        assert out == MIXED[:E_ACUTE_AT] + utf8.encode_width(0xE9, 4) + MIXED[E_ACUTE_AT + 2 :]
        # Same code point, wider encoding — that is the whole claim.
        assert out.replace(utf8.encode_width(0xE9, 4), "\u00e9".encode()) == MIXED

    def test_surrogate_splits_an_astral_code_point_into_a_cesu8_pair(self):
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_surrogate)])
        out = utf8.seq_mutate(MIXED, EMOJI_AT, rng=rng, max_len=64)
        hi, lo = 0xD83D, 0xDE00  # U+1F600 as a surrogate pair
        pair = utf8.encode_width(hi, 3) + utf8.encode_width(lo, 3)
        assert out == MIXED[:EMOJI_AT] + pair + MIXED[EMOJI_AT + 4 :]

    def test_surrogate_substitutes_a_lone_half_on_a_bmp_sequence(self):
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_surrogate), 0])
        out = utf8.seq_mutate(MIXED, CJK_AT, rng=rng, max_len=64)
        expected = utf8.encode_width(utf8.SURROGATE_CPS[0], 3)
        assert out == MIXED[:CJK_AT] + expected + MIXED[CJK_AT + 3 :]

    def test_boundary_substitutes_a_range_check_code_point(self):
        idx = utf8.BOUNDARY_CPS.index(0x110000)
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_boundary), idx])
        out = utf8.seq_mutate(MIXED, E_ACUTE_AT, rng=rng, max_len=64)
        assert out == MIXED[:E_ACUTE_AT] + b"\xf4\x90\x80\x80" + MIXED[E_ACUTE_AT + 2 :]

    def test_boundary_declines_when_it_draws_the_code_point_already_there(self):
        data = "x\u07ffy".encode()
        idx = utf8.BOUNDARY_CPS.index(0x7FF)
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_boundary), idx])
        assert utf8.seq_mutate(data, 1, rng=rng, max_len=64) is None

    def test_cont_flip_breaks_one_continuation_byte_and_keeps_the_length(self):
        # 0x30 is below 0x80, so it is returned unmapped.
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_cont_flip)], randints=[2, 0x30])
        out = utf8.seq_mutate(MIXED, EMOJI_AT, rng=rng, max_len=64)
        assert len(out) == len(MIXED)
        assert out[EMOJI_AT + 2] == 0x30
        assert out[:EMOJI_AT] == MIXED[:EMOJI_AT]

    def test_widen_can_never_draw_the_width_it_already_has(self):
        """Kills the off-by-one: `randint(len(seq), 6)` would re-emit the input.

        The identity guard in `seq_mutate` would turn that into a decline
        rather than a wrong answer, which is exactly why the bound needs
        its own assertion -- otherwise the operator quietly loses a share
        of its draws and nothing fails.
        """
        bounds = []

        class _Recorder(ScriptedRng):
            def randint(self_, a, b):
                bounds.append((a, b))
                return super().randint(a, b)

        for off, length in ((E_ACUTE_AT, 2), (CJK_AT, 3), (EMOJI_AT, 4)):
            rng = _Recorder(choice_idxs=[_mode(utf8.mode_widen)], randints=[6])
            utf8.seq_mutate(MIXED, off, rng=rng, max_len=512)
            assert bounds[-1] == (length + 1, 6)

    def test_cont_flip_never_writes_another_continuation_byte(self):
        """The replacement is drawn from [0x00,0x7F] u [0xC0,0xFF], never [0x80,0xBF]."""
        drawn = {utf8.non_cont_byte(v) for v in range(0xC0)}
        assert drawn == set(range(0x80)) | set(range(0xC0, 0x100))


# ── declines and the max_len contract ────────────────────────────────


class TestDeclines:
    def test_declines_when_there_is_nothing_to_decode(self):
        rng = ScriptedRng(choice_idxs=[0], randints=[1, 1])
        assert utf8.seq_mutate(b"pure ascii", 0, rng=rng, max_len=64) is None

    def test_declines_on_an_empty_buffer(self):
        assert utf8.seq_mutate(b"", 0, rng=ScriptedRng(), max_len=64) is None

    def test_declines_rather_than_truncating_past_max_len(self):
        """Adversarial: widening would overrun, and a clamp would cut the sequence."""
        data = "\u00e9ab".encode()  # 4 bytes
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_widen)], randints=[6])
        assert utf8.seq_mutate(data, 0, rng=rng, max_len=len(data)) is None

    def test_widens_when_there_is_room(self):
        data = "\u00e9ab".encode()
        rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_widen)], randints=[6])
        out = utf8.seq_mutate(data, 0, rng=rng, max_len=len(data) + 4)
        assert out == utf8.encode_width(0xE9, 6) + b"ab"

    def test_max_len_holds_over_every_mode_and_offset(self):
        cap = len(MIXED)
        for mode_idx in range(len(utf8.MODES)):
            for off in range(len(MIXED)):
                rng = ScriptedRng(choice_idxs=[mode_idx, 0], randints=[1, 0x41, 1])
                out = utf8.seq_mutate(MIXED, off, rng=rng, max_len=cap)
                assert out is None or len(out) <= cap

    def test_output_is_never_the_input(self):
        """A no-op returned as a mutation would credit the operator for nothing."""
        for mode_idx in range(len(utf8.MODES)):
            for off in range(len(MIXED)):
                rng = ScriptedRng(choice_idxs=[mode_idx, 0], randints=[1, 0x41, 1])
                out = utf8.seq_mutate(MIXED, off, rng=rng, max_len=512)
                assert out != MIXED


# ── wiring ───────────────────────────────────────────────────────────


class TestOperatorWiring:
    def setup_method(self):
        self.f = make_minimal_fuzzer(0x5EED)
        self.engine = OperatorEngine(self.f)

    def test_registered_alongside_its_two_siblings(self):
        assert REGISTRY.category_of("utf8_seq_mutate") == REGISTRY.category_of("utf8_widen")

    def test_reachable_through_the_dispatch_table(self):
        assert "utf8_seq_mutate" in self.engine.build_dispatch()

    def test_handler_mutates_the_sequence_at_the_drawn_offset(self):
        # `ctx` snapshots the fuzzer at construction, so the seam is `f._rng`.
        self.f._rng = ScriptedRng(choice_idxs=[_mode(utf8.mode_orphan)], randints=[1, 1])
        self.engine = OperatorEngine(self.f)
        buf = bytearray(MIXED)
        out = self.engine._op_utf8_seq_mutate(buf, CJK_AT, b"")
        assert bytes(out) == MIXED[:CJK_AT] + MIXED[CJK_AT + 1 : CJK_AT + 3] + MIXED[CJK_AT + 3 :]

    def test_handler_records_a_decline_instead_of_faking_a_mutation(self):
        buf = bytearray(b"pure ascii")
        out = self.engine._op_utf8_seq_mutate(buf, 0, b"")
        assert bytes(out) == bytes(buf)
        assert self.f._op_declines.get("utf8_seq_mutate") == 1
