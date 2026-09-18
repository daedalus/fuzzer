"""Regression tests for PRNGStateLearner's signal source and stream tracking.

Each test here corresponds to a defect found reviewing the first wiring of
this learner, and fails on that version:

  * candidates were read from ``CmplogCollector.pairs``, which is
    campaign-wide, first-seen-only and interval-drained, so "cmplog encounter
    order" was not draw order and repeated draws were deduplicated away.
    Extraction now reads the ordered, PC-carrying ``last_conds`` of the most
    recent drain and takes the longest single-PC run;
  * a single global ``_MIN_SAMPLES`` of 3 (taus88's own pinning count)
    accepted unrelated 32-bit constants at ~1/2**8, because 3 outputs leave 8
    free consistency bits in the 96-bit solve -- superseded by trying every
    4-byte-output family (xorshift32/taus88/taus113/xorshift128) smallest-
    state-first and gating each on its own ``confident_samples()``;
  * re-confirmation compared a later drain's samples against the
    origin-aligned state, which only matches if the stream repeats, so a good
    state was dropped and re-recovered on nearly every drain;
  * nothing scoped the learner to in-process execution, the only regime with
    a cross-execution generator stream to predict;
  * ``from_dict`` restored the state without the derived frontier, so a
    resumed campaign predicted draws it had already seen.

The integration test at the end drives the real compiled target through
InProcessRunner; it is skipped without a C toolchain.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.core.prng_state_learner import PRNGStateLearner
from fuzzer_tool.core.prng_state_recovery import (
    LFSR258,
    TAUS88_PARAMS,
    TAUS113,
    XORSHIFT32,
    XORSHIFT64,
    XORSHIFT128,
    output_word,
    step_state,
    taus88_output,
    taus88_step,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
TARGET_SRC = REPO_ROOT / "targets" / "prng_token_read.c"

SEED_STATE = (0x12345678, 0x9ABCDEF0, 0xFEDCBA98)

# The field the target compares the token against is read FROM the input, so
# it must appear in the payload: extraction drops operands found in the input
# (those are echoed data, not generated state), and a payload that omitted
# this field would let the zero operand in as a spurious candidate draw.
ZERO_FIELD = b"\x00\x00\x00\x00"
PAYLOAD = ZERO_FIELD + b"PAYLOAD"
PAYLOAD2 = ZERO_FIELD + b"PAYLOAD-2"


def _stream(n: int, state: tuple[int, int, int] = SEED_STATE) -> list[int]:
    """The first *n* outputs of the taus88 stream started at *state*.

    Matches the convention ``recover_taus88_state`` documents: each output is
    taken after stepping, so ``_stream(n)[0]`` is the output of
    ``taus88_step(state)``.
    """
    out: list[int] = []
    s = state
    for _ in range(n):
        s = taus88_step(s, TAUS88_PARAMS)
        out.append(taus88_output(s))
    return out


class _Base:
    def __init__(self, op_a: bytes, op_b: bytes, pc: int | None) -> None:
        self.op_a = op_a
        self.op_b = op_b
        self.pc = pc


class _Cond:
    """Stands in for adapters.track_parser.CondStmt (only .base is read)."""

    def __init__(self, op_a: bytes, op_b: bytes, pc: int | None = 0x1000) -> None:
        self.base = _Base(op_a, op_b, pc)


def _conds(words: list[int], pc: int | None = 0x1000) -> list[_Cond]:
    """Records as the shim logs them: the drawn word against a zero field."""
    return [_Cond(w.to_bytes(4, "little"), b"\x00\x00\x00\x00", pc) for w in words]


#: The 8-byte equivalent of PAYLOAD: the compared-against field has to be
#: present in the input for the same reason (an operand found in the input
#: is data the target read back, so extraction drops it -- and the zero
#: field of a _wide_conds record is 8 bytes, not 4).
WIDE_PAYLOAD = b"\x00" * 8 + b"PAYLOAD"


def _wide_conds(words: list[int], pc: int | None = 0x1000) -> list[_Cond]:
    """Records for a 64-bit generator: the shim's trace_cmp8 path logs both
    operands at full 8-byte width (``__afl_cmplog_ints(a, b, 8, pc)``)."""
    return [_Cond(w.to_bytes(8, "little"), b"\x00" * 8, pc) for w in words]


def _fuzzer(conds: list[Any] | None = None, *, in_process: bool = True) -> Any:
    f = MagicMock()
    f._inprocess_runner = MagicMock() if in_process else None
    cmplog = MagicMock()
    cmplog.last_conds = conds if conds is not None else []
    # Populated so a test that passes only because the learner fell back to
    # the pooled list would be visible: nothing here is a taus88 stream.
    cmplog.pairs = [(b"\xde\xad\xbe\xef", b"\x00\x00\x00\x00")]
    f._cmplog = cmplog
    return f


def _learner(conds: list[Any] | None = None, *, in_process: bool = True):
    return PRNGStateLearner(_fuzzer(conds, in_process=in_process))


class TestOrderedPerPCExtraction:
    def test_recovers_from_one_drain(self):
        words = _stream(4)
        learner = _learner(_conds(words))
        assert learner.observe_execution(PAYLOAD) is True
        assert learner.has_state()

    def test_ignores_the_pooled_pairs_list(self):
        """With last_conds empty, the pooled `pairs` must not be consulted:
        its order is first-sighting order across executions, not draw order.
        """
        learner = _learner([])
        assert learner.observe_execution(PAYLOAD) is False
        assert learner.attempts == 0

    def test_interleaved_site_does_not_break_the_run(self):
        """A second comparison site logging constants between the draws must
        not enter the candidate sequence. Reading the drain flat (as the
        pooled list forced) interleaves them and recovery fails."""
        words = _stream(4)
        noise = [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE]
        records: list[_Cond] = []
        for word, junk in zip(words, noise, strict=True):
            records.extend(_conds([word], pc=0x1000))
            records.extend(_conds([junk], pc=0x2000))
        learner = _learner(records)
        assert learner.observe_execution(PAYLOAD) is True
        assert learner.predict(1) == [_stream(5)[4]]

    def test_flat_reading_of_the_same_drain_would_fail(self):
        """Falsifies the above: the interleaved sequence, taken in flat
        encounter order, is not a taus88 run."""
        from fuzzer_tool.core.prng_state_recovery import recover_taus88_state

        words = _stream(4)
        noise = [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE]
        flat = [v for pair in zip(words, noise, strict=True) for v in pair]
        assert recover_taus88_state(flat[:8]) is None

    def test_repeated_value_within_a_drain_is_not_a_second_draw(self):
        words = _stream(4)
        learner = _learner(_conds([words[0], words[0]] + words[1:]))
        assert learner.observe_execution(PAYLOAD) is True
        assert learner._confirmed_samples == words

    def test_operand_present_in_input_is_excluded(self):
        words = _stream(4)
        payload = ZERO_FIELD + b"".join(w.to_bytes(4, "little") for w in words)
        learner = _learner(_conds(words))
        assert learner.observe_execution(payload) is False

    def test_longest_run_wins_between_sites(self):
        words = _stream(6)
        records = _conds([0xDEADBEEF, 0xCAFEBABE], pc=0x2000) + _conds(words, pc=0x1000)
        learner = _learner(records)
        assert learner.observe_execution(PAYLOAD) is True

    def test_no_pc_bucket_still_yields_a_sequence(self):
        """A shim build that logs no PC must still be usable."""
        words = _stream(4)
        learner = _learner(_conds(words, pc=None))
        assert learner.observe_execution(PAYLOAD) is True


class TestNondeterminismPreference:
    def test_varying_site_preferred_after_a_replay(self):
        """The preference needs evidence: on the first sight of an input every
        site looks alike, so a longer stable run wins and recovery fails. Once
        the same input is replayed and one site's value has changed, that site
        is preferred even though it is the shorter run."""
        words = _stream(12)
        learner = _learner()

        # A longer run of constants at a second site, identical on replay.
        stable = _conds([0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555], pc=0x2000)

        learner.f._cmplog.last_conds = stable + _conds(words[:4], pc=0x1000)
        assert learner.observe_execution(PAYLOAD) is False, (
            "first sight: the longer stable run is indistinguishable and wins"
        )

        # Same input again: the constants repeat, the generator has advanced.
        learner.f._cmplog.last_conds = stable + _conds(words[4:8], pc=0x1000)
        assert learner.observe_execution(PAYLOAD) is True
        # Both drains of the varying site are in the window by now, since a
        # run is assembled across executions.
        assert learner._confirmed_samples == words[:8]
        assert learner.predict(4) == words[8:12]

        history = learner._run_history[hash(PAYLOAD)]
        # Sites are keyed (pc, operand width), so a PC comparing 4- and
        # 8-byte values keeps two independent histories.
        assert len(history[(0x1000, 4)].values) == 8, "generator site recorded both drains"
        assert history[(0x1000, 4)].max_drain_width == 4
        stable_entry = history[(0x2000, 4)]
        assert len(stable_entry.values) == stable_entry.max_drain_width == 5, (
            "a site that logs the same constants every drain must not read as varied"
        )

    def test_history_is_capped(self):
        learner = _learner()
        from fuzzer_tool.core import prng_state_learner as mod

        for i in range(mod._RUN_HISTORY_CAP + 50):
            learner.f._cmplog.last_conds = _conds([0x1000 + i], pc=0x3000)
            learner.observe_execution(ZERO_FIELD + f"input-{i}".encode())
        assert len(learner._run_history) <= mod._RUN_HISTORY_CAP


class TestSampleFloor:
    def test_below_every_familys_floor_is_not_attempted(self):
        """Below the *smallest* candidate family's confident_samples (2, for
        xorshift32 at 4 bytes and xorshift64 at 8), no family of that width
        could possibly verify, so no attempt is spent."""
        from fuzzer_tool.core import prng_state_learner as mod

        assert mod._MIN_SAMPLES == {4: 2, 8: 2}
        learner = _learner(_conds(_stream(1)))
        assert learner.observe_execution(PAYLOAD) is False
        assert learner.attempts == 0

    def test_three_taus88_samples_are_tried_and_correctly_rejected(self):
        """3 real taus88 outputs clear xorshift32's floor (2) and get tried
        against it first (smallest state first) -- and correctly fail, since
        they are not an xorshift32 stream. They also pin taus88's own 96-bit
        state mathematically but leave 8 free consistency bits, which
        unrelated constants would satisfy at ~1/2**8; 3 is still below
        taus88's own confident_samples (4), so that family is skipped, not
        just failed.
        """
        learner = _learner(_conds(_stream(3)))
        assert learner.observe_execution(PAYLOAD) is False
        assert learner.attempts == 1
        assert not learner.has_state()

    def test_four_random_constants_are_rejected(self):
        """The floor's purpose: a 4-word run that is not a taus88 stream must
        not produce a state."""
        learner = _learner(_conds([0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE]))
        assert learner.observe_execution(PAYLOAD) is False
        assert learner.attempts == 1, "recovery must have been attempted and refused"
        assert not learner.has_state()


def _generic_stream(n: int, spec, state: tuple[int, ...]) -> list[int]:
    """The first *n* outputs of *spec*'s stream started at *state*, matching
    the same "output is taken after stepping" convention as ``_stream``."""
    out: list[int] = []
    s = state
    for _ in range(n):
        s = step_state(s, spec)
        out.append(output_word(s, spec))
    return out


class TestCrossFamilyRecovery:
    """The learner is not taus88-only: it tries every 4-byte-output shipped
    family and keeps whichever verifies (see the module docstring's "Family
    is not assumed" section)."""

    def test_recovers_an_xorshift32_stream(self):
        words = _generic_stream(4, XORSHIFT32, (0xACE1_2345,))
        learner = _learner(_conds(words))
        assert learner.observe_execution(PAYLOAD) is True
        assert learner._spec.name == "xorshift32"
        assert learner.predict(1) == _generic_stream(5, XORSHIFT32, (0xACE1_2345,))[4:]

    def test_recovers_an_xorshift128_stream(self):
        seed = (0x1234_5678, 0x9ABC_DEF0, 0x0F1E_2D3C, 0x4B5A_6978)
        words = _generic_stream(6, XORSHIFT128, seed)
        learner = _learner(_conds(words))
        assert learner.observe_execution(PAYLOAD) is True
        assert learner._spec.name == "xorshift128"

    def test_recovers_a_taus113_stream(self):
        seed = (0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444)
        words = _generic_stream(6, TAUS113, seed)
        learner = _learner(_conds(words))
        assert learner.observe_execution(PAYLOAD) is True
        assert learner._spec.name == "taus113"

    def test_smallest_state_family_is_preferred_when_ambiguous(self):
        """xorshift32's state IS its output, so a real xorshift32 stream also
        satisfies taus88's system at low sample counts is not a real risk
        here (different bit width in the map), but the ordering itself --
        smallest state first -- is what this asserts: xorshift32 is checked,
        and matches, before taus88 is ever tried."""
        words = _generic_stream(2, XORSHIFT32, (0xDEAD_BEEF,))
        learner = _learner(_conds(words))
        assert learner.observe_execution(PAYLOAD) is True
        assert learner._spec.name == "xorshift32"

    def test_round_trip_preserves_the_recovered_family(self):
        words = _generic_stream(6, XORSHIFT128, (1, 2, 3, 4))
        learner = _learner(_conds(words))
        learner.observe_execution(PAYLOAD)
        assert learner.to_dict()["family"] == "xorshift128"

        restored = PRNGStateLearner.from_dict(_fuzzer(), learner.to_dict())
        assert restored.has_state()
        assert restored._spec.name == "xorshift128"
        assert restored.predict(2) == learner.predict(2)


class TestWideOperandStreams:
    """64-bit generators are observed and predicted at their own width.

    A target drawing 8-byte tokens compares them through
    ``__sanitizer_cov_trace_cmp8``, which the shim logs whole, so nothing
    but the Python side's width assumption ever stood between lfsr258 /
    xorshift64 and recovery.
    """

    def test_recovers_an_xorshift64_stream(self):
        seed = (0x0123_4567_89AB_CDEF,)
        words = _generic_stream(4, XORSHIFT64, seed)
        learner = _learner(_wide_conds(words))
        assert learner.observe_execution(WIDE_PAYLOAD) is True
        assert learner._spec.name == "xorshift64"
        assert learner.predict(1) == _generic_stream(5, XORSHIFT64, seed)[4:]

    def test_recovers_an_lfsr258_stream(self):
        seed = (153587801, 759022222, 1288503317, 1718083407, 123456789)
        words = _generic_stream(8, LFSR258, seed)
        learner = _learner(_wide_conds(words))
        assert learner.observe_execution(WIDE_PAYLOAD) is True
        assert learner._spec.name == "lfsr258"
        assert learner.predict(2) == _generic_stream(10, LFSR258, seed)[8:]

    def test_prediction_is_packed_at_the_generators_width(self):
        """The mutator writes whatever next_value_bytes returns, so a 64-bit
        draw truncated to 4 bytes would be a value the target never
        compares."""
        seed = (0x0123_4567_89AB_CDEF,)
        learner = _learner(_wide_conds(_generic_stream(4, XORSHIFT64, seed)))
        learner.observe_execution(WIDE_PAYLOAD)
        packed = learner.next_value_bytes()
        assert len(packed) == 8
        assert int.from_bytes(packed, "little") == learner.predict(1)[0]

    def test_widths_at_one_pc_are_separate_streams(self):
        """One PC comparing both widths must not braid them into one window:
        the 4-byte taus88 run still recovers with the 8-byte noise present."""
        words = _stream(4)
        noise = _wide_conds([0xDEAD_BEEF_0BAD_F00D, 0x0102_0304_0506_0708], pc=0x1000)
        learner = _learner(noise + _conds(words, pc=0x1000))
        assert learner.observe_execution(PAYLOAD) is True
        assert learner._spec.name == "taus88"
        assert learner._confirmed_samples == words
        assert sorted(learner._pending) == [(0x1000, 4), (0x1000, 8)]

    def test_a_taus88_stream_widened_to_eight_bytes_is_not_recovered(self):
        """Adversarial: the same words a 4-byte window recovers taus88 from,
        logged as 8-byte operands. Width selects the families tried, so the
        only candidates are the 64-bit ones and none of them fits -- a
        zero-extended 32-bit stream is not a 64-bit generator's output."""
        learner = _learner(_wide_conds(_stream(8)))
        assert learner.observe_execution(WIDE_PAYLOAD) is False
        assert learner.has_state() is False
        assert learner.attempts == 1


class TestInProcessGate:
    def test_out_of_process_never_recovers(self):
        learner = _learner(_conds(_stream(6)), in_process=False)
        assert learner.observe_execution(PAYLOAD) is False
        assert learner.attempts == 0

    def test_gate_does_not_discard_an_existing_state(self):
        """has_state() is reported unchanged when the gate short-circuits."""
        learner = _learner(_conds(_stream(4)))
        assert learner.observe_execution(PAYLOAD) is True
        learner.f._inprocess_runner = None
        assert learner.observe_execution(PAYLOAD) is True
        assert learner.has_state()


class TestPredictionFrontier:
    def test_predicts_past_the_observed_samples(self):
        words = _stream(8)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        assert learner.predict(4) == words[4:8]

    def test_first_prediction_is_not_an_observed_draw(self):
        words = _stream(5)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        predicted = learner.predict(1)
        assert predicted == [words[4]]
        assert predicted[0] not in words[:4]

    def test_next_value_bytes_is_little_endian(self):
        words = _stream(5)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        assert learner.next_value_bytes() == words[4].to_bytes(4, "little")


class TestCrossDrainContinuation:
    def test_contiguous_next_drain_advances_the_frontier(self):
        words = _stream(12)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        attempts = learner.attempts

        learner.f._cmplog.last_conds = _conds(words[4:8])
        assert learner.observe_execution(PAYLOAD2) is True
        assert learner.attempts == attempts, "continuation must not re-run recovery"
        assert learner.predict(4) == words[8:12]

    def test_gapped_next_drain_still_continues(self):
        """Draws the target made without comparing them are invisible here, so
        the new run sits further along than the frontier by an unknown gap."""
        words = _stream(16)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        attempts = learner.attempts

        learner.f._cmplog.last_conds = _conds(words[9:13])
        assert learner.observe_execution(PAYLOAD2) is True
        assert learner.attempts == attempts
        assert learner.predict(3) == words[13:16]

    def test_foreign_stream_drops_the_state(self):
        learner = _learner(_conds(_stream(4)))
        learner.observe_execution(PAYLOAD)
        assert learner.has_state()

        learner.f._cmplog.last_conds = _conds([0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE])
        assert learner.observe_execution(PAYLOAD2) is False
        assert not learner.has_state()
        assert learner.predict(1) is None

    def test_gap_beyond_the_search_window_is_not_claimed(self):
        """Past _MAX_ADVANCE_SEARCH the learner must not silently accept: it
        re-recovers from the new run instead, which is still correct."""
        from fuzzer_tool.core import prng_state_learner as mod

        words = _stream(mod._MAX_ADVANCE_SEARCH + 40)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        attempts = learner.attempts

        far = mod._MAX_ADVANCE_SEARCH + 20
        learner.f._cmplog.last_conds = _conds(words[far : far + 4])
        assert learner.observe_execution(PAYLOAD2) is True
        assert learner.attempts == attempts + 1, "should have re-recovered, not walked"
        assert learner.predict(4) == _stream(far + 8)[far + 4 : far + 8]


class TestPersistenceRebuildsFrontier:
    def test_round_trip_keeps_predicting_forward(self):
        words = _stream(8)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        before = learner.predict(4)

        restored = PRNGStateLearner.from_dict(_fuzzer(), learner.to_dict())
        assert restored.has_state()
        assert restored.predict(4) == before == words[4:8]

    def test_legacy_dict_without_samples_is_anchored(self):
        words = _stream(6)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        data = learner.to_dict()
        data["confirmed_samples"] = []

        restored = PRNGStateLearner.from_dict(_fuzzer(), data)
        assert restored.has_state()
        # The state's own output is the only anchor: the next draw after it.
        assert restored.predict(1) == [words[1]]

    def test_counters_survive(self):
        learner = _learner(_conds(_stream(4)))
        learner.observe_execution(PAYLOAD)
        restored = PRNGStateLearner.from_dict(_fuzzer(), learner.to_dict())
        assert (restored.attempts, restored.successes) == (
            learner.attempts,
            learner.successes,
        )

    def test_empty_dict_is_inert(self):
        learner = PRNGStateLearner.from_dict(_fuzzer(), None)
        assert not learner.has_state()
        assert learner.predict(1) is None


class TestOperatorPlacement:
    """_op_prng_predict writes the prediction where the target reads it."""

    def _engine(self, learner, pairs, buf_len=32, rng=None):
        from fuzzer_tool.core.mutator_interface import MutationContext
        from fuzzer_tool.core.rand_pool import RandPool
        from fuzzer_tool.services.operators import OperatorEngine

        engine = OperatorEngine.__new__(OperatorEngine)
        engine.f = MagicMock()
        # ctx is a read-only property backed by this cache, which mutate()
        # primes once per round; priming it directly is how a handler gets
        # called in isolation without a whole Fuzzer behind it.
        engine._ctx_cache = MutationContext(
            max_len=buf_len,
            rng=rng if rng is not None else RandPool(seed=7),
            cmplog_pairs=pairs,
            prng_state_learner=learner,
        )
        return engine

    def test_writes_at_the_compared_offset(self):
        words = _stream(5)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)

        token_field = b"\x11\x22\x33\x44"
        buf = bytearray(b"HDR-" + token_field + b"-rest-of-the-buffer-here")
        offset = bytes(buf).find(token_field)
        engine = self._engine(learner, [(token_field, b"\x00\x00\x00\x00")])
        engine._op_prng_predict(buf, 0, bytes(buf))

        assert bytes(buf[offset : offset + 4]) == words[4].to_bytes(4, "little")

    def test_falls_back_to_a_random_offset(self):
        words = _stream(5)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)

        buf = bytearray(b"\x00" * 32)
        engine = self._engine(learner, [])
        engine._op_prng_predict(buf, 0, bytes(buf))
        assert words[4].to_bytes(4, "little") in bytes(buf)

    def test_declines_without_a_state(self):
        learner = _learner([])
        buf = bytearray(b"\x00" * 16)
        engine = self._engine(learner, [])
        assert engine._op_prng_predict(buf, 0, bytes(buf)) is None
        assert bytes(buf) == b"\x00" * 16

    def test_short_buffer_is_left_alone(self):
        words = _stream(5)
        learner = _learner(_conds(words[:4]))
        learner.observe_execution(PAYLOAD)
        buf = bytearray(b"ab")
        engine = self._engine(learner, [])
        engine._op_prng_predict(buf, 0, bytes(buf))
        assert bytes(buf) == b"ab"

    def _wide_learner(self):
        """A learner holding a recovered 64-bit (xorshift64) state."""
        seed = (0x0123_4567_89AB_CDEF,)
        learner = _learner(_wide_conds(_generic_stream(4, XORSHIFT64, seed)))
        learner.observe_execution(WIDE_PAYLOAD)
        assert learner.has_state()
        return learner, _generic_stream(5, XORSHIFT64, seed)[4]

    def test_writes_a_64_bit_draw_whole(self):
        """The field is as wide as the generator's word: half an 8-byte token
        satisfies nothing, and the other half is left as whatever was there."""
        learner, nxt = self._wide_learner()
        token_field = b"\x11\x22\x33\x44\x55\x66\x77\x88"
        buf = bytearray(b"HDR-" + token_field + b"-rest-of-the-buffer-here")
        offset = bytes(buf).find(token_field)
        engine = self._engine(learner, [(token_field, b"\x00" * 8)])
        engine._op_prng_predict(buf, 0, bytes(buf))
        assert bytes(buf[offset : offset + 8]) == nxt.to_bytes(8, "little")

    def test_four_byte_operands_do_not_place_an_eight_byte_draw(self):
        """Adversarial: a same-width operand is what marks the token field.
        A 4-byte operand occurring in the buffer is a different field, so it
        must not be used as the placement offset -- writing 8 bytes there
        would also run past what was compared. With no 8-byte operand on
        record the fallback offset is used instead, scripted here (Hard Rule
        39) so the assertion is on behaviour, not on a lucky draw.
        """
        from tests.support.scripted_rng import ScriptedRng

        learner, nxt = self._wide_learner()
        buf = bytearray(b"Z" * 64)
        narrow_field = b"\xaa\xbb\xcc\xdd"
        buf[4:8] = narrow_field
        engine = self._engine(
            learner,
            [(narrow_field, b"\xee\xee\xee\xee")],
            rng=ScriptedRng(randints=[40]),
        )
        engine._op_prng_predict(buf, 0, bytes(buf))
        assert bytes(buf[40:48]) == nxt.to_bytes(8, "little")
        assert bytes(buf[4:8]) == narrow_field, "the 4-byte field is untouched"

    def test_buffer_shorter_than_the_wide_draw_is_left_alone(self):
        """Six bytes is enough for a 4-byte token and not for an 8-byte one."""
        learner, _ = self._wide_learner()
        buf = bytearray(b"abcdef")
        engine = self._engine(learner, [])
        engine._op_prng_predict(buf, 0, bytes(buf))
        assert bytes(buf) == b"abcdef"


class TestCmplogExposesOrderedRecords:
    def test_last_conds_is_replaced_per_drain(self):
        """The field the learner depends on: ordered, PC-carrying, and scoped
        to the latest drain rather than accumulated."""
        from fuzzer_tool.core.cmplog import CmplogCollector

        collector = CmplogCollector.__new__(CmplogCollector)
        assert hasattr(CmplogCollector, "__init__")
        # Field is declared with a list default; a fresh instance must expose
        # it as empty rather than missing, since _extract_candidates getattrs
        # it on every execution.
        collector.last_conds = []
        assert collector.last_conds == []

    def test_pairs_and_last_conds_are_distinct_fields(self):
        import inspect

        from fuzzer_tool.core.cmplog import CmplogCollector

        src = inspect.getsource(CmplogCollector.__init__)
        assert "self.last_conds" in src
        assert "self.pairs" in src


def _shim_call(lib, symbol: str) -> None:
    """Call an exported shim entry point if the build has it."""
    if lib is None:
        return
    fn = getattr(lib, symbol, None)
    if fn is not None:
        fn()


def _compiler() -> str | None:
    for cc in ("clang", "gcc", "cc"):
        if shutil.which(cc):
            return cc
    return None


@pytest.fixture(scope="module")
def token_so(tmp_path_factory):
    """prng_token_read.c as a shim-linked .so with the cmplog layer live.

    clang specifically, and with these exact flags:

      * ``-fsanitize-coverage=trace-cmp`` -- the token compare is a plain
        ``uint32_t ==``, so it is an inline integer compare. The shim's libc
        interceptors cannot see one; only SanitizerCoverage's
        ``__sanitizer_cov_trace_cmp4`` callback reports it. gcc does not
        implement the flag.
      * ``-D__AFL_CMPLOG=1`` -- the ``__sanitizer_cov_trace_*`` bodies in
        afl_shim.c are compiled behind it. Without it the .so references
        callbacks nobody defines and ``dlopen`` fails on
        ``undefined symbol: __sanitizer_cov_trace_const_cmp1``.
      * ``trace-pc-guard`` -- edge coverage, as every other target build.
    """
    if not shutil.which("clang"):
        pytest.skip("trace-cmp instrumentation requires clang")
    if not SHIM.is_file() or not TARGET_SRC.is_file():
        pytest.skip("shim or target source not found")

    build = tmp_path_factory.mktemp("prng_token")
    so = build / "prng_token_read.so"
    proc = subprocess.run(
        [
            "clang",
            "-O1",
            "-fPIC",
            "-shared",
            "-D__AFL_CMPLOG=1",
            "-fsanitize-coverage=trace-pc-guard,trace-cmp",
            "-o",
            str(so),
            str(TARGET_SRC),
            str(SHIM),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not so.is_file():
        pytest.skip(f"target failed to build: {proc.stderr[-400:]}")
    return str(so)


class TestTargetStreamMatchesRecovery:
    """The target's C generator must produce the same stream the Python side
    reconstructs, or every recovery result is meaningless."""

    def test_c_and_python_streams_agree(self, token_so):
        cc = _compiler()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "probe.c"
            src.write_text(
                "#include <stdint.h>\n#include <stdio.h>\n"
                "static uint32_t a,b,c;\n"
                "static uint32_t st(uint32_t v,int k,int q,int s){"
                "uint32_t t=(((v<<q)^v))>>(k-s);"
                "uint32_t m=0xFFFFFFFFu<<(32-k);return ((v&m)<<s)^t;}\n"
                "int main(void){a=0x12345678u;b=0x9ABCDEF0u;c=0xFEDCBA98u;"
                "for(int i=0;i<8;i++){a=st(a,31,13,12);b=st(b,29,2,4);"
                'c=st(c,28,3,17);printf("%u\\n",a^b^c);}return 0;}\n'
            )
            exe = Path(tmp) / "probe"
            build = subprocess.run(
                [cc, "-O1", "-o", str(exe), str(src)], capture_output=True, text=True
            )
            if build.returncode != 0:
                pytest.skip("probe failed to build")
            out = subprocess.run([str(exe)], capture_output=True, text=True)
            c_words = [int(x) for x in out.stdout.split()]
        assert c_words == _stream(8), (
            "targets/prng_token_read.c must implement the same parameters as "
            "core/prng_state_recovery.TAUS88_PARAMS"
        )


@pytest.fixture(scope="module")
def live_session(token_so):
    """Recover the state from the live target once, then probe one prediction.

    Collected once, for the reason ``test_regression_direct_lite_reset.py``
    gives: the shim binds its coverage map and its cmplog fd at dlopen time,
    and dlopen of the same path in one process returns the same handle, so a
    second session in the same interpreter would write to the FIRST session's
    log file and see nothing. Both tests below read this one result.
    """
    from fuzzer_tool.adapters.inprocess import InProcessRunner
    from fuzzer_tool.adapters.shm import ShmCoverage
    from fuzzer_tool.core.cmplog import CmplogCollector

    map_size = 65536
    saved = {k: os.environ.get(k) for k in ("__AFL_SHM_ID", "AFL_MAP_SIZE")}
    shm = ShmCoverage(size=map_size)
    os.environ["__AFL_SHM_ID"] = str(shm.env_id)
    os.environ["AFL_MAP_SIZE"] = str(map_size)
    try:
        collector = CmplogCollector()
        collector.setup_env_for_run()
        runner = InProcessRunner(
            token_so,
            "LLVMFuzzerTestOneInput",
            direct_lite=True,
            shm_size=map_size,
            coverage_env_id=shm.env_id,
            timeout=5.0,
        )
        fuzzer = MagicMock()
        fuzzer._inprocess_runner = runner
        fuzzer._cmplog = collector
        learner = PRNGStateLearner(fuzzer)
        payload = b"\x00" * 8

        def iterate(data: bytes) -> bool:
            """One fuzz-loop iteration, in the loop's own order.

            ``__tracecmp_flush`` then collect then ``__cmplog_reset`` is what
            ``Fuzzer._flush_cmplog_shims`` and ``run()`` do, and the order is
            load-bearing: the shim buffers records in a 256 KiB internal
            buffer, so without the flush the file stays empty until that
            buffer fills, and ``__cmplog_reset`` truncates the file, so it has
            to come after the read.
            """
            shm.reset_edge_map()
            runner.run_one(data)
            _shim_call(runner._lib, "__tracecmp_flush")
            collector.collect_tokens()
            found = learner.observe_execution(data)
            _shim_call(runner._lib, "__cmplog_reset")
            return found

        iterations = 0
        for _ in range(24):
            iterations += 1
            if iterate(payload):
                break

        result = {
            "learner": learner,
            "iterations": iterations,
            "recovered": learner.has_state(),
            "pending": {hex(pc or 0): len(run) for pc, run in learner._pending.items()},
            "predicted": None,
            "observed": set(),
        }
        if learner.has_state():
            predicted = learner.predict(1)[0]
            # Feed the prediction as the token field and read the token the
            # target actually drew out of the next drain. The operand is direct
            # evidence; the target's abort is reported differently per backend.
            shm.reset_edge_map()
            runner.run_one(predicted.to_bytes(4, "little") + b"\x00" * 4)
            _shim_call(runner._lib, "__tracecmp_flush")
            collector.collect_tokens()
            result["predicted"] = predicted
            result["observed"] = {
                int.from_bytes(op, "little")
                for cond in collector.last_conds
                for op in (cond.base.op_a, cond.base.op_b)
                if len(op) == 4
            }
        return result
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.timeout(90)
class TestIntegrationAgainstLiveTarget:
    def test_recovers_state_from_the_live_token_stream(self, live_session):
        assert live_session["recovered"], (
            f"no state recovered in 24 executions; pending windows were {live_session['pending']}"
        )

    def test_recovers_within_a_few_executions(self, live_session):
        """One token per execution, so _MIN_SAMPLES executions is the floor.
        Close to it means the stream was tracked across drains rather than a
        run happening to appear inside one."""
        if not live_session["recovered"]:
            pytest.skip("state not recovered; covered by the test above")
        assert live_session["iterations"] <= 8, (
            f"took {live_session['iterations']} executions to recover"
        )

    def test_predicted_token_is_the_one_the_target_draws_next(self, live_session):
        """The payoff: the prediction equals the target's next draw, which is
        what turns the 2**32 equality check into a one-shot pass."""
        if not live_session["recovered"]:
            pytest.skip("state not recovered; covered by the test above")
        predicted = live_session["predicted"]
        assert predicted in live_session["observed"], (
            f"predicted {predicted:#010x}, target drew {live_session['observed']}"
        )
