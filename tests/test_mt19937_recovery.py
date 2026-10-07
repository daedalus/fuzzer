"""Tests for core/mt19937_recovery.py — full-word untemper recovery."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.mt19937_recovery import (
    MT19937,
    MT19937_SPEC,
    confident_samples,
    min_samples,
    output_word,
    predict_words,
    recover_state,
    step_state,
    untemper,
    verify_state,
    walk_stream,
)


def _stream_from_seed(seed: int, n: int) -> tuple[tuple[int, ...], list[int]]:
    """Produce n tempered outputs from a standard-seeded MT, plus origin state."""
    gen = MT19937()
    gen.seed(seed)
    # Align so state_tuple's next output is the first word we record.
    if gen.index >= 624:
        from fuzzer_tool.core.mt19937_recovery import _twist

        _twist(gen.mt)
        gen.index = 0
    state = gen.state_tuple()
    words = [gen.random_uint32() for _ in range(n)]
    return state, words


class TestUntemper:
    def test_roundtrip(self):
        for x in (0, 1, 0xFFFFFFFF, 0x12345678, 0x80000000, 0x7FFFFFFF):
            # temper then untemper must recover x
            from fuzzer_tool.core.mt19937_recovery import _temper

            assert untemper(_temper(x)) == x

    def test_many_random(self):
        from fuzzer_tool.core.mt19937_recovery import _temper

        rng = random.Random(0)
        for _ in range(200):
            x = rng.getrandbits(32)
            assert untemper(_temper(x)) == x


class TestRecover:
    def test_min_samples_is_624(self):
        assert min_samples() == 624
        assert confident_samples() == 625

    def test_raises_below_minimum(self):
        with pytest.raises(ValueError, match="624"):
            recover_state([0] * 100)

    @pytest.mark.parametrize("seed", [0, 1, 42, 0xC0FFEE, 0xDEADBEEF])
    def test_recovers_and_predicts(self, seed: int):
        _, words = _stream_from_seed(seed, 624 + 21)
        recovered = recover_state(words[:624])
        assert recovered is not None
        assert verify_state(recovered, words[:624])
        # Predictions after the recovered origin must match the live stream.
        # recovered is aligned so output_word(recovered) == words[0];
        # after walking 624 steps the state's own output is words[624], and
        # predict_words gives the draws AFTER it -- the convention every
        # other recovery module uses and the learner's predict() relies on.
        gen_state = recovered
        for w in words[:624]:
            assert output_word(gen_state) == w
            gen_state = step_state(gen_state)
        assert output_word(gen_state) == words[624]
        predicted = predict_words(gen_state, 20)
        assert predicted == words[625:645]

    def test_extra_words_reject_inconsistent(self):
        _, words = _stream_from_seed(99, 630)
        bad = list(words[:625])
        bad[624] ^= 0xFFFFFFFF  # corrupt the consistency word
        assert recover_state(bad) is None

    def test_cpython_random_compatibility(self):
        """Recover from CPython random.Random's getrandbits(32) stream."""
        rng = random.Random(12345)
        words = [rng.getrandbits(32) for _ in range(624 + 6)]
        recovered = recover_state(words[:624])
        assert recovered is not None
        assert verify_state(recovered, words[:624])
        # Future predictions must match CPython's own stream.
        gen_state = recovered
        for w in words[:624]:
            assert output_word(gen_state) == w
            gen_state = step_state(gen_state)
        assert output_word(gen_state) == words[624]
        assert predict_words(gen_state, 5) == words[625:630]


class TestWalkStream:
    def test_yields_state_word_pairs_like_the_other_families(self):
        _, words = _stream_from_seed(5, 640)
        state = recover_state(words[:625])
        walked = list(walk_stream(state, 6))
        assert [w for _, w in walked] == words[1:7]
        s = state
        for snap, word in walked:
            s = step_state(s)
            assert tuple(snap) == tuple(s)
            assert output_word(snap) == word

    def test_predict_words_matches_walk_stream(self):
        _, words = _stream_from_seed(6, 640)
        state = recover_state(words[:625])
        assert predict_words(state, 9) == [w for _, w in walk_stream(state, 9)]

