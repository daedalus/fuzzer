"""The generalized side of ``core/prng_state_recovery.py``.

``tests/test_prng_state_recovery.py`` covers the taus88-shaped façade this
module started as. This file covers what the generalization added: the
shipped family constants, the compiled concrete evaluator, the derived
sample counts, and the LFSR builders that have no shipped instance.

Every family here is checked against the published source it was read
from, transcribed as an expression rather than as parameters, because a
``(w, k, q, s)`` tuple copied wrong is invisible to any test written in
terms of the same tuple:

* taus88 -- GSL ``taus2`` / Boost ``taus88`` (L'Ecuyer 1996).
* taus113 -- Linux ``prandom_u32``'s ``TAUSWORTHE`` macro / GSL
  ``taus113`` (L'Ecuyer 1999 erratum, a.k.a. LFSR113).
* lfsr258 -- L'Ecuyer 1999's 64-bit generator, per the published C
  translation.
* xorshift32/64/128 -- Marsaglia 2003 "Xorshift RNGs", pp. 4-5.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from fuzzer_tool.core.prng_state_recovery import (
    FAMILIES,
    LFSR258,
    TAUS88,
    TAUS113,
    XORSHIFT32,
    XORSHIFT64,
    XORSHIFT128,
    LinearPRNG,
    Opcode,
    _sym_out_vector,
    _sym_step_vectors,
    confident_samples,
    fib_lfsr,
    galois_lfsr,
    min_samples,
    output_word,
    predict_next,
    predict_words,
    recover_state,
    spec_from_params,
    step_state,
    structural_equations,
    verify_state,
    walk_stream,
)

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


# ── Published references, transcribed ────────────────────────────────


def ref_taus88(s):
    z1, z2, z3 = s
    z1 = (((z1 & 4294967294) << 12) & M32) ^ ((((z1 << 13) & M32) ^ z1) >> 19)
    z2 = (((z2 & 4294967288) << 4) & M32) ^ ((((z2 << 2) & M32) ^ z2) >> 25)
    z3 = (((z3 & 4294967280) << 17) & M32) ^ ((((z3 << 3) & M32) ^ z3) >> 11)
    return (z1, z2, z3)


def _tausworthe(s, a, b, c, d):
    """Linux ``random32.c``: ``((s & c) << d) ^ (((s << a) ^ s) >> b)``."""
    return (((s & c) << d) & M32) ^ ((((s << a) & M32) ^ s) >> b)


def ref_taus113(s):
    z1, z2, z3, z4 = s
    return (
        _tausworthe(z1, 6, 13, 4294967294, 18),
        _tausworthe(z2, 2, 27, 4294967288, 2),
        _tausworthe(z3, 13, 21, 4294967280, 7),
        _tausworthe(z4, 3, 12, 4294967168, 13),
    )


def ref_lfsr258(s):
    s1, s2, s3, s4, s5 = s
    b = ((((s1 << 1) & M64) ^ s1) >> 53) & M64
    s1 = ((((s1 & (M64 - 1)) << 10) & M64) ^ b) & M64
    b = ((((s2 << 24) & M64) ^ s2) >> 50) & M64
    s2 = ((((s2 & (M64 - 511)) << 5) & M64) ^ b) & M64
    b = ((((s3 << 3) & M64) ^ s3) >> 23) & M64
    s3 = ((((s3 & (M64 - 4095)) << 29) & M64) ^ b) & M64
    b = ((((s4 << 5) & M64) ^ s4) >> 24) & M64
    s4 = ((((s4 & (M64 - 131071)) << 23) & M64) ^ b) & M64
    b = ((((s5 << 3) & M64) ^ s5) >> 33) & M64
    s5 = ((((s5 & (M64 - 8388607)) << 8) & M64) ^ b) & M64
    return (s1, s2, s3, s4, s5)


def ref_xorshift32(s):
    x = s[0]
    x ^= (x << 13) & M32
    x ^= x >> 17
    x ^= (x << 5) & M32
    return (x,)


def ref_xorshift64(s):
    x = s[0]
    x ^= (x << 13) & M64
    x ^= x >> 7
    x ^= (x << 17) & M64
    return (x,)


def ref_xorshift128(s):
    x, y, z, w = s
    t = x ^ ((x << 11) & M32)
    x, y, z = y, z, w
    w = w ^ (w >> 19) ^ t ^ (t >> 8)
    return (x, y, z, w)


#: (spec, reference step, a seed, the combined output of that state)
REFERENCES = [
    (TAUS88, ref_taus88, (0x12345678, 0x9ABCDEF0, 0xFEDCBA98)),
    (TAUS113, ref_taus113, (0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444)),
    (
        LFSR258,
        ref_lfsr258,
        (153587801, 759022222, 1288503317, 1718083407, 123456789),
    ),
    (XORSHIFT32, ref_xorshift32, (0xACE1_2345,)),
    (XORSHIFT64, ref_xorshift64, (0x0123_4567_89AB_CDEF,)),
    (
        XORSHIFT128,
        ref_xorshift128,
        (123456789, 362436069, 521288629, 88675123),
    ),
]

_IDS = [spec.name for spec, _, _ in REFERENCES]


def _seed_for(spec: LinearPRNG, rng: random.Random) -> tuple[int, ...]:
    """A non-degenerate seed: an all-zero LFSR word is a fixed point."""
    return tuple(rng.getrandbits(w) | 1 for w in spec.widths[: spec.n_slots])


def _off_by_one(spec: LinearPRNG) -> LinearPRNG:
    """*spec* with the first shift in its step decremented by one."""
    code = list(spec.step)
    for i, ins in enumerate(code):
        if ins.op in (Opcode.SHL, Opcode.SHR) and ins.b > 1:
            code[i] = replace(ins, b=ins.b - 1)
            return replace(spec, name=f"{spec.name}-wrong", step=tuple(code))
    raise AssertionError(f"{spec.name}: no shift to perturb")


@pytest.mark.parametrize(("spec", "reference", "seed"), REFERENCES, ids=_IDS)
class TestPublishedFamilies:
    def test_step_matches_the_published_expressions(self, spec, reference, seed):
        ours, theirs = seed, seed
        for _ in range(32):
            ours, theirs = step_state(ours, spec), reference(theirs)
            assert ours == theirs

    def test_a_wrong_constant_is_detected(self, spec, reference, seed):
        """Control for the oracle (Hard Rule 46): the comparison above has to
        be able to fail. One shift count off by one -- the exact way a
        transcribed parameter goes wrong -- must diverge from the reference
        within a few steps, otherwise agreement says nothing.
        """
        wrong = _off_by_one(spec)
        ours, theirs = seed, seed
        for _ in range(8):
            ours, theirs = step_state(ours, wrong), reference(theirs)
            if ours != theirs:
                return
        raise AssertionError(f"{spec.name}: a perturbed step still matched")


@pytest.mark.parametrize(("spec", "reference", "seed"), REFERENCES, ids=_IDS)
def test_compiled_step_equals_its_coefficients(spec, reference, seed):
    """The concrete path is a projection of the symbolic one, not a rewrite.

    ``step_state`` applies grouped (mask, shift) terms; the definition is a
    parity per output bit over that bit's coefficient mask. They must agree
    on every state, or forward simulation and recovery are solving different
    generators.
    """
    del reference
    rng = random.Random(4242)
    for _ in range(16):
        state = _seed_for(spec, rng)
        assert step_state(state, spec) == _by_coefficients(state, spec)
        assert output_word(state, spec) == _out_by_coefficients(state, spec)


def _parity(mask: int, bits: int) -> int:
    return bin(mask & bits).count("1") & 1


def _flat(state, spec) -> int:
    """State words packed into one bit vector, the recovery bit order."""
    bits, offset = 0, 0
    for word, width in zip(state, spec.widths[: spec.n_slots], strict=True):
        bits |= word << offset
        offset += width
    return bits


def _by_coefficients(state, spec) -> tuple[int, ...]:
    flat = _flat(state, spec)
    return tuple(
        sum(_parity(coeff, flat) << i for i, coeff in enumerate(vector))
        for vector in _sym_step_vectors(spec)
    )


def _out_by_coefficients(state, spec) -> int:
    flat = _flat(state, spec)
    return sum(_parity(coeff, flat) << i for i, coeff in enumerate(_sym_out_vector(spec)))


@pytest.mark.parametrize(("spec", "reference", "seed"), REFERENCES, ids=_IDS)
def test_recovery_round_trip(spec, reference, seed):
    del reference
    rng = random.Random(99)
    for _ in range(4):
        # One step first, so the state is in the step's image -- which is
        # what structural_equations assumes of anything it is asked to
        # recover, and true of any state a running generator is ever in.
        state = step_state(_seed_for(spec, rng), spec)
        words = [output_word(state, spec)] + predict_words(state, confident_samples(spec) + 3, spec)
        recovered = recover_state(words[: confident_samples(spec)], spec)
        assert recovered is not None
        assert verify_state(recovered, words, spec), "must predict unused words too"


@pytest.mark.parametrize(("spec", "reference", "seed"), REFERENCES, ids=_IDS)
def test_another_familys_stream_does_not_verify(spec, reference, seed):
    """Adversarial: recovery is only trustworthy because a wrong family
    fails. Feed each family a same-width sibling's stream and require that
    it either cannot solve it or produces a state that fails verification.
    """
    del reference
    rng = random.Random(7)
    for other, _, _ in REFERENCES:
        if other is spec or other.out_bits != spec.out_bits:
            continue
        source = step_state(_seed_for(other, rng), other)
        words = [output_word(source, other)] + predict_words(
            source, confident_samples(spec) + 2, other
        )
        recovered = recover_state(words, spec)
        assert recovered is None or not verify_state(recovered, words, spec), (
            f"{spec.name} accepted a {other.name} stream"
        )


class TestDerivedSampleCounts:
    """How many draws a family needs is measured from its own equations."""

    def test_measured_counts(self):
        assert {name: min_samples(spec) for name, spec in FAMILIES.items()} == {
            "xorshift32": 1,
            "xorshift64": 1,
            "taus88": 3,
            "taus113": 4,
            "xorshift128": 4,
            "lfsr258": 5,
        }

    def test_confident_adds_one_whole_word(self):
        for spec in FAMILIES.values():
            assert confident_samples(spec) == min_samples(spec) + 1

    def test_min_samples_is_the_floor_not_a_guess(self):
        """One word below the derived count, the system is not determined --
        so the count is tight, not merely sufficient."""
        rng = random.Random(3)
        for spec in FAMILIES.values():
            needed = min_samples(spec)
            if needed == 1:
                continue
            state = step_state(_seed_for(spec, rng), spec)
            words = [output_word(state, spec)] + predict_words(state, needed - 2, spec)
            with pytest.raises(ValueError):
                recover_state(words, spec)


class TestStructuralEquations:
    def test_combined_lfsr_gap_is_the_unused_word_bits(self):
        """Each ``linear_feedback_shift_engine`` component wastes ``w - k``
        bits of its word, and those bits are linear in the rest -- so the
        free equations are exactly that sum."""
        expected = {
            TAUS88: (32 - 31) + (32 - 29) + (32 - 28),
            TAUS113: (32 - 31) + (32 - 29) + (32 - 28) + (32 - 25),
            LFSR258: (64 - 63) + (64 - 55) + (64 - 52) + (64 - 47) + (64 - 41),
        }
        for spec, gap in expected.items():
            assert len(structural_equations(spec)) == gap

    def test_a_bijective_step_yields_none(self):
        """xorshift steps are invertible, so there is no image to exploit and
        nothing is claimed for free."""
        for spec in (XORSHIFT32, XORSHIFT64, XORSHIFT128):
            assert structural_equations(spec) == ()

    def test_equations_hold_on_real_states(self):
        rng = random.Random(11)
        for spec in FAMILIES.values():
            state = step_state(_seed_for(spec, rng), spec)
            flat = _flat(state, spec)
            for row in structural_equations(spec):
                assert _parity(row, flat) == 0


class TestBuilders:
    """The LFSR shapes with no shipped instance: a polynomial is a target's
    own, so these are built by the caller rather than looked up by name."""

    def test_galois_matches_a_hand_written_step(self):
        poly = 0xB400  # the usual 16-bit CRC-ish taps
        spec = galois_lfsr("galois16", 16, poly)
        ours = theirs = (0xACE1,)
        for _ in range(64):
            (x,) = theirs
            theirs = ((x >> 1) ^ (poly if x & 1 else 0),)
            ours = step_state(ours, spec)
            assert ours == theirs

    def test_fibonacci_matches_a_hand_written_step(self):
        taps = (0, 2, 3, 5)
        spec = fib_lfsr("fib16", 16, taps)
        ours = theirs = (0xACE1,)
        for _ in range(64):
            (x,) = theirs
            feedback = 0
            for tap in taps:
                feedback ^= (x >> tap) & 1
            theirs = ((x >> 1) | (feedback << 15),)
            ours = step_state(ours, spec)
            assert ours == theirs

    def test_a_galois_lfsr_state_is_recoverable(self):
        """The point of the builders: an unshipped generator goes through the
        same recovery with no special casing."""
        spec = galois_lfsr("galois32", 32, 0xEDB88320)
        state = step_state((0x1234_5678,), spec)
        words = [output_word(state, spec)] + predict_words(state, 6, spec)
        recovered = recover_state(words[: confident_samples(spec)], spec)
        assert recovered is not None
        assert verify_state(recovered, words, spec)


class TestWalkStream:
    def test_matches_stepping_one_at_a_time(self):
        for spec, _, seed in REFERENCES:
            state = step_state(seed, spec)
            walked = list(walk_stream(state, 5, spec))
            assert [word for _, word in walked] == predict_words(state, 5, spec)
            expected = state
            for reached, _ in walked:
                expected = step_state(expected, spec)
                assert reached == expected

    def test_compiles_the_program_once_per_walk(self, monkeypatch):
        """The reason it exists: the learner's continuation search walks up
        to 64 steps on every execution, and compiling per step put that back
        where the generalization found it.
        """
        import fuzzer_tool.core.prng_state_recovery as mod

        spec = galois_lfsr("galois-fresh", 16, 0xB400)
        calls = []
        original = mod._compile_terms
        monkeypatch.setattr(
            mod,
            "_compile_terms",
            lambda vec, spec: calls.append(1) or original(vec, spec),
        )

        for _ in walk_stream((0xACE1,), 64, spec):
            pass
        first = len(calls)
        assert first > 0, "the memo cannot be warm for a freshly built spec"

        for _ in walk_stream((0xACE1,), 64, spec):
            pass
        assert len(calls) == first, "second walk must reuse the memo"


def test_predict_next_honours_custom_params():
    """Regression: ``predict_next`` took a ``params`` argument its body
    ignored, so every non-taus88 parameter set predicted a taus88 stream --
    silently, since the values are well-formed either way.
    """
    custom = ((32, 30, 5, 3), (32, 27, 7, 2), (32, 25, 11, 6))
    state = (0x12345678, 0x9ABCDEF0, 0xFEDCBA98)
    spec = FAMILIES["taus88"]

    expected = predict_words(state, 4, spec_from_params(custom))
    assert predict_next(state, 4, custom) == expected
    assert predict_next(state, 4, custom) != predict_words(state, 4, spec)
