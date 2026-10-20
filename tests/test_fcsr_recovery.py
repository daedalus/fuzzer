"""Tests for core/fcsr_recovery.py: multiply-with-carry (FCSR) state recovery.

Streams come from a reference MWC written here, independent of the module.
"""

import pytest

from fuzzer_tool.core.fcsr_recovery import (
    MIN_UNKNOWN_WORDS,
    MWC_FAMILIES,
    MWCSpec,
    adic_synth,
    confident_samples,
    min_samples,
    mwc_family,
    predict_words,
    recover_state,
    recover_unknown,
    verify_state,
    walk_stream,
)
from fuzzer_tool.core.rand_pool import RandPool

BASE_BITS = 32
BASE = 1 << BASE_BITS
FAMILIES = sorted(MWC_FAMILIES)


def _ref_stream(a: int, x: int, c: int, n: int) -> list[int]:
    """Reference MWC: output x, then (x, c) = divmod(a*x + c, 2^32)."""
    out = []
    for _ in range(n):
        out.append(x)
        c, x = divmod(a * x + c, BASE)
    return out


def _words(spec: MWCSpec, rng: RandPool, n: int) -> tuple[list[int], tuple[int, int]]:
    x0, c0 = rng.randrange(BASE), rng.randrange(spec.a)
    return _ref_stream(spec.a, x0, c0, n), (x0, c0)


class TestSpec:
    def test_lookup_round_trips_every_family(self):
        for name, spec in MWC_FAMILIES.items():
            assert spec.name == name
            assert mwc_family(name) is spec

    def test_name_encodes_multiplier(self):
        """A learner dict written for an unknown multiplier must rebuild it."""
        spec = mwc_family("mwc32_4294883355")
        assert spec is not None and spec.a == 4294883355

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "mwc",
            "mwc32_",
            "mwc32_0",
            "mwc32_-5",
            "mwc32_4294967296",
            "mwc64_3",
            "mwc32_x",
            "taus88",
        ],
    )
    def test_bad_names_rejected(self, name):
        """Adversarial: persisted names are untrusted input."""
        assert mwc_family(name) is None

    def test_multiplier_must_fit_the_base(self):
        with pytest.raises(ValueError):
            MWCSpec("bad", a=BASE)
        with pytest.raises(ValueError):
            MWCSpec("bad", a=0)

    def test_sample_counts(self):
        for spec in MWC_FAMILIES.values():
            assert min_samples(spec) == 2
            assert confident_samples(spec) == min_samples(spec) + 1

    def test_prime_safe_multipliers(self):
        """Shipped multipliers: a*b - 1 prime (full-period l-sequence)."""
        for spec in MWC_FAMILIES.values():
            q = spec.a * BASE - 1
            assert all(pow(base, q - 1, q) == 1 for base in (2, 3, 5, 7))


class TestKnownMultiplier:
    @pytest.mark.parametrize("name", FAMILIES)
    def test_recovers_and_predicts(self, name):
        spec = MWC_FAMILIES[name]
        rng = RandPool(seed=21)
        n = confident_samples(spec)
        for _ in range(50):
            ys, state0 = _words(spec, rng, n + 8)
            state = recover_state(ys[:n], spec)
            assert state == state0
            assert verify_state(state, ys[:n], spec)
            # Falsification: predictions beyond the samples match the reference.
            assert predict_words(state, n + 7, spec) == ys[1 : n + 8]

    def test_walk_stream_yields_state_and_output(self):
        spec = MWC_FAMILIES[FAMILIES[0]]
        ys = _ref_stream(spec.a, 5, 9, 5)
        walked = list(walk_stream((5, 9), 4, spec))
        assert [w for _, w in walked] == ys[1:]
        assert walked[-1][0][0] == ys[4]

    def test_too_few_samples_raise(self):
        with pytest.raises(ValueError):
            recover_state([1], MWC_FAMILIES[FAMILIES[0]])

    def test_out_of_range_word_rejected(self):
        """Adversarial: a 32-bit generator cannot emit a 33-bit word."""
        spec = MWC_FAMILIES[FAMILIES[0]]
        ys = _ref_stream(spec.a, 1, 2, 4)
        ys[2] = BASE
        assert recover_state(ys, spec) is None

    def test_unreachable_carry_rejected(self):
        """Adversarial: the carry is < a, so x1 = a*x0 + c with c >= a has no state."""
        spec = MWCSpec("tiny", a=1000)
        x0, x1 = 7, (1000 * 7 + 5000) % BASE
        assert recover_state([x0, x1, 0], spec) is None

    def test_random_words_do_not_verify(self):
        """Adversarial: unrelated words must not fit at the confident count."""
        rng = RandPool(seed=5)
        spec = MWC_FAMILIES[FAMILIES[0]]
        n = confident_samples(spec)
        hits = sum(
            recover_state([rng.randrange(BASE) for _ in range(n)], spec) is not None
            for _ in range(200)
        )
        assert hits == 0

    def test_control_self_consistent(self):
        """Control (Rule 46): recovery on the same stream twice agrees."""
        spec = MWC_FAMILIES[FAMILIES[0]]
        ys = _ref_stream(spec.a, 123456, 789, 5)
        assert recover_state(ys, spec) == recover_state(ys, spec)


class TestUnknownMultiplier:
    """2-adic rational approximation: the word stream is the 2-adic expansion of p/q, q = a*b - 1."""

    @pytest.mark.parametrize("name", FAMILIES)
    def test_recovers_multiplier_from_five_words(self, name):
        true = MWC_FAMILIES[name]
        rng = RandPool(seed=33)
        assert MIN_UNKNOWN_WORDS == 5
        for _ in range(100):
            ys, state0 = _words(true, rng, MIN_UNKNOWN_WORDS)
            found = recover_unknown(ys)
            assert found is not None
            spec, state = found
            assert spec.a == true.a
            assert state == state0
            assert predict_words(state, 6, spec) == _ref_stream(true.a, *state0, 8)[1:7]

    def test_four_words_refused(self):
        """Falsification: 128 bits cannot pin a 64-bit-complexity register."""
        with pytest.raises(ValueError):
            recover_unknown([1, 2, 3, 4])

    def test_random_windows_never_fit(self):
        """Adversarial: random words have complexity ~N/2, far above 64 bits."""
        rng = RandPool(seed=8)
        hits = sum(
            recover_unknown([rng.randrange(BASE) for _ in range(MIN_UNKNOWN_WORDS)]) is not None
            for _ in range(300)
        )
        assert hits == 0

    def test_lcg_stream_is_not_an_mwc(self):
        """Adversarial: a different generator class must not be claimed."""
        x, ys = 0xCAFE, []
        for _ in range(8):
            x = (214013 * x + 2531011) % BASE
            ys.append(x)
        assert recover_unknown(ys) is None

    def test_composite_modulus_never_yields_a_wrong_multiplier(self):
        """a*b - 1 composite shrinks q; the answer is the true a or nothing, never another a."""
        true = MWCSpec("c", a=1000003)
        rng = RandPool(seed=2)
        for _ in range(100):
            ys, _ = _words(true, rng, 7)
            found = recover_unknown(ys)
            assert found is None or found[0].a == true.a

    def test_control_self_consistent(self):
        spec = MWC_FAMILIES[FAMILIES[0]]
        ys = _ref_stream(spec.a, 42, 7, MIN_UNKNOWN_WORDS)
        assert recover_unknown(ys) == recover_unknown(ys)


class TestAdicSynth:
    def test_recovers_small_fraction(self):
        """p/q with q odd is the shortest generator once N >= 2*bitlen(q) + 1."""
        q, p = 1019, 400
        bits = []
        for _ in range(40):
            b = p & 1
            bits.append(b)
            p = (p - b * q) // 2
        assert adic_synth(sum(b << i for i, b in enumerate(bits)), len(bits)) == (400, 1019)

    def test_all_zero_stream(self):
        assert adic_synth(0, 64) == (0, 1)

    def test_q_is_odd(self):
        """Adversarial: an even q is not a 2-adic denominator."""
        for s in (1, 6, 0xDEADBEEF, 2**63 + 5):
            _, q = adic_synth(s, 64)
            assert q & 1
