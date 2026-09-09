"""Tests for the NIST SP 800-22 detector/operator pairs.

Six SP 800-22 tests (cusum, approximate entropy, non-overlapping and
overlapping template matching, Maurer's universal statistical test, and
random excursions/variant) had neither a detector in ``core/randomness`` nor
a constructive inverse in ``core/mutations/structured``. This file tests
both halves added to close that gap, following the same round-trip
philosophy as ``test_structured_mutations.py``: a construction that doesn't
move its own detector's statistic is worthless as a mutation operator.

On seeding: see that file's docstring. Threshold assertions here draw from a
fixed seed for the same reason (a checked-in statement about one sample, not
a live hypothesis test on every CI run); properties that must hold for any
draw (length preservation, no exceptions) use fresh entropy.
"""

import math
import os
import random

import pytest

from fuzzer_tool.core import randomness as R
from fuzzer_tool.core.mutations import structured as S
from fuzzer_tool.core.rand_pool import RandPool

SEED = 20260909

# (constructive operator, detector, extra detector kwargs)
PAIRS = (
    (S.cusum_bias_run, R.cumulative_sums, {}),
    (S.apen_short_period, R.approximate_entropy, {"m": 2}),
    (S.template_saturate, R.non_overlapping_template_matching, {}),
    (S.overlapping_template_flood, R.overlapping_template_matching, {}),
    (S.maurer_dictionary_collapse, R.maurers_universal, {"length": 6}),
    (S.excursion_square_wave, R.random_excursions, {}),
    (S.excursion_square_wave, R.random_excursions_variant, {}),
)

ALL_NIST_OPS = (
    S.cusum_bias_run,
    S.apen_short_period,
    S.template_saturate,
    S.overlapping_template_flood,
    S.maurer_dictionary_collapse,
    S.excursion_square_wave,
)


@pytest.fixture
def rp():
    return RandPool()


def noise(n, seed=SEED):
    return random.Random(seed).randbytes(n)


# ── Properties every construction must hold ────────────────────────────


class TestOperatorProperties:
    @pytest.mark.parametrize("op", ALL_NIST_OPS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize("size", [0, 1, 2, 3, 7, 15, 16, 64, 300, 512, 4097])
    def test_length_preserved_and_no_exceptions(self, op, size, rp):
        data = os.urandom(size)
        out = op(data, rng=rp)
        assert len(out) == size

    @pytest.mark.parametrize("op", ALL_NIST_OPS, ids=lambda f: f.__name__)
    def test_deterministic_given_seeded_rng(self, op):
        data = os.urandom(4096)
        out_a = op(data, rng=random.Random(7))
        out_b = op(data, rng=random.Random(7))
        assert out_a == out_b

    @pytest.mark.parametrize("op", ALL_NIST_OPS, ids=lambda f: f.__name__)
    def test_stays_pure_no_op_only_when_buffer_too_small(self, op):
        """Below whatever floor an operator needs, it must return the
        input unchanged (never raise, never truncate/pad)."""
        for size in (0, 1, 5, 16):
            data = bytes(range(size))
            assert op(data, rng=random.Random(1)) == data


# ── Round trips: construction -> detector rejects; noise -> it doesn't ──


class TestRoundTrips:
    @pytest.mark.parametrize(
        ("op", "detector", "kwargs"), PAIRS, ids=[f"{o.__name__}-{d.__name__}" for o, d, _ in PAIRS]
    )
    def test_construction_is_rejected(self, op, detector, kwargs):
        base = noise(4096)
        rp = RandPool()
        rp.reseed(SEED)
        rejected = 0
        trials = 15
        for _ in range(trials):
            out = op(base, rng=rp)
            p = detector(out, **kwargs)
            if p < 0.01:
                rejected += 1
        # Not every draw is guaranteed to land in the strongest region an
        # operator's random choices can produce; the property under test is
        # that the construction reliably moves the statistic, not that it
        # always maximizes it.
        assert rejected >= trials * 0.6, (
            f"{op.__name__}/{detector.__name__} rejected {rejected}/{trials}"
        )

    @pytest.mark.parametrize(
        ("op", "detector", "kwargs"), PAIRS, ids=[f"{o.__name__}-{d.__name__}" for o, d, _ in PAIRS]
    )
    def test_pure_noise_is_not_rejected(self, op, detector, kwargs):
        false_positives = 0
        trials = 15
        for i in range(trials):
            data = noise(4096, seed=SEED + i)
            p = detector(data, **kwargs)
            if p < 0.01:
                false_positives += 1
        # At p<0.01 roughly 1 in 100 draws is a spurious rejection; well
        # under half the trials firing is the bar for "this isn't just
        # rejecting everything".
        assert false_positives <= trials * 0.3, (
            f"{detector.__name__} false-rejected {false_positives}/{trials}"
        )


# ── Detector-specific sanity checks ─────────────────────────────────────


class TestCumulativeSums:
    def test_all_same_bit_is_maximally_rejected(self):
        assert R.cumulative_sums(bytes(4096)) < 1e-6
        assert R.cumulative_sums(b"\xff" * 4096) < 1e-6

    def test_forward_and_backward_modes_both_valid(self):
        data = noise(4096)
        assert 0.0 <= R.cumulative_sums(data, mode="forward") <= 1.0
        assert 0.0 <= R.cumulative_sums(data, mode="backward") <= 1.0

    def test_short_input_returns_one(self):
        assert R.cumulative_sums(bytes(5)) == 1.0


class TestApproximateEntropy:
    def test_period_one_pattern_is_rejected(self):
        assert R.approximate_entropy(bytes(4096), m=2) < 1e-6

    def test_random_data_not_rejected(self):
        p = R.approximate_entropy(noise(4096), m=2)
        assert p > 0.01

    def test_too_short_returns_one(self):
        # m=2 needs n >= 1<<(m+2) = 16 bits; 1 byte (8 bits) is below that.
        assert R.approximate_entropy(bytes(1), m=2) == 1.0


class TestNonOverlappingTemplateMatching:
    def test_saturated_template_is_rejected(self):
        tpl = (0, 0, 0, 0, 0, 0, 0, 0, 1)
        bits = [tpl[i % len(tpl)] for i in range(4096 * 8)]
        block = bytearray(4096)
        for i in range(4096):
            b = 0
            for j in range(8):
                b = (b << 1) | bits[i * 8 + j]
            block[i] = b
        assert R.non_overlapping_template_matching(bytes(block)) < 1e-6

    def test_too_few_blocks_returns_one(self):
        assert R.non_overlapping_template_matching(bytes(10)) == 1.0


class TestOverlappingTemplateMatching:
    def test_all_ones_is_rejected(self):
        assert R.overlapping_template_matching(b"\xff" * 4096) < 1e-6

    def test_too_short_returns_one(self):
        assert R.overlapping_template_matching(bytes(10)) == 1.0


class TestMaurersUniversal:
    def test_two_value_alphabet_is_rejected(self):
        data = bytes(0 if i % 2 == 0 else 1 for i in range(4096))
        assert R.maurers_universal(data, length=6) < 1e-6

    def test_insufficient_blocks_returns_one(self):
        assert R.maurers_universal(bytes(50), length=6) == 1.0

    def test_reference_stats_are_finite_and_sane(self):
        # Exercise the numerically-summed (not table-driven) reference for
        # a range of L; expected value should increase with L (bigger
        # alphabet -> longer average wait between repeats) and stay
        # positive; variance should stay positive too.
        prev_expected = 0.0
        for length in (4, 6, 8, 10):
            expected, variance = R._maurer_reference(length)
            assert math.isfinite(expected)
            assert math.isfinite(variance)
            assert variance > 0
            assert expected > prev_expected
            prev_expected = expected


class TestRandomExcursions:
    def test_triangle_wave_is_rejected(self):
        peak = 4
        period = [1] * peak + [0] * peak
        block = bytearray(4096)
        for i in range(4096):
            b = 0
            for j in range(8):
                bit = period[(i * 8 + j) % len(period)]
                b = (b << 1) | bit
            block[i] = b
        data = bytes(block)
        assert R.random_excursions(data) < 1e-6
        assert R.random_excursions_variant(data) < 1e-6

    def test_too_few_cycles_returns_one(self):
        assert R.random_excursions(bytes(10)) == 1.0
        assert R.random_excursions_variant(bytes(10)) == 1.0

    def test_random_noise_not_rejected(self):
        data = noise(20000)
        assert R.random_excursions(data) > 0.01
        assert R.random_excursions_variant(data) > 0.01


# ── The zero-correction header in excursion_square_wave ─────────────────


class TestExcursionSquareWaveZeroCorrection:
    def test_embedded_region_still_rejected_regardless_of_preceding_offset(self):
        """The periodic pattern only registers as completed cycles if it
        actually crosses *absolute* zero -- whatever precedes the mutated
        region has already walked the running sum away from zero, so the
        operator must correct for that before laying down the pattern.
        This is the regression the earlier (broken) version of the
        operator would fail: full rejection standalone, near-zero
        rejection once embedded after arbitrary preceding content.
        """
        rng = random.Random(3)
        rejected = 0
        trials = 20
        for i in range(trials):
            base = random.Random(100 + i).randbytes(4096)
            out = S.excursion_square_wave(base, rng=rng)
            if R.random_excursions(out) < 0.01:
                rejected += 1
        assert rejected >= trials * 0.8

    def test_correction_never_consumes_more_than_the_region_budget(self):
        """Even against a pathologically biased prefix (maximal |S|), the
        operator must not raise or produce a region shorter than it
        started with (the docstring's own budget guard)."""
        prefix = b"\xff" * 4096  # maximal positive running sum
        rng = random.Random(1)
        for _ in range(10):
            out = S.excursion_square_wave(prefix + os.urandom(200), rng=rng)
            assert len(out) == 4296


# ── Registry wiring ──────────────────────────────────────────────────────


class TestRegistryWiring:
    def test_all_six_registered_in_regularity_category(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        names = {op.__name__ for op in ALL_NIST_OPS}
        assert names <= REGISTRY.categories()["regularity"]

    def test_all_six_have_dispatchable_handlers(self):
        """Same construction ``_make_fuzzer`` in test_regression_no_op_mutations
        uses, inlined here rather than imported cross-file so this test
        doesn't depend on pytest's import-mode/sys.path handling for
        resolving a sibling test module.
        """
        import tempfile
        from unittest.mock import patch

        from fuzzer_tool.core.operator_registry import REGISTRY
        from fuzzer_tool.services.fuzzer import Fuzzer

        tmp = tempfile.mkdtemp(prefix="nist_dispatch_")
        os.makedirs(f"{tmp}/corpus", exist_ok=True)
        with (
            patch.object(Fuzzer, "_setup_forkserver", lambda self: None),
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=f"{tmp}/corpus",
                crashes_dir=f"{tmp}/crashes",
                max_len=4096,
                timeout=1,
                mutations_per_input=2,
            )
        table = REGISTRY.dispatch(f._operators)
        data = noise(4096)
        for op in ALL_NIST_OPS:
            f._rng.reseed(SEED)
            handler = table[op.__name__]
            out = handler(bytearray(data), len(data) // 2, bytes(data))
            assert len(out) == len(data)
