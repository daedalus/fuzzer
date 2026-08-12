"""Tests for integer-modulus checksum recovery.

Covers the four areas the design handover called for: correctness against
known checksums, adversarial false-positive resistance (the highest-risk
failure mode -- a wrong model silently corrupts every subsequent mutation),
empirical characterization of the minimum pair count, and cost bounds.
"""

from __future__ import annotations

import random
import time
import zlib

import pytest

from fuzzer_tool.core.int_checksum import (
    ADLER32,
    FLETCHER16,
    FLETCHER32_BE,
    FLETCHER32_LE,
    KIND_FLETCHER,
    KIND_WEIGHTED_SUM,
    IntModel,
    eval_model,
    model_from_dict,
    model_to_dict,
)
from fuzzer_tool.core.int_checksum_solver import (
    INT_PAIRS_MAX,
    MIN_VERIFY_FLOOR,
    recover_int_model,
    verify_model,
)


def _pairs(model: IntModel, count: int, seed: int, lo: int = 600, hi: int = 4000):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(lo, hi)))
        out.append((data, eval_model(model, data)))
    return out


# ── correctness: known checksums ───────────────────────────────────────


def test_recovers_real_adler32():
    """Pairs built with zlib.adler32, not with this module's own evaluator."""
    rng = random.Random(11)
    pairs = []
    for _ in range(16):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(600, 4000)))
        pairs.append((data, zlib.adler32(data)))
    assert recover_int_model(pairs) == ADLER32


@pytest.mark.parametrize(
    ("model", "lo", "hi"),
    [
        (FLETCHER16, 8, 300),
        (FLETCHER32_LE, 8, 300),
        (FLETCHER32_BE, 8, 300),
        (ADLER32, 600, 4000),
    ],
)
def test_recovers_common_models(model, lo, hi):
    assert recover_int_model(_pairs(model, 16, 21, lo, hi)) == model


def test_recovers_unknown_fletcher_modulus():
    """Not in the common list -- exercises the general GCD path."""
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    assert recover_int_model(_pairs(model, 16, 31)) == model


def test_recovers_unknown_weighted_sum():
    model = IntModel(KIND_WEIGHTED_SUM, 4294967291, multiplier=31, init_a=7, out_bits=32)
    assert recover_int_model(_pairs(model, 16, 41, 8, 60)) == model


def test_recovers_plain_byte_sum_with_offset():
    model = IntModel(KIND_WEIGHTED_SUM, 65413, multiplier=1, init_a=1234, out_bits=16)
    assert recover_int_model(_pairs(model, 16, 51)) == model


def test_recovered_model_reproduces_held_out_pairs():
    """Generalization, not memorization: verify on data unused in recovery."""
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    recovered = recover_int_model(_pairs(model, 16, 61))
    assert recovered is not None
    held_out = _pairs(model, 20, 9999)
    assert all(eval_model(recovered, data) == checksum for data, checksum in held_out)


def test_recovered_model_survives_serialization():
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    recovered = recover_int_model(_pairs(model, 16, 71))
    assert model_from_dict(model_to_dict(recovered)) == recovered


def test_recovered_weighted_model_survives_serialization():
    """out_bits must land on a width model_from_dict accepts."""
    model = IntModel(KIND_WEIGHTED_SUM, 4294967291, multiplier=31, init_a=7, out_bits=32)
    recovered = recover_int_model(_pairs(model, 16, 81, 8, 60))
    assert recovered is not None
    assert model_from_dict(model_to_dict(recovered)) == recovered


# ── adversarial / false-positive resistance ────────────────────────────


def test_random_pairs_never_recover_a_model():
    """The highest-risk failure mode: a model that fits noise by chance."""
    false_positives = []
    for seed in range(200):
        rng = random.Random(seed)
        pairs = [
            (
                bytes(rng.randrange(256) for _ in range(rng.randrange(600, 3000))),
                rng.randrange(1 << 32),
            )
            for _ in range(12)
        ]
        model = recover_int_model(pairs)
        if model is not None:
            false_positives.append((seed, model))
    assert not false_positives


def test_random_short_pairs_never_recover_a_model():
    false_positives = []
    for seed in range(200):
        rng = random.Random(seed + 5000)
        pairs = [
            (bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64))), rng.randrange(1 << 16))
            for _ in range(12)
        ]
        if recover_int_model(pairs) is not None:
            false_positives.append(seed)
    assert not false_positives


def test_constant_checksum_pairs_rejected():
    """A degenerate 'everything maps to one value' fit must not verify."""
    rng = random.Random(123)
    pairs = [(bytes(rng.randrange(256) for _ in range(1000)), 7) for _ in range(12)]
    assert recover_int_model(pairs) is None


def test_verify_requires_two_distinct_checksums():
    """Matching a run of identical observations is not evidence."""
    model = IntModel(KIND_WEIGHTED_SUM, 251, multiplier=1, init_a=0, out_bits=16)
    # Distinct data, all with the same checksum under the model.
    pairs = [(b"\x01" * 251, 0), (b"\x01" * 502, 0), (b"\x01" * 753, 0), (b"\x01" * 1004, 0)]
    assert all(eval_model(model, d) == c for d, c in pairs)
    assert not verify_model(model, pairs)


def test_corrupted_checksums_are_rejected():
    """Right data, wrong checksums -- must not fall back to a partial fit."""
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    pairs = [(data, checksum ^ 0x5A5A) for data, checksum in _pairs(model, 16, 91)]
    assert recover_int_model(pairs) is None


def test_too_few_pairs_returns_none():
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    assert recover_int_model([]) is None
    assert recover_int_model(_pairs(model, 1, 101)) is None


def test_empty_data_pairs_carry_no_signal():
    """gzip extraction emits empty-payload pairs by design; they must not fit."""
    assert recover_int_model([(b"", 1), (b"", 2), (b"", 3), (b"", 4)]) is None


# ── empirical characterization (handover section 6) ────────────────────


def _corrupt(pairs, n_bad, seed):
    rng = random.Random(seed + 777)
    out = list(pairs)
    for idx in rng.sample(range(len(out)), n_bad):
        out[idx] = (out[idx][0], rng.randrange(1 << 20))
    return out


@pytest.mark.parametrize("n_bad", [1, 2, 3])
def test_recovers_despite_corrupt_pairs(n_bad):
    """cmplog pair extraction is explicitly heuristic, so bad pairs are expected.

    A single-chain GCD is all-or-nothing: one corrupt pair drags it to 1 and
    recovery fails outright (measured 0/200).  The consensus path scores
    candidates by congruence-class size instead and recovers ~200/200.
    """
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    recovered = 0
    for seed in range(20):
        pairs = _corrupt(_pairs(model, 16, seed * 5 + 400), n_bad, seed)
        got = recover_int_model(pairs)
        assert got in (None, model), f"wrong model from corrupt pairs: {got}"
        recovered += got == model
    assert recovered >= 18, f"only {recovered}/20 recovered with {n_bad} corrupt pairs"


def test_corrupt_pairs_never_yield_a_wrong_model():
    """Outlier tolerance must not become outlier *credulity*."""
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    for seed in range(30):
        # Majority garbage: no honest model should survive verification.
        pairs = _corrupt(_pairs(model, 12, seed * 11 + 900), 9, seed)
        assert recover_int_model(pairs) in (None, model)


def test_minimum_pair_count_for_reliable_recovery():
    """Characterizes min_pairs rather than assuming the GF(2) path's value.

    Measured across 60 seeds per pair count.  The important property is not
    just the success rate but the *shape* of the failures: recovery either
    finds the exact modulus or returns None.  It never returns a wrong model,
    which is what makes a low pair count merely unproductive rather than
    dangerous.
    """
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    rates = {}
    for count in (2, 4, 8, 12):
        exact = wrong = 0
        for seed in range(60):
            recovered = recover_int_model(_pairs(model, count, seed * 13 + count))
            if recovered == model:
                exact += 1
            elif recovered is not None:
                wrong += 1
        rates[count] = exact / 60
        assert wrong == 0, f"{count} pairs produced {wrong} wrong models"

    # Monotone-ish improvement, and 12 pairs is comfortably reliable. Stated
    # as bounds, not equalities, so the test does not break on tuning.
    assert rates[12] >= 0.95
    assert rates[12] >= rates[2]


def test_modulus_is_underdetermined_when_the_sum_never_wraps():
    """The precondition on pair size.

    With k=1 and sum(data) < N for every pair, the modulus is invisible: the
    checksum equals the raw sum, every pairwise difference is exactly zero,
    and *any* modulus above the largest observed sum fits the evidence
    equally well.  Nothing can do better here, because the information is not
    in the data.

    What recovery must guarantee here is not that it finds the true modulus
    but that whatever it returns reproduces every observation.  It does: it
    returns a congruent model (mod 2**16 rather than mod 65413), which agrees
    with the true model on the entire regime the evidence covers and diverges
    only on inputs that wrap -- inputs the fuzzer has never seen.
    """
    model = IntModel(KIND_WEIGHTED_SUM, 65413, multiplier=1, init_a=0, out_bits=16)
    # <=200 bytes -> max sum 51000 < 65413, so the sum never wraps.
    short = _pairs(model, 16, 111, 150, 200)
    assert all(sum(data) < model.modulus for data, _ in short)

    recovered = recover_int_model(short)
    assert recovered is not None
    assert recovered.modulus >= max(sum(data) for data, _ in short)
    assert all(eval_model(recovered, data) == checksum for data, checksum in short)

    # Given pairs long enough to wrap, the true modulus is pinned down.
    assert recover_int_model(_pairs(model, 16, 111, 600, 4000)) == model


def test_fails_closed_when_no_hypothesis_fits():
    """Non-wrapping pairs that no common model explains must yield None.

    Same no-wrap regime as above, but with a nonzero init the plain-sum
    models cannot absorb.  The general path's GCD is zero (all residuals
    identical), so recovery declines rather than inventing a modulus -- and
    this is exactly why the integer path cannot reuse the GF(2) path's
    256-byte pair cap, which would force every pair into this regime.
    """
    model = IntModel(KIND_WEIGHTED_SUM, 65413, multiplier=1, init_a=1234, out_bits=16)
    short = _pairs(model, 16, 112, 150, 200)
    assert all(sum(data) + 1234 < model.modulus for data, _ in short)
    assert recover_int_model(short) is None


def test_verify_floor_is_at_least_two_pairs():
    """The handover's non-negotiable: never activate on a single pair."""
    model = IntModel(KIND_WEIGHTED_SUM, 65413, multiplier=1, init_a=0, out_bits=16)
    data = b"\x7f" * 900
    assert MIN_VERIFY_FLOOR >= 2
    assert not verify_model(model, [(data, eval_model(model, data))], min_matches=1)


# ── cost bounds ────────────────────────────────────────────────────────


def test_pair_count_is_capped():
    """Reduction cost scales with pair count -- cap it."""
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    assert recover_int_model(_pairs(model, INT_PAIRS_MAX * 4, 121)) == model


def test_recovery_of_unrecoverable_pairs_is_fast():
    """The worst case is failure: every hypothesis tried, nothing verifies.

    A prior unbounded GCD path in this subsystem blocked fuzz_one() for 30+
    seconds; this path runs in the same place and must not regress that.
    """
    rng = random.Random(131)
    pairs = [
        (bytes(rng.randrange(256) for _ in range(4000)), rng.randrange(1 << 32)) for _ in range(64)
    ]
    start = time.perf_counter()
    assert recover_int_model(pairs) is None
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"failed recovery took {elapsed:.2f}s"


def test_long_pairs_with_large_multiplier_are_bounded():
    """weighted_raw with multiplier > 1 is O(len) limbs; long pairs are capped."""
    rng = random.Random(141)
    pairs = [
        (bytes(rng.randrange(256) for _ in range(200_000)), rng.randrange(1 << 32))
        for _ in range(8)
    ]
    start = time.perf_counter()
    recover_int_model(pairs)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"large-pair recovery took {elapsed:.2f}s"
