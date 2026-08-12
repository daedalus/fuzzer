"""Tests for the integer-modulus checksum dispatch module.

Correctness is asserted against *independent* references (``zlib.adler32`` and
straightforward running-loop implementations of Fletcher-16/32) rather than
against the module's own output, per the repo's equivalence-assertion rule.
"""

from __future__ import annotations

import random
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
    clear_active_int_model,
    compute_int_checksum,
    eval_model,
    get_active_int_model,
    model_from_dict,
    model_to_dict,
    set_active_int_model,
    sums,
    weighted_raw,
)


@pytest.fixture(autouse=True)
def _reset_active_model():
    clear_active_int_model()
    yield
    clear_active_int_model()


def _rand_bytes(rng: random.Random, lo: int, hi: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(rng.randrange(lo, hi)))


# ── independent references ─────────────────────────────────────────────


def _ref_fletcher16(data: bytes) -> int:
    a = b = 0
    for byte in data:
        a = (a + byte) % 255
        b = (b + a) % 255
    return (b << 8) | a


def _ref_fletcher32(data: bytes, big_endian: bool) -> int:
    if len(data) % 2:
        data += b"\x00"
    a = b = 0
    for i in range(0, len(data), 2):
        word = int.from_bytes(data[i : i + 2], "big" if big_endian else "little")
        a = (a + word) % 65535
        b = (b + a) % 65535
    return (b << 16) | a


# ── correctness ────────────────────────────────────────────────────────


def test_adler32_matches_zlib():
    rng = random.Random(1)
    for _ in range(200):
        data = _rand_bytes(rng, 0, 3000)
        assert eval_model(ADLER32, data) == zlib.adler32(data)


def test_adler32_known_vectors():
    # RFC 1950 style spot checks, computed independently of this module.
    assert eval_model(ADLER32, b"") == 1
    assert eval_model(ADLER32, b"a") == zlib.adler32(b"a")
    assert eval_model(ADLER32, b"Wikipedia") == zlib.adler32(b"Wikipedia")


def test_fletcher16_matches_reference():
    rng = random.Random(2)
    for _ in range(200):
        data = _rand_bytes(rng, 0, 400)
        assert eval_model(FLETCHER16, data) == _ref_fletcher16(data)


@pytest.mark.parametrize(("model", "big_endian"), [(FLETCHER32_LE, False), (FLETCHER32_BE, True)])
def test_fletcher32_matches_reference(model, big_endian):
    rng = random.Random(3)
    for _ in range(200):
        data = _rand_bytes(rng, 0, 400)
        assert eval_model(model, data) == _ref_fletcher32(data, big_endian)


def test_weighted_sum_matches_direct_formula():
    rng = random.Random(4)
    model = IntModel(KIND_WEIGHTED_SUM, 65521, multiplier=31, init_a=7)
    for _ in range(100):
        data = _rand_bytes(rng, 1, 40)
        expected = (sum(b * 31**j for j, b in enumerate(data)) + 7) % 65521
        assert eval_model(model, data) == expected


def test_closed_form_matches_running_loop():
    """The a/b closed forms must equal the step-by-step running recurrence."""
    rng = random.Random(5)
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    for _ in range(100):
        data = _rand_bytes(rng, 0, 500)
        a, b = 5, 9
        for byte in data:
            a = (a + byte) % 64007
            b = (b + a) % 64007
        assert eval_model(model, data) == (b << 16) | a


# ── vectorization equivalence (AGENTS.md rule 14) ──────────────────────


def test_sums_vectorized_matches_python():
    rng = random.Random(6)
    for _ in range(200):
        data = _rand_bytes(rng, 0, 900)
        n, s1, s2 = sums(data, 1, False)
        assert n == len(data)
        assert s1 == sum(data)
        assert s2 == sum((len(data) - t) * byte for t, byte in enumerate(data))


def test_sums_pads_odd_length_for_two_byte_words():
    n, s1, _s2 = sums(b"\x01\x00\x02", 2, False)
    assert n == 2
    assert s1 == 1 + 2  # trailing byte zero-padded into a 0x0002 word


def test_weighted_raw_multiplier_one_is_byte_sum():
    rng = random.Random(7)
    for _ in range(50):
        data = _rand_bytes(rng, 0, 2000)
        assert weighted_raw(data, 1) == sum(data)


def test_weighted_raw_horner_matches_power_series():
    rng = random.Random(8)
    for _ in range(50):
        data = _rand_bytes(rng, 1, 30)
        assert weighted_raw(data, 33) == sum(b * 33**j for j, b in enumerate(data))


# ── active model dispatch ──────────────────────────────────────────────


def test_compute_returns_none_without_active_model():
    assert compute_int_checksum(b"abc") is None


def test_active_model_round_trip():
    set_active_int_model(ADLER32)
    assert get_active_int_model() == ADLER32
    assert compute_int_checksum(b"abc") == zlib.adler32(b"abc")
    clear_active_int_model()
    assert compute_int_checksum(b"abc") is None


def test_explicit_model_overrides_active():
    set_active_int_model(ADLER32)
    assert compute_int_checksum(b"abc", FLETCHER16) == _ref_fletcher16(b"abc")


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown integer checksum kind"):
        eval_model(IntModel("nonsense", 255), b"abc")


def test_nbytes_tracks_output_width():
    assert ADLER32.nbytes == 4
    assert FLETCHER16.nbytes == 2


# ── serialization ──────────────────────────────────────────────────────


@pytest.mark.parametrize("model", [ADLER32, FLETCHER16, FLETCHER32_BE])
def test_model_dict_round_trip(model):
    assert model_from_dict(model_to_dict(model)) == model


def test_model_to_dict_none():
    assert model_to_dict(None) is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"kind": "bogus", "modulus": 255},
        {"kind": KIND_FLETCHER, "modulus": 1},
        {"kind": KIND_FLETCHER},
        {"kind": KIND_FLETCHER, "modulus": "not-an-int", "init_a": None},
    ],
)
def test_model_from_dict_rejects_malformed(payload):
    """A corrupt state.json must yield None, never a raise or a bogus model."""
    assert model_from_dict(payload) is None
