"""ChecksumLearner integration for the integer-modulus checksum family.

Covers the wiring the GF(2) path already had and the integer path needed:
extraction, recovery, activation, serialization, and the separation between
``compute_checksum`` (CRC-32 family) and ``compute_int_checksum``.
"""

from __future__ import annotations

import random
import struct
import zlib

import pytest

from fuzzer_tool.core.checksum_learner import ChecksumLearner
from fuzzer_tool.core.int_checksum import (
    ADLER32,
    KIND_FLETCHER,
    IntModel,
    clear_active_int_model,
    eval_model,
    get_active_int_model,
)


class _FakeFuzzer:
    def __init__(self):
        self._cmplog = None


@pytest.fixture(autouse=True)
def _reset_active_model():
    clear_active_int_model()
    yield
    clear_active_int_model()


def _png_with_idat(payload: bytes) -> bytes:
    """A PNG whose IDAT carries a real zlib stream over *payload*."""
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    for chunk_type, chunk_data in (
        (b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)),
        (b"IDAT", zlib.compress(payload)),
        (b"IEND", b""),
    ):
        out += struct.pack(">I", len(chunk_data))
        out += chunk_type + chunk_data
        out += struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
    return bytes(out)


# ── zlib / Adler-32 extraction ─────────────────────────────────────────


def test_zlib_extractor_pulls_adler_pair_from_png():
    rng = random.Random(1)
    payload = bytes(rng.randrange(256) for _ in range(4000))
    learner = ChecksumLearner(_FakeFuzzer())
    pairs = learner._extract_zlib_adler_pairs(_png_with_idat(payload))
    assert pairs == [(payload, zlib.adler32(payload))]


def test_zlib_extractor_handles_bare_stream():
    payload = b"the quick brown fox" * 200
    learner = ChecksumLearner(_FakeFuzzer())
    pairs = learner._extract_zlib_adler_pairs(zlib.compress(payload))
    assert pairs == [(payload, zlib.adler32(payload))]


@pytest.mark.parametrize(
    "data",
    [b"", b"not a png", b"\x89PNG\r\n\x1a\n", b"\x78\x9c" + b"\x00" * 40],
)
def test_zlib_extractor_tolerates_junk(data):
    """Runs on attacker-shaped input every iteration; must never raise."""
    learner = ChecksumLearner(_FakeFuzzer())
    assert learner.extract_format_pairs(data) is not None


def test_zlib_extractor_bounds_inflate_output():
    """A zip bomb must not be inflated in full inside fuzz_one()."""
    learner = ChecksumLearner(_FakeFuzzer())
    bomb = zlib.compress(b"\x00" * (4 << 20))
    assert learner._extract_zlib_adler_pairs(bomb) == []


# ── recovery, activation, persistence ──────────────────────────────────


def _adler_learner(count: int = 16) -> ChecksumLearner:
    rng = random.Random(2)
    learner = ChecksumLearner(_FakeFuzzer(), min_pairs=4)
    pairs = []
    for _ in range(count):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(600, 4000)))
        pairs.append((data, zlib.adler32(data)))
    learner.add_pairs(pairs)
    return learner


def test_learner_recovers_and_activates_adler():
    learner = _adler_learner()
    assert learner.ensure_int_model() == ADLER32
    assert get_active_int_model() == ADLER32
    assert learner.has_model()
    assert learner.ensure_model()


def test_learner_int_checksum_matches_zlib():
    learner = _adler_learner()
    assert learner.compute_int_checksum(b"payload data") == zlib.adler32(b"payload data")


def test_compute_checksum_stays_on_the_crc_family():
    """An Adler model must never leak into compute_checksum().

    _patch_png_crc/_patch_zip_crc call compute_checksum() meaning "the CRC-32
    this format specifies". PNG chunk CRCs are CRC-32 by spec and the
    Adler-32 lives one layer below, inside the IDAT zlib stream — routing the
    integer model through here would silently corrupt every mutated PNG.
    """
    learner = _adler_learner()
    assert learner.ensure_int_model() is not None
    assert learner.compute_checksum(b"payload data") == zlib.crc32(b"payload data")


def test_int_checksum_is_none_without_a_model():
    learner = ChecksumLearner(_FakeFuzzer())
    assert learner.compute_int_checksum(b"abc") is None
    assert not learner.has_model()


def test_int_model_survives_state_round_trip():
    learner = _adler_learner()
    assert learner.ensure_int_model() == ADLER32
    state = learner.to_dict()

    clear_active_int_model()
    restored = ChecksumLearner.from_dict(_FakeFuzzer(), state)
    assert restored._int_model == ADLER32
    # Reload must re-activate the model module-wide, as the GF(2) path does.
    assert get_active_int_model() == ADLER32
    assert restored.compute_int_checksum(b"abc") == zlib.adler32(b"abc")


def test_from_dict_tolerates_corrupt_int_model():
    state = {"poly": None, "int_model": {"kind": "bogus", "modulus": "x"}}
    restored = ChecksumLearner.from_dict(_FakeFuzzer(), state)
    assert restored._int_model is None
    assert not restored.has_model()


def test_recovery_stops_retrying_once_int_model_found():
    """The retry gate keys on has_model(), not on the polynomial alone."""
    learner = _adler_learner()
    assert learner.ensure_int_model() is not None
    attempted_at = learner._pairs_attempted_at
    learner.add_pairs([(b"x" * 700, zlib.adler32(b"x" * 700))] * 40)
    assert learner._pairs_attempted_at == attempted_at


def test_unrecoverable_pairs_leave_both_models_unset():
    rng = random.Random(3)
    learner = ChecksumLearner(_FakeFuzzer(), min_pairs=4)
    learner.add_pairs(
        [(bytes(rng.randrange(256) for _ in range(900)), rng.randrange(1 << 32)) for _ in range(32)]
    )
    assert learner.ensure_poly() is None
    assert learner.ensure_int_model() is None
    assert not learner.has_model()


def test_learner_recovers_non_standard_fletcher_end_to_end():
    model = IntModel(KIND_FLETCHER, 64007, init_a=5, init_b=9, word_bytes=1, out_bits=32)
    rng = random.Random(4)
    learner = ChecksumLearner(_FakeFuzzer(), min_pairs=4)
    pairs = []
    for _ in range(16):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(600, 4000)))
        pairs.append((data, eval_model(model, data)))
    learner.add_pairs(pairs)
    assert learner.ensure_int_model() == model
