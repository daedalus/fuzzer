"""Linear integer checksum family: ``c = (sum w_j * word_j + init) mod N``.

Recovered by integer relation (core/integer_relation.py) inside
``recover_int_model``; activated by ChecksumLearner; patched by crc_learn.
"""

from __future__ import annotations

import random
import time
import zlib
from types import SimpleNamespace

import pytest

from fuzzer_tool.core import int_checksum_solver as solver
from fuzzer_tool.core.analyzers.analyzer_checksum_learner import ChecksumLearner
from fuzzer_tool.core.int_checksum import (
    KIND_LINEAR,
    IntModel,
    clear_active_int_model,
    eval_model,
    model_from_dict,
    model_to_dict,
)
from fuzzer_tool.core.int_checksum_solver import _parity_consistent, recover_int_model
from fuzzer_tool.services.operators import OperatorEngine

BYTE_W = (3, -1, 7, 2, 0, 5, -4, 1)
BYTE_MODEL = IntModel(KIND_LINEAR, 1 << 16, init_a=1234, out_bits=16, weights=BYTE_W)
WORD_W = (2, -3, 1, 9)
WORD_MODEL = IntModel(
    KIND_LINEAR, 1 << 32, init_a=77, word_bytes=2, out_bits=32, big_endian=True, weights=WORD_W
)


class _FakeFuzzer:
    def __init__(self):
        self._cmplog = None
        self._op_declines = {}


@pytest.fixture(autouse=True)
def _reset_active_model():
    clear_active_int_model()
    yield
    clear_active_int_model()


def _byte_sum(data: bytes) -> int:
    """Reference checksum written out by hand, independent of eval_model."""
    return (sum(w * b for w, b in zip(BYTE_W, data, strict=False)) + 1234) % (1 << 16)


def _word_sum(data: bytes) -> int:
    words = [int.from_bytes(data[i : i + 2], "big") for i in range(0, len(data), 2)]
    return (sum(w * v for w, v in zip(WORD_W, words, strict=False)) + 77) % (1 << 32)


def _pairs(ref, count: int, size: int, seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        data = bytes(rng.randrange(256) for _ in range(size))
        out.append((data, ref(data)))
    return out


# ── evaluation ─────────────────────────────────────────────────────────


def test_eval_matches_reference():
    for data, checksum in _pairs(_byte_sum, 8, 8, 1) + _pairs(_byte_sum, 4, 3, 2):
        assert eval_model(BYTE_MODEL, data) == checksum


def test_eval_short_and_long_data_use_prefix():
    """Fewer words than weights: missing words count as zero; extra words ignored."""
    assert eval_model(BYTE_MODEL, b"\x01\x02") == (3 - 2 + 1234) % (1 << 16)
    assert eval_model(BYTE_MODEL, bytes(range(1, 20))) == _byte_sum(bytes(range(1, 20)))


# ── recovery ───────────────────────────────────────────────────────────


def test_recovers_byte_weights():
    assert recover_int_model(_pairs(_byte_sum, 16, 8, 3)) == BYTE_MODEL


def test_recovers_big_endian_word_weights():
    assert recover_int_model(_pairs(_word_sum, 16, 8, 4)) == WORD_MODEL


def test_falsification_random_fixed_length_pairs():
    """Noise over one fixed length must never yield a linear model."""
    rng = random.Random(5)
    for size in (4, 8, 16):
        pairs = [
            (bytes(rng.randrange(256) for _ in range(size)), rng.randrange(1 << 32))
            for _ in range(16)
        ]
        assert recover_int_model(pairs) is None


def test_adversarial_crc_is_not_linear():
    """CRC-32 over fixed-length data is affine over GF(2), not over Z/2^32."""
    rng = random.Random(6)
    pairs = []
    for _ in range(16):
        data = bytes(rng.randrange(256) for _ in range(8))
        pairs.append((data, zlib.crc32(data)))
    model = recover_int_model(pairs)
    assert model is None or model.kind != KIND_LINEAR


def test_adversarial_no_held_out_pairs_abstains():
    """Only the fit set: the fit reproduces itself, which is no evidence."""
    pairs = _pairs(_byte_sum, len(BYTE_W) + 2, 8, 7)
    assert recover_int_model(pairs) is None


def test_adversarial_corrupt_pair_never_wrong_model():
    pairs = _pairs(_byte_sum, 16, 8, 8)
    data, checksum = pairs[0]
    pairs[0] = (data, checksum ^ 0x5A5A)
    model = recover_int_model(pairs)
    assert model is None or model == BYTE_MODEL


def test_failed_recovery_is_bounded():
    """Worst case: every config tried at the word cap, nothing verifies."""
    rng = random.Random(9)
    pairs = [
        (bytes(rng.randrange(256) for _ in range(16)), rng.randrange(1 << 16)) for _ in range(64)
    ]
    start = time.perf_counter()
    assert recover_int_model(pairs) is None
    assert time.perf_counter() - start < 2.0


# ── persistence ────────────────────────────────────────────────────────


def test_round_trip_including_list_weights():
    assert model_from_dict(model_to_dict(WORD_MODEL)) == WORD_MODEL
    state = dict(model_to_dict(WORD_MODEL) or {})
    state["weights"] = list(WORD_W)  # JSON turns tuples into lists
    assert model_from_dict(state) == WORD_MODEL


@pytest.mark.parametrize("weights", [(), ["a", 1], [True, 2], "12", None])
def test_adversarial_bad_weights_rejected(weights):
    state = dict(model_to_dict(BYTE_MODEL) or {})
    state["weights"] = weights
    assert model_from_dict(state) is None


# ── wiring: learner + crc_learn ────────────────────────────────────────


def test_learner_activates_linear_model():
    learner = ChecksumLearner(_FakeFuzzer(), min_pairs=4)
    learner.add_pairs(_pairs(_byte_sum, 16, 8, 10))
    assert learner.ensure_int_model() == BYTE_MODEL
    assert learner.compute_int_checksum(b"\x09" * 8) == _byte_sum(b"\x09" * 8)


def test_crc_learn_patches_trailing_field():
    learner = ChecksumLearner(_FakeFuzzer())
    learner._set_int_model(BYTE_MODEL)
    eng = SimpleNamespace(ctx=SimpleNamespace(checksum_learner=learner, _rng=None))
    eng._try_format_int_patch = lambda buf, m: OperatorEngine._try_format_int_patch(eng, buf, m)

    body = bytes(range(10, 18))
    buf = bytearray(body + b"\x00\x00")
    OperatorEngine._op_crc_learn(eng, buf, 0, bytes(buf))
    assert bytes(buf) == body + _byte_sum(body).to_bytes(2, "big")


# ── parity pre-filter ──────────────────────────────────────────────────


def test_parity_filter_accepts_linear_rows():
    rows = [(list(d), c) for d, c in _pairs(_byte_sum, 16, 8, 11)]
    assert _parity_consistent(rows)


def test_parity_filter_rejects_noise():
    """Falsification: 16 random rows over 8 unknowns + offset are inconsistent mod 2."""
    rng = random.Random(12)
    rows = [([rng.randrange(256) for _ in range(8)], rng.randrange(1 << 16)) for _ in range(16)]
    assert not _parity_consistent(rows)


def test_parity_filter_skips_relation_search(monkeypatch):
    """Noise must not reach the integer-relation solve (Hard Rule 41)."""
    rng = random.Random(13)
    pairs = [
        (bytes(rng.randrange(256) for _ in range(16)), rng.randrange(1 << 16)) for _ in range(64)
    ]
    calls = []
    real = solver.find_relation
    monkeypatch.setattr(solver, "find_relation", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert recover_int_model(pairs) is None
    assert not calls


def test_adversarial_short_pairs_among_long_ones():
    """PNG-sized pairs must not crowd the short linear pairs out of the search."""
    rng = random.Random(14)
    noise = [
        (bytes(rng.randrange(256) for _ in range(700)), rng.randrange(1 << 32)) for _ in range(40)
    ]
    assert recover_int_model(noise + _pairs(_byte_sum, 16, 8, 15)) == BYTE_MODEL
