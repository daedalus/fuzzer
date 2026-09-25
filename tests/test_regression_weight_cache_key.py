"""Regression: ``weighted_pick_seed`` lazily initialised its cache on
``hasattr(f, "_weight_cache")``, seeding the key with the tuple ``(-1, -1)``.

Corpus pruning (``deprioritize_near_duplicates``, ``auto_minimize_corpus``)
also writes ``_weight_cache`` / ``_cached_weights``; if it ran before the
first pick, the key was never created and the pick raised AttributeError.
The key is now its own guard and an int, matching ``cadence.bucket``.
"""

import pytest

from fuzzer_tool.core.cadence import bucket
from tests.test_regression_resume_state import _fuzzer

_SEEDS = [bytes([i]) * 32 for i in range(12)]


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)


@pytest.fixture
def fz(tmp_path):
    f = _fuzzer(tmp_path)
    f.corpus = list(_SEEDS)
    return f


def test_regression_prune_before_first_pick(fz):
    cm = fz._corpus_manager
    a, b = cm.seed_key(_SEEDS[0]), cm.seed_key(_SEEDS[1])
    fz._edge_tracker.find_near_duplicate_seeds = lambda max_hamming: [(a, b, 0.0)]
    cm.deprioritize_near_duplicates()
    assert fz._weight_cache is None  # pruning touched the cache first

    assert fz._seed_picker.weighted_pick_seed() in fz.corpus


def test_key_is_the_int_bucket(fz):
    fz._seed_picker.weighted_pick_seed()

    assert fz._weight_cache_key == bucket(fz.exec_count, 200, "seed_picker.weights")


def test_same_bucket_reuses_weights(fz):
    """Falsification: a second pick in the same bucket must not recompute."""
    fz._seed_picker.weighted_pick_seed()
    cached = fz._weight_cache

    fz._seed_picker.weighted_pick_seed()

    assert cached is not None
    assert fz._weight_cache is cached
