"""Seed weights recompute on corpus growth relative to its size.

A fixed +20 seeds forced a full O(corpus) weight pass about every 77 execs
while an FFmpeg corpus grew (104 passes × 23 ms in 8k execs, 6% of
fuzz_one). The trigger is now max(WEIGHT_GROWTH_MIN, WEIGHT_GROWTH_FRAC ×
corpus); the 200-exec interval is unchanged. Seeds admitted in between keep
weight 1.0 until the next pass, as before.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.services.seed_picker import (
    WEIGHT_GROWTH_FRAC,
    WEIGHT_GROWTH_MIN,
    weight_growth_due,
)
from tests.test_regression_resume_state import _fuzzer


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)


def _seeds(n: int, start: int = 0) -> list[bytes]:
    return [i.to_bytes(4, "little") * 8 for i in range(start, start + n)]


@pytest.fixture
def fz(tmp_path):
    f = _fuzzer(tmp_path)
    f.corpus = _seeds(1000)
    return f


def _passes(fz, monkeypatch) -> list[int]:
    calls = []
    real = fz._seed_picker._compute_weights
    monkeypatch.setattr(
        fz._seed_picker, "_compute_weights", lambda now: calls.append(1) or real(now)
    )
    return calls


def test_threshold_scales_with_corpus():
    big = 1000
    assert not weight_growth_due(big, big + WEIGHT_GROWTH_MIN)
    assert weight_growth_due(big, big + int(big * WEIGHT_GROWTH_FRAC))
    assert weight_growth_due(100, 100 + WEIGHT_GROWTH_MIN)  # floor holds
    assert not weight_growth_due(100, 100 + WEIGHT_GROWTH_MIN - 1)


def test_small_growth_on_a_large_corpus_reuses_weights(fz, monkeypatch):
    fz._seed_picker.weighted_pick_seed()
    calls = _passes(fz, monkeypatch)

    fz.corpus += _seeds(30, start=5000)
    fz._seed_picker.weighted_pick_seed()

    assert calls == []


def test_falsify_relative_growth_recomputes(fz, monkeypatch):
    fz._seed_picker.weighted_pick_seed()
    calls = _passes(fz, monkeypatch)

    fz.corpus += _seeds(int(1000 * WEIGHT_GROWTH_FRAC), start=5000)
    fz._seed_picker.weighted_pick_seed()

    assert calls == [1]


def test_adversarial_shrink_uses_the_same_threshold(fz, monkeypatch):
    """Pruning half the corpus must still invalidate (abs growth)."""
    fz._seed_picker.weighted_pick_seed()
    calls = _passes(fz, monkeypatch)

    fz.corpus = fz.corpus[:500]
    fz._seed_picker.weighted_pick_seed()

    assert calls == [1]
