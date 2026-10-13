"""SeedPicker._weight_cached spreads edge analyses over several passes."""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.services import seed_picker as sp
from fuzzer_tool.services.seed_picker import SeedPicker


class _Tracker:
    def __init__(self, keys):
        self.seed_edges = {k: {1} for k in keys}
        self.computed: list[str] = []

    def compute_subsumption_weight(self, key):
        self.computed.append(key)
        return 2.0

    def compute_hitcount_diversity_weight(self, _key):
        return 1.0

    def compute_wasserstein_weight(self, _key):
        return 1.0

    def compute_coverage_proximity(self, _key):
        return 0.5


def _picker(keys):
    tracker = _Tracker(keys)
    f = SimpleNamespace(_cached_weights={}, _edge_tracker=tracker, _saturation_gated=False)
    picker = SeedPicker.__new__(SeedPicker)
    picker.f = f
    return picker, f, tracker


def test_fill_budget_defers_without_caching() -> None:
    keys = [f"k{i}" for i in range(5)]
    picker, f, tracker = _picker(keys)
    picker._weight_fill_left = 2
    out = [picker._weight_cached(k, 1.0, {}, f)[0] for k in keys]
    assert tracker.computed == ["k0", "k1"]  # budget spent after two
    assert set(f._cached_weights) == {"k0", "k1"}  # deferred seeds are not cached
    assert out[0] == out[1] == 2.0 * 1.0  # (sub * div * spa) * (0.5 + cov)
    assert out[2] == out[3] == out[4] == 1.0  # neutral for this pass

    picker._weight_fill_left = 2  # next pass reaches the next seeds
    for k in keys:
        picker._weight_cached(k, 1.0, {}, f)
    assert tracker.computed == ["k0", "k1", "k2", "k3"]


def test_default_budget_is_unlimited() -> None:
    keys = [f"k{i}" for i in range(50)]
    picker, f, tracker = _picker(keys)
    for k in keys:
        picker._weight_cached(k, 1.0, {}, f)
    assert len(tracker.computed) == 50


def test_budget_constants_are_sane() -> None:
    assert 0 < sp.WEIGHT_FILL_PER_PASS < sp.LAZY_WEIGHT_MIN_CORPUS
