"""Regression: the ablation log must report the weight the picker used.

``_log_pick_signals`` recomputed a weight of its own instead of reading the
vector the pick was drawn from, and that recomputation had drifted from the
live formula: no temperature in the explore term or the burst factor, a fixed
staleness threshold of 50 rather than 50*T, no momentum, and none of the
phase-2 multipliers. The log's only purpose is attributing outcomes to
signals, so a number produced by no signal is worse than no number.
"""

import types

import pytest

from fuzzer_tool.services.seed_picker import SeedPicker


class _Fuzzer:
    def __init__(self, temperature=1.0, cached=None):
        self.corpus = [b"seed-one", b"seed-two"]
        self.seed_meta = {
            b"seed-one": {
                "fuzz_count": 4,
                "coverage_edges": 10,
                "added_at": 0.0,
                "momentum": 0.25,
            },
            b"seed-two": {
                "fuzz_count": 1,
                "coverage_edges": 1,
                "added_at": 0.0,
                "momentum": 0.0,
            },
        }
        self._ablation_file = object()  # truthy: logging enabled
        self._cached_weights = dict(cached or {})
        self._temperature = temperature
        self.markov_trained = False
        self.markov = types.SimpleNamespace(codelength_ratio=lambda d: 0.0)
        self._last_pick_signals: dict = {}

    def _seed_key(self, data: bytes) -> str:
        return data.decode()


@pytest.fixture
def picker():
    f = _Fuzzer()
    return f, SeedPicker(f)


class TestFinalWeightIsTheOneUsed:
    def test_final_w_comes_from_the_weight_vector(self, picker):
        f, p = picker
        weights = [123.456789, 0.5]
        p._log_pick_signals(b"seed-one", 120.0, weights)
        assert float(f._last_pick_signals["final_w"]) == pytest.approx(123.456789, rel=1e-6)

    def test_second_seed_reads_its_own_slot(self, picker):
        f, p = picker
        p._log_pick_signals(b"seed-two", 120.0, [1.0, 42.0])
        assert float(f._last_pick_signals["final_w"]) == pytest.approx(42.0, rel=1e-6)
        assert f._last_pick_signals["seed_idx"] == 1

    def test_short_weight_vector_falls_back_without_raising(self, picker):
        f, p = picker
        p._log_pick_signals(b"seed-two", 120.0, [1.0])
        assert float(f._last_pick_signals["final_w"]) > 0.0


class TestScalarTermsMatchTheLivePath:
    """base_w and burst must come from _weight_exploit_parts, the same
    helper the vectorised path in _compute_weights mirrors."""

    @pytest.mark.parametrize("temperature", [0.2, 1.0, 2.5])
    def test_base_and_burst_match_the_helper(self, temperature):
        f = _Fuzzer(temperature=temperature)
        p = SeedPicker(f)
        now = 300.0
        meta = f.seed_meta[b"seed-one"]
        age = now - meta["added_at"]
        expected_w, expected_burst = p._weight_exploit_parts(
            meta, max(meta["fuzz_count"], 1), meta["coverage_edges"], age, temperature
        )
        p._log_pick_signals(b"seed-one", now, [1.0, 1.0])
        ps = f._last_pick_signals
        assert float(ps["base_w"]) == pytest.approx(expected_w, abs=1e-4)
        assert float(ps["burst"]) == pytest.approx(expected_burst, abs=1e-2)

    def test_temperature_reaches_the_burst_factor(self):
        """The old recomputation used max(1, 5 - age/60) with no T at all, so
        burst read identically at every temperature."""
        seen = set()
        for temperature in (0.1, 1.0, 4.0):
            f = _Fuzzer(temperature=temperature)
            p = SeedPicker(f)
            p._log_pick_signals(b"seed-one", 60.0, [1.0, 1.0])
            seen.add(f._last_pick_signals["burst"])
        assert len(seen) == 3

    def test_momentum_reaches_base_w(self):
        f_hot = _Fuzzer()
        f_cold = _Fuzzer()
        f_cold.seed_meta[b"seed-one"]["momentum"] = 0.0
        for f in (f_hot, f_cold):
            SeedPicker(f)._log_pick_signals(b"seed-one", 60.0, [1.0, 1.0])
        assert f_hot._last_pick_signals["base_w"] != f_cold._last_pick_signals["base_w"]

    def test_staleness_threshold_scales_with_temperature(self):
        # fuzz_count / (coverage + 1) = 60 / 2 = 30: stale at T=0.2
        # (threshold 10), not stale at T=1.0 (threshold 50).
        penalties = {}
        for temperature in (0.2, 1.0):
            f = _Fuzzer(temperature=temperature)
            f.seed_meta[b"seed-one"]["fuzz_count"] = 60
            f.seed_meta[b"seed-one"]["coverage_edges"] = 1
            SeedPicker(f)._log_pick_signals(b"seed-one", 10.0, [1.0, 1.0])
            penalties[temperature] = float(f._last_pick_signals["penalty"])
        assert penalties[0.2] == pytest.approx(0.01)
        assert penalties[1.0] == pytest.approx(1.0)

    def test_temperature_is_reported(self, picker):
        f, p = picker
        p._log_pick_signals(b"seed-one", 60.0, [1.0, 1.0])
        assert float(f._last_pick_signals["temperature"]) == pytest.approx(1.0)


class TestMissingCacheEntry:
    def test_uncached_seed_does_not_raise(self, picker):
        """Adversarial: the default was a 3-tuple while the recombination
        indexes cached[3], so any seed absent from _cached_weights raised
        IndexError -- reachable whenever the saturation gate flushes the
        cache, and only with ablation logging on."""
        f, p = picker
        assert f._cached_weights == {}
        p._log_pick_signals(b"seed-one", 60.0, None)
        assert float(f._last_pick_signals["final_w"]) > 0.0

    def test_cached_entry_is_reported(self):
        f = _Fuzzer(cached={"seed-one": (0.25, 2.0, 1.5, 0.75)})
        SeedPicker(f)._log_pick_signals(b"seed-one", 60.0, [1.0, 1.0])
        ps = f._last_pick_signals
        assert float(ps["subsumption"]) == pytest.approx(0.25)
        assert float(ps["diversity"]) == pytest.approx(2.0)
        assert float(ps["spatial"]) == pytest.approx(1.5)


class TestLoggingStaysOptional:
    def test_no_ablation_file_writes_nothing(self, picker):
        f, p = picker
        f._ablation_file = None
        f._last_pick_signals = {}
        p._log_pick_signals(b"seed-one", 60.0, [1.0, 1.0])
        assert f._last_pick_signals == {}

    def test_unknown_seed_writes_nothing(self, picker):
        f, p = picker
        f._last_pick_signals = {}
        p._log_pick_signals(b"not-in-corpus", 60.0, [1.0, 1.0])
        assert f._last_pick_signals == {}
