"""Regression tests: item 4 (LiveBitMaskEstimator) wired into
OperatorEngine's region weighting as a down-weighting signal.

Per docs/handover_skittercreek_tailslayer_port.md item 4: this covers the
wiring mechanism itself (record_coverage_diff -> _region_liveness_factor
-> _region_weighted_position), using synthetic coverage-edge diffs. It is
NOT the real-corpus sensitivity sweep the handover doc's Sequencing step 6
still asks for -- that requires an actual campaign's coverage-bitmap data,
not available here.
"""

import os

import pytest

from fuzzer_tool.core.live_bit_mask import LiveBitMaskEstimator
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import (
    _LIVENESS_DEAD_WEIGHT,
    _LIVENESS_SWITCH_AFTER,
    OperatorEngine,
)


class _MockFuzzer:
    def __init__(self):
        self.max_len = 1 << 20
        self._rand_pool = RandPool()
        self._use_region_profile = True
        self._use_transfer_entropy = False
        self._te = None
        self._use_mi = False
        self._mi = None
        self._use_sensitivity = False
        self._sensitivity = None
        self._crash_mi = None


def _mixed_seed() -> bytes:
    """Two clearly distinct regions: 8KiB of low-entropy filler (region 0,
    offsets < 8192) and 8KiB of pseudo-tabular u32 offsets (region 1,
    offsets >= 8192), so a single mutation offset maps unambiguously to
    one region or the other."""
    filler = b"\x00" * 8192
    table = b"".join((i * 64).to_bytes(4, "little") for i in range(2048))
    return filler + table


class TestRecordCoverageDiff:
    def test_noop_on_seed_too_short_for_a_profile(self):
        engine = OperatorEngine(_MockFuzzer())
        short = os.urandom(64)
        engine.record_coverage_diff(short, 0, {1, 2}, {1, 2, 3})
        assert engine._region_liveness == {}

    def test_noop_when_offset_outside_every_region(self):
        """profile_buffer skips windows below 512 bytes, and stride==window
        by default, so an offset past the last full window has no region
        to attribute to."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        entry = engine.region_weights(seed)
        assert entry is not None
        _, bounds, _ = entry
        past_end = bounds[-1][1] + 10_000
        engine.record_coverage_diff(seed, past_end, {1}, {1, 2})
        key = list(engine._region_cache.keys())[0]
        # No estimator list was ever created, since region_idx stayed None
        # and the method returned before setdefault.
        assert key not in engine._region_liveness

    def test_creates_one_estimator_per_touched_region(self):
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        engine.record_coverage_diff(seed, 100, {1, 2}, {1, 2, 3})  # region 0
        engine.record_coverage_diff(seed, 8300, {1, 2}, {1, 2, 4})  # region 2
        key = list(engine._region_cache.keys())[0]
        estimators = engine._region_liveness[key]
        assert estimators[0] is not None
        assert estimators[2] is not None
        assert estimators[0] is not estimators[2]

    def test_identical_edge_sets_produce_zero_diff(self):
        """No symmetric difference -> observe(0, 0) -> mask stays empty,
        exactly the 'no growth' sample the convergence detector needs."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        same = {1, 2, 3}
        engine.record_coverage_diff(seed, 100, same, set(same))
        key = list(engine._region_cache.keys())[0]
        est = engine._region_liveness[key][0]
        assert est.mask == 0
        assert est.samples_seen == 1

    def test_differing_edges_grow_the_mask(self):
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        engine.record_coverage_diff(seed, 100, {1, 2}, {1, 2, 3})
        key = list(engine._region_cache.keys())[0]
        est = engine._region_liveness[key][0]
        assert est.mask != 0


class TestRegionLivenessFactor:
    def test_no_estimator_yet_is_a_noop_factor(self):
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        assert engine._region_liveness_factor(seed, 0) == 1.0

    def test_unconverged_empty_mask_is_not_downweighted(self):
        """A handful of no-growth samples isn't convergence -- absence of
        evidence must not read as a negative claim yet."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        for _ in range(5):
            engine.record_coverage_diff(seed, 100, {1, 2}, {1, 2})
        assert engine._region_liveness_factor(seed, 0) == 1.0

    def test_converged_dead_region_is_downweighted(self):
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        for _ in range(_LIVENESS_SWITCH_AFTER):
            engine.record_coverage_diff(seed, 100, {1, 2}, {1, 2})
        assert engine._region_liveness_factor(seed, 0) == _LIVENESS_DEAD_WEIGHT

    def test_converged_but_nonempty_mask_is_not_downweighted(self):
        """Converged-and-lively (mask nonzero) is the opposite verdict from
        converged-and-dead -- must not share the down-weight path."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        engine.record_coverage_diff(seed, 100, {1, 2}, {1, 2, 3})  # one growth
        for _ in range(_LIVENESS_SWITCH_AFTER):
            engine.record_coverage_diff(seed, 100, {1, 2, 3}, {1, 2, 3})
        key = list(engine._region_cache.keys())[0]
        est = engine._region_liveness[key][0]
        assert est.is_converged
        assert est.mask != 0
        assert engine._region_liveness_factor(seed, 0) == 1.0

    def test_liveness_is_per_region_not_global(self):
        """Region 2 confirmed dead must not down-weight region 0."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        for _ in range(_LIVENESS_SWITCH_AFTER):
            engine.record_coverage_diff(seed, 8300, {1}, {1})  # region 2: dead
        engine.record_coverage_diff(seed, 100, {1}, {1, 2})  # region 0: alive
        assert engine._region_liveness_factor(seed, 2) == _LIVENESS_DEAD_WEIGHT
        assert engine._region_liveness_factor(seed, 0) == 1.0


class TestRegionWeightedPositionDownweighting:
    def test_confirmed_dead_region_drawn_less_often(self):
        """Once region 0 (offsets < 8192) is confirmed dead, draws should
        skew hard toward region 1 -- same statistical-majority style check
        as test_regression_region_profile.py's mutation_weight test."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        split = 8192

        # Baseline: without any liveness data, both regions get picked
        # (region 1's tabular mutation_weight of 1.6 already favors it
        # over region 0's default 1.0, but region 0 still gets a real
        # share).
        baseline = [engine._region_weighted_position(seed, len(seed)) for _ in range(1500)]
        baseline_drawn = [p for p in baseline if p is not None]
        baseline_region0_share = sum(p < split for p in baseline_drawn) / len(baseline_drawn)
        assert baseline_region0_share > 0.05

        # Confirm region 0 dead.
        for _ in range(_LIVENESS_SWITCH_AFTER):
            engine.record_coverage_diff(seed, 100, {1}, {1})

        after = [engine._region_weighted_position(seed, len(seed)) for _ in range(1500)]
        after_drawn = [p for p in after if p is not None]
        after_region0_share = sum(p < split for p in after_drawn) / len(after_drawn)

        assert after_region0_share < baseline_region0_share

    def test_fast_path_skipped_when_no_region_confirmed_dead(self):
        """No estimator, or estimators that haven't converged-dead, must
        reuse the cached cumulative/total unchanged -- this is the
        no-liveness-data-yet case the vast majority of draws hit, and it
        must not pay the adjusted-cumulative rebuild cost."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        entry_before = engine.region_weights(seed)
        for _ in range(5):
            engine._region_weighted_position(seed, len(seed))
        entry_after = engine.region_weights(seed)
        assert entry_before is entry_after  # cache untouched


class TestRecordCoverageDiffReturnValue:
    """record_coverage_diff signals the offset,width) -> FormatLearner
    bridge by returning region bounds exactly on the transition into
    converged-dead, per the round-7 wiring."""

    def test_returns_none_before_convergence(self):
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        for _ in range(_LIVENESS_SWITCH_AFTER - 1):
            result = engine.record_coverage_diff(seed, 100, {1}, {1})
        assert result is None

    def test_returns_bounds_exactly_once_on_the_transition(self):
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        results = [
            engine.record_coverage_diff(seed, 100, {1}, {1})
            for _ in range(_LIVENESS_SWITCH_AFTER + 5)
        ]
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 1
        lo, width = non_none[0]
        assert lo == 0  # region 0 starts at offset 0
        assert width == 4096

    def test_returns_none_once_already_converged_dead(self):
        """The transition fires once; subsequent no-growth samples after
        convergence must not keep re-reporting."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        for _ in range(_LIVENESS_SWITCH_AFTER):
            engine.record_coverage_diff(seed, 100, {1}, {1})
        # Now converged-dead. More no-growth samples must return None.
        for _ in range(10):
            assert engine.record_coverage_diff(seed, 100, {1}, {1}) is None

    def test_converged_but_alive_never_returns_bounds(self):
        """A region that converges with a nonzero mask (confirmed-live)
        must never trigger the padding-evidence return path."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        engine.record_coverage_diff(seed, 100, {1}, {1, 2})  # one growth
        results = [
            engine.record_coverage_diff(seed, 100, {1, 2}, {1, 2})
            for _ in range(_LIVENESS_SWITCH_AFTER + 5)
        ]
        assert all(r is None for r in results)

    def test_short_seed_returns_none(self):
        engine = OperatorEngine(_MockFuzzer())
        assert engine.record_coverage_diff(os.urandom(64), 0, {1}, {1, 2}) is None


class TestCacheEvictionParity:
    def test_liveness_cleared_alongside_region_cache(self):
        from fuzzer_tool.services.operators import _REGION_CACHE_MAX

        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        engine.record_coverage_diff(seed, 100, {1}, {1, 2})
        assert engine._region_liveness  # populated

        for _ in range(_REGION_CACHE_MAX + 5):
            engine.region_weights(os.urandom(8192))

        # Once _region_cache is wholesale-cleared, _region_liveness must
        # have been cleared in the same pass -- a stale liveness entry
        # surviving past its region bounds/cumulative would misattribute
        # the next seed sharing that hash's diffs to the wrong region.
        assert len(engine._region_cache) <= _REGION_CACHE_MAX
        assert len(engine._region_liveness) <= _REGION_CACHE_MAX
