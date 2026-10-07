"""--joint-liveness: pair probes over coverage-dead regions.

`LiveBitMaskEstimator` calls a region dead when mutations inside it alone
never move coverage. Block sensitivity can exceed single-bit sensitivity
(openai/math family 132), so a region can be dead alone and live together
with another. These tests pin the mechanism in `core/joint_liveness.py` and
its wiring in `OperatorEngine`, against a model of the paired-region target
(`tools/gen_synthetic_target.py --pair-len`): coverage moves only when BOTH
regions of the pair differ from the seed.

Falsification: with joint probing off (or the ledger never revoking), the
paired regions keep `_LIVENESS_DEAD_WEIGHT` forever, which is what the
first engine test asserts as the baseline. Control: an independent dead
pair (no joint effect) must stay dead, so "revoke everything" cannot pass.
"""

from __future__ import annotations

import inspect

import pytest
import xxhash

from fuzzer_tool.core.joint_liveness import JointLiveness
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import (
    _LIVENESS_DEAD_WEIGHT,
    _LIVENESS_SWITCH_AFTER,
    OperatorEngine,
)


class _MockFuzzer:
    def __init__(self, joint=None):
        self.max_len = 1 << 20
        self._rng = RandPool(seed=5)
        self._use_region_profile = True
        self._use_transfer_entropy = False
        self._te = None
        self._use_mi = False
        self._mi = None
        self._use_sensitivity = False
        self._sensitivity = None
        self._crash_mi = None
        self._joint_liveness = joint
        self._last_joint_probe = None


def _mixed_seed() -> bytes:
    """Four regions: two of zero filler, two of tabular u32 offsets."""
    filler = b"\x00" * 8192
    table = b"".join((i * 64).to_bytes(4, "little") for i in range(2048))
    return filler + table


BASE = {1, 2, 3}
JOINT_EDGE = 99


def _model_edges(seed: bytes, mutant: bytes, bounds, pair=(0, 2)) -> set:
    """The paired target: JOINT_EDGE appears iff both regions of *pair* differ."""
    changed = {
        i for i, (lo, hi) in enumerate(bounds) if seed[lo : min(hi, len(seed))] != mutant[lo:hi]
    }
    return BASE | ({JOINT_EDGE} if set(pair) <= changed else set())


def _dead_engine(joint):
    """Engine with every region of `_mixed_seed()` converged dead."""
    engine = OperatorEngine(_MockFuzzer(joint))
    seed = _mixed_seed()
    _c, bounds, _t = engine.region_weights(seed)
    for lo, _hi in bounds:
        for _ in range(_LIVENESS_SWITCH_AFTER + 1):
            engine.record_coverage_diff(seed, lo + 1, set(BASE), set(BASE))
    key = xxhash.xxh3_64_intdigest(seed)
    for idx in range(len(bounds)):
        assert engine._region_liveness_factor(seed, idx) == _LIVENESS_DEAD_WEIGHT
    return engine, seed, bounds, key


class TestJointLivenessLedger:
    def test_rejects_bad_parameters(self):
        rng = RandPool(seed=1)
        for kwargs in (
            {"rate": 0.0},
            {"rate": 1.5},
            {"probes_per_pair": 0},
            {"confirm_hits": 0},
            {"confirm_hits": 99},
            {"max_pairs": 0},
        ):
            with pytest.raises(ValueError):
                JointLiveness(rng, **kwargs)

    def test_pick_needs_two_dead_regions(self):
        jl = JointLiveness(RandPool(seed=1))
        assert jl.pick_pair(1, []) is None
        assert jl.pick_pair(1, [3]) is None

    def test_pick_returns_ordered_distinct_pair(self):
        jl = JointLiveness(RandPool(seed=1))
        for _ in range(50):
            a, b = jl.pick_pair(1, [0, 2, 5, 7])
            assert a < b
            assert {a, b} <= {0, 2, 5, 7}

    def test_one_hit_is_not_enough(self):
        """Unstable edges produce an occasional spurious 'moved'."""
        jl = JointLiveness(RandPool(seed=1), confirm_hits=2)
        assert jl.record(1, (0, 2), moved=True) is False
        assert not jl.is_revoked(1, 0)
        assert jl.record(1, (0, 2), moved=True) is True
        assert jl.is_revoked(1, 0) and jl.is_revoked(1, 2)
        assert not jl.is_revoked(1, 1)

    def test_revoked_regions_leave_the_candidate_pool(self):
        jl = JointLiveness(RandPool(seed=1), confirm_hits=1)
        jl.record(1, (0, 2), moved=True)
        # Only region 1 is left undecided: no pair can be formed.
        assert jl.pick_pair(1, [0, 1, 2]) is None

    def test_resolved_pair_is_not_redrawn(self):
        jl = JointLiveness(RandPool(seed=1), probes_per_pair=3, confirm_hits=2)
        for _ in range(3):
            jl.record(1, (0, 2), moved=False)
        assert all(jl.pick_pair(1, [0, 2]) is None for _ in range(20))

    def test_pair_cap_bounds_tracked_pairs(self):
        jl = JointLiveness(RandPool(seed=1), max_pairs=2, probes_per_pair=1, confirm_hits=1)
        for pair in ((0, 1), (0, 2)):
            jl.record(1, pair, moved=False)
        # Both tracked pairs are exhausted and no new pair is allowed.
        assert all(jl.pick_pair(1, [0, 1, 2, 3]) is None for _ in range(30))

    def test_drop_forgets_a_seed(self):
        jl = JointLiveness(RandPool(seed=1), confirm_hits=1)
        jl.record(1, (0, 2), moved=True)
        jl.drop(1)
        assert not jl.is_revoked(1, 0)
        jl.drop(1)  # idempotent


class TestEngineJointProbe:
    def test_off_by_default_no_probe_and_regions_stay_dead(self):
        engine, seed, bounds, _key = _dead_engine(joint=None)
        assert engine.joint_liveness_probe(seed) is None
        assert engine.record_joint_coverage_diff(1, (0, 2), set(BASE), BASE | {9}) is False
        assert all(
            engine._region_liveness_factor(seed, i) == _LIVENESS_DEAD_WEIGHT
            for i in range(len(bounds))
        )

    def test_no_probe_before_regions_are_dead(self):
        jl = JointLiveness(RandPool(seed=3), rate=1.0)
        engine = OperatorEngine(_MockFuzzer(jl))
        seed = _mixed_seed()
        engine.record_coverage_diff(seed, 100, set(BASE), set(BASE))  # 1 sample only
        assert engine.joint_liveness_probe(seed) is None

    def test_probe_touches_exactly_two_dead_regions_one_byte_each(self):
        jl = JointLiveness(RandPool(seed=3), rate=1.0)
        engine, seed, bounds, key = _dead_engine(jl)
        for _ in range(40):
            mutant = engine.joint_liveness_probe(seed)
            assert mutant is not None and len(mutant) == len(seed)
            diffs = [i for i in range(len(seed)) if seed[i] != mutant[i]]
            assert len(diffs) == 2
            regions = {next(r for r, (lo, hi) in enumerate(bounds) if lo <= d < hi) for d in diffs}
            assert len(regions) == 2
            published_key, pair = engine.f._last_joint_probe
            assert published_key == key and set(pair) == regions

    def test_paired_target_regions_regain_weight_and_others_stay_dead(self):
        """The end-to-end claim: dead alone, live together, found by probing."""
        jl = JointLiveness(RandPool(seed=3), rate=1.0)
        engine, seed, bounds, key = _dead_engine(jl)
        assert len(bounds) == 4

        # Baseline (what single-region probing alone leaves): both stay dead.
        for idx in (0, 2):
            assert engine._region_liveness_factor(seed, idx) == _LIVENESS_DEAD_WEIGHT

        for _ in range(3000):
            mutant = engine.joint_liveness_probe(seed)
            if mutant is None:
                break
            _k, pair = engine.f._last_joint_probe
            edges = _model_edges(seed, mutant, bounds)
            engine.record_joint_coverage_diff(key, pair, set(BASE), edges)

        assert engine._region_liveness_factor(seed, 0) == 1.0
        assert engine._region_liveness_factor(seed, 2) == 1.0
        # Control: regions with no joint effect are NOT revoked.
        assert engine._region_liveness_factor(seed, 1) == _LIVENESS_DEAD_WEIGHT
        assert engine._region_liveness_factor(seed, 3) == _LIVENESS_DEAD_WEIGHT
        assert jl.revoked_regions == 2

    def test_independent_dead_regions_stay_dead_and_probing_terminates(self):
        """Control: no joint effect anywhere -> nothing revoked, budget bounded."""
        jl = JointLiveness(RandPool(seed=3), rate=1.0, probes_per_pair=4)
        engine, seed, bounds, key = _dead_engine(jl)
        probes = 0
        for _ in range(500):
            mutant = engine.joint_liveness_probe(seed)
            if mutant is None:
                break
            probes += 1
            _k, pair = engine.f._last_joint_probe
            engine.record_joint_coverage_diff(key, pair, set(BASE), set(BASE))
        # 4 regions -> 6 pairs x 4 probes each, then the pool is exhausted.
        assert probes <= 6 * 4
        assert jl.revoked_regions == 0
        assert all(
            engine._region_liveness_factor(seed, i) == _LIVENESS_DEAD_WEIGHT
            for i in range(len(bounds))
        )

    def test_revoked_region_is_drawn_again_by_position_weighting(self):
        jl = JointLiveness(RandPool(seed=3), rate=1.0, confirm_hits=1)
        engine, seed, bounds, key = _dead_engine(jl)
        engine.record_joint_coverage_diff(key, (0, 2), set(BASE), BASE | {JOINT_EDGE})

        def share(region):
            lo, hi = bounds[region]
            hits = 0
            for _ in range(4000):
                pos = engine._region_weighted_position(seed, len(seed))
                hits += lo <= pos < hi
            return hits / 4000

        # Regions 0/2 are back at full weight, 1/3 sit at the dead weight.
        assert share(0) > 3 * share(1)
        assert share(2) > 3 * share(3)

    def test_eviction_drops_joint_state_with_the_layout(self):
        jl = JointLiveness(RandPool(seed=3), rate=1.0, confirm_hits=1)
        engine, seed, _bounds, key = _dead_engine(jl)
        engine.record_joint_coverage_diff(key, (0, 2), set(BASE), BASE | {JOINT_EDGE})
        assert jl.is_revoked(key, 0)
        engine._drop_region_liveness(key)
        assert not jl.is_revoked(key, 0)


class TestWiring:
    """Source pins, as test_delocalised_offset.py does for the same reason:
    reaching these lines through a real round needs a whole Fuzzer."""

    def test_probe_round_publishes_no_single_region_offset(self):
        src = inspect.getsource(OperatorEngine.mutate)
        hook = src[src.index("joint_mutant = self.joint_liveness_probe") :]
        # _reset_round_ops publishes offset 0 (region 0 would be credited);
        # the probe round must overwrite it with None before returning.
        assert hook.index("_reset_round_ops") < hook.index("_last_mutation_offset = None")
        assert hook.index("_last_mutation_offset = None") < hook.index("return joint_mutant")

    def test_stale_probe_is_cleared_at_the_top_of_every_round(self):
        src = inspect.getsource(OperatorEngine.mutate)
        assert src.index("f._last_joint_probe = None") < src.index(
            "det_mutant = self.maybe_deterministic_mutation"
        )

    def test_exec_loop_routes_probe_to_the_pair_ledger(self):
        from fuzzer_tool.services import fuzz_round

        src = inspect.getsource(fuzz_round)
        assert "record_joint_coverage_diff" in src
        # The pair branch returns before the single-region estimator runs.
        body = src[src.index("_joint_probe = getattr") :]
        assert body.index("return") < body.index("record_coverage_diff")

    def test_cli_flag_reaches_fuzzer(self):
        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert '"--joint-liveness"' in src
        assert 'joint_liveness=getattr(args, "joint_liveness", False)' in src
        assert "joint_liveness" not in commands._HAIL_MARY_FLAGS
