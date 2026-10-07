"""PositionLevyScheduler: heavy-tailed jumps around a seed's last gain offset.

Covers core/schedulers/pos_levy.py. Draw order inside ``propose`` is fixed and
the scripted tests rely on it: spark check, then u (step size), then sign.
"""

import json

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_levy import (
    GAP_RING,
    MAX_SEEDS,
    SPARK_RATE,
    STALE,
    PositionLevyScheduler,
    _reflect,
)

SEED = bytes(1000)
NO_SPARK = 0.99  # random() draw above SPARK_RATE
PLUS = 0.1  # sign draw: below 0.5 -> +1
MINUS = 0.9  # sign draw: at/above 0.5 -> -1


class ScriptedRng:
    """Deterministic stand-in: scripted random(), randint -> lower bound."""

    def __init__(self, randoms=()):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_SPARK

    def randint(self, a, b):
        return a


def _lv(*randoms):
    return PositionLevyScheduler(ScriptedRng(randoms))


def _gain(lv, *offsets, data=SEED):
    lv.record(data, list(offsets), Outcome.GAIN, 1.0)


def _propose(lv, u, sign, buf_len=1000, data=SEED):
    """One proposal with a scripted step-size draw *u* and sign draw *sign*."""
    lv._rng._randoms[:] = [NO_SPARK, u, sign]
    return lv.propose(data, buf_len)


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_lv(), PositionScheduler)

    def test_name(self):
        assert PositionLevyScheduler.name == "levy"


class TestReflect:
    @pytest.mark.parametrize(
        ("pos", "last", "want"),
        [
            (5, 9, 5),
            (0, 9, 0),
            (9, 9, 9),
            (-1, 9, 1),
            (-3, 9, 3),
            (10, 9, 8),
            (12, 9, 6),
            (18, 9, 0),
            (19, 9, 1),
            (-19, 9, 1),
            (7, 0, 0),
            (-7, 0, 0),
        ],
    )
    def test_triangle_fold(self, pos, last, want):
        assert _reflect(pos, last) == want

    def test_always_in_range(self):
        for last in (1, 2, 9, 255):
            for pos in range(-3 * last - 5, 3 * last + 6):
                assert 0 <= _reflect(pos, last) <= last


class TestColdAndAnchor:
    def test_cold_seed_declines(self):
        assert _lv().propose(SEED, len(SEED)) is None

    def test_miss_on_unknown_seed_creates_no_walk(self):
        lv = _lv()
        lv.record(SEED, [10], Outcome.MISS, 0.0)
        assert lv.seed_count() == 0

    def test_gain_sets_the_anchor(self):
        lv = _lv()
        _gain(lv, 123)
        assert lv.anchor(SEED) == 123

    def test_newer_gain_replaces_the_anchor(self):
        lv = _lv()
        _gain(lv, 123)
        _gain(lv, 400)
        assert lv.anchor(SEED) == 400

    def test_gain_without_valid_offsets_creates_no_walk(self):
        lv = _lv()
        _gain(lv)
        _gain(lv, -1, -50)
        assert lv.seed_count() == 0

    def test_negative_offsets_are_ignored(self):
        lv = _lv()
        _gain(lv, -5, 7)
        assert lv.anchor(SEED) == 7

    def test_empty_seed_is_ignored(self):
        lv = _lv()
        lv.record(b"", [0], Outcome.GAIN, 1.0)
        assert lv.seed_count() == 0

    def test_offsets_past_the_seed_end_are_accepted(self):
        lv = _lv()
        _gain(lv, len(SEED) + 500)  # the child grew
        assert lv.anchor(SEED) == len(SEED) + 500

    def test_multi_offset_gain_anchors_on_one_of_them(self):
        offsets = [3, 50, 700]
        for seed in range(20):
            lv = PositionLevyScheduler(RandPool(seed=seed))
            _gain(lv, *offsets)
            assert lv.anchor(SEED) in offsets

    def test_multi_offset_gain_can_pick_each_offset(self):
        offsets = [3, 50, 700]
        seen = set()
        for seed in range(60):
            lv = PositionLevyScheduler(RandPool(seed=seed))
            _gain(lv, *offsets)
            seen.add(lv.anchor(SEED))
        assert seen == set(offsets)

    def test_weight_does_not_change_the_anchor(self):
        lv = _lv()
        lv.record(SEED, [77], Outcome.GAIN, 0.25)
        assert lv.anchor(SEED) == 77


class TestStaleness:
    def test_miss_streak_drops_the_anchor(self):
        lv = _lv()
        _gain(lv, 100)
        for _ in range(STALE):
            lv.record(SEED, [5], Outcome.MISS, 0.0)
        assert lv.anchor(SEED) is None
        assert lv.propose(SEED, len(SEED)) is None

    def test_one_short_of_stale_keeps_the_anchor(self):
        lv = _lv()
        _gain(lv, 100)
        for _ in range(STALE - 1):
            lv.record(SEED, [5], Outcome.MISS, 0.0)
        assert lv.anchor(SEED) == 100

    def test_gain_resets_the_streak(self):
        lv = _lv()
        _gain(lv, 100)
        for _ in range(STALE - 1):
            lv.record(SEED, [5], Outcome.MISS, 0.0)
        _gain(lv, 100)
        for _ in range(STALE - 1):
            lv.record(SEED, [5], Outcome.MISS, 0.0)
        assert lv.anchor(SEED) == 100

    def test_gain_after_stale_reanchors(self):
        lv = _lv()
        _gain(lv, 100)
        for _ in range(STALE):
            lv.record(SEED, [5], Outcome.MISS, 0.0)
        _gain(lv, 640)
        assert lv.anchor(SEED) == 640
        assert lv.walk_state(SEED)[1] == 0

    def test_misses_after_stale_do_not_count_against_a_dropped_anchor(self):
        lv = _lv()
        _gain(lv, 100)
        for _ in range(STALE + 10):
            lv.record(SEED, [5], Outcome.MISS, 0.0)
        assert lv.walk_state(SEED)[1] == 0

    def test_staleness_is_per_seed(self):
        lv = _lv()
        other = bytes(range(200))
        _gain(lv, 100)
        _gain(lv, 50, data=other)
        for _ in range(STALE):
            lv.record(SEED, [5], Outcome.MISS, 0.0)
        assert lv.anchor(SEED) is None
        assert lv.anchor(other) == 50


class TestPropose:
    def test_high_u_lands_on_the_anchor_itself(self):
        # step = floor(1/u - 1): u=0.9 -> 0. A plain Pareto floor(1/u) could
        # never re-propose the byte that gained.
        lv = _lv()
        _gain(lv, 500)
        assert _propose(lv, 0.9, PLUS) == 500

    @pytest.mark.parametrize(("u", "step"), [(0.9, 0), (0.4, 1), (0.2, 4), (0.011, 89)])
    def test_step_follows_the_lomax_quantile(self, u, step):
        lv = _lv()
        _gain(lv, 500)
        assert _propose(lv, u, PLUS) == 500 + step
        assert _propose(lv, u, MINUS) == 500 - step

    def test_sign_draw_picks_the_direction(self):
        lv = _lv()
        _gain(lv, 500)
        assert _propose(lv, 0.4, PLUS) == 501
        assert _propose(lv, 0.4, MINUS) == 499

    def test_zero_u_is_guarded_and_capped_at_the_buffer(self):
        # random() == 0.0 would be an infinite step; the floor and the buf_len
        # cap turn it into "as far as the buffer goes", then reflection folds
        # it back inside.
        lv = _lv()
        _gain(lv, 100)
        pos = _propose(lv, 0.0, PLUS)
        assert pos == 898  # 100 + 1000 = 1100, folded over last=999
        assert 0 <= pos <= 999

    def test_overshoot_reflects_off_the_low_end(self):
        lv = _lv()
        _gain(lv, 0)
        assert _propose(lv, 0.25, MINUS) == 3  # step 3 below 0 -> 3, not a pile at 0

    def test_overshoot_reflects_off_the_high_end(self):
        lv = _lv()
        _gain(lv, 999)
        assert _propose(lv, 0.25, PLUS) == 996

    def test_position_is_clamped_to_a_shrunken_buffer(self):
        lv = _lv()
        _gain(lv, 900)
        assert _propose(lv, 0.9, PLUS, buf_len=100) == 99

    def test_anchor_past_the_buffer_end_is_clamped_first(self):
        lv = _lv()
        _gain(lv, len(SEED) + 500)
        assert 0 <= _propose(lv, 0.4, PLUS, buf_len=len(SEED)) <= len(SEED) - 1

    def test_empty_buffer_declines(self):
        lv = _lv()
        _gain(lv, 10)
        assert lv.propose(SEED, 0) is None

    def test_one_byte_buffer(self):
        lv = _lv()
        _gain(lv, 10)
        assert _propose(lv, 0.2, PLUS, buf_len=1) == 0

    def test_spark_escapes_the_anchor(self):
        lv = _lv()
        _gain(lv, 500)
        lv._rng._randoms[:] = [SPARK_RATE / 2]
        assert lv.propose(SEED, len(SEED)) == 0  # ScriptedRng.randint -> lower bound

    def test_spark_boundary_is_exclusive(self):
        lv = _lv()
        _gain(lv, 500)
        lv._rng._randoms[:] = [SPARK_RATE, 0.9, PLUS]
        assert lv.propose(SEED, len(SEED)) == 500  # random() == SPARK_RATE: no spark

    def test_propose_does_not_touch_the_state(self):
        lv = _lv()
        _gain(lv, 500)
        before = lv.walk_state(SEED)
        for _ in range(10):
            _propose(lv, 0.3, MINUS)
        assert lv.walk_state(SEED) == before

    def test_never_leaves_the_buffer_over_many_draws(self):
        for buf_len in (1, 2, 7, 100, 1000):
            lv = PositionLevyScheduler(RandPool(seed=buf_len))
            _gain(lv, 0)
            _gain(lv, buf_len // 2)
            for _ in range(500):
                pos = lv.propose(SEED, buf_len)
                assert pos is not None
                assert 0 <= pos < buf_len


class TestTail:
    """Statistical shape with a real, seeded RandPool (deterministic)."""

    N = 20000
    ANCHOR = 5000
    BUF = 10000

    def _distances(self):
        lv = PositionLevyScheduler(RandPool(seed=7))
        seed = bytes(self.BUF)
        _gain(lv, self.ANCHOR, data=seed)
        return [abs(lv.propose(seed, self.BUF) - self.ANCHOR) for _ in range(self.N)]

    def test_most_mass_is_near_the_anchor(self):
        d = self._distances()
        near = sum(1 for x in d if x <= 2) / self.N
        # (1 - SPARK_RATE) * P(step <= 2) = 0.95 * (1 - 1/3) ~ 0.633
        assert 0.55 <= near <= 0.72

    def test_mass_on_the_anchor_byte_itself_is_about_half(self):
        d = self._distances()
        on = sum(1 for x in d if x == 0) / self.N
        assert 0.42 <= on <= 0.53  # 0.95 * 0.5 ~ 0.475

    def test_tail_reaches_far_beyond_a_gaussian_kernel(self):
        d = self._distances()
        far = sum(1 for x in d if x > 100) / self.N
        # 0.95 * P(step >= 101) + spark mass ~ 0.0093 + 0.049 ~ 0.058. A
        # Gaussian kernel of sigma 3 would put ~0 here outside the spark.
        assert 0.03 <= far <= 0.09
        assert sum(1 for x in d if 100 < x <= 1000) / self.N > 0.008


class TestGaps:
    def test_gaps_record_anchor_to_anchor_distance(self):
        lv = _lv()
        _gain(lv, 10)
        _gain(lv, 30)
        _gain(lv, 25)
        assert lv.walk_state(SEED)[2] == [20, 5]

    def test_first_gain_records_no_gap(self):
        lv = _lv()
        _gain(lv, 10)
        assert lv.walk_state(SEED)[2] == []

    def test_gap_after_a_dropped_anchor_is_not_recorded(self):
        lv = _lv()
        _gain(lv, 10)
        for _ in range(STALE):
            lv.record(SEED, [5], Outcome.MISS, 0.0)
        _gain(lv, 900)
        assert lv.walk_state(SEED)[2] == []

    def test_gap_ring_is_bounded_and_keeps_the_newest(self):
        lv = _lv()
        _gain(lv, 0)
        for i in range(1, GAP_RING + 30):
            _gain(lv, i * 2)
        gaps = lv.walk_state(SEED)[2]
        assert len(gaps) == GAP_RING
        assert set(gaps) == {2}

    def test_walk_state_returns_a_copy(self):
        lv = _lv()
        _gain(lv, 10)
        _gain(lv, 30)
        lv.walk_state(SEED)[2].append(999)
        assert lv.walk_state(SEED)[2] == [20]

    def test_walk_state_unknown_seed(self):
        assert _lv().walk_state(SEED) is None
        assert _lv().anchor(SEED) is None


class TestLRU:
    def test_seed_table_is_lru_bounded(self):
        lv = _lv()
        seeds = [i.to_bytes(4, "big") + bytes(10) for i in range(MAX_SEEDS + 5)]
        for s in seeds:
            _gain(lv, 3, data=s)
        assert lv.seed_count() == MAX_SEEDS
        assert lv.anchor(seeds[0]) is None
        assert lv.anchor(seeds[-1]) == 3

    def test_recent_use_protects_a_seed_from_eviction(self):
        lv = _lv()
        seeds = [i.to_bytes(4, "big") + bytes(10) for i in range(MAX_SEEDS + 1)]
        for s in seeds[:MAX_SEEDS]:
            _gain(lv, 3, data=s)
        _gain(lv, 4, data=seeds[0])  # refresh the oldest
        _gain(lv, 3, data=seeds[MAX_SEEDS])  # forces one eviction
        assert lv.anchor(seeds[0]) == 4
        assert lv.anchor(seeds[1]) is None


class TestPersistence:
    def _populated(self):
        lv = _lv()
        _gain(lv, 100)
        _gain(lv, 160)
        lv.record(SEED, [1], Outcome.MISS, 0.0)
        _gain(lv, 7, data=b"other-seed-bytes")
        return lv

    def test_round_trip_preserves_state(self):
        a = self._populated()
        b = _lv()
        b.from_dict(a.to_dict())
        assert b.walk_state(SEED) == (160, 1, [60])
        assert b.anchor(b"other-seed-bytes") == 7

    def test_round_trip_through_json(self):
        # state_store is JSON-backed: int keys become str, tuples become lists.
        a = self._populated()
        b = _lv()
        b.from_dict(json.loads(json.dumps(a.to_dict())))
        assert b.walk_state(SEED) == (160, 1, [60])

    def test_dropped_anchor_survives_a_round_trip(self):
        a = _lv()
        _gain(a, 100)
        for _ in range(STALE):
            a.record(SEED, [5], Outcome.MISS, 0.0)
        b = _lv()
        b.from_dict(json.loads(json.dumps(a.to_dict())))
        assert b.walk_state(SEED) == (None, 0, [])

    @pytest.mark.parametrize(
        "payload",
        [
            {"version": 999, "walks": {}},
            {"walks": {}},
            {"version": 1},
            {"version": 1, "walks": "nope"},
            {"version": 1, "walks": {"1": [5, 0]}},
            {"version": 1, "walks": {"x": [5, 0, []]}},
            {"version": 1, "walks": {"1": [-5, 0, []]}},
            {"version": 1, "walks": {"1": [5, -1, []]}},
            {"version": 1, "walks": {"1": [5, 0, [-3]]}},
            {"version": 1, "walks": {"1": ["abc", 0, []]}},
            [1, 2, 3],
            "garbage",
            42,
        ],
    )
    def test_malformed_state_resets_cleanly(self, payload):
        lv = self._populated()
        lv.from_dict(payload)
        assert lv.seed_count() == 0

    def test_empty_state_is_a_no_op(self):
        lv = _lv()
        lv.from_dict({})
        lv.from_dict(None)
        assert lv.seed_count() == 0

    def test_restore_respects_the_lru_cap(self):
        walks = {i: (3, 0, []) for i in range(MAX_SEEDS + 10)}
        lv = _lv()
        lv.from_dict({"version": 1, "walks": walks})
        assert lv.seed_count() == MAX_SEEDS

    def test_restore_trims_an_oversized_gap_ring(self):
        lv = _lv()
        lv.from_dict({"version": 1, "walks": {7: (3, 0, list(range(GAP_RING + 20)))}})
        key = next(iter(lv._walks))
        assert len(lv._walks[key].gaps) == GAP_RING

    def test_restore_replaces_existing_state(self):
        lv = self._populated()
        lv.from_dict({"version": 1, "walks": {}})
        assert lv.seed_count() == 0
