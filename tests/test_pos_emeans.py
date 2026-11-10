"""PositionEMeansScheduler: offsets from an EMA-trained gain codebook.

Covers core/schedulers/pos_emeans.py.
"""

import math

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import MAX_SEEDS
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_emeans import (
    ALPHA0,
    ALPHA_MIN_FACTOR,
    HORIZON,
    INIT_USAGE,
    REVIVE_EVERY,
    K,
    PositionEMeansScheduler,
    scheduled_alpha,
)

SEED = bytes(1000)  # floor = max(1, 1000 / 256) = 3.9, leader radius 7.8


class ScriptedRng:
    """random() above EXPLORE (never escapes) and 0.5 into the mass cumsum; gauss -> 0."""

    def random(self):
        return 0.5

    def randint(self, a, b):
        return a

    def gauss(self, mu=0.0, sigma=1.0):
        return mu


def _sched(rng=None, **kw):
    return PositionEMeansScheduler(rng or ScriptedRng(), **kw)


def _gain(s, offsets, data=SEED):
    s.record(data, offsets if isinstance(offsets, list) else [offsets], Outcome.GAIN)


def test_satisfies_protocol():
    s = _sched()
    assert isinstance(s, PositionScheduler)
    assert s.name == "emeans"


def test_declines_before_a_gain_and_misses_carry_no_point():
    s = _sched()
    assert s.propose(SEED, len(SEED)) is None
    s.record(SEED, [100], Outcome.MISS)
    assert s.propose(SEED, len(SEED)) is None
    assert s.centroids(SEED) is None


def test_degenerate_inputs():
    s = _sched()
    _gain(s, 100)
    assert s.propose(SEED, 0) is None
    assert s.propose(b"", 10) is None
    s.record(b"", [1], Outcome.GAIN)
    s.record(SEED, [-1], Outcome.GAIN)  # negative offsets are not points
    assert len(s.centroids(SEED)) == 1


def test_first_gain_snaps_a_slot_to_the_point():
    s = _sched()
    _gain(s, 700)
    rows = s.centroids(SEED)
    assert rows[0][0] == pytest.approx(700, abs=1e-2)
    assert s.propose(SEED, len(SEED)) == 700  # gauss -> 0, one slot


def test_single_slot_follows_the_ema_recurrence():
    # Points within the leader radius share slot 0; compare to the closed form.
    s = _sched()
    pts = [100, 104, 102, 106, 101, 105, 103, 100, 104, 102]
    c, n = None, 0.0
    for t, x in enumerate(pts):
        a = scheduled_alpha(ALPHA0, t, HORIZON)
        if c is None:
            c, n = float(x), INIT_USAGE
        n *= 1.0 - a
        mass = n + a * 1.0
        c += (a * 1.0 / mass) * (x - c)
        n = mass
        _gain(s, x)
    row = s.centroids(SEED)[0]
    assert len(s.centroids(SEED)) == 1
    assert row[0] == pytest.approx(c, abs=1e-6)
    assert row[1] == pytest.approx(n, abs=1e-9)


def test_cosine_step_matches_the_reference_schedule():
    assert scheduled_alpha(0.5, 0, 100) == pytest.approx(0.5)
    assert scheduled_alpha(0.5, 100, 100) == pytest.approx(0.5 * ALPHA_MIN_FACTOR)
    assert scheduled_alpha(0.5, 500, 100) == pytest.approx(0.5 * ALPHA_MIN_FACTOR)  # holds
    mid = scheduled_alpha(0.5, 50, 100)
    assert mid == pytest.approx(0.5 * (ALPHA_MIN_FACTOR + (1 - ALPHA_MIN_FACTOR) * 0.5))
    steps = [scheduled_alpha(0.5, t, 100) for t in range(101)]
    assert all(a >= b for a, b in zip(steps, steps[1:], strict=False))
    assert math.isfinite(sum(steps))


def test_total_mass_settles_at_one_and_a_round_splits_evenly():
    s = _sched()
    for _ in range(60):
        _gain(s, [100, 900])  # two offsets: 1/2 of a point each
    rows = s.centroids(SEED)
    assert len(rows) == 2
    assert rows[:, 1].sum() == pytest.approx(1.0, abs=1e-2)
    assert rows[0][1] == pytest.approx(rows[1][1], rel=1e-3)
    assert rows[0][0] == pytest.approx(100, abs=1e-2)
    assert rows[1][0] == pytest.approx(900, abs=1e-2)


def test_offsets_past_the_end_clamp_to_the_last_byte():
    s = _sched()
    _gain(s, 5000)
    assert s.centroids(SEED)[0][0] == pytest.approx(len(SEED) - 1, abs=1e-2)


def test_draws_are_in_range_and_concentrate_on_the_gain_regions():
    rng = RandPool(seed=7)
    s = _sched(rng)
    gen = np.random.default_rng(0)
    for _ in range(300):
        x = gen.normal(1000, 30) if gen.random() < 0.7 else gen.normal(3000, 50)
        _gain(s, int(x), data=bytes(4000))
    data = bytes(4000)
    draws = [s.propose(data, 4000) for _ in range(3000)]
    assert all(0 <= d < 4000 for d in draws)
    hot = sum(1 for d in draws if 800 <= d <= 1200 or 2800 <= d <= 3200)
    assert hot / len(draws) > 0.75  # 1 - EXPLORE leaves ~0.9 on the hotspots


def test_codebook_follows_a_drifting_gain_region():
    s = _sched()
    for _ in range(80):
        _gain(s, 100)
    for _ in range(160):
        _gain(s, 800)
    rows = s.centroids(SEED)
    near_new = rows[np.abs(rows[:, 0] - 800) < 20][:, 1].sum()
    assert near_new / rows[:, 1].sum() > 0.95  # the old region has decayed away


def test_seeding_stops_at_k_slots():
    s = _sched()
    for x in range(0, 1000, 20):  # 50 well-separated points
        _gain(s, x)
    assert len(s.centroids(SEED)) == K


def test_revival_is_off_by_default_and_splits_donors_when_enabled():
    off = _sched()
    on = _sched(RandPool(seed=3), unused_threshold=0.1)
    for _ in range(REVIVE_EVERY):  # first pass: every target slot is empty
        _gain(off, 400)
        _gain(on, 400)
    assert len(off.centroids(SEED)) == 1
    rows = on.centroids(SEED)
    assert len(rows) == 2  # an empty slot was refilled by splitting the donor
    # A split halves the donor's mass: no mass is added or dropped on this pass.
    assert rows[:, 1].sum() == pytest.approx(off.centroids(SEED)[:, 1].sum(), rel=1e-9)
    assert rows[0][0] < 400 < rows[1][0]  # children straddle the donor
    for _ in range(REVIVE_EVERY * 2):  # later passes keep refilling; mass stays <= no-revival
        _gain(off, 400)
        _gain(on, 400)
    assert len(on.centroids(SEED)) > 2
    assert on.centroids(SEED)[:, 1].sum() <= off.centroids(SEED)[:, 1].sum() + 1e-9


def test_per_seed_state_is_lru_bounded():
    s = _sched()
    for i in range(MAX_SEEDS + 40):
        data = i.to_bytes(4, "big") + bytes(60)
        s.record(data, [3], Outcome.GAIN)
    assert len(s._books) == MAX_SEEDS


def test_constructor_validation():
    for bad in ({"alpha0": 0.0}, {"alpha0": 1.5}, {"horizon": 0}, {"unused_threshold": -0.1}):
        with pytest.raises(ValueError):
            _sched(**bad)
