"""PositionFractalScheduler: adaptive-resolution offset selection.

Covers core/schedulers/pos_fractal.py. Mirrors the shape of TestBurnFront
in test_position_arena.py, adapted for a tree instead of flat bins.
"""

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_fractal import (
    COOL_EVERY,
    FUEL_FLOOR,
    MAX_CELLS,
    MAX_SEEDS,
    SPARK_RATE,
    SPLIT_THRESHOLD,
    PositionFractalScheduler,
)

SEED = bytes(1000)
NO_SPARK = 0.99  # random() draw above SPARK_RATE


class ScriptedRng:
    """Deterministic stand-in: scripted random(), argmax weighted_choice.

    Same contract as test_position_arena.py's ScriptedRng (kept local so
    this file has no import-order dependency on that one).
    """

    def __init__(self, randoms=()):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_SPARK

    def randint(self, a, b):
        if (a, b) == (0, 9):
            return int(self.random() * 10)
        return a

    def weighted_choice(self, seq, weights):
        return seq[max(range(len(seq)), key=weights.__getitem__)]


def _fr(rng=None):
    return PositionFractalScheduler(rng or ScriptedRng())


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_fr(), PositionScheduler)


class TestFractal:
    def test_cold_seed_declines(self):
        assert _fr().propose(SEED, len(SEED)) is None

    def test_miss_deposits_nothing(self):
        # FALSIFICATION: a MISS that heated a cell would make every
        # execution a gain.
        s = _fr()
        s.record(SEED, [100], Outcome.MISS)
        assert s.cell_count(SEED) == 0
        assert s.propose(SEED, len(SEED)) is None

    def test_gain_lights_the_root_cell(self):
        s = _fr()
        s.record(SEED, [100], Outcome.GAIN)
        heat, fuel, forked = s.cell_state(SEED)
        assert heat == pytest.approx(1.0)
        assert fuel == 1.0
        assert not forked  # below SPLIT_THRESHOLD
        assert 0 <= s.propose(SEED, len(SEED)) < len(SEED)

    def test_weight_sums_when_offsets_share_a_cell(self):
        # At depth 0 there is only one cell (the whole seed), so two
        # offsets in the same round still land in it together.
        s = _fr()
        s.record(SEED, [100, 500], Outcome.GAIN, weight=2.0)
        assert s.cell_state(SEED)[0] == pytest.approx(2.0)

    def test_weight_is_split_once_offsets_land_in_different_children(self):
        s = _fr()
        s.record(SEED, [100], Outcome.GAIN, weight=SPLIT_THRESHOLD)  # forks root only
        s.record(SEED, [100, 900], Outcome.GAIN, weight=2.0)  # share=1.0 each
        left = s.cell_state(SEED, depth=1, idx=0)  # covers [0, 500) -> offset 100
        right = s.cell_state(SEED, depth=1, idx=1)  # covers [500, 1000) -> offset 900
        assert left[0] == pytest.approx(1.0)
        assert right[0] == pytest.approx(1.0)

    def test_root_forks_once_heat_crosses_the_threshold(self):
        s = _fr()
        s.record(SEED, [100], Outcome.GAIN, weight=SPLIT_THRESHOLD)
        heat, _, forked = s.cell_state(SEED)
        assert heat == pytest.approx(SPLIT_THRESHOLD)
        assert forked
        # No deposit has reached a child yet: the tree is root-only.
        assert s.cell_count(SEED) == 1

    def test_second_deposit_after_a_fork_reaches_a_child(self):
        s = _fr()
        s.record(SEED, [100], Outcome.GAIN, weight=SPLIT_THRESHOLD)  # forks root
        s.record(SEED, [100], Outcome.GAIN, weight=1.0)  # now descends
        assert s.cell_count(SEED) == 2
        assert s.cell_state(SEED, depth=1, idx=0) is not None

    def test_deep_fork_narrows_the_pick_to_the_frontier_cell(self):
        # 7 deposits at offset 750 fork the root (call 3), then depth-1's
        # right child (call 6); call 7 reaches depth 2. With ScriptedRng's
        # argmax weighted_choice, propose must descend root -> (1,1) ->
        # (2,3) and land exactly on that cell's start (span 250, offset
        # 750 // 250 == 3), landing on 750 itself.
        s = _fr(ScriptedRng())
        s.record(SEED, [750] * 7, Outcome.GAIN, weight=7.0)
        assert s.propose(SEED, len(SEED)) == 750

    def test_fuel_burns_and_pick_shifts(self):
        # FALSIFICATION: without fuel burn the argmax child never changes
        # and the pick is pinned to the same cell forever.
        s = _fr(RandPool(seed=7))
        s.record(SEED, [100] * 7, Outcome.GAIN, weight=7.0)
        picks = {s.propose(SEED, len(SEED)) for _ in range(200)}
        assert len(picks) > 1

    def test_fuel_never_reaches_zero(self):
        # ADVERSARIAL: a zero-weight cell would make weighted_choice raise.
        s = PositionFractalScheduler(RandPool(seed=5))
        s.record(SEED, [100] * 7, Outcome.GAIN, weight=7.0)
        for _ in range(5000):
            s.propose(SEED, len(SEED))
        heat, fuel, forked = s.cell_state(SEED)
        assert fuel >= FUEL_FLOOR

    def test_spark_escapes_the_tree(self):
        s = PositionFractalScheduler(ScriptedRng([SPARK_RATE / 2]))
        s.record(SEED, [500], Outcome.GAIN)
        assert s.propose(SEED, len(SEED)) == 0  # ScriptedRng.randint -> lo

    def test_cooling_extinguishes_non_root_cells(self):
        s = _fr()
        s.record(SEED, [100] * 7, Outcome.GAIN, weight=7.0)
        assert s.cell_count(SEED) > 1
        for _ in range(COOL_EVERY * 400):
            s.propose(SEED, len(SEED))
        # The root cell is always kept (it anchors the tree so propose can
        # still fall back to a coarse pick); its descendants cool away.
        assert s.cell_count(SEED) == 1
        assert 0 <= s.propose(SEED, len(SEED)) < len(SEED)

    def test_position_is_clamped_to_a_shrunken_buffer(self):
        # ADVERSARIAL: earlier ops in the round may already have shrunk the
        # buffer below the hot offset.
        s = _fr()
        s.record(SEED, [900] * 7, Outcome.GAIN, weight=7.0)
        for _ in range(50):
            assert 0 <= s.propose(SEED, 10) < 10

    def test_empty_buffer_declines(self):
        s = _fr()
        s.record(SEED, [1], Outcome.GAIN)
        assert s.propose(SEED, 0) is None

    def test_offsets_past_the_seed_end_are_accepted(self):
        # ADVERSARIAL: the buffer may have grown past the parent seed; the
        # deposit clamps to the tree's last cell instead of indexing an
        # unbounded new depth.
        s = _fr()
        s.record(SEED, [len(SEED) + 500], Outcome.GAIN)
        assert 0 <= s.propose(SEED, len(SEED) + 600) < len(SEED) + 600

    def test_negative_offsets_are_ignored(self):
        s = _fr()
        s.record(SEED, [-5], Outcome.GAIN)
        assert s.cell_count(SEED) == 0

    def test_cells_are_capped_and_keep_the_root(self):
        wide = bytes(20000)
        s = _fr()
        # Fork a different branch per offset so distinct cells accumulate
        # faster than a single hot spot would.
        for i in range(200):
            off = (i * 97) % len(wide)
            s.record(wide, [off] * 7, Outcome.GAIN, weight=7.0)
        assert s.cell_count(wide) <= MAX_CELLS
        assert s.cell_state(wide, 0, 0) is not None

    def test_seed_table_is_lru_bounded(self):
        s = _fr()
        for i in range(MAX_SEEDS + 50):
            s.record(i.to_bytes(4, "big") * 4, [1], Outcome.GAIN)
        assert s.seed_count() == MAX_SEEDS
        assert s.cell_count((0).to_bytes(4, "big") * 4) == 0  # oldest evicted


class TestPersistence:
    def test_round_trip_preserves_tree_state(self):
        s = _fr()
        s.record(SEED, [750] * 7, Outcome.GAIN, weight=7.0)
        before = s.cell_state(SEED, depth=2, idx=3)

        restored = _fr()
        restored.from_dict(s.to_dict())
        assert restored.cell_count(SEED) == s.cell_count(SEED)
        assert restored.cell_state(SEED, depth=2, idx=3) == before

    def test_malformed_state_resets_cleanly(self):
        s = _fr()
        s.record(SEED, [100], Outcome.GAIN)
        s.from_dict({"version": 999, "trees": {}})
        assert s.seed_count() == 0
        assert s.propose(SEED, len(SEED)) is None

    def test_empty_state_is_a_no_op(self):
        s = _fr()
        s.from_dict({})
        assert s.seed_count() == 0

    def test_restore_respects_the_lru_cap(self):
        s = _fr()
        payload = {
            "version": 1,
            "trees": {i: (100, {"0:0": (1.0, 1.0, False)}, 1) for i in range(MAX_SEEDS + 20)},
        }
        s.from_dict(payload)
        assert s.seed_count() == MAX_SEEDS
