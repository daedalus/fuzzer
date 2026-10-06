"""PositionKadaneScheduler: mutate inside the seed's maximum-excess-gain run.

Covers core/schedulers/pos_kadane.py.
"""

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import MAX_BINS
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_kadane import (
    EXPLORE,
    MIN_EXCESS,
    PositionKadaneScheduler,
)

SEED = bytes(1000)  # 1000 < MAX_BINS, so one bin per byte


class ScriptedRng:
    """random() above EXPLORE (never escapes); randint -> a."""

    def random(self):
        return 0.5

    def randint(self, a, b):
        return a


def _sched(rng=None):
    return PositionKadaneScheduler(rng or ScriptedRng())


def _gain(s, offset, times=1, data=SEED):
    for _ in range(times):
        s.record(data, [offset], Outcome.GAIN)


def _miss(s, offset, times=1, data=SEED):
    for _ in range(times):
        s.record(data, [offset], Outcome.MISS)


def _background(s, data=SEED, times=50):
    """Misses at byte 0: a pooled rate below 1, or no bin can beat expectation."""
    _miss(s, 0, times, data=data)


def test_satisfies_protocol_and_name():
    s = _sched()
    assert isinstance(s, PositionScheduler)
    assert s.name == "kadane"


def test_declines_with_no_evidence_and_with_misses_only():
    s = _sched()
    assert s.propose(SEED, 1000) is None
    assert s.window(SEED, 1000) is None
    _miss(s, 10, 30)
    assert s.propose(SEED, 1000) is None


def test_declines_on_empty_input_and_zero_length():
    s = _sched()
    _gain(s, 5, 3)
    assert s.propose(b"", 10) is None
    assert s.propose(SEED, 0) is None
    assert s.window(SEED, 0) is None


def test_window_is_the_gain_cluster_and_excludes_a_missed_bin():
    s = _sched()
    _miss(s, 10, 50)
    _gain(s, 500, 3)
    _gain(s, 520, 3)
    assert s.window(SEED, 1000) == (500, 21)


def test_untried_gap_between_two_gains_is_bridged():
    s = _sched()
    _background(s)
    _gain(s, 100, 3)
    _gain(s, 300, 3)
    assert s.window(SEED, 1000) == (100, 201)


def test_heavily_missed_bin_between_gains_splits_them():
    s = _sched()
    _gain(s, 100, 3)
    _miss(s, 150, 200)
    _gain(s, 200, 3)
    off, length = s.window(SEED, 1000)
    assert (off, length) in {(100, 1), (200, 1)}


def test_excess_floor_declines_a_single_early_gain():
    s = _sched()
    _miss(s, 10, 40)
    _gain(s, 700, 1)
    # one gain over 41 trials: excess is 1 - 1/41 < MIN_EXCESS
    assert MIN_EXCESS >= 1.0
    assert s.window(SEED, 1000) is None


def test_propose_stays_inside_window_unless_escaping():
    s = PositionKadaneScheduler(RandPool(seed=7))
    _miss(s, 10, 50)
    _gain(s, 500, 3)
    _gain(s, 520, 3)
    draws = [s.propose(SEED, 1000) for _ in range(4000)]
    inside = sum(1 for d in draws if 500 <= d <= 520)
    # EXPLORE of draws are uniform over 1000 bytes; the rest land in the 21-byte window
    assert inside / len(draws) > 1 - EXPLORE - 0.02
    assert all(0 <= d < 1000 for d in draws)


def test_window_is_clipped_to_a_shrunk_buffer():
    s = _sched()
    _background(s)
    _gain(s, 900, 3)
    _gain(s, 905, 3)
    assert s.window(SEED, 1000) == (900, 6)
    # buffer shrank below the window: no live bin carries a gain
    assert s.window(SEED, 800) is None
    # only the first of the two gain bins is still live
    assert s.window(SEED, 903) == (900, 1)


def test_binning_matches_bin_rates_on_a_large_seed():
    big = bytes(MAX_BINS * 4)  # width 4
    s = _sched()
    _background(s, data=big)
    _gain(s, 4000, 3, data=big)
    _gain(s, 4004, 3, data=big)
    off, length = s.window(big, len(big))
    assert off == 4000 and length == 8  # two adjacent 4-byte bins


def test_cache_is_refreshed_after_new_credit():
    s = _sched()
    _background(s)
    _gain(s, 100, 3)
    first = s.window(SEED, 1000)
    assert first == (100, 1)
    _gain(s, 110, 6)
    assert s.window(SEED, 1000) == (100, 11)


def test_scores_sum_to_zero_and_untried_bins_are_zero():
    s = _sched()
    _miss(s, 10, 20)
    _gain(s, 500, 4)
    x = s.scores(SEED, 1000)
    assert isinstance(x, np.ndarray) and len(x) == 1000
    assert abs(float(x.sum())) < 1e-9
    assert float(x[0]) == 0.0
    assert float(x[500]) > 0 > float(x[10])


def test_gain_only_evidence_has_nothing_to_stand_out_against():
    s = _sched()
    _gain(s, 100, 5)
    # every trial gained: the pooled rate is 1, every bin sits exactly on it
    assert s.window(SEED, 1000) is None


def test_ignores_negative_offsets():
    s = _sched()
    s.record(SEED, [-1], Outcome.GAIN)
    assert s.window(SEED, 1000) is None


def test_seeds_do_not_share_evidence():
    s = _sched()
    other = bytes([1]) * 1000
    _background(s)
    _gain(s, 100, 3)
    assert s.window(SEED, 1000) is not None
    assert s.window(other, 1000) is None
