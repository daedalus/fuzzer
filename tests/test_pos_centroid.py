"""PositionCentroidScheduler: mutate where the seed reaches its cluster's missing members.

Covers core/schedulers/pos_centroid.py.
"""

from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_centroid import (
    PRIOR_A,
    PRIOR_B,
    RETARGET_EVERY,
    PositionCentroidScheduler,
)

SEED = bytes(64)


class ScriptedRng:
    def random(self):
        return 0.5

    def randint(self, a, b):
        return a


class _Env:
    """Seed gaps (missing cluster members) and this exec's hit set, scripted."""

    def __init__(self, gaps, hit):
        self.gaps = gaps
        self.hit = hit
        self.queries = []
        self.version = 0

    def sched(self):
        return PositionCentroidScheduler(
            ScriptedRng(),
            gaps_of=lambda d: self.gaps,
            hit_any=self._hit_any,
            version=lambda: self.version,
        )

    def _hit_any(self, edges):
        self.queries.append(tuple(edges))
        return None if self.hit is None else any(e in self.hit for e in edges)


def _rate(s, n):
    return (s + PRIOR_A) / (n + PRIOR_A + PRIOR_B)


def test_satisfies_the_protocol():
    assert isinstance(_Env([5], hit=set()).sched(), PositionScheduler)


def test_no_gaps_declines_and_never_scans():
    env = _Env([], hit={5})
    s = env.sched()
    s.record(SEED, [3], Outcome.GAIN)
    assert s.propose(SEED, len(SEED)) is None
    assert s.weights(SEED, len(SEED)) is None
    assert env.queries == []


def test_reaching_a_gap_credits_the_bins():
    env = _Env([5, 7], hit={7})
    s = env.sched()
    s.record(SEED, [4, 9], Outcome.GAIN)
    w = s.weights(SEED, len(SEED))
    assert w[4] == w[9] == _rate(1, 1)
    assert env.queries == [(5, 7)]


def test_missing_every_gap_charges_the_bins():
    env = _Env([5], hit=set())
    s = env.sched()
    s.record(SEED, [4], Outcome.GAIN)  # outcome irrelevant: no gap was reached
    assert s.weights(SEED, len(SEED))[4] == _rate(0, 1)


def test_falsification_rates_separate_bins():
    env = _Env([5], hit={5})
    s = env.sched()
    for _ in range(10):
        s.record(SEED, [10], Outcome.MISS)
    env.hit = set()
    for _ in range(10):
        s.record(SEED, [20], Outcome.MISS)
    w = s.weights(SEED, len(SEED))
    assert w[10] > w[0] > w[20]


def test_unknown_hit_and_negative_offsets_not_credited():
    s = _Env([5], hit=None).sched()
    s.record(SEED, [4], Outcome.MISS)
    assert s.weights(SEED, len(SEED)) is None
    s2 = _Env([5], hit={5}).sched()
    s2.record(SEED, [-1], Outcome.MISS)
    assert s2.weights(SEED, len(SEED)) is None


def test_retarget_resets_on_changed_gaps():
    env = _Env([5], hit={5})
    s = env.sched()
    s.record(SEED, [4], Outcome.MISS)
    env.gaps = [9]  # the cluster refit moved the seed's gaps
    for _ in range(RETARGET_EVERY):
        s.record(SEED, [6], Outcome.MISS)
    assert s.gaps(SEED) == (9,)
    assert s.weights(SEED, len(SEED))[4] == _rate(0, 0)


def test_gaps_cached_between_retargets():
    env = _Env([5], hit={5})
    s = env.sched()
    s.record(SEED, [4], Outcome.MISS)
    env.gaps = [9]
    s.record(SEED, [4], Outcome.MISS)
    assert s.gaps(SEED) == (5,)


def test_refit_retargets_immediately():
    # A cluster refit (version bump) must not wait RETARGET_EVERY credits.
    env = _Env([], hit={9})
    s = env.sched()
    assert s.propose(SEED, len(SEED)) is None
    env.gaps, env.version = [9], 1
    s.record(SEED, [4], Outcome.MISS)
    assert s.gaps(SEED) == (9,)
    assert s.weights(SEED, len(SEED))[4] == _rate(1, 1)
