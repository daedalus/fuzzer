"""PositionChangedScheduler: pooled group testing on "did the path move?".

Covers core/schedulers/pos_changed.py.
"""

from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_changed import PRIOR_A, PRIOR_B, PositionChangedScheduler

SEED = bytes(64)


class ScriptedRng:
    def __init__(self, randoms=()):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0) if self._randoms else 0.5

    def randint(self, a, b):
        return a


def _sched(moved):
    box = {"m": moved}
    return PositionChangedScheduler(ScriptedRng(), moved=lambda d: box["m"]), box


def _rate(s, n):
    return (s + PRIOR_A) / (n + PRIOR_A + PRIOR_B)


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_sched(True)[0], PositionScheduler)

    def test_cold_seed_declines(self):
        assert _sched(True)[0].propose(SEED, len(SEED)) is None


class TestPooledCredit:
    def test_unmoved_round_clears_every_offset(self):
        # Group test: nothing moved, so every mutated byte is inert-ish.
        s, _ = _sched(False)
        s.record(SEED, [1, 2, 3], Outcome.MISS)
        w = s.weights(SEED, len(SEED))
        assert all(w[b] == _rate(0, 1) for b in (1, 2, 3))
        assert w[0] == _rate(0, 0)

    def test_moved_round_splits_the_success(self):
        s, _ = _sched(True)
        s.record(SEED, [1, 2], Outcome.MISS)
        w = s.weights(SEED, len(SEED))
        assert w[1] == w[2] == _rate(0.5, 1)

    def test_moved_counts_regardless_of_gain(self):
        # The signal is trace movement, not coverage: a MISS that moved is live.
        a, _ = _sched(True)
        b, _ = _sched(True)
        a.record(SEED, [5], Outcome.MISS)
        b.record(SEED, [5], Outcome.GAIN)
        assert a.weights(SEED, len(SEED))[5] == b.weights(SEED, len(SEED))[5]

    def test_unknown_signal_is_not_credited(self):
        s, _ = _sched(None)
        s.record(SEED, [5], Outcome.MISS)
        assert s.weights(SEED, len(SEED)) is None

    def test_falsification_inert_bins_lose_mass(self):
        s, box = _sched(False)
        for _ in range(20):
            s.record(SEED, [10], Outcome.MISS)
        box["m"] = True
        for _ in range(20):
            s.record(SEED, [20], Outcome.MISS)
        w = s.weights(SEED, len(SEED))
        assert w[20] > w[0] > w[10]

    def test_adversarial_empty_and_negative_offsets(self):
        s, _ = _sched(True)
        s.record(SEED, [], Outcome.MISS)
        s.record(SEED, [-3], Outcome.MISS)
        assert s.weights(SEED, len(SEED)) is None

    def test_proposes_after_evidence(self):
        s, _ = _sched(True)
        s.record(SEED, [7], Outcome.MISS)
        off = s.propose(SEED, len(SEED))
        assert 0 <= off < len(SEED)
