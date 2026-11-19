"""Regression: sensitivity trackers evict the oldest seed, not an arbitrary one.

``_analyzed`` is a set; bytes hashing is salted per process, so ``next(iter(set))``
picked a hash-order victim and evicted the newest seed under some PYTHONHASHSEEDs.
"""

import pytest

from fuzzer_tool.core.analyzers.analyzer_influence import WindowInfluenceTracker
from fuzzer_tool.core.analyzers.analyzer_sensitivity import ByteSensitivityTracker

BASE = {1, 2, 3}
SEEDS = [bytes([i]) * 16 for i in range(1, 5)]


class _NewestFirstSet(set):
    """Set iterating newest-first: the adversarial hash order."""

    def __init__(self, items=()):
        super().__init__()
        self._order: list[bytes] = []
        for item in items:
            self.add(item)

    def add(self, item):
        if item not in self:
            self._order.append(item)
        super().add(item)

    def discard(self, item):
        if item in self:
            self._order.remove(item)
        super().discard(item)

    def __iter__(self):
        return iter(reversed(self._order))


TRACKERS = [ByteSensitivityTracker, WindowInfluenceTracker]


def _fill(cls, max_seeds):
    t = cls(max_seeds=max_seeds, sample_rate=1.0)
    t._analyzed = _NewestFirstSet()
    for seed in SEEDS:
        t.analyze_seed(seed, BASE, lambda d: BASE)
    return t


@pytest.mark.parametrize("cls", TRACKERS)
def test_regression_evicts_oldest_seed(cls):
    keep = 2
    t = _fill(cls, keep)

    # Falsification: the first seeds went, the last `keep` stayed.
    assert [t.has_data(s) for s in SEEDS] == [False] * (len(SEEDS) - keep) + [True] * keep


@pytest.mark.parametrize("cls", TRACKERS)
def test_regression_eviction_keeps_sets_in_sync(cls):
    t = _fill(cls, 1)

    # Adversarial: hash order newest-first must not desync the two indexes.
    assert set(t._analyzed) == set(t._sensitivity) == {SEEDS[-1][:64]}
