"""``_EdgeVectors.indices`` via ``np.fromiter``: same arrays, same new-id order.

~2.5% faster on mostly-known sets. Reference: the list/map loop, copied.
"""

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.seed_picker import _EdgeVectors


class _Reference:
    # Pre-change indices(), without the cache.
    def __init__(self) -> None:
        self.index: dict[int, int] = {}
        self.ids: list[int] = []

    def indices(self, edges: set[int]) -> np.ndarray:
        order = list(edges)
        found = list(map(self.index.get, order))
        for k, i in enumerate(found):
            if i is None:
                i = self.index[order[k]] = len(self.ids)
                self.ids.append(order[k])
                found[k] = i
        return np.array(found, dtype=np.int64)


def _sets(seed: int) -> list[set[int]]:
    rng = RandPool(seed=seed)
    return [set(rng.randint_list(0, 500, rng.randint(1, 60))) for _ in range(40)]


def _drive(make, seed: int) -> tuple[list, list]:
    vec = make()
    arrays = [vec.indices(s).tolist() for s in _sets(seed)]
    ids = vec.ids if isinstance(vec, _Reference) else vec._ids
    return arrays, list(ids)


def test_regression_indices_control():
    # Control (Hard Rule 46).
    assert _drive(_Reference, 1) == _drive(_Reference, 1)


def test_regression_indices_match_loop():
    for seed in range(6):
        assert _drive(_EdgeVectors, seed) == _drive(_Reference, seed), seed


def test_regression_indices_grown_set_reindexed():
    # Adversarial: a set grown in place is re-indexed, new ids appended.
    vec = _EdgeVectors()
    s = {3, 1}
    first = vec.indices(s)
    s.add(99)
    grown = vec.indices(s)
    assert sorted(first.tolist()) == [0, 1]
    assert sorted(grown.tolist()) == [0, 1, 2]
    assert vec._ids[2] == 99


def test_regression_indices_empty_set():
    # Falsification: an empty set yields an empty int64 array, no ids.
    vec = _EdgeVectors()
    out = vec.indices(set())
    assert out.dtype == np.int64 and out.size == 0
    assert vec._ids == []
