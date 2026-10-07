"""Pool eviction picks the same victims as the full (value, index) sort.

``_eviction_victims`` stops at the first *excess* zero-credit entries
instead of sorting the whole 10,000-token / 5,000-pair pool on every drain.
Credit is a non-negative count, so those are the smallest keys whenever
there are enough of them; the sort remains for when there are not.
"""

import random

from fuzzer_tool.core.cmplog import _eviction_victims


def _reference(pool, value, excess):
    order = sorted(range(len(pool)), key=lambda i: (value.get(pool[i], 0), i))
    return set(order[:excess])


def test_matches_the_sort():
    rng = random.Random(4)
    for trial in range(3000):
        n = rng.randint(1, 80)
        pool = [bytes([trial % 251, i % 256, i // 256]) for i in range(n)]
        density = rng.random()
        value = {p: rng.randint(1, 6) for p in pool if rng.random() < density}
        excess = rng.randint(1, n)
        assert _eviction_victims(pool, value, excess) == _reference(pool, value, excess)


def test_all_credited_falls_back_to_the_sort():
    pool = [b"a", b"b", b"c", b"d"]
    value = {b"a": 3, b"b": 1, b"c": 2, b"d": 1}
    assert _eviction_victims(pool, value, 2) == {1, 3}
