"""``TupleWalk``: non-repeating m-permutation draws, wired into ``_swap_tuple``."""

import math

import pytest

from fuzzer_tool.core.mutations.generic import _swap_tuple
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.tuple_walk import MAX_KEYS, TupleWalk


def test_full_period_covers_space_exactly_once():
    walk, rng = TupleWalk(), RandPool(seed=1)
    total = math.perm(5, 3)
    seen = [walk.draw(5, 3, rng) for _ in range(total)]
    assert len(set(seen)) == total
    assert all(len(set(p)) == 3 and max(p) < 5 for p in seen)


def test_wraps_after_exhaustion():
    walk, rng = TupleWalk(), RandPool(seed=2)
    total = math.perm(4, 2)
    first = [walk.draw(4, 2, rng) for _ in range(total)]
    assert [walk.draw(4, 2, rng) for _ in range(total)] == first


def test_control_two_walks_same_seed_identical():
    a = [TupleWalk().draw(50, 4, RandPool(seed=7)) for _ in range(1)]
    b = [TupleWalk().draw(50, 4, RandPool(seed=7)) for _ in range(1)]
    assert a == b


def test_different_seeds_diverge():
    a = TupleWalk().draw(500, 5, RandPool(seed=1))
    b = TupleWalk().draw(500, 5, RandPool(seed=2))
    assert a != b


def test_huge_space_no_repeat():
    walk, rng = TupleWalk(), RandPool(seed=3)
    draws = [walk.draw(100_000, 5, rng) for _ in range(2000)]
    assert len(set(draws)) == 2000


def test_too_small_domain_is_none():
    assert TupleWalk().draw(3, 4, RandPool(seed=1)) is None
    assert TupleWalk().draw(4, 1, RandPool(seed=1)) is None


def test_state_bounded():
    walk, rng = TupleWalk(), RandPool(seed=4)
    for n in range(10, 10 + 3 * MAX_KEYS):
        walk.draw(n, 4, rng)
    assert len(walk) <= MAX_KEYS


@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_swap_tuple_with_walk_valid(m):
    walk, rng = TupleWalk(), RandPool(seed=5)
    for _ in range(200):
        picked, permuted = _swap_tuple(12, rng, m, walk=walk)
        assert len(set(picked)) == m
        assert sorted(picked) == sorted(permuted)
        assert picked != permuted


def test_swap_tuple_walk_respects_start():
    walk, rng = TupleWalk(), RandPool(seed=6)
    for _ in range(100):
        picked, _ = _swap_tuple(12, rng, 4, start=1, walk=walk)
        assert min(picked) >= 1


def test_swap_tuple_walk_ignored_for_sequence_domain():
    walk, rng = TupleWalk(), RandPool(seed=8)
    picked, _ = _swap_tuple([3, 5, 7, 9, 11], rng, 3, walk=walk)
    assert set(picked) <= {3, 5, 7, 9, 11}
    assert len(walk) == 0


class _Scripted:
    """Deterministic rng: always the lowest draw, so the m>2 branch fires."""

    def randint(self, a, b):
        return a

    def sample(self, population, k):
        return list(population[:k])

    def choice(self, seq):
        return seq[0]


def _swap_bytes(walk):
    from types import SimpleNamespace

    from fuzzer_tool.services.operators import OperatorEngine

    eng = SimpleNamespace(ctx=SimpleNamespace(_rng=_Scripted()), f=SimpleNamespace(swap_walk=walk))
    buf = bytearray(range(16))
    OperatorEngine._op_swap_bytes(eng, buf, 0, b"")
    return buf


def test_op_swap_bytes_uses_walk_when_set():
    walk = TupleWalk()
    buf = _swap_bytes(walk)
    assert len(walk) == 1
    assert sorted(buf) == list(range(16)) and buf != bytearray(range(16))


def test_op_swap_bytes_without_walk_untouched():
    walk = TupleWalk()
    _swap_bytes(None)
    assert len(walk) == 0
