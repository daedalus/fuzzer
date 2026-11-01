"""EdgeSet: sorted uint32 edge set standing in for ``set[int]`` in seed_edges.

Oracle: a builtin set built from the same input. Every operation is checked
against it, both operand orders, and the control (oracle vs a second oracle
built from a shuffled copy) must agree first (Hard Rule 46).
"""

import pickle
import sys

import numpy as np
import pytest

from fuzzer_tool.core.edge_set import EdgeSet

_MAX_U32 = 2**32 - 1

# Unsorted, duplicated, spanning small ints (cached objects) and large ones.
_A = [9, 3, 3, 70000, 0, 4_000_000_000, 512, 9]
_B = [3, 512, 8, 70001, 4_000_000_000]


def _ops(x, y):
    """Every binary result a consumer can read, as builtin values."""
    return {
        "and": x & y,
        "or": x | y,
        "sub": x - y,
        "xor": x ^ y,
        "le": x <= y,
        "ge": x >= y,
        "lt": x < y,
        "eq": x == y,
        "disjoint": x.isdisjoint(y),
    }


def test_control_oracle_matches_itself():
    shuffled = list(reversed(_A))
    assert _ops(set(_A), set(_B)) == _ops(set(shuffled), set(_B))


def test_iter_sorted_unique():
    assert list(EdgeSet(_A)) == sorted(set(_A))
    assert len(EdgeSet(_A)) == len(set(_A))


def test_empty():
    e = EdgeSet()
    assert not e
    assert len(e) == 0
    assert list(e) == []
    assert e == set()


def test_contains():
    e = EdgeSet(_A)
    for x in set(_A):
        assert x in e
    for x in (1, 70001, _MAX_U32, -1, 2**40):
        assert x not in e
    assert np.uint32(512) in e


def test_contains_non_int_is_false():
    e = EdgeSet(_A)
    assert "a" not in e
    assert None not in e


@pytest.mark.parametrize("other_type", [set, frozenset, EdgeSet])
def test_algebra_matches_set(other_type):
    want = _ops(set(_A), set(_B))
    assert _ops(EdgeSet(_A), other_type(_B)) == want


@pytest.mark.parametrize("other_type", [set, frozenset])
def test_reflected_algebra_matches_set(other_type):
    want = _ops(set(_B), set(_A))
    assert _ops(other_type(_B), EdgeSet(_A)) == want


def test_results_are_mutable_builtin_sets():
    e = EdgeSet(_A)
    for r in (e & set(_B), set(_B) & e, e | set(_B), set(_B) | e, e - set(_B), set(_B) - e):
        assert type(r) is set


def test_eq_and_unhashable():
    assert EdgeSet(_A) == EdgeSet(reversed(_A))
    assert EdgeSet(_A) == frozenset(_A)
    assert EdgeSet(_A) != set(_B)
    assert EdgeSet(_A) != list(_A)
    with pytest.raises(TypeError):
        hash(EdgeSet(_A))


def test_update_merges():
    e = EdgeSet(_A)
    e.update({1, 9}, [70001])
    assert e == set(_A) | {1, 70001}
    assert list(e) == sorted(e)


def test_update_empty_is_noop():
    e = EdgeSet(_A)
    e.update(set())
    assert e == set(_A)


@pytest.mark.parametrize("bad", [-1, _MAX_U32 + 1])
def test_out_of_range_raises(bad):
    with pytest.raises(OverflowError):
        EdgeSet([1, bad])
    e = EdgeSet([1])
    with pytest.raises(OverflowError):
        e.update([bad])
    assert e == {1}


def test_from_numpy_unsorted():
    e = EdgeSet(np.array([5, 1, 5, 3], dtype=np.int64))
    assert list(e) == [1, 3, 5]


def test_as_numpy_view():
    arr = EdgeSet(_A).as_numpy()
    assert arr.dtype == np.uint32
    assert arr.tolist() == sorted(set(_A))
    assert not arr.flags.writeable


def test_pickle_roundtrip():
    e = EdgeSet(_A)
    back = pickle.loads(pickle.dumps(e))
    assert type(back) is EdgeSet
    assert back == e


def test_smaller_than_set():
    edges = range(1000, 1000 + 3000 * 7, 7)
    s = set(edges)
    set_bytes = sys.getsizeof(s) + sum(map(sys.getsizeof, s))
    assert sys.getsizeof(EdgeSet(edges)) * 10 < set_bytes


def test_copy_is_independent():
    e = EdgeSet(_A)
    c = e.copy()
    c.update([1])
    assert 1 not in e


def test_cache_revalidates_after_update():
    e = EdgeSet(_A)
    assert not (e & {1})  # warms the materialized set
    e.update([1])
    assert e & {1} == {1}
    assert {1} - e == set()


def test_results_do_not_alias_cache():
    e = EdgeSet(_A)
    r = e - set()
    r.add(1)
    assert e & {1} == set()
    assert len(e | set()) == len(set(_A))


def test_cache_bounded():
    from fuzzer_tool.core import edge_set

    sets = [EdgeSet([i]) for i in range(3 * edge_set._HOT_SLOTS)]
    for s in sets:
        _ = s & {0}
    assert len(edge_set._HOT) <= edge_set._HOT_SLOTS
