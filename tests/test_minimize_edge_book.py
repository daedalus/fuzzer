"""cmin keeps per-seed edges as sorted uint32 arrays and collapses twins.

A ``set[int]`` of ~2k edges costs ~15x a uint32 array of the same ids, and
cmin held one per seed for the whole corpus. Seeds with identical coverage
are interchangeable for set-cover / rate-distortion, so only the smallest is
kept; minimax-robust keeps them, since a twin is a free backup.
"""

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.services.minimize import PruneMode, Twins, _EdgeBook, _prune


def _ids(*edges: int) -> np.ndarray:
    return np.array(sorted(edges), dtype=np.uint32)


@pytest.fixture
def shm():
    cov = ShmCoverage(size=8192)
    yield cov
    cov.cleanup()


def test_edge_id_array_sorted_uint32(shm):
    """Same ids as get_edge_ids, sorted, uint32, detached from the segment."""
    for edge in (900, 7, 31):
        shm.record_edge(edge)

    arr = shm.edge_id_array()
    assert arr.dtype == np.uint32
    assert arr.tolist() == sorted(shm.get_edge_ids())

    # A reset must not rewrite an array already handed out.
    shm.reset_edge_map()
    shm.record_edge(5)
    assert arr.tolist() == [7, 31, 900]


def test_edge_id_array_empty(shm):
    arr = shm.edge_id_array()
    assert arr.dtype == np.uint32
    assert arr.size == 0


def test_collapse_keeps_smallest_twin():
    """a, b share coverage; b is smaller, so a goes. c is distinct."""
    book = _EdgeBook(Twins.COLLAPSE)
    book.add("a", _ids(1, 2), 10)
    book.add("b", _ids(1, 2), 4)
    book.add("c", _ids(1, 3), 50)

    assert set(book.edges) == {"b", "c"}
    assert book.edges["b"].tolist() == [1, 2]
    assert book.sizes == {"b": 4, "c": 50}


def test_collapse_tie_keeps_first():
    book = _EdgeBook(Twins.COLLAPSE)
    book.add("first", _ids(5), 8)
    book.add("second", _ids(5), 8)

    assert list(book.edges) == ["first"]


def test_falsify_near_twins_not_merged():
    """One differing edge: both stay, or coverage would be lost."""
    book = _EdgeBook(Twins.COLLAPSE)
    book.add("x", _ids(1, 2, 3), 1)
    book.add("y", _ids(1, 2, 4), 100)

    assert set(book.edges) == {"x", "y"}


def test_keep_mode_retains_twins():
    book = _EdgeBook(Twins.KEEP)
    book.add("a", _ids(1, 2), 10)
    book.add("b", _ids(1, 2), 4)

    assert set(book.edges) == {"a", "b"}


def test_adversarial_empty_and_huge_ids():
    """Empty coverage collapses to one entry; ids at the uint32 top survive."""
    top = np.iinfo(np.uint32).max
    book = _EdgeBook(Twins.COLLAPSE)
    book.add("e1", _ids(), 3)
    book.add("e2", _ids(), 1)
    book.add("big", _ids(top, 1), 2)

    assert set(book.edges) == {"e2", "big"}
    assert book.edges["big"].tolist() == [1, top]


def test_adversarial_caller_buffer_reuse():
    """The book must not alias the caller's array."""
    book = _EdgeBook(Twins.COLLAPSE)
    ids = _ids(1, 2)
    book.add("a", ids, 1)
    ids[:] = 9

    assert book.edges["a"].tolist() == [1, 2]


@pytest.mark.parametrize("prune", list(PruneMode))
def test_prune_covers_all_edges(prune):
    """Every mode runs on array-backed books and keeps full coverage here.

    target_frac=1.0 so rate-distortion must reach every edge too.
    """
    book = _EdgeBook(Twins.KEEP if prune is PruneMode.MINIMAX_ROBUST else Twins.COLLAPSE)
    book.add("a", _ids(1, 10, 11), 1)
    book.add("b", _ids(1, 20, 21), 1)
    book.add("c", _ids(1, 10, 11, 30), 1)

    kept = _prune(book, prune, target_frac=1.0, map_entries=8192)

    covered = set().union(*(book.edges[k].tolist() for k in kept))
    assert covered == {1, 10, 11, 20, 21, 30}
    assert {"b", "c"} <= set(kept)
