"""``--prune-density``: break eviction ties by edges per bit, least dense first.

Same seam as ``--prune-cost`` (``EdgeTracker._prune_victims``): loss of
unique coverage orders eviction, the tie decides nearly every one. With
``EvictTie.DENSITY`` the seed hitting the fewest edges per bit of input goes
first, so of equally redundant seeds the compact ones stay tracked.
"""

from __future__ import annotations

import subprocess
import sys

from fuzzer_tool.core.edge_density import edges_per_bit
from fuzzer_tool.core.edge_tracker import EdgeTracker, EvictTie

CEILING = 10
N_NARROW = 30
WIDE_EDGES = set(range(1, 1 + N_NARROW))
UNIQUE_EDGE = 10_000


def _size(i: int) -> int:
    """Sizes not monotone in age, so AGE and DENSITY orders differ."""
    return (i * 7) % N_NARROW + 1


def _tracker(tie: EvictTie) -> tuple[EdgeTracker, list[str]]:
    """A wide seed covering every narrow seed's edge, then narrow seeds."""
    et = EdgeTracker(max_tracked_seeds=CEILING)
    et.evict_tie = tie
    et.record_edges("wide", WIDE_EDGES, nbytes=N_NARROW)
    narrow = []
    for i in range(N_NARROW):
        key = f"n{i}"
        et.record_edges(key, {1 + i}, nbytes=_size(i))
        narrow.append(key)
    return et, narrow


def _evicted(et: EdgeTracker, narrow: list[str]) -> list[int]:
    return [i for i, k in enumerate(narrow) if k not in et.seed_edges]


def test_density_evicts_least_dense_first():
    """Every evicted redundant seed is at most as dense as every survivor."""
    et, narrow = _tracker(EvictTie.DENSITY)
    gone = _evicted(et, narrow)
    kept = [i for i in range(N_NARROW) if i not in gone]
    assert gone and kept

    density = [edges_per_bit(1, _size(i)) for i in range(N_NARROW)]
    assert max(density[i] for i in gone) <= min(density[i] for i in kept)


def test_density_differs_from_age():
    """Falsification: the fixture separates the orders, else the test above is vacuous."""
    dense, narrow = _tracker(EvictTie.DENSITY)
    age, _ = _tracker(EvictTie.AGE)
    assert _evicted(dense, narrow) != _evicted(age, narrow)


def test_loss_dominates_density():
    """Adversarial: a sparse seed owning an edge outlives dense redundant ones."""
    et = EdgeTracker(max_tracked_seeds=CEILING)
    et.evict_tie = EvictTie.DENSITY
    et.record_edges("wide", WIDE_EDGES, nbytes=N_NARROW)
    et.record_edges("owner", {UNIQUE_EDGE}, nbytes=1 << 20)
    for i in range(N_NARROW):
        et.record_edges(f"n{i}", {1 + i}, nbytes=1)
    assert "owner" in et.seed_edges
    assert "wide" in et.seed_edges


def test_unsized_seed_evicted_last():
    """Adversarial: a seed recorded without a size is never ranked least dense."""
    et = EdgeTracker(max_tracked_seeds=CEILING)
    et.evict_tie = EvictTie.DENSITY
    et.record_edges("wide", WIDE_EDGES, nbytes=N_NARROW)
    et.record_edges("unsized", {1})
    for i in range(N_NARROW):
        et.record_edges(f"n{i}", {1 + i}, nbytes=_size(i))
    assert "unsized" in et.seed_edges
    assert "unsized" not in et.seed_size


def test_sizes_pruned_with_seeds():
    """Memory bound: no size entry outlives its seed."""
    et, _ = _tracker(EvictTie.DENSITY)
    assert set(et.seed_size) <= set(et.seed_edges)


def test_age_records_no_size():
    """AGE keeps no size map at all."""
    et, _ = _tracker(EvictTie.AGE)
    assert not et.seed_size


def test_cli_flag():
    out = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"], capture_output=True, text=True
    ).stdout
    assert "--prune-density" in out
