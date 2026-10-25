"""``--prune-cost``: break eviction ties by exec cost, not age (backlog D5).

``EdgeTracker._prune_victims`` evicts by unique coverage lost; 99.9% of
candidates lose none, so the tie decides nearly every eviction and the tie
was insertion order. With ``EvictTie.COST`` the most expensive of equally
redundant seeds goes first. Loss still dominates: cost never evicts a seed
that owns an edge ahead of a redundant one. Default (AGE) is unchanged.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker, EvictTie

CEILING = 10
N_NARROW = 30
WIDE_EDGES = set(range(1, 1 + N_NARROW))
UNIQUE_EDGE = 10_000


def _cost(i: int) -> float:
    """Costs not monotone in age, so AGE and COST orders differ."""
    return float((i * 7) % N_NARROW + 1) * 1e-4


def _tracker(tie: EvictTie) -> tuple[EdgeTracker, list[str]]:
    """A wide seed covering every narrow seed's edge, then narrow seeds."""
    et = EdgeTracker(max_tracked_seeds=CEILING)
    et.evict_tie = tie
    et.record_edges("wide", WIDE_EDGES, exec_s=1.0)
    narrow = []
    for i in range(N_NARROW):
        key = f"n{i}"
        et.record_edges(key, {1 + i}, exec_s=_cost(i))
        narrow.append(key)
    return et, narrow


def _evicted(et: EdgeTracker, narrow: list[str]) -> list[int]:
    return [i for i, k in enumerate(narrow) if k not in et.seed_edges]


def test_regression_prune_cost_tie():
    """COST: every evicted redundant seed cost at least as much as every survivor."""
    et, narrow = _tracker(EvictTie.COST)
    gone = _evicted(et, narrow)
    kept = [i for i in range(N_NARROW) if i not in gone]
    assert gone and kept
    assert min(_cost(i) for i in gone) >= max(_cost(i) for i in kept)


def test_age_default_unchanged():
    """Falsification: the default still evicts oldest first."""
    et = EdgeTracker(max_tracked_seeds=CEILING)
    assert et.evict_tie is EvictTie.AGE
    et, narrow = _tracker(EvictTie.AGE)
    gone = _evicted(et, narrow)
    kept = [i for i in range(N_NARROW) if i not in gone]
    assert gone and max(gone) < min(kept)


def test_loss_dominates_cost():
    """Adversarial: a cheap seed owning an edge outlives costly redundant ones."""
    et = EdgeTracker(max_tracked_seeds=CEILING)
    et.evict_tie = EvictTie.COST
    et.record_edges("wide", WIDE_EDGES, exec_s=1.0)
    et.record_edges("owner", {UNIQUE_EDGE}, exec_s=1e-9)
    for i in range(N_NARROW):
        et.record_edges(f"n{i}", {1 + i}, exec_s=1.0 + i)
    assert "owner" in et.seed_edges
    assert "wide" in et.seed_edges


def test_costs_pruned_with_seeds():
    """Memory bound: no cost entry outlives its seed."""
    et, _ = _tracker(EvictTie.COST)
    assert set(et.seed_cost) <= set(et.seed_edges)


def test_age_records_no_cost():
    """AGE keeps no cost map at all."""
    et, _ = _tracker(EvictTie.AGE)
    assert not et.seed_cost


def test_cli_flag():
    out = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"], capture_output=True, text=True
    ).stdout
    assert "--prune-cost" in out


@pytest.mark.parametrize("tie", list(EvictTie))
def test_fuzzer_wires_tie(tie, tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "s").write_bytes(b"x")
    f = Fuzzer(
        target="/bin/true",
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "k"),
        use_coverage=False,
        prune_cost=tie is EvictTie.COST,
    )
    assert f._edge_tracker.evict_tie is tie
