"""Fused ``_EdgeCounts.add``: one slot lookup pass, ``difference`` only for new edges.

The fused path hashed every edge twice per exec (``edges.difference(slot)``
then ``itemgetter``): 1.3 vs 0.93 ms at 12k edges. New edges are rare after
warmup, so the lookup now runs first and falls back on KeyError.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.shapley import FUSED_MIN_WORK, _EdgeCounts

OPS = ("flip", "splice")


class _NoDiff(set):
    """Edge set whose ``difference`` must not run (all edges already slotted)."""

    def difference(self, *_a):
        raise AssertionError("difference on known edges")


def _wide(start: int) -> set[int]:
    return set(range(start, start + FUSED_MIN_WORK))


def _pairs(c: _EdgeCounts) -> dict[tuple[str, int], int]:
    """Oracle-side readout: every nonzero (op, edge) count."""
    c.fold()
    return {
        (op, c.ids[s]): int(c.mat[lane, s])
        for op, lane in c.lanes.items()
        for s in np.flatnonzero(c.mat[lane, : len(c.ids)])
    }


def test_known_edges_skip_difference():
    c = _EdgeCounts()
    c.add(OPS, _wide(0))
    c.add(OPS, _NoDiff(_wide(0)))
    assert int(c.total[c.slot[5]]) == 2


def test_mixed_new_edges_counted():
    """Adversarial: half-new wide exec counts every edge exactly once."""
    c = _EdgeCounts()
    c.add(OPS, _wide(0))
    c.add(OPS[:1], _wide(FUSED_MIN_WORK // 2))
    got = _pairs(c)
    for e in _wide(0) | _wide(FUSED_MIN_WORK // 2):
        first, second = e < FUSED_MIN_WORK, e >= FUSED_MIN_WORK // 2
        assert got.get(("flip", e), 0) == first + second
        assert got.get(("splice", e), 0) == int(first)
        assert int(c.total[c.slot[e]]) == first + second


def test_single_new_edge():
    """Falsification edge: a lone unseen edge with many ops still lands."""
    c = _EdgeCounts()
    ops = [f"op{i}" for i in range(FUSED_MIN_WORK)]
    c.add(ops, {1 << 40})
    assert int(c.total[c.slot[1 << 40]]) == 1
    assert _pairs(c)[("op7", 1 << 40)] == 1
