"""Unit tests for core.causal_sector primitives (no wiring)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.causal_sector import CausalSectorGraph


class TestCausalSectorGraph:
    def test_empty_snapshot(self):
        g = CausalSectorGraph(stability_windows=2)
        snap = g.snapshot()
        assert snap.n_nodes == 0
        assert snap.n_edges == 0
        assert snap.stable is False
        assert snap.observation_windows == 0
        assert snap.top_edges == []

    def test_observe_pair_and_te(self):
        g = CausalSectorGraph()
        g.observe_pair(1, 2, 0.5)
        assert g.te(1, 2) == pytest.approx(0.5)
        assert g.te(2, 1) == 0.0
        assert len(g) == 1
        assert g.successors(1) == [(2, pytest.approx(0.5))]
        assert g.predecessors(2) == [(1, pytest.approx(0.5))]

    def test_ignores_self_and_tiny_te(self):
        g = CausalSectorGraph(min_te=0.01)
        g.observe_pair(1, 1, 1.0)
        g.observe_pair(1, 2, 0.001)
        assert len(g) == 0

    def test_observe_flow_batch(self):
        g = CausalSectorGraph(stability_windows=1, min_asymmetry=0.05)
        flow = {(10, 20): 0.4, (20, 10): 0.05, (30, 40): 0.3}
        g.observe_flow(flow)
        assert g.te(10, 20) == pytest.approx(0.4)
        assert g.observation_windows == 1
        assert g.aligns(10, 20) is True
        assert g.aligns(20, 10) is False

    def test_stability_requires_consistent_windows(self):
        g = CausalSectorGraph(stability_windows=3, min_asymmetry=0.1, decay=1.0)
        flow = {(1, 2): 0.5, (2, 1): 0.0}
        g.observe_flow(flow)
        assert g.stable is False
        g.observe_flow(flow)
        assert g.stable is False
        g.observe_flow(flow)
        assert g.stable is True
        assert g.observation_windows == 3

    def test_flip_breaks_stability(self):
        g = CausalSectorGraph(stability_windows=2, min_asymmetry=0.1, decay=1.0)
        g.observe_flow({(1, 2): 0.5, (2, 1): 0.0})
        g.observe_flow({(1, 2): 0.5, (2, 1): 0.0})
        assert g.stable is True
        # reverse orientation
        g.observe_flow({(1, 2): 0.0, (2, 1): 0.5})
        assert g.stable is False

    def test_mean_asymmetry(self):
        g = CausalSectorGraph(decay=1.0)
        g.observe_pair(1, 2, 0.4)
        g.observe_pair(2, 1, 0.1)
        assert g.mean_asymmetry() == pytest.approx(0.3)

    def test_caps_max_edges(self):
        g = CausalSectorGraph(max_nodes=100, max_edges=5, min_te=0.0)
        for i in range(20):
            g.observe_pair(i, i + 100, 0.1 + i * 0.01)
        assert len(g) <= 5

    def test_clear(self):
        g = CausalSectorGraph()
        g.observe_flow({(1, 2): 0.3})
        g.clear()
        assert len(g) == 0
        assert g.stable is False
        assert g.observation_windows == 0

    def test_top_edges_order(self):
        g = CausalSectorGraph()
        g.observe_pair(1, 2, 0.1)
        g.observe_pair(3, 4, 0.9)
        g.observe_pair(5, 6, 0.5)
        top = g.top_edges(2)
        assert top[0][:2] == (3, 4)
        assert top[1][:2] == (5, 6)

    def test_invalid_decay(self):
        with pytest.raises(ValueError):
            CausalSectorGraph(decay=0.0)
        with pytest.raises(ValueError):
            CausalSectorGraph(max_nodes=0)
