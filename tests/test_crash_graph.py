"""Igor crash graphs: builders, WL subtree kernel, threshold linkage (E4)."""

import itertools
import math

import numpy as np
import pytest

from fuzzer_tool.core.crash_graph import (
    cluster_kernel,
    graph_from_edges,
    graph_from_fires,
    kernel_matrix,
    wl_features,
)


def _cosine(a: dict, b: dict) -> float:
    """Reference cosine over sparse count dicts, independent of the module."""
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class TestBuilders:
    def test_fires_nodes_and_arcs(self):
        # 1 -> 2 -> 1 -> 3: three nodes, arcs (1,2) (2,1) (1,3).
        g = graph_from_fires([1, 2, 1, 3])

        assert g.labels == [1, 2, 3]
        arcs = {(g.labels[s], g.labels[d]) for s, nxt in enumerate(g.succ) for d in nxt}
        assert arcs == {(1, 2), (2, 1), (1, 3)}

    def test_edges_have_no_arcs(self):
        g = graph_from_edges([5, 3, 5])

        assert sorted(g.labels) == [3, 5]
        assert all(not s for s in g.succ)

    def test_adversarial_unbounded_fire_log_is_capped(self):
        # An endless fire stream must not hang or grow memory past the cap.
        g = graph_from_fires(itertools.count(), cap=4)

        assert g.labels == [0, 1, 2, 3]


class TestKernel:
    def test_identity_is_one(self):
        f = wl_features(graph_from_fires([1, 2, 3, 1]))

        assert kernel_matrix([f, f])[0, 1] == pytest.approx(1.0)

    def test_falsification_disjoint_labels_are_zero(self):
        a = wl_features(graph_from_fires([1, 2, 3]))
        b = wl_features(graph_from_fires([7, 8, 9]))

        assert kernel_matrix([a, b])[0, 1] == pytest.approx(0.0)

    def test_edge_sets_share_two_of_three(self):
        # Arc-less nodes relabel 1:1 every round, so each round repeats round 0: 2/3.
        a = wl_features(graph_from_edges([1, 2, 3]), iters=2)
        b = wl_features(graph_from_edges([1, 2, 4]), iters=2)

        assert kernel_matrix([a, b])[0, 1] == pytest.approx(2 / 3)

    def test_structure_separates_same_edge_set(self):
        # Same edges, different order: round 0 cannot tell, round 1 can.
        a = graph_from_fires([1, 2, 3])
        b = graph_from_fires([1, 3, 2])

        flat = kernel_matrix([wl_features(a, iters=0), wl_features(b, iters=0)])[0, 1]
        deep = kernel_matrix([wl_features(a, iters=1), wl_features(b, iters=1)])[0, 1]

        assert flat == pytest.approx(1.0)
        assert deep < flat

    def test_adversarial_empty_graph(self):
        # An uninstrumented target yields no edges: similar to nothing, itself 1.
        e = wl_features(graph_from_edges([]))
        f = wl_features(graph_from_fires([1, 2]))

        k = kernel_matrix([e, e, f])

        assert k[0, 1] == 0.0
        assert k[0, 2] == 0.0
        assert np.allclose(np.diag(k), 1.0)

    def test_matches_reference_cosine(self):
        graphs = [
            graph_from_fires([1, 2, 3, 1, 4]),
            graph_from_fires([1, 2, 3, 5]),
            graph_from_fires([4, 1, 2]),
            graph_from_edges([1, 2, 3]),
            graph_from_fires([9, 8]),
        ]
        feats = [wl_features(g) for g in graphs]
        n = len(feats)

        def ref(fs):
            return np.array([[_cosine(fs[i], fs[j]) for j in range(n)] for i in range(n)])

        # Control (Hard Rule 46): the reference against itself on reordered dicts.
        reordered = [dict(reversed(list(f.items()))) for f in feats]
        assert np.allclose(ref(feats), ref(reordered))

        got = kernel_matrix(feats)
        assert np.allclose(got, ref(feats))


class TestCluster:
    def test_threshold_groups(self):
        sim = np.array([[1, 0.9, 0.1], [0.9, 1, 0.2], [0.1, 0.2, 1]])

        assert cluster_kernel(sim, 0.5) == [[0, 1], [2]]

    def test_threshold_is_inclusive(self):
        sim = np.array([[1, 0.5], [0.5, 1]])

        assert cluster_kernel(sim, 0.5) == [[0, 1]]

    def test_single_linkage_chains(self):
        # 0~1 and 1~2 merge 0 with 2 although 0 and 2 are dissimilar.
        sim = np.array([[1, 0.8, 0.0], [0.8, 1, 0.8], [0.0, 0.8, 1]])

        assert cluster_kernel(sim, 0.7) == [[0, 1, 2]]

    def test_empty(self):
        assert cluster_kernel(np.zeros((0, 0)), 0.5) == []
