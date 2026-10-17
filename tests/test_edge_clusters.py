"""core/edge_clusters.py: incidence classes, near-duplicate clusters, near-misses."""

import numpy as np
import pytest

from fuzzer_tool.core import edge_clusters as mod
from fuzzer_tool.core.edge_clusters import CLUSTER_JACCARD, RARE_OWNERS, EdgeClusters
from fuzzer_tool.core.rand_pool import RandPool

# A = {1,2,3} on s0..s3 (4 owners); B = {5} on s0..s2 (3 owners, J(A,B) = 3/4);
# C = {9,10,11} on s4 only: one rare block of three edges.
SEEDS = {
    "s0": {1, 2, 3, 5},
    "s1": {1, 2, 3, 5},
    "s2": {1, 2, 3, 5},
    "s3": {1, 2, 3},
    "s4": {9, 10, 11},
}


def _fit(seeds=SEEDS) -> EdgeClusters:
    c = EdgeClusters()
    c.refit(seeds)
    return c


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def test_stats_collapse_blocks():
    st = _fit().stats()
    assert st == {
        "edges": 7,
        "classes": 3,
        "clusters": 2,
        "singleton_edges": 3,
        "singleton_classes": 1,
        "largest_class": 3,
    }


def test_rare_classes_count_blocks_once():
    c = _fit()
    # Per edge, s4 holds three rare edges; per class, one.
    assert c.rare_classes(SEEDS["s4"]) == 1
    assert c.rare_classes(SEEDS["s0"]) == 1  # B (3 owners <= RARE_OWNERS)
    assert c.rare_classes({999}) == 0  # unseen edges are ignored


def test_unique_fraction_by_class():
    c = _fit()
    assert c.unique_fraction(SEEDS["s4"]) == 1.0
    assert c.unique_fraction(SEEDS["s0"]) == 0.0
    assert c.unique_fraction(set()) == 0.0


def test_near_misses_mark_partial_clusters():
    c = _fit()
    # s3 is on cluster {A, B} but lacks rare B: one branch away.
    assert {k: c.near_misses(k) for k in SEEDS} == {"s0": 0, "s1": 0, "s2": 0, "s3": 1, "s4": 0}
    assert c.near_misses("unknown") == 0


def test_class_owner_counts():
    assert sorted(_fit().class_owner_counts().tolist()) == [1, 3, 4]


def test_leader_clustering_does_not_chain():
    # Adversarial: J(A,B)=0.8, J(B,C)=0.75, J(A,C)=0.6. Single linkage at 0.75
    # merges all three; members must reach the threshold against their leader.
    seeds = {f"s{i}": set() for i in range(10)}
    for i in range(10):
        seeds[f"s{i}"].add(100)  # A on 10 seeds
    for i in range(8):
        seeds[f"s{i}"].add(200)  # B on 8
    for i in range(6):
        seeds[f"s{i}"].add(300)  # C on 6
    c = _fit(seeds)
    assert c.stats()["clusters"] == 2
    assert c.cluster_of(100) == c.cluster_of(200) != c.cluster_of(300)


def test_cluster_members_reach_threshold_vs_leader():
    # Falsification on random incidence: recompute every member's Jaccard to
    # its cluster's leader from the raw sets.
    # Planted structure: 8 features, 6 edges each; an edge's owners are the
    # feature's owners minus up to 2 dropped seeds.
    rng = RandPool(7)
    seeds = {f"s{i}": set() for i in range(40)}
    for feat in range(8):
        base = rng.sample(sorted(seeds), 20)
        for j in range(6):
            dropped = set(rng.sample(base, rng.randint(0, 2)))
            for k in base:
                if k not in dropped:
                    seeds[k].add(feat * 100 + j)
    c = _fit(seeds)
    assert c.stats()["clusters"] < c.stats()["classes"]  # not vacuous
    owners = {e: {k for k, s in seeds.items() if e in s} for e in set().union(*seeds.values())}
    for e, own in owners.items():
        leader = c.cluster_of(e)
        assert _jaccard(own, owners[leader]) >= CLUSTER_JACCARD - 1e-12


def test_over_budget_skips_clustering(monkeypatch):
    monkeypatch.setattr(mod, "MAX_CLUSTER_CLASSES", 2)
    c = _fit()
    assert c.stats()["clusters"] == c.stats()["classes"]
    assert c.near_misses("s3") == 0


def test_empty_refit():
    c = _fit({})
    assert c.stats()["edges"] == 0
    assert c.rare_classes({1}) == 0
    assert c.class_owner_counts().size == 0


def test_refit_replaces_previous_fit():
    c = _fit()
    c.refit({"a": {1}, "b": {1}, "c": {2}})
    assert c.stats()["classes"] == 2
    assert c.near_misses("s3") == 0


@pytest.mark.parametrize("threshold", [RARE_OWNERS])
def test_rare_threshold_matches_owner_counts(threshold):
    c = _fit()
    edges = set().union(*SEEDS.values())
    owners = {e: sum(e in s for s in SEEDS.values()) for e in edges}
    rare_heads = {c.cluster_of(e) for e in edges if owners[e] <= threshold}
    assert c.rare_classes(edges) == len({c.class_of(e) for e in edges if owners[e] <= threshold})
    assert len(rare_heads) >= 1
    assert np.all(c.class_owner_counts() >= 1)
