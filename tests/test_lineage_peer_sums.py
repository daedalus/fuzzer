"""Tests for ``LineageTree.peer_distance_sums`` and the O(n) ``_lca_diversity``.

``_lca_diversity`` built 64 (seed, peer) pairs per seed and ran Tarjan's
offline LCA over all of them: 614 ms per weights pass at 4,000 seeds
(5.5% of a GStreamer ``--elo all`` campaign). Per-subtree peer counts and a
top-down prefix sum give every seed's distance total to a fixed peer set in
O(n); the multiplier must be bit-identical to the pair-based one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.seed_picker import SeedPicker

from .test_lineage_batch_lca import _build_chain_like_tree, _build_random_forest

SAMPLE_CAP = 64
SIZES = (3, 40, 66, 300, 1200)


def _reference(tree, all_sk, max_depth, rng):
    """Pre-change ``_lca_diversity`` (pairs + batch LCA), verbatim logic."""
    n_sk = len(all_sk)
    pool_idx = rng.sample(n_sk, SAMPLE_CAP + 1) if n_sk > SAMPLE_CAP + 1 else range(n_sk)
    pool_idx = list(pool_idx)
    pairs, samples = [], []
    for i, sk_i in enumerate(all_sk):
        sample = [all_sk[j] for j in pool_idx if j != i][:SAMPLE_CAP]
        samples.append(sample)
        pairs.extend((sk_i, k) for k in sample)
    dist = tree.batch_lca_distances(pairs)
    out = {}
    for i, sk_i in enumerate(all_sk):
        valid = [d for d in (dist[(sk_i, k)] for k in samples[i]) if d >= 0]
        avg = sum(valid) / len(valid) if valid else 0.0
        out[sk_i] = 1.0 + 0.5 * min(avg / (2.0 * max_depth), 1.0)
    return out


def _new(tree, all_sk, max_depth, seed):
    sp = SeedPicker.__new__(SeedPicker)
    sp.f = SimpleNamespace(_rng=RandPool(seed=seed))
    return sp._lca_diversity(tree, all_sk, max_depth)


def _trees():
    for n in SIZES:
        yield _build_chain_like_tree(n, seed=n)
        yield _build_random_forest(n, n_roots=max(1, n // 50), seed=n)


def _max_depth(tree):
    return max(nd.depth for nd in tree.nodes.values())


def test_reference_matches_itself():
    """Control (Hard Rule 46): the oracle is deterministic per seed."""
    tree, keys = _build_chain_like_tree(300, seed=1)
    md = _max_depth(tree)
    assert _reference(tree, keys, md, RandPool(seed=5)) == _reference(
        tree, keys, md, RandPool(seed=5)
    )


@pytest.mark.parametrize("seed", range(3))
def test_matches_pair_reference(seed):
    """Bit-identical multipliers on chains and forests, members and non-members."""
    for tree, keys in _trees():
        md = _max_depth(tree)
        assert _new(tree, keys, md, seed) == _reference(tree, keys, md, RandPool(seed=seed))


def test_peer_sums_match_batch():
    """Per-node totals equal summed batch distances over a fixed peer list."""
    tree, keys = _build_random_forest(200, n_roots=4, seed=9)
    peers = keys[::7] + keys[:2]  # duplicates count twice
    sums = tree.peer_distance_sums(peers)
    dist = tree.batch_lca_distances([(k, p) for k in keys for p in peers])
    for k in keys:
        valid = [dist[(k, p)] for p in peers if dist[(k, p)] >= 0]
        assert sums.get(k, (0, 0)) == (sum(valid), len(valid))


def test_far_seed_scores_higher():
    """Falsification: a seed on a long separate branch outscores one beside the pool."""
    tree = LineageTree()
    tree.insert(None, "r", [], [], 0)
    prev = "r"
    for i in range(20):  # the pool lives on this branch
        tree.insert(prev, f"a{i}", [], [], 0)
        prev = f"a{i}"
    prev = "r"
    for i in range(20):  # a far branch
        tree.insert(prev, f"b{i}", [], [], 0)
        prev = f"b{i}"
    keys = [f"a{i}" for i in range(20)] + ["b19"]
    div = _new(tree, keys, _max_depth(tree), 0)
    assert div["b19"] > div["a10"]


def test_cycle_and_unknown_keys():
    """Adversarial: a parent cycle and unknown keys neither hang nor break equality."""
    tree, keys = _build_chain_like_tree(120, seed=3)
    a, b = keys[50], keys[51]
    tree.nodes[a].parent_key = b  # a <-> b cycle
    tree.nodes[b].parent_key = a
    keys = keys + ["ghost1", "ghost2"]
    md = _max_depth(tree)
    assert _new(tree, keys, md, 4) == _reference(tree, keys, md, RandPool(seed=4))
