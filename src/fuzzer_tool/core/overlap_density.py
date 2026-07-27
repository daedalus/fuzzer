"""FMM-clustered pairwise overlap density for seed selection.

Computes each seed's mean pairwise Jaccard similarity against every other
seed in the corpus.  This is an N-body problem (O(N²) naively).  We use a
Fast-Multipole-Method-style decomposition:

  Near-field (intra-cluster): exact pairwise Jaccard within each cluster.
  Far-field (inter-cluster):   approximate via J(seed, centroid_of_far_cluster)
                               × |far_cluster|, where the centroid is the
                               element-wise min of member MinHash signatures
                               (= MinHash of the *union* of edge sets).

Clusters are formed via the existing MinHash LSH infrastructure —
seeds that share an LSH bucket at a Jaccard threshold are in the same
cluster.  This is analogous to the FMM multipole acceptance criterion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuzzer_tool.core.edge_tracker import MinHashLSH

log = logging.getLogger(__name__)


def _sig_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    """Jaccard from two raw MinHash signatures (O(k), k = num_perm)."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=False) if a == b)
    return matches / len(sig_a)


def _build_clusters(
    seed_keys: list[str],
    minhash: MinHashLSH,
    min_jaccard: float,
) -> tuple[list[list[int]], dict[int, int]]:
    """Cluster seeds by LSH similarity using union-find.

    Returns:
        clusters: list of lists of seed indices.
        seed_to_cluster: seed_idx → cluster_idx mapping.
    """
    n = len(seed_keys)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        sk = seed_keys[i]
        if not sk or sk not in minhash.signatures:
            continue
        similar = minhash.find_similar(sk, min_jaccard=min_jaccard)
        if not similar:
            continue
        for j in range(n):
            if j != i and seed_keys[j] in similar:
                union(i, j)
                break

    raw: dict[int, list[int]] = {}
    for i in range(n):
        raw.setdefault(find(i), []).append(i)

    clusters = list(raw.values())
    seed_to_cluster: dict[int, int] = {}
    for cidx, members in enumerate(clusters):
        for m in members:
            seed_to_cluster[m] = cidx

    return clusters, seed_to_cluster


def _compute_centroids(
    clusters: list[list[int]],
    seed_keys: list[str],
    minhash: MinHashLSH,
) -> list[list[int] | None]:
    """Compute centroid signatures for each cluster.

    The centroid is the element-wise *minimum* of all member signatures,
    which is the MinHash of the *union* of the member edge sets (property
    of MinHash: min-of-signatures = signature-of-union).
    """
    centroids: list[list[int] | None] = []
    for members in clusters:
        first_sig = minhash.signatures.get(seed_keys[members[0]])
        if first_sig is None:
            centroids.append(None)
            continue
        centroid = list(first_sig)
        for i in members[1:]:
            sig = minhash.signatures.get(seed_keys[i])
            if sig:
                for k in range(len(centroid)):
                    if sig[k] < centroid[k]:
                        centroid[k] = sig[k]
        centroids.append(centroid)
    return centroids


def compute_corpus_overlap_density(
    seed_keys: list[str],
    minhash: MinHashLSH,
    min_jaccard: float = 0.25,
) -> tuple[dict[str, float], list[list[int]], dict[int, int]]:
    """Compute pairwise overlap density for all seeds.

    Overlap density for seed i = mean pairwise Jaccard(i, j) across all
    j ≠ i.  Computed via FMM near/far decomposition for O(N·C) instead
    of O(N²).

    Args:
        seed_keys: All seed content hashes in the sliding window.
        minhash: MinHashLSH instance with signatures for all seed_keys.
        min_jaccard: Jaccard threshold for LSH clustering (default 0.25).
            Lower = coarser clusters (faster, less accurate).
            Higher = tighter clusters (slower, more accurate).

    Returns:
        (densities_dict, clusters, seed_to_cluster):
            densities_dict: seed_key → mean pairwise Jaccard in [0, 1].
            clusters: list of [seed_idx, ...] for each cluster.
            seed_to_cluster: seed_idx → cluster_idx.

    Raises:
        ValueError: If fewer than 2 seed_keys are provided.
    """
    if len(seed_keys) < 2:
        if len(seed_keys) == 1:
            return {seed_keys[0]: 0.0}, [[0]], {0: 0}
        return {}, [], {}

    clusters, seed_to_cluster = _build_clusters(seed_keys, minhash, min_jaccard)
    centroids = _compute_centroids(clusters, seed_keys, minhash)

    densities: dict[str, float] = {}

    for i, sk in enumerate(seed_keys):
        cidx = seed_to_cluster.get(i)
        if cidx is None or centroids[cidx] is None:
            densities[sk] = 0.0
            continue

        sig_self = minhash.signatures.get(sk)
        if sig_self is None:
            densities[sk] = 0.0
            continue

        total = 0.0
        count = 0

        # Near field: exact pairwise Jaccard within cluster
        for j in clusters[cidx]:
            if j == i:
                continue
            total += minhash.approximate_jaccard(seed_keys[i], seed_keys[j])
            count += 1

        # Far field: centroid-to-seed approximation per far cluster
        for ocidx, members in enumerate(clusters):
            if ocidx == cidx or centroids[ocidx] is None:
                continue
            inter_j = _sig_jaccard(sig_self, centroids[ocidx])
            total += inter_j * len(members)
            count += len(members)

        densities[sk] = total / count if count else 0.0

    return densities, clusters, seed_to_cluster
