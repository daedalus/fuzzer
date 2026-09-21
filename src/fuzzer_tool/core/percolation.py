"""Percolation primitives for coverage-guided fuzzing.

Central home for percolation-framed types and algorithms:

- ``CoverageRegime`` enum — phase labels (subcritical / critical / supercritical)
  used by both the coverage regime detector and the bootstrap minimizer.
- ``bootstrap_minimize_corpus()`` — iterative k-rigid-core reduction that
  captures transitive redundancy single-pass greedy set-cover misses.
"""

import enum


class CoverageRegime(enum.Enum):
    """Percolation phase of coverage exploration.

    - SUBCRITICAL    — discovery rate decays exponentially; fuzzer stuck in isolated clusters
    - CRITICAL       — power-law regime; maximum sensitivity; near a coverage jump
    - SUPERCRITICAL  — compounding discovery; each input unlocks multiple paths
    """

    SUBCRITICAL = "subcritical"
    CRITICAL = "critical"
    SUPERCRITICAL = "supercritical"


def bootstrap_minimize_corpus(
    corpus: list[bytes],
    edge_tracker,
    k: int = 1,
) -> tuple[list[bytes], list[bytes]]:
    """Iteratively remove seeds with < k unique edges to fixed point.

    A seed's "unique edges" are those covered by no other seed currently in
    the corpus. Seeds with fewer than k such edges are removed, and the
    unique-edge counts are recomputed after removal, until no seed changes
    state.

    Two removal disciplines, chosen by ``k``:

    - ``k == 1`` — **coverage-preserving.** Seeds are removed *one at a time*
      (fewest edges first, ties by corpus order), so two seeds that only cover
      an edge jointly are never both dropped: after the first goes, the
      survivor owns the edge and stays. The union of covered edges is
      unchanged (seeds with no tracked edges contribute nothing and are
      removed up front). Batch removal used to drop such pairs together and
      lose the edge — see ``docs/handover/handover_generators_2026-09-20.md``
      P1-1.
    - ``k >= 2`` — **the k-rigid core.** Every seed below the threshold goes in
      the same round. This is deliberately lossy: it keeps only seeds that
      individually own at least k edges, whether or not coverage survives.

    Args:
        corpus: list of seed bytes (e.g. ``f.corpus``).
        edge_tracker: EdgeTracker with ``seed_edges`` populated
            (``dict[seed_key -> set[edge_id]]``).
        k: Minimum unique edges required to keep a seed. Default 1.

    Returns:
        ``(kept, removed)`` tuple of byte lists. ``kept`` preserves corpus order.
    """
    if not corpus:
        return [], []

    if not edge_tracker.seed_edges:
        return list(corpus), []

    if k == 1:
        return _minimize_preserving_coverage(corpus, edge_tracker)

    kept: list[bytes] = list(corpus)
    removed: list[bytes] = []

    while True:
        # Build edge → seeds map and seed → edges map from current kept set.
        edge_to_seeds: dict[int, set[int]] = {}
        seed_to_edges: dict[int, set[int]] = {}  # seed_index -> edge set

        for idx, seed in enumerate(kept):
            sk = _seed_key(seed)
            edges = edge_tracker.seed_edges.get(sk, set())
            if not edges:
                # Seed has no tracked edges → always removable.
                seed_to_edges[idx] = set()
                continue
            seed_to_edges[idx] = edges
            for e in edges:
                edge_to_seeds.setdefault(e, set()).add(idx)

        # Find seeds with < k unique edges.
        to_remove: set[int] = set()
        for idx, _seed in enumerate(kept):
            if idx not in seed_to_edges:
                # No tracked edges → remove.
                to_remove.add(idx)
                continue

            unique = sum(1 for e in seed_to_edges[idx] if len(edge_to_seeds.get(e, set())) == 1)
            if unique < k:
                to_remove.add(idx)

        if not to_remove:
            break

        # Remove marked seeds (preserve order, remove by index descending).
        for idx in sorted(to_remove, reverse=True):
            removed.append(kept.pop(idx))

    return kept, removed


def _minimize_preserving_coverage(
    corpus: list[bytes],
    edge_tracker,
) -> tuple[list[bytes], list[bytes]]:
    """k == 1 path: sequential removal of seeds that own no edge.

    A seed with zero unique edges has every edge covered by another seed, so
    removing it alone cannot shrink the covered union. Removing it can only
    *raise* other seeds' unique counts (an edge whose owner count falls to one
    becomes that owner's unique edge), so once a seed has a unique edge it
    keeps it, and a lazy min-heap of zero-unique seeds is exact. Total cost is
    O(sum of |edges| * log n).
    """
    import heapq

    n = len(corpus)
    seed_edges = edge_tracker.seed_edges
    edges_of: list[frozenset | set] = []
    owners: dict[int, set[int]] = {}
    for idx, seed in enumerate(corpus):
        edges = seed_edges.get(_seed_key(seed), set())
        edges_of.append(edges)
        for e in edges:
            owners.setdefault(e, set()).add(idx)

    removed_idx: list[int] = []
    alive = [True] * n
    unique = [0] * n
    for idx in range(n):
        unique[idx] = sum(1 for e in edges_of[idx] if len(owners[e]) == 1)

    # Seeds with no tracked edges are always removable and touch no owner set.
    for idx in range(n):
        if not edges_of[idx]:
            alive[idx] = False
            removed_idx.append(idx)

    heap = [(len(edges_of[i]), i) for i in range(n) if alive[i] and unique[i] == 0]
    heapq.heapify(heap)
    while heap:
        _, idx = heapq.heappop(heap)
        if not alive[idx] or unique[idx] != 0:
            continue
        alive[idx] = False
        removed_idx.append(idx)
        for e in edges_of[idx]:
            own = owners[e]
            own.discard(idx)
            if len(own) == 1:
                (sole,) = own
                unique[sole] += 1

    kept = [corpus[i] for i in range(n) if alive[i]]
    removed = [corpus[i] for i in removed_idx]
    return kept, removed


def _seed_key(seed: bytes) -> str:
    """Hash a seed to a 16-char hex string for EdgeTracker lookup."""
    import xxhash

    return xxhash.xxh64(seed).hexdigest()[:16]
