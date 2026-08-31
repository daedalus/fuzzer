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
    the corpus. Seeds with fewer than k such edges are removed. After each
    removal round, the unique-edge counts are recomputed from scratch, and the
    process repeats until no seed changes state.

    The result is the k-rigid core: the smallest corpus where every seed has
    at least k singleton edges.

    Args:
        corpus: list of seed bytes (e.g. ``f.corpus``).
        edge_tracker: EdgeTracker with ``seed_edges`` populated
            (``dict[seed_key -> set[edge_id]]``).
        k: Minimum unique edges required to keep a seed. Default 1.

    Returns:
        ``(kept, removed)`` tuple of byte lists. ``kept`` is the k-rigid core.
    """
    if not corpus:
        return [], []

    if not edge_tracker.seed_edges:
        return list(corpus), []

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


def _seed_key(seed: bytes) -> str:
    """Hash a seed to a 16-char hex string for EdgeTracker lookup."""
    import xxhash

    return xxhash.xxh64(seed).hexdigest()[:16]
