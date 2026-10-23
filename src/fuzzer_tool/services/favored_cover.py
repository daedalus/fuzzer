"""AFL top_rated / favored greedy cover over dense edge index arrays.

``Fuzzer._cull_queue`` walked every (seed, edge) pair through dicts: ffmpeg
--lightweight, 4k seeds, 2.7M pairs per call, 1.7 s each (77 s of a 459 s
run). Same cover here from numpy sorts over ``_EdgeVectors`` index arrays,
which the seed picker already caches per seed set.

    pairs (edge, seed rank) --minimum.at--> cheapest seed per edge (top_rated)
    edges --lexsort (owners, id)--> rarest first --greedy--> favored seeds
"""

import numpy as np


def favored_cover(keys, sets, costs, vectors, owner_count) -> set[str]:
    """Favored seed keys: the greedy rarest-edge-first cover.

    Args:
        keys: Seed keys, in the tracker's iteration order.
        sets: Each key's edge set.
        costs: Each key's cost (exec_us * input_size).
        vectors: ``_EdgeVectors``: per-set dense indices and their edge ids.
        owner_count: Edge id -> number of seeds covering it.

    Returns:
        The keys the cover picked. Ties keep the loop's rules: the cheapest
        seed owns an edge, the first in *keys* order on equal cost; edges
        are visited by (owner count, id).
    """
    idx = [vectors.indices(s) for s in sets]
    lens = np.fromiter(map(len, idx), dtype=np.int64, count=len(idx))
    if not lens.any():
        return set()

    # top_rated: per edge, the best-ranked seed holding it, ranked by
    # (cost, position in keys); np.minimum.at, not a 3-key lexsort over
    # every pair (1.2 s at 5M pairs vs 34 ms).
    n = len(idx)
    by_rank = np.lexsort((np.arange(n), np.asarray(costs, dtype=np.float64)))
    rank = np.empty(n, dtype=np.int32)  # int32: halves the per-pair array
    rank[by_rank] = np.arange(n, dtype=np.int32)
    best = np.full(len(vectors), n, dtype=np.int32)
    np.minimum.at(best, np.concatenate(idx), np.repeat(rank, lens))
    edges = np.flatnonzero(best < n)
    winner = by_rank[best[edges]]

    # Rarest edge first; the edge id breaks ties.
    ids = vectors.id_array()[edges]
    owners = np.fromiter(map(owner_count, ids.tolist()), dtype=np.int64, count=ids.size)
    visit = np.lexsort((ids, owners))

    # Greedy: an uncovered edge pulls its winner in, covering its whole set.
    covered = np.zeros(len(vectors), dtype=bool)
    seen = memoryview(covered)
    favored: set[str] = set()
    for e, k in zip(edges[visit].tolist(), winner[visit].tolist(), strict=True):
        if seen[e]:
            continue
        favored.add(keys[k])
        covered[idx[k]] = True
    return favored
