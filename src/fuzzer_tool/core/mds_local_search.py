"""Weighted bounded local-search Maximum Disjoint Set (Chan & Har-Peled 2012).

Corpus minimization currently treats near-duplicate pressure and seed value
as two separate concerns: `PoissonDiskAdmission` (services/corpus_manager.py)
rejects incoming seeds that fall inside a *fixed* Jaccard radius of an
already-admitted seed, and `auto_minimize_corpus` separately ranks the
survivors by a composite value score and keeps the top-K. Neither view lets
a seed's *value* change how close a neighbor is allowed to be.

This module treats each seed as a disk in Jaccard-distance space whose
radius shrinks for high-value seeds (they tolerate closer neighbors and
pack densely) and grows for low-value seeds (they need more clearance to
be worth the slot). Picking a maximum-weight set of non-overlapping disks
is exactly the weighted "fat objects of arbitrary sizes" Maximum Disjoint
Set problem:
https://en.wikipedia.org/wiki/Maximum_disjoint_set#Fat_objects_with_arbitrary_sizes:_PTAS

The grid-based PTASs on that page (Erlebach-Jansen-Seidel, Chan's shifted
quadtree) need actual 2D coordinates, which MinHash signatures don't have.
The bounded local-search algorithm (Chan & Har-Peled, "Approximation
Algorithms for Maximum Independent Set of Pseudo-Disks", 2012) only needs
a pairwise conflict predicate, which `MinHashLSH.approximate_jaccard`
already gives us for free -- no embedding required.

Algorithm: start from a weight-sorted greedy independent set. Repeatedly
look for a small conflict-free group of currently-excluded candidates
(size 1..c) whose combined weight beats the weight of every selected item
they'd have to displace, and swap it in. Every accepted swap strictly
increases total weight, so the loop terminates; `max_rounds` bounds work
on large corpora where chasing the exact fixed point isn't worth it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

# Brackets the old PoissonDiskAdmission fixed radius (0.25): a corpus of
# uniform-score seeds maps every radius to the midpoint of this range, so
# behavior degrades gracefully toward the old fixed-radius admission gate
# rather than diverging from it.
DEFAULT_R_MIN = 0.12
DEFAULT_R_MAX = 0.40

# Local search is n^O(c) in the number of excluded candidates considered
# per round (Wikipedia's own bound for this algorithm family). Capping the
# excluded pool to the highest-weight stragglers keeps each round bounded
# without changing the outcome in the common case, since a low-weight
# excluded candidate essentially never wins a swap against a higher-weight
# selected neighbor anyway.
DEFAULT_CANDIDATE_LIMIT = 400


def disk_radius(
    score: float,
    score_min: float,
    score_max: float,
    r_min: float = DEFAULT_R_MIN,
    r_max: float = DEFAULT_R_MAX,
) -> float:
    """Map a seed's value score to an exclusion radius in Jaccard-distance space.

    Higher score -> smaller radius (packs densely, tolerates near
    neighbors). Lower score -> larger radius (needs more clearance to
    justify a slot). Seeds with equal scores all get the same radius, the
    midpoint of [r_min, r_max].
    """
    if score_max <= score_min:
        return (r_min + r_max) / 2.0
    frac = (score - score_min) / (score_max - score_min)
    frac = min(1.0, max(0.0, frac))
    return r_max - frac * (r_max - r_min)


def _conflicts(
    key_a: str,
    key_b: str,
    radius: dict[str, float],
    jaccard_fn: Callable[[str, str], float],
) -> bool:
    """Disk-intersection test: two seeds conflict when the Jaccard *distance*
    between their signatures is smaller than the sum of their radii -- the
    usual "distance between centers < sum of radii" disk overlap test, with
    (1 - Jaccard) standing in for Euclidean distance.
    """
    if key_a == key_b:
        return True
    d = 1.0 - jaccard_fn(key_a, key_b)
    return d < (radius.get(key_a, 0.0) + radius.get(key_b, 0.0))


@dataclass
class MDSResult:
    selected: list[str] = field(default_factory=list)
    rounds: int = 0
    swaps: int = 0


def local_search_mds(
    keys: list[str],
    weight: dict[str, float],
    radius: dict[str, float],
    jaccard_fn: Callable[[str, str], float],
    c: int = 2,
    max_rounds: int = 4,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> MDSResult:
    """Bounded local-search weighted MDS over an arbitrary conflict graph.

    `jaccard_fn(key_a, key_b) -> float` is the only geometric primitive
    needed -- no coordinates. Returns the selected subset (unordered) plus
    round/swap counts for logging.
    """
    order = sorted(keys, key=lambda k: weight.get(k, 0.0), reverse=True)

    selected: list[str] = []
    for k in order:
        if not any(_conflicts(k, s, radius, jaccard_fn) for s in selected):
            selected.append(k)

    selected_set = set(selected)
    excluded = [k for k in order if k not in selected_set]

    swaps = 0
    rounds = 0
    improved = True
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        pool = excluded[:candidate_limit]
        for size in range(1, c + 1):
            for group in combinations(pool, size):
                if any(
                    _conflicts(a, b, radius, jaccard_fn) for a, b in combinations(group, 2)
                ):
                    continue  # group isn't itself conflict-free
                conflicting_selected = {
                    s for s in selected if any(_conflicts(g, s, radius, jaccard_fn) for g in group)
                }
                gain = sum(weight.get(g, 0.0) for g in group)
                cost = sum(weight.get(s, 0.0) for s in conflicting_selected)
                if gain > cost:
                    selected = [s for s in selected if s not in conflicting_selected] + list(group)
                    selected_set = set(selected)
                    excluded = [k for k in order if k not in selected_set]
                    swaps += 1
                    improved = True
                    break
            if improved:
                break

    return MDSResult(selected=selected, rounds=rounds, swaps=swaps)
