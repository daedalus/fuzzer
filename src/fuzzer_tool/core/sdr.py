"""Distinct representatives for rare edges: Hall's theorem (``minimize --hall``).

Pure function, no fuzzer state (same split as ``core/assignment.py``).
Set cover keeps one seed per edge but lets one seed hold many rare edges,
so losing that seed loses all of them. Here every rare edge (held by at
most *rare_owners* seeds) gets a seed of its own: a maximum bipartite
matching edge -> seed (Hopcroft-Karp).

    seeds  a{1,2}  b{1}        cover: {a}
    rare   1:{a,b}  2:{a}      reps:  1 -> b, 2 -> a

When no such assignment exists, Hall's condition fails: some edge set S
has fewer holders than edges, |N(S)| < |S|. The Kőnig witness is S = the
edges reachable by alternating paths from the unmatched ones; N(S) are the
seeds reached, all matched, so |S| - |N(S)| = number left unmatched.

    a{1,2,3} b{3}  ->  1 -> a, 3 -> b, 2 unmatched; S = {1,2}, N(S) = {a}
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

K = TypeVar("K", bound=Hashable)

UNMATCHED = -1

# Same rarity cut as ``services/seed_picker.RARE_EDGE_OWNERS``.
RARE_OWNERS = 3

TITLE = "Distinct Representatives"

_INF = 1 << 62


@dataclass(frozen=True)
class Reps(Generic[K]):
    """Matching of rare edges to distinct seeds, and its Hall witness."""

    owner: dict[int, K]  # rare edge -> its own seed
    deficient: tuple[int, ...]  # rare edges left without one
    violator: tuple[int, ...]  # Hall witness S
    witness: tuple[K, ...]  # N(S): |S| - |N(S)| = len(deficient)
    n_rare: int


def distinct_reps(
    seed_edges: Mapping[K, np.ndarray],
    rare_owners: int = RARE_OWNERS,
    prefer: Collection[K] = (),
) -> Reps[K]:
    """Maximum matching of rare edges to distinct seeds.

    Args:
        seed_edges: Edge ids per seed key.
        rare_owners: An edge is rare when at most this many seeds hold it.
        prefer: Seeds tried first, e.g. the set cover, so fewer are added.
    """
    keys = list(dict.fromkeys(k for k in prefer if k in seed_edges))
    first = set(keys)
    keys += [k for k in seed_edges if k not in first]
    rare_ids, ptr, nbr = _rare_csr([seed_edges[k] for k in keys], rare_owners)

    match_l, match_r = _hopcroft_karp(ptr, nbr, len(keys))
    free = [u for u, v in enumerate(match_l) if v == UNMATCHED]
    s_left, s_right = _witness(ptr, nbr, match_r, free)

    ids = rare_ids.tolist()
    return Reps(
        owner={ids[u]: keys[v] for u, v in enumerate(match_l) if v != UNMATCHED},
        deficient=tuple(ids[u] for u in free),
        violator=tuple(ids[u] for u in s_left),
        witness=tuple(keys[v] for v in s_right),
        n_rare=len(ids),
    )


def _rare_csr(edges: list[np.ndarray], rare_owners: int) -> tuple[np.ndarray, list[int], list[int]]:
    """Rare edge ids, and their holders as CSR over seed indices."""
    n = len(edges)
    sizes = np.fromiter((e.size for e in edges), dtype=np.int64, count=n)
    if not sizes.sum():
        return np.zeros(0, dtype=np.int64), [0], []

    # Stable sort by edge keeps seeds ascending within each edge. One
    # radix argsort: 10x faster than np.unique on packed (edge, seed) pairs
    # at 10k seeds x 300 edges.
    seed_idx = np.repeat(np.arange(n, dtype=np.int64), sizes)
    cat = np.concatenate(edges)
    order = np.argsort(cat, kind="stable")
    e, s = cat[order], seed_idx[order]

    # Repeated (edge, seed) incidences are adjacent now: drop them.
    keep = np.ones(e.size, dtype=bool)
    keep[1:] = (e[1:] != e[:-1]) | (s[1:] != s[:-1])
    e, s = e[keep], s[keep]

    start = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    held = np.diff(np.r_[start, e.size])
    rare = held <= rare_owners
    ptr = np.zeros(int(rare.sum()) + 1, dtype=np.int64)
    np.cumsum(held[rare], out=ptr[1:])
    return e[start[rare]].astype(np.int64), ptr.tolist(), s[np.repeat(rare, held)].tolist()


def _hopcroft_karp(ptr: list[int], nbr: list[int], n_right: int) -> tuple[list[int], list[int]]:
    """Maximum matching; phases of BFS layering, then disjoint augmenting paths."""
    n_left = len(ptr) - 1
    match_l = [UNMATCHED] * n_left
    match_r = [UNMATCHED] * n_right
    while True:
        dist = _layers(ptr, nbr, match_l, match_r)
        if dist is None:
            return match_l, match_r

        it = ptr[:-1]
        for u in range(n_left):
            if match_l[u] == UNMATCHED:
                _augment(u, ptr, nbr, dist, it, match_l, match_r)


def _layers(
    ptr: list[int], nbr: list[int], match_l: list[int], match_r: list[int]
) -> list[int] | None:
    """BFS distance per left vertex from the free ones; None if no augmenting path."""
    dist = [0 if v == UNMATCHED else _INF for v in match_l]
    queue = [u for u, v in enumerate(match_l) if v == UNMATCHED]
    found = False
    for u in queue:
        for v in nbr[ptr[u] : ptr[u + 1]]:
            w = match_r[v]
            if w == UNMATCHED:
                found = True
                continue
            if dist[w] == _INF:
                dist[w] = dist[u] + 1
                queue.append(w)
    return dist if found else None


def _augment(
    root: int,
    ptr: list[int],
    nbr: list[int],
    dist: list[int],
    it: list[int],
    match_l: list[int],
    match_r: list[int],
) -> None:
    """Iterative layered DFS from *root*; flips the path when it ends free.

    stack[i] is matched to path[i] on success. A long chain (edge i held by
    seeds i, i+1) recurses n deep, so no recursion.
    """
    stack = [root]
    path: list[int] = []
    while stack:
        u = stack[-1]

        # Dead end: no layer-respecting edge left out of u.
        if it[u] == ptr[u + 1]:
            dist[u] = _INF
            stack.pop()
            if path:
                path.pop()
            continue

        v = nbr[it[u]]
        it[u] += 1
        w = match_r[v]
        if w == UNMATCHED:
            path.append(v)
            for uu, vv in zip(stack, path, strict=True):
                match_l[uu] = vv
                match_r[vv] = uu
            return

        if dist[w] == dist[u] + 1:
            stack.append(w)
            path.append(v)


def _witness(
    ptr: list[int], nbr: list[int], match_r: list[int], free: list[int]
) -> tuple[list[int], list[int]]:
    """Kőnig: left/right vertices reachable by alternating paths from *free*."""
    seen_l = set(free)
    seen_r: set[int] = set()
    queue = list(free)
    for u in queue:
        for v in nbr[ptr[u] : ptr[u + 1]]:
            if v in seen_r:
                continue
            seen_r.add(v)

            # v is matched: an unmatched v would be an augmenting path.
            w = match_r[v]
            if w not in seen_l:
                seen_l.add(w)
                queue.append(w)
    return sorted(seen_l), sorted(seen_r)


def reps_report(reps: Reps, added: int, rare_owners: int = RARE_OWNERS) -> str:
    """Report text for ``minimize -c --hall``."""
    lines = ["", f"--- {TITLE} (rare: <= {rare_owners} owners) ---"]
    if not reps.n_rare:
        lines.append("  No rare edges.")
        return "\n".join(lines)

    lines += [
        f"  {'Rare edges:':<16s}{reps.n_rare}",
        f"  {'Matched:':<16s}{len(reps.owner)} ({len(reps.owner) / reps.n_rare:.1%})",
        f"  {'Deficiency:':<16s}{len(reps.deficient)}",
        f"  {'Hall violator:':<16s}{len(reps.violator)} edges, {len(reps.witness)} seeds",
        f"  {'Seeds added:':<16s}{added}",
    ]
    return "\n".join(lines)
