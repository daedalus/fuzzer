"""Approximate kNN graph of the seeds (``minimize -c --seed-knn K``).

A seed is its set of edges; two seeds are close when their Jaccard is high.
The exact graph needs N x N pair scores. NN-descent (Dong et al., as in faiss
``NNDescent``) skips them: a neighbour of my neighbour is likely my neighbour,
so each round scores only those candidates and keeps the K best.

    sig[s, p] = min over e in s of h_p(e)            MinHash, P perms
    sim(s, t) = #{p : sig[s,p] == sig[t,p]}          ~ P * Jaccard(s, t)

    round:  cand(s) = nbr(s) + rev(s) + nbr(nbr(s) + rev(s))
            nbr(s)  = top K of cand(s) by sim

Edges held by more than *strip_frac* of the seeds are dropped first (as in
``edge_components``). Seeds left with no edge are not in the graph. At or
below ``BRUTE_MAX`` seeds the exact scan on the same signatures is used.
Memory: N x P x 4 bytes of signatures, scoring in blocks of ``PAIR_BLOCK``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from fuzzer_tool.core.edge_components import STRIP_FRAC, _incidence
from fuzzer_tool.core.edge_components import components as exact_components
from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool

TITLE = "Seed kNN"
PERMS = 64
MAX_ITERS = 12
DELTA = 0.001  # stop when fewer than this fraction of neighbour slots changed
BRUTE_MAX = 2048
REV_CAP = 5  # reverse neighbours kept per seed
ROW_BLOCK = 4096
PAIR_BLOCK = 1 << 16
INC_BLOCK = 1 << 22  # incidences hashed at once
BRUTE_CELLS = 1 << 26  # bytes of comparisons per exhaustive block
RECALL_SAMPLE = 100
NONE = -1  # no neighbour in this slot
UNKNOWN = -2  # similarity not scored yet
HASH_SHIFT = np.uint64(32)


class Mode(Enum):
    BRUTE = "exhaustive"
    DESCENT = "descent"


@dataclass
class KnnGraph:
    """Neighbour ids (-1 = none) and matching-permutation counts per seed."""

    all_keys: list[str]
    keys: list[str]  # seeds with at least one kept edge
    sig: np.ndarray  # (n, perms) uint32
    nbr: np.ndarray  # (n, k) int32, best first
    sim: np.ndarray  # (n, k) int16, -1 where nbr is -1
    mode: Mode
    iters: int
    rate: float  # fraction of slots changed in the last round

    @property
    def perms(self) -> int:
        return self.sig.shape[1]


def _check(k: int, perms: int, strip_frac: float, brute_max: int, max_iters: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if perms < 1:
        raise ValueError("perms must be >= 1")
    if not 0.0 < strip_frac <= 1.0:
        raise ValueError("strip_frac must be in (0, 1]")
    if brute_max < 0 or max_iters < 0:
        raise ValueError("brute_max and max_iters must be >= 0")


def _signatures(
    edges: dict[str, np.ndarray], strip_frac: float, perms: int, rng: RandPool
) -> tuple[list[str], np.ndarray]:
    """MinHash signature of every seed that keeps at least one edge."""
    keys, seed_idx, edge_ids = _incidence(edges)
    uniq, dense, held = np.unique(edge_ids, return_inverse=True, return_counts=True)
    keep = held[dense] <= strip_frac * len(keys)
    seed_idx, dense = seed_idx[keep], dense[keep]

    # seed_idx ascends, so each active seed is one contiguous run of `dense`.
    active, starts = np.unique(seed_idx, return_index=True)
    mult = np.frombuffer(rng.randbytes(8 * perms), dtype=np.uint64) | np.uint64(1)
    sig = np.empty((active.size, perms), dtype=np.uint32)

    lo = 0
    while lo < active.size:
        hi = max(lo + 1, int(np.searchsorted(starts, starts[lo] + INC_BLOCK)))
        end = int(starts[hi]) if hi < active.size else dense.size
        x = dense[starts[lo] : end].astype(np.uint64) + np.uint64(
            1
        )  # 0 would hash to 0 in every perm
        rel = starts[lo:hi] - starts[lo]
        for p in range(perms):
            h = ((x * mult[p]) >> HASH_SHIFT).astype(np.uint32)
            sig[lo:hi, p] = np.minimum.reduceat(h, rel)
        lo = hi
    return [keys[i] for i in active.tolist()], sig


def _pair_sims(sig: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Matching-permutation count of each (i, j) pair, in blocks."""
    out = np.empty(i.size, dtype=np.int16)
    for lo in range(0, i.size, PAIR_BLOCK):
        s = slice(lo, lo + PAIR_BLOCK)
        out[s] = np.count_nonzero(sig[i[s]] == sig[j[s]], axis=1)
    return out


def brute(sig: np.ndarray, k: int, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact K nearest per row by signature match; ties go to the lower id."""
    n, perms = sig.shape
    rows = np.arange(n) if rows is None else np.asarray(rows)
    nbr = np.full((rows.size, k), NONE, dtype=np.int32)
    sim = np.full((rows.size, k), NONE, dtype=np.int16)
    if k == 0:
        return nbr, sim

    step = max(1, BRUTE_CELLS // max(1, n * perms))
    rank = n - np.arange(n)
    for lo in range(0, rows.size, step):
        r = rows[lo : lo + step]
        eq = np.count_nonzero(sig[r][:, None, :] == sig[None, :, :], axis=2)
        key = eq.astype(np.int64) * (n + 1) + rank[None, :]
        key[np.arange(r.size), r] = NONE

        top = np.argpartition(-key, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(key, top, 1), axis=1), 1)
        nbr[lo : lo + step] = top
        sim[lo : lo + step] = np.take_along_axis(eq, top, 1)
    return nbr, sim


def _merge(
    sig: np.ndarray, rows: np.ndarray, ids: np.ndarray, known: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """K best distinct non-self candidates per row; scores only the unknown."""
    order = np.argsort(ids, axis=1, kind="stable")
    ids = np.take_along_axis(ids, order, 1)
    sim = np.take_along_axis(known, order, 1)

    bad = (ids < 0) | (ids == rows[:, None])
    bad[:, 1:] |= ids[:, 1:] == ids[:, :-1]
    r, c = np.nonzero(~bad & (sim == UNKNOWN))
    sim[r, c] = _pair_sims(sig, rows[r], ids[r, c])
    sim[bad] = NONE

    short = k - ids.shape[1]
    if short > 0:
        ids = np.pad(ids, ((0, 0), (0, short)), constant_values=NONE)
        sim = np.pad(sim, ((0, 0), (0, short)), constant_values=NONE)
    top = np.argsort(-sim.astype(np.int32), axis=1, kind="stable")[:, :k]
    nbr = np.take_along_axis(ids, top, 1)
    sim = np.take_along_axis(sim, top, 1)
    nbr[sim < 0] = NONE
    return nbr.astype(np.int32), sim


def _reverse(nbr: np.ndarray) -> np.ndarray:
    """Up to REV_CAP seeds that list each seed as a neighbour."""
    n, k = nbr.shape
    src = np.repeat(np.arange(n, dtype=np.int32), k)
    dst = nbr.ravel()
    ok = dst >= 0
    src, dst = src[ok], dst[ok]

    order = np.argsort(dst, kind="stable")
    src, dst = src[order], dst[order]
    rank = np.arange(dst.size) - np.searchsorted(dst, dst)
    fit = rank < REV_CAP

    rev = np.full((n, REV_CAP), NONE, dtype=np.int32)
    rev[dst[fit], rank[fit]] = src[fit]
    return rev


def _round(sig: np.ndarray, nbr: np.ndarray, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """One synchronous NN-descent round; returns the new graph and slots changed."""
    n, k = nbr.shape
    pool = np.concatenate([nbr, _reverse(nbr)], axis=1)
    new_nbr, new_sim = np.empty_like(nbr), np.empty_like(sim)
    changed = 0

    for lo in range(0, n, ROW_BLOCK):
        rows = np.arange(lo, min(lo + ROW_BLOCK, n))
        a = pool[rows]
        hop = np.where(a[..., None] >= 0, nbr[a], NONE).reshape(rows.size, -1)
        ids = np.concatenate([nbr[rows], a, hop], axis=1)
        unseen = np.full((rows.size, ids.shape[1] - k), UNKNOWN, dtype=np.int16)
        nb, sm = _merge(sig, rows, ids, np.concatenate([sim[rows], unseen], axis=1), k)

        fresh = (nb[:, :, None] != nbr[rows][:, None, :]).all(axis=2) & (nb >= 0)
        changed += int(np.count_nonzero(fresh))
        new_nbr[rows], new_sim[rows] = nb, sm
    return new_nbr, new_sim, changed


def _descend(
    sig: np.ndarray, k: int, max_iters: int, rng: RandPool
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """NN-descent from a random graph; returns (nbr, sim, rounds, last change rate)."""
    n = sig.shape[0]
    start = (rng.random_array((n, k)) * n).astype(np.int32)
    start = np.where(start == np.arange(n)[:, None], (start + 1) % n, start)
    unknown = np.full(start.shape, UNKNOWN, dtype=np.int16)
    nbr, sim = _merge(sig, np.arange(n), start, unknown, k)

    rate = 1.0
    for done in range(1, max_iters + 1):
        nbr, sim, changed = _round(sig, nbr, sim)
        rate = changed / (n * k)
        if rate < DELTA:
            return nbr, sim, done, rate
    return nbr, sim, max_iters, rate


def build(
    edges: dict[str, np.ndarray],
    k: int,
    strip_frac: float = STRIP_FRAC,
    perms: int = PERMS,
    brute_max: int = BRUTE_MAX,
    max_iters: int = MAX_ITERS,
    rng: RandPool | None = None,
) -> KnnGraph:
    """K nearest seeds (by estimated Jaccard) for every seed with a kept edge."""
    _check(k, perms, strip_frac, brute_max, max_iters)
    rng = rng or get_default_rand_pool()
    keys, sig = (
        _signatures(edges, strip_frac, perms, rng)
        if edges
        else ([], np.empty((0, perms), np.uint32))
    )
    n = len(keys)
    k_eff = max(0, min(k, n - 1))

    if n <= brute_max or k_eff == 0:
        nbr, sim = brute(sig, k_eff)
        return KnnGraph(list(edges), keys, sig, nbr, sim, Mode.BRUTE, 0, 0.0)

    nbr, sim, iters, rate = _descend(sig, k_eff, max_iters, rng)
    return KnnGraph(list(edges), keys, sig, nbr, sim, Mode.DESCENT, iters, rate)


def recall(g: KnnGraph, sample: int = RECALL_SAMPLE, rng: RandPool | None = None) -> float:
    """Mean share of each sampled seed's K slots that are as good as the true K-th.

    Compared by score, not id, so equal-score ties cannot count as misses.
    """
    n, k = g.nbr.shape
    if k == 0:
        return 1.0
    rng = rng or get_default_rand_pool()
    rows = np.arange(n) if sample >= n else np.asarray(sorted(rng.sample(range(n), sample)))

    _, true_sim = brute(g.sig, k, rows)
    kth = true_sim[:, -1:]
    return float((g.sim[rows] >= kth).sum(axis=1).mean() / k)


def _labels(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Connected-component label per node: min-label propagation with pointer jumping."""
    lab = np.arange(n, dtype=np.int64)
    while True:
        new = lab.copy()
        np.minimum.at(new, src, lab[dst])
        np.minimum.at(new, dst, lab[src])
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def knn_components(g: KnnGraph) -> dict[str, int]:
    """Component label per seed of the symmetrised graph (edges with score > 0)."""
    n, k = g.nbr.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = g.nbr.ravel().astype(np.int64)
    ok = (dst >= 0) & (g.sim.ravel() > 0)
    lab = _labels(n, src[ok], dst[ok])

    out = dict(zip(g.keys, lab.tolist(), strict=True))
    own = n
    for key in g.all_keys:
        if key in out:
            continue
        out[key] = own
        own += 1
    return out


def _split_stats(exact: np.ndarray, knn: np.ndarray) -> tuple[int, int, int]:
    """(exact components, how many the graph splits, seeds off their main piece)."""
    width = int(knn.max()) + 1
    uniq, size = np.unique(exact.astype(np.int64) * width + knn, return_counts=True)
    comp = uniq // width
    ids, first = np.unique(comp, return_index=True)
    pieces = np.diff(np.append(first, comp.size))
    main = np.maximum.reduceat(size, first)
    return ids.size, int((pieces > 1).sum()), int(exact.size - main.sum())


def seed_knn_report(
    edges: dict[str, np.ndarray],
    k: int,
    strip_frac: float = STRIP_FRAC,
    rng: RandPool | None = None,
) -> str:
    """Report text for ``minimize -c --seed-knn K``."""
    lines = ["", f"--- {TITLE} (k={k}, perms={PERMS}, strip edges held by > {strip_frac:.0%}) ---"]
    start = time.perf_counter()
    g = build(edges, k, strip_frac, rng=rng)
    took = time.perf_counter() - start
    if not g.keys:
        lines.append("  No seeds with edge data.")
        return "\n".join(lines)

    knn = knn_components(g)
    exact, _ = exact_components(edges, strip_frac)
    knn_lab = np.fromiter((knn[x] for x in g.all_keys), dtype=np.int64, count=len(g.all_keys))
    ex_lab = np.fromiter((exact[x] for x in g.all_keys), dtype=np.int64, count=len(g.all_keys))
    n_exact, split, off = _split_stats(ex_lab, knn_lab)

    sizes = np.unique(knn_lab, return_counts=True)[1]
    live = g.sim[g.nbr >= 0]
    how = (
        "exhaustive scan"
        if g.mode is Mode.BRUTE
        else f"descent, {g.iters} rounds, {g.rate:.2%} changed in the last"
    )
    lines += [
        f"  {'Seeds:':<18s}{len(g.all_keys)} ({len(g.all_keys) - len(g.keys)} without kept edges)",
        f"  {'Mode:':<18s}{how}",
        f"  {'Recall:':<18s}{recall(g, rng=rng):.1%} of true neighbours (sample {RECALL_SAMPLE})",
        f"  {'Mean similarity:':<18s}{live.mean() / g.perms if live.size else 0.0:.3f} (est. Jaccard)",
        f"  {'Components:':<18s}{sizes.size}, largest {sizes.max()} seeds",
        f"  {'Exact components:':<18s}{n_exact}; split by the graph: {split}; "
        f"seeds off their main piece: {off} ({off / len(g.all_keys):.1%})",
        f"  {'Build:':<18s}{took:.2f} s",
    ]
    return "\n".join(lines)
