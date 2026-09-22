#!/usr/bin/env python3
"""Analyse the 2-D (edge_id, hit_count) matrix a target's SHM table produces.

Motivation
----------
It is tempting to treat the SHM table as a plot: edge_id on x, hit count on
y, then reach for the usual 2-D toolbox (regression, autocorrelation, FFT,
Wasserstein over the index, clustering, fractal dimension).  Almost none of
that is defined here, because x is not a coordinate.  The shim builds

    edge_id = (caller_ctx ^ prev_loc ^ cur_loc) | 1

from small sequential guard values (``__sanitizer_cov_trace_pc_guard_init``),
where ``caller_ctx`` is a ``__AFL_CTX_BITS``-wide slice of a splitmix64 hash
of a return address.  XOR is an ultrametric, so "adjacent ids" means "share a
high-bit prefix", never "adjacent in the program".  The one earlier attempt
to use the id axis as a metric space -- Wasserstein/CRPS over the edge index
and ``compute_coverage_proximity`` -- measured nothing and was moved onto the
``log2(1 + hit count)`` axis for exactly this reason.

What *is* defined splits three ways, and this tool reports all three so the
distinction stops having to be re-derived:

1. **Blocked x.**  With ``__AFL_CTX_SENSITIVE=1`` (the default) the context
   term only perturbs the low ``__AFL_CTX_BITS``, so ``id >> ctx_bits`` is a
   context-free edge family and ``id & mask`` is the caller tag.  The
   defined statistic is a nested variance decomposition (an ICC), not a
   curve fit.  Two controls are reported alongside it per Hard Rule 46: a
   global permutation null, and a within-family shuffle that must *preserve*
   the effect if the effect is really the blocking.
2. **Permutation-invariant y.**  Anything that ignores x: the AFL count-class
   ladder (reused from ``core.count_class``, not reimplemented), Shannon
   entropy as an effective-edge count, Gini, and a Zipf fit with its
   residual so a bad fit cannot be quoted as a good one.
3. **Substituted x.**  Four axes that *do* carry meaning and already exist in
   the tree: owner count, first-seen index, per-edge max count, and (not
   collected here) the distance-table ``node_idx``.  Reported as rank
   correlations against total count.

Three further sections take the *seed x edge* matrix rather than the (id,
count) columns, which are not a matrix at all: its singular spectrum, its
GF(2) rank against the redundancy that rank is a proxy for, and -- opt-in --
its integer relations. All three are sensitive detectors of the id drift
described next, far more sensitive than an edge count is.

``--transpose`` runs those on the edge x seed matrix instead and adds section
[7], the edge equivalence classes. The spectrum and both ranks are
transpose-invariant and are printed as controls; everything derived from them
is not, and the derived numbers are where the orientation earns its keep.

``--positions`` adds section [8], the (edge_pos, edge_id, count) matrix:
edge_pos is the SHM slot index the live edge occupies, home = edge_id %
map_size plus a linear-probe displacement.  It falsifies three shim-design
predictions directly -- stable placement (a slot is never reclaimed), count
independent of position, and the per-run (pos, id) view being a partial
permutation whose spectrum is exactly the count histogram (F12, quantified).

It also measures cross-process id stability, which is what makes or breaks
every number above: the context hash is taken over a return address, so
under PIE + ASLR the same call chain hashes differently in every process
unless the shim resolves that address relative to the load base first.
``services.fuzzer`` calls ``adapters.process.disable_aslr`` once at startup
and children inherit it; this tool does the same.

``--keep-aslr`` skips that call and nothing else, which is the *raw* regime:
ASLR on with the shim hashing addresses verbatim. That is no longer what a
``FUZZER_KEEP_ASLR=1`` run sees -- the variable now also puts the shim in
base-relative mode, and ``services.fuzzer`` sets it for the target whenever
ASLR survives startup. To reproduce a production run with ASLR on, export
``FUZZER_KEEP_ASLR=1`` alongside the flag; the flag by itself reproduces a
target built before the shim grew that mode, which is still the sharpest
canary the tool has for the sections below.

Usage::

    # collect and analyse
    python3 tools/edge_matrix_analysis.py --target targets/fuzzgoat_read \\
        --corpus ~/fuzzing/corpus/json --save /tmp/edges.npz

    # re-analyse a saved collection (no target needed)
    python3 tools/edge_matrix_analysis.py --load /tmp/edges.npz --json out.json

    # quantify what context sensitivity buys, and what ASLR costs it
    python3 tools/edge_matrix_analysis.py --target targets/fuzzgoat_read \\
        --corpus ~/fuzzing/corpus/json --keep-aslr
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzzer_tool.adapters.process import disable_aslr
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.count_class import classify_single

COUNT_MASK = 0xFFFFFF  # the top byte of the SHM count field is the generation tag


# ── Collection ────────────────────────────────────────────────────────


def _run_one(target: Path, cov: ShmCoverage, path: Path, timeout: float):
    """Execute *target* on *path* and return its live (positions, ids, counts)."""
    cov.reset_edge_map()
    env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(cov.num_entries))
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run([str(target), str(path)], env=env, capture_output=True, timeout=timeout)
    positions, ids, counts = cov._scan_with_positions()
    return (
        positions.astype(np.int64),
        ids.astype(np.int64),
        (counts & COUNT_MASK).astype(np.int64),
    )


def collect(target: Path, inputs: list[Path], map_size: int, timeout: float):
    """Run every input once, returning one (positions, ids, counts) triple per execution.

    The first execution against a *clean* table does not produce the same
    ids as every later execution against a generation-reset one -- measured,
    see the warm-up note in ``measure_stability``.  One discarded warm-up run
    puts the whole collection in the same regime the fuzzer spends its
    campaign in, instead of leaving input 0 in a regime of its own.
    """
    cov = ShmCoverage(size=map_size)
    try:
        _run_one(target, cov, inputs[0], timeout)  # warm-up, discarded
        return [_run_one(target, cov, p, timeout) for p in inputs]
    finally:
        cov.cleanup()


def measure_stability(target: Path, path: Path, repeats: int, map_size: int, timeout: float):
    """Jaccard of the edge-id sets across *repeats* executions of one input.

    1.0 means edge identity is reproducible across processes.  Anything less
    with a deterministic target means the ids themselves are moving, which
    invalidates every per-edge statistic downstream -- the headline case
    being a PIE target under ASLR with ``__AFL_CTX_SENSITIVE=1``, where the
    context term is a hash of a raw return address and therefore differs in
    every process.

    Two regimes are reported separately because they have different causes.
    ``jaccard`` covers the steady state, after one warm-up execution has
    populated the table; ``first_exec_matches`` says whether the very first
    execution against a *clean* table agreed with it.  Measured on fuzzgoat
    with ASLR off: steady state is exact, and the first execution differs --
    same edge count and same total fires, different rolling path hash, and
    under ``__AFL_CTX_SENSITIVE=1`` a handful of different ids.
    """
    cov = ShmCoverage(size=map_size)
    try:
        first = frozenset(_run_one(target, cov, path, timeout)[1].tolist())
        sets = [frozenset(_run_one(target, cov, path, timeout)[1].tolist()) for _ in range(repeats)]
    finally:
        cov.cleanup()
    union = set().union(*sets)
    inter = set(sets[0]).intersection(*sets[1:]) if len(sets) > 1 else set(sets[0])
    return {
        "repeats": repeats,
        "sizes": [len(s) for s in sets],
        "union": len(union),
        "intersection": len(inter),
        "jaccard": (len(inter) / len(union)) if union else 1.0,
        "first_exec_matches": first == sets[0],
        "first_exec_only": len(first - union),
    }


def save_runs(path: Path, runs, map_size: int | None = None) -> None:
    """Store the ragged per-execution columns as three flat arrays."""
    offsets = np.cumsum([0] + [len(p) for p, _, _ in runs])
    np.savez_compressed(
        path,
        positions=np.concatenate([p for p, _, _ in runs]) if runs else np.empty(0, np.int64),
        ids=np.concatenate([i for _, i, _ in runs]) if runs else np.empty(0, np.int64),
        counts=np.concatenate([c for _, _, c in runs]) if runs else np.empty(0, np.int64),
        offsets=offsets,
        map_size=map_size or 0,
    )


def saved_map_size(path: Path):
    """Table size a collection was recorded with, if the file carries it."""
    with np.load(path) as z:
        size = int(z["map_size"]) if "map_size" in z else 0
    return size if size else None


def load_runs(path: Path):
    z = np.load(path)
    off = z["offsets"]
    runs = [
        (z["ids"][off[i] : off[i + 1]], z["counts"][off[i] : off[i + 1]])
        for i in range(len(off) - 1)
    ]
    if "positions" in z:
        return [
            (z["positions"][off[i] : off[i + 1]], ids, counts)
            for i, (ids, counts) in enumerate(runs)
        ]
    return runs


# ── Aggregation ───────────────────────────────────────────────────────


def aggregate(runs):
    """Collapse per-execution columns into the per-edge vectors under study.

    ``total`` mirrors ``EdgeTracker._global_edge_hits`` (execution volume),
    ``owners`` mirrors ``_edge_owner_count`` (incidence), ``first_seen``
    mirrors ``_edge_first_seen`` and ``peak`` mirrors ``_max_counts``.  The
    point of returning all four is that the first is the one most easily
    mistaken for the others.
    """
    total: Counter[int] = Counter()
    owners: Counter[int] = Counter()
    peak: dict[int, int] = {}
    first: dict[int, int] = {}
    for idx, (ids, counts) in enumerate(runs):
        for edge, count in zip(ids.tolist(), counts.tolist(), strict=True):
            total[edge] += count
            owners[edge] += 1
            if count > peak.get(edge, 0):
                peak[edge] = count
            first.setdefault(edge, idx)
    ids = np.array(sorted(total), dtype=np.int64)
    return {
        "ids": ids,
        "total": np.array([total[e] for e in ids], dtype=float),
        "owners": np.array([owners[e] for e in ids], dtype=float),
        "peak": np.array([peak[e] for e in ids], dtype=float),
        "first_seen": np.array([first[e] for e in ids], dtype=float),
    }


# ── Statistics (numpy only; scipy is not a dependency) ────────────────


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    return float((xc * yc).sum() / denom) if denom else 0.0


def _ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks, so ties (very common in hit counts) do not bias rho."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=float)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_ranks(x), _ranks(y))


def _lag1(v: np.ndarray) -> float:
    vc = v - v.mean()
    denom = float((vc * vc).sum())
    return float((vc[:-1] * vc[1:]).sum() / denom) if denom else 0.0


# ── Section 1: the x axis ─────────────────────────────────────────────


def axis_structure(ids: np.ndarray, total: np.ndarray, ctx_bits: int, perms: int, seed: int):
    """Test whether the id axis carries anything, and if so, at what scale."""
    rng = np.random.default_rng(seed)
    families = defaultdict(list)
    for edge, count in zip(ids.tolist(), total.tolist(), strict=True):
        families[edge >> ctx_bits].append(count)

    # Nested variance decomposition on log2(1+count): how much of the
    # spread is *between* context-free families rather than within one.
    y = np.log2(1.0 + total)
    grand = y.mean()
    between = 0.0
    for members in families.values():
        fam = np.log2(1.0 + np.asarray(members, dtype=float))
        between += len(fam) * (fam.mean() - grand) ** 2
    ss_total = float(((y - grand) ** 2).sum())

    observed = _lag1(total)
    null = np.array([_lag1(rng.permutation(total)) for _ in range(perms)])
    # Control (Hard Rule 46): shuffling only *within* families must leave the
    # effect standing if the effect is the blocking.  If it collapses, the
    # signal is a gradient along x, which would be a genuine surprise worth
    # chasing rather than the context layout.
    fam_index: dict[int, list[int]] = defaultdict(list)
    for i, edge in enumerate(ids.tolist()):
        fam_index[edge >> ctx_bits].append(i)
    within = total.copy()
    for idxs in fam_index.values():
        within[idxs] = rng.permutation(total[idxs])

    odd = int((ids & 1).sum())
    sizes = sorted((len(v) for v in families.values()), reverse=True)
    return {
        "n_edges": int(len(ids)),
        "id_min": int(ids.min()) if len(ids) else 0,
        "id_max": int(ids.max()) if len(ids) else 0,
        "ctx_bits": ctx_bits,
        "families": len(families),
        "largest_families": sizes[:8],
        "ctx_tags_reachable": (1 << ctx_bits) // 2 if ctx_bits else 1,
        "all_ids_odd": odd == len(ids),
        "icc_family": (between / ss_total) if ss_total else 0.0,
        "lag1_observed": observed,
        "lag1_null_mean": float(null.mean()),
        "lag1_null_sd": float(null.std()),
        "lag1_z": float((observed - null.mean()) / null.std()) if null.std() else 0.0,
        "lag1_within_family_shuffle": _lag1(within),
        "spearman_id_vs_count": _spearman(ids.astype(float), total),
    }


# ── Section 2: the y marginal ─────────────────────────────────────────


def y_marginal(total: np.ndarray):
    """Everything about the matrix that survives permuting the id axis."""
    p = total / total.sum()
    entropy = float(-(p * np.log2(p)).sum())
    desc = np.sort(total)[::-1]
    asc = desc[::-1]
    n = len(total)
    gini = float(1.0 - 2.0 * np.sum(np.cumsum(asc)) / (n * asc.sum()) + 1.0 / n)
    rank = np.arange(1, n + 1, dtype=float)
    slope, intercept = np.polyfit(np.log(rank), np.log(desc), 1)
    resid = np.log(desc) - (slope * np.log(rank) + intercept)
    classes = Counter(classify_single(int(c)) for c in total)
    return {
        "mean": float(total.mean()),
        "median": float(np.median(total)),
        "max": float(total.max()),
        "entropy_bits": entropy,
        "entropy_max_bits": math.log2(n) if n else 0.0,
        "effective_edges": 2.0**entropy,
        "gini": gini,
        "top_1pct_share": float(desc[: max(1, n // 100)].sum() / desc.sum()),
        "zipf_slope": float(slope),
        "zipf_resid_sd": float(resid.std()),
        "count_classes": {str(k): v for k, v in sorted(classes.items())},
    }


# ── Section 3: axes that mean something ───────────────────────────────


def substituted_axes(agg):
    """Rank correlation of each meaningful x candidate against total count."""
    total = agg["total"]
    return {
        "spearman_owners_vs_total": _spearman(agg["owners"], total),
        "spearman_first_seen_vs_total": _spearman(agg["first_seen"], total),
        "spearman_peak_vs_total": _spearman(agg["peak"], total),
    }


# ── Sections 4 and 5: the seed x edge matrix ──────────────────────────
#
# The (edge_id, count) columns are not a matrix -- two incommensurable
# columns. The matrix the counts live in is seed x edge, which is what
# core/schedulers/seed_tang.py factorises. Both sections below exist mainly
# as F1 canaries: a spectrum and a GF(2) rank are far more sensitive to
# per-process id drift than an edge count is, because every phantom id is a
# column owned by exactly one seed.

SVD_CELL_BUDGET = 20_000_000  # m*n above which the dense SVD is skipped
GREEDY_CELL_BUDGET = 50_000_000


def seed_edge_matrix(runs):
    """Dense seed x edge hit-count matrix, rows in execution order."""
    ids = sorted({int(e) for eids, _ in runs for e in eids.tolist()})
    col = {e: j for j, e in enumerate(ids)}
    mat = np.zeros((len(runs), len(ids)), dtype=float)
    for r, (eids, counts) in enumerate(runs):
        for edge, count in zip(eids.tolist(), counts.tolist(), strict=True):
            mat[r, col[int(edge)]] = count
    return mat


def spectrum(mat, ks=(1, 3, 10, 20)):
    """Singular spectrum under all three cell semantics.

    Reporting the three together is deliberate: a harness bug upstream of the
    transform once produced *identical* spectra for hit counts, log1p and
    binary (`handover_done_2026-09-06.md` §14, third round), and identical
    rows here are the signature of that class of bug rather than a result.
    """
    if mat.size > SVD_CELL_BUDGET:
        return {"skipped": f"{mat.shape[0]}x{mat.shape[1]} exceeds the dense-SVD budget"}
    out = {}
    for label, cells in (
        ("raw", mat),
        ("log1p", np.log1p(mat)),
        ("binary", (mat > 0).astype(float)),
    ):
        sv = np.linalg.svd(cells, compute_uv=False)
        sq = sv**2
        mass = float(sq.sum())
        out[label] = {
            "dominance": float(sq[0] / mass) if mass else 0.0,
            # participation ratio of the squared spectrum: 1.0 means rank-1,
            # min(m, n) means flat.
            "effective_rank": float(mass**2 / float((sq**2).sum())) if mass else 0.0,
            "rho": {str(k): float(np.sqrt(sq[k:].sum() / mass)) if mass else 0.0 for k in ks},
        }
    return out


def _gf2_reduce(binary):
    """Row-reduce over GF(2) using one big int per row.

    Returns (rank, pivot columns, indices of rows that reduced to zero). A
    row reduces to zero exactly when its coverage is the symmetric difference
    of earlier rows -- which, since XOR can only set bits present in an
    operand, implies its edges are contained in their union. That makes this
    a *sound* redundancy detector; it is not a complete one, because a seed
    whose edges are merely a subset of another's produces no cancellation and
    stays independent.
    """
    pivots: dict[int, int] = {}
    zero_rows: list[int] = []
    for idx, row in enumerate(binary):
        cur = int.from_bytes(np.packbits(row[::-1].astype(np.uint8)).tobytes(), "big")
        cur >>= (-len(row)) % 8  # packbits pads the high end; drop the padding
        while cur:
            bit = cur.bit_length() - 1
            if bit in pivots:
                cur ^= pivots[bit]
            else:
                pivots[bit] = cur
                break
        if cur == 0:
            zero_rows.append(idx)
    return len(pivots), sorted(pivots), zero_rows


def gf2_structure(mat):
    """GF(2) rank of the binary matrix, against the redundancy it is a proxy for."""
    binary = mat > 0
    rank, pivots, zero_rows = _gf2_reduce(binary)
    owners = binary.sum(axis=0)
    # A seed's edges are contained in the union of the others exactly when it
    # owns no singleton edge -- O(m*n) and exact, no pairwise comparison.
    has_singleton = (binary & (owners == 1)).any(axis=1)
    union_redundant = int((~has_singleton).sum())

    greedy = None
    if binary.size <= GREEDY_CELL_BUDGET:
        need = np.ones(binary.shape[1], dtype=bool)
        greedy = 0
        while need.any():
            gains = binary[:, need].sum(axis=1)
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            need &= ~binary[best]
            greedy += 1

    pivot_cols = list(pivots)
    separated = len({binary[i][pivot_cols].tobytes() for i in range(binary.shape[0])})
    return {
        "rows": int(binary.shape[0]),
        "gf2_rank": rank,
        "real_rank": int(np.linalg.matrix_rank(binary.astype(float))),
        "distinct_rows": len({r.tobytes() for r in binary}),
        "xor_dependent_rows": len(zero_rows),
        "union_redundant_seeds": union_redundant,
        "pivot_edges": len(pivot_cols),
        "distinct_rows_on_pivots": separated,
        "greedy_cover_seeds": greedy,
    }


# ── Section 6: integer relations (opt-in) ─────────────────────────────

LLL_ROW_BUDGET = 100  # LLL here is ~2 minutes at 100 rows and superlinear


def _lll_reduce(basis, delta=0.75):
    """LLL-reduce *basis* (a list of integer row vectors) in place, returning it.

    Written out rather than imported: Hard Rule 51, and the only outside
    implementation that would fit here is sympy's, which is a dependency this
    repo does not carry.

    The basis vectors stay exact Python integers -- every subtraction and
    swap below is integer arithmetic -- while the Gram-Schmidt coefficients
    are float.  That split is the usual engineering compromise and it is safe
    for what this tool reports: float error in ``mu`` can only make the
    reduction *weaker* (a size reduction skipped, a swap not taken), never
    turn a non-relation into one, because the caller decides what is a
    relation by testing exact integer entries for zero.

    The Gram-Schmidt row for ``k`` is recomputed from the orthogonalised rows
    below it whenever ``B[k]`` changes, and both affected rows are refreshed
    after a swap.  Rows above ``k`` are never read before ``k`` reaches them,
    so nothing stale is ever used.
    """
    rows = [list(map(int, r)) for r in basis]
    n = len(rows)
    if n < 2:
        return rows
    dim = len(rows[0])
    ortho = np.zeros((n, dim))
    mu = np.zeros((n, n))
    norms = np.zeros(n)

    def orthogonalise(k):
        v = np.array(rows[k], dtype=float)
        for j in range(k):
            if norms[j] > 0.0:
                mu[k, j] = float(np.dot(v, ortho[j]) / norms[j])
                v = v - mu[k, j] * ortho[j]
            else:
                mu[k, j] = 0.0
        ortho[k] = v
        norms[k] = float(np.dot(v, v))

    orthogonalise(0)
    k = 1
    while k < n:
        orthogonalise(k)
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k, j]))
            if q:
                rows[k] = [a - q * b for a, b in zip(rows[k], rows[j], strict=True)]
                orthogonalise(k)
        if norms[k] >= (delta - mu[k, k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            rows[k], rows[k - 1] = rows[k - 1], rows[k]
            orthogonalise(k - 1)
            orthogonalise(k)
            k = max(k - 1, 1)
    return rows


def integer_relations(mat, max_rows=LLL_ROW_BUDGET):
    """Integer structure of the raw-count matrix: duplicates, sparse relations, LLL.

    Off by default (``--lll``) because it is the most expensive analysis here
    by three orders of magnitude and, measured on fuzzgoat, the least
    informative: the exact relations LLL recovers are dense (support ~n/2,
    coefficients into the tens) and the only short vectors in the lattice are
    duplicate-row differences, which the hash below finds in O(m).

    The exhaustive scalar-multiple and A = B + C searches are what make a
    negative meaningful: they prove absence of sparse relations at support
    <= 3 outright, where LLL proves nothing -- its approximation factor is
    2^((n-1)/2), vacuous at these dimensions.
    """
    rows = [r for r in mat.astype(int)]
    first: dict[bytes, int] = {}
    distinct_idx = []
    for i, row in enumerate(rows):
        key = row.tobytes()
        if key in first:
            continue
        first[key] = i
        distinct_idx.append(i)
    dist = mat.astype(int)[distinct_idx]

    index = {dist[i].tobytes(): i for i in range(len(dist))}
    triples = 0
    multiples = 0
    for i in range(len(dist)):
        for j in range(i + 1, len(dist)):
            if (dist[i] + dist[j]).tobytes() in index:
                triples += 1
            a, b = dist[i], dist[j]
            if np.array_equal(a > 0, b > 0) and (a > 0).any():
                ratio = a[a > 0] / b[a > 0]
                if np.allclose(ratio, ratio[0]) and abs(ratio[0] - 1.0) > 1e-9:
                    multiples += 1

    out = {
        "rows": int(mat.shape[0]),
        "distinct_rows": len(dist),
        "duplicate_rows": int(mat.shape[0]) - len(dist),
        "scalar_multiple_pairs": multiples,
        "sum_triples": triples,
    }

    sub = dist[:max_rows]
    sub = sub[:, sub.any(axis=0)]
    n = sub.shape[0]
    rank = int(np.linalg.matrix_rank(sub.astype(float)))
    # [I | N*A]: a reduced row whose A-part vanishes carries an exact integer
    # relation in its I-part. N only has to outweigh the coefficients we care
    # about, so that a row keeping any coverage mass cannot look short.
    scale = 10**4
    basis = [
        [1 if j == i else 0 for j in range(n)] + [scale * int(v) for v in sub[i]] for i in range(n)
    ]
    start = time.perf_counter()
    reduced = _lll_reduce(basis)
    elapsed = time.perf_counter() - start
    rels = [row[:n] for row in reduced if not any(row[n:])]
    support = [sum(1 for v in c if v) for c in rels]
    l1 = [sum(abs(v) for v in c) for c in rels]
    peak = [max(abs(v) for v in c) for c in rels]
    out["lll"] = {
        "rows": n,
        "cols": int(sub.shape[1]),
        "rank": rank,
        "kernel_dim": n - rank,
        "relations_found": len(rels),
        "support_median": float(np.median(support)) if rels else None,
        "l1_median": float(np.median(l1)) if rels else None,
        "max_coeff_median": float(np.median(peak)) if rels else None,
        "seconds": elapsed,
    }
    return out


# ── Section 7: edge equivalence classes (transposed orientation) ──────


def duplicate_classes(mat, row_ids, ctx_bits):
    """Group edges whose count profile is identical across every execution.

    Only meaningful on the transposed matrix, where a row is an edge. Two
    edges in one class carry the same information in this corpus: whatever
    distinguishes them, no input here exercised it.

    The split by ``id >> ctx_bits`` is the actionable part. A class confined
    to one family is a base edge whose context tags never once differed --
    map slots context sensitivity bought and did not use, countable exactly
    rather than inferred from the ICC in section [1]. A class spanning
    several families is a straight-line block chain, a property of the target
    rather than of the instrumentation.
    """
    groups: dict[bytes, list[int]] = {}
    for i, row in enumerate(mat.astype(int)):
        groups.setdefault(row.tobytes(), []).append(i)
    multi = [g for g in groups.values() if len(g) > 1]
    within = [g for g in multi if len({int(row_ids[i]) >> ctx_bits for i in g}) == 1]
    across = [g for g in multi if len({int(row_ids[i]) >> ctx_bits for i in g}) > 1]
    largest = max(multi, key=len) if multi else []
    return {
        "rows": int(mat.shape[0]),
        "classes": len(groups),
        "duplicate_rows": sum(len(g) - 1 for g in multi),
        "within_family_classes": len(within),
        "within_family_rows": sum(len(g) - 1 for g in within),
        "across_family_classes": len(across),
        "across_family_rows": sum(len(g) - 1 for g in across),
        "largest_class": len(largest),
        "largest_class_families": sorted({int(row_ids[i]) >> ctx_bits for i in largest})[:4],
    }


# ── Section 8: the (edge_pos, edge_id, count) matrix (opt-in) ─────────
#
# x = edge_pos is the SHM slot index, derived by the shim's linear probe
# from home = edge_id % map_size; y = edge_id; z = hit count.  The 2-D
# projection (pos, id) with the count as its value and the 3-D binary
# tensor (pos, id, count) are the two matrices under study.
#
# Three falsifiable predictions fall straight out of the shim design:
#
# *  a slot is claimed once and never reclaimed, so an id's position must
#    be fixed for the whole run once its edge first fires;
# *  an id's count is a property of the *input*, its position a property of
#    the *insertion* -- so count and position should be independent;
# *  per run the (pos, id) matrix is a partial permutation (one live id per
#    slot, one slot per id), so its singular spectrum is exactly the
#    per-edge count histogram: a fold is a bijection onto its image and
#    cannot add information (handover F12, now made quantitative).
#
# Every correlation is tested against a null that permutes the companion
# margin among the triples (Hard Rule 46), so the marginal distributions
# are fixed under the null.

SVD_CELL_BUDGET_POS = 20_000_000  # trimmed per-run matrix cells


def _placement_rollup(pos_runs):
    """Flatten every run into (run_idx, pos, id, count) plus slot/id membership."""
    pos_of: dict[int, set[int]] = defaultdict(set)
    ids_at_slot: dict[int, set[int]] = defaultdict(set)
    triples = []
    for run_idx, (positions, ids, counts) in enumerate(pos_runs):
        for pos, eid, cnt in zip(positions.tolist(), ids.tolist(), counts.tolist(), strict=True):
            pos_of[eid].add(pos)
            ids_at_slot[pos].add(eid)
            triples.append((run_idx, pos, eid, cnt))
    return pos_of, ids_at_slot, triples


def _placement_overview(arr, pos_of, ids_at_slot, map_size):
    """Placement identity stats; the F14 machinery, end to end."""
    arr_pos, arr_id, arr_cnt = arr[:, 1], arr[:, 2], arr[:, 3]
    with np.errstate(invalid="ignore"):
        home = arr_id % map_size
        disp = (arr_pos - home) % map_size
    n_ids = len(pos_of)
    ids_multi_position = sum(1 for s in pos_of.values() if len(s) > 1)
    slots_multi_id = sum(1 for ids_at_pos in ids_at_slot.values() if len(ids_at_pos) > 1)
    return (
        {
            "unique_ids": n_ids,
            "ids_single_position": n_ids - ids_multi_position,
            "ids_single_position_pct": 100.0 * (n_ids - ids_multi_position) / n_ids,
            "ids_multi_position": ids_multi_position,
            "slots_multi_id": slots_multi_id,
            "displacement_mean": float(disp.mean()),
            "home_hit_frac": float((disp == 0).mean()),
            "beyond_probe_max": int((disp >= ShmCoverage.PROBE_MAX).sum()),
            "disp_hist_top": [int(c) for _k, c in Counter(disp.tolist()).most_common(8)],
        },
        home,
        disp,
        arr_cnt,
    )


def _spearman_null_block(a, b, perms, rng, prefix):
    """Observed rank correlation against a b-preserving permutation null."""
    obs = _spearman(a, b)
    null = np.array([_spearman(a, rng.permutation(b)) for _ in range(perms)])
    return {
        f"{prefix}": obs,
        f"{prefix}_null_mean": float(null.mean()),
        f"{prefix}_null_sd": float(null.std()),
        f"{prefix}_null_z": float((obs - null.mean()) / null.std()) if null.std() else 0.0,
    }


def _spectral_check(pos_runs, budget):
    """F14 quantitative check: first-run (pos, id) spectrum == count histogram."""
    first = pos_runs[0]
    cells = len(first[0]) * len(first[1])
    if not 0 < cells <= budget:
        return {}
    rowset = {int(p) for p in first[0].tolist()}
    colset = {int(i) for i in first[1].tolist()}
    rm = {r: k for k, r in enumerate(sorted(rowset))}
    cm = {c: k for k, c in enumerate(sorted(colset))}
    mat = np.zeros((len(rm), len(cm)), dtype=float)
    for p, i, c in zip(first[0].tolist(), first[1].tolist(), first[2].tolist(), strict=True):
        mat[rm[p], cm[i]] = c
    sv = np.linalg.svd(mat, compute_uv=False)
    ref = np.sort(first[2].astype(float))[::-1]
    return {
        "rank_2d": int(np.linalg.matrix_rank(mat)),
        "single_run_sv_max_relerr": float(
            np.max(np.abs(sv - ref) / ref.max()) if ref.size else 0.0
        ),
    }


def positions_structure(pos_runs, map_size: int, perms: int, seed: int):
    """Placement statistics over the (pos, id, count) triples of every run."""
    if pos_runs is None:
        return {"available": False}
    rng = np.random.default_rng(seed)

    pos_of, ids_at_slot, triples = _placement_rollup(pos_runs)
    if not triples:
        return {"available": True, "runs": len(pos_runs), "triples": 0, "unique_ids": 0}

    arr = np.array(triples, dtype=np.int64)
    first_run = {}
    for run_idx, (_, ids, _) in enumerate(pos_runs):
        for eid in ids.tolist():
            first_run.setdefault(eid, run_idx)
    arr_first = np.array([first_run[e] for e in arr[:, 2].tolist()], dtype=np.int64)

    overview, home, disp, arr_cnt = _placement_overview(arr, pos_of, ids_at_slot, map_size)
    out = {
        "available": True,
        "runs": len(pos_runs),
        "triples": int(len(triples)),
        "table_size": map_size,
        **overview,
        **_spearman_null_block(
            disp.astype(float), arr_cnt.astype(float), perms, rng, "spearman_disp_count"
        ),
        **_spearman_null_block(
            home.astype(float), disp.astype(float), perms, rng, "spearman_home_disp"
        ),
    }

    # Confound check: displacement is decided by table occupancy at the
    # moment of insertion, so any disp-count association should trace to
    # first-seen run, not to the count itself.
    out["spearman_disp_first_run"] = _spearman(disp.astype(float), arr_first.astype(float))
    out["spearman_first_run_count"] = _spearman(arr_first.astype(float), arr_cnt.astype(float))
    out.update(_spectral_check(pos_runs, SVD_CELL_BUDGET_POS))
    return out


# ── Reporting ─────────────────────────────────────────────────────────


def _report(result) -> None:
    coll = result["collection"]
    print(
        f"inputs {coll['inputs']}  median live edges {coll['median_live_edges']}  "
        f"union {coll['union_edges']}  total hits {coll['total_hits']}"
    )
    print(f"matrix orientation for sections [4]-[7]: {coll['orientation']}")
    if "stability" in result:
        s = result["stability"]
        print(f"\n[0] cross-process id stability ({s['repeats']} runs of one input)")
        print(
            f"    sizes {s['sizes']}  union {s['union']}  intersection {s['intersection']}  "
            f"jaccard {s['jaccard']:.3f}"
        )
        if s["jaccard"] < 1.0:
            print("    WARNING: edge ids are not reproducible across processes. Every per-edge")
            print("    statistic below is aggregating ids that do not denote the same edge.")
            print("    First suspect: a PIE target under ASLR with __AFL_CTX_SENSITIVE=1.")
        if not s["first_exec_matches"]:
            print(
                f"    first execution (clean table) disagreed with the steady state; "
                f"{s['first_exec_only']} of its ids never reappear"
            )

    a = result["axis"]
    print(f"\n[1] x axis: {a['n_edges']} distinct ids in [{a['id_min']}, {a['id_max']}]")
    print(
        f"    ctx_bits {a['ctx_bits']}: {a['families']} context-free families "
        f"(id >> {a['ctx_bits']}), largest {a['largest_families']}"
    )
    print(
        f"    reachable ctx tags per family {a['ctx_tags_reachable']} "
        f"(|= 1 kills tag bit 0; all ids odd: {a['all_ids_odd']})"
    )
    degenerate = (
        " (degenerate: one edge per family, nothing to decompose)"
        if a["families"] == a["n_edges"]
        else ""
    )
    print(f"    ICC: family explains {a['icc_family']:.3f} of log2(1+count) variance{degenerate}")
    print(
        f"    lag-1 autocorr along id order {a['lag1_observed']:+.4f}  "
        f"null {a['lag1_null_mean']:+.4f} +/- {a['lag1_null_sd']:.4f}  z={a['lag1_z']:+.2f}"
    )
    print(
        f"    control, shuffled within family: {a['lag1_within_family_shuffle']:+.4f} "
        "(should stay near observed)"
    )
    print(
        f"    Spearman(id, count) {a['spearman_id_vs_count']:+.4f} "
        "-- not interpretable as a trend; reported to show it is not zero either"
    )

    y = result["y_marginal"]
    print(f"\n[2] y marginal: mean {y['mean']:.1f} median {y['median']:.0f} max {y['max']:.0f}")
    print(
        f"    Shannon {y['entropy_bits']:.2f} of {y['entropy_max_bits']:.2f} bits "
        f"-> {y['effective_edges']:.0f} effective edges"
    )
    print(
        f"    Gini {y['gini']:.3f}  top 1% of edges carry {y['top_1pct_share'] * 100:.1f}% of hits"
    )
    print(f"    Zipf log-log slope {y['zipf_slope']:+.3f} (residual sd {y['zipf_resid_sd']:.3f})")
    print(f"    AFL count classes {y['count_classes']}")

    s = result["substituted_axes"]
    print("\n[3] substituted x axes, Spearman vs total count")
    print(f"    owner count (incidence)  {s['spearman_owners_vs_total']:+.3f}")
    print(f"    first-seen index         {s['spearman_first_seen_vs_total']:+.3f}")
    print(f"    per-edge max count       {s['spearman_peak_vs_total']:+.3f}")
    print("    distance-table node_idx: not collected here -- needs a __AFL_DISTANCE_MODE build")

    sp = result["spectrum"]
    print(
        f"\n[4] {result['collection']['orientation']} spectrum "
        "(not the (id, count) columns -- those are not a matrix)"
    )
    if result["collection"]["orientation"] == "edge x seed":
        print("    identical to the seed x edge spectrum by construction, sigma(A) = sigma(A^T);")
        print("    printed as a control -- a difference here means the pipeline is wrong.")
    if "skipped" in sp:
        print(f"    skipped: {sp['skipped']}")
    else:
        for label in ("raw", "log1p", "binary"):
            e = sp[label]
            print(
                f"    {label:7s} dominance {e['dominance']:.3f}  effective rank "
                f"{e['effective_rank']:5.1f}  rho k=1/3/10/20 "
                + "/".join(f"{e['rho'][k]:.3f}" for k in ("1", "3", "10", "20"))
            )
        if abs(sp["raw"]["dominance"] - sp["binary"]["dominance"]) < 1e-6:
            print("    WARNING: raw and binary spectra agree to 1e-6. That is a bug upstream")
            print("    of both transforms, not a result -- check the cells are counts.")

    g = result["gf2"]
    transposed = result["collection"]["orientation"] == "edge x seed"
    row, col = ("edges", "seeds") if transposed else ("seeds", "edges")
    print(
        f"\n[5] GF(2) structure: rank {g['gf2_rank']} (real rank {g['real_rank']}, "
        f"{g['distinct_rows']} distinct rows of {g['rows']} {row})"
    )
    print(f"    XOR-dependent {row} {g['xor_dependent_rows']} -- sound but incomplete: each one")
    print(f"    is union-redundant, but {g['union_redundant_seeds']} {row} actually are")
    if g["greedy_cover_seeds"] is not None:
        print(
            f"    as a minimiser: GF(2) basis {g['gf2_rank']} {row} vs greedy set cover "
            f"{g['greedy_cover_seeds']}"
        )
    print(
        f"    {g['pivot_edges']} pivot {col} separate {g['distinct_rows_on_pivots']} of "
        f"{g['distinct_rows']} distinct profiles"
    )
    if transposed:
        print("    note: union redundancy and greedy cover are seed-side metrics. Transposed")
        print("    they degenerate -- every edge sits inside the union of the others, and the")
        print("    cover is the single edge every input reaches.")

    if "integer_relations" in result:
        r = result["integer_relations"]
        print(
            f"\n[6] integer relations over the raw counts ({r['rows']} rows, "
            f"{r['distinct_rows']} distinct, {r['duplicate_rows']} exact duplicates)"
        )
        print(
            f"    sparse relations, exhaustive: {r['scalar_multiple_pairs']} scalar-multiple "
            f"pairs, {r['sum_triples']} A=B+C triples"
        )
        lll = r["lll"]
        print(
            f"    LLL on {lll['rows']}x{lll['cols']} (rank {lll['rank']}, kernel "
            f"{lll['kernel_dim']}): {lll['relations_found']} exact relations in "
            f"{lll['seconds']:.1f}s"
        )
        if lll["relations_found"]:
            print(
                f"    support median {lll['support_median']:.0f} of {lll['rows']}, "
                f"L1 median {lll['l1_median']:.0f}, max|coeff| median "
                f"{lll['max_coeff_median']:.0f}"
            )
            sparse = lll["support_median"] <= 8 and lll["max_coeff_median"] <= 2
            if sparse:
                print("    sparse, near-unit relations: on the edge orientation these are")
                print("    flow conservation on the CFG (Ball-Larus). Cross-check against")
                print("    core/icfg.py before treating any of them as structural -- holding")
                print("    over one corpus does not distinguish structural from coincidental.")
            else:
                print("    dense relations are not actionable: the only short vectors here")
                print("    are duplicate-row differences, which the hash above finds in O(m).")

    if "duplicate_classes" in result:
        d = result["duplicate_classes"]
        print(
            f"\n[7] edge equivalence classes: {d['classes']} distinct profiles for "
            f"{d['rows']} edges ({d['duplicate_rows']} exact copies)"
        )
        print(
            f"    within one id>>ctx_bits family: {d['within_family_rows']} edges in "
            f"{d['within_family_classes']} classes -- context tags that never differed"
        )
        print(
            f"    across families: {d['across_family_rows']} edges in "
            f"{d['across_family_classes']} classes -- straight-line block chains"
        )
        print(
            f"    largest class {d['largest_class']} edges, families {d['largest_class_families']}"
        )

    if "positions" in result:
        p = result["positions"]
        print("\n[8] edge_pos placement (SHM slot index), (pos, id, count) matrix")
        if not p["available"]:
            print("    unavailable: this collection has no positions -- re-collect with --save")
            return
        print(
            f"    table {p['table_size']} slots; {p['triples']} triples over {p['runs']} runs, "
            f"{p['unique_ids']} distinct ids"
        )
        print(
            f"    ids at a single position {p['ids_single_position']} "
            f"({p['ids_single_position_pct']:.1f}%)  -- slots are never reclaimed, so any"
        )
        print("    multi-position id contradicts the shim's design (first-fit is final)")
        print(
            f"    slots hosting >1 distinct id across runs: {p['slots_multi_id']}"
            "  (expected: only broken placements)"
        )
        print(
            f"    probe displacement: mean {p['displacement_mean']:.3f}, "
            f"home hits {p['home_hit_frac'] * 100:.1f}%, "
            f"beyond PROBE_MAX {p['beyond_probe_max']}  "
            f"top bins {p['disp_hist_top']}"
        )
        obs = p["spearman_disp_count"]
        print(
            f"    Spearman(displacement, count) {obs:+.4f}  "
            f"null {p['spearman_disp_count_null_mean']:+.4f} +/- "
            f"{p['spearman_disp_count_null_sd']:.4f}  z={p['spearman_disp_count_null_z']:+.2f}"
        )
        obs2 = p["spearman_home_disp"]
        print(
            f"    Spearman(home, displacement)  {obs2:+.4f}  "
            f"null {p['spearman_home_disp_null_mean']:+.4f} +/- "
            f"{p['spearman_home_disp_null_sd']:.4f}  z={p['spearman_home_disp_null_z']:+.2f}"
        )
        print(
            f"    confound: Spearman(disp, first_seen) {p['spearman_disp_first_run']:+.4f}, "
            f"Spearman(first_seen, count) {p['spearman_first_run_count']:+.4f}"
        )
        if "rank_2d" in p:
            print(
                f"    first-run (pos x id) matrix rank {p['rank_2d']}; "
                f"singular spectrum max|rel err| vs the count histogram "
                f"{p['single_run_sv_max_relerr']:.2e}"
            )
            print("    (a fold is a bijection onto its image: the pos x id view is the (id, count)")
            print("     view re-rendered, per handover F12 -- this measures that, F15)")


# ── Entry point ───────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--target", type=Path, help="instrumented target taking an input path in argv[1]"
    )
    ap.add_argument("--corpus", type=Path, help="directory of inputs to run (one execution each)")
    ap.add_argument("--load", type=Path, help="re-analyse a collection saved with --save")
    ap.add_argument("--save", type=Path, help="write the collected columns to an .npz")
    ap.add_argument("--json", type=Path, help="write the full result dict as JSON")
    ap.add_argument("--map-size", type=int, default=65536, help="SHM table entries (default 65536)")
    ap.add_argument(
        "--ctx-bits",
        type=int,
        default=8,
        help="__AFL_CTX_BITS the target was built with (default 8; 0 for a "
        "__AFL_CTX_SENSITIVE=0 build)",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=6,
        help="executions of one input for the stability check (0 to skip)",
    )
    ap.add_argument(
        "--keep-aslr",
        action="store_true",
        help="do not disable ASLR (raw-address ctx regime; export "
        "FUZZER_KEEP_ASLR=1 too for what a production run with ASLR on sees)",
    )
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument(
        "--transpose",
        action="store_true",
        help="run sections [4]-[6] on the edge x seed matrix and add section [7]",
    )
    ap.add_argument(
        "--lll",
        action="store_true",
        help="run section [6]: integer relations over the raw counts (slow)",
    )
    ap.add_argument(
        "--lll-rows",
        type=int,
        default=LLL_ROW_BUDGET,
        help=f"distinct rows fed to LLL (default {LLL_ROW_BUDGET}; cost is superlinear)",
    )
    ap.add_argument("--perms", type=int, default=2000, help="permutation null samples")
    ap.add_argument(
        "--positions",
        action="store_true",
        help="run section [8]: the (edge_pos, edge_id, count) placement matrix",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    if args.load is None and (args.target is None or args.corpus is None):
        ap.error("either --load, or both --target and --corpus")

    if args.load is not None:
        runs_all = load_runs(args.load)
        stability = None
    else:
        if not args.keep_aslr:
            disable_aslr()
        inputs = sorted(p for p in args.corpus.rglob("*") if p.is_file())
        if not inputs:
            ap.error(f"no inputs under {args.corpus}")
        runs_all = collect(args.target, inputs, args.map_size, args.timeout)
        stability = (
            measure_stability(args.target, inputs[0], args.repeats, args.map_size, args.timeout)
            if args.repeats > 1
            else None
        )
        if args.save is not None:
            save_runs(args.save, runs_all, args.map_size)

    if runs_all and len(runs_all[0]) == 3:
        pos_runs = runs_all
        runs = [(ids, counts) for _, ids, counts in runs_all]
    else:
        pos_runs = None
        runs = runs_all
    collected_map = saved_map_size(args.load) if args.load is not None else args.map_size

    agg = aggregate(runs)
    mat = seed_edge_matrix(runs)
    # The spectrum and both ranks are transpose-invariant; everything derived
    # from them is not. See the handover's F10 for the comparison.
    if args.transpose:
        mat = mat.T
    if len(agg["ids"]) == 0:
        print("no edges recorded -- is the target instrumented and linked against the shim?")
        return 1

    live = sorted(len(ids) for ids, _ in runs)
    result = {
        "collection": {
            "inputs": len(runs),
            "median_live_edges": live[len(live) // 2],
            "union_edges": int(len(agg["ids"])),
            "total_hits": int(agg["total"].sum()),
            "aslr_disabled": (not args.keep_aslr) if args.load is None else None,
            "orientation": "edge x seed" if args.transpose else "seed x edge",
        },
        "axis": axis_structure(agg["ids"], agg["total"], args.ctx_bits, args.perms, args.seed),
        "y_marginal": y_marginal(agg["total"]),
        "substituted_axes": substituted_axes(agg),
        "spectrum": spectrum(mat),
        "gf2": gf2_structure(mat),
    }
    if args.transpose:
        result["duplicate_classes"] = duplicate_classes(mat, agg["ids"], args.ctx_bits)
    if args.lll:
        result["integer_relations"] = integer_relations(mat, args.lll_rows)
    if args.positions:
        result["positions"] = positions_structure(pos_runs, collected_map, args.perms, args.seed)
    if stability is not None:
        result["stability"] = stability

    _report(result)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
