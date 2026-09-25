"""Analyses for ``edge_diagnostic.py matrix`` beyond sections [0]-[9].

Section numbers continue the tool's own -- [10] is ``--flow`` (P1-2, closed
negative, F17): [11] flaky edges, [12] subsumption, [13] admission replay,
[14] rarefaction, [15] bootstrap, [16] length confound, [17] prefix fold,
[18] score audit.

Pure numpy over the same collected columns the tool already holds (``mat``: one row
per execution in corpus order, one column per edge id, cell = hit count), plus one
extra input the tool did not collect before: repeated runs of each input (flaky
edges). Nothing here uses an id-axis statistic (F17 already closed the
count-relation half of that question via ``--flow``; see
``handover_edge_id_axis_2026-09-18.md``, "Not defined on the id axis"). The one
positional analysis, ``prefix_fold``, is the bit-prefix bucketing that section names
as the defined replacement. Randomness comes from ``RandPool`` (Hard Rule 16).

Screening tools, not verdicts: ``score_audit`` can retire a seed score, never
promote one -- observational correlation has been wrong twice on exactly this
question, so promotion stays with ``tools/bench_paired.py``.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from fuzzer_tool.core.count_class import classify_single
from fuzzer_tool.core.edge_matrix import (
    build_fold,
    partial_rank_corr,
    rank_corr,
    residualize_ranks,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.scheduler_substrate import EdgeCanonicalizer

SUBSUME_BUDGET = 2_000  # distinct rows for the poset
WIDTH_BUDGET = 800  # distinct rows for the antichain matching
SVD_CELL_BUDGET = 20_000_000
BUCKET_SPAN = 256  # token = column * BUCKET_SPAN + bucket
MIN_RARE_SEEDS = 4
RARE_FRACTIONS = (0.25, 0.5, 0.75)
CURVE_POINTS = 10
TAIL_FRACTION = 0.1
CHAO_BIAS_BELOW = 10  # mirrors edge_tracker._CHAO_BIAS_CORRECT_BELOW
CI_Z = 1.96
CI_LOW_PCT, CI_HIGH_PCT = 2.5, 97.5
MIN_AUDIT_SEEDS = 8
AUDIT_RANK = 3
KILL_PARTIAL = 0.1  # screening cut, not a pre-registered threshold
TOP_LIST = 5
MAX_IDS = 1_000


def _off(reason: str) -> dict:
    return {"available": False, "reason": reason}


def greedy_cover(binary: np.ndarray) -> int:
    """Seeds a greedy set cover of the union needs (rows = seeds)."""
    need = np.ones(binary.shape[1], dtype=bool)
    picked = 0
    while need.any():
        gains = binary[:, need].sum(axis=1)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        need &= ~binary[best]
        picked += 1
    return picked


def _eff_rank(cells: np.ndarray) -> float:
    sq = np.linalg.svd(cells, compute_uv=False) ** 2
    mass = float(sq.sum())
    return float(mass**2 / float((sq**2).sum())) if mass else 0.0


def _pool_perms(m: int, count: int, seed: int) -> list[list[int]]:
    pool = RandPool(seed)
    out = []
    for _ in range(count):
        order = list(range(m))
        pool.shuffle(order)
        out.append(order)
    return out


# ── [11] flaky edges ──────────────────────────────────────────────────


def _input_variance(reps) -> tuple[set[int], set[int]]:
    """(edges absent from some repeat, edges present in all with varying counts)."""
    per = [dict(zip(ids.tolist(), counts.tolist(), strict=True)) for ids, counts in reps]
    seen = set().union(*per)
    absent = {e for e in seen if any(e not in d for d in per)}
    varying = {e for e in seen - absent if len({d[e] for d in per}) > 1}
    return absent, varying


def _variance_tally(rep_runs) -> tuple[Counter, Counter, int]:
    """Presence/count-flakiness tallies over every input, and how many are affected."""
    presence: Counter[int] = Counter()
    varying: Counter[int] = Counter()
    affected = 0
    for reps in rep_runs:
        absent, vary = _input_variance(reps)
        presence.update(absent)
        varying.update(vary)
        affected += bool(absent or vary)
    return presence, varying, affected


def _private_edges(rep_runs) -> list[set[int]]:
    """Per input, the edges only it (of the first repeat's set) reaches."""
    firsts = [set(reps[0][0].tolist()) for reps in rep_runs]
    owners = Counter(e for s in firsts for e in s)
    return [{e for e in s if owners[e] == 1} for s in firsts]


def flaky_edges(rep_runs) -> dict:
    """Edges whose presence or count changes across repeats of the *same* input.

    ``rep_runs``: per input, a list of (ids, counts) from repeated executions. A
    flaky edge is a false-novelty source: it can make a seed look private.
    """
    if not rep_runs or min(len(r) for r in rep_runs) < 2:
        return _off("needs at least two repeats of every input")
    presence, varying, affected = _variance_tally(rep_runs)
    union = set().union(*(set(ids.tolist()) for reps in rep_runs for ids, _ in reps))
    flaky = set(presence) | set(varying)
    has_private = [p for p in _private_edges(rep_runs) if p]
    return {
        "available": True,
        "inputs": len(rep_runs),
        "repeats": min(len(r) for r in rep_runs),
        "union_edges": len(union),
        "presence_flaky": len(set(presence)),
        "count_flaky": len(set(varying) - set(presence)),
        "flaky_fraction": len(flaky) / len(union) if union else 0.0,
        "flaky_inputs": affected,
        "seeds_with_private": len(has_private),
        "private_all_flaky": sum(1 for p in has_private if p <= flaky),
        "worst": [[e, n] for e, n in (presence + varying).most_common(TOP_LIST)],
        "flaky_ids": sorted(flaky)[:MAX_IDS],
    }


# ── [12] subsumption ──────────────────────────────────────────────────


def _row_bits(binary: np.ndarray) -> list[int]:
    packed = np.packbits(binary, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def _above(vals: list[int]) -> list[list[int]]:
    """For each value, the indices of strict supersets (values are distinct)."""
    pops = [v.bit_count() for v in vals]
    return [
        [j for j, w in enumerate(vals) if pops[j] > pops[i] and v & w == v]
        for i, v in enumerate(vals)
    ]


def _chain_depth(vals: list[int], above: list[list[int]]) -> int:
    depth = [1] * len(vals)
    for i in sorted(range(len(vals)), key=lambda k: vals[k].bit_count()):
        for j in above[i]:
            depth[j] = max(depth[j], depth[i] + 1)
    return max(depth, default=0)


def _matching(adj: list[list[int]]) -> int:
    """Maximum bipartite matching (Kuhn, iterative augmenting paths)."""
    match_r = [-1] * len(adj)
    size = 0
    for start in range(len(adj)):
        seen = [False] * len(adj)
        stack, iters, chosen = [start], {start: iter(adj[start])}, []
        while stack:
            u = stack[-1]
            step = next((v for v in iters[u] if not seen[v]), None)
            if step is None:
                stack.pop()
                if chosen:
                    chosen.pop()
                continue
            seen[step] = True
            chosen.append(step)
            nxt = match_r[step]
            if nxt == -1:
                for lu, rv in zip(stack, chosen, strict=True):
                    match_r[rv] = lu
                size += 1
                break
            stack.append(nxt)
            iters[nxt] = iter(adj[nxt])
    return size


def _seed_poset(vals: list[int], rows: int) -> dict:
    n = len(vals)
    base = {"rows": rows, "distinct_rows": n, "duplicate_rows": rows - n}
    if n > SUBSUME_BUDGET:
        return {**base, "skipped": f"{n} distinct rows over budget {SUBSUME_BUDGET}"}
    above = _above(vals)
    maximal = sum(1 for a in above if not a)
    width = n - _matching(above) if n <= WIDTH_BUDGET else None
    return {
        **base,
        "subsumed_distinct": n - maximal,
        "maximal_rows": maximal,
        "width": width,
        "height": _chain_depth(vals, above),
    }


def _edge_poset(binary: np.ndarray, ids: np.ndarray) -> dict:
    groups: dict[int, list[int]] = {}
    for e, bits in enumerate(_row_bits(binary.T.copy())):
        groups.setdefault(bits, []).append(e)
    vals = list(groups)
    n_edges = binary.shape[1]
    base = {"edges": n_edges, "classes": len(vals), "duplicate_edges": n_edges - len(vals)}
    if len(vals) > SUBSUME_BUDGET:
        return {**base, "skipped": f"{len(vals)} classes over budget {SUBSUME_BUDGET}"}
    above = _above(vals)
    sizes = [len(groups[v]) for v in vals]
    has_below = {j for a in above for j in a}
    implied = [sum(sizes[j] for j in a) for a in above]
    universal = (1 << binary.shape[0]) - 1
    order = sorted(range(len(vals)), key=lambda k: (-implied[k], -vals[k].bit_count(), k))
    gateways = [
        {
            "ids": [int(ids[e]) for e in groups[vals[k]][:3]],
            "size": sizes[k],
            "support": vals[k].bit_count(),
            "implied_edges": implied[k],
        }
        for k in order[:TOP_LIST]
        if implied[k] > 0
    ]
    return {
        **base,
        "universal_edges": len(groups.get(universal, [])),
        "implication_pairs": sum(len(a) for a in above),
        "depth": _chain_depth(vals, above),
        "deepest_classes": len(vals) - len(has_below),
        "gateways": gateways,
    }


def subsumption(mat: np.ndarray, ids: np.ndarray) -> dict:
    """Seed-side poset (subset order on edge sets) and edge-side implications.

    Seeds: how many are strictly contained in another, the maximal seeds (a valid
    cover, so an upper bound on the minimum one), antichain width (Dilworth) and
    longest chain. GF(2) elimination cannot see subset dominance -- its docstring
    says so -- and this can, exactly.

    Edges: ``a => b`` when every seed reaching ``a`` also reaches ``b``. Equal
    supports collapse into one class (section [7]'s duplicates); the strict order
    between classes is the empirical dominator structure. ``gateways`` are the
    classes whose firing implies the most other edges.
    """
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return _off("empty matrix")
    binary = mat > 0
    return {
        "seeds": _seed_poset(sorted(set(_row_bits(binary))), binary.shape[0]),
        "edges": _edge_poset(binary, np.asarray(ids)),
    }


# ── [13] admission replay and resolution ladder ───────────────────────


def _bucketize(mat: np.ndarray) -> np.ndarray:
    out = np.zeros(mat.shape, dtype=np.int64)
    for value in np.unique(mat[mat > 0]).astype(np.int64).tolist():
        out[mat == value] = classify_single(value)
    return out


def _token_rows(mat: np.ndarray, buckets: np.ndarray) -> list[list[int]]:
    rows = []
    for i in range(mat.shape[0]):
        cols = np.flatnonzero(mat[i])
        rows.append((cols * BUCKET_SPAN + buckets[i, cols]).tolist())
    return rows


def _admit_tokens(rows: list[list[int]], order: list[int]) -> list[int]:
    seen: set[int] = set()
    admitted = []
    for i in order:
        new = [t for t in rows[i] if t not in seen]
        if not new:
            continue
        admitted.append(i)
        seen.update(new)
    return admitted


def _admit_max(mat: np.ndarray, order: list[int]) -> list[int]:
    top = np.zeros(mat.shape[1])
    admitted = []
    for i in order:
        if not (mat[i] > top).any():
            continue
        admitted.append(i)
        np.maximum(top, mat[i], out=top)
    return admitted


def _spread(counts: list[int]) -> dict:
    if not counts:
        return {"shuffled_mean": None, "shuffled_sd": None, "shuffled_min": None,
                "shuffled_max": None}  # fmt: skip
    arr = np.array(counts, dtype=float)
    return {
        "shuffled_mean": float(arr.mean()),
        "shuffled_sd": float(arr.std()),
        "shuffled_min": int(arr.min()),
        "shuffled_max": int(arr.max()),
    }


def resolution_ladder(mat: np.ndarray) -> dict:
    """Distinct rows and effective rank as the cell semantics get finer."""
    cells = {
        "binary": (mat > 0).astype(float),
        "bucket": _bucketize(mat).astype(float),
        "log1p": np.log1p(mat),
        "raw": mat,
    }
    return {
        label: {
            "distinct_rows": len({r.tobytes() for r in c}),
            "effective_rank": _eff_rank(c) if c.size <= SVD_CELL_BUDGET else None,
        }
        for label, c in cells.items()
    }


def admission_replay(mat: np.ndarray, shuffles: int, seed: int) -> dict:
    """Replay the corpus through three admission rules, in order and shuffled.

    ``edge`` admits on a new edge, ``bucket`` on a new (edge, AFL count class) --
    the hit-count axis of the matrix -- and ``maxcount`` on any count above the
    edge's previous maximum. Admission is order dependent, so the corpus order is
    reported next to the spread over *shuffles* random orders. Every rule admits the
    first seed to reach any edge, hence ``retains_union`` is a theorem, not a result.
    """
    m = mat.shape[0]
    if m == 0 or not (mat > 0).any():
        return _off("no coverage")
    tokens = {
        "edge": _token_rows(mat, np.zeros(mat.shape, dtype=np.int64)),
        "bucket": _token_rows(mat, _bucketize(mat)),
    }
    rules = {
        "edge": lambda o: _admit_tokens(tokens["edge"], o),
        "bucket": lambda o: _admit_tokens(tokens["bucket"], o),
        "maxcount": lambda o: _admit_max(mat, o),
    }
    orders = [list(range(m))] + _pool_perms(m, shuffles, seed)
    got = {name: [fn(o) for o in orders] for name, fn in rules.items()}
    union = set(np.flatnonzero((mat > 0).any(axis=0)).tolist())
    edge_n = [len(a) for a in got["edge"]]
    out = {}
    for name, runs in got.items():
        sizes = [len(a) for a in runs]
        extra = [s - e for s, e in zip(sizes[1:], edge_n[1:], strict=True)]
        out[name] = {
            "corpus_order": sizes[0],
            **_spread(sizes[1:]),
            "extra_over_edge_mean": float(np.mean(extra)) if extra else None,
            "retains_union": all(
                set(np.flatnonzero((mat[a] > 0).any(axis=0)).tolist()) == union for a in runs
            ),
        }
    return {
        "available": True,
        "seeds": m,
        "shuffles": shuffles,
        "rules": out,
        "greedy_cover": greedy_cover(mat > 0),
        "ladder": resolution_ladder(mat),
    }


# ── [14] rarefaction and Chao2 calibration ────────────────────────────


def chao2(owners: np.ndarray, m: int) -> dict:
    """Chao2 richness from incidence counts; the formulas of
    ``EdgeTracker.good_turing_estimate`` (pinned against it by a test)."""
    counts = np.asarray(owners)
    s_obs = int(len(counts))
    base = {"n": s_obs, "m": m, "chao2": float(s_obs), "f0": 0.0, "ci_low": float(s_obs),
            "ci_high": float(s_obs), "sample_coverage": 1.0 if s_obs else 0.0}  # fmt: skip
    if s_obs == 0 or m < 2:
        return base
    q1, q2 = int((counts == 1).sum()), int((counts == 2).sum())
    a = (m - 1) / m
    f0 = a * q1 * q1 / (2.0 * q2) if q2 >= CHAO_BIAS_BELOW else a * q1 * (q1 - 1) / (2.0 * (q2 + 1))
    total = s_obs + f0
    low, high = _chao_ci(q1, q2, a, f0, total, s_obs)
    denom = (m - 1) * q1 + 2 * q2
    cov = 1.0 - (q1 / int(counts.sum())) * ((m - 1) * q1 / denom) if denom else 1.0
    return {**base, "chao2": total, "f0": f0, "ci_low": low, "ci_high": high,
            "sample_coverage": min(max(cov, 0.0), 1.0), "n1": q1, "n2": q2}  # fmt: skip


def _chao_ci(q1: int, q2: int, a: float, f0: float, total: float, s_obs: int):
    if q2 > 0:
        r = q1 / q2
        var = q2 * ((a / 2.0) * r**2 + (a**2) * r**3 + (a**2 / 4.0) * r**4)
    elif q1 > 0 and total > 0:
        var = max(
            a * q1 * (q1 - 1) / 2.0
            + (a**2) * q1 * (2 * q1 - 1) ** 2 / 4.0
            - (a**2) * q1**4 / (4.0 * total),
            0.0,
        )
    else:
        var = 0.0
    if f0 <= 0 or var <= 0:
        return float(total), float(total)
    k = math.exp(CI_Z * math.sqrt(math.log(1.0 + var / (f0 * f0))))
    return s_obs + f0 / k, s_obs + f0 * k


def _first_seen_curve(binary: np.ndarray, order: list[int]) -> np.ndarray:
    """Edges discovered after each seed, when seeds arrive in *order*."""
    sub = binary[order]
    live = sub.any(axis=0)
    first = sub.argmax(axis=0)[live]
    return np.cumsum(np.bincount(first, minlength=len(order)))


def _calibrate(binary: np.ndarray, perms, full: int) -> list[dict]:
    m = binary.shape[0]
    rows = []
    for frac in RARE_FRACTIONS:
        k = max(2, int(frac * m))
        est, covers, ratio = [], [], []
        for order in perms:
            owners = binary[order[:k]].sum(axis=0)
            owners = owners[owners > 0]
            e = chao2(owners, k)
            est.append((e["chao2"] - full) / full)
            covers.append(e["ci_low"] <= full <= e["ci_high"])
            ratio.append(len(owners) / full)
        rows.append({
            "fraction": frac,
            "seeds": k,
            "obs_ratio": float(np.mean(ratio)),
            "mean_rel_error": float(np.mean(est)),
            "ci_covers_full": float(np.mean(covers)),
        })  # fmt: skip
    return rows


def rarefaction(mat: np.ndarray, resamples: int, seed: int) -> dict:
    """Accumulation curve against its permutation null, and Chao2 calibration.

    ``order``: area under the corpus-order discovery curve against random orders
    (a front-loaded corpus scores above its null). ``calibration``: Chao2 fitted on
    the first fraction of a random order, against the union of the whole corpus --
    the live estimator's bias and how often its 95% interval contains the truth.
    Seeds are not an iid sample of an input population, so read the bias as a
    property of this corpus, not of the estimator.
    """
    binary = mat > 0
    m = binary.shape[0]
    if m < MIN_RARE_SEEDS or not binary.any():
        return _off(f"needs {MIN_RARE_SEEDS}+ seeds with coverage")
    full = int(binary.any(axis=0).sum())
    perms = _pool_perms(m, resamples, seed)
    obs = _first_seen_curve(binary, list(range(m)))
    null = np.array([_first_seen_curve(binary, o) for o in perms], dtype=float)
    area = float(obs.mean() / full)
    null_area = null.mean(axis=1) / full
    sd = float(null_area.std())
    tail = max(int((1 - TAIL_FRACTION) * m), 1)
    grid = sorted({max(1, round(f * m)) for f in np.linspace(1 / CURVE_POINTS, 1, CURVE_POINTS)})
    return {
        "available": True,
        "seeds": m,
        "union_edges": full,
        "resamples": resamples,
        "grid": grid,
        "curve_obs": [int(obs[k - 1]) for k in grid],
        "curve_mean": [float(null[:, k - 1].mean()) for k in grid],
        "order": {
            "obs_area": area,
            "null_mean": float(null_area.mean()),
            "null_sd": sd,
            "z": float((area - null_area.mean()) / sd) if sd else 0.0,
            "percentile": float((null_area <= area).mean()),
            "tail_gain_obs": float((full - obs[tail - 1]) / full),
            "tail_gain_null": float(((full - null[:, tail - 1]) / full).mean()),
        },
        "full": chao2(binary.sum(axis=0)[binary.any(axis=0)], m),
        "calibration": _calibrate(binary, perms, full),
    }


# ── [15] bootstrap ────────────────────────────────────────────────────


def bootstrap_ci(mat: np.ndarray, stat_fn, resamples: int, seed: int) -> dict:
    """Percentile intervals of ``stat_fn(mat)`` over seed resamples.

    ``stat_fn`` maps a matrix to ``{name: float}``. Resampling seeds with
    replacement shrinks any count of *distinct* things (union edges, distinct rows),
    so ``bias`` is reported next to the interval rather than hidden in it.
    """
    m = mat.shape[0]
    if resamples < 1 or m < 2:
        return _off("needs 2+ seeds and 1+ resamples")
    pool = RandPool(seed)
    point = stat_fn(mat)
    reps = [stat_fn(mat[pool.randrange_list(m, m)]) for _ in range(resamples)]
    stats = {}
    for key, value in point.items():
        arr = np.array([r[key] for r in reps], dtype=float)
        stats[key] = {
            "point": float(value),
            "mean": float(arr.mean()),
            "sd": float(arr.std()),
            "low": float(np.percentile(arr, CI_LOW_PCT)),
            "high": float(np.percentile(arr, CI_HIGH_PCT)),
            "bias": float(arr.mean() - value),
        }
    return {"available": True, "resamples": resamples, "seeds": m, "stats": stats}


# ── shared per-seed quantities ────────────────────────────────────────


def _profiles(mat: np.ndarray, ids: np.ndarray) -> dict[str, dict[int, int]]:
    return {
        str(i): {int(ids[j]): int(mat[i, j]) for j in np.flatnonzero(mat[i])}
        for i in range(mat.shape[0])
    }


def _pc1(mat: np.ndarray) -> np.ndarray | None:
    if mat.size > SVD_CELL_BUDGET:
        return None
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    vec = u[:, 0] * s[0]
    return -vec if vec.sum() < 0 else vec


# ── [16] length confound ──────────────────────────────────────────────


def length_confound(mat: np.ndarray, ids: np.ndarray, sizes) -> dict:
    """Is "volume" just input size?

    Spearman of input size against the seed quantities the scheduler arms read.
    ``residual`` is exactly ``seed_residual``'s score (mass after regressing out
    rank(total)); if it still tracks size, the arm's "beyond volume" component is
    length by another name.
    """
    if sizes is None:
        return _off("no input sizes in this collection (re-collect with --save)")
    sizes = np.asarray(sizes, dtype=float)
    if len(sizes) != mat.shape[0]:
        raise ValueError(f"sizes has {len(sizes)} entries for {mat.shape[0]} seeds")
    fold, reason = build_fold(_profiles(mat, ids), EdgeCanonicalizer())
    if fold is None:
        return _off(reason)
    quantities = {
        "total": fold.total,
        "degree": fold.degree,
        "mass": fold.mass,
        "residual": residualize_ranks(fold.mass, fold.total),
    }
    pc1 = _pc1(mat)
    if pc1 is not None:
        quantities["pc1"] = pc1
    out = {"available": True, "seeds": int(len(sizes))}
    for name, values in quantities.items():
        out[f"rho_size_{name}"] = rank_corr(sizes, values)
    out["r2_total_on_size"] = out["rho_size_total"] ** 2
    per_byte = fold.total[sizes > 0] / sizes[sizes > 0]
    mean = float(per_byte.mean()) if len(per_byte) else 0.0
    out["hits_per_byte_cv"] = float(per_byte.std() / mean) if mean else 0.0
    return out


# ── [17] prefix fold ──────────────────────────────────────────────────


def _fold_bits(mat: np.ndarray, ids: np.ndarray, bits: int) -> np.ndarray:
    keys = ids >> bits
    starts = np.concatenate([[0], np.flatnonzero(np.diff(keys)) + 1])
    return np.add.reduceat(mat, starts, axis=1)


def prefix_fold(mat: np.ndarray, ids: np.ndarray, ctx_bits: int) -> dict:
    """Distinct rows and rank as the low id bits (the context tag) are folded away.

    ``ctx_only_rows_*`` is how many rows exist only because of the context tag. A
    flat tail means those bits bought nothing on this corpus (input to
    ``__AFL_CTX_BITS`` sizing). Bit-prefix bucketing is defined on the id axis;
    hash collisions between locations blur it, so confirm with ``--ground-truth``.
    """
    if ctx_bits < 1:
        return _off("ctx_bits is 0: nothing to fold")
    order = np.argsort(ids, kind="stable")
    ids, mat = np.asarray(ids)[order], mat[:, order]
    per_bit = []
    for b in range(ctx_bits + 1):
        folded = _fold_bits(mat, ids, b)
        binary = (folded > 0).astype(float)
        per_bit.append({
            "bit": b,
            "edges": int(folded.shape[1]),
            "rows_binary": len({r.tobytes() for r in binary}),
            "rows_raw": len({r.tobytes() for r in folded}),
            "binary_rank": int(np.linalg.matrix_rank(binary)),
        })  # fmt: skip
    return {
        "available": True,
        "ctx_bits": ctx_bits,
        "per_bit": per_bit,
        "ctx_only_rows_binary": per_bit[0]["rows_binary"] - per_bit[-1]["rows_binary"],
        "ctx_only_rows_raw": per_bit[0]["rows_raw"] - per_bit[-1]["rows_raw"],
    }


# ── [18] score audit ──────────────────────────────────────────────────


def loo_edges(binary: np.ndarray) -> np.ndarray:
    """Edges each seed alone reaches: what removing it would lose."""
    return (binary & (binary.sum(axis=0) == 1)).sum(axis=1)


def _loo_classes(fold) -> np.ndarray:
    return np.array(
        [sum(1 for c in cls.tolist() if fold.class_owners[c] == 1) for cls in fold.seed_classes],
        dtype=float,
    )


def _subspace(binary: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """(row leverage, distance from the rank-k subspace) per seed."""
    cells = binary.astype(float)
    u, s, vt = np.linalg.svd(cells, full_matrices=False)
    k = min(rank, len(s))
    leverage = (u[:, :k] ** 2).sum(axis=1)
    proj = cells @ vt[:k].T
    dist = np.sqrt(np.maximum((cells**2).sum(axis=1) - (proj**2).sum(axis=1), 0.0))
    return leverage, dist


def _audit_row(score, outcome, total, degree) -> dict:
    return {
        "rho": rank_corr(score, outcome),
        "partial_total": partial_rank_corr(score, outcome, total),
        "partial_degree": partial_rank_corr(score, outcome, degree),
    }


def score_audit(mat: np.ndarray, ids: np.ndarray, rank: int = AUDIT_RANK, outcomes=None) -> dict:
    """Screen seed scores against leave-one-out outcomes, controlling for volume.

    ``dead``: the score's partial correlation with the outcome, controlling for
    total hits or for degree, is under ``KILL_PARTIAL`` -- it is a row sum in
    disguise (Tang: +0.006, Wilcoxon p = 1.0). Surviving proves nothing; see the
    module docstring. ``leverage`` and ``subspace_distance`` are the rank-*rank*
    row leverage and residual norm of a seed, i.e. P3-2's "column leverage" in the
    seed x edge orientation.
    """
    if mat.shape[0] < MIN_AUDIT_SEEDS:
        return _off(f"needs {MIN_AUDIT_SEEDS}+ seeds")
    fold, reason = build_fold(_profiles(mat, ids), EdgeCanonicalizer())
    if fold is None:
        return _off(reason)
    binary = mat > 0
    leverage, distance = _subspace(binary, rank)
    outs = outcomes or {
        "loo_edges": loo_edges(binary).astype(float),
        "loo_classes": _loo_classes(fold),
    }
    scores = {
        "mass": fold.mass,
        "residual": residualize_ranks(fold.mass, fold.total),
        "leverage": leverage,
        "subspace_distance": distance,
    }
    pc1 = _pc1(mat)
    if pc1 is not None:
        scores["pc1"] = pc1
    candidates: dict[str, dict] = {}
    for name, score in scores.items():
        rows = {o: _audit_row(score, y, fold.total, fold.degree) for o, y in outs.items()}
        for row in rows.values():
            row["dead"] = bool(min(row["partial_total"], row["partial_degree"]) < KILL_PARTIAL)
        candidates[name] = rows
    baselines = {
        name: {o: _audit_row(v, y, fold.total, fold.degree) for o, y in outs.items()}
        for name, v in (("total", fold.total), ("degree", fold.degree))
    }
    return {"available": True, "seeds": int(mat.shape[0]), "rank": rank,
            "baselines": baselines, "candidates": candidates}  # fmt: skip


# ── Reporting ─────────────────────────────────────────────────────────


def _say_flaky(r: dict) -> None:
    print(f"\n[11] flaky edges ({r['inputs']} inputs x {r['repeats']} repeats)")
    print(
        f"    union {r['union_edges']}  presence-flaky {r['presence_flaky']}  "
        f"count-flaky {r['count_flaky']}  fraction {r['flaky_fraction']:.3f}  "
        f"inputs affected {r['flaky_inputs']}"
    )
    print(
        f"    seeds with private edges {r['seeds_with_private']}, "
        f"of which private edges all flaky {r['private_all_flaky']}"
    )


def _say_subsumption(r: dict) -> None:
    s, e = r["seeds"], r["edges"]
    print("\n[12] subsumption")
    print(
        f"    seeds: {s['rows']} rows, {s['distinct_rows']} distinct, "
        f"strictly subsumed {s.get('subsumed_distinct', '-')}, "
        f"maximal {s.get('maximal_rows', '-')}, width {s.get('width', '-')}, "
        f"height {s.get('height', '-')}"
    )
    print(
        f"    edges: {e['edges']} in {e['classes']} classes, "
        f"implication pairs {e.get('implication_pairs', '-')}, depth {e.get('depth', '-')}, "
        f"deepest classes {e.get('deepest_classes', '-')}"
    )
    for g in e.get("gateways", []):
        print(f"      gateway ids {g['ids']} support {g['support']} implies {g['implied_edges']}")


def _say_admission(r: dict) -> None:
    print(f"\n[13] admission replay ({r['seeds']} seeds, {r['shuffles']} shuffles)")
    print(f"    greedy cover {r['greedy_cover']}")
    for name, v in r["rules"].items():
        spread = (
            ""
            if v["shuffled_mean"] is None
            else (f"  shuffled {v['shuffled_mean']:.1f} +/- {v['shuffled_sd']:.1f}")
        )
        print(f"    {name:<9} corpus order {v['corpus_order']}{spread}")
    for label, v in r["ladder"].items():
        rank = "-" if v["effective_rank"] is None else f"{v['effective_rank']:.2f}"
        print(f"    ladder {label:<7} distinct rows {v['distinct_rows']}  effective rank {rank}")


def _say_rarefaction(r: dict) -> None:
    o = r["order"]
    print(f"\n[14] rarefaction ({r['seeds']} seeds, union {r['union_edges']})")
    print(
        f"    corpus-order area {o['obs_area']:.3f} vs null {o['null_mean']:.3f} "
        f"(z {o['z']:+.2f}, pct {o['percentile']:.2f}); tail gain {o['tail_gain_obs']:.3f} "
        f"vs {o['tail_gain_null']:.3f}"
    )
    f = r["full"]
    print(
        f"    Chao2 on the full corpus {f['chao2']:.1f} (CI {f['ci_low']:.1f}-{f['ci_high']:.1f})"
    )
    for c in r["calibration"]:
        print(
            f"    Chao2 from {c['fraction']:.0%}: rel error {c['mean_rel_error']:+.3f}, "
            f"CI covers full {c['ci_covers_full']:.2f}, observed/full {c['obs_ratio']:.2f}"
        )


def _say_bootstrap(r: dict) -> None:
    print(f"\n[15] bootstrap over seeds ({r['resamples']} resamples)")
    for name, s in r["stats"].items():
        print(
            f"    {name:<16} point {s['point']:.2f}  95% [{s['low']:.2f}, {s['high']:.2f}]  "
            f"sd {s['sd']:.2f}  bias {s['bias']:+.2f}"
        )


def _say_length(r: dict) -> None:
    print(f"\n[16] length confound ({r['seeds']} seeds)")
    keys = [k for k in r if k.startswith("rho_size_")]
    print("    " + "  ".join(f"{k[9:]} {r[k]:+.3f}" for k in keys))
    print(
        f"    R^2 total~size {r['r2_total_on_size']:.3f}  hits/byte CV {r['hits_per_byte_cv']:.3f}"
    )


def _say_prefix(r: dict) -> None:
    print(f"\n[17] prefix fold (ctx_bits {r['ctx_bits']})")
    for row in r["per_bit"]:
        print(
            f"    fold {row['bit']} bits: {row['edges']} edges, rows binary "
            f"{row['rows_binary']} raw {row['rows_raw']}, rank {row['binary_rank']}"
        )
    print(f"    rows owed to the tag: binary {r['ctx_only_rows_binary']}, "
          f"raw {r['ctx_only_rows_raw']}")  # fmt: skip


def _say_audit(r: dict) -> None:
    print(f"\n[18] score audit ({r['seeds']} seeds, rank {r['rank']}; screens, never promotes)")
    for group in ("baselines", "candidates"):
        for name, rows in r[group].items():
            for outcome, v in rows.items():
                tag = "  DEAD" if v.get("dead") else ""
                print(
                    f"    {group[:-1]:<9} {name:<17} vs {outcome:<11} rho {v['rho']:+.3f}  "
                    f"|total {v['partial_total']:+.3f}  |degree {v['partial_degree']:+.3f}{tag}"
                )


_SECTIONS = (
    ("flaky", _say_flaky),
    ("subsumption", _say_subsumption),
    ("admission", _say_admission),
    ("rarefaction", _say_rarefaction),
    ("bootstrap", _say_bootstrap),
    ("length_confound", _say_length),
    ("prefix_fold", _say_prefix),
    ("score_audit", _say_audit),
)


def print_sections(result: dict) -> None:
    """Print every section present in *result*; unavailable ones say why."""
    for key, say in _SECTIONS:
        section = result.get(key)
        if section is None:
            continue
        if section.get("available") is False:
            print(f"\n[{key}] unavailable: {section['reason']}")
            continue
        say(section)
