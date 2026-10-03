"""Mutual information between operator/position and the edge set (handover 7.2).

``te_position.update_te_causal_map`` collapses each execution's coverage to
``max(edge_set)``. This module keeps the whole vector and provides:

* ``mutual_information``: plug-in ``I(X;Y)`` in bits over discrete pairs.
* ``permutation_null``: shuffle Y against X, return the null mean/std and a
  z-score. The analytic TE bias correction was evaluated and rejected
  (residual 1.7-3.4 bits), so bias is handled by the null, as ``seed_entropy_kl``
  does, never by an analytic term.
* ``edge_presence_mi``: ``I(X ; edge e present)`` for every edge over a list of
  ``(x, edge_set)`` observations (X = byte-position bin or operator id).
* ``channel_capacity``: Blahut-Arimoto capacity of ``p(y|x)`` and the capacity
  achieving input distribution (usable as operator weights).
* ``operator_new_edge_capacity``: the per-operator ``new_edge`` channel.

Pure functions, stdlib only, deterministic given ``seed``. Not wired into any
scheduler; calibrate the null on a real clang build before doing so.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence


def mutual_information(xs: Sequence[Hashable], ys: Sequence[Hashable]) -> float:
    """Plug-in ``I(X;Y)`` in bits. Biased upward for small samples."""
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must have equal length")
    if n == 0:
        return 0.0
    cx, cy, cxy = Counter(xs), Counter(ys), Counter(zip(xs, ys))
    mi = 0.0
    for (x, y), c in cxy.items():
        mi += (c / n) * math.log2(c * n / (cx[x] * cy[y]))
    return max(mi, 0.0)


def permutation_null(
    xs: Sequence[Hashable],
    ys: Sequence[Hashable],
    n_perm: int = 200,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Return ``(mi, null_mean, null_std, z)`` with Y shuffled against X."""
    mi = mutual_information(xs, ys)
    rng = random.Random(seed)
    pool = list(ys)
    nulls = []
    for _ in range(max(n_perm, 1)):
        rng.shuffle(pool)
        nulls.append(mutual_information(xs, pool))
    mean = sum(nulls) / len(nulls)
    var = sum((v - mean) ** 2 for v in nulls) / max(len(nulls) - 1, 1)
    std = math.sqrt(var)
    z = (mi - mean) / std if std > 0 else (math.inf if mi > mean else 0.0)
    return mi, mean, std, z


def edge_presence_mi(
    observations: Iterable[tuple[Hashable, Iterable[int]]],
    n_perm: int = 100,
    seed: int = 0,
    min_support: int = 5,
    top: int | None = None,
) -> list[tuple[int, float, float]]:
    """Per-edge ``I(X ; present)``; returns ``[(edge, mi_bits, z)]`` by z desc.

    Edges present in fewer than ``min_support`` executions (or absent from
    fewer than ``min_support``) are skipped: their null is degenerate.
    """
    obs = [(x, frozenset(es)) for x, es in observations]
    n = len(obs)
    xs = [x for x, _ in obs]
    support: Counter[int] = Counter()
    for _, es in obs:
        support.update(es)
    out = []
    for e, k in support.items():
        if k < min_support or n - k < min_support:
            continue
        ys = [e in es for _, es in obs]
        mi, _, _, z = permutation_null(xs, ys, n_perm, seed)
        out.append((e, mi, z))
    out.sort(key=lambda t: (-t[2], -t[1], t[0]))
    return out[:top] if top else out


def channel_capacity(
    p_y_given_x: Sequence[Sequence[float]],
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> tuple[float, list[float]]:
    """Blahut-Arimoto. Rows are ``p(y|x)``; returns ``(capacity_bits, q(x))``."""
    m = len(p_y_given_x)
    if m == 0:
        return 0.0, []
    k = len(p_y_given_x[0])
    for row in p_y_given_x:
        if len(row) != k or abs(sum(row) - 1.0) > 1e-6 or min(row) < 0:
            raise ValueError("each row must be a probability distribution")
    q = [1.0 / m] * m
    cap = 0.0
    for _ in range(max_iter):
        py = [sum(q[x] * p_y_given_x[x][y] for x in range(m)) for y in range(k)]
        d = []
        for x in range(m):
            d.append(
                sum(
                    p * math.log2(p / py[y])
                    for y, p in enumerate(p_y_given_x[x])
                    if p > 0
                )
            )
        lo = sum(q[x] * d[x] for x in range(m))
        hi = max(d)
        cap = lo
        if hi - lo < tol:
            break
        w = [q[x] * 2.0 ** d[x] for x in range(m)]
        s = sum(w)
        q = [v / s for v in w]
    return max(cap, 0.0), q


def operator_new_edge_capacity(
    records: Iterable[tuple[Hashable, bool]],
    alpha: float = 0.5,
) -> tuple[float, dict[Hashable, float]]:
    """Capacity of the ``op -> new_edge`` channel and the capacity-achieving mix.

    Caution: the achieving mix maximises ``I(op; new_edge)``, so it favours the
    most *distinguishable* operators, including a near-deterministic "never
    hits" one. It is NOT a productivity ranking and must not be used directly
    as scheduler weights; use the capacity as an upper bound on what operator
    choice can tell you, and rank by hit rate (or Good-Turing M0) instead.

    ``alpha`` is an additive (Jeffreys) smoothing so an operator with zero
    hits is not treated as a noiseless "never".
    """
    hits: Counter[Hashable] = Counter()
    tot: Counter[Hashable] = Counter()
    for op, new in records:
        tot[op] += 1
        hits[op] += bool(new)
    ops = sorted(tot, key=repr)
    if not ops:
        return 0.0, {}
    rows = []
    for op in ops:
        p = (hits[op] + alpha) / (tot[op] + 2 * alpha)
        rows.append([1.0 - p, p])
    cap, q = channel_capacity(rows)
    return cap, dict(zip(ops, q))
