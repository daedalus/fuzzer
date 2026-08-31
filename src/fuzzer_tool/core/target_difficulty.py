"""Static pre-fuzz difficulty estimation via the isoperimetric function.

Percolation handover Module 3 (docs/handover/handover_percolation_theory_2026-08-31.md
§4), revised per Diskin, Easo, Radhakrishnan, Sudakov, Tassion, "Supercritical
sharpness of percolation" (arXiv:2603.03257): supercritical cluster-size decay
is governed by the isoperimetric function

    Φ(n) = min{|∂S| : S subset of V, n <= |S| < ∞}

on ANY infinite transitive graph — no assumption about degree distribution
or clustering is needed. A CFG-derived coverage graph is finite and not
transitive, so this module treats Φ as a heuristic difficulty signal, not a
literal application of the theorem (see handover §9.1).

Exact Φ(n) is a min-cut-style computation over all size-n subsets and is
intractable in general; ``estimate_isoperimetric_profile`` approximates it
with a greedy boundary-growth heuristic, giving an upper bound (a witness
cutset), not a proof of optimality.
"""

from __future__ import annotations

from typing import Union

from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.core.rand_pool import RandPool

_DEFAULT_RESTARTS = 4

# A caller passes either a real ICFG or a plain adjacency dict (tests, or any
# graph reconstructed without a live binary).
Graph = Union[InterproceduralCFG, "dict[int, set[int]]"]


def _adjacency_from_icfg(icfg: InterproceduralCFG) -> dict[int, set[int]]:
    """Undirected adjacency from an InterproceduralCFG (core/icfg.py).

    Φ is defined over undirected edge boundaries (cut size); a caller→callee
    edge and a branch edge both count as one adjacency link regardless of
    direction.
    """
    adj: dict[int, set[int]] = {i: set() for i in range(icfg.n_nodes)}
    for u, v in zip(icfg.src.tolist(), icfg.dst.tolist(), strict=True):
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _as_adjacency(graph: Graph) -> dict[int, set[int]]:
    """Accept either a plain adjacency dict or an InterproceduralCFG."""
    if isinstance(graph, dict):
        return graph
    return _adjacency_from_icfg(graph)


def _greedy_grow_sequence(adj: dict[int, set[int]], seed: int, max_n: int) -> list[int]:
    """Grow {seed} up to size max_n, recording |∂S| at every size along the way.

    Boundary size is tracked incrementally: adding vertex v to S changes
    the boundary by (neighbors of v outside S) - (neighbors of v inside
    S), so the full boundary never needs recomputing from scratch.

    Returns:
        ``[boundary_at_size_1, boundary_at_size_2, ...]``, stopping early
        (shorter than max_n) when the component containing seed is
        exhausted — the caller must check the sequence length before
        trusting an entry past it as reachable at all.
    """
    s = {seed}
    boundary = len(adj[seed])
    frontier = set(adj[seed])
    seq = [boundary]

    while len(s) < max_n and frontier:
        best_v: int | None = None
        best_delta: int | None = None
        for v in frontier:
            in_s = sum(1 for u in adj[v] if u in s)
            delta = len(adj[v]) - 2 * in_s
            if best_delta is None or delta < best_delta:
                best_delta, best_v = delta, v
        assert best_v is not None and best_delta is not None  # frontier was non-empty
        s.add(best_v)
        boundary += best_delta
        frontier.discard(best_v)
        frontier |= adj[best_v] - s
        seq.append(boundary)

    return seq


def estimate_isoperimetric_profile(
    graph: Graph,
    sizes: list[int],
    restarts: int = _DEFAULT_RESTARTS,
    seed: int | None = None,
) -> dict[int, int]:
    """Approximate Φ(n) for each requested cluster size n.

    Args:
        graph: an InterproceduralCFG (core/icfg.py), or a plain undirected
            adjacency dict ``{node: set(neighbors)}`` for testing without a
            real binary.
        sizes: cluster sizes n to estimate Φ(n) for.
        restarts: number of independent greedy-growth probes; each probe
            grows once up to the largest requested size, so the best
            (smallest) boundary found across restarts is kept, since each
            probe only gives an upper bound on the true minimum.
        seed: RNG seed for restart-node selection (reproducible profiles).

    Returns:
        ``{n: approx Φ(n)}``, restricted to n values reachable within some
        connected component of the graph — Φ is undefined (infinite) for n
        larger than every component, so those sizes are omitted rather than
        silently reported as some finite number.

    A witness set of size k is also a valid (if slack) witness for any
    n <= k, since Φ(n) minimizes over |S| >= n, not |S| == n exactly. Each
    restart's growth sequence is therefore reduced to a suffix-min before
    merging across restarts, which is what keeps the returned profile
    non-decreasing in n — a plain per-size independent probe (one greedy
    run targeting exactly n, discarding the path once n is reached) can
    otherwise report a smaller boundary at a larger n than at a smaller
    one, since the two sizes may take different greedy paths from the same
    seed and the larger path can wander through a better cut the shorter
    one hadn't reached yet.
    """
    adj = _as_adjacency(graph)
    if not adj:
        return {}

    requested = sorted(n for n in sizes if n >= 1)
    if not requested:
        return {}
    max_n = requested[-1]

    nodes = list(adj)
    pool = RandPool(seed=seed)
    restart_nodes = pool.sample(nodes, min(restarts, len(nodes)))

    best: dict[int, int] = {}
    for start in restart_nodes:
        seq = _greedy_grow_sequence(adj, start, max_n)
        running_min = None
        for k in range(len(seq), 0, -1):
            running_min = seq[k - 1] if running_min is None else min(running_min, seq[k - 1])
            prev = best.get(k)
            best[k] = running_min if prev is None else min(prev, running_min)

    return {n: best[n] for n in requested if n in best}


def estimate_percolation_threshold(graph: Graph) -> dict[str, float | int]:
    """Cheap fallback difficulty estimate: p_c ~= 1/<k> (Erdos-Renyi formula).

    Kept as a fallback for targets where a full Φ profile isn't affordable
    (huge binary, no time budget for the greedy probes). Prefer
    ``estimate_isoperimetric_profile`` when it fits the budget — see
    handover §4 for why the degree-only estimate can mislead on CFGs with
    heavy-tailed branching (a single dispatch switch or checksum gate).
    """
    adj = _as_adjacency(graph)
    if not adj:
        return {"avg_degree": 0.0, "p_c_estimate": float("inf"), "n_nodes": 0}

    degrees = [len(neighbors) for neighbors in adj.values()]
    avg_degree = sum(degrees) / len(degrees)
    p_c = 1.0 / avg_degree if avg_degree > 0 else float("inf")
    return {"avg_degree": avg_degree, "p_c_estimate": p_c, "n_nodes": len(adj)}


def _phi_at(phi_profile_sorted_keys: list[int], phi_profile: dict[int, int], v: float) -> float:
    """Nearest known Φ(n) at or below v (Φ is non-decreasing in n)."""
    lo = None
    for sz in phi_profile_sorted_keys:
        if sz <= v:
            lo = sz
        else:
            break
    key = lo if lo is not None else phi_profile_sorted_keys[0]
    return float(phi_profile[key])


def estimate_growth_curve(
    phi_profile: dict[int, int],
    initial_size: int,
    max_steps: int,
    c: float = 1.0,
) -> list[float]:
    """Estimate v_n — reachable-neighborhood size after n steps (Theorem 3,
    arXiv:2603.03257), by Euler-integrating dv/dn = c * Φ(v).

    Theorem 3 defines v_n implicitly via the integral relation
    ``integral from |S| to v_n of 1/(c * Phi(t)) dt = n``, equivalent to the
    ODE ``dv/dn = c * Phi(v)``. This is a heuristic prioritization signal,
    not a formal bound: Φ here is only the greedy upper-bound approximation
    from ``estimate_isoperimetric_profile``, the step below is a first-order
    Euler approximation, and ``c`` is not derived from the graph — comparing
    growth curves across targets is only meaningful at a fixed c.

    Args:
        phi_profile: ``{n: approx Φ(n)}`` from ``estimate_isoperimetric_profile``.
        initial_size: |S|, the starting cluster size (v_0).
        max_steps: number of discrete steps n to simulate.
        c: the constant from the paper's Theorem 1/3.

    Returns:
        ``[v_0, v_1, ..., v_max_steps]`` (length ``max_steps + 1``).
    """
    v = float(initial_size)
    if max_steps <= 0:
        return [v]
    if not phi_profile:
        return [v] * (max_steps + 1)

    sorted_keys = sorted(phi_profile)
    curve = [v]
    for _ in range(max_steps):
        step = c * _phi_at(sorted_keys, phi_profile, v)
        v += max(step, 1e-9)  # Φ >= 0 in principle; guard against stalling
        curve.append(v)
    return curve
