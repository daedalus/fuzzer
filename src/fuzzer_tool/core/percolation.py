"""Percolation primitives for coverage-guided fuzzing.

Central home for percolation-framed types and algorithms:

- ``CoverageRegime`` enum — phase labels (subcritical / critical / supercritical)
  used by both the coverage regime detector and the bootstrap minimizer.
- ``bootstrap_minimize_corpus()`` — iterative k-rigid-core reduction that
  captures transitive redundancy single-pass greedy set-cover misses.
- ``estimate_time_to_next_discovery()`` — first-passage estimate of executions
  until the next coverage milestone (Module 5 / P4-8). Inverts the integral
  form of Diskin et al. (arXiv:2603.03257) Thm 3 using a Φ profile from
  ``target_difficulty.estimate_isoperimetric_profile``. Pure heuristic;
  production wiring of Φ remains gated on the P2-1 decision.
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


# Regime multipliers applied after the integral estimate. Subcritical
# discovery is exponentially rare; supercritical compounds. These are
# order-of-magnitude priors, not fitted constants — falsify against
# inter-discovery CV before trusting absolute budgets (handover §Open
# falsifier checks).
_REGIME_TIME_SCALE: dict[CoverageRegime, float] = {
    CoverageRegime.SUBCRITICAL: 8.0,
    CoverageRegime.CRITICAL: 1.5,
    CoverageRegime.SUPERCRITICAL: 0.35,
}

# Fallback when no Φ profile is supplied: executions-per-new-edge prior
# by regime (used only for the pure regime path).
_REGIME_FALLBACK_EXEC_PER_EDGE: dict[CoverageRegime, float] = {
    CoverageRegime.SUBCRITICAL: 5000.0,
    CoverageRegime.CRITICAL: 400.0,
    CoverageRegime.SUPERCRITICAL: 40.0,
}


def _phi_at_profile(
    sorted_keys: list[int], phi_profile: dict[int, int], v: float
) -> float:
    """Nearest known Φ(n) at or below *v* (Φ is non-decreasing in n).

    Mirrors ``target_difficulty._phi_at`` so percolation.py does not import
    the diagnostic-only module at call time.
    """
    lo = None
    for sz in sorted_keys:
        if sz <= v:
            lo = sz
        else:
            break
    key = lo if lo is not None else sorted_keys[0]
    return float(phi_profile[key])


def _success_rate_from_operator_stats(operator_stats) -> float:
    """Extract a success fraction in (0, 1] from heterogeneous operator stats.

    Accepts:
    - ``None`` → 1.0 (no penalty)
    - a float already in (0, 1] → itself
    - a mapping with ``success_rate`` / ``rate`` / ``p_success``
    - a mapping of ``{op_name: {successes, trials} | float}`` → pooled rate
    - an object with ``.success_rate`` attribute
    """
    if operator_stats is None:
        return 1.0
    if isinstance(operator_stats, (int, float)):
        r = float(operator_stats)
        return r if 0.0 < r <= 1.0 else 1.0
    if hasattr(operator_stats, "success_rate"):
        r = float(operator_stats.success_rate)
        return r if 0.0 < r <= 1.0 else 1.0
    if isinstance(operator_stats, dict):
        for key in ("success_rate", "rate", "p_success"):
            if key in operator_stats:
                r = float(operator_stats[key])
                return r if 0.0 < r <= 1.0 else 1.0
        # Pooled {op: stats}
        succ = trials = 0.0
        for v in operator_stats.values():
            if isinstance(v, dict):
                s = float(v.get("successes", v.get("success", 0)))
                t = float(v.get("trials", v.get("pulls", v.get("n", 0))))
                succ += s
                trials += t
            elif isinstance(v, (int, float)):
                # treat as a rate contribution with unit weight
                r = float(v)
                if 0.0 < r <= 1.0:
                    succ += r
                    trials += 1.0
        if trials > 0:
            r = succ / trials
            return r if 0.0 < r <= 1.0 else 1.0
    return 1.0


def estimate_time_to_next_discovery(
    edge_tracker,
    operator_stats=None,
    coverage_regime: CoverageRegime | None = None,
    *,
    phi_profile: dict[int, int] | None = None,
    target_delta: int = 1,
    target_size: int | None = None,
    c: float = 1.0,
    max_n: float = 1_000_000.0,
) -> float:
    """First-passage estimate: executions until the next coverage milestone.

    Implements percolation handover Module 5 (P4-8). Inverts the integral
    form of Diskin et al. (arXiv:2603.03257) Theorem 3:

        ∫_{|S| → v_n} 1 / (c · Φ(t)) dt = n

    which is the continuous inverse of the forward Euler growth curve in
    ``target_difficulty.estimate_growth_curve``.

    This is a **heuristic prioritization / budgeting signal**, not a formal
    bound:

    - Φ is the greedy upper-bound approximation from
      ``estimate_isoperimetric_profile`` (finite, non-transitive coverage
      graphs are outside the theorem's hypotheses).
    - ``c`` is not derived from the graph.
    - Regime and operator-rate scales are order-of-magnitude priors.

    Production consumers of Φ remain gated on the P2-1 decision
    (``target_difficulty`` is still diagnostic-only). Callers that already
    hold a profile (offline reports, future multi-target budgeters) pass
    it explicitly via ``phi_profile``.

    Args:
        edge_tracker: object with ``cumulative_edges`` (a set/collection of
            currently covered edge ids). Typically ``EdgeTracker``.
        operator_stats: optional success-rate source (float, dict, or object
            with ``.success_rate``). Scales the estimate by 1/rate so sparse
            operators stretch the budget.
        coverage_regime: current ``CoverageRegime``. Defaults to CRITICAL
            when omitted. Multiplies the integral result by a regime scale.
        phi_profile: ``{n: approx Φ(n)}`` from
            ``target_difficulty.estimate_isoperimetric_profile``. When
            ``None``, falls back to a pure regime-based prior
            (``_REGIME_FALLBACK_EXEC_PER_EDGE``).
        target_delta: number of additional edges to aim for when
            ``target_size`` is not given. Default 1 ("next discovery").
        target_size: absolute coverage size to reach. Overrides
            ``target_delta`` when provided.
        c: constant from the paper's Theorem 1/3 (same role as in
            ``estimate_growth_curve``).
        max_n: hard ceiling on the returned estimate (guards against
            near-zero Φ stalls).

    Returns:
        Estimated number of executions (float ≥ 0). ``0.0`` when the
        milestone is already reached; ``max_n`` when the integral diverges
        or the ceiling is hit.
    """
    if coverage_regime is None:
        coverage_regime = CoverageRegime.CRITICAL

    try:
        current = len(getattr(edge_tracker, "cumulative_edges", ()) or ())
    except TypeError:
        current = 0

    if target_size is not None:
        goal = int(target_size)
    else:
        goal = current + max(1, int(target_delta))

    if goal <= current:
        return 0.0

    rate = _success_rate_from_operator_stats(operator_stats)
    # 1/rate stretches time when operators rarely produce new coverage.
    rate_scale = 1.0 / rate

    regime_scale = _REGIME_TIME_SCALE.get(coverage_regime, 1.5)

    if not phi_profile:
        # Pure regime prior: no geometric information available.
        per_edge = _REGIME_FALLBACK_EXEC_PER_EDGE.get(coverage_regime, 400.0)
        estimate = (goal - current) * per_edge * rate_scale
        return float(min(max(estimate, 0.0), max_n))

    sorted_keys = sorted(phi_profile)
    if not sorted_keys:
        per_edge = _REGIME_FALLBACK_EXEC_PER_EDGE.get(coverage_regime, 400.0)
        estimate = (goal - current) * per_edge * rate_scale
        return float(min(max(estimate, 0.0), max_n))

    # Trapezoidal / stepwise integration of 1/(c·Φ(t)) from current → goal.
    # Step in unit coverage; Φ is only known at discrete profile keys, so
    # we evaluate Φ at the left endpoint of each unit interval (consistent
    # with the forward Euler in estimate_growth_curve).
    n_acc = 0.0
    v = float(current)
    # Guard: if Φ is zero/tiny the integral diverges → return ceiling.
    min_phi = 1e-12
    while v < goal:
        phi = _phi_at_profile(sorted_keys, phi_profile, v)
        denom = c * max(phi, min_phi)
        # dv = 1 edge of progress; dn = dv / (c·Φ)
        dn = 1.0 / denom
        n_acc += dn
        v += 1.0
        if n_acc * regime_scale * rate_scale >= max_n:
            return float(max_n)

    estimate = n_acc * regime_scale * rate_scale
    return float(min(max(estimate, 0.0), max_n))


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
    edges_of, owners = _edge_owners(corpus, edge_tracker.seed_edges)

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
        _release_edges(idx, edges_of[idx], owners, unique)

    kept = [corpus[i] for i in range(n) if alive[i]]
    removed = [corpus[i] for i in removed_idx]
    return kept, removed


def _edge_owners(
    corpus: list[bytes], seed_edges: dict
) -> tuple[list[frozenset | set], dict[int, set[int]]]:
    """Per-seed edge sets and, per edge, the indices of seeds covering it."""
    edges_of: list[frozenset | set] = []
    owners: dict[int, set[int]] = {}
    for idx, seed in enumerate(corpus):
        edges = seed_edges.get(_seed_key(seed), set())
        edges_of.append(edges)
        for e in edges:
            owners.setdefault(e, set()).add(idx)
    return edges_of, owners


def _release_edges(idx: int, edges, owners: dict[int, set[int]], unique: list[int]) -> None:
    """Drop seed *idx* as owner; a now-sole owner gains a unique edge."""
    for e in edges:
        own = owners[e]
        own.discard(idx)
        if len(own) == 1:
            (sole,) = own
            unique[sole] += 1


def _seed_key(seed: bytes) -> str:
    """Hash a seed to a 16-char hex string for EdgeTracker lookup."""
    import xxhash

    return xxhash.xxh64(seed).hexdigest()[:16]
