"""Transfer entropy position selection for mutation targeting."""

from __future__ import annotations

from fuzzer_tool.core.circular_stats import (
    DEFAULT_ALPHA,
    MIN_STRIDE,
    PhaseConcentration,
    concentration,
)


def update_te_causal_map(
    te,
    input_history: list[bytes],
    edge_history: list[set[int]],
    map_size: int,
    byte_edges: dict[int, dict[int, int]],
) -> None:
    """Update byte→edge causal map using transfer entropy.

    Mutates ``byte_edges`` in place.
    """
    if not te or len(input_history) < 10:
        return
    max_pos = min(64, min(len(b) for b in input_history))
    for pos in range(max_pos):
        source = [b[pos] if pos < len(b) else 0 for b in input_history]
        target = []
        for edge_set in edge_history:
            if edge_set:
                target.append(max(edge_set))
            else:
                target.append(0)
        te_val = te.transfer_entropy(source, target)
        if te_val > 0.01:
            edge_counts: dict[int, int] = {}
            for edge_set in edge_history[-50:]:
                for eid in edge_set:
                    edge_counts[eid] = edge_counts.get(eid, 0) + 1
            if edge_counts:
                byte_edges[pos] = edge_counts


def get_te_weighted_position(
    byte_edges: dict[int, dict[int, int]],
    input_length: int,
) -> int | None:
    """Get a byte position weighted by transfer entropy causal influence.

    Returns position with highest TE to coverage, or None if no TE data.
    """
    if not byte_edges:
        return None
    best_pos = max(byte_edges, key=lambda pos: sum(byte_edges[pos].values()))
    return best_pos if best_pos < input_length else None


def phase_lock(
    byte_edges: dict[int, dict[int, int]],
    stride: int | None,
) -> PhaseConcentration | None:
    """Test the causal byte map for a phase lock on a *stride*-byte record.

    The expensive half of :func:`get_phase_weighted_position`, split out so
    callers on the mutation hot path can memoise it: the causal map only
    changes on the TE observation cadence, while the position is drawn on
    every mutation.

    Returns ``None`` when the question is not askable — no causal data, or
    no usable stride. A returned concentration still has to clear its
    significance gate; that is :func:`draw_phase_position`'s job.
    """
    if not byte_edges or not stride or stride < MIN_STRIDE:
        return None

    offsets = list(byte_edges)
    weights = [float(sum(byte_edges[pos].values())) for pos in offsets]

    return concentration(offsets, weights, stride)


def draw_phase_position(
    lock: PhaseConcentration | None,
    input_length: int,
    rng,
    alpha: float = DEFAULT_ALPHA,
) -> int | None:
    """Pick a byte position at the locked field of a uniformly drawn record.

    Returns ``None`` when *lock* is absent, fails its significance test at
    *alpha*, or names an offset no record slot can reach inside
    ``input_length``. The RNG is consulted only on the accepted path.
    """
    if lock is None or not lock.is_locked(alpha):
        return None

    # Slots at the locked offset that fit: offset, offset+stride, ...
    n_slots = (input_length - lock.offset + lock.stride - 1) // lock.stride
    if n_slots <= 0:
        return None

    return rng.randint(0, n_slots - 1) * lock.stride + lock.offset


def get_phase_weighted_position(
    byte_edges: dict[int, dict[int, int]],
    input_length: int,
    stride: int | None,
    rng,
    alpha: float = DEFAULT_ALPHA,
) -> int | None:
    """Extrapolate the causal byte map across a record-structured buffer.

    ``update_te_causal_map`` only ever observes offsets below its own
    64-byte cap, and :func:`get_te_weighted_position` can therefore never
    name a position outside it. When the seed has an inferred
    ``record_stride`` and the causal offsets are phase-locked on it — tested
    at *alpha* by :func:`core.circular_stats.concentration` — the same field
    recurs at ``offset + k*stride`` for every record in the buffer, so a
    uniformly drawn record extends the evidence to the full length.

    Returns ``None`` whenever the extrapolation is unwarranted: no stride,
    no causal data, a rejected lock, or a buffer with no slot at the locked
    offset. The RNG is consulted only on the accepted path.
    """
    return draw_phase_position(phase_lock(byte_edges, stride), input_length, rng, alpha)


def edge_sets_to_flow(
    te,
    edge_history: list[set[int]],
    top_k: int = 10,
) -> dict[tuple[int, int], float]:
    """Directed edge-to-edge transfer entropy from a history of edge sets.

    Same binary-series construction as ``TransferEntropy.edge_to_edge_flow``,
    adapted to the sparse ``list[set[int]]`` representation already
    accumulated in ``Fuzzer._te_edge_history`` (SHM edge ids), so callers
    never need to materialise a per-run dense bitmap just to feed the
    causal-sector graph. The top-k ranking differs, though: ``edge_history``
    carries no per-exec hit counts, only per-step presence, so ``total_hits``
    here counts the number of steps each edge appears in (at most 1 per
    step) rather than summed hit counts as in ``edge_to_edge_flow``.
    """
    if not te or len(edge_history) < 3:
        return {}

    total_hits: dict[int, int] = {}
    for edges in edge_history:
        for eid in edges:
            total_hits[eid] = total_hits.get(eid, 0) + 1
    top_edges = sorted(total_hits, key=lambda e: total_hits[e], reverse=True)[:top_k]
    if not top_edges:
        return {}

    edge_series = {e: [1 if e in edges else 0 for edges in edge_history] for e in top_edges}

    flow: dict[tuple[int, int], float] = {}
    for src in top_edges:
        for tgt in top_edges:
            if src == tgt:
                continue
            te_val = te.transfer_entropy(edge_series[src], edge_series[tgt])
            if te_val > 0:
                flow[(src, tgt)] = te_val
    return flow
