"""Transfer entropy position selection for mutation targeting."""

from __future__ import annotations


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
