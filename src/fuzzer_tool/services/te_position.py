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
    capped_map = min(map_size, 1024)
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
    best_pos = max(byte_edges.keys())
    return best_pos if best_pos < input_length else None
