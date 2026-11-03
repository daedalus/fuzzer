"""Regression for P0-5: shapley prune discarded the numerically low half of edge ids."""

from fuzzer_tool.core.shapley import SHAPLEY_EDGES_MAX, ShapleyAttribution


def test_prune_does_not_bias_by_edge_id_half():
    """Two equally productive operators in disjoint id ranges must keep similar credit.

    Pre-fix, _prune_edges sorted by edge id and dropped the low half, so an
    operator whose edges lived in [0, 32768) was systematically under-credited
    relative to one in [32768, 65536) despite identical productivity.
    """
    s = ShapleyAttribution(n_samples=50, window_size=1000)
    low_op, high_op = "op_low", "op_high"
    # Flood past SHAPLEY_EDGES_MAX with equal productivity, disjoint ranges.
    n_rounds = (SHAPLEY_EDGES_MAX // 4) + 500
    for i in range(n_rounds):
        low_edges = {i % 30000, (i + 1) % 30000, (i + 2) % 30000, (i + 3) % 30000}
        high_edges = {
            32768 + (i % 30000),
            32768 + ((i + 1) % 30000),
            32768 + ((i + 2) % 30000),
            32768 + ((i + 3) % 30000),
        }
        s.record([low_op], new_edges=len(low_edges), edge_indices=low_edges)
        s.record([high_op], new_edges=len(high_edges), edge_indices=high_edges)

    values = s.shapley_values()
    assert low_op in values and high_op in values
    lo, hi = values[low_op], values[high_op]
    # Within noise of equality (was 5.3× under the id-half prune).
    ratio = max(lo, hi) / max(min(lo, hi), 1e-12)
    assert ratio < 2.0, f"credit distortion {ratio:.2f}x: low={lo:.4f} high={hi:.4f}"


def test_prune_keeps_operator_sets_consistent_with_edge_index():
    """Pruning via the edge->operator index must leave no dropped edge in any operator set."""
    s = ShapleyAttribution(n_samples=50, window_size=1000)
    ops = [f"op{i}" for i in range(20)]
    for i in range(SHAPLEY_EDGES_MAX // 2):
        edges = {(i * 7 + k) % 40000 for k in range(4)}
        s.record({ops[i % 20], ops[(i * 3 + 1) % 20]}, len(edges), edges)
    assert len(s._all_edges) <= SHAPLEY_EDGES_MAX

    kept = s._all_edges
    for edges in s._operator_edges.values():
        assert edges <= kept
    assert set(s._edge_op_count) == set(s._edge_total) == kept
    held = {(op, e) for op, edges in s._operator_edges.items() for e in edges}
    indexed = {(op, e) for e, per_op in s._edge_op_count.items() for op in per_op}
    assert held == indexed
