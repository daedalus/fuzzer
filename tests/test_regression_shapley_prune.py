"""Regression for P0-5: shapley prune discarded the numerically low half of edge ids."""

import pytest

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


def test_edge_view_tracks_records_and_prunes():
    """The edge-major view is cached, so record() and prune must invalidate it."""
    s = ShapleyAttribution(n_samples=50, window_size=100)
    s.record({"a", "b"}, 2, {1, 2})
    assert s._edge_op_count == {1: {"a": 1, "b": 1}, 2: {"a": 1, "b": 1}}
    s.record({"a"}, 1, {2, 3})
    assert s._edge_op_count[2] == {"a": 2, "b": 1}
    assert s._edge_op_count[3] == {"a": 1}

    for i in range(SHAPLEY_EDGES_MAX):
        s.record({"c"}, 1, {100 + i})
    assert len(s._all_edges) <= SHAPLEY_EDGES_MAX
    view = s._edge_op_count
    assert set(view) == set(s._edge_total)
    assert all(
        op in s._operator_edges and e in s._operator_edges[op] for e, d in view.items() for op in d
    )


def test_shapley_values_match_per_edge_formula_after_prune():
    """Vectorised credit equals the per-edge definition on the pruned state."""
    s = ShapleyAttribution(n_samples=50, window_size=500)
    ops = [f"op{i}" for i in range(6)]
    for i in range(SHAPLEY_EDGES_MAX // 3):
        edges = {(i * 5 + k) % 20000 for k in range(6)}
        s.record({ops[i % 6], ops[(i + 2) % 6]}, len(edges), edges)

    expected = dict.fromkeys(ops, 0.0)
    for per_op in s._edge_op_count.values():
        total, k = sum(per_op.values()), len(per_op)
        for op, c in per_op.items():
            expected[op] += c / total / k
    norm = sum(expected.values())
    got = s.shapley_values(ops)
    assert got == {op: pytest.approx(v / norm, abs=1e-12) for op, v in expected.items()}


def test_operator_edges_is_a_live_view_of_op_edge_count():
    """One store: _operator_edges mirrors _op_edge_count, never creates keys on read."""
    s = ShapleyAttribution()
    s.record({"a", "b"}, 2, {1, 2})
    assert set(s._operator_edges["a"]) == {1, 2}
    assert s._operator_edges.get("zzz", set()) == set()
    assert "zzz" not in s._operator_edges and "zzz" not in s._op_edge_count
    s.record({"a"}, 1, {3})
    assert s._operator_edges["a"] == {1, 2, 3}
    with pytest.raises(KeyError):
        s._operator_edges["zzz"]


def _expected_kept(totals: dict[int, int]) -> set[int]:
    """Independent prune oracle: drop the rarest half by (total, id)."""
    ranked = sorted(totals, key=lambda e: (totals[e], e))
    return set(ranked[len(ranked) // 2 :])


def test_regression_all_edges_is_a_view_of_edge_total():
    """One store: _all_edges is the key view of _edge_total, not a second set updated per exec."""
    s = ShapleyAttribution()
    s.record({"a"}, 2, {1, 2})
    assert isinstance(s._all_edges, type({}.keys()))
    s.record({"b"}, 1, {3})
    assert s._all_edges == {1, 2, 3} == s._edge_total.keys()


def test_falsify_prune_keeps_oracle_half():
    """Kept edges equal the oracle's; a wrong tie-break or direction would differ."""
    s = ShapleyAttribution()
    hot = set(range(SHAPLEY_EDGES_MAX // 2))
    s.record({"a"}, len(hot), hot)  # every hot edge seen twice, the rest once
    s.record({"b"}, len(hot), hot)
    totals = dict(s._edge_total)
    cold = set(range(-(SHAPLEY_EDGES_MAX // 2) - 1, 0))  # pushes past the cap
    totals.update(dict.fromkeys(cold, 1))
    s.record({"c"}, len(cold), cold)
    assert s.prune_count == 1
    assert set(s._all_edges) == _expected_kept(totals) == set(s._edge_total)


def test_adversarial_prune_all_ties_break_by_id():
    """All totals equal: the numerically lower half (negative ids included) goes."""
    s = ShapleyAttribution()
    edges = set(range(-SHAPLEY_EDGES_MAX // 2, SHAPLEY_EDGES_MAX // 2 + 1))
    s.record({"a"}, len(edges), edges)
    assert s.prune_count == 1
    assert set(s._all_edges) == _expected_kept(dict.fromkeys(edges, 1))
    assert set(s._operator_edges["a"]) == set(s._all_edges)
