"""Marginal contribution of operators to union edge coverage, and its report wiring."""

import types

import pytest

from fuzzer_tool.core.marginal_coverage import DIRECT_MAX_ID, marginal_table
from fuzzer_tool.core.shapley import ShapleyAttribution
from fuzzer_tool.services.report import _marginal_coverage


def _rows(sets):
    return {r.op: r for r in marginal_table(sets)}


def test_disjoint_sets_each_keep_everything():
    rows = _rows({"a": {1, 2, 3}, "b": {4, 5}})
    assert (rows["a"].gain, rows["a"].sole, rows["a"].shapley) == (3, 3, 3.0)
    assert (rows["b"].gain, rows["b"].sole, rows["b"].shapley) == (2, 2, 2.0)


def test_falsify_duplicates_are_not_ranked_equal():
    """Raw size ties a and b; marginal gain must not: the second copy adds nothing."""
    rows = marginal_table({"a": {1, 2, 3}, "b": {1, 2, 3}})
    assert [r.gain for r in rows] == [3, 0]
    assert all(r.sole == 0 for r in rows)
    assert all(r.shapley == 1.5 for r in rows)


def test_greedy_order_and_gain_over_union():
    rows = marginal_table({"big": set(range(10)), "sub": {1, 2}, "new": {50, 51, 52}})
    assert [(r.op, r.gain) for r in rows] == [("big", 10), ("new", 3), ("sub", 0)]


def test_invariants_on_overlapping_sets():
    sets = {"a": {1, 2, 3, 4}, "b": {3, 4, 5}, "c": {4, 5, 6, 7}, "d": set()}
    rows = marginal_table(sets)
    union = set().union(*sets.values())

    assert sum(r.gain for r in rows) == len(union)
    assert sum(r.shapley for r in rows) == pytest.approx(len(union))
    for r in rows:
        assert r.sole <= r.gain <= r.size
        assert r.sole <= r.shapley <= r.size


def test_sole_counts_only_unshared_edges():
    rows = _rows({"a": {1, 2, 3}, "b": {3, 4}})
    assert (rows["a"].sole, rows["b"].sole) == (2, 1)
    assert (rows["a"].shapley, rows["b"].shapley) == (2.5, 1.5)


def test_ties_break_by_name_deterministically():
    sets = {"z": {1}, "a": {2}, "m": {3}}
    assert [r.op for r in marginal_table(sets)] == ["a", "m", "z"]


@pytest.mark.parametrize("sets", [{}, {"a": set()}, {"a": set(), "b": set()}])
def test_adversarial_empty_inputs(sets):
    rows = marginal_table(sets)
    assert all(r.gain == r.sole == r.size == 0 and r.shapley == 0.0 for r in rows)
    assert len(rows) == len(sets)


def test_adversarial_sparse_and_negative_edge_ids():
    """Ids are hashes, not dense indices: huge and negative values must not blow up."""
    rows = _rows({"a": {2**62, -5}, "b": {-5, 7}})
    assert rows["a"].sole == 1 and rows["b"].sole == 1
    assert rows["a"].gain + rows["b"].gain == 3


def test_adversarial_input_not_mutated():
    sets = {"a": {1, 2}, "b": {2, 3}}
    marginal_table(sets)
    assert sets == {"a": {1, 2}, "b": {2, 3}}


def test_adversarial_many_operators_one_shared_edge():
    sets = {f"op{i}": {0} for i in range(200)}
    rows = marginal_table(sets)
    assert sum(r.gain for r in rows) == 1
    assert all(r.shapley == pytest.approx(1 / 200) for r in rows)


def test_shapley_attribution_exposes_table():
    sa = ShapleyAttribution()
    sa.record({"a"}, 2, {1, 2})
    sa.record({"b"}, 2, {2, 3})
    rows = {r.op: r for r in sa.marginal_coverage()}
    assert (rows["a"].sole, rows["b"].sole) == (1, 1)
    assert rows["a"].gain + rows["b"].gain == 3


def test_attribution_edge_sets_are_copies():
    sa = ShapleyAttribution()
    sa.record({"a"}, 1, {1})
    sa.operator_edge_sets()["a"].add(99)
    assert sa.operator_edge_sets()["a"] == {1}


def _fuzzer(sa):
    return types.SimpleNamespace(_use_shapley=sa is not None, _shapley=sa)


def test_report_lists_rows_and_zero_gain_operators():
    sa = ShapleyAttribution()
    sa.record({"a"}, 3, {1, 2, 3})
    sa.record({"b"}, 3, {1, 2, 3})
    out = _marginal_coverage(_fuzzer(sa))

    assert "Marginal Coverage" in out
    assert "union 3 edges" in out
    assert "zero gain" in out and "b" in out.split("zero gain", 1)[1]


@pytest.mark.parametrize("fz", [_fuzzer(None), _fuzzer(ShapleyAttribution())])
def test_report_empty_when_disabled_or_no_data(fz):
    assert _marginal_coverage(fz) == ""


def test_report_truncates_long_tables():
    sa = ShapleyAttribution()
    for i in range(40):
        sa.record({f"op{i:02d}"}, 1, {i})
    out = _marginal_coverage(_fuzzer(sa))
    assert "more" in out
    assert out.count("\n") < 30


def _naive(sets):
    """Reference: full-scan greedy and per-edge loops over plain sets."""
    left, covered, order = dict(sets), set(), []
    while left:
        op = min(left, key=lambda o: (-len(left[o] - covered), o))
        order.append((op, len(left[op] - covered)))
        covered |= left.pop(op)
    holders = {}
    for edges in sets.values():
        for e in edges:
            holders[e] = holders.get(e, 0) + 1
    return {
        op: (
            gain,
            sum(1 for e in sets[op] if holders[e] == 1),
            sum(1 / holders[e] for e in sets[op]),
        )
        for op, gain in order
    }, [op for op, _ in order]


def _skewed(n_ops):
    """Deterministic overlapping sets with many gain ties."""
    return {
        f"op{i:02d}": {(i * 7 + j * 13) % 211 for j in range(5 + (i * 3) % 40)}
        for i in range(n_ops)
    }


@pytest.mark.parametrize("n_ops", [2, 9, 40])
def test_matches_naive_reference(n_ops):
    sets = _skewed(n_ops)
    want, want_order = _naive(sets)
    got = marginal_table(sets)

    assert [r.op for r in got] == want_order
    for r in got:
        gain, sole, shapley = want[r.op]
        assert (r.gain, r.sole) == (gain, sole)
        assert r.shapley == pytest.approx(shapley)


def test_control_reference_equals_itself():
    """The oracle agrees with a second run of itself, so a mismatch above is the code."""
    sets = _skewed(40)
    assert _naive(sets) == _naive(sets)
    assert marginal_table(sets) == marginal_table(sets)


@pytest.mark.parametrize("shift", [0, DIRECT_MAX_ID - 100, DIRECT_MAX_ID - 1, DIRECT_MAX_ID, -300])
def test_dense_index_paths_agree(shift):
    """Direct-index and sort paths give the same table; ids straddle the limit."""
    base = _skewed(12)
    moved = {op: {e + shift for e in edges} for op, edges in base.items()}
    assert marginal_table(moved) == marginal_table(base)


def _brute_greedy(sets):
    left, covered, out = dict(sets), set(), []
    while left:
        op = min(left, key=lambda o: (-len(left[o] - covered), o))
        out.append((op, len(left[op] - covered)))
        covered |= left.pop(op)
    return out


@pytest.mark.parametrize("space", [6, 60, 5000, 1 << 40])
def test_falsify_incremental_greedy_matches_brute_force(space):
    """Gain order must equal the full-scan greedy, ties by name, in any id space."""
    import random

    rng = random.Random(space)
    for _ in range(40):
        sets = {
            f"op{rng.randrange(50)}_{i}": {rng.randrange(space) for _ in range(rng.randint(0, 30))}
            for i in range(rng.randint(0, 25))
        }
        assert [(r.op, r.gain) for r in marginal_table(sets)] == _brute_greedy(sets)


def test_marginal_coverage_does_not_mutate_or_copy_state():
    """Report reads the live per-operator sets; it must leave them untouched."""
    sa = ShapleyAttribution()
    sa.record({"a"}, 3, {1, 2, 3})
    sa.record({"b"}, 2, {3, 4})
    before = {k: set(v) for k, v in sa._operator_edges.items()}
    rows = sa.marginal_coverage()
    assert {k: set(v) for k, v in sa._operator_edges.items()} == before
    assert sum(r.gain for r in rows) == 4


@pytest.mark.parametrize("big", [1 << 63, (1 << 64) + 7, 1 << 200])
def test_ids_outside_int64_are_ranked_by_identity(big):
    """Hash-sized ids must not crash the report; only equality matters."""
    rows = {r.op: r for r in marginal_table({"a": {big, 1}, "b": {big}, "c": {big + 1}})}
    assert (rows["a"].gain, rows["a"].sole) == (2, 1)
    assert (rows["b"].gain, rows["b"].sole) == (0, 0)
    assert rows["c"].sole == 1


@pytest.mark.parametrize("n", [1 << 16, (1 << 16) + 1])
def test_radix_key_boundary_matches_brute_force(n):
    """65536 columns use the 16-bit radix key, 65537 fall back to int64: same answer."""
    sets = {"a": set(range(n)), "b": set(range(n // 2, n + 10)), "c": {0, n + 5}}
    assert [(r.op, r.gain) for r in marginal_table(sets)] == _brute_greedy(sets)


def test_tiny_input_with_huge_ids_skips_the_direct_table():
    """Three entries must not allocate a 4M-slot count table (was ~25 ms, now sort path)."""
    import time

    sets = {"a": {DIRECT_MAX_ID - 1}, "b": {DIRECT_MAX_ID - 2, 3}}
    t = time.perf_counter()
    for _ in range(20):
        marginal_table(sets)
    assert (time.perf_counter() - t) / 20 < 0.010


def test_report_has_no_edge_cap_note():
    """No edge is ever evicted, so the report never warns about a cap."""
    sa = ShapleyAttribution()
    for i in range(20_000):
        sa.record({"b"}, 1, {100 + i})
    assert len(sa._all_edges) == 20_000
    assert "edge cap" not in _marginal_coverage(_fuzzer(sa))
