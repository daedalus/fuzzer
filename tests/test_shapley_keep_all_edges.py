"""ShapleyAttribution keeps every edge: no cap, no eviction, exact counts on both backends.

Every test runs on the Counter backend and on the dense one (forced by a zero
pair budget, so the store migrates after the first record).
"""

import collections
import pickle

import numpy as np
import pytest

from fuzzer_tool.core import shapley
from fuzzer_tool.core.marginal_coverage import marginal_table
from fuzzer_tool.core.shapley import FUSED_MIN_WORK, ShapleyAttribution, _CounterCounts, _EdgeCounts


@pytest.fixture(params=["counters", "dense"], autouse=True)
def backend(request, monkeypatch):
    if request.param == "dense":
        monkeypatch.setattr(shapley, "COUNTER_PAIRS_MAX", 0)
    return request.param


class _CounterRef:
    """Uncapped reference: one Counter per operator plus per-edge totals."""

    def __init__(self):
        self.op = collections.defaultdict(collections.Counter)
        self.total = collections.Counter()

    def record(self, ops, edges):
        for op in ops:
            self.op[op].update(edges)
        self.total.update(edges)


# Three ops per exec: below SMALL_W the Counter buffer takes it, from FUSED_W the matrix.
SMALL_W = FUSED_MIN_WORK // 4 - 1
FUSED_W = FUSED_MIN_WORK // 4 + 1
STORES = {
    "small": lambda i: SMALL_W,
    "fused": lambda i: FUSED_W,
    "mixed": lambda i: FUSED_W if i % 3 == 0 else SMALL_W,
}


def _workload(n, width=STORES["mixed"], ops=40, space=60_000):
    """Deterministic (ops, edges) stream: overlapping windows, >10k distinct edges."""
    for i in range(n):
        picked = {f"op{(i * k) % ops}" for k in (1, 3, 7)}
        yield picked, {(i * 37 + k * k) % space for k in range(width(i))}


def _run(n, store):
    sa, ref = ShapleyAttribution(), _CounterRef()
    width = STORES[store] if isinstance(store, str) else store
    for ops, edges in _workload(n, width):
        sa.record(ops, len(edges), edges)
        ref.record(ops, edges)
    return sa, ref


def _state(sa):
    return {op: dict(c) for op, c in sa._op_edge_count.items()}, dict(sa._edge_total)


def _ref_state(ref):
    return {op: dict(c) for op, c in ref.op.items()}, dict(ref.total)


def test_regression_wide_exec_keeps_every_edge():
    """One exec of 19,000 edges: the capped tracker kept 0 (Misra-Gries) or half."""
    wide = set(range(19_000))
    sa = ShapleyAttribution()
    for _ in range(3):
        sa.record({"havoc"}, len(wide), wide)
    assert set(sa._all_edges) == wide
    assert dict(sa._op_edge_count["havoc"]) == dict.fromkeys(wide, 3)
    assert dict(sa._edge_total) == dict.fromkeys(wide, 3)


def test_regression_wide_execs_do_not_drain_hot_edges():
    """Wide execs after a hot set leave the hot counts untouched."""
    hot, wide = set(range(100_000, 103_000)), set(range(19_000))
    sa = ShapleyAttribution()
    for _ in range(5):
        sa.record({"flip"}, len(hot), hot)
    for _ in range(2):
        sa.record({"havoc"}, len(wide), wide)
    assert sa._edge_total[100_000] == 5
    assert set(sa._all_edges) == hot | wide


def test_control_reference_equals_itself():
    """Hard Rule 46: the oracle must agree with a second run of itself."""
    a, b = _CounterRef(), _CounterRef()
    for ops, edges in _workload(600):
        a.record(ops, edges)
        b.record(ops, edges)
    assert _ref_state(a) == _ref_state(b)


@pytest.mark.parametrize("store", sorted(STORES))
def test_matches_uncapped_reference(store):
    """Counts and totals equal the uncapped reference past 10k edges, on every path."""
    sa, ref = _run(1500, store)
    assert len(ref.total) > 10_000
    assert _state(sa) == _ref_state(ref)
    assert set(sa._all_edges) == set(ref.total)
    assert {op: set(e) for op, e in sa._operator_edges.items()} == {
        op: set(c) for op, c in ref.op.items()
    }


@pytest.mark.parametrize("store", sorted(STORES))
def test_marginal_coverage_matches_reference_sets(store):
    sa, ref = _run(700, store)
    got = sa.marginal_coverage()
    want = marginal_table({op: set(c) for op, c in ref.op.items()})
    assert [r[:4] for r in got] == [r[:4] for r in want]
    assert [r.shapley for r in got] == pytest.approx([r.shapley for r in want], rel=1e-12)


@pytest.mark.parametrize("store", sorted(STORES))
@pytest.mark.parametrize("subset", [None, ["op0", "op3", "missing"]])
def test_shapley_values_match_reference(subset, store):
    """credit(e, op) / |T_e| from the reference counts, restricted to *subset*."""
    sa, ref = _run(400, store)
    keep = sorted(ref.op) if subset is None else subset
    expected = dict.fromkeys(keep, 0.0)
    for e in ref.total:
        held = {op: ref.op[op][e] for op in keep if ref.op.get(op, {}).get(e)}
        tot = sum(held.values())
        for op, c in held.items():
            expected[op] += c / tot / len(held)
    norm = sum(expected.values())
    got = sa.shapley_values(subset)
    assert got == {op: pytest.approx(v / norm, abs=1e-12) for op, v in expected.items()}
    assert np.isclose(sum(got.values()), 1.0)


def test_falsify_one_extra_count_is_detected():
    """The comparison is not vacuous: a single off-by-one count must fail it."""
    sa, ref = _run(50, "mixed")
    ref.op["op0"][next(iter(ref.op["op0"]))] += 1
    assert _state(sa) != _ref_state(ref)


def test_reads_between_writes_fold_once():
    """A read folds pending counts in; later writes add on top, never twice."""
    sa, ref = ShapleyAttribution(), _CounterRef()
    for i, (ops, edges) in enumerate(_workload(90, STORES["mixed"])):
        sa.record(ops, len(edges), edges)
        ref.record(ops, edges)
        if i % 7 == 0:
            assert _state(sa) == _ref_state(ref)
    assert _state(sa) == _ref_state(ref)


def test_pending_buffer_is_bounded(monkeypatch):
    """Dense store: small execs buffer in Counters; past the bound they fold."""
    monkeypatch.setattr(shapley, "COUNTER_PAIRS_MAX", 0)
    monkeypatch.setattr(shapley, "PENDING_MAX", 100)
    sa, ref = ShapleyAttribution(), _CounterRef()
    for ops, edges in _workload(200, STORES["small"]):
        sa.record(ops, len(edges), edges)
        ref.record(ops, edges)
        assert sa._counts.pending_size() <= 100 + len(edges) * (len(ops) + 1)
    assert _state(sa) == _ref_state(ref)


def test_adversarial_ids_outside_int64_and_negative():
    """Edge ids are only hashed: negative and > int64 ids survive both paths."""
    big = 1 << 70
    edges = {big + k for k in range(FUSED_MIN_WORK)} | {-1, -2}
    sa = ShapleyAttribution()
    sa.record({"a"}, len(edges), edges)  # fused
    sa.record({"a"}, 1, {big})  # small, on a fused-seen edge
    sa.record({"b"}, 1, {-(1 << 70)})  # small, new edge
    expected = dict.fromkeys(edges, 1)
    expected[big] = 2
    assert dict(sa._op_edge_count["a"]) == expected
    assert dict(sa._op_edge_count["b"]) == {-(1 << 70): 1}
    assert sa._edge_total[big] == 2


def test_adversarial_growth_past_initial_capacity():
    """One exec wider than the initial slots, more operators than initial lanes."""
    sa = ShapleyAttribution()
    wide = set(range(shapley.COUNTS_EDGES0 * 2 + 1))
    ops = {f"op{i}" for i in range(shapley.COUNTS_OPS0 * 2 + 1)}
    sa.record(ops, len(wide), wide)
    sa.record({"op0"}, len(wide), wide)
    assert set(sa._op_edge_count) == ops
    assert dict(sa._op_edge_count["op0"]) == dict.fromkeys(wide, 2)
    assert dict(sa._op_edge_count["op1"]) == dict.fromkeys(wide, 1)


def test_record_without_edges_creates_no_operator():
    sa = ShapleyAttribution()
    sa.record({"a"}, 0, set())
    assert "a" not in sa._operator_edges and len(sa._all_edges) == 0


def test_pickle_round_trip_with_pending_counts():
    sa, _ = _run(30, "mixed")
    clone = pickle.loads(pickle.dumps(sa))
    assert _state(clone) == _state(sa)


def test_narrow_run_stays_on_counters(monkeypatch):
    """Ifs up: narrow execs within the pair budget never leave the Counter store."""
    monkeypatch.setattr(shapley, "COUNTER_PAIRS_MAX", 1_000_000)
    sa, _ = _run(shapley.DENSE_WARMUP + 100, "small")
    assert isinstance(sa._counts, _CounterCounts)


def test_wide_run_migrates_after_warmup(monkeypatch):
    """Mean work per exec >= FUSED_MIN_WORK after warmup: migrate, counts unchanged."""
    monkeypatch.setattr(shapley, "COUNTER_PAIRS_MAX", 1_000_000)
    monkeypatch.setattr(shapley, "DENSE_WARMUP", 20)
    sa, ref = _run(60, lambda i: FUSED_MIN_WORK)  # >= FUSED_MIN_WORK edges even at 1 op
    assert isinstance(sa._counts, _EdgeCounts)
    assert _state(sa) == _ref_state(ref)


def test_pair_budget_migrates_mid_stream_exactly(monkeypatch):
    """Crossing the pair budget mid-run migrates once; counts stay exact across it."""
    monkeypatch.setattr(shapley, "COUNTER_PAIRS_MAX", 20_000)
    monkeypatch.setattr(shapley, "PAIRS_CHECK_EVERY", 1)
    monkeypatch.setattr(shapley, "DENSE_WARMUP", 10**9)
    sa, ref = ShapleyAttribution(), _CounterRef()
    kinds = []
    for ops, edges in _workload(600, STORES["small"]):
        sa.record(ops, len(edges), edges)
        ref.record(ops, edges)
        kinds.append(type(sa._counts))
    assert kinds[0] is _CounterCounts and kinds[-1] is _EdgeCounts
    assert sum(1 for a, b in zip(kinds, kinds[1:], strict=False) if a is not b) == 1
    assert _state(sa) == _ref_state(ref)
