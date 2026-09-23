"""``edge_diagnostic.py matrix --flow``: are the count relations Kirchhoff identities?

P1-2 of handover_edge_id_axis_2026-09-18. The tracer's (prev, cur, call site)
log is a walk, so its per-run edge counts are circulations of the observed
graph once closed with a virtual exit. A relation among counts is structural
iff it is orthogonal to every cycle; anything else is corpus coincidence or an invariant flow cannot express.
These tests pin the graph build, the cycle basis, the exact rank, and the
node-law/other split on graphs small enough to check by hand.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "edge_diagnostic.py"


def _load():
    spec = importlib.util.spec_from_file_location("edge_diagnostic", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ED = _load()
EDGE = ED.FlowGraph.EDGE.value
CALL_SITE = ED.FlowGraph.CALL_SITE.value


def _walk(*steps) -> np.ndarray:
    """Tracer records for a walk; a step is a location or (location, call site)."""
    recs, prev = [], 0
    for s in steps:
        loc, site = s if isinstance(s, tuple) else (s, 0)
        recs.append((prev, loc, site))
        prev = loc
    return np.array(recs, dtype=np.uint64).reshape(-1, 3)


def _incidence(src, dst, n):
    b = np.zeros((n, len(src)), dtype=np.int64)
    b[src, np.arange(len(src))] -= 1
    b[dst, np.arange(len(src))] += 1
    return b


# Two diamonds in series, 1 -> 2 -> {3|4} -> 5 -> {6|7} -> 8, where the two
# choices are correlated in the corpus but not in the graph.
DIAMONDS = [_walk(1, 2, 3, 5, 6, 8), _walk(1, 2, 4, 5, 7, 8), _walk(1, 2, 3, 5, 6, 8)]


# ── exact rank ────────────────────────────────────────────────────────


def test_rank_p_agrees_with_float_rank_on_its_control():
    m = np.array([[1, 2, 3], [4, 5, 6], [5, 7, 9], [0, 0, 0]])
    assert np.linalg.matrix_rank(m.astype(float)) == 2  # control: the oracle itself
    assert ED._rank_p(m) == 2
    assert ED._rank_p(np.eye(4, dtype=np.int64)) == 4
    assert ED._rank_p(np.zeros((3, 5), dtype=np.int64)) == 0


def test_rank_p_is_exact_where_float_rank_is_not():
    # Rank 2 over Q, but the entries span 1e17: float64 loses the dependence.
    big = 10**17
    m = np.array([[big, 1], [2 * big, 2], [1, 0]], dtype=object)
    assert ED._rank_p(m) == 2
    assert ED._rank_p(m[:2]) == 1


# ── cycle basis ───────────────────────────────────────────────────────


def _check_basis(src, dst, n, components):
    src, dst = np.array(src), np.array(dst)
    z = ED._cycle_basis(src, dst, n)
    assert z.shape == (len(src) - n + components, len(src))
    assert not np.count_nonzero(_incidence(src, dst, n) @ z.T)
    assert ED._rank_p(z) == z.shape[0]


def test_cycle_basis_spans_the_kernel_of_the_incidence_matrix():
    # diamond with a return edge: 0->1, 0->2, 1->3, 2->3, 3->0
    _check_basis([0, 0, 1, 2, 3], [1, 2, 3, 3, 0], 4, components=1)


def test_cycle_basis_adversarial_shapes():
    # parallel edges 0->1 twice, a self-loop at 4, a 2-cycle 3<->5, and
    # isolated nodes 2 and 6: five components
    _check_basis([0, 0, 1, 4, 3, 5], [1, 1, 0, 4, 5, 3], 7, components=5)


# ── node laws vs the rest ─────────────────────────────────────────────


def test_correlated_branches_are_not_node_laws_chains_are():
    got = ED.flow_structure(DIAMONDS, lll_rows=0)[EDGE]
    assert got["discontinuities"] == 0
    assert got["kirchhoff_violations"] == 0
    # nodes: entry, 1..8, exit; edges: 10 real + (8, exit) + (exit, entry)
    assert (got["nodes"], got["columns"], got["cycle_rank"]) == (10, 10, 3)
    assert got["count_rank"] == 2
    assert got["non_flow_dims"] == 1
    dups = got["relations"]["duplicates"]
    # Pairs are (first of class, member). [1,1,1]: (0,1)~(1,2), a chain.
    # [1,0,1] and [0,1,0]: one chain pair each, two cross-diamond pairs each.
    assert (dups["found"], dups["structural"]) == (7, 3)
    shown = {tuple(map(tuple, r)) for r in dups["non_flow"]}
    assert (((2, 3), 1), ((5, 6), -1)) in shown


def test_a_node_law_is_structural_whatever_the_corpus():
    # Conservation at node 5 of DIAMONDS: (3,5) + (4,5) - (5,6) - (5,7) = 0.
    g = ED._flow_graph(DIAMONDS, ED.FlowGraph.EDGE)
    r = np.zeros(len(g["cols"]), dtype=np.int64)
    for key, c in (((3, 5), 1), ((4, 5), 1), ((5, 6), -1), ((5, 7), -1)):
        r[g["cols"][key]] = c
    assert ED._is_structural(g["zp"], r)
    r[g["cols"][(5, 7)]] = 0  # drop one term: no longer a law
    assert not ED._is_structural(g["zp"], r)


def test_derivable_is_columns_minus_independent_cycles():
    got = ED.flow_structure(DIAMONDS, lll_rows=0)[EDGE]
    assert got["derivable"] == 7  # 10 columns, 3 independent cycles


def test_abort_mid_graph_closes_through_the_virtual_exit():
    runs = [_walk(1, 2, 3, 5, 6, 8), _walk(1, 2, 4)]  # second run aborts at 4
    got = ED.flow_structure(runs, lll_rows=0)[EDGE]
    assert got["kirchhoff_violations"] == 0
    assert got["exits"] == 2
    assert got["non_flow_dims"] == 0


def test_discontinuous_log_is_counted_not_trusted():
    bad = _walk(1, 2, 3)
    bad[2, 0] = 9  # record claims prev 9, the walk was at 2
    got = ED.flow_structure([bad, _walk(1, 2, 3)], lll_rows=0)[EDGE]
    assert got["discontinuities"] == 1


def test_empty_run_is_a_zero_row_not_a_crash():
    empty = np.empty((0, 3), dtype=np.uint64)
    got = ED.flow_structure([empty, *DIAMONDS], lll_rows=0)[EDGE]
    assert got["runs"] == 4
    assert got["count_rank"] == 2


def test_call_site_graph_derives_call_return_matching():
    # F = locations 10, 11, called from site A=100 (caller 1, returns to 2)
    # and site B=200 (caller 3, returns to 4). Flat Kirchhoff at 10/11 mixes
    # the two returns; the call-site graph keeps them apart.
    a, b = 100, 200
    runs = [
        _walk(1, (10, a), (11, a), 2, 5),
        _walk(3, (10, b), (11, b), 4, 5),
        _walk(1, (10, a), (11, a), 2, 3, (10, b), (11, b), 4, 5),
    ]
    got = ED.flow_structure(runs, lll_rows=0)
    # Pairs are (first of class, member): (0,1) heads the [1,0,1] class.
    call, ret = ((0, 1), 1), ((11, 2), -1)
    flat = {tuple(map(tuple, r)) for r in got[EDGE]["relations"]["duplicates"]["non_flow"]}
    assert (call, ret) in flat
    ctx = got[CALL_SITE]["relations"]["duplicates"]
    shown = {tuple((k[:2], c) for k, c in r) for r in ctx["non_flow"]}
    assert (call, ret) not in shown


def test_lll_recovers_the_node_law_and_classifies_it():
    # Distinct profiles [1,1,1], [1,0,1], [0,1,0]: one relation, the law at
    # node 2, found by both the triple search and LLL.
    got = ED.flow_structure(DIAMONDS, lll_rows=100)[EDGE]["relations"]
    assert (got["triples"]["found"], got["triples"]["structural"]) == (1, 1)
    assert (got["lll"]["found"], got["lll"]["structural"]) == (1, 1)
    assert ED.flow_structure(DIAMONDS, lll_rows=0)[EDGE]["relations"]["lll"]["found"] == 0


# ── control against the shim's collection ─────────────────────────────


def test_profile_match_is_order_free_and_detects_a_changed_column():
    counts = np.array([[1, 0, 3], [2, 5, 0]])
    assert ED._profile_match(counts, counts)  # control against itself first
    assert ED._profile_match(counts, counts[:, [2, 0, 1]])
    other = counts.copy()
    other[1, 1] = 4
    assert not ED._profile_match(counts, other)
    assert not ED._profile_match(counts, counts[:, :2])
