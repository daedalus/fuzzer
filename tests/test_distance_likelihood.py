"""Likelihood-weighted block distance (``DistanceMetric.LIKELIHOOD``).

Edge u -> v costs ``-log2 p(v|u)`` bits; with no observed branch counts p is
the Laplace prior ``1 / outdeg(u)`` (distinct successors, out-of-function
ones included). Reverse Dijkstra per target, AFLGo harmonic aggregation:
value = |T| / sum 1/(1+d) (targets 0). Oracles are hand arithmetic; the control is a
graph of 2-way branches only, where bits equal hops and the metric must
reproduce the harmonic BFS values exactly.
"""

import math

import pytest

from fuzzer_tool.core.analyzers.analyzer_distance import DistanceMetric, TargetDistance
from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG

FUNC_NAME = "lk_fn"


def _cfg(edges: dict[int, list[int]]) -> FunctionCFG:
    blocks = {a: BasicBlock(start=a, end=a + 1, successors=list(s)) for a, s in edges.items()}
    return FunctionCFG(name=FUNC_NAME, start=min(edges), end=max(edges) + 1, blocks=blocks)


def _h(*bits: float) -> float:
    """AFLGo harmonic value over per-target distances (in bits)."""
    return len(bits) / sum(1.0 / (1.0 + d) for d in bits)


def _values(edges, targets, metric=DistanceMetric.LIKELIHOOD) -> dict:
    td = TargetDistance("/nonexistent", targets=[FUNC_NAME], metric=metric)
    cfg = _cfg(edges)
    td.functions = {FUNC_NAME: (cfg.start, cfg.end)}
    td._func_addrs_sorted = [(cfg.start, FUNC_NAME)]
    td.target_addrs = set(targets)
    td._cfgs = {FUNC_NAME: cfg}
    td._compute_bb_values()
    return td._bb_value


# Ranking flip. 10 is one hop away behind a 4-way switch (2 bits); 20 is
# three hops away on a straight line (0 bits).
#
#   0 -> 10 -> {30*, 11, 12, 13}
#     -> 20 -> 21 -> 22 -> 30*
FLIP = {
    0: [10, 20],
    10: [30, 11, 12, 13],
    11: [],
    12: [],
    13: [],
    20: [21],
    21: [22],
    22: [30],
    30: [],
}

# Every non-sink block is a 2-way branch: bits == hops.
BINARY = {0: [1, 2], 1: [3, 9], 2: [1, 9], 3: [4, 9], 4: [0, 9], 9: []}


def test_metric_value_is_cli_name():
    assert DistanceMetric("likelihood") is DistanceMetric.LIKELIHOOD


def test_control_harmonic_matches_itself():
    # Rule 46: the reference against a second run of itself.
    assert _values(BINARY, {4}, DistanceMetric.HARMONIC) == _values(
        BINARY, {4}, DistanceMetric.HARMONIC
    )


def test_binary_branches_equal_harmonic():
    # 2-way branches only: each edge costs exactly 1 bit = 1 hop.
    assert _values(BINARY, {4}) == _values(BINARY, {4}, DistanceMetric.HARMONIC)


def test_falsification_ranking_flips():
    h = _values(FLIP, {30}, DistanceMetric.HARMONIC)
    lk = _values(FLIP, {30})
    assert h[10] < h[20]
    assert lk[20] < lk[10]
    assert lk[10] == _h(math.log2(4))
    assert lk[20] == _h(0.0)
    # 0 is a 2-way branch: 1 bit, then the free straight line.
    assert lk[0] == _h(1.0)


def test_adversarial_dijkstra_not_first_found():
    # BFS reaches 0 first through 1 (2 hops, 3 bits: 8-way switch); the
    # cheaper route is 0 -> 2 -> 3 -> T (3 hops, 1 bit). A FIFO BFS that
    # settles on first discovery gets 1 + 3 = 4 bits; Dijkstra gets 1 + 0.
    sw = list(range(100, 107))
    edges = {0: [1, 2], 1: [50, *sw], 2: [3], 3: [50], 50: []} | {s: [] for s in sw}
    vals = _values(edges, {50})
    assert vals[1] == _h(3.0)
    assert vals[2] == vals[3] == _h(0.0)
    assert vals[0] == _h(1.0)


def test_adversarial_harmonic_over_two_targets():
    # 0 -> {A, mid}, mid -> {B, dead}. d(0,A) = 1 bit, d(0,B) = 2 bits.
    vals = _values({0: [5, 1], 1: [6, 9], 5: [], 6: [], 9: []}, {5, 6})
    assert vals[0] == pytest.approx(_h(1.0, 2.0))
    assert vals[1] == _h(1.0)  # only B reachable


def test_adversarial_unreachable_and_cycle_get_no_value():
    vals = _values({0: [1, 5], 1: [], 5: [6], 6: [5]}, {1})
    assert 5 not in vals and 6 not in vals
    assert vals[0] == _h(1.0)


def test_adversarial_duplicate_successor_counted_once():
    assert _values({0: [1, 1], 1: []}, {1})[0] == _h(0.0)


def test_adversarial_out_of_function_successor_widens_branch():
    # The 0x999 successor is outside the CFG but still a branch outcome.
    assert _values({0: [1, 0x999], 1: []}, {1})[0] == _h(1.0)


def test_adversarial_zero_weight_self_loop_terminates():
    vals = _values({0: [0], 1: [0], 2: [1, 2]}, {0})
    assert vals == {0: 0.0, 1: _h(0.0), 2: _h(1.0)}


def test_adversarial_target_with_successors_stays_zero():
    vals = _values({0: [1], 1: [0, 2], 2: []}, {1})
    assert vals[1] == 0.0 and vals[0] == _h(0.0)
    assert 2 not in vals
