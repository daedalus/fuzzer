"""SelectFuzz reaching-probability block fitness (``DistanceMetric.REACH_PROB``).

P(target) = 1; P(b) = mean P over b's distinct successors; exits and
out-of-function successors count 0. The stored value is the surprisal
``-log2 P`` so "lower is closer" holds for every distance consumer.
"""

import ast
import inspect
import math

import pytest

from fuzzer_tool.core.analyzers.analyzer_distance import DistanceMetric, TargetDistance
from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG

FUNC_NAME = "reach_fn"


def _cfg(edges: dict[int, list[int]]) -> FunctionCFG:
    blocks = {a: BasicBlock(start=a, end=a + 1, successors=list(s)) for a, s in edges.items()}
    return FunctionCFG(name=FUNC_NAME, start=min(edges), end=max(edges) + 1, blocks=blocks)


def _values(edges, targets, metric=DistanceMetric.REACH_PROB, gate_bonus=0.0) -> dict:
    td = TargetDistance("/nonexistent", targets=[FUNC_NAME], metric=metric, gate_bonus=gate_bonus)
    cfg = _cfg(edges)
    td.functions = {FUNC_NAME: (cfg.start, cfg.end)}
    td._func_addrs_sorted = [(cfg.start, FUNC_NAME)]
    td.target_addrs = set(targets)
    td._cfgs = {FUNC_NAME: cfg}
    td._compute_bb_values()
    return td._bb_value


# Loop with the target inside: 0 -> 1 -> {2(target), 3(exit)}, 2 -> 1.
LOOP = {0: [1], 1: [2, 3], 2: [1], 3: []}

# Ranking flip. 10 is one hop from the target but behind a 4-way switch
# (P = 1/4); 20 is three hops away on a straight line (P = 1).
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


def test_default_metric_is_harmonic():
    assert TargetDistance("/nonexistent")._metric is DistanceMetric.HARMONIC


def test_harmonic_unchanged_control():
    # Same numbers as tests/test_distance_gate_bonus.py: the default path is untouched.
    assert _values(LOOP, {2}, DistanceMetric.HARMONIC) == {2: 0.0, 1: 2.0, 0: 3.0}


def test_loop_surprisal():
    # P(1) = (P(2) + P(3)) / 2 = 0.5 -> 1 bit; P(0) = P(1).
    assert _values(LOOP, {2}) == {2: 0.0, 1: 1.0, 0: 1.0}


def test_falsification_ranking_flips():
    h = _values(FLIP, {30}, DistanceMetric.HARMONIC)
    r = _values(FLIP, {30})
    assert h[10] < h[20]  # distance prefers the short, unlikely path
    assert r[20] < r[10]  # reaching probability prefers the certain one
    assert r[10] == -math.log2(1 / 4)
    assert r[20] == 0.0


def test_adversarial_closed_cycle_gets_no_value():
    # 5 <-> 6 never reaches the target: minimal solution P = 0, no value.
    vals = _values({0: [1, 5], 1: [], 5: [6], 6: [5]}, {1})
    assert 5 not in vals and 6 not in vals
    assert vals[0] == 1.0


def test_adversarial_out_of_function_successor_counts_zero():
    assert _values({0: [1, 0x999], 1: []}, {1})[0] == 1.0


def test_adversarial_duplicate_successor_counted_once():
    # jcc whose taken and fall-through targets coincide is one branch.
    assert _values({0: [1, 1], 1: []}, {1})[0] == 0.0


def test_adversarial_self_loop_converges_to_certain():
    # P(0) = (P(0) + 1) / 2 has the single solution P = 1.
    assert _values({0: [0, 1], 1: []}, {1})[0] == pytest.approx(0.0, abs=1e-9)


def test_two_block_loop_closed_form():
    # 0 -> {1, dead}, 1 -> {0, T}: P0 = P1/2, P1 = (P0 + 1)/2 -> P1 = 2/3, P0 = 1/3.
    # The cycle defeats the Gauss-Seidel pass, so this runs the numpy Jacobi path.
    vals = _values({0: [1, 9], 1: [0, 5], 5: [], 9: []}, {5})
    assert vals[0] == pytest.approx(-math.log2(1 / 3), rel=1e-9)
    assert vals[1] == pytest.approx(-math.log2(2 / 3), rel=1e-9)
    assert 9 not in vals


def test_adversarial_deep_chain_exact():
    # i -> {i+1, dead}: P(i) = 2^-(n-i). Deep enough to defeat a fixed
    # sweep count in a bad visiting order, shallow enough for doubles.
    n = 300
    edges = {i: [i + 1, 10_000 + i] for i in range(n)}
    edges[n] = []
    edges |= {10_000 + i: [] for i in range(n)}
    vals = _values(edges, {n})
    for i in range(n):
        assert vals[i] == pytest.approx(n - i, rel=1e-9)


def test_adversarial_target_with_successors_stays_zero():
    vals = _values({0: [1], 1: [0, 2], 2: []}, {1})
    assert vals[1] == 0.0 and vals[0] == 0.0
    assert 2 not in vals


def test_gate_bonus_composes():
    # Gates 0 and 1 dominate the target; 1 bit each, halved.
    assert _values(LOOP, {2}, gate_bonus=0.5) == {2: 0.0, 1: 0.5, 0: 0.5}


# ── wiring ──────────────────────────────────────────────────────────────


def test_cli_flag_wired():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    assert calls and all("distance_metric" in {k.arg for k in c.keywords} for c in calls)
    assert "distance_metric" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
    assert "distance_metric" not in commands._HAIL_MARY_FLAGS


def test_activation_passes_metric(monkeypatch):
    from fuzzer_tool.core import analyzer_registry

    seen = {}

    class _FakeTD:
        def __init__(self, *a, **kw):
            seen.update(kw)

        def load(self):
            return False

    monkeypatch.setattr("fuzzer_tool.core.analyzers.analyzer_distance.TargetDistance", _FakeTD)
    f = type(
        "F",
        (),
        {
            "target": "/bin/true",
            "_distance_targets": ["main"],
            "_use_cfg_cache": False,
            "debug": False,
            "_distance_metric": DistanceMetric.REACH_PROB,
        },
    )()
    analyzer_registry._activate_distance(f)
    assert seen["metric"] is DistanceMetric.REACH_PROB
