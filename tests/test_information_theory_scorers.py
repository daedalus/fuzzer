"""Handover 7.2-7.4 scorers: mutual information, conditional entropy, path novelty.

Expected values are closed forms (BSC / Z-channel capacity, the Kaspar-Schuster
LZ76 example) or planted ground truth, never the module's own output.
"""

from __future__ import annotations

import math
import random

import pytest

from fuzzer_tool.core.cond_entropy import (
    conditional_entropy,
    context_report,
    energy_weights,
    prefix_context,
)
from fuzzer_tool.core.mutual_info import (
    channel_capacity,
    edge_presence_mi,
    mutual_information,
    operator_new_edge_capacity,
    permutation_null,
)
from fuzzer_tool.core.path_novelty import (
    PathNovelty,
    lz76_complexity,
    lz76_normalised,
    ncd,
)


def _h2(p: float) -> float:
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# ---- 7.2 mutual information -------------------------------------------------


def test_mi_perfect_dependence_is_log_alphabet():
    xs = [0, 1, 2, 3] * 25
    assert mutual_information(xs, xs) == pytest.approx(2.0)


def test_mi_full_cross_product_is_zero():
    xs = [a for a in range(3) for _ in range(4)]
    ys = [b for _ in range(3) for b in range(4)]
    assert mutual_information(xs, ys) == pytest.approx(0.0, abs=1e-12)


def test_mi_length_mismatch_raises():
    with pytest.raises(ValueError):
        mutual_information([1], [1, 2])


def test_permutation_null_separates_dependent_from_independent():
    rng = random.Random(1)
    xs = [rng.randrange(4) for _ in range(400)]
    dep = [x % 2 for x in xs]
    ind = [rng.randrange(2) for _ in range(400)]
    _, _, _, z_dep = permutation_null(xs, dep, 100, seed=3)
    _, _, _, z_ind = permutation_null(xs, ind, 100, seed=3)
    assert z_dep > 10
    assert abs(z_ind) < 3.5


def test_permutation_null_is_deterministic():
    xs = [i % 3 for i in range(60)]
    ys = [(i * 7) % 5 for i in range(60)]
    assert permutation_null(xs, ys, 50, seed=9) == permutation_null(xs, ys, 50, seed=9)


def test_edge_presence_mi_ranks_planted_edge_first():
    rng = random.Random(5)
    obs = []
    for _ in range(300):
        x = rng.randrange(4)
        es = {100} | ({7} if x == 0 else set()) | ({8} if rng.random() < 0.3 else set())
        obs.append((x, es))
    ranked = edge_presence_mi(obs, n_perm=60, seed=1)
    assert ranked[0][0] == 7
    ids = [e for e, _, _ in ranked]
    assert 100 not in ids  # always present: below min_support on the absent side


def test_blahut_arimoto_bsc_capacity():
    cap, q = channel_capacity([[0.9, 0.1], [0.1, 0.9]])
    assert cap == pytest.approx(1 - _h2(0.1), abs=1e-6)
    assert q == pytest.approx([0.5, 0.5], abs=1e-4)


def test_blahut_arimoto_z_channel_capacity():
    # Z-channel with p(1|1) = 1/2: C = log2(5/4), optimal P(x=1) = 2/5.
    cap, q = channel_capacity([[1.0, 0.0], [0.5, 0.5]])
    assert cap == pytest.approx(math.log2(1.25), abs=1e-6)
    assert q[1] == pytest.approx(0.4, abs=1e-3)


def test_blahut_arimoto_rejects_bad_rows():
    with pytest.raises(ValueError):
        channel_capacity([[0.5, 0.4]])


def test_operator_capacity_is_achieved_by_returned_mix():
    recs = [("good", i % 2 == 0) for i in range(200)] + [("dead", False)] * 200
    cap, mix = operator_new_edge_capacity(recs, alpha=0.5)
    # Independent recomputation of I(X;Y) under the returned input mix.
    p = {"good": (100 + 0.5) / 201, "dead": 0.5 / 201}
    py1 = sum(mix[o] * p[o] for o in p)
    mi = 0.0
    for o in p:
        for py, pyx in ((py1, p[o]), (1 - py1, 1 - p[o])):
            mi += mix[o] * pyx * math.log2(pyx / py)
    assert cap > 0.1
    assert mi == pytest.approx(cap, abs=1e-6)
    assert sum(mix.values()) == pytest.approx(1.0)


def test_operator_capacity_mix_is_not_a_productivity_ranking():
    # Documented caveat: the near-deterministic dead operator is MORE
    # distinguishable, so the achieving mix weights it above the 50% operator.
    recs = [("good", i % 2 == 0) for i in range(200)] + [("dead", False)] * 200
    _, mix = operator_new_edge_capacity(recs)
    assert mix["dead"] > mix["good"]


# ---- 7.3 conditional entropy -----------------------------------------------


def test_conditional_entropy_deterministic_is_zero():
    pairs = [(c, frozenset({c})) for c in range(5) for _ in range(20)]
    assert conditional_entropy(pairs) == pytest.approx(0.0, abs=1e-12)


def test_conditional_entropy_uniform_two_paths_is_one_bit():
    pairs = [("c", i % 2) for i in range(4000)]
    assert conditional_entropy(pairs, miller_madow=False) == pytest.approx(1.0)
    # Miller-Madow adds (k-1)/(2 n ln2) = 1/(8000 ln2)
    assert conditional_entropy(pairs) == pytest.approx(
        1.0 + 1 / (2 * 4000 * math.log(2))
    )


def test_context_report_statuses_and_weights():
    pairs = (
        [("magic", "A")] * 30
        + [("body", i % 4) for i in range(40)]
        + [("rare", "A")] * 2
    )
    rep = context_report(pairs, min_n=8)
    assert rep["magic"]["status"] == "determined"
    assert rep["body"]["status"] == "open"
    assert rep["rare"]["status"] == "undetermined"  # no evidence: never "determined"
    w = energy_weights(rep)
    assert w["magic"] < w["rare"] < w["body"]


def test_prefix_context():
    assert prefix_context(b"\x89PNG\r\n\x1a\n", 4) == b"\x89PNG"


# ---- 7.4 path novelty -------------------------------------------------------


def test_lz76_kaspar_schuster_example():
    assert lz76_complexity("0001101001000101") == 6


def test_lz76_constant_vs_random():
    assert lz76_complexity([1] * 64) == 2
    rng = random.Random(0)
    rnd = [rng.randrange(4) for _ in range(512)]
    assert lz76_complexity(rnd) > 10 * lz76_complexity([1, 2, 3, 4] * 128)
    assert lz76_normalised(rnd) > 0.7


def test_lz76_edges():
    assert lz76_complexity([]) == 0
    assert lz76_complexity([3]) == 1


def test_ncd_identical_small_unrelated_large():
    rng = random.Random(2)
    a = [rng.randrange(1 << 20) for _ in range(300)]
    b = [rng.randrange(1 << 20) for _ in range(300)]
    assert ncd(a, a) < 0.1
    assert ncd(a, b) > 0.8


def test_ncd_detects_reordering_with_identical_edge_set():
    rng = random.Random(4)
    a = [rng.randrange(1 << 20) for _ in range(300)]
    shuffled = a[:]
    rng.shuffle(shuffled)
    assert set(a) == set(shuffled)
    assert ncd(a, shuffled) > 0.5 > ncd(a, a)


def test_path_novelty_order_only_signal():
    rng = random.Random(8)
    t = [rng.randrange(1 << 20) for _ in range(200)]
    pn = PathNovelty()
    assert pn.novelty(t) == 1.0  # empty reference set
    assert pn.observe(t) is True
    assert pn.observe(t) is False  # identical ordering already held
    assert pn.order_novelty(t) < 0.1
    reordered = t[:]
    rng.shuffle(reordered)
    assert pn.order_novelty(reordered) > 0.5
    assert pn.order_novelty([1, 2, 3]) is None  # different set: map already sees it


def test_path_novelty_capacity_evicts_oldest():
    pn = PathNovelty(capacity=3)
    for i in range(5):
        pn.observe([i, i + 1, i + 2])
    assert len(pn) == 3
