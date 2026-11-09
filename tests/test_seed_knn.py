"""``--seed-knn K``: approximate kNN graph of the seeds (MinHash + NN-descent).

Seeds are edge sets; closeness is Jaccard, estimated from MinHash signatures.
NN-descent (neighbours of neighbours) finds each seed's K closest seeds
without the N x N matrix. Oracles: an exhaustive scan on the same signatures,
and exact set Jaccard.

    a={1,2,3} b={1,2,3} c={7,8}     k=1   ->   a->b, b->a, c->nobody
"""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.edge_components import components as exact_components
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.seed_knn import (
    Mode,
    brute,
    build,
    knn_components,
    recall,
    seed_knn_report,
)


def _u32(*ids: int) -> np.ndarray:
    return np.asarray(ids, dtype=np.uint32)


def _clusters(
    n_clusters: int = 30, per: int = 40, core: int = 30, noise: int = 6, seed: int = 0
) -> dict[str, np.ndarray]:
    """Seeds sharing a cluster-private core of edges plus a few random ones."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for c in range(n_clusters):
        base = c * 1000
        for s in range(per):
            keep = rng.choice(core, size=core * 2 // 3, replace=False) + base
            extra = rng.choice(n_clusters * 1000, size=noise, replace=False)
            out[f"c{c}s{s}"] = np.unique(np.concatenate([keep, extra])).astype(np.uint32)
    return out


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def _rng(seed: int = 1) -> RandPool:
    return RandPool(seed=seed)


def test_twins_are_each_others_neighbour():
    g = build({"a": _u32(1, 2, 3), "b": _u32(1, 2, 3), "c": _u32(7, 8)}, k=1, strip_frac=1.0)
    idx = {k: i for i, k in enumerate(g.keys)}
    assert g.nbr[idx["a"], 0] == idx["b"]
    assert g.nbr[idx["b"], 0] == idx["a"]
    assert g.sim[idx["a"], 0] == g.perms


def test_estimate_tracks_exact_jaccard():
    book = _clusters(n_clusters=4, per=10)
    g = build(book, k=5, strip_frac=1.0, perms=512, rng=_rng())
    err = [
        abs(g.sim[i, j] / g.perms - _jaccard(book[g.keys[i]], book[g.keys[g.nbr[i, j]]]))
        for i in range(len(g.keys))
        for j in range(5)
    ]
    assert float(np.mean(err)) < 0.03


def test_control_exhaustive_scan_agrees_with_itself():
    g = build(_clusters(n_clusters=6, per=20), k=8, strip_frac=1.0, rng=_rng())
    assert g.mode is Mode.BRUTE
    assert recall(g, sample=len(g.keys), rng=_rng()) == 1.0


def test_descent_recall_on_clusters():
    g = build(_clusters(), k=10, strip_frac=1.0, brute_max=0, rng=_rng())
    assert g.mode is Mode.DESCENT
    assert recall(g, sample=200, rng=_rng()) >= 0.9


def test_descent_beats_random_graph():
    """Falsification: recall of the initial random graph is far below the result."""
    book = _clusters()
    cold = build(book, k=10, strip_frac=1.0, brute_max=0, max_iters=0, rng=_rng())
    warm = build(book, k=10, strip_frac=1.0, brute_max=0, rng=_rng())
    assert recall(cold, sample=200, rng=_rng()) < 0.3
    assert recall(warm, sample=200, rng=_rng()) > recall(cold, sample=200, rng=_rng()) + 0.5


def test_deterministic_for_a_seed():
    book = _clusters(n_clusters=10, per=30)
    a = build(book, k=6, strip_frac=1.0, brute_max=0, rng=_rng(5))
    b = build(book, k=6, strip_frac=1.0, brute_max=0, rng=_rng(5))
    assert np.array_equal(a.nbr, b.nbr) and np.array_equal(a.sim, b.sim)


def test_no_self_no_duplicates():
    g = build(_clusters(n_clusters=10, per=30), k=8, strip_frac=1.0, brute_max=0, rng=_rng())
    for i in range(len(g.keys)):
        row = g.nbr[i][g.nbr[i] >= 0]
        assert i not in row
        assert len(set(row.tolist())) == len(row)


def test_disjoint_seeds_have_no_positive_neighbour():
    book = {f"s{i}": _u32(i * 10, i * 10 + 1) for i in range(20)}
    g = build(book, k=3, strip_frac=1.0, rng=_rng())
    assert int(g.sim.max()) <= 0
    labels = knn_components(g)
    assert len(set(labels.values())) == 20


def test_universal_edges_are_stripped():
    book = {f"s{i}": _u32(0, 100 + i) for i in range(10)}
    g = build(book, k=2, strip_frac=0.5, rng=_rng())
    assert int(g.sim.max()) <= 0


def test_edgeless_and_missing_seeds_get_own_label():
    book = {"a": _u32(1, 2), "b": _u32(1, 2), "e": _u32()}
    labels = knn_components(build(book, k=1, strip_frac=1.0, rng=_rng()))
    assert labels["a"] == labels["b"] != labels["e"]
    assert set(labels) == {"a", "b", "e"}


def test_k_larger_than_corpus_pads_with_none():
    g = build({"a": _u32(1), "b": _u32(1)}, k=5, strip_frac=1.0, rng=_rng())
    assert g.nbr.shape == (2, 1)


def test_tiny_inputs():
    assert build({}, k=3, strip_frac=1.0).keys == []
    assert build({"a": _u32(1)}, k=3, strip_frac=1.0).nbr.size == 0
    assert build({"a": _u32(), "b": _u32()}, k=3, strip_frac=1.0).keys == []


def test_large_edge_ids():
    big = 2**31 + 5
    g = build({"a": _u32(big, big + 1), "b": _u32(big, big + 1)}, k=1, strip_frac=1.0)
    assert g.sim[0, 0] == g.perms


def test_identical_seeds_do_not_break_descent():
    book = {f"s{i}": _u32(1, 2, 3, 4) for i in range(300)}
    g = build(book, k=5, strip_frac=1.0, brute_max=0, rng=_rng())
    assert int(g.sim.min()) == g.perms


def test_components_match_exact_on_planted_clusters():
    book = _clusters(n_clusters=8, per=25, noise=0)
    g = build(book, k=6, strip_frac=1.0, rng=_rng())
    knn = knn_components(g)
    exact, _ = exact_components(book, strip_frac=1.0)
    pair = {(exact[k], knn[k]) for k in book}
    assert len(pair) == len(set(exact.values())) == len(set(knn.values())) == 8


def test_bad_arguments():
    for kw in ({"k": 0}, {"k": 2, "perms": 0}, {"k": 2, "strip_frac": 0.0}):
        with pytest.raises(ValueError):
            build({"a": _u32(1)}, **{"strip_frac": 1.0, **kw})


def test_brute_ties_are_stable():
    sig = np.zeros((6, 4), dtype=np.uint32)
    nbr, _ = brute(sig, k=2)
    assert nbr.tolist()[0] == [1, 2] and nbr.tolist()[5] == [0, 1]


def test_report_text():
    text = seed_knn_report(_clusters(n_clusters=5, per=12), k=4, strip_frac=1.0, rng=_rng())
    for word in ("Seed kNN", "Recall", "Components", "Exact components"):
        assert word in text


def test_report_empty():
    assert "No seeds" in seed_knn_report({}, k=3, strip_frac=1.0)


def test_minimize_flag_registered(monkeypatch, capsys):
    import sys

    from fuzzer_tool.cli import commands

    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "minimize", "--help"])
    with pytest.raises(SystemExit):
        commands.main()
    assert "--seed-knn" in capsys.readouterr().out
