"""``--seed-cut K``: max-k-cut of the seed overlap graph (B = A.A^T).

Seeds i, j weigh B_ij = number of shared (non-universal) edges. Splitting the
corpus into K groups so that overlapping seeds land in *different* groups
leaves each group internally diverse. The randomized 1/K-split of the MaxCut
post cuts (1 - 1/K) of the total weight in expectation; the greedy pass here
(each seed goes to the group it overlaps least) never does worse.

    a={1,2} b={1,2} c={3,4} d={3,4}   k=2
    best split: {a,c} | {b,d}  (a-b and c-d are both cut)
"""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.seed_cut import cut_weight, partition, seed_cut_report


def _u32(*ids: int) -> np.ndarray:
    return np.asarray(ids, dtype=np.uint32)


def _book(seed: int, n: int = 60, n_edges: int = 80, per: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        f"s{i}": np.sort(rng.choice(n_edges, size=per, replace=False)).astype(np.uint32)
        for i in range(n)
    }


def _brute(edges: dict, labels: dict, strip_frac: float) -> tuple[int, int]:
    """Pairwise oracle: (cut weight, total weight) over non-universal edges."""
    keys = list(edges)
    hold: dict[int, int] = {}
    for k in keys:
        for e in edges[k].tolist():
            hold[e] = hold.get(e, 0) + 1
    keep = {e for e, c in hold.items() if c <= strip_frac * len(keys)}
    cut = total = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            w = len(keep & set(edges[a].tolist()) & set(edges[b].tolist()))
            total += w
            cut += w if labels[a] != labels[b] else 0
    return cut, total


def test_cut_weight_matches_pairwise_oracle():
    edges = _book(1)
    labels = {k: i % 3 for i, k in enumerate(edges)}
    assert cut_weight(edges, labels, 0.5) == _brute(edges, labels, 0.5)


def test_twin_pairs_are_split():
    edges = {"a": _u32(1, 2), "b": _u32(1, 2), "c": _u32(3, 4), "d": _u32(3, 4)}
    labels = partition(edges, k=2, strip_frac=1.0)
    assert labels["a"] != labels["b"]
    assert labels["c"] != labels["d"]
    cut, total = cut_weight(edges, labels, 1.0)
    assert (cut, total) == (4, 4)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_greedy_beats_random_expectation(k, seed):
    edges = _book(seed)
    labels = partition(edges, k=k, strip_frac=0.5, passes=0, slack=None)
    cut, total = cut_weight(edges, labels, 0.5)
    assert cut * k >= (k - 1) * total


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_search_never_lowers_the_cut(seed):
    edges = _book(seed)
    base = cut_weight(edges, partition(edges, 3, 0.5, passes=0, slack=None), 0.5)[0]
    better = cut_weight(edges, partition(edges, 3, 0.5, passes=3, slack=None), 0.5)[0]
    assert better >= base


def test_balance_cap_is_respected():
    edges = _book(5, n=50)
    labels = partition(edges, k=4, strip_frac=0.5, slack=0.0)
    sizes = np.bincount(list(labels.values()), minlength=4)
    assert sizes.max() <= int(np.ceil(50 / 4))


def test_every_key_labelled_and_edgeless_seeds_spread():
    edges = {f"e{i}": _u32() for i in range(6)}
    labels = partition(edges, k=3, strip_frac=0.5)
    assert set(labels) == set(edges)
    assert sorted(np.bincount(list(labels.values())).tolist()) == [2, 2, 2]


def test_deterministic():
    edges = _book(7)
    assert partition(edges, 3, 0.5) == partition(edges, 3, 0.5)


def test_universal_edges_do_not_count():
    edges = {k: _u32(1, 2) for k in "abcd"}
    assert cut_weight(edges, {k: i % 2 for i, k in enumerate(edges)}, 0.5) == (0, 0)


def test_bad_arguments():
    with pytest.raises(ValueError):
        partition({"a": _u32(1)}, k=1, strip_frac=0.5)
    with pytest.raises(ValueError):
        partition({"a": _u32(1)}, k=2, strip_frac=0.0)
    with pytest.raises(ValueError):
        partition({"a": _u32(1)}, k=2, strip_frac=0.5, slack=-0.1)


def test_empty_book():
    assert partition({}, k=2, strip_frac=0.5) == {}
    assert "No seeds" in seed_cut_report({}, k=2, strip_frac=0.5)


def test_report_text():
    text = seed_cut_report(_book(2), k=3, strip_frac=0.5)
    assert "Seed Cut" in text
    assert "Cut weight" in text
    assert "random" in text.lower()
    assert "Group sizes" in text


def test_minimize_flag_registered(monkeypatch, capsys):
    import sys

    from fuzzer_tool.cli import commands

    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "minimize", "--help"])
    with pytest.raises(SystemExit):
        commands.main()
    assert "--seed-cut" in capsys.readouterr().out
