"""``--edge-components``: connected components of B = A.A^T after stripping
near-universal edges (handover_coverage_gram_matrix section 6, item 4).

Two seeds are linked when they share a non-universal edge. Edges held by more
than ``strip_frac`` of the seeds link everything and are dropped first.

    a={1,10}  b={1,10,11}  c={1,20}  d={1,20,21}  e={1,30}  f={}
    edge 1 is held by 5 of 6 seeds -> stripped at strip_frac 0.5
    components: {a,b} {c,d} {e} {f}
"""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.edge_components import components, components_report


def _u32(*ids: int) -> np.ndarray:
    return np.asarray(ids, dtype=np.uint32)


BOOK = {
    "a": _u32(1, 10),
    "b": _u32(1, 10, 11),
    "c": _u32(1, 20),
    "d": _u32(1, 20, 21),
    "e": _u32(1, 30),
    "f": _u32(),
}


def test_strip_splits_clusters():
    labels, stripped = components(BOOK, strip_frac=0.5)
    assert stripped == 1
    assert labels["a"] == labels["b"]
    assert labels["c"] == labels["d"]
    assert len({labels["a"], labels["c"], labels["e"], labels["f"]}) == 4


def test_no_strip_merges_everything_with_edges():
    labels, stripped = components(BOOK, strip_frac=1.0)
    assert stripped == 0
    assert len({labels[k] for k in "abcde"}) == 1
    assert labels["f"] != labels["a"]


def test_chain_links_transitively():
    book = {"x": _u32(1, 2), "y": _u32(2, 3), "z": _u32(3, 4), "w": _u32(9)}
    labels, _ = components(book, strip_frac=1.0)
    assert labels["x"] == labels["y"] == labels["z"] != labels["w"]


def test_empty_book():
    labels, stripped = components({}, strip_frac=0.5)
    assert labels == {} and stripped == 0


def test_bad_strip_frac():
    with pytest.raises(ValueError):
        components(BOOK, strip_frac=0.0)
    with pytest.raises(ValueError):
        components(BOOK, strip_frac=1.5)


def _reference_partition(book: dict[str, np.ndarray]) -> set[frozenset[str]]:
    """Union-find over seeds sharing an edge (no stripping)."""
    parent = {k: k for k in book}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    owner: dict[int, str] = {}
    for k, e in book.items():
        for x in e.tolist():
            if x in owner:
                parent[find(k)] = find(owner[x])
            else:
                owner[x] = k
    return {frozenset(k for k in book if find(k) == r) for r in {find(k) for k in book}}


def test_matches_union_find_on_random_books():
    rng = np.random.default_rng(7)
    for _ in range(20):
        n, m = int(rng.integers(2, 40)), int(rng.integers(2, 60))
        book = {
            f"s{i}": np.unique(rng.integers(0, m, int(rng.integers(0, 6))).astype(np.uint32))
            for i in range(n)
        }
        labels, _ = components(book, strip_frac=1.0)
        got = {frozenset(k for k in book if labels[k] == lab) for lab in set(labels.values())}
        assert got == _reference_partition(book)


def test_report_lines():
    text = components_report(BOOK, strip_frac=0.5)
    assert "Edge Components" in text
    assert "Components:" in text and "4" in text
    assert "Stripped edges:" in text
    assert "Largest:" in text
    assert "Isolated seeds:" in text


def test_report_empty():
    assert "No seeds" in components_report({}, strip_frac=0.5)


def test_minimize_flag_registered(monkeypatch, capsys):
    import sys

    from fuzzer_tool.cli import commands

    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "minimize", "--help"])
    with pytest.raises(SystemExit):
        commands.main()
    assert "--edge-components" in capsys.readouterr().out
