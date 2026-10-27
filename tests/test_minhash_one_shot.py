"""compute_signature: the one-shot block and the banded path agree with the
pure-Python reference for every set size, including both sides of the cutoff."""

import random
from array import array

import pytest

from fuzzer_tool.core import edge_tracker as et
from fuzzer_tool.core.edge_tracker import MinHashLSH


def _reference(m: MinHashLSH, edge_set: set[int]) -> array:
    sig = array("Q", [m._mask]) * m.num_perm
    for edge in edge_set:
        for i, (a, b) in enumerate(m._coeffs):
            h = (a * edge + b) & m._mask
            if h < sig[i]:
                sig[i] = h
    return sig


@pytest.mark.parametrize(
    "n", [1, 2, 7, 64, 1023, et._MINHASH_ONE_SHOT_MAX, et._MINHASH_ONE_SHOT_MAX + 1]
)
def test_signature_matches_python_reference(n):
    rng = random.Random(n)
    m = MinHashLSH()
    edges = set(rng.sample(range(1 << 20), n))
    assert m.compute_signature(edges) == _reference(m, edges)


def test_empty_set_is_all_ones_mask():
    m = MinHashLSH()
    assert m.compute_signature(set()) == _reference(m, set())


def test_edge_ids_near_uint64_top_wrap_identically():
    m = MinHashLSH()
    edges = {(1 << 64) - 1, (1 << 63), (1 << 63) + 12345, 0, 1}
    assert m.compute_signature(edges) == _reference(m, edges)


def test_both_paths_agree_on_the_same_set(monkeypatch):
    rng = random.Random(99)
    edges = set(rng.sample(range(1 << 20), 300))
    m = MinHashLSH()
    one_shot = m.compute_signature(edges)
    monkeypatch.setattr(et, "_MINHASH_ONE_SHOT_MAX", 0)
    assert m.compute_signature(edges) == one_shot
