"""compute_average_jaccard must agree with the pairwise definition exactly.

The vectorized path was rewritten from an (n, n, p) broadcast to the
pair-counting identity

    mean_{i<j} (1/p) sum_k [s_ik == s_jk]
        == (1 / (p * C(n,2))) * sum_k sum_v C(m_kv, 2)

which is an algebraic rearrangement, not an approximation. These tests use
the naive double loop as the oracle and require exact agreement, so a future
rewrite that quietly turns the metric into a sample or an estimate fails
here rather than drifting the number the stats line prints.

The threshold matters: the vectorized branch only fires above 20 signatures,
so every case below is sized past it.
"""

from array import array

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker

np = pytest.importorskip("numpy")


def _oracle(tracker) -> float:
    """Average pairwise Jaccard by the literal definition."""
    keys = list(tracker._minhash.signatures.keys())
    n = len(keys)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += tracker._minhash.approximate_jaccard(keys[i], keys[j])
            count += 1
    return total / count if count else 0.0


def _tracker_with(sigs: list[list[int]]) -> EdgeTracker:
    tracker = EdgeTracker()
    for idx, sig in enumerate(sigs):
        tracker._minhash.signatures[f"seed{idx}"] = array("Q", sig)
    tracker._minhash.num_perm = len(sigs[0])
    return tracker


def test_matches_pairwise_oracle_on_random_signatures():
    rng = np.random.default_rng(0)
    # Small value alphabet so collisions actually occur; with 64-bit random
    # values every multiplicity would be 1 and the identity would be tested
    # only at its trivial point.
    sigs = rng.integers(0, 40, size=(64, 32)).tolist()
    tracker = _tracker_with(sigs)
    assert tracker.compute_average_jaccard() == pytest.approx(_oracle(tracker), abs=1e-12)


def test_all_identical_signatures_give_one():
    sigs = [[7, 11, 13, 17]] * 25
    tracker = _tracker_with(sigs)
    assert tracker.compute_average_jaccard() == pytest.approx(1.0)


def test_all_disjoint_signatures_give_zero():
    # Every seed unique in every column: no pair agrees anywhere.
    sigs = [[i * 4 + c for c in range(4)] for i in range(25)]
    tracker = _tracker_with(sigs)
    assert tracker.compute_average_jaccard() == pytest.approx(0.0)


def test_two_tight_clusters():
    # 15 copies of A and 15 of B, disjoint. Agreeing pairs are the two
    # within-cluster blocks: 2 * C(15,2) out of C(30,2).
    sigs = [[1, 2, 3, 4]] * 15 + [[5, 6, 7, 8]] * 15
    tracker = _tracker_with(sigs)
    expected = 2 * (15 * 14 // 2) / (30 * 29 // 2)
    assert tracker.compute_average_jaccard() == pytest.approx(expected)
    assert tracker.compute_average_jaccard() == pytest.approx(_oracle(tracker), abs=1e-12)


def test_partial_column_agreement():
    rng = np.random.default_rng(7)
    sigs = rng.integers(0, 3, size=(40, 16)).tolist()
    tracker = _tracker_with(sigs)
    got = tracker.compute_average_jaccard()
    assert 0.0 < got < 1.0
    assert got == pytest.approx(_oracle(tracker), abs=1e-12)


def test_vectorized_and_scalar_paths_agree_across_the_threshold():
    """n=21 takes the vectorized branch, n=20 the pure-Python one."""
    rng = np.random.default_rng(13)
    base = rng.integers(0, 5, size=(21, 8)).tolist()
    big = _tracker_with(base)
    small = _tracker_with(base[:20])
    assert big.compute_average_jaccard() == pytest.approx(_oracle(big), abs=1e-12)
    assert small.compute_average_jaccard() == pytest.approx(_oracle(small), abs=1e-12)


def test_fewer_than_two_seeds_is_zero():
    tracker = EdgeTracker()
    assert tracker.compute_average_jaccard() == 0.0
    tracker._minhash.signatures["only"] = array("Q", [1, 2, 3])
    assert tracker.compute_average_jaccard() == 0.0
