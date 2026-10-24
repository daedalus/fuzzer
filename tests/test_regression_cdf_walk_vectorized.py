"""``EdgeTracker._cdf_walk`` in numpy: bit-identical to the merged-support loop.

The corpus hit-count profile has one support point per distinct mean count
(~88k on ffmpeg), and the loop sorted and walked it per Wasserstein call:
~9 ms per seed in ``_compute_weights``. ``np.cumsum`` accumulates left to
right like the loop, so results compare with ``==``, not a tolerance.
"""

import math

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.rand_pool import RandPool


def _reference(prof_a: dict, prof_b: dict) -> tuple:
    # The pre-change loop, verbatim.
    xs = sorted(set(prof_a) | set(prof_b))
    if not xs:
        return 0.0, 0.0, 0.0
    cdf_diff = wasserstein = ks = crps = 0.0
    prev = xs[0]
    for x in xs:
        gap = x - prev
        abs_diff = abs(cdf_diff)
        wasserstein += abs_diff * gap
        ks = max(ks, abs_diff)
        crps += cdf_diff * cdf_diff * gap
        cdf_diff += prof_a.get(x, 0.0) - prof_b.get(x, 0.0)
        prev = x
    ks = max(ks, abs(cdf_diff))
    return wasserstein, ks, crps


def _profile(rng: RandPool, n: int, top: int) -> dict:
    # Real shape: log2(1 + count) support, equal mass per observation.
    prof: dict[float, float] = {}
    w = 1.0 / n
    for count in rng.randint_list(0, top, n):
        x = math.log2(1.0 + count)
        prof[x] = prof.get(x, 0.0) + w
    return prof


def test_regression_cdf_walk_control():
    # Control (Hard Rule 46).
    rng = RandPool(seed=1)
    a, b = _profile(rng, 300, 50), _profile(rng, 900, 400)
    assert _reference(a, b) == _reference(a, b)


def test_regression_cdf_walk_bit_identical():
    for seed in range(8):
        rng = RandPool(seed=seed)
        a = _profile(rng, rng.randint(1, 500), rng.randint(1, 3000))
        b = _profile(rng, rng.randint(1, 5000), rng.randint(1, 3000))
        assert EdgeTracker._cdf_walk(a, b) == _reference(a, b), seed
        assert EdgeTracker._cdf_walk(b, a) == _reference(b, a), seed


def test_regression_cdf_walk_disjoint_and_identical():
    # Adversarial: no shared support; identical profiles (distance 0).
    a, b = {0.0: 0.5, 1.0: 0.5}, {2.0: 0.25, 3.0: 0.75}
    assert EdgeTracker._cdf_walk(a, b) == _reference(a, b)
    assert EdgeTracker._cdf_walk(a, dict(a)) == _reference(a, a) == (0.0, 0.0, 0.0)


def test_regression_cdf_walk_empty_sides():
    # Falsification: an empty side still walks the other's support.
    a = {0.0: 0.2, 4.5: 0.8}
    assert EdgeTracker._cdf_walk(a, {}) == _reference(a, {})
    assert EdgeTracker._cdf_walk({}, a) == _reference({}, a)
    assert EdgeTracker._cdf_walk({}, {}) == (0.0, 0.0, 0.0)


def test_regression_cdf_walk_returns_floats():
    out = EdgeTracker._cdf_walk({1.0: 1.0}, {2.0: 1.0})
    assert all(type(v) is float for v in out)


def _tracker(seed: int) -> EdgeTracker:
    rng = RandPool(seed=seed)
    et = EdgeTracker()
    for i in range(30):
        edges = set(rng.randint_list(1, 2000, rng.randint(5, 300)))
        et.record_edges(f"s{i}", edges, hit_counts={e: rng.randint(1, 900) for e in edges})
    return et


def test_regression_aggregate_norms_match_loop():
    # The cached, sorted corpus side gives the loop's (wasserstein, ks).
    et = _tracker(3)
    corpus = et._corpus_hitcount_profile()
    for key in ("s0", "s7", "s29"):
        hc = et.seed_hit_counts[key]
        want = _reference(et._hitcount_profile(hc), corpus)[:2]
        assert et._aggregate_norms(hc) == want, key


def test_regression_corpus_sorted_follows_profile():
    # Adversarial: a record invalidates the profile; the sorted arrays follow.
    et = _tracker(4)
    hc = et.seed_hit_counts["s1"]
    et._aggregate_norms(hc)
    et.record_edges("new", {5000, 5001}, hit_counts={5000: 7000, 5001: 1})
    corpus = et._corpus_hitcount_profile()
    assert et._aggregate_norms(hc) == _reference(et._hitcount_profile(hc), corpus)[:2]
    assert et._corpus_sorted[0] is corpus
