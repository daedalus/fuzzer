"""Regression tests for the ground metric of the edge-distance family.

Edge ids come out of ``caller_ctx ^ prev_loc ^ cur_loc`` (afl_shim.c), and the
ptrace backend hashes addresses the same way, so |e_a - e_b| describes the XOR
rather than the program. The Wasserstein/KS/CRPS family and coverage proximity
are defined on the log2 hit-count axis and on the coverage clock instead.
"""

import random

from fuzzer_tool.core.edge_tracker import EdgeTracker


def _xor_corpus(n_seeds=30, seed=7):
    """A tracker populated with realistically hashed edge ids."""
    rng = random.Random(seed)
    locs = [rng.getrandbits(16) for _ in range(400)]
    et = EdgeTracker()
    keys = []
    for i in range(n_seeds):
        key = f"s{i}"
        keys.append(key)
        edges = {(rng.choice(locs) ^ rng.choice(locs)) | 1 for _ in range(rng.randint(20, 120))}
        loud = i % 5 == 0
        hits = {e: (rng.choice([200, 400, 800]) if loud else rng.choice([1, 2, 3])) for e in edges}
        et.record_edges(key, edges, hit_counts=hits)
    return et, keys


def test_relabelling_edge_ids_does_not_move_distances():
    """The metric must not depend on the hash. Permuting every edge id is a
    relabelling of the same coverage; the old id-axis distance changed."""
    et_a = EdgeTracker()
    et_b = EdgeTracker()
    profiles = {"x": {1: 4, 2: 90, 3: 4}, "y": {1: 5, 2: 5, 3: 900}}
    remap = {1: 51234, 2: 7, 3: 60001}
    for key, hc in profiles.items():
        et_a.record_edges(key, set(hc), hit_counts=hc)
        et_b.record_edges(
            key, {remap[e] for e in hc}, hit_counts={remap[e]: c for e, c in hc.items()}
        )
    assert et_a.compute_wasserstein_distance("x", "y") == et_b.compute_wasserstein_distance(
        "x", "y"
    )
    assert et_a.compute_ks_distance("x", "y") == et_b.compute_ks_distance("x", "y")


def test_wasserstein_weight_separates_loop_heavy_seeds():
    """The stated purpose of this family: a seed hitting a loop 500 times is
    behaviourally distinct from one hitting it 5 times. On the id axis the
    whole corpus landed in a 1.2x band and the ordering tracked the hash."""
    et, keys = _xor_corpus()
    weights = {k: et.compute_wasserstein_weight(k) for k in keys}
    loud = [weights[f"s{i}"] for i in range(30) if i % 5 == 0]
    quiet = [weights[f"s{i}"] for i in range(30) if i % 5]
    assert sum(loud) / len(loud) > 1.5 * (sum(quiet) / len(quiet))
    assert all(0.5 <= w <= 2.0 for w in weights.values())


def test_coverage_proximity_is_not_constant():
    """Measured over this corpus the id-space version returned exactly 1.0 for
    every seed, making ``w *= 0.5 + cov`` a uniform 1.5x."""
    et, keys = _xor_corpus()
    values = {et.compute_coverage_proximity(k) for k in keys}
    assert len(values) > 1
    assert all(0.0 <= v <= 1.0 for v in values)


def test_coverage_proximity_tracks_the_discovery_frontier():
    et = EdgeTracker()
    for i in range(20):
        et.record_edges(f"old{i}", {i})
    # A seed made entirely of edges discovered last.
    et.record_edges("frontier", {900, 901, 902})
    assert et.compute_coverage_proximity("frontier") == 1.0
    assert et.compute_coverage_proximity("old0") == 0.0


def test_proximity_radius_argument_is_accepted_and_ignored():
    """Kept for call compatibility; it described the old id-space window."""
    et, keys = _xor_corpus(n_seeds=6)
    assert et.compute_coverage_proximity(keys[0], radius=5) == et.compute_coverage_proximity(
        keys[0], radius=500
    )


def test_unknown_seed_is_neutral():
    et, _ = _xor_corpus(n_seeds=4)
    assert et.compute_coverage_proximity("nope") == 0.5
    assert et.compute_wasserstein_weight("nope") == 1.0
