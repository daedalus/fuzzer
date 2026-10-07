"""The dense-vector edge reductions in a weight pass match the scalar loop.

``_weight_edge_penalties`` reduces a seed's edge set against the owner counts
and the recent-pick window. Inside ``_compute_weights`` it now reads those
from ``_EdgeVectors`` -- one index array per seed set, three numpy gathers per
seed -- instead of looping over the set in Python. The change is meant to be
a pure refactor of the largest per-execution cost of a long-input FFmpeg
campaign, so these tests pin bit-for-bit equality with the scalar path,
which stays in place for calls outside a pass and serves as the oracle.
"""

import random
import tempfile
from unittest.mock import patch

import numpy as np

from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.seed_picker import SeedPicker, _EdgeVectors


def _make_fuzzer():
    tmpdir = tempfile.mkdtemp(prefix="fuzz_edgevec_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )


def _random_campaign(rng, trial):
    f = _make_fuzzer()
    et = f._edge_tracker
    universe = list(range(1, rng.choice([40, 400, 4000])))
    keys = []
    for i in range(rng.randint(3, 15)):
        key = f"s{trial}_{i}"
        keys.append(key)
        et.record_edges(key, set(rng.sample(universe, rng.randint(1, min(300, len(universe))))))
    f._recent_seed_edges = [et.seed_edges[rng.choice(keys)] for _ in range(rng.randint(0, 20))]
    return f, keys


def test_vector_stats_match_the_scalar_penalty():
    rng = random.Random(20261007)
    for trial in range(40):
        f, keys = _random_campaign(rng, trial)
        et = f._edge_tracker
        picker = SeedPicker(f)
        counts = picker._recent_edge_counts(f)
        fuzz = {k: rng.randint(1, 30) for k in keys}
        scalar = {k: picker._weight_edge_penalties(k, 1.0, fuzz[k], f, counts) for k in keys}

        vectors = _EdgeVectors()
        vectors.prepare(
            [et.seed_edges[k] for k in keys], et._edge_owner_count, f._recent_seed_edges
        )
        picker._pass_vectors = vectors
        try:
            vec = {k: picker._weight_edge_penalties(k, 1.0, fuzz[k], f, counts) for k in keys}
        finally:
            picker._pass_vectors = None
        assert vec == scalar


def test_compute_weights_is_unchanged_by_the_vector_path():
    """Whole weight vectors, vector pass vs scalar pass, frozen clock."""
    rng = random.Random(7)
    for trial in range(10):
        f, keys = _random_campaign(rng, trial)
        corpus = [k.encode() for k in keys]
        f.corpus = corpus
        for i, seed in enumerate(corpus):
            f.seed_meta[seed] = {
                "fuzz_count": rng.randint(1, 30),
                "coverage_edges": rng.randint(1, 300),
                "added_at": 1000.0 - i,
                "momentum": 0.0,
            }
        f._seed_key = lambda data: data.decode()
        picker = SeedPicker(f)
        f._cached_weights = {}
        vec = picker._compute_weights(1000.0)
        f._cached_weights = {}
        with patch.object(SeedPicker, "_prepare_edge_vectors", lambda *a, **k: None):
            scalar = picker._compute_weights(1000.0)
        assert len(set(vec)) > 1, "degenerate fixture: every weight equal"
        assert vec == scalar


def test_a_grown_set_is_reindexed():
    """Seed sets only grow in place; the cached array must follow."""
    v = _EdgeVectors()
    edges = {5, 9}
    first = v.indices(edges)
    edges.add(11)
    again = v.indices(edges)
    assert len(first) == 2 and len(again) == 3
    assert sorted(v._ids[i] for i in again) == [5, 9, 11]


def test_window_counts_are_incidences():
    v = _EdgeVectors()
    a, b = {1, 2}, {2, 3}
    v.prepare([a, b], {1: 1, 2: 2, 3: 1}, [a, b, b])
    idx = v.indices({2})
    assert int(v.recent[idx].sum()) == 3


def test_tables_follow_the_current_pass_only():
    """Arrays for sets that left the corpus and window are dropped."""
    v = _EdgeVectors()
    a, b = {1, 2}, {3}
    v.prepare([a, b], {}, [])
    v.prepare([a], {}, [])
    assert id(b) not in v._arrays
    assert isinstance(v.owners, np.ndarray) and v.recent is None
