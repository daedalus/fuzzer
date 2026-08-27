"""Regression tests for the fused recent-overlap penalty in SeedPicker.

The penalty used to be computed as ``sum(len(seed_edges & recent) for recent
in f._recent_seed_edges)`` -- one set intersection per (seed, window slot),
rebuilt on every seed of every weight pass. It is now folded into the
owner-count pass over the same edges via a window occurrence map built once
per pass (``SeedPicker._recent_edge_counts``).

These tests pin the *equality* of the two formulations, since the change is
meant to be a pure refactor of a hot path: any drift here is a scheduling
change that would not otherwise show up as a failure.
"""

import random
import tempfile
from unittest.mock import patch

from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.seed_picker import SeedPicker


def _make_fuzzer():
    tmpdir = tempfile.mkdtemp(prefix="fuzz_overlap_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )


def _reference_overlap(seed_edges, window):
    """The pre-fusion expression, kept verbatim as the oracle."""
    return sum(len(seed_edges & recent) for recent in window)


def test_window_counts_are_incidences_not_unique_edges():
    """An edge in three window slots contributes three, not one.

    The overlap penalty measures how much recently-fuzzed ground a seed
    re-treads, so a slot that covers the same edge again counts again. A
    plain set union over the window would collapse those and under-penalise
    the most redundant seeds -- exactly the ones the penalty exists for.
    """
    f = _make_fuzzer()
    f._recent_seed_edges = [{1, 2}, {2, 3}, {2, 9}]
    counts = SeedPicker._recent_edge_counts(f)
    assert counts[2] == 3
    assert counts[1] == 1
    assert counts.get(42, 0) == 0
    assert sum(counts.get(e, 0) for e in {1, 2, 3}) == _reference_overlap(
        {1, 2, 3}, f._recent_seed_edges
    )


def test_absent_or_empty_window_yields_no_counts():
    """None stands in for the old ``hasattr`` guard and for an empty window."""
    f = _make_fuzzer()
    assert SeedPicker._recent_edge_counts(f) is None
    f._recent_seed_edges = []
    assert SeedPicker._recent_edge_counts(f) is None


def test_fused_weight_matches_reference_over_random_corpora():
    """Fused penalty reproduces the intersection-sum formulation exactly.

    Random shapes rather than one hand-built case: the fusion moved the
    counting earlier in the function but left the multiplications in place,
    so the two must agree bit for bit, including on seeds that trip the
    crowding branch and seeds the window never touches.
    """
    rng = random.Random(20260827)
    for trial in range(25):
        f = _make_fuzzer()
        et = f._edge_tracker
        universe = list(range(1, 60))
        keys = []
        for i in range(rng.randint(3, 12)):
            key = f"s{trial}_{i}"
            keys.append(key)
            et.record_edges(key, set(rng.sample(universe, rng.randint(1, 25))))
        window = [et.seed_edges[rng.choice(keys)] for _ in range(rng.randint(0, 20))]
        f._recent_seed_edges = window
        picker = SeedPicker(f)
        counts = picker._recent_edge_counts(f)
        for key in keys:
            fuzz_count = rng.randint(1, 30)
            fused = picker._weight_edge_penalties(key, 1.0, fuzz_count, f, counts)
            # Self-derived path: callers outside _compute_weights pass nothing.
            derived = picker._weight_edge_penalties(key, 1.0, fuzz_count, f)
            assert fused == derived

            seed_edges = et.seed_edges[key]
            expected_overlap = _reference_overlap(seed_edges, window)
            actual_overlap = sum(counts.get(e, 0) for e in seed_edges) if counts else 0
            assert actual_overlap == expected_overlap


def test_penalty_still_fires_without_precomputed_counts():
    """A seed sitting entirely on recently-fuzzed edges is still discounted."""
    f = _make_fuzzer()
    et = f._edge_tracker
    et.record_edges("target", {1, 2, 3, 4})
    picker = SeedPicker(f)
    f._recent_seed_edges = []
    baseline = picker._weight_edge_penalties("target", 1.0, 1, f)
    f._recent_seed_edges = [{1, 2, 3, 4}, {1, 2, 3, 4}]
    penalised = picker._weight_edge_penalties("target", 1.0, 1, f)
    assert penalised < baseline
