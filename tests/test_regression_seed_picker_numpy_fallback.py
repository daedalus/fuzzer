"""Regression: a numpy ImportError must not disable seed weighting.

The metadata extraction pass lived inside the ``try: import numpy`` block in
``_compute_weights``, so an ImportError left ``has_meta`` all-False. The
phase-2 loop skips any seed without metadata, so it did nothing: every weight
stayed at its 1.0 initial value, ``seed_keys`` stayed all-None, and seed
selection silently became uniform random. numpy is a hard dependency, so this
path is defensive only -- but a defensive path that turns off the scheduler
without a symptom is worse than none.

The fallback computes the same scalar terms through ``_weight_exploit_parts``,
the helper the vector math mirrors.
"""

import builtins
import time
import types

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.seed_picker import SeedPicker

SEEDS = [b"seed-aaa", b"seed-bbb", b"seed-ccc"]


class _Fuzzer:
    def __init__(self, now: float):
        self.corpus = list(SEEDS)
        self.seed_meta = {
            SEEDS[0]: {"fuzz_count": 1, "coverage_edges": 50, "added_at": now, "momentum": 0.0},
            SEEDS[1]: {"fuzz_count": 40, "coverage_edges": 1, "added_at": now - 600, "momentum": 0.0},
            SEEDS[2]: {"fuzz_count": 4, "coverage_edges": 10, "added_at": now - 60, "momentum": 0.5},
        }
        self._edge_tracker = EdgeTracker()
        self._edge_tracker.good_turing_estimate = lambda: {"saturation": 0.0}
        self.exec_count = 0
        self._last_new_edge_exec = 0
        self._cached_weights: dict = {}
        self._temperature = 1.0
        self._secretary = None
        self._seed_secretary: dict = {}
        self._distance = None
        self._profile = types.SimpleNamespace(format_signature=None, hot_functions=None)
        self._use_lineage = False
        self._use_lineage_backtrack = False
        self._lineage = None
        self.markov_trained = False
        self._ablation_file = None
        self.max_len = 4096

    def _seed_key(self, data: bytes) -> str:
        return data.decode()


@pytest.fixture
def no_numpy(monkeypatch):
    """Make `import numpy` inside _compute_weights raise, nothing else."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("numpy disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _picker(f) -> SeedPicker:
    """Phase-2 modifiers stubbed to identity: these tests assert Phase 1,
    which is the pass the numpy branch owns."""
    sp = SeedPicker(f)
    sp._weight_secretary_and_cached = lambda sk, w, classifications, ff: (w, 1.0, 1.0)
    sp._weight_edge_penalties = lambda sk, w, fuzz_count, ff, recent_counts=None: w
    sp._weight_entropy_and_distance = lambda seed, sk, meta, w, ff, em, me, md: w
    sp._weight_static_features = lambda seed, cov, w, ff: w
    sp._weight_length_and_cross_target = lambda seed, meta, w, ff: w
    sp._weight_overlap_density = lambda sk, w, ff: w
    sp._weight_validity = lambda meta, w, ff: w
    sp._weight_lineage_backtrack = lambda sk, w, fuzz_count, ff, bt: w
    return sp


def _weights(f, now: float | None = None):
    return _picker(f)._compute_weights(time.time() if now is None else now)


class TestScalarFallback:
    def test_weights_are_not_all_uniform(self, no_numpy):
        f = _Fuzzer(time.time())
        w = _weights(f)
        assert len(w) == len(SEEDS)
        # Three seeds with three different (fuzz_count, coverage, age) triples
        # must get three different weights. When Phase 1 is skipped they all
        # keep the 1.0 initial value and only the Pareto pass touches them,
        # which can produce at most the two values {0.5, 1.0}.
        assert len(set(w)) == len(SEEDS), "Phase 1 never ran: weights are untouched"

    def test_weights_are_not_the_untouched_initial_value(self, no_numpy):
        f = _Fuzzer(time.time())
        w = _weights(f)
        assert not set(w) <= {0.5, 1.0}, "weights are the initial value scaled by Pareto only"

    def test_fresh_high_coverage_seed_outranks_the_stale_one(self, no_numpy):
        f = _Fuzzer(time.time())
        w = _weights(f)
        assert w[0] > w[1]

    def test_matches_the_numpy_path(self):
        """The scalar branch and the vector branch must agree. Compared with
        a frozen clock so age terms are identical across the two runs."""
        now = time.time()
        vector = _weights(_Fuzzer(now), now)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy disabled")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            scalar = _weights(_Fuzzer(now), now)
        finally:
            builtins.__import__ = real_import

        assert len(scalar) == len(vector)
        for s, v in zip(scalar, vector, strict=True):
            assert s == pytest.approx(v, rel=1e-9)

    def test_momentum_reaches_the_scalar_branch(self, no_numpy):
        now = time.time()
        hot = _Fuzzer(now)
        cold = _Fuzzer(now)
        cold.seed_meta[SEEDS[2]]["momentum"] = 0.0
        assert _weights(hot)[2] > _weights(cold)[2]

    def test_empty_corpus_is_handled(self, no_numpy):
        f = _Fuzzer(time.time())
        f.corpus = []
        f.seed_meta = {}
        assert _weights(f) == []

    def test_seed_without_metadata_is_treated_the_same_as_under_numpy(self):
        """A seed with no metadata is skipped by both branches, so its weight
        must come out identical either way (the Pareto pass still scales it)."""
        now = time.time()

        def _with_extra():
            f = _Fuzzer(now)
            f.corpus.append(b"unregistered")
            return f

        vector = _weights(_with_extra(), now)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy disabled")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            scalar = _weights(_with_extra(), now)
        finally:
            builtins.__import__ = real_import
        assert scalar[-1] == pytest.approx(vector[-1], rel=1e-9)
