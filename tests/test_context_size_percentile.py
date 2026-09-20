"""The contextual schedulers' corpus-size percentile feature.

Two defects are pinned here, one of wiring and one of mathematics:

  1. ``_corpus_log_size_stats`` (formerly ``_corpus_size_stats``) was
     created in ``Fuzzer.__init__`` and read in ``_context_vector``, but
     nothing ever called ``update()`` on it -- its comment described what
     ``corpus_manager``'s *other* moments object (``_seed_size_moments``)
     does.  The ``count >= 5`` guard therefore never passed and the
     feature was the constant 0.5 for every execution of every run.

  2. The feature squashed a raw-byte z-score through ``1/(1+exp(-z))``.
     That is not the normal CDF (they differ by up to 0.117), and the
     raw-byte axis is the wrong axis for a right-skewed quantity anyway.

Every test below fails on the pre-fix implementation.
"""

from __future__ import annotations

import bisect
import math
import random
import statistics

import pytest

from fuzzer_tool.core.gaussian import norm_cdf
from fuzzer_tool.core.running_stats import RunningMoments


class TestNormCdf:
    def test_matches_known_values(self):
        assert norm_cdf(0.0) == pytest.approx(0.5)
        assert norm_cdf(1.0) == pytest.approx(0.8413447461, abs=1e-9)
        assert norm_cdf(-1.96) == pytest.approx(0.0249978952, abs=1e-9)
        assert norm_cdf(2.576) == pytest.approx(0.9950030, abs=1e-6)

    def test_loc_and_scale(self):
        assert norm_cdf(7.0, loc=7.0, scale=3.0) == pytest.approx(0.5)
        assert norm_cdf(10.0, loc=7.0, scale=3.0) == pytest.approx(norm_cdf(1.0))

    def test_degenerate_scale_is_a_step_not_a_raise(self):
        # A corpus of identical seed sizes reaches this legitimately.
        assert norm_cdf(1.0, loc=0.0, scale=0.0) == 1.0
        assert norm_cdf(-1.0, loc=0.0, scale=0.0) == 0.0
        assert norm_cdf(0.0, loc=0.0, scale=0.0) == 1.0

    def test_beats_the_unscaled_logistic_it_replaced(self):
        """The old surrogate is off by >0.1 where Phi is off by ~0."""
        worst = max(
            abs(1.0 / (1.0 + math.exp(-z / 100.0)) - norm_cdf(z / 100.0))
            for z in range(-600, 601)
        )
        assert worst > 0.11


class TestLogAxisPercentile:
    """Phi on log sizes recovers the true percentile; the old form does not."""

    @staticmethod
    def _corpus(n=4000, seed=7):
        rng = random.Random(seed)
        return [math.exp(rng.gauss(7.0, 1.0)) for _ in range(n)]

    def test_log_axis_tracks_the_empirical_percentile(self):
        sizes = self._corpus()
        srt = sorted(sizes)
        moments = RunningMoments()
        for s in sizes:
            moments.update(math.log1p(s))

        errs = [
            abs(
                norm_cdf(math.log1p(x), moments.mean, moments.stddev)
                - bisect.bisect_left(srt, x) / len(srt)
            )
            for x in sizes[::7]
        ]
        assert statistics.mean(errs) < 0.01
        assert max(errs) < 0.05

    def test_raw_axis_forms_are_an_order_of_magnitude_worse(self):
        sizes = self._corpus()
        srt = sorted(sizes)
        mean = statistics.mean(sizes)
        sd = statistics.pstdev(sizes)

        def err(fn):
            return statistics.mean(
                abs(fn(x) - bisect.bisect_left(srt, x) / len(srt)) for x in sizes[::7]
            )

        logistic_raw = err(lambda x: 1.0 / (1.0 + math.exp(-((x - mean) / sd))))
        phi_raw = err(lambda x: norm_cdf(x, mean, sd))
        assert logistic_raw > 0.10
        assert phi_raw > 0.05

    def test_uses_the_full_unit_range(self):
        """A true percentile is uniform on [0, 1] -- stddev 1/sqrt(12)."""
        sizes = self._corpus()
        moments = RunningMoments()
        for s in sizes:
            moments.update(math.log1p(s))
        vals = [norm_cdf(math.log1p(x), moments.mean, moments.stddev) for x in sizes]
        assert statistics.pstdev(vals) == pytest.approx(1 / math.sqrt(12), abs=0.02)


_PCTILE_INDEX = 5  # log_size, entropy, edge_frac, lineage, cmplog, PCTILE


def _engine_with_corpus(sizes):
    """OperatorEngine over a stub whose log-size moments have seen *sizes*."""
    from fuzzer_tool.services.operators import OperatorEngine

    class _Stub:
        max_len = 1 << 20
        seed_meta: dict = {}
        _edge_tracker = None
        _cmplog = None
        _op_time_ema: dict = {}
        _corpus_log_size_stats = RunningMoments()

    stub = _Stub()
    for s in sizes:
        stub._corpus_log_size_stats.update(math.log1p(s))
    return OperatorEngine(stub), stub


class TestFeatureIsLiveAndCalibrated:
    """The wiring half, exercised through the real feature builder."""

    @staticmethod
    def _sizes(n=500, seed=3):
        rng = random.Random(seed)
        return [int(math.exp(rng.gauss(7.0, 1.0))) + 1 for _ in range(n)]

    def test_feature_is_neutral_before_five_seeds(self):
        engine, _ = _engine_with_corpus([100, 200, 300])
        vec = engine._build_shared_context(b"x" * 150)
        assert vec[_PCTILE_INDEX] == 0.5

    def test_feature_leaves_the_neutral_default_once_fed(self):
        """Pre-fix this stayed 0.5 forever: nothing updated the moments."""
        sizes = self._sizes()
        engine, _ = _engine_with_corpus(sizes)
        small = engine._build_shared_context(b"x" * 40)[_PCTILE_INDEX]
        large = engine._build_shared_context(b"x" * 40000)[_PCTILE_INDEX]
        assert small < 0.05
        assert large > 0.95

    def test_feature_matches_the_empirical_percentile(self):
        sizes = self._sizes(n=3000)
        srt = sorted(sizes)
        engine, _ = _engine_with_corpus(sizes)
        errs = []
        for x in srt[::37]:
            got = engine._build_shared_context(b"x" * x)[_PCTILE_INDEX]
            errs.append(abs(got - bisect.bisect_left(srt, x) / len(srt)))
        assert statistics.mean(errs) < 0.02
        assert max(errs) < 0.06

    def test_feature_spans_the_unit_interval(self):
        """The old logistic-on-raw-bytes form bottomed out around 0.3."""
        sizes = self._sizes(n=2000)
        engine, _ = _engine_with_corpus(sizes)
        vals = [engine._build_shared_context(b"x" * x)[_PCTILE_INDEX] for x in sizes]
        assert min(vals) < 0.02
        assert max(vals) > 0.98
        assert statistics.pstdev(vals) == pytest.approx(1 / math.sqrt(12), abs=0.03)

    def test_identical_seed_sizes_do_not_blow_up(self):
        """stddev == 0 is a legitimate state, not an error."""
        engine, _ = _engine_with_corpus([512] * 20)
        vec = engine._build_shared_context(b"x" * 512)
        assert 0.0 <= vec[_PCTILE_INDEX] <= 1.0


class TestCorpusManagerFeedsTheMoments:
    def test_save_to_corpus_updates_the_object_the_feature_reads(self):
        """The twin-object defect: written here, read in operators.py."""
        import inspect

        from fuzzer_tool.services import corpus_manager
        from fuzzer_tool.services import fuzzer as fuzzer_mod
        from fuzzer_tool.services import operators as operators_mod

        written = inspect.getsource(corpus_manager)
        created = inspect.getsource(fuzzer_mod)
        read = inspect.getsource(operators_mod)
        assert "log_size_moments.update(math.log1p(len(data)))" in written
        assert "self._corpus_log_size_stats = RunningMoments()" in created
        assert 'getattr(f, "_corpus_log_size_stats", None)' in read
        # The old name must be gone from all three, or the defect (one
        # object written, a different one read) can reappear silently.
        for src in (written, created, read):
            assert "_corpus_size_stats" not in src
