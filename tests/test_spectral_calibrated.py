"""Null calibration applied to the spectral score.

The raw spectral score keeps a length bias (Spearman -0.99 against seed
length on single-distribution corpora), so it gets the same treatment as
the plug-in KL: ``clip((F - mean0(n)) / sd0(n), 0, Z_CAP)`` with the null
mean and spread of F for an n-byte sample drawn from the pool. F has no
closed-form null, so both come from fixed-seed Monte-Carlo (Hard Rule 39:
deterministic, no retry-until-hit).

Oracle: at n = 1 the sample is one byte, so the null is an exact
enumeration over the 256 one-hot rows weighted by q -- computed here
through the (separately tested) row scorer, not through the module's null
curve. Control (Rule 46): the *uncalibrated* spectral score must still show
the bias, or the regression test could not fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.schedulers.seed_entropy_kl import Z_CAP, EntropyKLSeedStrategy
from fuzzer_tool.core.spectral_kl import (
    nibble_features,
    null_spectral_curve,
    spectral_kl_rows_bits,
)
from tests.test_entropy_kl_miller_madow import _mixed_corpus, auc
from tests.test_regression_entropy_kl_length_bias import (
    SEED,
    _null_corpus,
    _spearman,
    _zipf,
)


class TestNullCurve:
    def test_n1_matches_exact_enumeration(self):
        rng = np.random.default_rng(SEED)
        q = _zipf(rng)
        phi = nibble_features()
        f_bits = spectral_kl_rows_bits(np.eye(256), q, phi)
        mean = float(np.dot(q, f_bits))
        sd = math.sqrt(float(np.dot(q, (f_bits - mean) ** 2)))

        got_mean, got_sd = null_spectral_curve(q, [1], draws=6000, seed=1)
        assert got_mean[0] == pytest.approx(mean, rel=0.06)
        assert got_sd[0] == pytest.approx(sd, rel=0.06)

    def test_control_same_seed_reproduces_exactly(self):
        q = _zipf(np.random.default_rng(SEED))
        a = null_spectral_curve(q, [4, 64], draws=50, seed=7)
        b = null_spectral_curve(q, [4, 64], draws=50, seed=7)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_mean_falls_with_sample_length(self):
        q = _zipf(np.random.default_rng(SEED))
        mean, _ = null_spectral_curve(q, [8, 64, 512, 4096], draws=100, seed=3)
        assert list(mean) == sorted(mean, reverse=True) and mean[-1] >= 0.0


class TestCalibratedSpectral:
    def test_control_raw_spectral_still_tracks_length(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        lengths = [len(s) for s in seeds]
        assert _spearman(lengths, EntropyKLSeedStrategy(None).spectral_scores(seeds)) < -0.8

    def test_regression_calibrated_spectral_uncorrelated_with_length(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        lengths = [len(s) for s in seeds]
        strat = EntropyKLSeedStrategy(None)
        assert abs(_spearman(lengths, strat.calibrated_spectral_scores(seeds))) < 0.3

    def test_short_seeds_do_not_dominate_selection_weight(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        short = np.array([len(s) <= 64 for s in seeds])
        w = np.asarray(EntropyKLSeedStrategy(None).calibrated_spectral_scores(seeds)) + 1e-6
        assert w[short].sum() / w.sum() < 1.5 * short.mean()

    def test_auc_beats_uncalibrated_spectral(self):
        raw, cal = [], []
        for trial in range(4):
            seeds, labels = _mixed_corpus(trial)
            strat = EntropyKLSeedStrategy(None)
            raw.append(auc(strat.spectral_scores(seeds), labels))
            cal.append(auc(strat.calibrated_spectral_scores(seeds), labels))
        assert np.mean(cal) > np.mean(raw) + 0.05


class TestAdversarial:
    def test_degenerate_inputs_stay_finite_bounded_and_empty_is_zero(self):
        corpus = [b"", b"\x00", b"\xff", bytes(4096), bytes(5000), b"ab" * 40]
        scores = EntropyKLSeedStrategy(None).calibrated_spectral_scores(corpus)
        assert all(math.isfinite(s) and 0.0 <= s <= Z_CAP for s in scores)
        assert scores[0] == 0.0

    def test_empty_corpus(self):
        assert EntropyKLSeedStrategy(None).calibrated_spectral_scores([]) == []

    def test_follows_a_large_pool_shift(self):
        rng = np.random.default_rng(SEED)
        first = _null_corpus(rng, _zipf(rng), count=40)
        second = _null_corpus(rng, _zipf(rng), count=40)
        live = EntropyKLSeedStrategy(None)
        live.calibrated_spectral_scores(first)
        got = live.calibrated_spectral_scores(second)
        want = EntropyKLSeedStrategy(None).calibrated_spectral_scores(second)
        assert got == pytest.approx(want, abs=1e-9)

    def test_outlier_is_capped(self):
        rng = np.random.default_rng(SEED)
        nulls = _null_corpus(rng, _zipf(rng), count=60)
        scores = EntropyKLSeedStrategy(None).calibrated_spectral_scores(
            [*nulls, bytes([0xFF]) * 4096]
        )
        assert scores[-1] == Z_CAP
