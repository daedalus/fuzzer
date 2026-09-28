"""Bach's spectral KL estimator (core/spectral_kl.py).

Reference: https://francisbach.com/spectral_log_density_estimation/ (Eq. 7
and the generalized-eigenvalue form). The oracle integrates Eq. (7) over rho
by Gauss-Legendre quadrature with a linear solve per node -- a different
route from the module's one-shot eigendecomposition, so agreement is
evidence, not tautology. Draws are fixed-seed Generators (Hard Rule 39).

Falsification: F <= KL always (it is a lower bound), F = 0 at p = q.
Identity: one-hot features make F *equal* the plug-in KL. Adversarial:
singular / degenerate moments, unseen bins, empty rows.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.spectral_kl import (
    nibble_features,
    spectral_kl_nats,
    spectral_kl_rows_bits,
)

RNG_SEED = 20260927


def _moments(p: np.ndarray, phi: np.ndarray):
    return p @ phi, (phi.T * p) @ phi


def _kl_nats(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def _dirichlet(rng, k: int) -> np.ndarray:
    return rng.dirichlet(np.ones(k))


def _quadrature_reference(mu_p, sp, mu_q, sq, ridge: float) -> float:
    """Eq. (7) with d nu = 2 (1 - rho) d rho, one linear solve per node."""
    nodes, weights = np.polynomial.legendre.leggauss(400)
    rho, w = 0.5 * (nodes + 1.0), 0.5 * weights
    d = mu_p - mu_q
    eye = np.eye(len(d))
    total = 0.0
    for r, wt in zip(rho, w, strict=True):
        a = r * sp + (1.0 - r) * sq + ridge * eye
        total += wt * 0.5 * 2.0 * (1.0 - r) * float(d @ np.linalg.solve(a, d))
    return total


class TestEstimator:
    def test_matches_the_quadrature_of_equation_7(self):
        rng = np.random.default_rng(RNG_SEED)
        phi = nibble_features()
        p, q = _dirichlet(rng, 256), _dirichlet(rng, 256)
        mu_p, sp = _moments(p, phi)
        mu_q, sq = _moments(q, phi)
        got = spectral_kl_nats(mu_p, sp, mu_q, sq, ridge=1e-3)
        assert got == pytest.approx(_quadrature_reference(mu_p, sp, mu_q, sq, 1e-3), rel=1e-6)

    def test_control_quadrature_is_stable_against_itself(self):
        # Rule 46: the oracle must reproduce itself at a different order.
        rng = np.random.default_rng(RNG_SEED)
        phi = nibble_features()
        p, q = _dirichlet(rng, 256), _dirichlet(rng, 256)
        args = (*_moments(p, phi), *_moments(q, phi), 1e-3)
        first = _quadrature_reference(*args)
        assert first == pytest.approx(_quadrature_reference(*args), rel=1e-12)

    def test_one_hot_features_reduce_exactly_to_plug_in_kl(self):
        rng = np.random.default_rng(RNG_SEED)
        p, q = _dirichlet(rng, 8), _dirichlet(rng, 8)
        eye = np.eye(8)
        got = spectral_kl_nats(*_moments(p, eye), *_moments(q, eye), ridge=1e-12)
        assert got == pytest.approx(_kl_nats(p, q), rel=1e-6)

    def test_lower_bound_on_the_true_kl(self):
        rng = np.random.default_rng(RNG_SEED)
        phi = nibble_features()
        for _ in range(20):
            p, q = _dirichlet(rng, 256), _dirichlet(rng, 256)
            f = spectral_kl_nats(*_moments(p, phi), *_moments(q, phi), ridge=1e-9)
            assert f <= _kl_nats(p, q) + 1e-9

    def test_identical_distributions_give_zero(self):
        rng = np.random.default_rng(RNG_SEED)
        phi = nibble_features()
        p = _dirichlet(rng, 256)
        args = (*_moments(p, phi), *_moments(p, phi))
        assert spectral_kl_nats(*args, ridge=1e-3) == pytest.approx(0.0, abs=1e-12)

    def test_non_negative(self):
        rng = np.random.default_rng(RNG_SEED)
        phi = nibble_features()
        for _ in range(10):
            p, q = _dirichlet(rng, 256), _dirichlet(rng, 256)
            assert spectral_kl_nats(*_moments(p, phi), *_moments(q, phi), ridge=1e-3) >= 0.0


class TestRows:
    def test_batch_matches_the_per_row_estimator(self):
        rng = np.random.default_rng(RNG_SEED)
        phi = nibble_features()
        rows = np.stack([_dirichlet(rng, 256) for _ in range(5)])
        q = _dirichlet(rng, 256)
        mu_q, sq = _moments(q, phi)
        want = [
            spectral_kl_nats(*_moments(r, phi), mu_q, sq, ridge=1e-3) / math.log(2) for r in rows
        ]
        assert spectral_kl_rows_bits(rows, q, phi, ridge=1e-3).tolist() == pytest.approx(
            want, rel=1e-7, abs=1e-12
        )

    def test_empty_row_scores_zero(self):
        rng = np.random.default_rng(RNG_SEED)
        rows = np.zeros((2, 256))
        rows[1] = _dirichlet(rng, 256)
        out = spectral_kl_rows_bits(rows, _dirichlet(rng, 256), nibble_features(), ridge=1e-3)
        assert out[0] == 0.0 and out[1] > 0.0

    def test_pool_with_unseen_bins_stays_finite(self):
        # Smoothed pools carry ~1e-9 mass on bins never seen: Sigma_q is
        # near-singular, the ridge is what keeps this finite.
        q = np.full(256, 1e-9)
        q[:4] = (1.0 - 252e-9) / 4
        row = np.zeros((1, 256))
        row[0, 200:210] = 0.1
        out = spectral_kl_rows_bits(row, q, nibble_features(), ridge=1e-3)
        assert np.isfinite(out).all() and (out >= 0).all()

    def test_point_mass_pool_and_row_do_not_blow_up(self):
        q = np.full(256, 1e-12)
        q[0] = 1.0 - 255e-12
        rows = np.zeros((2, 256))
        rows[0, 0] = 1.0
        rows[1, 255] = 1.0
        out = spectral_kl_rows_bits(rows, q, nibble_features(), ridge=1e-3)
        assert np.isfinite(out).all() and (out >= 0).all()
        assert out[1] > out[0]


class TestFeatures:
    def test_nibble_features_are_two_one_hots(self):
        phi = nibble_features()
        assert phi.shape == (256, 32)
        assert (phi.sum(axis=1) == 2).all()
        assert phi[0x4A, 0x4] == 1 and phi[0x4A, 16 + 0xA] == 1


class TestStrategyBaseline:
    """spectral_scores() next to raw / Miller-Madow / calibrated, by AUC."""

    def test_matches_the_module_on_the_live_pool(self):
        from fuzzer_tool.core.schedulers.seed_entropy_kl import EntropyKLSeedStrategy

        seeds = [b"abcabcabc" * 3, b"aaaa" * 10, bytes(range(64)), b"the quick brown fox " * 2]
        strat = EntropyKLSeedStrategy(None)
        got = strat.spectral_scores(seeds)

        counts = np.zeros((len(seeds), 256))
        pool = np.zeros(256)
        for i, s in enumerate(seeds):
            for b in s:
                counts[i, b] += 1
                pool[b] += 1
        rows = counts / counts.sum(axis=1, keepdims=True)
        q = (pool + 1 / 256) / (pool.sum() + 1)
        want = spectral_kl_rows_bits(rows, q, nibble_features())
        assert got == pytest.approx(want.tolist(), rel=1e-9, abs=1e-12)

    def test_empty_corpus_and_empty_seed(self):
        from fuzzer_tool.core.schedulers.seed_entropy_kl import EntropyKLSeedStrategy

        assert EntropyKLSeedStrategy(None).spectral_scores([]) == []
        assert EntropyKLSeedStrategy(None).spectral_scores([b"", b"abc"])[0] == 0.0

    def test_auc_ranking_raw_lt_spectral_lt_calibrated(self):
        from fuzzer_tool.core.schedulers.seed_entropy_kl import EntropyKLSeedStrategy
        from tests.test_entropy_kl_miller_madow import _mixed_corpus, auc

        raw, spec, cal = [], [], []
        for trial in range(4):
            seeds, labels = _mixed_corpus(trial)
            strat = EntropyKLSeedStrategy(None)
            raw.append(auc(strat.raw_scores(seeds), labels))
            spec.append(auc(strat.spectral_scores(seeds), labels))
            cal.append(auc(strat.scores(seeds), labels))
        # Measured over 6 runs: raw 0.72, spectral 0.83, calibrated 0.97.
        assert np.mean(raw) < np.mean(spec) < np.mean(cal)

    def test_spectral_alone_does_not_remove_the_length_bias(self):
        # Pins the honest limit: with 32 shared features and a 1e-3 ridge the
        # score still tracks seed length on a single-distribution corpus, so
        # it is a baseline, not a replacement for the null calibration.
        from fuzzer_tool.core.schedulers.seed_entropy_kl import EntropyKLSeedStrategy
        from tests.test_regression_entropy_kl_length_bias import (
            SEED,
            _null_corpus,
            _spearman,
            _zipf,
        )

        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        lengths = [len(s) for s in seeds]
        strat = EntropyKLSeedStrategy(None)
        assert _spearman(lengths, strat.spectral_scores(seeds)) < -0.8
        assert abs(_spearman(lengths, strat.scores(seeds))) < 0.3
