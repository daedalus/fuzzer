"""Regression: ``entropy_zscore`` weights tracked seed length.

Plug-in byte entropy of an n-byte sample reads low by exactly the KL null
mean E0(n) (E[H(P_hat)] = H(q) - E[KL(P_hat || q)] for draws from q), and
it is far noisier for short samples. On corpora whose seeds all come from
one distribution the uncalibrated arm gave seeds <= 64 B about 0.62x their
uniform weight share (Spearman(weight, length) +0.10).

``calibrate_length=True`` adds the exact bias back and standardises each
seed by its own null spread: z_i = (H_i + E0(n_i) - mean) / sqrt(between^2 +
sd0(n_i)^2), between^2 = max(var - mean(sd0^2), 0). Adding the bias alone
made it worse (short seeds scatter into the tails): that is why the spread
is part of the fix.

Oracles are independent of the module: the n = 2 null is enumerated by hand
(E[H] = 1 - sum q^2 bits), draws come from fixed-seed Generators (Hard Rule
39), and the control (Rule 46) is the *uncalibrated* arm, which must still
show the bias.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.schedulers.seed_entropy_kl import null_kl_bits
from fuzzer_tool.core.schedulers.seed_entropy_zscore import (
    EntropyLengthNull,
    EntropyZScoreSeedStrategy,
)
from tests.test_regression_entropy_kl_length_bias import (
    SEED,
    _null_corpus,
    _spearman,
    _zipf,
)


class _Rng:
    def weighted_choice(self, seq, weights):
        return seq[0]


def _arm(**kw) -> EntropyZScoreSeedStrategy:
    return EntropyZScoreSeedStrategy(_Rng(), **kw)


def _short_share(seeds, weights) -> float:
    lengths = np.array([len(s) for s in seeds])
    w = np.asarray(weights)
    short = lengths <= 64
    return float((w[short].sum() / w.sum()) / short.mean())


class TestLengthNullOracle:
    @staticmethod
    def _q() -> np.ndarray:
        q = 1.0 / np.arange(1, 257) ** 1.3
        return q / q.sum()

    def test_bias_is_the_kl_null_mean_at_n2(self):
        # E[H(P_hat_2)] = 1 - sum q^2 bits, enumerated by hand: a pair of
        # equal bytes has entropy 0, an unequal pair exactly 1 bit.
        q = self._q()
        entropy_q = -float(np.dot(q, np.log2(q)))
        want_bias = entropy_q - (1.0 - float(np.dot(q, q)))
        assert null_kl_bits(q, 2) == pytest.approx(want_bias, abs=1e-9)

        null = EntropyLengthNull()
        null.fit(q)
        assert null.bias_pct(np.array([2]))[0] == pytest.approx(100.0 * want_bias / 8.0, rel=1e-6)

    def test_spread_at_n2_matches_the_two_point_law(self):
        # H is 0 with probability p = sum q^2 and 1 bit otherwise.
        q = self._q()
        p = float(np.dot(q, q))
        want_sd_pct = 100.0 * math.sqrt(p * (1.0 - p)) / 8.0
        null = EntropyLengthNull(draws=6000)
        null.fit(q)
        assert null.sd_pct(np.array([2]))[0] == pytest.approx(want_sd_pct, rel=0.06)

    def test_n0_and_n1_have_no_spread_but_stay_finite(self):
        null = EntropyLengthNull()
        null.fit(self._q())
        sd = null.sd_pct(np.array([0, 1]))
        assert np.isfinite(sd).all() and (sd > 0).all()
        assert null.bias_pct(np.array([0]))[0] == 0.0

    def test_control_same_seed_reproduces_exactly(self):
        a, b = EntropyLengthNull(), EntropyLengthNull()
        a.fit(self._q())
        b.fit(self._q())
        ns = np.array([3, 40, 900])
        assert np.array_equal(a.sd_pct(ns), b.sd_pct(ns))


class TestLengthBias:
    def test_control_uncalibrated_arm_still_underweights_short_seeds(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        assert _short_share(seeds, _arm().scores(seeds)) < 0.8

    def test_regression_calibrated_arm_gives_short_seeds_their_share(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        share = _short_share(seeds, _arm(calibrate_length=True).scores(seeds))
        assert 0.85 < share < 1.15

    def test_regression_calibrated_weights_uncorrelated_with_length(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        w = _arm(calibrate_length=True).scores(seeds)
        assert abs(_spearman([len(s) for s in seeds], w)) < 0.2


class TestFalsification:
    def test_a_true_entropy_outlier_still_leaves_the_bulk(self):
        # Calibration must not flatten real signal: a 4 KiB near-uniform
        # seed in a skewed (Zipf) corpus has far higher entropy than any
        # null seed, and must be weighted well below the typical seed.
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        odd = bytes(rng.integers(0, 256, 4096, dtype=np.uint8))
        w = _arm(calibrate_length=True).scores([*seeds, odd])
        assert w[-1] < np.percentile(w[:-1], 5)


class TestAdversarial:
    def test_default_is_the_uncalibrated_arm(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng), count=60)
        assert _arm().scores(seeds) == _arm(calibrate_length=False).scores(seeds)

    def test_degenerate_corpora_stay_finite_and_positive(self):
        for corpus in (
            [b"", b"\x00", b"\xff", bytes(4096), bytes(5000), b"ab" * 40],
            [bytes(64)] * 3,
            [b"x"],
        ):
            w = _arm(calibrate_length=True).scores(corpus)
            assert all(math.isfinite(v) and v > 0.0 for v in w)

    def test_follows_a_large_pool_shift(self):
        rng = np.random.default_rng(SEED)
        first = _null_corpus(rng, _zipf(rng), count=40)
        second = _null_corpus(rng, _zipf(rng), count=40)
        live = _arm(calibrate_length=True)
        live.scores(first)
        got = live.scores(second)
        assert got == pytest.approx(_arm(calibrate_length=True).scores(second), abs=1e-9)

    def test_null_is_refit_only_when_the_pool_changes(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng), count=40)
        arm = _arm(calibrate_length=True)
        calls = []
        real = arm._null.fit
        arm._null.fit = lambda q: (calls.append(1), real(q))[1]

        arm.scores(seeds)
        arm.scores(seeds)
        assert len(calls) == 1
        arm.scores([*seeds, b"a new seed with fresh bytes"])
        assert len(calls) == 2
        arm.scores(seeds)  # the new seed is evicted: the pool moved back
        assert len(calls) == 3

    def test_evicted_seeds_leave_the_pool(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng), count=60)
        live = _arm(calibrate_length=True)
        live.scores(seeds)
        kept = seeds[:30]
        assert live.scores(kept) == pytest.approx(
            _arm(calibrate_length=True).scores(kept), rel=0.2, abs=1e-6
        )
