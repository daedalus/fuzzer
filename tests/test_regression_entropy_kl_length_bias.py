"""Regression: ``entropy_kl`` scores tracked seed length, not divergence.

Plug-in KL(P_s || Q) from an n-byte sample is biased upward by roughly
(K - 1) / (2n) nats, so a 16-byte seed drawn from the pool's own
distribution (true KL 0) out-scored a 4 KiB seed that truly diverged, and
since selection is proportional to score, short seeds were over-picked.
Measured before the fix: Spearman(score, length) = -0.995 on seeds all
drawn from one distribution; seeds <= 64 B were 23.5 % of the corpus and
drew 55.5 % of the selection weight.

The fix subtracts the exact expected null KL for the seed's sample length
under the current pool (``null_kl_bits``), clipped at zero.

Oracles here are independent of the module: the exact null is enumerated
pair-by-pair for n = 1, 2, and every draw comes from a fixed-seed numpy
Generator (deterministic, no retry-until-hit loops -- Hard Rule 39). The
control (Hard Rule 46) is that the *uncalibrated* ``raw_scores`` must still
show the bias, so the test can fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.byte_entropy import ENTROPY_SAMPLE_CAP
from fuzzer_tool.core.schedulers.seed_entropy_kl import (
    Z_CAP,
    EntropyKLSeedStrategy,
    null_kl_bits,
)

SEED = 20260927


def _zipf(rng: np.random.Generator) -> np.ndarray:
    q = 1.0 / np.arange(1, 257) ** 1.1
    q = rng.permutation(q)
    return q / q.sum()


def _draw(rng: np.random.Generator, p: np.ndarray, n: int) -> bytes:
    return bytes(rng.choice(256, n, p=p).astype(np.uint8))


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def _spearman(a, b) -> float:
    ra, rb = _ranks(np.asarray(a, dtype=np.float64)), _ranks(np.asarray(b, dtype=np.float64))
    return float(np.corrcoef(ra, rb)[0, 1])


def _null_corpus(rng: np.random.Generator, q: np.ndarray, count: int = 400) -> list[bytes]:
    seeds: list[bytes] = []
    seen: set[bytes] = set()
    while len(seeds) < count:
        n = int(math.exp(rng.uniform(math.log(16), math.log(4096))))
        s = _draw(rng, q, n)
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


class TestLengthBias:
    def test_regression_score_uncorrelated_with_length_under_the_null(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        lengths = [len(s) for s in seeds]

        strategy = EntropyKLSeedStrategy(None)
        assert abs(_spearman(lengths, strategy.scores(seeds))) < 0.3

    def test_control_raw_scores_still_show_the_bias(self):
        # Rule 46: if the uncalibrated score did not fail this, the test
        # above would be a broken oracle.
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        lengths = [len(s) for s in seeds]

        raw = EntropyKLSeedStrategy(None).raw_scores(seeds)
        assert _spearman(lengths, raw) < -0.9

    def test_regression_short_seeds_do_not_dominate_selection_weight(self):
        rng = np.random.default_rng(SEED)
        seeds = _null_corpus(rng, _zipf(rng))
        short = np.array([len(s) <= 64 for s in seeds])

        scores = np.asarray(EntropyKLSeedStrategy(None).scores(seeds)) + 1e-6
        share = scores[short].sum() / scores.sum()
        # Uniform would be short.mean(); before the fix this was ~2.4x that.
        assert share < 1.5 * short.mean()


class TestFalsification:
    def test_diverging_long_seed_outscores_matching_short_seeds(self):
        # The calibration must not flatten real signal: a 4 KiB seed drawn
        # from a shifted distribution has to beat what null seeds of every
        # length reach.
        rng = np.random.default_rng(SEED)
        q = _zipf(rng)
        hi = (np.arange(256) >= 0xC0).astype(np.float64)
        shifted = 0.6 * q + 0.4 * hi / hi.sum()

        nulls = _null_corpus(rng, q)
        odd = _draw(rng, shifted, 4096)
        scores = EntropyKLSeedStrategy(None).scores([*nulls, odd])

        assert scores[-1] > np.percentile(scores[:-1], 95)

    def test_same_entropy_disjoint_pairs_still_separate(self):
        a, b = bytes([0x41, 0x42] * 32), bytes([0x61, 0x62] * 32)
        scores = EntropyKLSeedStrategy(None).scores([a, a + a, b])
        assert scores[2] > scores[0]

    def test_score_is_capped_so_one_outlier_cannot_starve_the_rest(self):
        rng = np.random.default_rng(SEED)
        nulls = _null_corpus(rng, _zipf(rng), count=60)
        odd = bytes([0xFF]) * 4096
        scores = EntropyKLSeedStrategy(None).scores([*nulls, odd])
        assert scores[-1] == Z_CAP
        assert max(scores[:-1]) <= Z_CAP


class TestNullOracle:
    @staticmethod
    def _q() -> np.ndarray:
        q = 1.0 / np.arange(1, 257) ** 1.3
        return q / q.sum()

    def test_n1_is_the_entropy_of_q(self):
        q = self._q()
        want = -float(np.dot(q, np.log2(q)))
        assert null_kl_bits(q, 1) == pytest.approx(want, abs=1e-9)

    def test_n2_matches_pair_enumeration(self):
        q = self._q()
        lq = np.log2(q)
        # All ordered pairs (b, c); a pair's KL is spelled out directly.
        b, c = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
        same = b == c
        kl = np.where(same, -lq[b], 0.5 * (-lq[b] - lq[c]) - 1.0)
        want = float((q[b] * q[c] * kl).sum())
        assert null_kl_bits(q, 2) == pytest.approx(want, abs=1e-9)

    def test_large_n_approaches_the_chi_square_expansion(self):
        # E[KL] -> (K-1) / (2n ln 2) bits when every bin's mean count is
        # large; a uniform Q makes that hold at n = 4096 (16 per bin).
        q = np.full(256, 1 / 256)
        assert null_kl_bits(q, 4096) == pytest.approx(255 / (2 * 4096 * math.log(2)), rel=0.02)

    def test_null_is_non_negative_and_decreasing_in_n(self):
        q = self._q()
        vals = [null_kl_bits(q, n) for n in (1, 2, 8, 64, 512, 4096)]
        assert min(vals) >= 0.0
        assert vals == sorted(vals, reverse=True)

    def test_interpolated_curve_tracks_the_exact_null(self):
        rng = np.random.default_rng(SEED)
        strategy = EntropyKLSeedStrategy(None)
        seeds = _null_corpus(rng, _zipf(rng), count=60)
        strategy.scores(seeds)
        q = np.asarray(strategy._pool.freq_dist())
        for n in (100, 300, 1500):
            assert strategy._null_bits(n) == pytest.approx(null_kl_bits(q, n), rel=0.03)


class TestAdversarial:
    def test_degenerate_lengths_stay_finite_and_non_negative(self):
        cap = ENTROPY_SAMPLE_CAP
        corpus = [b"", b"\x00", b"\xff", bytes(cap), bytes(cap + 500), b"ab" * 40]
        scores = EntropyKLSeedStrategy(None).scores(corpus)
        assert all(math.isfinite(s) and s >= 0.0 for s in scores)
        assert scores[0] == 0.0

    def test_single_byte_pool_does_not_blow_up(self):
        # q collapses onto one value: log1p(-q) is at its worst here.
        scores = EntropyKLSeedStrategy(None).scores([bytes(64), bytes(1024)])
        assert all(math.isfinite(s) and s >= 0.0 for s in scores)

    def test_calibration_follows_a_large_pool_shift(self):
        # A stale null curve would mis-calibrate after the pool turns over.
        rng = np.random.default_rng(SEED)
        p1, p2 = _zipf(rng), _zipf(rng)
        first = _null_corpus(rng, p1, count=40)
        second = _null_corpus(rng, p2, count=40)

        live = EntropyKLSeedStrategy(None)
        live.scores(first)
        got = live.scores(second)
        want = EntropyKLSeedStrategy(None).scores(second)
        assert got == pytest.approx(want, abs=1e-9)

    def test_small_pool_drift_stays_within_tolerance_of_fresh(self):
        rng = np.random.default_rng(SEED)
        q = _zipf(rng)
        corpus = _null_corpus(rng, q, count=300)
        extra = _draw(rng, q, 200)

        live = EntropyKLSeedStrategy(None)
        live.scores(corpus)
        got = live.scores([*corpus, extra])
        want = EntropyKLSeedStrategy(None).scores([*corpus, extra])
        # The exact mean is rebuilt on every change. The spread is a
        # 200-draw Monte-Carlo estimate (1/sqrt(2*200) = 5 % standard error),
        # so a reused build and a fresh one differ by MC noise, ~7 % between
        # two builds; bound it at 20 % rather than pretending it is exact.
        assert got == pytest.approx(want, rel=0.2, abs=1e-6)
