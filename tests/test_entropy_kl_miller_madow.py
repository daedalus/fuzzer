"""Miller-Madow variant of the entropy_kl score, and the AUC that ranks it.

``miller_madow_scores`` subtracts the first-order plug-in bias
(K_hat - 1) / (2 n ln 2) bits, K_hat = distinct byte values in the seed's
sample. The oracle is spelled out here from the raw histogram, independent
of the module's vectorised path. AUC is the rank statistic P(score_alt >
score_null), computed here by ranks (no scipy). Draws come from fixed-seed
Generators (Hard Rule 39); the control (Rule 46) is AUC against labels that
carry no information, which must sit near 0.5 or the metric is broken.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.schedulers.seed_entropy_kl import EntropyKLSeedStrategy
from tests.test_regression_entropy_kl_length_bias import SEED, _draw, _null_corpus, _zipf


def auc(scores, labels) -> float:
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    ranks = np.argsort(np.argsort(s)).astype(np.float64)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def _mixed_corpus(trial: int):
    rng = np.random.default_rng(SEED + trial)
    q = _zipf(rng)
    hi = (np.arange(256) >= 0xC0).astype(np.float64)
    shifted = 0.6 * q + 0.4 * hi / hi.sum()
    seeds = _null_corpus(rng, q, 300)
    labels = [0] * len(seeds)
    for _ in range(40):
        n = int(math.exp(rng.uniform(math.log(16), math.log(4096))))
        seeds.append(_draw(rng, shifted, n))
        labels.append(1)
    return seeds, np.asarray(labels)


class TestMillerMadow:
    def test_matches_the_scalar_oracle(self):
        seeds = [b"abcabcabc", b"aaaa", bytes(range(32)), b"the quick brown fox"]
        strat = EntropyKLSeedStrategy(None)
        raw = strat.raw_scores(seeds)
        want = [
            max(r - (len(set(s)) - 1) / (2 * len(s) * math.log(2)), 0.0)
            for r, s in zip(raw, seeds, strict=True)
        ]
        assert strat.miller_madow_scores(seeds) == pytest.approx(want, abs=1e-12)

    def test_single_value_seed_gets_no_correction(self):
        strat = EntropyKLSeedStrategy(None)
        (mm,) = strat.miller_madow_scores([b"aaaa"])
        (raw,) = strat.raw_scores([b"aaaa"])
        assert mm == pytest.approx(raw, abs=1e-12)

    def test_never_negative_and_empty_seed_is_zero(self):
        scores = EntropyKLSeedStrategy(None).miller_madow_scores(
            [b"", b"\x00", b"ab" * 8, bytes(300)]
        )
        assert min(scores) >= 0.0
        assert scores[0] == 0.0

    def test_never_above_the_raw_score(self):
        seeds, _ = _mixed_corpus(0)
        strat = EntropyKLSeedStrategy(None)
        assert all(
            m <= r + 1e-12
            for m, r in zip(strat.miller_madow_scores(seeds), strat.raw_scores(seeds), strict=True)
        )


class TestAuc:
    def test_control_uninformative_labels_sit_near_half(self):
        seeds, labels = _mixed_corpus(0)
        perm = np.random.default_rng(SEED).permutation(labels)
        assert abs(auc(EntropyKLSeedStrategy(None).scores(seeds), perm) - 0.5) < 0.12

    def test_auc_is_one_for_a_perfect_ranking_and_zero_for_the_reverse(self):
        y = np.array([0, 0, 1, 1])
        assert auc([0.1, 0.2, 0.8, 0.9], y) == 1.0
        assert auc([0.9, 0.8, 0.2, 0.1], y) == 0.0

    def test_ranking_raw_lt_miller_madow_lt_calibrated(self):
        raw, mm, cal = [], [], []
        for trial in range(4):
            seeds, labels = _mixed_corpus(trial)
            strat = EntropyKLSeedStrategy(None)
            raw.append(auc(strat.raw_scores(seeds), labels))
            mm.append(auc(strat.miller_madow_scores(seeds), labels))
            cal.append(auc(strat.scores(seeds), labels))
        # Miller-Madow removes part of the bias (sparse regime: K_hat << K
        # under-corrects), the null calibration removes it.
        assert np.mean(raw) < np.mean(mm) < np.mean(cal)
