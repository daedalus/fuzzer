"""spectral_scores() caches its rows per pool version.

The spectral bound of a seed depends on its own histogram and on the pool
(through Sigma_q), so a row stays valid exactly until the pool changes
(admission or eviction bumps ``_pool_version``). Before this cache every
call recomputed all N eigendecompositions (~230 ms at N = 2000, 85 % of it
``eigh``) even when nothing had changed.

The spy counts calls to the row scorer the strategy imported, so "no
recompute" is observed, not assumed. Oracle for the values is a strategy
built fresh on the same corpus (no shared cache). Fixed-seed draws (Rule 39);
the control (Rule 46) is the first call, which must reach the scorer.
"""

from __future__ import annotations

import numpy as np
import pytest

import fuzzer_tool.core.schedulers.seed_entropy_kl as kl
from fuzzer_tool.core.schedulers.seed_entropy_kl import EntropyKLSeedStrategy
from tests.test_regression_entropy_kl_length_bias import SEED, _draw, _null_corpus, _zipf


@pytest.fixture
def calls(monkeypatch):
    real = kl.spectral_kl_rows_bits
    seen: list[int] = []

    def spy(rows, *args, **kwargs):
        seen.append(len(rows))
        return real(rows, *args, **kwargs)

    monkeypatch.setattr(kl, "spectral_kl_rows_bits", spy)
    return seen


def _corpus(count: int = 60):
    rng = np.random.default_rng(SEED)
    q = _zipf(rng)
    return rng, q, _null_corpus(rng, q, count)


class TestCache:
    def test_control_first_call_reaches_the_scorer(self, calls):
        _, _, seeds = _corpus()
        EntropyKLSeedStrategy(None).spectral_scores(seeds)
        assert calls == [len(seeds)]

    def test_repeat_call_does_not_recompute(self, calls):
        _, _, seeds = _corpus()
        strat = EntropyKLSeedStrategy(None)
        first = strat.spectral_scores(seeds)
        second = strat.spectral_scores(seeds)
        assert second == first
        assert len(calls) == 1

    def test_subset_and_reordering_are_served_from_the_same_rows(self, calls):
        _, _, seeds = _corpus()
        strat = EntropyKLSeedStrategy(None)
        strat.spectral_scores(seeds)
        # A subset is a different corpus: the pool moved, so this recomputes
        # (and must equal a fresh strategy on that subset).
        subset = seeds[::2]
        got = strat.spectral_scores(subset)
        want = EntropyKLSeedStrategy(None).spectral_scores(subset)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)
        # Same corpus, other order: cached, same values per seed.
        before = len(calls)
        again = strat.spectral_scores(list(reversed(subset)))
        assert len(calls) == before
        assert again == pytest.approx(list(reversed(got)), abs=0.0)

    def test_admission_recomputes_once_and_matches_fresh(self, calls):
        rng, q, seeds = _corpus()
        strat = EntropyKLSeedStrategy(None)
        strat.spectral_scores(seeds)
        grown = [*seeds, _draw(rng, q, 200)]
        got = strat.spectral_scores(grown)
        strat.spectral_scores(grown)
        assert len(calls) == 2
        want = EntropyKLSeedStrategy(None).spectral_scores(grown)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)

    def test_eviction_recomputes_and_matches_fresh(self, calls):
        _, _, seeds = _corpus()
        strat = EntropyKLSeedStrategy(None)
        strat.spectral_scores(seeds)
        kept = seeds[:-10]
        got = strat.spectral_scores(kept)
        assert len(calls) == 2
        want = EntropyKLSeedStrategy(None).spectral_scores(kept)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)

    def test_calibrated_repeat_is_cheap_and_stable(self, calls):
        # The spy sees only the strategy's own row scoring; the Monte-Carlo
        # null curve calls the scorer inside spectral_kl and is not counted.
        _, _, seeds = _corpus()
        strat = EntropyKLSeedStrategy(None)
        first = strat.calibrated_spectral_scores(seeds)
        second = strat.calibrated_spectral_scores(seeds)
        assert second == first
        assert len(calls) == 1

    def test_empty_and_unknown_seeds(self, calls):
        strat = EntropyKLSeedStrategy(None)
        assert strat.spectral_scores([]) == []
        assert strat.spectral_scores([b"", b"abc"])[0] == 0.0
