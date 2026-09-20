"""Tests for the entropy-deviation seed strategy (core/schedulers/seed_entropy_deviation.py).

Proposal 1 of docs/handover/handover_entropy_seed_schedulers_2026-09-19.md.

The byte-content analogue of `_weight_entropy_and_distance`'s edge-hit
deviation bonus: score a seed by how far its byte entropy sits from the
corpus's own mean, not from any fixed scale. Deliberately scores against
the *mean of per-seed entropies*, not the pooled corpus distribution
`seed_entropy_kl.py` scores against -- see the module docstring for why
those are different numbers.

Mean oracle is recomputed here from the seed list rather than read off the
strategy. Draws are captured, never sampled (Hard Rule 39).
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.byte_entropy import byte_entropy_pct
from fuzzer_tool.core.schedulers.seed_entropy_deviation import (
    MIN_OBSERVATIONS,
    EntropyDeviationSeedStrategy,
    deviation_weight,
)


class CapturingRng:
    """Records the weights it is handed and returns a fixed index."""

    def __init__(self, index: int = 0):
        self._index = index
        self.weights: list[float] = []

    def weighted_choice(self, seq, weights):
        self.weights = list(weights)
        return seq[self._index]


def _varied(count: int, length: int = 64) -> list[bytes]:
    """`count` seeds whose byte variety -- and so entropy -- climbs with i."""
    return [bytes((j * 7 + i) % (i + 1) for j in range(length)) for i in range(count)]


def _mean_oracle(seeds: list[bytes]) -> float:
    """Corpus mean byte entropy, spelled out independently of the strategy."""
    return sum(byte_entropy_pct(s) for s in seeds) / len(seeds)


def _strategy(rng=None, **kwargs) -> EntropyDeviationSeedStrategy:
    return EntropyDeviationSeedStrategy(rng or CapturingRng(), **kwargs)


class TestWarmup:
    def test_not_warmed_before_the_minimum_sample(self):
        strategy = _strategy()
        strategy.scores(_varied(MIN_OBSERVATIONS - 1))
        assert not strategy.warmed

    def test_warmed_once_the_minimum_is_reached(self):
        strategy = _strategy()
        strategy.scores(_varied(MIN_OBSERVATIONS))
        assert strategy.warmed

    def test_select_declines_until_warmed(self):
        strategy = _strategy()
        assert strategy.select(_varied(MIN_OBSERVATIONS - 1)) is None

    def test_select_picks_once_warmed(self):
        strategy = _strategy()
        seeds = _varied(MIN_OBSERVATIONS + 2)
        assert strategy.select(seeds) in seeds

    def test_empty_corpus_never_warms(self):
        strategy = _strategy()
        assert strategy.select([]) is None
        assert not strategy.warmed


class TestScoreMath:
    def test_weight_matches_an_independently_computed_deviation(self):
        seeds = _varied(MIN_OBSERVATIONS + 5)
        mean = _mean_oracle(seeds)
        strategy = _strategy()
        weights = strategy.scores(seeds)
        expected = [deviation_weight(byte_entropy_pct(s), mean) for s in seeds]
        assert weights == pytest.approx(expected)

    def test_typical_seed_gets_no_bonus(self):
        # A seed sitting exactly on the mean deviates by zero.
        assert deviation_weight(50.0, 50.0) == pytest.approx(1.0)

    def test_deviation_is_capped_at_a_50pct_bonus(self):
        # Deviation ratio is huge here (entropy=100 against mean=1), but the
        # weight must not exceed 1.0 + 1.0 * 0.5 = 1.5.
        assert deviation_weight(100.0, 1.0) == pytest.approx(1.5)

    def test_zero_mean_is_a_no_op(self):
        assert deviation_weight(42.0, 0.0) == 1.0

    def test_larger_deviation_scores_higher(self):
        # Both above the mean, but one farther away.
        near = deviation_weight(55.0, 50.0)
        far = deviation_weight(90.0, 50.0)
        assert far > near

    def test_deviation_is_symmetric_above_and_below_the_mean(self):
        above = deviation_weight(60.0, 50.0)
        below = deviation_weight(40.0, 50.0)
        assert above == pytest.approx(below)


class TestObservation:
    def test_each_distinct_seed_is_observed_once(self):
        strategy = _strategy()
        seeds = _varied(MIN_OBSERVATIONS)
        strategy.scores(seeds)
        strategy.scores(seeds)
        assert strategy.stats()["observed"] == MIN_OBSERVATIONS

    def test_entropy_cache_does_not_outlive_the_corpus(self):
        strategy = _strategy()
        seeds = _varied(MIN_OBSERVATIONS + 4)
        strategy.scores(seeds)
        shrunk = seeds[:MIN_OBSERVATIONS]
        strategy.scores(shrunk)
        assert strategy.stats()["cached"] == len(shrunk)

    def test_mean_updates_as_the_corpus_grows(self):
        strategy = _strategy()
        seeds = _varied(MIN_OBSERVATIONS)
        strategy.scores(seeds)
        mean_before = strategy.stats()["mean_entropy"]
        grown = seeds + _varied(5, length=512)  # push the mean higher
        strategy.scores(grown)
        mean_after = strategy.stats()["mean_entropy"]
        assert mean_after != pytest.approx(mean_before)
        assert mean_after == pytest.approx(_mean_oracle(grown))


class TestSelect:
    def test_weights_are_the_scores(self):
        rng = CapturingRng()
        strategy = _strategy(rng)
        seeds = _varied(MIN_OBSERVATIONS + 3)
        expected = strategy.scores(list(seeds))
        strategy.select(seeds)
        assert rng.weights == pytest.approx(expected)

    def test_empty_corpus_selects_nothing(self):
        assert _strategy().select([]) is None

    def test_selected_counter_increments_only_on_success(self):
        strategy = _strategy()
        strategy.select(_varied(MIN_OBSERVATIONS - 1))
        assert strategy.stats()["selected"] == 0
        strategy.select(_varied(MIN_OBSERVATIONS))
        assert strategy.stats()["selected"] == 1


class TestWiring:
    def test_registered_as_seed_strategy(self):
        from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

        assert "entropy_deviation" in _SEED_STRATEGY_NAMES

    def test_constructor_takes_the_flag(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters
        assert params["entropy_deviation"].default is False

    def test_cli_passes_flag_to_both_constructions(self):
        from fuzzer_tool.cli import commands
        from fuzzer_tool.services import parallel

        def kws(fn, callee):
            tree = ast.parse(inspect.getsource(fn))
            return [
                {k.arg for k in c.keywords}
                for c in ast.walk(tree)
                if isinstance(c, ast.Call) and getattr(c.func, "id", None) == callee
            ]

        assert all("entropy_deviation" in k for k in kws(commands.cmd_fuzz, "Fuzzer"))
        assert all("entropy_deviation" in k for k in kws(commands.cmd_fuzz, "run_parallel"))
        assert all("entropy_deviation" in k for k in kws(parallel._worker_main, "Fuzzer"))
        assert "entropy_deviation" in commands._HAIL_MARY_FLAGS

    def test_parser_declares_flag(self):
        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
        assert "entropy_deviation" in dests


class TestSeedPickerWiring:
    """Elo pool eligibility, warm-up gating, and the non-Elo fallback."""

    def _fuzzer(self, strategy=True, corpus=None, warm=False):
        from fuzzer_tool.core.rand_pool import RandPool

        corpus = _varied(MIN_OBSERVATIONS + 4) if corpus is None else list(corpus)
        f = SimpleNamespace(
            corpus=corpus,
            seed_meta={},
            _use_elo=True,
            _elo=SimpleNamespace(select_strategy=lambda keys, **_: "seed_entropy_deviation"),
            _seed_strategy=None,
            _seed_strategy_pool=[],
            _seed_strategies_used=set(),
            _stall_recovery_active=False,
            _rng=RandPool(seed=4),
            ga=None,
            qea=None,
            markov_generate=False,
            markov_trained=False,
            _use_bayesian=False,
            _use_boltzmann=False,
            _use_ecofuzz=False,
            _profile=SimpleNamespace(format_signature=None),
        )
        f._entropy_deviation = EntropyDeviationSeedStrategy(f._rng) if strategy else None
        if warm and strategy:
            f._entropy_deviation.scores(corpus)
        return f

    def _picker(self, f):
        from fuzzer_tool.services.seed_picker import SeedPicker

        sp = SeedPicker.__new__(SeedPicker)
        sp.f = f
        sp._rng = f._rng
        return sp

    def test_eligible_and_dispatched_once_warm(self):
        f = self._fuzzer(warm=True)
        picked = self._picker(f)._pick_seed_elo()
        assert "entropy_deviation" in f._seed_strategy_pool
        assert f._seed_strategy == "entropy_deviation"
        assert picked in f.corpus

    def test_listed_even_before_warm(self):
        # Unlike entropy_zscore (which withholds while cold to avoid a
        # phantom opponent once warm-without-spread), this arm has no
        # permanent unready state -- a mean is always computable -- so it
        # is listed unconditionally, same as entropy_kl.
        f = self._fuzzer(warm=False)
        self._picker(f)._pick_seed_elo()
        assert "entropy_deviation" in f._seed_strategy_pool

    def test_not_eligible_when_disabled_or_empty(self):
        for f in (self._fuzzer(strategy=False), self._fuzzer(corpus=[])):
            f._use_boltzmann = True
            f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
            self._picker(f)._pick_seed_elo()
            assert "entropy_deviation" not in f._seed_strategy_pool

    def test_non_elo_fallback_dispatches_before_bayesian(self, monkeypatch):
        f = self._fuzzer(warm=True)
        f._use_elo = False
        f._use_bayesian = True
        f._seed_quality = {"x": 1}
        f._entropy_kl = None
        f._entropy_zscore = None
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)
        monkeypatch.setattr(sp, "_pick_bayesian_seed", lambda: pytest.fail("bayesian won"))
        assert sp.pick_seed() in f.corpus

    def test_non_elo_fallback_hands_back_while_cold(self, monkeypatch):
        f = self._fuzzer(corpus=_varied(3))
        f._use_elo = False
        f._use_bayesian = True
        f._seed_quality = {"x": 1}
        f._entropy_kl = None
        f._entropy_zscore = None
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)
        monkeypatch.setattr(sp, "_pick_bayesian_seed", lambda: b"bayesian")
        assert sp.pick_seed() == b"bayesian"


class TestReporting:
    def _f(self, enabled=True):
        strategy = _strategy() if enabled else None
        if strategy is not None:
            strategy.select(_varied(MIN_OBSERVATIONS + 2))
        return SimpleNamespace(_entropy_kl=None, _entropy_zscore=None, _entropy_deviation=strategy)

    def test_report_lines(self):
        from fuzzer_tool.services.report import _entropy_seed_lines

        lines = _entropy_seed_lines(self._f())
        assert lines[0].split()[:2] == ["Entropy", "deviation:"]
        assert f"observed={MIN_OBSERVATIONS + 2} selected=1 warmed=True" in lines[1]

    def test_report_absent_when_disabled(self):
        from fuzzer_tool.services.report import _entropy_seed_lines

        assert _entropy_seed_lines(self._f(enabled=False)) == []

    def test_status_field(self):
        from fuzzer_tool.services.stats import _entropy_seed_str

        assert _entropy_seed_str(self._f()).startswith(" | ent-dev: mu=")
        assert _entropy_seed_str(self._f(enabled=False)) == ""
