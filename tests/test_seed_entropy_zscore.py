"""Tests for the entropy z-score seed strategy (core/schedulers/seed_entropy_zscore.py).

Proposal 3 of docs/handover/handover_entropy_seed_schedulers_2026-09-19.md.

The point of the arm is self-calibration: SeedScorer's 25/62/93 breakpoints
are target-agnostic, so on a target whose every well-formed seed sits above
93% they all land in the same "probably random noise" bucket. The z-score is
taken against the corpus's own spread, so the same corpus still separates.

Mean and stddev oracles are recomputed here from the seed list rather than
read off the strategy. Draws are captured, never sampled (Hard Rule 39).
"""

from __future__ import annotations

import ast
import inspect
import math
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.byte_entropy import byte_entropy_pct
from fuzzer_tool.core.schedulers.seed_entropy_zscore import (
    MIN_OBSERVATIONS,
    MIN_WEIGHT,
    STATE_VERSION,
    EntropyZScoreSeedStrategy,
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


def _high_entropy(count: int, length: int = 512) -> list[bytes]:
    """Seeds that all sit above ENTROPY_RANDOM_PCT=93, as a compressed
    container target's corpus does, but are not identical to each other.

    Lengths vary so the byte histograms do: a full 256-cycle is exactly
    uniform whatever the stride, which would give every seed the same
    entropy and nothing to calibrate against.
    """
    return [
        bytes((j * (2 * i + 3) + i * 31) % 256 for j in range(length - 5 * i)) for i in range(count)
    ]


def _z(value: float, sample: list[float]) -> float:
    """Bessel-corrected z-score, spelled out independently of RunningMoments."""
    n = len(sample)
    mean = sum(sample) / n
    var = sum((x - mean) ** 2 for x in sample) / (n - 1)
    return (value - mean) / math.sqrt(var)


def _strategy(rng=None, **kwargs) -> EntropyZScoreSeedStrategy:
    return EntropyZScoreSeedStrategy(rng or CapturingRng(), **kwargs)


class TestWarmup:
    def test_not_ready_before_the_minimum_sample(self):
        strategy = _strategy()
        strategy.scores(_varied(MIN_OBSERVATIONS - 1))
        assert not strategy.ready

    def test_ready_once_the_minimum_is_reached(self):
        strategy = _strategy()
        strategy.scores(_varied(MIN_OBSERVATIONS))
        assert strategy.ready

    def test_select_declines_until_ready(self):
        # Handing back None lets the caller fall through to another arm
        # rather than picking off a variance estimate of one or two seeds.
        strategy = _strategy()
        assert strategy.select(_varied(3)) is None

    def test_zero_spread_is_never_ready(self):
        # Adversarial: a corpus of copies of one byte value has stddev 0,
        # and every z-score would be a division by it.
        strategy = _strategy()
        strategy.scores([bytes([7]) * (i + 1) for i in range(MIN_OBSERVATIONS + 5)])
        assert not strategy.ready
        assert strategy.select([bytes([7]) * 4]) is None


class TestScoreMath:
    def test_weight_is_the_gaussian_of_an_independently_computed_z(self):
        corpus = _varied(MIN_OBSERVATIONS + 6)
        strategy = _strategy()
        got = strategy.scores(corpus)

        sample = [byte_entropy_pct(s) for s in corpus]
        want = [math.exp(-0.5 * _z(byte_entropy_pct(s), sample) ** 2) + MIN_WEIGHT for s in corpus]
        assert got == pytest.approx(want, abs=1e-9)

    def test_peak_sits_on_the_target_z(self):
        corpus = _varied(MIN_OBSERVATIONS + 6)
        sample = [byte_entropy_pct(s) for s in corpus]
        zs = [_z(x, sample) for x in sample]
        top = max(range(len(corpus)), key=lambda i: zs[i])

        scores = _strategy(target_z=zs[top]).scores(corpus)
        assert max(range(len(corpus)), key=lambda i: scores[i]) == top

    def test_default_target_favours_the_typical_seed(self):
        corpus = _varied(MIN_OBSERVATIONS + 6)
        sample = [byte_entropy_pct(s) for s in corpus]
        zs = [abs(_z(x, sample)) for x in sample]
        scores = _strategy().scores(corpus)

        closest = min(range(len(corpus)), key=lambda i: zs[i])
        assert max(range(len(corpus)), key=lambda i: scores[i]) == closest

    def test_width_flattens_the_curve(self):
        corpus = _varied(MIN_OBSERVATIONS + 6)
        narrow = _strategy(width=0.5).scores(corpus)
        wide = _strategy(width=4.0).scores(corpus)
        assert max(wide) - min(wide) < max(narrow) - min(narrow)

    def test_separates_a_corpus_the_fixed_thresholds_cannot(self):
        # Every seed is above ENTROPY_RANDOM_PCT, so SeedScorer's fixed
        # breakpoint puts all of them in one bucket. Self-calibration must
        # still produce a spread.
        from fuzzer_tool.core.schedules import ENTROPY_RANDOM_PCT

        corpus = _high_entropy(MIN_OBSERVATIONS + 6)
        assert min(byte_entropy_pct(s) for s in corpus) > ENTROPY_RANDOM_PCT

        scores = _strategy().scores(corpus)
        assert max(scores) - min(scores) > 0.05


class TestObservation:
    def test_each_distinct_seed_is_observed_once(self):
        corpus = _varied(MIN_OBSERVATIONS)
        strategy = _strategy()
        strategy.scores(corpus)
        strategy.scores(corpus)
        strategy.scores(corpus)
        assert strategy.stats()["observed"] == len(corpus)

    def test_entropy_cache_does_not_outlive_the_corpus(self):
        strategy = _strategy()
        corpus = _varied(MIN_OBSERVATIONS)
        strategy.scores(corpus)
        strategy.scores(corpus[:4])
        assert strategy.stats()["cached"] == 4

    def test_moments_keep_evicted_seeds(self):
        # The tracker is a streaming estimate of the target's entropy
        # regime, not a corpus census: pruning the corpus must not reset
        # the calibration it took the whole run to build.
        strategy = _strategy()
        corpus = _varied(MIN_OBSERVATIONS)
        strategy.scores(corpus)
        strategy.scores(corpus[:2])
        assert strategy.ready
        assert strategy.stats()["observed"] == len(corpus)


class TestSelect:
    def test_weights_are_the_scores(self):
        corpus = _varied(MIN_OBSERVATIONS + 2)
        rng = CapturingRng(index=3)
        strategy = _strategy(rng)
        chosen = strategy.select(corpus)
        assert chosen is corpus[3]
        assert rng.weights == pytest.approx(strategy.scores(corpus), abs=1e-12)

    def test_empty_corpus_selects_nothing(self):
        assert _strategy().select([]) is None


class TestState:
    def test_round_trip_keeps_the_counters(self):
        corpus = _varied(MIN_OBSERVATIONS + 4)
        strategy = _strategy()
        strategy.select(corpus)

        restored = EntropyZScoreSeedStrategy.from_dict(strategy.to_dict(), CapturingRng())
        assert restored.stats()["observed"] == strategy.stats()["observed"]
        assert restored.stats()["selected"] == strategy.stats()["selected"]

    def test_resume_recalibrates_from_the_corpus(self):
        # The moments are not persisted: the corpus is reloaded whole, so
        # the first scoring pass rebuilds them from the surviving seeds and
        # a restored window would count every one of them twice.
        corpus = _varied(MIN_OBSERVATIONS + 4)
        strategy = _strategy()
        before = strategy.scores(corpus)

        restored = EntropyZScoreSeedStrategy.from_dict(strategy.to_dict(), CapturingRng())
        assert not restored.ready
        assert restored.scores(corpus) == pytest.approx(before, abs=1e-12)
        assert restored.ready

    def test_target_and_width_survive(self):
        strategy = _strategy(target_z=1.5, width=2.0)
        strategy.scores(_varied(MIN_OBSERVATIONS))
        restored = EntropyZScoreSeedStrategy.from_dict(strategy.to_dict(), CapturingRng())
        assert restored.stats()["target_z"] == 1.5
        assert restored.stats()["width"] == 2.0

    @pytest.mark.parametrize(
        "payload",
        [None, {}, {"version": STATE_VERSION + 1}, {"version": STATE_VERSION, "observed": -1}],
    )
    def test_malformed_payload_starts_fresh(self, payload):
        restored = EntropyZScoreSeedStrategy.from_dict(payload, CapturingRng())
        assert restored.stats()["observed"] == 0
        assert not restored.ready


class TestWiring:
    def test_registered_as_seed_strategy(self):
        from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

        assert "entropy_zscore" in _SEED_STRATEGY_NAMES

    def test_constructor_takes_the_flags(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters
        assert params["entropy_zscore"].default is False
        assert params["entropy_zscore_target"].default == 0.0

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

        for flag in ("entropy_zscore", "entropy_zscore_target"):
            assert all(flag in k for k in kws(commands.cmd_fuzz, "Fuzzer"))
            assert all(flag in k for k in kws(commands.cmd_fuzz, "run_parallel"))
            assert all(flag in k for k in kws(parallel._worker_main, "Fuzzer"))
        assert "entropy_zscore" in commands._HAIL_MARY_FLAGS

    def test_parser_declares_flag(self):
        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
        assert {"entropy_zscore", "entropy_zscore_target"} <= dests


class TestSeedPickerWiring:
    """Elo pool eligibility, warm-up gating, and the non-Elo fallback."""

    def _fuzzer(self, strategy=True, corpus=None, warm=False):
        from fuzzer_tool.core.rand_pool import RandPool

        corpus = _varied(MIN_OBSERVATIONS + 4) if corpus is None else list(corpus)
        f = SimpleNamespace(
            corpus=corpus,
            seed_meta={},
            _use_elo=True,
            _elo=SimpleNamespace(select_strategy=lambda keys, **_: "seed_entropy_zscore"),
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
        f._entropy_zscore = EntropyZScoreSeedStrategy(f._rng) if strategy else None
        if warm and strategy:
            f._entropy_zscore.scores(corpus)
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
        assert "entropy_zscore" in f._seed_strategy_pool
        assert f._seed_strategy == "entropy_zscore"
        assert picked in f.corpus

    def test_warms_up_within_the_first_call(self):
        # It observes the corpus only when picked, so excluding an unwarmed
        # arm would keep it unwarmed forever. Scoring is what warms it, and
        # that happens before the readiness check -- so a corpus this size
        # costs no declined pick at all.
        f = self._fuzzer()
        assert not f._entropy_zscore.warmed
        picked = self._picker(f)._pick_seed_elo()
        assert "entropy_zscore" in f._seed_strategy_pool
        assert f._entropy_zscore.ready
        assert picked in f.corpus

    def test_declines_while_the_corpus_is_too_small(self):
        f = self._fuzzer(corpus=_varied(3))
        assert self._picker(f)._pick_seed_elo() is None

    def test_dropped_once_warm_without_spread(self):
        # Adversarial: a corpus with no entropy spread never becomes ready,
        # and must stop being offered instead of declining forever.
        flat = [bytes([3]) * (i + 2) for i in range(MIN_OBSERVATIONS + 4)]
        f = self._fuzzer(corpus=flat, warm=True)
        f._use_boltzmann = True  # a second arm, so Elo is consulted
        f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
        self._picker(f)._pick_seed_elo()
        assert "entropy_zscore" not in f._seed_strategy_pool

    def test_not_eligible_when_disabled_or_empty(self):
        for f in (self._fuzzer(strategy=False), self._fuzzer(corpus=[])):
            f._use_boltzmann = True
            f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
            self._picker(f)._pick_seed_elo()
            assert "entropy_zscore" not in f._seed_strategy_pool

    def test_non_elo_fallback_dispatches_before_bayesian(self, monkeypatch):
        f = self._fuzzer(warm=True)
        f._use_elo = False
        f._use_bayesian = True
        f._seed_quality = {"x": 1}
        f._entropy_kl = None
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
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)
        monkeypatch.setattr(sp, "_pick_bayesian_seed", lambda: b"bayesian")
        assert sp.pick_seed() == b"bayesian"


class TestReporting:
    def _f(self, enabled=True):
        strategy = _strategy() if enabled else None
        if strategy is not None:
            strategy.select(_varied(MIN_OBSERVATIONS + 2))
        return SimpleNamespace(_entropy_kl=None, _entropy_zscore=strategy)

    def test_report_lines(self):
        from fuzzer_tool.services.report import _entropy_seed_lines

        lines = _entropy_seed_lines(self._f())
        assert lines[0].split()[:2] == ["Entropy", "z-score:"]
        assert f"observed={MIN_OBSERVATIONS + 2} selected=1 ready=True" in lines[1]

    def test_report_absent_when_disabled(self):
        from fuzzer_tool.services.report import _entropy_seed_lines

        assert _entropy_seed_lines(self._f(enabled=False)) == []

    def test_status_field(self):
        from fuzzer_tool.services.stats import _entropy_seed_str

        assert _entropy_seed_str(self._f()).startswith(" | ent-z: mu=")
        assert _entropy_seed_str(self._f(enabled=False)) == ""
