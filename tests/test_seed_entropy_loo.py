"""Entropy §5, leave-one-out step: ``core/schedulers/seed_entropy_loo.py``.

Score ``Δ(s) = H(pool) - H(pool without s)`` over pooled corpus bytes. The
doc's plan is LOO first, full Shapley only if LOO's known near-duplicate
misprice matters; ``TestShapleyReference`` pins that misprice.
"""

from __future__ import annotations

import ast
import inspect
import itertools
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.byte_entropy import CumulativeByteEntropy
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_entropy_loo import MIN_WEIGHT, EntropyLOOSeedStrategy

DIVERSE = bytes(range(256))
FLAT = b"\x00" * 256
HALF = bytes(range(128)) * 2
TOP = bytes(range(128, 256)) * 2


class CapturingRng:
    def __init__(self, index=0):
        self._index = index
        self.weights = []

    def weighted_choice(self, seq, weights):
        self.weights = list(weights)
        return seq[self._index]


def _bits(seeds):
    pool = CumulativeByteEntropy()
    for s in seeds:
        pool.add(s)
    return pool.bits()


def _loo_oracle(corpus, i):
    """H(all) - H(all but i), via fresh pools (independent of the strategy)."""
    return _bits(corpus) - _bits(corpus[:i] + corpus[i + 1 :])


def _shapley(corpus):
    """Exact Shapley over all orderings (small n only)."""
    n = len(corpus)
    phi = [0.0] * n
    perms = list(itertools.permutations(range(n)))
    for order in perms:
        seen = []
        for i in order:
            before = _bits([corpus[j] for j in seen]) if seen else 0.0
            seen.append(i)
            phi[i] += _bits([corpus[j] for j in seen]) - before
    return [p / len(perms) for p in phi]


# --------------------------------------------------------------------------
# score math
# --------------------------------------------------------------------------


class TestScores:
    def test_matches_independent_oracle(self):
        corpus = [DIVERSE, FLAT, HALF, TOP, b"hello world"]
        got = EntropyLOOSeedStrategy(CapturingRng()).scores(corpus)
        for i, g in enumerate(got):
            assert g == pytest.approx(_loo_oracle(corpus, i), abs=1e-9)

    def test_single_seed_scores_its_own_entropy(self):
        assert EntropyLOOSeedStrategy(CapturingRng()).scores([HALF]) == [pytest.approx(7.0)]

    def test_flat_seed_can_score_negative(self):
        """Removing a flat seed raises pooled entropy, so Δ < 0."""
        scores = EntropyLOOSeedStrategy(CapturingRng()).scores([DIVERSE, FLAT])
        assert scores[1] < 0

    def test_tracks_evictions_and_admissions(self):
        s = EntropyLOOSeedStrategy(CapturingRng())
        s.scores([DIVERSE, FLAT, HALF])
        corpus = [DIVERSE, TOP]
        got = s.scores(corpus)
        for i, g in enumerate(got):
            assert g == pytest.approx(_loo_oracle(corpus, i), abs=1e-9)

    def test_empty_seed_is_zero(self):
        corpus = [b"", HALF]
        assert EntropyLOOSeedStrategy(CapturingRng()).scores(corpus)[0] == pytest.approx(0.0)


class TestShapleyReference:
    def test_control_shapley_is_efficient(self):
        """Rule 46 control: the reference must pass its own axiom first."""
        corpus = [HALF, TOP, FLAT]
        assert sum(_shapley(corpus)) == pytest.approx(_bits(corpus))

    def test_falsification_loo_misprices_duplicate_pair(self):
        """The doc's stated LOO failure: two same-histogram seeds each look
        free to drop, while Shapley splits the pool's entropy between them.
        Pins the known limitation."""
        corpus = [HALF, HALF[::-1]]
        loo = EntropyLOOSeedStrategy(CapturingRng()).scores(corpus)
        phi = _shapley(corpus)
        assert loo == [pytest.approx(0.0, abs=1e-9)] * 2
        assert phi == [pytest.approx(_bits([HALF]) / 2)] * 2


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------


class TestSelect:
    def test_weights_are_floored_scores(self):
        rng = CapturingRng()
        corpus = [DIVERSE, FLAT, HALF]
        s = EntropyLOOSeedStrategy(rng)
        s.select(corpus)
        expected = [max(v, 0.0) + MIN_WEIGHT for v in s.scores(corpus)]
        assert rng.weights == pytest.approx(expected)

    def test_adversarial_all_nonpositive_does_not_raise(self):
        s = EntropyLOOSeedStrategy(RandPool(seed=3))
        assert s.select([FLAT, FLAT]) == FLAT

    def test_empty_corpus(self):
        assert EntropyLOOSeedStrategy(CapturingRng()).select([]) is None

    def test_stats(self):
        s = EntropyLOOSeedStrategy(CapturingRng())
        s.select([DIVERSE, HALF])
        st = s.stats()
        assert st["selected"] == 1
        assert st["pooled"] == 2


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


class TestWiring:
    def test_registered_as_seed_strategy(self):
        from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

        assert "entropy_loo" in _SEED_STRATEGY_NAMES

    def test_constructor_takes_the_flag(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        assert inspect.signature(Fuzzer.__init__).parameters["entropy_loo"].default is False

    def test_cli_passes_flag(self):
        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
        calls = [
            c
            for c in ast.walk(tree)
            if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "Fuzzer"
        ]
        assert all("entropy_loo" in {k.arg for k in c.keywords} for c in calls)
        assert "entropy_loo" in commands._HAIL_MARY_FLAGS
        assert "entropy_loo" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))

    def test_banner_lists_it(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        src = inspect.getsource(Fuzzer)
        assert src.count('"entropy-loo"') >= 2


class TestSeedPickerWiring:
    def _fuzzer(self, strategy=True, corpus=(DIVERSE, HALF, TOP)):
        f = SimpleNamespace(
            corpus=list(corpus),
            seed_meta={},
            _use_elo=True,
            _elo=SimpleNamespace(select_strategy=lambda keys, **_: "seed_entropy_loo"),
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
        f._entropy_loo = EntropyLOOSeedStrategy(f._rng) if strategy else None
        return f

    def _picker(self, f):
        from fuzzer_tool.services.seed_picker import SeedPicker

        sp = SeedPicker.__new__(SeedPicker)
        sp.f = f
        sp._rng = f._rng
        return sp

    def test_eligible_and_dispatched(self):
        f = self._fuzzer()
        picked = self._picker(f)._pick_seed_elo()
        assert "entropy_loo" in f._seed_strategy_pool
        assert f._seed_strategy == "entropy_loo"
        assert picked in f.corpus

    def test_not_eligible_when_disabled_or_empty(self):
        for f in (self._fuzzer(strategy=False), self._fuzzer(corpus=())):
            f._use_boltzmann = True
            f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
            self._picker(f)._pick_seed_elo()
            assert "entropy_loo" not in f._seed_strategy_pool

    def test_non_elo_fallback_dispatches_before_bayesian(self, monkeypatch):
        f = self._fuzzer()
        f._use_elo = False
        f._use_bayesian = True
        f._seed_quality = {"x": 1}
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)
        monkeypatch.setattr(sp, "_pick_bayesian_seed", lambda: pytest.fail("bayesian won"))
        assert sp.pick_seed() in f.corpus


class TestReporting:
    def _f(self, enabled=True):
        strategy = EntropyLOOSeedStrategy(CapturingRng()) if enabled else None
        if strategy is not None:
            strategy.select([DIVERSE, HALF])
        return SimpleNamespace(_entropy_loo=strategy)

    def test_report_lines(self):
        from fuzzer_tool.services.report import _entropy_seed_lines

        lines = _entropy_seed_lines(self._f())
        assert lines[0].split()[:2] == ["Entropy", "LOO:"]
        assert "selected=1 pooled=2" in lines[1]

    def test_report_absent_when_disabled(self):
        from fuzzer_tool.services.report import _entropy_seed_lines

        assert _entropy_seed_lines(self._f(enabled=False)) == []

    def test_status_field(self):
        from fuzzer_tool.services.stats import _entropy_seed_str

        assert _entropy_seed_str(self._f()).startswith(" | ent-loo: mean=")
        assert _entropy_seed_str(self._f(enabled=False)) == ""
