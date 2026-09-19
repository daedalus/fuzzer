"""Tests for the entropy-KL seed strategy (core/schedulers/seed_entropy_kl.py).

Proposal 2 of docs/handover/handover_entropy_seed_schedulers_2026-09-19.md.

The oracle is a scalar KL spelled out here over 256 bins -- deliberately
independent of the module's cross-entropy matvec, which is the thing under
test. Draws are captured, never sampled (Hard Rule 39).
"""

from __future__ import annotations

import ast
import inspect
import math
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.byte_entropy import POOL_SMOOTHING
from fuzzer_tool.core.schedulers.seed_entropy_kl import (
    MIN_WEIGHT,
    STATE_VERSION,
    EntropyKLSeedStrategy,
)

# Two seeds with the same scalar entropy (1 bit) over disjoint byte pairs,
# plus a third the pool is built around. §2's whole claim is that a scalar
# entropy gap cannot tell A from B and a KL can.
PAIR_A = bytes([0x41, 0x42] * 32)
PAIR_B = bytes([0x61, 0x62] * 32)
TEXT = b"the quick brown fox jumps over the lazy dog " * 4
ZEROS = bytes(64)


class CapturingRng:
    """Records the weights it is handed and returns a fixed index."""

    def __init__(self, index: int = 0):
        self._index = index
        self.weights: list[float] = []

    def weighted_choice(self, seq, weights):
        self.weights = list(weights)
        return seq[self._index]


def _reference_kl(seed: bytes, pool: list[bytes], smoothing: float = POOL_SMOOTHING) -> float:
    """KL(P_seed || P_pool) in bits, one bin at a time."""
    seed_freq = [0] * 256
    for b in seed:
        seed_freq[b] += 1

    pool_freq = [0] * 256
    pool_total = 0
    for other in pool:
        for b in other:
            pool_freq[b] += 1
            pool_total += 1

    denom = pool_total + smoothing * 256
    out = 0.0
    for value, count in enumerate(seed_freq):
        if not count:
            continue
        p = count / len(seed)
        q = (pool_freq[value] + smoothing) / denom
        out += p * math.log2(p / q)
    return out


def _strategy(rng=None) -> EntropyKLSeedStrategy:
    return EntropyKLSeedStrategy(rng or CapturingRng())


class TestScoreMath:
    def test_matches_the_scalar_reference(self):
        corpus = [PAIR_A, PAIR_B, TEXT]
        got = _strategy().scores(corpus)
        want = [_reference_kl(s, corpus) for s in corpus]
        assert got == pytest.approx(want, abs=1e-9)

    def test_lone_seed_diverges_from_itself_by_nothing(self):
        # A pool of one seed *is* that seed's distribution; only the
        # smoothing floor separates them.
        (score,) = _strategy().scores([TEXT])
        assert 0.0 <= score < 1e-2

    def test_same_entropy_different_bytes_score_differently(self):
        # The discriminating claim: PAIR_A and PAIR_B carry identical
        # scalar entropy, so any |entropy - mean| scorer ranks them equal.
        from fuzzer_tool.core.byte_entropy import byte_entropy_bits

        assert byte_entropy_bits(PAIR_A) == byte_entropy_bits(PAIR_B)

        # Pool dominated by PAIR_A: B is the novel one.
        corpus = [PAIR_A, PAIR_A + PAIR_A, PAIR_B]
        scores = _strategy().scores(corpus)
        assert scores[2] > scores[0]

    def test_scores_are_never_negative(self):
        corpus = [PAIR_A, PAIR_B, TEXT, ZEROS, b"\xff"]
        assert min(_strategy().scores(corpus)) >= 0.0

    def test_empty_seed_scores_zero(self):
        scores = _strategy().scores([b"", TEXT])
        assert scores[0] == 0.0

    def test_respects_the_cap(self):
        # Past the cap the seeds are identical, so only the head counts.
        tail = bytes(4096)
        strategy = EntropyKLSeedStrategy(CapturingRng(), cap=8)
        a, b = _strategy().scores([PAIR_A, TEXT]), strategy.scores([PAIR_A + tail, TEXT + tail])
        assert a != pytest.approx(b, abs=1e-9)
        assert strategy.stats()["pool_bytes"] == 16


class TestPoolLifecycle:
    def test_admitting_a_seed_moves_the_other_scores(self):
        strategy = _strategy()
        before = strategy.scores([TEXT, PAIR_A])[0]
        after = strategy.scores([TEXT, PAIR_A, PAIR_B, ZEROS])[0]
        assert before != pytest.approx(after, abs=1e-9)

    def test_evicted_seed_leaves_the_pool(self):
        # Adversarial: the failure mode _edge_owner_count had -- counts that
        # only ever climb keep crediting bytes to seeds that are gone. After
        # eviction the pool must be exactly TEXT again.
        strategy = _strategy()
        alone = strategy.scores([TEXT])
        strategy.scores([TEXT, ZEROS, PAIR_A])
        assert strategy.scores([TEXT]) == pytest.approx(alone, abs=1e-12)
        assert strategy.stats()["pool_bytes"] == len(TEXT)

    def test_cache_does_not_outlive_the_corpus(self):
        strategy = _strategy()
        strategy.scores([TEXT, PAIR_A, PAIR_B])
        strategy.scores([TEXT])
        assert strategy.stats()["pooled"] == 1

    def test_pool_bytes_track_the_live_corpus(self):
        strategy = _strategy()
        strategy.scores([PAIR_A, PAIR_B])
        assert strategy.stats()["pool_bytes"] == len(PAIR_A) + len(PAIR_B)


class TestSelect:
    def test_weights_are_scores_plus_the_floor(self):
        rng = CapturingRng(index=1)
        strategy = EntropyKLSeedStrategy(rng)
        corpus = [TEXT, PAIR_B, ZEROS]
        chosen = strategy.select(corpus)
        assert chosen is PAIR_B
        assert rng.weights == pytest.approx(
            [s + MIN_WEIGHT for s in strategy.scores(corpus)], abs=1e-12
        )

    def test_empty_corpus_selects_nothing(self):
        assert _strategy().select([]) is None

    def test_identical_corpus_gives_every_seed_the_same_weight(self):
        # All-zero scores must fall back to uniform, not to a single arm.
        rng = CapturingRng()
        EntropyKLSeedStrategy(rng).select([ZEROS, ZEROS[:32]])
        assert rng.weights[0] == pytest.approx(rng.weights[1], abs=1e-9)


class TestStats:
    def test_counters(self):
        strategy = _strategy()
        strategy.select([TEXT, PAIR_A])
        strategy.select([TEXT, PAIR_A])
        st = strategy.stats()
        assert st["scored"] == 2
        assert st["selected"] == 2
        assert st["mean_kl"] == pytest.approx(sum(strategy.scores([TEXT, PAIR_A])) / 2)


class TestState:
    def test_round_trip(self):
        strategy = _strategy()
        strategy.select([TEXT, PAIR_A])
        restored = EntropyKLSeedStrategy.from_dict(strategy.to_dict(), CapturingRng())
        assert restored.stats()["scored"] == strategy.stats()["scored"]
        assert restored.stats()["selected"] == strategy.stats()["selected"]

    def test_version_is_carried(self):
        assert _strategy().to_dict()["version"] == STATE_VERSION

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {},
            {"version": STATE_VERSION + 1, "scored": 5},
            {"version": STATE_VERSION, "scored": -1},
        ],
    )
    def test_malformed_payload_starts_fresh(self, payload):
        restored = EntropyKLSeedStrategy.from_dict(payload, CapturingRng())
        assert restored.stats()["scored"] == 0


class TestWiring:
    def test_registered_as_seed_strategy(self):
        from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

        assert "entropy_kl" in _SEED_STRATEGY_NAMES

    def test_constructor_takes_the_flag(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters
        assert params["entropy_kl"].default is False

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

        assert all("entropy_kl" in k for k in kws(commands.cmd_fuzz, "Fuzzer"))
        assert all("entropy_kl" in k for k in kws(commands.cmd_fuzz, "run_parallel"))
        assert all("entropy_kl" in k for k in kws(parallel._worker_main, "Fuzzer"))
        assert "entropy_kl" in commands._HAIL_MARY_FLAGS

    def test_parser_declares_flag(self):
        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        assert "entropy_kl" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))


class TestSeedPickerWiring:
    """Elo pool eligibility, handler dispatch, and the non-Elo fallback."""

    def _fuzzer(self, strategy=True, corpus=(TEXT, PAIR_B, ZEROS)):
        from fuzzer_tool.core.rand_pool import RandPool

        f = SimpleNamespace(
            corpus=list(corpus),
            seed_meta={},
            _use_elo=True,
            _elo=SimpleNamespace(select_strategy=lambda keys, **_: "seed_entropy_kl"),
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
        f._entropy_kl = EntropyKLSeedStrategy(f._rng) if strategy else None
        return f

    def _picker(self, f):
        from fuzzer_tool.services.seed_picker import SeedPicker

        sp = SeedPicker.__new__(SeedPicker)
        sp.f = f
        sp._rng = f._rng
        return sp

    def test_eligible_and_dispatched_under_elo(self):
        f = self._fuzzer()
        picked = self._picker(f)._pick_seed_elo()
        assert "entropy_kl" in f._seed_strategy_pool
        assert f._seed_strategy == "entropy_kl"
        assert picked in f.corpus

    def test_not_eligible_when_disabled_or_empty(self):
        for f in (self._fuzzer(strategy=False), self._fuzzer(corpus=())):
            f._use_boltzmann = True  # two arms, so Elo is consulted
            f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
            self._picker(f)._pick_seed_elo()
            assert "entropy_kl" not in f._seed_strategy_pool

    def test_empty_corpus_declines(self):
        assert self._picker(self._fuzzer(corpus=()))._pick_entropy_kl_seed() is None

    def test_non_elo_fallback_dispatches_before_bayesian(self, monkeypatch):
        f = self._fuzzer()
        f._use_elo = False
        f._use_bayesian = True
        f._seed_quality = {"x": 1}
        f._entropy_zscore = None
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)
        monkeypatch.setattr(sp, "_pick_bayesian_seed", lambda: pytest.fail("bayesian won"))
        assert sp.pick_seed() in f.corpus


class TestReporting:
    def _f(self, enabled=True):
        strategy = _strategy() if enabled else None
        if strategy is not None:
            strategy.select([TEXT, PAIR_A, PAIR_B])
        return SimpleNamespace(_entropy_kl=strategy, _entropy_zscore=None)

    def test_report_lines(self):
        from fuzzer_tool.services.report import _entropy_seed_lines

        lines = _entropy_seed_lines(self._f())
        assert lines[0].split() == ["Entropy", "KL:", "enabled"]
        assert "scored=3 selected=1 pooled=3" in lines[1]

    def test_report_absent_when_disabled(self):
        from fuzzer_tool.services.report import _entropy_seed_lines

        assert _entropy_seed_lines(self._f(enabled=False)) == []
        assert _entropy_seed_lines(SimpleNamespace()) == []

    def test_status_field(self):
        from fuzzer_tool.services.stats import _entropy_seed_str

        assert _entropy_seed_str(self._f()).startswith(" | ent-kl: mean=")
        assert _entropy_seed_str(self._f(enabled=False)) == ""
