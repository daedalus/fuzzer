"""Time-normalised Elo strategy credit (--elo-time-credit, autofz port).

autofz (Fu et al., arXiv 2302.12879) budgets fuzzers by CPU time, so a slow
fuzzer pays for its cost. Here a timed round is judged against each idle
arm's score *per second* times this round's elapsed time, instead of its
score per round::

    b: 1.0 score over 1.0 s      -> rate 1.0 / s
    a: 1.0 score in   2.0 s      -> bar 2.0, s_ab = 0.5 + (1 - 2) / 2 = 0
    untimed                      -> bar 1.0, s_ab = 0.5 (draw)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker
from fuzzer_tool.core.clock import VIRTUAL_EXEC_S, Clock, ClockMode
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer

A, B = "op_a", "op_b"


def _primed(elapsed_b: float | None = 1.0) -> BayesianEloTracker:
    """Tracker where B played one round scoring 1.0 in *elapsed_b* seconds."""
    elo = BayesianEloTracker()
    elo.record_strategy_matches(B, [], 1.0, elapsed_b)
    return elo


def _expected_mu(elo: BayesianEloTracker, s_ab: float) -> float:
    """One Bayesian Elo step from the prior, derived from the update rule."""
    sig = elo.initial_sigma**2
    var = sig / (sig + elo.beta**2)
    return elo.initial_mu + var * elo._base_k * (s_ab - 0.5)


def _replay(elapsed: float | None) -> BayesianEloTracker:
    """A fixed match sequence, every round timed at *elapsed* (or untimed)."""
    elo = BayesianEloTracker()
    rounds = [(A, 1.0), (B, 0.0), (A, 0.3), (B, 0.8), (A, 0.0), (B, 1.0)]
    for name, score in rounds:
        other = B if name == A else A
        elo.record_strategy_matches(name, [other], score, elapsed)
    return elo


def _ratings(elo: BayesianEloTracker) -> tuple[dict, dict]:
    return elo._strategy_mu, elo._strategy_sigma_sq


# -- Tracker -------------------------------------------------------------


def test_slow_arm_loses_at_equal_yield():
    """Same score per round, twice the time: a loss, not a draw."""
    elo = _primed()
    elo.record_strategy_matches(A, [B], 1.0, 2.0)

    assert elo._strategy_mu[A] == pytest.approx(_expected_mu(elo, 0.0))
    assert elo._strategy_mu[A] < elo.initial_mu


def test_fast_arm_wins_at_equal_yield():
    """Half the time per round: bar 0.5, s_ab = 0.75."""
    elo = _primed()
    elo.record_strategy_matches(A, [B], 1.0, 0.5)

    assert elo._strategy_mu[A] == pytest.approx(_expected_mu(elo, 0.75))


def test_untimed_is_a_draw():
    """Baseline the slow-arm test is measured against: untimed, s_ab = 0.5."""
    elo = _primed(None)
    elo.record_strategy_matches(A, [B], 1.0)

    assert elo._strategy_mu[A] == pytest.approx(elo.initial_mu)


def test_control_untimed_replays_identically():
    """Control (Hard Rule 46): the untimed oracle against itself."""
    assert _ratings(_replay(None)) == _ratings(_replay(None))


def test_equal_cost_reduces_to_untimed():
    """Falsification: every round costing the same is the per-round rule."""
    timed_mu, timed_sig = _ratings(_replay(0.25))
    mu, sig = _ratings(_replay(None))

    assert timed_mu == pytest.approx(mu)
    assert timed_sig == pytest.approx(sig)


@pytest.mark.parametrize("elapsed", [0.0, -1.0])
def test_non_positive_elapsed_is_untimed(elapsed):
    """Adversarial: a zero/backward clock reading neither divides nor pays."""
    elo = _primed()
    elo.record_strategy_matches(A, [B], 1.0, elapsed)

    assert elo._strategy_mu[A] == pytest.approx(elo.initial_mu)
    assert A not in elo._strategy_time_sum


def test_huge_elapsed_clamps_to_loss():
    """Adversarial: a 1e9 s round scores 0, not a negative match score."""
    huge, slow = _primed(), _primed()
    huge.record_strategy_matches(A, [B], 1.0, 1e9)
    slow.record_strategy_matches(A, [B], 1.0, 2.0)

    assert huge._strategy_mu[A] == pytest.approx(slow._strategy_mu[A])


def test_untimed_opponent_uses_per_round_bar():
    """B has no timed history (old state): A is judged per round."""
    elo = _primed(None)
    elo.record_strategy_matches(A, [B], 1.0, 2.0)

    assert elo._strategy_mu[A] == pytest.approx(elo.initial_mu)


def test_timed_sums_only_count_timed_rounds():
    """Mixed history: the rate divides timed score by timed seconds only."""
    elo = _primed()
    elo.record_strategy_matches(B, [], 1.0)  # untimed: per-round sums only
    elo.record_strategy_matches(A, [B], 1.0, 2.0)

    assert elo._strategy_timed_sum[B] == pytest.approx(1.0)
    assert elo._strategy_time_sum[B] == pytest.approx(1.0)
    assert elo._strategy_mu[A] == pytest.approx(_expected_mu(elo, 0.0))


def test_time_sums_round_trip():
    elo = _primed(0.5)
    restored = BayesianEloTracker()
    restored.from_dict(elo.to_dict())

    assert restored._strategy_time_sum == {B: 0.5}
    assert restored._strategy_timed_sum == {B: 1.0}


def test_legacy_state_loads_without_time_sums():
    data = _primed(None).to_dict()
    data.pop("strategy_time_sum", None)
    data.pop("strategy_timed_sum", None)
    restored = BayesianEloTracker()
    restored.from_dict(data)

    assert restored._strategy_time_sum == {}
    assert restored._strategy_timed_sum == {}


# -- Wiring --------------------------------------------------------------


class _Calls:
    def __init__(self):
        self.op: list[tuple] = []
        self.seed: list[tuple] = []


def _fake_fuzzer(time_credit: bool, calls: _Calls) -> SimpleNamespace:
    return SimpleNamespace(
        _clock=Clock(ClockMode.VIRTUAL),
        _elo_time_credit=time_credit,
        _use_elo=True,
        _elo=object(),
        _meta_strategy="bandit",
        _seed_strategy="weighted",
        _record_operator_strategy_matches=lambda s, e=None: calls.op.append((s, e)),
        _record_seed_strategy_matches=lambda s, e=None: calls.seed.append((s, e)),
        _settle_positions=lambda *a: None,
        _settle_targets=lambda *a: None,
    )


def test_round_passes_elapsed_when_on():
    calls = _Calls()
    f = _fake_fuzzer(True, calls)
    rnd = FuzzRound(f, b"x")
    rnd._stamp_round()
    for _ in range(3):
        f._clock.tick()
    rnd._record_arenas()

    want = 3 * VIRTUAL_EXEC_S
    assert calls.op == [(0.0, pytest.approx(want))]
    assert calls.seed == [(0.0, pytest.approx(want))]


def test_round_passes_none_when_off():
    """Falsification: flag off, the arenas see the untimed call."""
    calls = _Calls()
    f = _fake_fuzzer(False, calls)
    rnd = FuzzRound(f, b"x")
    rnd._stamp_round()
    f._clock.tick()
    rnd._record_arenas()

    assert calls.op == [(0.0, None)]
    assert calls.seed == [(0.0, None)]


def test_seed_matches_forward_elapsed():
    elo = _primed()
    f = SimpleNamespace(_use_elo=True, _elo=elo, _seed_strategy="a", _seed_strategy_pool=["a", "b"])
    elo.record_strategy_matches("seed_b", [], 1.0, 1.0)
    Fuzzer._record_seed_strategy_matches(f, 1.0, 2.0)

    assert elo._strategy_time_sum["seed_a"] == pytest.approx(2.0)
