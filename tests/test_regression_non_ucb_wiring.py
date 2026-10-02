"""Wiring for the Elo-only non-UCB operator schedulers.

Each must be reachable end to end: flag -> construction on the fuzzer's
RandPool -> arms registered -> ballot -> select_op dispatch -> record
fan-out. Tsallis-INF is importance-weighted, so it is fed on-policy only.
"""

import pytest

from fuzzer_tool.core.schedulers import (
    AntColonyScheduler,
    EXP3IXScheduler,
    IDSScheduler,
    KalmanTSScheduler,
    LearningAutomatonScheduler,
    PHEScheduler,
    RegretMatchingScheduler,
    TsallisINFScheduler,
)
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import _OPERATOR_STRATEGY_NAMES
from fuzzer_tool.services.operators import _FALLBACK_PRECEDENCE, operator_strategy_pool
from tests.test_regression_resume_state import _fuzzer

#: Ballot name -> (scheduler class, CLI flags, Fuzzer kwargs they set).
_CASES = {
    "tsallis": (TsallisINFScheduler, ["--tsallis", "--tsallis-eta", "1.5"], {"tsallis_eta": 1.5}),
    "kalman_ts": (
        KalmanTSScheduler,
        ["--kalman-ts", "--kalman-ts-q0", "1e-6"],
        {"kalman_ts_q0": 1e-6},
    ),
    "ids": (IDSScheduler, ["--ids", "--ids-samples", "64"], {"ids_samples": 64}),
    "phe": (PHEScheduler, ["--phe", "--phe-a", "2.5"], {"phe_a": 2.5}),
    "exp3_ix": (
        EXP3IXScheduler,
        ["--exp3-ix", "--exp3-ix-eta-scale", "0.5"],
        {"exp3_ix_eta_scale": 0.5},
    ),
    "regret_matching": (
        RegretMatchingScheduler,
        ["--regret-matching", "--regret-matching-mix", "0.2"],
        {"regret_matching_mix": 0.2},
    ),
    "automaton": (
        LearningAutomatonScheduler,
        ["--automaton", "--automaton-rate", "0.1"],
        {"automaton_rate": 0.1},
    ),
    "ant_colony": (
        AntColonyScheduler,
        ["--ant-colony", "--ant-colony-rho", "0.1"],
        {"ant_colony_rho": 0.1},
    ),
}


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")
    monkeypatch.setattr("fuzzer_tool.core.elf.detect_ctx_bits", lambda _t: 4)


def _registered(scheduler) -> int:
    return len(scheduler._names)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_regression_flag_builds_and_arms(tmp_path, name):
    f = _fuzzer(tmp_path, **{name: True})
    sched = getattr(f, f"_{name}")

    assert isinstance(sched, _CASES[name][0])
    assert sched._rng is f._rng
    assert f._track_op_effect
    assert _registered(sched) > 0
    assert name in operator_strategy_pool(f)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_off_by_default(tmp_path, name):
    """Falsification: the flag, not construction, is what enables it."""
    f = _fuzzer(tmp_path)

    assert getattr(f, f"_{name}") is None
    assert name not in operator_strategy_pool(f)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_ballot_name_registered_and_elo_only(name):
    assert name in _OPERATOR_STRATEGY_NAMES
    assert name not in _FALLBACK_PRECEDENCE


class _PickElo:
    """Elo stand-in that always elects *name*."""

    def __init__(self, name: str) -> None:
        self.name = name

    def select_strategy(self, _available):
        return self.name


@pytest.mark.parametrize("name", sorted(_CASES))
def test_select_op_dispatches_to_it(tmp_path, name):
    f = _fuzzer(tmp_path, **{name: True})
    sched = getattr(f, f"_{name}")
    calls = []
    original = sched.select_op
    sched.select_op = lambda ops: calls.append(list(ops)) or original(ops)
    f._use_elo, f._elo, f._meta_strategy_cached = True, _PickElo(name), None
    f._last_mopt_particles = []  # per-exec state mutate() normally resets
    ops = ["havoc", "bitflip"]

    op = f._operators.select_op(ops)

    assert f._op_selector == name
    assert calls == [ops] and op in ops


@pytest.mark.parametrize(
    ("name", "selector", "fed"),
    [
        ("tsallis", "tsallis", True),
        ("tsallis", "bandit", False),  # on-policy: another selector's round
        ("kalman_ts", "bandit", True),  # off-policy safe: every round
        ("ids", "bandit", True),
        ("phe", "bandit", True),
        ("exp3_ix", "exp3_ix", True),
        ("exp3_ix", "bandit", False),  # on-policy
        ("regret_matching", "regret_matching", True),
        ("regret_matching", "bandit", False),  # on-policy
        ("automaton", "automaton", True),
        ("automaton", "bandit", False),  # on-policy
        ("ant_colony", "ant_colony", True),
        ("ant_colony", "bandit", True),
    ],
)
def test_record_fan_out(tmp_path, name, selector, fed):
    f = _fuzzer(tmp_path, **{name: True})
    sched = getattr(f, f"_{name}")
    seen = []
    sched.record = lambda op, ok, weight=1.0: seen.append((op, ok, weight))
    f._op_selector = selector
    rnd = FuzzRound.__new__(FuzzRound)
    rnd._f = f

    rnd._record_schedulers([("havoc", True, 0.5)])

    assert seen == ([("havoc", True, 0.5)] if fed else [])


@pytest.mark.parametrize("name", sorted(_CASES))
def test_cli_flags_reach_fuzzer(monkeypatch, tmp_path, name):
    """Adversarial: a parsed flag the CLI forgets to forward is silently off."""
    from fuzzer_tool.cli import commands

    seen = {}

    class _Stop(Exception):
        pass

    def fake_fuzzer(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(commands, "Fuzzer", fake_fuzzer)
    target = tmp_path / "t"
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    _cls, flags, kwargs = _CASES[name]
    argv = ["fuzzer-tool", "fuzz", str(target), "-d", str(tmp_path / "c"), *flags]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(_Stop):
        commands.main()

    assert seen[name] is True
    for key, value in kwargs.items():
        assert seen[key] == value
