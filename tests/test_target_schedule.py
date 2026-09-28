"""--target-schedule: weighted (default) vs strict per-exec round robin."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.target_schedule import TargetSchedule
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.support.scripted_rng import ScriptedRng
from tests.test_dirichlet_wiring import _fuzzer_call_kwargs, _parse, build  # noqa: F401

TARGETS = ["t0", "t1", "t2"]
PAST_WARMUP = 1000  # weighted mode engages after exec 100


def _stub(schedule, edges, randoms=()):
    """Fuzzer with only the state _select_next_target reads."""
    f = Fuzzer.__new__(Fuzzer)
    f.multi_targets = list(TARGETS)
    f._active_target_idx = 0
    f._rr_turn = 0
    f.exec_count = PAST_WARMUP
    f._target_shm_covs = {
        t: SimpleNamespace(cumulative_edges=e) for t, e in zip(TARGETS, edges, strict=True)
    }
    f._rng = ScriptedRng(randoms=randoms)
    f._target_schedule = schedule
    f.target = TARGETS[0]
    return f


def _picks(f, n):
    out = []
    for _ in range(n):
        Fuzzer._select_next_target(f)
        out.append(f.target)
    return out


def test_round_robin_cycles_every_exec():
    """Falsification: each exec goes to the next target, past warm-up, in order."""
    f = _stub(TargetSchedule.ROUND_ROBIN, edges=[10, 10, 10])

    assert _picks(f, 7) == [TARGETS[i % 3] for i in range(7)]


def test_round_robin_ignores_weights_and_rng():
    """Adversarial: skewed coverage and an empty RNG script -- RR draws nothing, weights nothing."""
    f = _stub(TargetSchedule.ROUND_ROBIN, edges=[1, 50_000, 50_000], randoms=())

    picks = _picks(f, 30)

    assert [picks.count(t) for t in TARGETS] == [10, 10, 10]


def test_weighted_default_prefers_least_covered():
    """Default unchanged: r lands in t0's 1/edges slice -> t0, with one RNG draw per exec."""
    edges = [1, 1000, 1000]
    weights = [1 / e for e in edges]
    r_in_t0 = 0.5 * weights[0] / sum(weights)  # random() is scaled by the weight total
    f = _stub(TargetSchedule.WEIGHTED, edges=edges, randoms=[r_in_t0] * 5)

    assert _picks(f, 5) == ["t0"] * 5


def test_default_is_weighted(build):  # noqa: F811
    assert build()._target_schedule is TargetSchedule.WEIGHTED


def test_cli_default_and_value(monkeypatch):
    assert _parse(monkeypatch).target_schedule == TargetSchedule.WEIGHTED.value
    assert _parse(monkeypatch, "--target-schedule", "round-robin").target_schedule == "round-robin"


def test_cli_rejects_unknown(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--target-schedule", "lottery")


def test_cmd_fuzz_forwards():
    assert all("target_schedule" in k for k in _fuzzer_call_kwargs())


def test_round_robin_first_exec_is_first_target():
    """Adversarial: the first exec must not skip target 0 (N execs -> each target once)."""
    f = _stub(TargetSchedule.ROUND_ROBIN, edges=[10, 10, 10])

    assert _picks(f, 3) == TARGETS


def test_target_schedule_appended_last():
    """Fuzzer.__init__ is positional: new parameters go at the end."""
    import inspect

    params = list(inspect.signature(Fuzzer.__init__).parameters)
    assert params[-1] == "cuckoo_seed_filter"
