"""--target-schedule: weighted (default) vs strict per-exec round robin."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.fair_queue import SmoothWRR, WeightedFairQueue
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
    f._target_wrr = SmoothWRR()
    f._target_wfq = WeightedFairQueue()
    f._fq_last_t = None
    f._fq_clock = lambda: 0.0
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
    assert params[-1] == "target_arena"


def _clocked(f):
    """Drive WFQ's clock so each iteration takes the scripted per-target cost."""
    state = {"t": 0.0, "i": 0}

    def clock():
        return state["t"]

    f._fq_clock = clock
    return state


def _wfq_picks(f, state, costs, n):
    out = []
    for _ in range(n):
        Fuzzer._select_next_target(f)
        out.append(f.target)
        state["t"] += costs[f.target]  # the iteration just started runs on f.target
    return out


def test_wrr_share_follows_inverse_edges():
    """Falsification: edges 1:2:4 -> weights 4:2:1 -> exactly 4/2/1 per 7 picks, no RNG."""
    f = _stub(TargetSchedule.WRR, edges=[1, 2, 4], randoms=())

    picks = _picks(f, 70)

    assert [picks.count(t) for t in TARGETS] == [40, 20, 10]


def test_wrr_engages_before_warmup():
    """Adversarial: no exec-count gate -- WRR is on from exec 0."""
    f = _stub(TargetSchedule.WRR, edges=[1, 1000, 1000])
    f.exec_count = 0

    picks = _picks(f, 20)

    assert picks.count("t0") > picks.count("t1")


def test_wrr_zero_edges_is_not_a_division_error():
    """Adversarial: a target with no edges yet weighs 1.0, not inf."""
    f = _stub(TargetSchedule.WRR, edges=[0, 0, 0])

    assert _picks(f, 6) == TARGETS * 2


def test_wfq_charges_time_not_execs():
    """Equal edges, t1 iterations cost 3x: t1 gets a third of the picks, equal wall time."""
    costs = {"t0": 1.0, "t1": 3.0, "t2": 1.0}
    f = _stub(TargetSchedule.WFQ, edges=[10, 10, 10])
    state = _clocked(f)

    picks = _wfq_picks(f, state, costs, 500)
    time = {t: picks.count(t) * costs[t] for t in TARGETS}

    assert max(time.values()) - min(time.values()) <= max(costs.values())
    assert picks.count("t1") < picks.count("t0")


def test_wfq_first_select_charges_nothing():
    """Adversarial: no previous target on the first call, so no bogus elapsed charge."""
    f = _stub(TargetSchedule.WFQ, edges=[10, 10, 10])
    f._fq_clock = lambda: 1e9

    Fuzzer._select_next_target(f)

    assert f._target_wfq.virtual_time == 0.0


def test_wfq_backwards_clock_does_not_poison():
    """Adversarial: a clock step back yields a neutral charge, never a negative one."""
    f = _stub(TargetSchedule.WFQ, edges=[10, 10, 10])
    ticks = iter([100.0, 50.0, 60.0, 70.0])
    f._fq_clock = lambda: next(ticks)

    picks = _picks(f, 4)

    assert len(picks) == 4
    assert f._target_wfq.virtual_time >= 0.0


def test_cli_accepts_wrr_and_wfq(monkeypatch):
    assert _parse(monkeypatch, "--target-schedule", "wrr").target_schedule == "wrr"
    assert _parse(monkeypatch, "--target-schedule", "wfq").target_schedule == "wfq"


def test_weighted_default_still_draws_rng():
    """Regression: the shared weight helper must not change the default's RNG use."""
    edges = [1, 1000, 1000]
    f = _stub(TargetSchedule.WEIGHTED, edges=edges, randoms=[0.0])

    _picks(f, 1)

    with pytest.raises(StopIteration):  # one scripted draw per pick, then exhausted
        _picks(f, 1)
