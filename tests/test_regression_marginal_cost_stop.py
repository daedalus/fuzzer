"""Regression tests for ReplicatorScheduler's optional marginal-cost
stopping rule (handover §1: docs/handover/handover_decision_game_theory_survey_2026-09-13.md).

Consumer #2 from the handover's proposal: replace/augment the fixed
``window_size`` cutoff with a marginal-cost stopping rule that penalizes
an operator whose executions-per-discovery has run away relative to the
population average, using ``core/marginal_cost.py``.
"""

from fuzzer_tool.core.marginal_cost import MarginalCostTracker
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.replicator import ReplicatorScheduler


def _make_scheduler(**kwargs):
    # window_size=8 covers exactly one _run_window call below (4 execs for
    # each of the two operators) -- _execs_in_window counts total records
    # across all operators, not per operator, so the window must be sized
    # to close after both operators have been fed, not after either alone.
    sched = ReplicatorScheduler(
        window_size=8, learning_rate=0.1, mutation_rate=0.0, rng=RandPool(seed=1), **kwargs
    )
    sched.init_arm("cheap")
    sched.init_arm("expensive")
    return sched


def _run_window(sched, cheap_hits, expensive_hits):
    """Feed exactly one window's worth of records: cheap succeeds
    ``cheap_hits`` times out of 4, expensive succeeds ``expensive_hits``
    times out of 4 (each success worth weight=1.0)."""
    for i in range(4):
        sched.record("cheap", success=(i < cheap_hits), weight=1.0)
    for i in range(4):
        sched.record("expensive", success=(i < expensive_hits), weight=1.0)


# ── default behavior is unchanged ────────────────────────────────────────


def test_marginal_cost_stop_disabled_by_default():
    sched = _make_scheduler()
    assert sched.marginal_cost_stop_multiplier is None


def test_disabled_multiplier_never_calls_should_stop(monkeypatch):
    sched = _make_scheduler()
    called = []
    original = MarginalCostTracker.should_stop
    monkeypatch.setattr(
        MarginalCostTracker,
        "should_stop",
        lambda self, *a, **kw: called.append(1) or original(self, *a, **kw),
    )
    _run_window(sched, cheap_hits=4, expensive_hits=0)
    _run_window(sched, cheap_hits=4, expensive_hits=0)
    assert called == []


# ── snapshots accumulate regardless of whether the multiplier is set ────


def test_operator_marginal_costs_none_before_two_windows():
    sched = _make_scheduler()
    assert sched.operator_marginal_costs() == {"cheap": None, "expensive": None}
    _run_window(sched, cheap_hits=4, expensive_hits=4)
    # One window boundary crossed -- still only one snapshot per operator.
    assert sched.operator_marginal_costs() == {"cheap": None, "expensive": None}


def test_operator_marginal_costs_defined_after_two_windows():
    sched = _make_scheduler()
    _run_window(sched, cheap_hits=4, expensive_hits=1)
    _run_window(sched, cheap_hits=4, expensive_hits=1)
    costs = sched.operator_marginal_costs()
    # cheap: 4 execs / 4 discoveries = 1.0; expensive: 4 execs / 1 discovery = 4.0
    assert costs["cheap"] == 1.0
    assert costs["expensive"] == 4.0


# ── the stopping rule actually penalizes the expensive operator ─────────


def test_expensive_operator_shrinks_more_with_stop_rule_enabled():
    """Same two windows fed to two otherwise-identical schedulers; the one
    with marginal_cost_stop_multiplier set should end up assigning
    'expensive' a strictly smaller population share than the one without,
    because the extra shrink can only ever push growth down further."""
    baseline = _make_scheduler()
    gated = _make_scheduler(marginal_cost_stop_multiplier=1.5)

    for sched in (baseline, gated):
        # Window 1: establishes the first snapshot -- no MC signal yet,
        # both schedulers must behave identically here.
        _run_window(sched, cheap_hits=4, expensive_hits=1)
        # Window 2: expensive's cost-per-discovery is still bad, so its MC
        # this window is high relative to cheap's -- MC signal now exists.
        _run_window(sched, cheap_hits=4, expensive_hits=1)

    baseline_share = baseline.population_distribution()["expensive"]
    gated_share = gated.population_distribution()["expensive"]
    assert gated_share < baseline_share


def test_stop_rule_never_helps_the_flagged_operator():
    """The extra penalty is a min() against the ordinary update's growth,
    so an operator that already shrinks under the ordinary rule can only
    shrink the same amount or more with the stop rule enabled -- never
    less, regardless of parameters."""
    for multiplier in (1.01, 1.5, 5.0, 50.0):
        baseline = _make_scheduler()
        gated = _make_scheduler(marginal_cost_stop_multiplier=multiplier)
        for sched in (baseline, gated):
            _run_window(sched, cheap_hits=4, expensive_hits=1)
            _run_window(sched, cheap_hits=4, expensive_hits=1)
        assert (
            gated.population_distribution()["expensive"]
            <= baseline.population_distribution()["expensive"]
        )
