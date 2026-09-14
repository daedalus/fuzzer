"""Regression tests for MOptScheduler's optional marginal-cost fitness
penalty (handover §1 candidate #2, added after the initial replicator.py
wiring -- see
docs/handover/handover_decision_game_theory_survey_2026-09-13.md §1).

MOpt's within-window fitness (disc/execs) is blind to whether the window
just evaluated is better or worse than the one before it. These tests
check the optional penalty that shrinks a particle's fitness -- before the
pbest/gbest comparison -- when its marginal cost-per-discovery, between
the last two PSO windows, has run away relative to the swarm average.
"""

from fuzzer_tool.core.marginal_cost import MarginalCostTracker
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.mopt import MOptScheduler


def _make_scheduler(**kwargs):
    # n_particles=2 keeps the swarm-average comparison legible: particle
    # "p0" cheap, "p1" expensive. window_size=8 covers exactly one
    # _run_window call below (4 execs routed to each particle) --
    # record() triggers a PSO update on self._total_execs % window_size,
    # which counts total records across all particles/operators, not per
    # particle, so the window must close only after both have been fed.
    sched = MOptScheduler(
        n_particles=2, window_size=8, rng=RandPool(seed=1), **kwargs
    )
    sched.init_arm("mut_a")
    sched.init_arm("mut_b")
    return sched


def _run_window(sched, p0_hits, p1_hits):
    """Feed one window's worth of records: particle 0 (p0, 'cheap') gets 4
    execs with p0_hits successes; particle 1 (p1, 'expensive') gets 4 execs
    with p1_hits successes. Operator choice doesn't matter for the fitness
    signal here, so both particles always use 'mut_a'."""
    for i in range(4):
        sched.record("mut_a", success=(i < p0_hits), particle_id=0, weight=1.0)
    for i in range(4):
        sched.record("mut_a", success=(i < p1_hits), particle_id=1, weight=1.0)


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
    _run_window(sched, p0_hits=4, p1_hits=0)
    _run_window(sched, p0_hits=4, p1_hits=0)
    assert called == []


# ── snapshots accumulate regardless of whether the multiplier is set ────


def test_particle_marginal_costs_none_before_two_windows():
    sched = _make_scheduler()
    assert sched.particle_marginal_costs() == {"p0": None, "p1": None}
    _run_window(sched, p0_hits=4, p1_hits=4)
    assert sched.particle_marginal_costs() == {"p0": None, "p1": None}


def test_particle_marginal_costs_defined_after_two_windows():
    sched = _make_scheduler()
    _run_window(sched, p0_hits=4, p1_hits=1)
    _run_window(sched, p0_hits=4, p1_hits=1)
    costs = sched.particle_marginal_costs()
    # p0: 4 execs / 4 discoveries = 1.0; p1: 4 execs / 1 discovery = 4.0
    assert costs["p0"] == 1.0
    assert costs["p1"] == 4.0


# ── the penalty actually shrinks the expensive particle's fitness ───────


def test_expensive_particle_fitness_lower_with_stop_rule_enabled():
    """Same two windows fed to two otherwise-identical schedulers; the one
    with marginal_cost_stop_multiplier set should end up with a strictly
    lower fitness recorded for particle 'p1' than the one without, since
    the extra shrink can only ever push fitness down further."""
    baseline = _make_scheduler()
    gated = _make_scheduler(marginal_cost_stop_multiplier=1.5)

    for sched in (baseline, gated):
        _run_window(sched, p0_hits=4, p1_hits=1)
        _run_window(sched, p0_hits=4, p1_hits=1)

    baseline_p1 = next(p for p in baseline.particles if p.name == "p1")
    gated_p1 = next(p for p in gated.particles if p.name == "p1")
    assert gated_p1.fitness < baseline_p1.fitness


def test_stop_rule_never_helps_the_flagged_particle():
    """The penalty is a min() against the ordinary fitness, so a particle
    flagged by should_stop() can only end with equal or lower fitness
    across any multiplier -- never higher."""
    for multiplier in (1.01, 1.5, 5.0, 50.0):
        baseline = _make_scheduler()
        gated = _make_scheduler(marginal_cost_stop_multiplier=multiplier)
        for sched in (baseline, gated):
            _run_window(sched, p0_hits=4, p1_hits=1)
            _run_window(sched, p0_hits=4, p1_hits=1)
        baseline_p1 = next(p for p in baseline.particles if p.name == "p1")
        gated_p1 = next(p for p in gated.particles if p.name == "p1")
        assert gated_p1.fitness <= baseline_p1.fitness
