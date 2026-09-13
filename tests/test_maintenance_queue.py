"""Tests for services/maintenance.py (P3-3, step 4 of 6)."""

import pytest

from fuzzer_tool.services.maintenance import MaintenanceJob, MaintenanceQueue


def _counter():
    """Return (call_log_list, callable) -- callable appends its call index."""
    calls = []
    return calls, lambda: calls.append(len(calls))


# --------------------------------------------------------------------------
# MaintenanceJob validation
# --------------------------------------------------------------------------


def test_job_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        MaintenanceJob("x", interval_execs=0, action=lambda: None)
    with pytest.raises(ValueError):
        MaintenanceJob("x", interval_execs=-5, action=lambda: None)


def test_job_defaults_to_never_run_and_always_active():
    job = MaintenanceJob("x", interval_execs=500, action=lambda: None)
    assert job.last_run_exec < 0
    assert job.active() is True


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_register_rejects_duplicate_id():
    q = MaintenanceQueue()
    q.register(MaintenanceJob("x", 500, action=lambda: None))
    with pytest.raises(ValueError):
        q.register(MaintenanceJob("x", 1000, action=lambda: None))


def test_register_rejects_unknown_predecessor():
    q = MaintenanceQueue()
    with pytest.raises(ValueError):
        q.register(
            MaintenanceJob("x", 500, action=lambda: None, predecessors=frozenset({"missing"}))
        )


def test_register_accepts_known_predecessor():
    q = MaintenanceQueue()
    q.register(MaintenanceJob("a", 500, action=lambda: None))
    q.register(MaintenanceJob("b", 500, action=lambda: None, predecessors=frozenset({"a"})))
    assert "a" in q and "b" in q
    assert len(q) == 2


def test_constructor_accepts_job_list_in_order():
    q = MaintenanceQueue(
        [
            MaintenanceJob("a", 500, action=lambda: None),
            MaintenanceJob("b", 500, action=lambda: None, predecessors=frozenset({"a"})),
        ]
    )
    assert len(q) == 2


# --------------------------------------------------------------------------
# due_jobs / tick: never-run jobs are always due
# --------------------------------------------------------------------------


def test_never_run_job_is_due_immediately():
    calls, action = _counter()
    q = MaintenanceQueue([MaintenanceJob("x", 1000, action=action)])
    assert q.due_jobs(0) == ["x"]
    ran = q.tick(0)
    assert ran == ["x"]
    assert calls == [0]


def test_job_not_due_before_interval_elapses():
    calls, action = _counter()
    q = MaintenanceQueue([MaintenanceJob("x", 1000, action=action)])
    q.tick(0)
    assert q.due_jobs(500) == []
    assert q.tick(500) == []
    assert calls == [0]  # only the first run


def test_job_due_again_exactly_at_interval():
    calls, action = _counter()
    q = MaintenanceQueue([MaintenanceJob("x", 1000, action=action)])
    q.tick(0)
    assert q.due_jobs(1000) == ["x"]
    q.tick(1000)
    assert calls == [0, 1]


def test_job_due_after_interval_elapses_with_slack():
    # Simulates the stats-interval sampling not landing exactly on a
    # multiple of interval_execs -- due as soon as *at least* interval_execs
    # have elapsed, not only exactly at multiples.
    calls, action = _counter()
    q = MaintenanceQueue([MaintenanceJob("x", 1000, action=action)])
    q.tick(0)
    assert q.due_jobs(1200) == ["x"]


# --------------------------------------------------------------------------
# active gating
# --------------------------------------------------------------------------


def test_inactive_job_is_never_run_and_clock_does_not_reset():
    calls, action = _counter()
    active = False
    q = MaintenanceQueue([MaintenanceJob("x", 500, action=action, active=lambda: active)])
    assert q.due_jobs(0) == []
    assert q.tick(0) == []
    assert calls == []
    # Later it becomes active -- still due, since it never actually ran.
    active = True
    assert q.due_jobs(9999) == ["x"]
    assert q.tick(9999) == ["x"]
    assert calls == [0]


def test_active_flips_off_after_running_does_not_rerun():
    calls, action = _counter()
    active = True
    q = MaintenanceQueue([MaintenanceJob("x", 500, action=action, active=lambda: active)])
    q.tick(0)
    active = False
    assert q.due_jobs(1000) == []
    assert q.tick(1000) == []
    assert calls == [0]


# --------------------------------------------------------------------------
# ordering: most-overdue-first, precedence honored
# --------------------------------------------------------------------------


def test_tick_runs_all_due_jobs_independent_of_registration_order():
    calls, action_a = _counter()
    _, action_b = _counter()
    q = MaintenanceQueue(
        [
            MaintenanceJob("b", 500, action=action_b),
            MaintenanceJob("a", 500, action=action_a),
        ]
    )
    ran = q.tick(0)
    assert set(ran) == {"a", "b"}


def test_more_overdue_job_runs_first():
    order = []
    q = MaintenanceQueue(
        [
            MaintenanceJob("frequent", 100, action=lambda: order.append("frequent")),
            MaintenanceJob("rare", 1000, action=lambda: order.append("rare")),
        ]
    )
    q.tick(0)  # both never-run -> both run once, order among equals is
    # implementation-defined for ties, so just reset and test the real case:
    order.clear()
    # frequent last ran at 1000, due again at 1100; rare last ran at 1000,
    # due again at 2000. At exec_count=3000 both are due, but rare has been
    # waiting since 2000 (1000 execs overdue) while frequent has been
    # waiting since 1100 (1900 execs overdue) -- frequent should run first.
    q._jobs["frequent"].last_run_exec = 1000
    q._jobs["rare"].last_run_exec = 1000
    ran = q.tick(3000)
    assert ran == ["frequent", "rare"]


def test_precedence_overrides_due_date_ordering():
    order = []
    q = MaintenanceQueue(
        [
            MaintenanceJob("later_due", 1000, action=lambda: order.append("later_due")),
        ]
    )
    q.register(
        MaintenanceJob(
            "must_go_first",
            1000,
            action=lambda: order.append("must_go_first"),
            predecessors=frozenset({"later_due"}),
        )
    )
    # later_due has never run (due_date=-inf, maximally overdue) while
    # must_go_first also never run (due_date=-inf too) -- without precedence
    # this is a tie broken by id ("later_due" < "must_go_first"), which
    # would put later_due first. The precedence constraint must override
    # that and force must_go_first's predecessor (later_due) to still run
    # before it regardless -- so assert the precedence is respected, i.e.
    # later_due appears before must_go_first.
    ran = q.tick(0)
    assert ran.index("later_due") < ran.index("must_go_first")


def test_predecessor_not_due_this_tick_imposes_no_constraint():
    order = []
    q = MaintenanceQueue(
        [
            MaintenanceJob("pred", 10_000, action=lambda: order.append("pred")),
        ]
    )
    q.register(
        MaintenanceJob(
            "dependent",
            500,
            action=lambda: order.append("dependent"),
            predecessors=frozenset({"pred"}),
        )
    )
    q.tick(0)  # both run once (both never-run)
    order.clear()
    # pred is not due again until exec 10_000; dependent is due at 500.
    ran = q.tick(1000)
    assert ran == ["dependent"]


# --------------------------------------------------------------------------
# integration-shaped: modeling the three absorbed ad-hoc gates
# --------------------------------------------------------------------------


def test_absorbs_the_three_ad_hoc_gate_shapes():
    """Regression-shaped test mirroring the real callers this replaces:
    crash/sanitizer replays (i % 500, guarded by replay_n / asan/ubsan
    targets), memory prune (1000-exec throttle, guarded by a config
    threshold), and gc.collect (i % 500, unconditional)."""
    replay_n = 0
    asan_target = None
    prune_threshold = 0

    crash_calls, crash_action = _counter()
    san_calls, san_action = _counter()
    mem_calls, mem_action = _counter()
    gc_calls, gc_action = _counter()

    q = MaintenanceQueue(
        [
            MaintenanceJob("gc", 500, action=gc_action),
            MaintenanceJob(
                "crash_replays", 500, action=crash_action, active=lambda: replay_n > 0
            ),
            MaintenanceJob(
                "sanitizer_replays", 500, action=san_action, active=lambda: asan_target is not None
            ),
            MaintenanceJob(
                "memory_prune", 1000, action=mem_action, active=lambda: prune_threshold > 0
            ),
        ]
    )

    # Nothing configured yet: only gc (always active) fires.
    ran = q.tick(500)
    assert ran == ["gc"]
    assert gc_calls == [0]
    assert crash_calls == san_calls == mem_calls == []

    # Enable replay_n and the memory threshold; gc not due again until 1000.
    replay_n = 5
    prune_threshold = 80
    assert set(q.due_jobs(999)) == {"crash_replays", "memory_prune"}
    ran = q.tick(1000)
    assert set(ran) == {"gc", "crash_replays", "memory_prune"}
    assert san_calls == []  # asan_target still None
