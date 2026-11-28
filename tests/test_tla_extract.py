"""tools/tla_extract.py: AST -> TLA+ extraction and TLC checking."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import tla_extract  # noqa: E402

from fuzzer_tool.adapters.forkserver import ForkserverRunner  # noqa: E402
from fuzzer_tool.core import tla  # noqa: E402
from tests.support import tla_fixtures as fx  # noqa: E402

_JAR = tla_extract.find_jar()
needs_tlc = pytest.mark.skipif(_JAR is None, reason="tla2tools.jar missing (tools/vendor_tla.sh)")


def _check(cls, tmp_path: Path) -> tla_extract.CheckResult:
    model = tla_extract.extract(cls)
    return tla_extract.check(model, tmp_path, _JAR, tla_extract.Bounds())


# ── Falsification + control (Hard Rules 23, 46) ─────────────────────


@needs_tlc
def test_orphan_only_when_leaky(tmp_path):
    clean = _check(fx.CleanRunner, tmp_path / "clean")
    assert clean.completed and not clean.violations

    leaky = _check(fx.LeakyRunner, tmp_path / "leaky")
    assert "NoOrphan" in leaky.violations


@needs_tlc
def test_desync_only_without_kill(tmp_path):
    fresh = _check(fx.FreshReplyRunner, tmp_path / "fresh")
    assert fresh.completed and not fresh.violations

    stale = _check(fx.StaleReplyRunner, tmp_path / "stale")
    assert "NoDesync" in stale.violations


@needs_tlc
def test_unguarded_recursion(tmp_path):
    res = _check(fx.RecursiveRunner, tmp_path)
    assert "RecursionBound" in res.violations


@needs_tlc
def test_setup_leak_only_unguarded(tmp_path):
    clean = _check(fx.CleanJob, tmp_path / "clean")
    assert clean.completed and not clean.violations

    leaky = _check(fx.LeakyJob, tmp_path / "leaky")
    assert "SetupCleanup" in leaky.violations


@needs_tlc
def test_use_after_close_flagged(tmp_path):
    res = _check(fx.RereadJob, tmp_path)
    assert "NoUncaught" in res.violations


@needs_tlc
def test_orphan_only_write_first(tmp_path):
    held = _check(fx.HoldFirstNotifier, tmp_path / "held")
    assert held.completed and not held.violations

    leaky = _check(fx.WriteFirstNotifier, tmp_path / "leaky")
    assert "NoOrphan" in leaky.violations


@needs_tlc
def test_short_read_desyncs(tmp_path):
    full = _check(fx.FullReadRunner, tmp_path / "full")
    assert full.completed and not full.violations

    short = _check(fx.ShortReadRunner, tmp_path / "short")
    assert "NoDesync" in short.violations


@needs_tlc
def test_stale_kill_only_unforgotten(tmp_path):
    fixed = _check(fx.ReapForgetRunner, tmp_path / "fixed")
    assert fixed.completed and not fixed.violations

    stale = _check(fx.StaleKillRunner, tmp_path / "stale")
    assert "NoStaleKill" in stale.violations


@needs_tlc
def test_foreign_stop_orphans(tmp_path):
    reaped = _check(fx.ReapStoppedRunner, tmp_path / "reaped")
    assert reaped.completed and not reaped.violations

    forgot = _check(fx.ForgetStoppedRunner, tmp_path / "forgot")
    assert "NoOrphan" in forgot.violations


@needs_tlc
def test_timer_before_handler_kills(tmp_path):
    ordered = _check(fx.HandlerFirstSampler, tmp_path / "ordered")
    assert ordered.completed and not ordered.violations

    racy = _check(fx.TimerFirstSampler, tmp_path / "racy")
    assert "NoFatalSignal" in racy.violations


@needs_tlc
def test_install_once_after_stop(tmp_path):
    checked = _check(fx.InstallCheckRunner, tmp_path / "checked")
    assert checked.completed and not checked.violations

    once = _check(fx.InstallOnceRunner, tmp_path / "once")
    assert "NoFatalSignal" in once.violations


@needs_tlc
def test_assumed_init_reuse(tmp_path):
    gated = _check(fx.GatedOnceRunner, tmp_path / "gated")
    assert gated.completed and not gated.violations

    reused = _check(fx.ReusedOnceRunner, tmp_path / "reused")
    assert "NoFatalSignal" in reused.violations


@needs_tlc
def test_lost_wakeup_deadlocks(tmp_path):
    fixed = _check(fx.FixedOracle, tmp_path / "fixed")
    assert fixed.completed and not fixed.violations

    lost = _check(fx.LostWakeupOracle, tmp_path / "lost")
    assert "NoDeadlock" in lost.violations


@needs_tlc
def test_status_zero_false_exit(tmp_path):
    reaped = _check(fx.ReapRunningRunner, tmp_path / "reaped")
    assert reaped.completed and not reaped.violations

    false_exit = _check(fx.FalseExitRunner, tmp_path / "false")
    assert "NoOrphan" in false_exit.violations


@needs_tlc
def test_caller_locals_survive_inline(tmp_path):
    """Joins after an inlined call see the caller's locals, not the callee's frame."""
    kept = _check(fx.CallerLocalRunner, tmp_path / "kept")
    assert kept.completed and not kept.violations

    leak = _check(fx.CallerLeakRunner, tmp_path / "leak")
    assert "NoOrphan" in leak.violations


@needs_tlc
def test_gdb_queue_publish_contained(tmp_path):
    """CrashGdbQueue._publish catching only OSError: a failed refresh escapes tick/close."""
    clean = _check(fx.CleanGdbQueue, tmp_path / "clean")
    assert clean.completed and not clean.violations

    buggy = _check(fx.OsErrorOnlyGdbQueue, tmp_path / "buggy")
    assert "NoUncaught" in buggy.violations


@needs_tlc
def test_adversarial_gdb_queue_drop_live(tmp_path):
    """close() without waiting drops a running gdb."""
    res = _check(fx.NoWaitGdbQueue, tmp_path)
    assert "NoOrphan" in res.violations


@needs_tlc
def test_loop_bound_only_past_unroll(tmp_path):
    """A loop done in exactly 3 iterations is clean; one needing a 4th is LoopBound."""
    three = _check(fx.ThreeStepLoop, tmp_path / "three")
    assert three.completed and not three.violations

    four = _check(fx.FourStepLoop, tmp_path / "four")
    assert "LoopBound" in four.violations


def test_bounded_wait_always_reaps():
    """`job.wait` maps to WaitBounded (no Hang outcome), not Popen's Wait."""
    model = tla_extract.extract(fx.CleanGdbQueue)
    assert "WaitBounded(" in model.tla
    assert "Wait(" not in model.tla.replace("WaitBounded(", "")


# ── Adversarial: outside the subset fails loudly ────────────────────


def _line_of(cls, needle: str) -> int:
    src = Path(fx.__file__).read_text().splitlines()
    start = next(i for i, s in enumerate(src) if s.startswith(f"class {cls.__name__}"))
    return next(i for i in range(start, len(src)) if needle in src[i]) + 1


def test_loop_rejected():
    with pytest.raises(tla_extract.ExtractError) as err:
        tla_extract.extract(fx.LoopRunner)
    assert f"tla_fixtures.py:{_line_of(fx.LoopRunner, 'match self')}" in str(err.value)


def test_unmodeled_call_rejected():
    with pytest.raises(tla_extract.ExtractError) as err:
        tla_extract.extract(fx.UnmodeledRunner)
    line = _line_of(fx.UnmodeledRunner, "os.system(")
    assert f"tla_fixtures.py:{line}" in str(err.value)


def test_effectful_loop_rejected():
    with pytest.raises(tla_extract.ExtractError) as err:
        tla_extract.extract(fx.EffectLoopRunner)
    assert f"tla_fixtures.py:{_line_of(fx.EffectLoopRunner, 'for _ in')}" in str(err.value)


def test_nonlocal_handler_rejected():
    with pytest.raises(tla_extract.ExtractError) as err:
        tla_extract.extract(fx.NonlocalHandlerRunner)
    assert f"tla_fixtures.py:{_line_of(fx.NonlocalHandlerRunner, 'nonlocal hit')}" in str(err.value)


def test_nested_blocking_call_rejected():
    with pytest.raises(tla_extract.ExtractError) as err:
        tla_extract.extract(fx.NestedBlockingOracle)
    assert f"tla_fixtures.py:{_line_of(fx.NestedBlockingOracle, 'return 1 + self')}" in str(
        err.value
    )


def test_cli_writes_nothing(tmp_path):
    rc = tla_extract.main(["tests.support.tla_fixtures:LoopRunner", "--out", str(tmp_path)])
    assert rc != 0
    assert not any(tmp_path.iterdir())


def test_undecorated_rejected():
    with pytest.raises(tla_extract.ExtractError):
        tla_extract.extract(fx.PipeReader)


# ── Real target ─────────────────────────────────────────────────────


def test_decorators_are_identity():
    def f():
        return 1

    assert tla.entry()(f) is f
    assert tla.opaque(1)(f) is f


def test_forkserver_entries():
    model = tla_extract.extract(ForkserverRunner)
    for name in ("run_one", "stop", "set_timeout", "update_shm_after_resize"):
        assert f"Call_{name} ==" in model.tla


@needs_tlc
def test_forkserver_check_runs(tmp_path):
    res = _check(ForkserverRunner, tmp_path)
    assert res.completed
    known = set(tla_extract.BUILTIN_INVARIANTS) | set(ForkserverRunner.__tla_machine__.invariants)
    assert set(res.violations) <= known


@pytest.mark.parametrize("name", sorted(tla_extract.MACHINES))
def test_machine_extracts(name):
    """Every registered machine stays inside the subset (no TLC needed)."""
    model = tla_extract.extract(tla_extract._resolve(name))
    assert "Init ==" in model.tla


@needs_tlc
def test_shm_leak_and_dangling(tmp_path):
    clean = _check(fx.CleanResize, tmp_path / "clean")
    assert clean.completed and not clean.violations
    leaky = _check(fx.LeakyResize, tmp_path / "leaky")
    assert "NoShmLeak" in leaky.violations
    dangling = _check(fx.DanglingResize, tmp_path / "dangling")
    assert "NoDanglingSeg" in dangling.violations


@needs_tlc
def test_join_keeps_none(tmp_path):
    """A tuple-or-None result stays exact through the return join."""
    exact = _check(fx.EarlyTupleRunner, tmp_path / "exact")
    assert exact.completed and not exact.violations
    leak = _check(fx.EarlyNoneLeakRunner, tmp_path / "leak")
    assert "NoOrphan" in leak.violations


# ── One TLC run for every invariant ─────────────────────────────────

# x counts 0..3; Small fails at x = 2 and Tiny at x = 1; Bounded always holds.
_MULTI = """---- MODULE Multi ----
EXTENDS Naturals
CONSTANTS MaxProcs, MaxCalls, MaxQueue
VARIABLE w
Init == w = [x |-> 0]
Next == w.x < 3 /\\ w' = [w EXCEPT !.x = @ + 1]
Small == w.x < 2
Tiny == w.x < 1
Bounded == w.x <= 3
Broken == w.nope = 1
====
"""


def _count_tlc(monkeypatch) -> list[int]:
    calls = [0]
    real = tla_extract._run_tlc

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(tla_extract, "_run_tlc", counted)
    return calls


@needs_tlc
def test_one_tlc_run_all_invariants(tmp_path, monkeypatch):
    """Every invariant's verdict comes from a single TLC run, shortest trace first."""
    calls = _count_tlc(monkeypatch)
    model = tla_extract.Model("Multi", _MULTI, ["Small", "Tiny", "Bounded"])
    res = tla_extract.check(model, tmp_path, _JAR, tla_extract.Bounds())

    assert calls[0] == 1
    assert res.completed and res.states == 4  # x = 0..3: -continue explores past violations
    assert set(res.violations) == {"Small", "Tiny"}
    assert "x |-> 1" in res.violations["Tiny"] and "x |-> 2" not in res.violations["Tiny"]


@needs_tlc
def test_tlc_error_leaves_no_ok(tmp_path):
    """Adversarial: an evaluation error is no verdict for any invariant, not a pass."""
    model = tla_extract.Model("Multi", _MULTI, ["Bounded", "Broken"])
    res = tla_extract.check(model, tmp_path, _JAR, tla_extract.Bounds())

    assert not res.completed
    assert set(res.violations) == {"Bounded", "Broken"}


def test_violations_keep_first_trace():
    log = (
        "Error: Invariant A is violated.\nError: The behavior up to this point is:\nfirst\n"
        "Error: Invariant A is violated.\nError: The behavior up to this point is:\nsecond\n"
        "Error: Invariant B is violated.\nError: The behavior up to this point is:\nthird\n"
        "Model checking completed. No error has been found.\n"
        "9 states generated, 4 distinct states found, 0 states left on queue.\n"
    )
    found = tla_extract._violations(log)
    assert set(found) == {"A", "B"}
    assert "first" in found["A"] and "second" not in found["A"]
    assert "third" in found["B"] and "Model checking" not in found["B"]


# ── Inert havoc branches collapse to one path ───────────────────────


def _havoc_in(cls, needle: str) -> bool:
    return any(needle in h for h in tla_extract.extract(cls).havoc)


def test_inert_havoc_if_skipped():
    """No branch over `_times` / `_eps` / `avg`: none of them reach modeled state."""
    model = tla_extract.extract(fx.InertHavocRunner)
    assert not any(n in h for h in model.havoc for n in ("_times", "_eps", "avg"))


@pytest.mark.parametrize(
    "cls", [fx.HavocWriteRunner, fx.HavocLocalRunner, fx.HavocReturnRunner],
    ids=lambda c: c.__name__,
)  # fmt: skip
def test_effectful_havoc_if_kept(cls):
    """Adversarial: a modeled write, a modeled local, or an early return keeps both branches."""
    assert _havoc_in(cls, "len(self._times) > 4")


@needs_tlc
def test_inert_havoc_same_states(tmp_path):
    """Skipping inert branches changes no reachable state; the bare twin agrees with itself."""
    bounds = tla_extract.Bounds()
    bare = [
        tla_extract.check(tla_extract.extract(fx.BareTickRunner), tmp_path / f"b{i}", _JAR, bounds)
        for i in (0, 1)
    ]
    inert = tla_extract.check(
        tla_extract.extract(fx.InertHavocRunner), tmp_path / "i", _JAR, bounds
    )

    assert (bare[0].states, bare[0].violations) == (bare[1].states, bare[1].violations)
    assert inert.completed and (inert.states, inert.violations) == (bare[0].states, {})


def test_regression_initial_state_violation_parsed():
    """TLC words an Init violation differently; it was dropped, hiding SetupCleanup."""
    log = (
        "Error: Invariant SetupCleanup is violated by the initial state:\n"
        'w = [setup |-> "OSError"]\n'
        "\n"
        "Finished computing initial states: 2 distinct states generated at x.\n"
        "Model checking completed. No error has been found.\n"
    )
    found = tla_extract._violations(log)
    assert set(found) == {"SetupCleanup"}
    assert "OSError" in found["SetupCleanup"] and "Finished" not in found["SetupCleanup"]
