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
