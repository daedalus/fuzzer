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


# ── Adversarial: outside the subset fails loudly ────────────────────


def _line_of(cls, needle: str) -> int:
    src = Path(fx.__file__).read_text().splitlines()
    start = next(i for i, s in enumerate(src) if s.startswith(f"class {cls.__name__}"))
    return next(i for i in range(start, len(src)) if needle in src[i]) + 1


def test_loop_rejected():
    with pytest.raises(tla_extract.ExtractError) as err:
        tla_extract.extract(fx.LoopRunner)
    assert f"tla_fixtures.py:{_line_of(fx.LoopRunner, 'while not')}" in str(err.value)


def test_unmodeled_call_rejected():
    with pytest.raises(tla_extract.ExtractError) as err:
        tla_extract.extract(fx.UnmodeledRunner)
    line = _line_of(fx.UnmodeledRunner, "os.system(")
    assert f"tla_fixtures.py:{line}" in str(err.value)


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
