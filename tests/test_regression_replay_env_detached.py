"""Regression: crash replays wrote into the live run's shim channels.

The background GDB replay (``GdbJob``) inherited ``os.environ``, which holds
``_CMPLOG_COUNTS_SHM``, the cmplog sinks and the ``__AFL_*`` bitmap ids. Its
counts landed in whichever execution's drain was running, so comparison
progress -- and admission -- depended on timing: seed-7 ``--clock virtual``
campaigns forked by round 7-15 (``test_virtual_clock_repro``). The triage
loader, synchronous, added a whole interpreter's comparisons (~400k strcmp)
to the next execution's vector.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.process import detached_env
from fuzzer_tool.core.analyzers import analyzer_trace
from fuzzer_tool.core.analyzers.analyzer_trace import CrashTracer, GdbJob, TraceReport
from fuzzer_tool.services import runner as runner_mod
from fuzzer_tool.services.runner import TargetRunner

LIVE = {
    "__AFL_SHM_ID": "123",
    "__AFL_NODE_BITMAP_ID": "456",
    "_CMPLOG_COUNTS_SHM": "789",
    "_CMPLOG_OUT": "/tmp/cmplog_out",
}
# Not shim channels: must survive (names only share a substring).
KEEP = {"KEEP_ME": "1", "AFL_PATH": "/opt/afl", "X__AFL_SHM_ID": "2"}


@pytest.fixture
def live_env(monkeypatch):
    for k, v in {**LIVE, **KEEP}.items():
        monkeypatch.setenv(k, v)


def _dump_env(tmp_path: Path) -> dict[str, str]:
    """Run a GdbJob whose 'gdb' prints its environment; parse it back."""
    out = tmp_path / "env.txt"
    inp = tmp_path / "in.bin"
    inp.write_bytes(b"x")
    tracer = CrashTracer("/bin/true")
    job = GdbJob(tracer, TraceReport(), str(inp), ["sh", "-c", f"env > {out}"], None, None)
    job.wait()
    return dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)


def test_regression_gdb_job_detached(live_env, tmp_path):
    env = _dump_env(tmp_path)

    assert not set(LIVE) & set(env)


def test_gdb_job_keeps_unrelated_env(live_env, tmp_path):
    """Falsification: detaching must not drop the rest of the environment."""
    env = _dump_env(tmp_path)

    for k, v in KEEP.items():
        assert env.get(k) == v


@pytest.mark.parametrize("method", ["_run_gdb", "_run_strace"])
def test_regression_sync_tracer_runs_detached(live_env, monkeypatch, tmp_path, method):
    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw.get("env") or {"<inherited>": "1"})
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    tracer = CrashTracer("/bin/true")  # before the patch: its `which` probes inherit
    # Fixed argv: /bin/true is PIE, so _gdb_argv would probe it with `nm` first.
    monkeypatch.setattr(tracer, "_gdb_argv", lambda _p: (["gdb"], None, None))
    monkeypatch.setattr(analyzer_trace.subprocess, "run", fake_run)

    getattr(tracer, method)(str(tmp_path / "in.bin"), TraceReport())

    assert "<inherited>" not in seen
    assert not set(LIVE) & set(seen)
    assert seen["KEEP_ME"] == "1"


class _Stop(Exception):
    pass


def test_regression_triage_drops_cmplog_only(live_env, monkeypatch, tmp_path):
    """Adversarial: triage keeps __AFL_* (its crash record), loses _CMPLOG*."""
    seen = {}

    def fake_popen(argv, **kw):
        seen.update(kw["env"])
        raise _Stop

    monkeypatch.setattr(runner_mod.subprocess, "Popen", fake_popen)
    loader = tmp_path / "loader.py"
    loader.write_text("")
    f = SimpleNamespace(
        _inprocess_runner=SimpleNamespace(_loader_path=str(loader), function_name="fuzz"),
        target="/bin/true",
        timeout=1,
    )

    with pytest.raises(_Stop):
        TargetRunner(f)._run_triage_ptrace(b"x")

    assert not {k for k in seen if k.startswith("_CMPLOG")}
    assert seen["__AFL_SHM_ID"] == LIVE["__AFL_SHM_ID"]
    assert seen["_PTRACE_TRACEME"] == "1"


def test_detached_env_default_prefixes(live_env):
    env = detached_env()

    assert not set(LIVE) & set(env)
    assert all(env[k] == v for k, v in KEEP.items())
