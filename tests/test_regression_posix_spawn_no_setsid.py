"""Regression: spawn path failed on Pythons without posix_spawn ``setsid``.

python-build-standalone (uv's default interpreter) is built without
``POSIX_SPAWN_SETSID``: ``posix_spawnp(setsid=True)`` raises
``NotImplementedError`` and every ``run_target_stdin``/``run_target_file``
returned rc -2. The spawn now falls back to ``setpgroup=0``: the child still
leads its own process group, which is all ``killpg`` cleanup needs.
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.adapters.process import run_target_file, run_target_stdin

TIMEOUT = 5.0
HANG_TIMEOUT = 0.5
# Deadline slack for scheduler jitter.
SLACK = 2.0

# /proc/<pid>/stat field 5 is the process group id.
PGRP = 'read -r _ _ _ _ g _ < /proc/$$/stat; echo "$$ $g" >&2\n'
# Backgrounded sleeper holds stderr open; group kill must reach it.
STRAGGLER = 'sleep 30 &\necho "$!" >&2\nsleep 30\n'


def _script(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def _alive(pid: int) -> bool:
    """True when *pid* runs and is not a zombie (PID 1 may never reap it)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ")[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


@pytest.fixture
def no_setsid(monkeypatch):
    """Emulate a Python built without POSIX_SPAWN_SETSID."""
    real = os.posix_spawnp
    calls: list[dict] = []

    def _spawnp(path, argv, env, **kw):
        calls.append(kw)
        if kw.get("setsid"):
            raise NotImplementedError("posix_spawnp: setsid unavailable on this platform")
        return real(path, argv, env, **kw)

    monkeypatch.setattr(process.os, "posix_spawnp", _spawnp)
    # Fallback choice is cached per process; restore it after the test.
    monkeypatch.setattr(process, "_spawn_group", process._SPAWN_SETSID)
    return calls


def test_regression_posix_spawn_no_setsid(tmp_path, no_setsid):
    rc, err, pid = run_target_stdin(_script(tmp_path, "s", PGRP), b"", TIMEOUT)
    assert rc == 0, err

    # Child leads its own group: pgid == pid, so killpg stays scoped.
    child, pgrp = err.split()
    assert (int(child), int(pgrp)) == (pid, pid)

    rc, err, _ = run_target_file(_script(tmp_path, "f", PGRP), b"x", TIMEOUT, str(tmp_path), [])
    assert rc == 0, err


def test_fallback_is_decided_once(tmp_path, no_setsid):
    """Falsification: setsid is tried once, then never again."""
    target = _script(tmp_path, "s", "exit 0\n")
    for _ in range(3):
        assert run_target_stdin(target, b"", TIMEOUT)[0] == 0

    tried = [kw.get("setsid", False) for kw in no_setsid]
    assert tried == [True, False, False, False]


def test_fallback_timeout_kills_stragglers(tmp_path, no_setsid):
    """Adversarial: a backgrounded child holding stderr dies with the group."""
    start = time.monotonic()
    rc, err, _ = run_target_stdin(_script(tmp_path, "h", STRAGGLER), b"", HANG_TIMEOUT)
    assert rc == -1
    assert time.monotonic() - start < HANG_TIMEOUT + SLACK

    # Background child shared the group, so the timeout kill reached it.
    straggler = int(err.split()[0])
    assert not _alive(straggler)


def test_regression_setsid_fallback_race(tmp_path, monkeypatch):
    """Another thread switched the group between our setsid try and its failure."""
    real = os.posix_spawnp
    calls: list[dict] = []

    def _spawnp(path, argv, env, **kw):
        calls.append(kw)
        if not kw.get("setsid"):
            return real(path, argv, env, **kw)

        # Simulate the other thread winning the switch first.
        process._spawn_group = process._SPAWN_SETPGROUP
        raise NotImplementedError("posix_spawnp: setsid unavailable on this platform")

    monkeypatch.setattr(process.os, "posix_spawnp", _spawnp)
    monkeypatch.setattr(process, "_spawn_group", process._SPAWN_SETSID)

    assert run_target_stdin(_script(tmp_path, "s", "exit 0\n"), b"", TIMEOUT)[0] == 0
    assert [kw.get("setsid", False) for kw in calls] == [True, False]


def test_setpgroup_failure_propagates(monkeypatch):
    """Adversarial: the fallback itself unsupported raises, no retry loop."""
    calls: list[dict] = []

    def _spawnp(path, argv, env, **kw):
        calls.append(kw)
        raise NotImplementedError("unsupported")

    monkeypatch.setattr(process.os, "posix_spawnp", _spawnp)
    monkeypatch.setattr(process, "_spawn_group", process._SPAWN_SETSID)

    with pytest.raises(NotImplementedError):
        process._spawn_grouped(["/bin/true"], {}, [])
    assert len(calls) == 2
