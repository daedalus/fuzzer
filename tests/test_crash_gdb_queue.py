"""GDB crash replay runs beside the fuzz loop, not inside save_crash.

A novel crash used to block the loop for one gdb run (~2 s on fuzzgoat ASAN,
31% of fuzz_one over 10k execs). The report is written at once without the
GDB section; the queue polls gdb at maintenance ticks and rewrites the .txt
and .json when it exits.

A fake ``gdb`` on PATH prints a fixed crash and waits for a release file, so
"before" and "after" are observed deterministically, never by timing.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from fuzzer_tool.core.crash_metadata import CrashMetadata
from fuzzer_tool.services import corpus_manager as cm_service
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.crash_gdb import MAX_PENDING, CrashGdbQueue, TraceMode
from fuzzer_tool.services.fuzz_round import FuzzRound
from tests.test_crash_explain import ASAN_STDERR, _Fuzzer

GDB_MARK = "=== GDB crash replay ==="
FRAME = "#0  0x0000000000401136 in boom () at boom.c:3"
FAKE_GDB = """#!/bin/sh
echo run >> "{release}.runs"
sleep 60 &
echo $! > "{release}.child"
while [ ! -e "{release}" ]; do sleep 0.02; done
echo "Program received signal SIGSEGV, Segmentation fault."
echo "{frame}"
"""


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Install the fake gdb; touching the returned path lets it finish."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gate = tmp_path / "release"
    gdb = bindir / "gdb"
    gdb.write_text(FAKE_GDB.format(release=gate, frame=FRAME))
    gdb.chmod(gdb.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return gate


def _wired(tmp_path: Path, timeout: int = 10) -> tuple[_Fuzzer, CrashGdbQueue]:
    f = _Fuzzer(tmp_path, None)
    f._crash_gdb = CrashGdbQueue(f.target, f.crashes_dir, timeout)
    return f, f._crash_gdb


def _report(f: _Fuzzer, name: str) -> tuple[str, dict]:
    txt = (f.crashes_dir / f"{name}.txt").read_text()
    return txt, json.loads((f.crashes_dir / f"{name}.json").read_text())


def test_save_crash_returns_before_gdb_finishes(tmp_path, release):
    f, q = _wired(tmp_path)

    name = CorpusManager(f).save_crash(b"boom", -11, ASAN_STDERR)

    txt, d = _report(f, name)
    assert GDB_MARK not in txt
    assert d["gdb_replay"] == ""
    assert q.pending == 1

    release.touch()
    q.close()

    txt, d = _report(f, name)
    assert GDB_MARK in txt and FRAME in txt
    assert FRAME in d["gdb_replay"]
    assert q.pending == 0


def test_tick_publishes_only_finished_jobs(tmp_path, release):
    f, q = _wired(tmp_path)
    name = CorpusManager(f).save_crash(b"boom", -11, ASAN_STDERR)

    q.tick()
    assert GDB_MARK not in _report(f, name)[0]

    release.touch()
    q.close()
    assert GDB_MARK in _report(f, name)[0]


def test_falsify_without_queue_replay_stays_inline(tmp_path, release, monkeypatch):
    """No queue (partial fuzzer-likes): the old synchronous path still runs."""
    calls = []
    monkeypatch.setattr(cm_service, "_gdb_crash_replay", lambda f, d, rc: calls.append(rc) or "")
    f = _Fuzzer(tmp_path, None)

    CorpusManager(f).save_crash(b"boom", -11, ASAN_STDERR)

    assert calls == [-11]


def test_adversarial_hung_gdb_is_killed(tmp_path, release):
    """Past the timeout gdb is killed; like the inline path, only the signal lands."""
    f, q = _wired(tmp_path, timeout=1)
    name = CorpusManager(f).save_crash(b"boom", -11, ASAN_STDERR)

    q.close()  # release never touched: gdb hangs until the deadline

    txt, d = _report(f, name)
    assert FRAME not in txt
    assert "Signal: SIGSEGV" in d["gdb_replay"]
    assert q.pending == 0
    assert not _alive(int(Path(f"{release}.child").read_text()))


def _alive(pid: int) -> bool:
    """Running, not a zombie awaiting its (dead) parent's reap."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def test_adversarial_evicted_crash_not_recreated(tmp_path, release):
    f, q = _wired(tmp_path)
    name = CorpusManager(f).save_crash(b"boom", -11, ASAN_STDERR)
    for p in f.crashes_dir.glob(f"{name}.*"):
        p.unlink()

    release.touch()
    q.close()

    assert list(f.crashes_dir.glob(f"{name}.*")) == []


def test_queue_is_bounded(tmp_path, release):
    """Hard Rule 54: crashes past MAX_PENDING (live job included) are dropped."""
    f, q = _wired(tmp_path)
    for i in range(MAX_PENDING + 3):
        q.enqueue(CrashMetadata(), b"boom", -11, f"crash_{i}")

    assert q.pending == MAX_PENDING
    release.touch()
    q.close()
    assert q.pending == 0


def _runs(release: Path) -> int:
    return len(Path(f"{release}.runs").read_text().split())


def test_trace_report_reuses_the_sidecar_gdb_run(tmp_path, release):
    """--trace (default on) used to run gdb a second time, inline."""
    f, q = _wired(tmp_path)
    name = CorpusManager(f).save_crash(b"boom", -11, ASAN_STDERR)

    q.want_trace(b"boom", -11, name)
    release.touch()
    q.close()

    assert FRAME in (f.crashes_dir / f"{name}.trace").read_text()
    assert GDB_MARK in _report(f, name)[0]
    assert _runs(release) == 1


def test_trace_only_job_leaves_the_sidecar_alone(tmp_path, release):
    """A crash with no queued sidecar job (a smaller trigger) still gets its .trace."""
    f, q = _wired(tmp_path)
    q.enqueue(CrashMetadata(), b"other", -11, "crash_other")  # occupies the live slot
    (f.crashes_dir / "crash_x.bin").write_bytes(b"x")

    q.want_trace(b"x", -11, "crash_x")
    release.touch()
    q.close()

    assert FRAME in (f.crashes_dir / "crash_x.trace").read_text()
    assert not (f.crashes_dir / "crash_x.txt").exists()
    assert _runs(release) == 2


def test_trace_mode_is_an_enum():
    assert {m.name for m in TraceMode} == {"SIDECAR", "FULL"}


def test_round_hands_trace_to_the_queue():
    """Wiring: _on_crash defers the trace when the fuzzer has a queue."""
    import inspect

    src = inspect.getsource(FuzzRound._on_crash)
    assert "_crash_gdb" in src and "want_trace" in src
