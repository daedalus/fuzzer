"""Regression: the spawn path started threads per exec.

``run_target_stdin`` ran a writer thread and a watchdog thread per exec;
``run_target_file`` a watchdog thread. Both now spawn with ``posix_spawnp``
and poll stdin/stderr/exit on one thread, like ``run_target_fast``. They
serve ``--file-mode``, ``--target-args``, ``--clock virtual`` and the
replay tools (tmin, minimize, root_cause, crash_isolate, crash_bisect).
"""

from __future__ import annotations

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
# Larger than a pipe buffer: stdin and stderr must be serviced together.
BIG = 300_000
EXIT_MOD = 256
ASAN_LINE = "ERROR: AddressSanitizer: fake"

# Exit status = bytes read mod 256, so delivery is checked end to end.
STDIN_COUNT = "n=$(wc -c); exit $((n % 256))\n"
FILE_COUNT = 'n=$(wc -c < "$1"); exit $((n % 256))\n'
# Writes a pipe-buffer's worth of stderr before reading stdin.
STDERR_FIRST = f'head -c {BIG} /dev/zero | tr "\\0" x >&2; n=$(wc -c); exit $((n % 256))\n'
HANG = f'echo "{ASAN_LINE}" >&2; sleep 30\n'
IGNORE_STDIN = "exit 0\n"


def _script(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


@pytest.fixture
def no_threads(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("thread started per exec")

    monkeypatch.setattr(process.threading, "Thread", _boom)


def test_regression_spawn_no_threads(tmp_path, no_threads):
    data = b"a" * 1000
    rc, err, pid = run_target_stdin(_script(tmp_path, "s", STDIN_COUNT), data, TIMEOUT)
    assert (rc, err) == (len(data) % EXIT_MOD, "")
    assert pid > 0


def test_file_mode_no_threads(tmp_path, no_threads):
    data = b"b" * 777
    target = _script(tmp_path, "f", FILE_COUNT)
    rc, _, _ = run_target_file(target, data, TIMEOUT, str(tmp_path), [])
    assert rc == len(data) % EXIT_MOD
    assert not list(tmp_path.glob("fuzz_*")), "input file left behind"


def test_file_mode_target_args(tmp_path, no_threads):
    data = b"c" * 300
    target = _script(tmp_path, "g", FILE_COUNT)
    rc, _, _ = run_target_file(target, data, TIMEOUT, str(tmp_path), ["{file}"])
    assert rc == len(data) % EXIT_MOD


def test_stdin_and_stderr_together(tmp_path, no_threads):
    """Adversarial: child fills stderr before reading a large stdin."""
    data = b"d" * BIG
    rc, err, _ = run_target_stdin(_script(tmp_path, "e", STDERR_FIRST), data, TIMEOUT)
    assert rc == len(data) % EXIT_MOD
    assert err and set(err) == {"x"}


def test_hang_times_out_with_partial_stderr(tmp_path, no_threads):
    """Falsification: a hang is killed at the deadline; its report survives."""
    for run in (
        lambda t: run_target_stdin(t, b"x", HANG_TIMEOUT),
        lambda t: run_target_file(t, b"x", HANG_TIMEOUT, str(tmp_path), []),
    ):
        t0 = time.monotonic()
        rc, err, _ = run(_script(tmp_path, "h", HANG))
        assert rc == -1
        assert ASAN_LINE in err
        assert time.monotonic() - t0 < HANG_TIMEOUT + SLACK


def test_target_ignores_stdin(tmp_path, no_threads):
    """Adversarial: exit without reading a large stdin is not an error."""
    rc, _, _ = run_target_stdin(_script(tmp_path, "i", IGNORE_STDIN), b"z" * BIG, TIMEOUT)
    assert rc == 0


def test_missing_target(tmp_path):
    rc, _, _ = run_target_stdin(str(tmp_path / "nope"), b"x", TIMEOUT)
    assert rc == -2
