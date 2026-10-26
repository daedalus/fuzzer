"""Regression: return-code decoding across modes (128+sig, persistent exits, stderr sync)."""

from __future__ import annotations

import shutil
import signal
import subprocess
import threading
import time

import pytest

from fuzzer_tool.adapters import persistent_subprocess
from fuzzer_tool.adapters.filesystem import classify_crash
from fuzzer_tool.adapters.process import signal_number
from fuzzer_tool.services.runner import TargetRunner

_EXIT_BASE = 128

# Signals afl_shim and the loaders report as 128+sig.
_TRAP_SIGNALS = (signal.SIGILL, signal.SIGTRAP, signal.SIGSYS)


class _Fuzzer:
    extra_crash_codes: set[int] = set()
    last_report = None


@pytest.fixture
def runner() -> TargetRunner:
    return TargetRunner(_Fuzzer())  # type: ignore[arg-type]


@pytest.mark.parametrize("sig", _TRAP_SIGNALS)
def test_regression_128_plus_trap_signal_is_crash(runner: TargetRunner, sig: int) -> None:
    # __builtin_trap / UBSAN-trap / seccomp through the 128+sig paths.
    assert runner.is_crash(_EXIT_BASE + sig, "")
    assert runner.is_interesting(_EXIT_BASE + sig, "")


def test_plain_failing_exit_not_crash(runner: TargetRunner) -> None:
    # Falsification: exit(1) without a report stays a non-crash.
    assert not runner.is_crash(1, "")


def test_regression_echoed_asan_not_interesting(runner: TargetRunner) -> None:
    # Adversarial: a clean exit whose log merely mentions ASAN.
    assert not runner.is_interesting(0, "built with ASAN; AddressSanitizer: ok\n")


def test_real_report_interesting_on_clean_exit(runner: TargetRunner) -> None:
    stderr = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n"

    assert runner.is_interesting(0, stderr)


@pytest.mark.parametrize("sig", (signal.SIGABRT, signal.SIGSEGV, signal.SIGILL))
def test_regression_signal_signature_mode_independent(sig: int) -> None:
    # Spawn reports -sig, in-process/persistent 128+sig: one bug, one signature.
    a = classify_crash(b"a", -sig, "", set(), {})
    b = classify_crash(b"a", _EXIT_BASE + sig, "", set(), {})

    assert a.signature == b.signature == f"signal:{sig}"


def test_signal_number_keeps_other_codes() -> None:
    # Edge: a non-crash exit code is not reinterpreted as a signal.
    assert signal_number(-11) == signal.SIGSEGV
    assert signal_number(_EXIT_BASE + signal.SIGSEGV) == signal.SIGSEGV
    assert signal_number(_EXIT_BASE + 1) == _EXIT_BASE + 1
    assert signal_number(255) == 255


# ── Persistent loader (needs clang) ─────────────────────────────────────

_SO_SRC = r"""
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {
    if (n >= 2 && d[0] == 'X') _exit(d[1]);
    if (n >= 1 && d[0] == 'E') { fputs("LATE-REPORT\n", stderr); fflush(stderr); }
    return 0;
}
"""


@pytest.fixture
def so(tmp_path):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "t.c"
    src.write_text(_SO_SRC)
    out = tmp_path / "t.so"
    subprocess.run(["clang", "-shared", "-fPIC", "-o", str(out), str(src)], check=True)
    return out


@pytest.fixture
def persistent(so):
    loader = persistent_subprocess.PersistentLoader(str(so), timeout=5.0)
    if not loader.start():
        loader.stop()
        pytest.skip("persistent loader failed to start")
    yield loader
    loader.stop()


@pytest.mark.parametrize("code", (_EXIT_BASE + 1, _EXIT_BASE + 2, 255))
def test_regression_persistent_exit_code_not_signal(persistent, code: int) -> None:
    # exit(129) read as -1 (timeout), exit(130) as -2 (dropped), 255 as "signal 127".
    rc, _ = persistent.run_one(b"X" + bytes([code]))

    assert rc == code


def test_persistent_shim_crash_exit_is_signal(persistent) -> None:
    # Falsification: afl_shim's _exit(128+SIGSEGV) is still a SIGSEGV.
    rc, _ = persistent.run_one(b"X" + bytes([_EXIT_BASE + signal.SIGSEGV]))

    assert rc == -signal.SIGSEGV


# Drain-thread start delay standing in for a descheduled reader.
_DRAIN_DELAY = 0.3


def test_regression_persistent_stderr_reaches_its_run(so, monkeypatch) -> None:
    # The report must be read with the run that wrote it, not the next one.
    real_thread = threading.Thread

    def _slow_thread(*, target, daemon):
        def _late():
            time.sleep(_DRAIN_DELAY)
            target()

        return real_thread(target=_late, daemon=daemon)

    monkeypatch.setattr(persistent_subprocess.threading, "Thread", _slow_thread)
    loader = persistent_subprocess.PersistentLoader(str(so), timeout=5.0)
    try:
        assert loader.start()
        loader.run_one(b"E")
        first = loader.consume_stderr()
        loader.run_one(b"q")
        second = loader.consume_stderr()
    finally:
        loader.stop()

    assert "LATE-REPORT" in first
    assert "LATE-REPORT" not in second
    # Synced by the run markers, not by timing out the wait.
    assert loader._stderr_marks == 2
