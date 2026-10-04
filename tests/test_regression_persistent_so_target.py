"""Persistent mode must not execve() a shared object.

--hail-mary forces persistent mode on. PersistentRunner.start() execve()s the
target; a shared object has no PT_INTERP and e_entry 0, so the kernel jumps to
its base (the non-executable ELF header) and the child segfaults (error 0x15,
ip == mapping base) on every attempt, with a kernel log line each time.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from fuzzer_tool.adapters import persistent_signal
from fuzzer_tool.services.fuzzer import Fuzzer

CC = shutil.which("clang") or shutil.which("gcc")
pytestmark = pytest.mark.skipif(CC is None, reason="C compiler required")

SRC = "int fuzz_shm_run(const unsigned char *d, unsigned long n) { return 0; }\n"
MAIN = "int main(void) { return 0; }\n"


def _cc(tmp_path: Path, name: str, src: str, *flags: str) -> Path:
    c = tmp_path / f"{name}.c"
    c.write_text(src)
    out = tmp_path / name
    subprocess.run([CC, *flags, "-o", str(out), str(c)], check=True)
    return out


def _fuzzer(tmp_path: Path, target: Path, **kw) -> Fuzzer:
    return Fuzzer(
        target=str(target),
        corpus_dir=str(tmp_path / "corpus"),
        crashes_dir=str(tmp_path / "crashes"),
        persistent=True,
        max_len=64,
        timeout=1,
        mutations_per_input=1,
        cmplog=False,
        **kw,
    )


@pytest.mark.parametrize("inproc", [{}, {"inprocess_direct": True}, {"inprocess": True}])
def test_regression_so_never_exec_persistent(tmp_path, inproc):
    """Falsification: a .so with persistent=True never reaches execve."""
    so = _cc(tmp_path, "t.so", SRC, "-shared", "-fPIC")

    with mock.patch.object(persistent_signal.os, "execve") as ex:
        f = _fuzzer(tmp_path, so, **inproc)

    ex.assert_not_called()
    assert f.persistent is False
    assert f._persistent_runner is None


def test_regression_pie_exe_keeps_persistent(tmp_path):
    """Adversarial: a real executable is still allowed to try persistent mode."""
    exe = _cc(tmp_path, "t", SRC + MAIN, "-fPIE", "-pie")

    with mock.patch.object(persistent_signal.PersistentRunner, "start", return_value=False) as st:
        f = _fuzzer(tmp_path, exe)

    st.assert_called_once()
    # start() failed -> fell back to fork, and the flag says so
    assert f._persistent_runner is None
    assert f.persistent is False


def test_regression_persistent_started_keeps_flag(tmp_path):
    """Adversarial: when start() succeeds the flag stays True."""
    exe = _cc(tmp_path, "t", SRC + MAIN, "-fPIE", "-pie")

    with mock.patch.object(persistent_signal.PersistentRunner, "start", return_value=True):
        f = _fuzzer(tmp_path, exe)

    assert f.persistent is True
    assert f._persistent_runner is not None


def test_regression_auto_tune_timeout_never_execs_so(tmp_path):
    """--auto-timeout (forced on by --hail-mary) must not exec a shared object.

    _auto_tune_timeout ran the target 10 times through run_target_file/stdin;
    on a .so each run died on an NX fetch at the mapping base and the tuned
    "timeout" was just the time-to-crash (the 0.05s floor).
    """
    from fuzzer_tool.cli import commands

    so = _cc(tmp_path, "t.so", SRC, "-shared", "-fPIC")

    with (
        mock.patch("fuzzer_tool.adapters.process.run_target_file") as rf,
        mock.patch("fuzzer_tool.adapters.process.run_target_stdin") as rs,
    ):
        assert commands._auto_tune_timeout(str(so), file_mode=True) is None
        assert commands._auto_tune_timeout(str(so), file_mode=False) is None

    rf.assert_not_called()
    rs.assert_not_called()


def test_regression_auto_tune_timeout_still_tunes_executables(tmp_path):
    """Adversarial: a real executable is still measured."""
    from fuzzer_tool.cli import commands

    exe = _cc(tmp_path, "t", SRC + MAIN, "-fPIE", "-pie")

    with mock.patch("fuzzer_tool.adapters.process.run_target_stdin") as rs:
        got = commands._auto_tune_timeout(str(exe), file_mode=False, runs=3)

    assert rs.call_count == 3
    assert got == pytest.approx(0.05, abs=1.0)
