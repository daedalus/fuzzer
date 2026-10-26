"""Regression: per-call subprocess loader stderr was unbounded, or dropped.

``InProcessRunner._run_c_subprocess`` read the loader's stderr with
``communicate()``: no cap, so a chatty target held all of it in memory. The
loader's standalone-executable path captured the target's stderr and never
relayed it, so its sanitizer reports were lost.
"""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.adapters.inprocess import _LOADER_SCRIPT, InProcessRunner

REPORT = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1"

# Well past the retained-stderr cap.
_NOISE_BYTES = 16 * process._STDERR_CAP

_SO_SRC = f"""
#include <stddef.h>
#include <stdio.h>
#include <string.h>
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {{
    static char noise[{_NOISE_BYTES}];
    if (n >= 1 && d[0] == 'N') {{
        memset(noise, 'n', sizeof noise);
        fwrite(noise, 1, sizeof noise, stderr);
    }}
    fputs("\\n{REPORT}\\n", stderr);
    return 0;
}}
"""


def _runner(tmp_path, target: str) -> SimpleNamespace:
    loader = tmp_path / "loader.py"
    loader.write_text(_LOADER_SCRIPT)
    return SimpleNamespace(
        _loader_path=str(loader),
        timeout=10.0,
        _shim=None,
        _bitmap_out=None,
        coverage_env_id=None,
        shm_size=8192,
        target=target,
        function_name="LLVMFuzzerTestOneInput",
    )


@pytest.fixture
def so(tmp_path) -> str:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "t.c"
    src.write_text(_SO_SRC)
    out = tmp_path / "t.so"
    subprocess.run(["clang", "-shared", "-fPIC", "-o", str(out), str(src)], check=True)
    return str(out)


def test_regression_loader_stderr_bounded(tmp_path, so) -> None:
    rc, stderr = InProcessRunner._run_c_subprocess(_runner(tmp_path, so), b"N")  # type: ignore[arg-type]

    assert rc == 0
    assert len(stderr) <= process._STDERR_CAP
    assert REPORT in stderr


def test_loader_short_stderr_whole(tmp_path, so) -> None:
    # Falsification: under the cap nothing is dropped.
    rc, stderr = InProcessRunner._run_c_subprocess(_runner(tmp_path, so), b"q")  # type: ignore[arg-type]

    assert (rc, stderr) == (0, f"\n{REPORT}\n")


def test_regression_loader_relays_executable_stderr(tmp_path) -> None:
    exe = tmp_path / "t.sh"
    exe.write_text(f"#!/bin/sh\necho '{REPORT}' >&2\nexit 1\n")
    exe.chmod(0o755)

    rc, stderr = InProcessRunner._run_c_subprocess(_runner(tmp_path, str(exe)), b"x")  # type: ignore[arg-type]

    assert rc == 1
    assert REPORT in stderr
