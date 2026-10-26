"""Regression: fuzz_loader.c kept the stderr head, and the forkserver drained only after exit.

A target printing more than the 64 KiB pipe buffer blocked on write until the
timeout (forkserver path), and either path dropped the sanitizer report that
follows the noise.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.adapters.shm import ShmCoverage

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(_ROOT, "src", "fuzzer_tool", "adapters", "afl_shim.c")

_REPORT = "==7==ERROR: AddressSanitizer: heap-use-after-free on address 0x1"
_FAILING_EXIT = 3

# Noise past both the pipe buffer and the loader's MAX_ERR.
_NOISE_BYTES = 256 * 1024

_TARGET = f"""
#include <stdio.h>
#include <string.h>
#include <unistd.h>
int main(void) {{
    char buf[8];
    ssize_t n = read(0, buf, sizeof buf);
    if (n > 0 && buf[0] == 'N') {{
        static char noise[{_NOISE_BYTES}];
        memset(noise, 'n', sizeof noise);
        fwrite(noise, 1, sizeof noise, stderr);
    }}
    fputs("\\n{_REPORT}\\n", stderr);
    return {_FAILING_EXIT};
}}
"""


def _build(tmp_path, *extra: str) -> str:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    src = tmp_path / "t.c"
    src.write_text(_TARGET)
    exe = tmp_path / "t"
    r = subprocess.run(
        ["clang", "-O0", *extra, "-o", str(exe), str(src)], capture_output=True, text=True
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(exe)


@pytest.fixture(params=["forkserver", "exec"])
def runner(request, tmp_path):
    shim = ("-fsanitize-coverage=trace-pc-guard", "-include", SHIM)
    exe = _build(tmp_path, *(shim if request.param == "forkserver" else ()))
    shm = ShmCoverage(size=8192)
    r = ForkserverRunner(
        exe, timeout=5.0, env={"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size)}
    )
    try:
        assert r.start()
        yield r
    finally:
        r.stop()
        shm.cleanup()


def test_regression_report_after_noise_survives(runner) -> None:
    rc, stderr = runner.run_one(b"N")

    assert rc == _FAILING_EXIT
    assert _REPORT in stderr


def test_short_stderr_kept_whole(runner) -> None:
    # Falsification: no noise, nothing dropped.
    rc, stderr = runner.run_one(b"q")

    assert rc == _FAILING_EXIT
    assert stderr == f"\n{_REPORT}\n"
