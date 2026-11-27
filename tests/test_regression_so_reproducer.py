"""Regression: crash reproducers for .so targets exec'd the library.

``format_reproducer`` / ``_build_repro`` piped the input into the target
path. A shared object is not an executable (Hard Rule 63): the GStreamer
qtdemux assert reproducer segfaulted in the loader instead of aborting.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.core.analyzers.analyzer_trace import CrashTracer, TraceReport
from fuzzer_tool.core.crash_metadata import CrashMetadata

SIGABRT_RC = 134  # bash: 128 + SIGABRT
CRASH = b"BOOM"

LIB_C = r"""
#include <stdlib.h>
#include <string.h>
int fuzz_shm_run(const unsigned char *d, size_t n)
{
    if (n == 4 && !memcmp(d, "BOOM", 4)) abort();
    return 0;
}
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """A real .so whose entry point aborts on CRASH; path has a space and a quote."""
    clang = shutil.which("clang")
    if not clang:
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("so repro")
    src = d / "t.c"
    src.write_text(LIB_C)
    so = d / "it's lib.so"
    subprocess.run([clang, "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    return str(so)


def _script(lib: str, data: bytes, tmp_path) -> int:
    meta = CrashMetadata()
    sh = tmp_path / "repro.sh"
    sh.write_text(meta.format_reproducer(data, lib))
    return subprocess.run(["bash", str(sh)], capture_output=True).returncode


def test_regression_so_reproducer(lib, tmp_path):
    """The script dlopens the .so and the crash input aborts."""
    assert _script(lib, CRASH, tmp_path) == SIGABRT_RC


def test_benign_input_exits_clean(lib, tmp_path):
    """Falsification: a non-crashing input exits 0."""
    assert _script(lib, b"ok", tmp_path) == 0


def test_trace_repro_cmd(lib):
    """The .trace repro command dlopens too."""
    report = TraceReport()
    analyzer = CrashTracer.__new__(CrashTracer)
    analyzer.target_path = lib
    analyzer._build_repro(CRASH, report)
    rc = subprocess.run(["bash", "-c", report.repro_cmd], capture_output=True).returncode
    assert rc == SIGABRT_RC


def test_binary_target_unchanged():
    """Executables are still fed on stdin."""
    out = CrashMetadata().format_reproducer(CRASH, "/usr/bin/target")
    assert "  /usr/bin/target" in out and "ctypes" not in out
