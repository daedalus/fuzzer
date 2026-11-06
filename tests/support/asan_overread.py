"""ASAN ``.so`` whose ``fuzz_shm_run`` reads ``buf[n]`` when ``buf[0] == 'R'``.

Shared by the exact-size input tests (loader and in-process). Other first
bytes stay in bounds, so a large safe input can leave a stale buffer behind
before the overread probe runs.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.cli.ldpreload_wrapper import _resolve_asan

#: First byte that makes the target read one past the input.
OVERREAD = ord("R")

#: ASAN report kind the overread must produce.
REPORT = "heap-buffer-overflow"

_SRC = r"""
#include <stddef.h>
int fuzz_shm_run(const unsigned char *b, size_t n) {
    if (n == 0 || b[0] != 'R') return 0;
    return b[n];
}
"""


def asan_env(base: dict[str, str], log_dir) -> dict[str, str]:
    """*base* plus libasan preloaded, reports to *log_dir*; skip without libasan.

    A file, not stderr: direct mode points fd 2 at its capture pipe, and an
    aborting report written there dies with the process.
    """
    libasan = _resolve_asan()
    if libasan is None:
        pytest.skip("libasan not installed")
    opts = f"detect_leaks=0:log_path={log_dir / 'asan'}"
    return {**base, "LD_PRELOAD": libasan, "ASAN_OPTIONS": opts}


def asan_reports(log_dir) -> str:
    """All ASAN reports written under *log_dir* (``asan.<pid>`` files)."""
    return "".join(p.read_text(errors="replace") for p in sorted(log_dir.glob("asan.*")))


def build_overread_so(tmp_path) -> str:
    """Compile the target into *tmp_path*; skip when clang/libasan are missing."""
    if not shutil.which("clang"):
        pytest.skip("clang not installed")

    src = tmp_path / "overread.c"
    obj = tmp_path / "overread.o"
    so = tmp_path / "overread.so"
    src.write_text(_SRC)

    # Two passes: clang -shared -fsanitize=address wants its own static
    # runtime; linking gcc's libasan explicitly matches tools/build_targets.sh.
    steps = (
        ["clang", "-O0", "-fsanitize=address", "-fPIC", "-c", str(src), "-o", str(obj)],
        ["clang", "-shared", str(obj), "-o", str(so), "-lasan"],
    )
    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"ASAN target failed to build: {r.stderr[:300]}")
    return str(so)
