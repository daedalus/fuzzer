"""Regression: interpreter shutdown must not spew EBADF from runner teardown.

A runner still alive when the interpreter exits takes the __del__ ->
stop() -> _close_streams() path. _close_streams() deliberately skips
``stream.close()`` on the wrapper objects while ``sys.is_finalizing()``
(the drain thread may hold a stream's lock, and a frozen daemon thread
never releases it) -- but the fds were already given away by os.close(),
so the stdin/stdout wrappers' own dealloc at finalization flushes/reads a
closed fd and raises:

    OSError: [Errno 9] Bad file descriptor

printed as an "Exception ignored in" traceback on *every* clean exit that
left an un-stopped runner behind. Only stderr is contended by a persistent
drain thread; stdin/stdout wrappers are safe to close even while finalizing.
"""

import os
import shutil
import subprocess
import sys

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.adapters.shm import ShmCoverage


def _top_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SHIM = os.path.join(_top_dir(), "src", "fuzzer_tool", "adapters", "afl_shim.c")

_TARGET = """
#include <stdlib.h>
#include <unistd.h>
int main(void) {
    char buf[64];
    ssize_t n = read(0, buf, sizeof buf);
    if (n > 0) return buf[0] % 255;
    return 0;
}
"""


@pytest.fixture
def target(tmp_path):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "t.c"
    src.write_text(_TARGET)
    exe = tmp_path / "t"
    r = subprocess.run(
        [
            "clang",
            "-O0",
            "-fsanitize-coverage=trace-pc-guard",
            "-include",
            SHIM,
            "-o",
            str(exe),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(exe)


def _child_proc(target: str) -> subprocess.CompletedProcess:
    child = (
        "import sys;"
        "from fuzzer_tool.adapters.forkserver import ForkserverRunner;"
        "from fuzzer_tool.adapters.shm import ShmCoverage;"
        f"shm = ShmCoverage(size=8192);"
        f"r = ForkserverRunner({target!r}, timeout=2.0,"
        f"  env={{'__AFL_SHM_ID': shm.env_id, 'AFL_MAP_SIZE': str(shm.size)}});"
        "assert r.start();"
        "assert r.run_one(b'AAAA')[0] >= 0;"
        "# no stop(): interpreter shutdown must run __del__ -> stop()"
    )
    return subprocess.run(
        [sys.executable, "-c", child],
        env={**os.environ, "PYTHONPATH": _top_dir()},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_no_ebadf_in_teardown_on_interpreter_exit(target):
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    proc = _child_proc(target)
    assert proc.returncode == 0, proc.stderr
    assert "Bad file descriptor" not in proc.stderr
    assert "Errno 9" not in proc.stderr


def test_explicit_stop_still_cleans_up_quietly(tmp_path, target):
    """The explicit-stop path must remain noise-free too."""
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    shm = ShmCoverage(size=8192)
    r = ForkserverRunner(
        target,
        timeout=2.0,
        env={"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size)},
    )
    try:
        assert r.start()
        assert r.run_one(b"AAAA")[0] >= 0
    finally:
        r.stop()
        shm.cleanup()
