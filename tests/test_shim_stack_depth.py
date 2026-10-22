"""The shim must publish the max stack depth of the run at SHM offset 0.

``__afl_max_stack_depth`` used to be reset to 0 and copied to the header,
never assigned in between: ``read_stack_depth()`` was always 0 and the
stack-depth boost in ``core/schedules.py`` never fired.

Real ``-fsanitize-coverage=trace-pc-guard`` builds under clang, like
tests/test_shim_ctx_instrumented.py.
"""

import inspect
import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.shim_health import STACK_DEPTH_ENV, export_stack_depth
from fuzzer_tool.services.fuzzer import Fuzzer

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

FRAME_BYTES = 256
SHALLOW_LIMIT = 4 * 1024
BOOST_FLOOR = 16 * 1024
THREAD_LIMIT = 1024 * 1024

# Each frame holds a FRAME_BYTES buffer whose address escapes into an asm
# barrier, so the optimizer cannot shrink it: depth n costs at least
# n * FRAME_BYTES of stack. Mode "t" runs shallow work on the
# main thread and on a second thread with its own stack.
_DRIVER = """
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define FRAME @FRAME@

__attribute__((noinline)) static int rec(int n) {
    char pad[FRAME];
    pad[0] = (char)n;
    __asm__ volatile("" : : "r"(pad) : "memory");
    if (n <= 0) return pad[0];
    return rec(n - 1) + pad[0];
}

static void *worker(void *arg) {
    (void)arg;
    return (void *)(long)rec(2);
}

int main(int argc, char **argv) {
    if (argc < 3) return 0;
    int n = atoi(argv[2]);
    if (argv[1][0] == 't') {
        pthread_t t;
        int r = rec(2);
        pthread_create(&t, 0, worker, 0);
        pthread_join(t, 0);
        return r == 0x7fffffff;
    }
    return rec(n) == 0x7fffffff;
}
""".replace("@FRAME@", str(FRAME_BYTES))

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")


def _build(tmp_path, *flags):
    src = tmp_path / "drv.c"
    src.write_text(_DRIVER)
    exe = tmp_path / "drv"
    proc = subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize-coverage=trace-pc-guard",
            *flags,
            "-include",
            SHIM,
            "-o",
            str(exe),
            str(src),
            "-ldl",
            "-lpthread",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"shim build failed: {proc.stderr[:400]}"
    return str(exe)


def _depth(exe, mode, n, **extra):
    cov = ShmCoverage(size=65536)
    try:
        env = dict(
            os.environ,
            __AFL_SHM_ID=str(cov.shm_id),
            AFL_MAP_SIZE=str(cov.num_entries),
            **extra,
        )
        proc = subprocess.run([exe, mode, str(n)], env=env, capture_output=True, timeout=10)
        assert proc.returncode == 0, f"target died: rc={proc.returncode}"
        return cov.read_stack_depth()
    finally:
        cov.cleanup()


@needs_clang
def test_deep_recursion_reports_depth(tmp_path):
    exe = _build(tmp_path)
    depth = _depth(exe, "r", 400)
    assert depth >= 400 * FRAME_BYTES, f"depth {depth} below the recursion's own stack use"
    assert depth > BOOST_FLOOR, "deep input must clear the schedules.py boost threshold"


@needs_clang
def test_depth_grows_with_recursion(tmp_path):
    exe = _build(tmp_path)
    shallow = _depth(exe, "r", 100)
    deep = _depth(exe, "r", 800)
    assert deep > shallow, f"depth not monotonic: n=100 -> {shallow}, n=800 -> {deep}"
    assert deep - shallow >= 600 * FRAME_BYTES


@needs_clang
def test_shallow_input_reports_small_depth(tmp_path):
    """Falsification: a nonzero value must come from depth, not a constant."""
    exe = _build(tmp_path)
    assert _depth(exe, "r", 1) < SHALLOW_LIMIT


@needs_clang
def test_other_thread_stack_is_not_depth(tmp_path):
    """Adversarial: a second thread's stack sits at unrelated addresses."""
    exe = _build(tmp_path)
    assert _depth(exe, "t", 0) < THREAD_LIMIT


@needs_clang
def test_deep_recursion_reports_depth_under_asan(tmp_path):
    exe = _build(tmp_path, "-fsanitize=address")
    assert _depth(exe, "r", 400) >= 400 * FRAME_BYTES


# ── __AFL_STACK_DEPTH: off when nothing reads it ───────────────────────
# Only the --honggfuzz power schedule reads the depth; elsewhere the per-edge
# frame sampling was paid for nothing.


@needs_clang
def test_depth_off_skips_tracking(tmp_path):
    """Falsification: =0 publishes no depth for the same deep recursion."""
    exe = _build(tmp_path)
    assert _depth(exe, "r", 400, **{STACK_DEPTH_ENV: "0"}) == 0


@needs_clang
@pytest.mark.parametrize("value", ["1", "", "yes"])
def test_depth_on_unless_exactly_zero(tmp_path, value):
    """Adversarial: "1", empty and a typo all keep the build default (on)."""
    exe = _build(tmp_path)
    assert _depth(exe, "r", 400, **{STACK_DEPTH_ENV: value}) >= 400 * FRAME_BYTES


def test_export_stack_depth(monkeypatch):
    monkeypatch.setenv(STACK_DEPTH_ENV, "x")
    export_stack_depth(False)
    assert os.environ[STACK_DEPTH_ENV] == "0"
    export_stack_depth(True)
    assert os.environ[STACK_DEPTH_ENV] == "1"


def test_fuzzer_exports_its_reader():
    """The honggfuzz schedule is the only reader, so it decides."""
    assert "export_stack_depth(honggfuzz)" in inspect.getsource(Fuzzer.__init__)
