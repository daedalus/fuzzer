"""Regression: direct modes resolved shim symbols on every exec.

``_call_guarded`` looked up ``__afl_guarded_call_timeout`` /
``__afl_guarded_call``, reassigned ``restype``/``argtypes`` and re-cast the
entry pointer per exec; ``_flush_distance_tail`` looked up
``__afl_dist_flush`` per exec (a failing ``dlsym`` on non-distance builds).
They are now bound once at load.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters import inprocess
from fuzzer_tool.adapters.inprocess import InProcessRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
SIGSEGV_RC = 128 + 11

PROBE_SRC = r"""
#include <stddef.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    if (len && buf[0] == 'S') { volatile int *p = 0; *p = 1; }
    return len ? buf[0] : 0;
}
"""


class _NoLookup:
    """Stands in for the CDLL: any symbol lookup fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"per-exec symbol lookup: {name}")


@pytest.fixture(scope="module")
def probe_so(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    build = tmp_path_factory.mktemp("guard_bind")
    src = build / "probe.c"
    src.write_text(PROBE_SRC)
    so = build / "probe_guard.so"
    proc = subprocess.run(
        ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"probe target failed to build: {proc.stderr[-400:]}")
    return str(so)


@pytest.fixture
def lite(probe_so):
    r = InProcessRunner(probe_so, "fuzz_test", direct_lite=True, shm_size=4096, timeout=5.0)
    r._start()
    yield r
    r.stop()


def test_regression_guard_bind_once(lite, monkeypatch):
    """After load, an exec performs no symbol lookup and no pointer cast."""
    monkeypatch.setattr(lite, "_lib", _NoLookup())

    def _no_cast(*_a, **_k):
        raise AssertionError("per-exec ctypes.cast")

    monkeypatch.setattr(inprocess.ctypes, "cast", _no_cast)
    assert lite.run_one(b"\x05") == (5, "")


def test_guard_still_catches_crash(probe_so):
    """Adversarial: the bound guard still escapes a real SIGSEGV."""
    r = InProcessRunner(probe_so, "fuzz_test", direct=True, shm_size=4096, timeout=5.0)
    r._start()
    try:
        rc, _ = r.run_one(b"S")
        assert rc == SIGSEGV_RC
        assert r.run_one(b"\x03")[0] == 3
    finally:
        r.stop()


def test_stop_unbinds(lite):
    """Falsification: after stop() nothing bound survives to be called."""
    lite.stop()
    assert lite.run_one(b"\x01") == (-2, "runner not initialized")
    assert lite._dist_flush is None
    assert lite._guard_fn is None


def test_runner_freed_without_gc(probe_so):
    """Adversarial: binding must not create a cycle that defers __del__/stop()."""
    import gc
    import weakref

    r = InProcessRunner(probe_so, "fuzz_test", direct_lite=True, shm_size=4096, timeout=5.0)
    r._start()
    ref = weakref.ref(r)
    gc.disable()
    try:
        del r
        assert ref() is None
    finally:
        gc.enable()
