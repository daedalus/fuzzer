"""An abandoned call on the shared glibc heap taints it, whatever abandoned it."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters import libc_mem
from fuzzer_tool.adapters.inprocess import InProcessRunner
from tests.conftest import requires_clang

SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"

_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
int fuzz_shm_run(const unsigned char *d, size_t n) {
    if (n && d[0] == 'S') { volatile int *p = 0; *p = 1; }
    if (n && d[0] == 'T') { for (;;) { void *q = malloc(16); free(q); } }
    return 0;
}
"""


@pytest.fixture(scope="module")
def so(tmp_path_factory):
    d = tmp_path_factory.mktemp("taint")
    (d / "t.c").write_text(_SRC)
    for name, extra in (("plain", []), ("priv", ["-D__AFL_PRIVATE_HEAP=1"])):
        cmd = ["clang", "-O0", "-shared", "-fPIC", "-fno-builtin", "-include", str(SHIM), *extra, "-o", str(d / f"{name}.so"), str(d / "t.c")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-800:]
    return d


@pytest.fixture(autouse=True)
def _clean_taint(monkeypatch):
    monkeypatch.setattr(libc_mem, "_heap_tainted", False)


def _run(path, data):
    r = InProcessRunner(target=str(path), function_name="fuzz_shm_run", timeout=1, direct_lite=True)
    return r.run_one(data)[0]


@requires_clang
@pytest.mark.parametrize(("data", "rc"), [(b"S" * 16, -11), (b"T" * 16, -1)])
def test_segv_and_timeout_taint_a_shared_heap(so, data, rc):
    assert _run(so / "plain.so", data) == rc
    assert libc_mem.heap_tainted()


@requires_clang
@pytest.mark.parametrize("data", [b"S" * 16, b"T" * 16])
def test_private_heap_stays_untainted(so, data):
    _run(so / "priv.so", data)
    assert not libc_mem.heap_tainted()


@requires_clang
def test_clean_call_does_not_taint(so):
    assert _run(so / "plain.so", b"x" * 16) == 0
    assert not libc_mem.heap_tainted()
