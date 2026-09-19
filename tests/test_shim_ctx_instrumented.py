"""A real trace-pc-guard build must not re-enter the coverage callback.

dd834d1 added ``__afl_ctx_resolve_base()`` and ``__afl_ctx_use_relative()``
to the caller-context path without ``__AFL_NO_COV``.  Both are ``static
inline``, and at -O2 clang leaves them out of line -- ``nm`` shows them as
local text symbols -- so in a build that actually carries
``-fsanitize-coverage=trace-pc-guard`` they are instrumented like any other
function in the translation unit.  They are called from inside
``__afl_map_edge``, which runs inside the callback, so each hit re-enters the
callback and the target dies of stack exhaustion on its first edge.

tests/test_edge_id_stability_guard.py and tests/test_ctx_and_map_size.py
cannot see this by construction: their drivers invoke
``__sanitizer_cov_trace_pc_guard`` directly and build under gcc, which has no
trace-pc-guard support, so no part of the shim is ever instrumented there.
Only a clang build with real instrumentation reaches the recursion, which is
what this file adds.

Measured on the unfixed shim (clang 18, -O2, PIE): SIGSEGV on the first
input, two edges recorded out of 59.  With ``__AFL_NO_COV`` on both helpers:
exit 0, 59 edges.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

# Branches and a nested call, so the guards fire from more than one frame --
# a straight-line main would exercise the ctx path only once.
_DRIVER = """
#include <stdlib.h>
#include <string.h>

static int leaf(int v) { return v > 3 ? v * 2 : v - 1; }

static int branchy(const char *s) {
    int acc = 0;
    for (size_t i = 0; s[i]; i++) {
        if (s[i] == 'a') acc += leaf((int)i);
        else if (s[i] == 'b') acc -= leaf((int)i);
        else acc ^= leaf((int)i);
    }
    return acc;
}

int main(int argc, char **argv) {
    if (argc < 2) return 0;
    return branchy(argv[1]) == 0x7fffffff;
}
"""

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")


def _build(tmp_path, *flags):
    src = tmp_path / "drv.c"
    src.write_text(_DRIVER)
    exe = tmp_path / "drv"
    proc = subprocess.run(
        [
            "clang",
            "-O2",
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
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"shim failed to build under clang: {proc.stderr[:300]}")
    return str(exe)


def _run(exe, env_extra=None):
    """Execute under a live SHM map; return (returncode, edges recorded)."""
    cov = ShmCoverage(size=65536)
    try:
        env = dict(
            os.environ,
            __AFL_SHM_ID=str(cov.shm_id),
            AFL_MAP_SIZE=str(cov.num_entries),
            **(env_extra or {}),
        )
        proc = subprocess.run([exe, "abcabcabc"], env=env, capture_output=True, timeout=10)
        return proc.returncode, len(cov.get_edge_ids())
    finally:
        cov.cleanup()


@needs_clang
def test_instrumented_ctx_build_survives_its_own_callback(tmp_path):
    exe = _build(tmp_path)
    rc, edges = _run(exe)
    assert rc >= 0, (
        f"target died of signal {-rc} on its first edges -- the ctx path is calling an "
        "instrumented helper from inside the coverage callback (missing __AFL_NO_COV)"
    )
    assert edges > 1, "no coverage recorded from an instrumented build"


@needs_clang
def test_relative_mode_is_reached_without_recursing(tmp_path):
    """FUZZER_KEEP_ASLR=1 is the path that calls __afl_ctx_resolve_base()."""
    exe = _build(tmp_path)
    rc, edges = _run(exe, {"FUZZER_KEEP_ASLR": "1"})
    assert rc >= 0, f"target died of signal {-rc} in base-relative context mode"
    assert edges > 1


@needs_clang
def test_context_free_build_is_unaffected(tmp_path):
    """The control: with ctx off neither helper is reachable."""
    exe = _build(tmp_path, "-D__AFL_CTX_SENSITIVE=0")
    rc, edges = _run(exe)
    assert rc >= 0
    assert edges > 1
