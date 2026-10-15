"""Regression: ``__afl_ctx_use_relative()`` must inline into the guard path.

It was ``static inline`` but clang -O2 kept it out of line (its body holds
the one-time ``getenv``), so every context-sensitive edge paid a call/ret:
~0.3 ns/edge on ``tools/shim-micro-benchmark.c`` ``ctx`` mode.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage

_SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)
_HELPER = "__afl_ctx_use_relative"

# Nested calls so guards fire from more than one frame.
_DRIVER = """
static int leaf(int v) { return v > 3 ? v * 2 : v - 1; }

int main(int argc, char **argv) {
    int acc = 0;
    if (argc < 2) return 0;
    for (int i = 0; argv[1][i]; i++) acc += argv[1][i] == 'a' ? leaf(i) : -leaf(i);
    return acc == 0x7fffffff;
}
"""


def _build(tmp_path) -> str:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")

    src = tmp_path / "drv.c"
    src.write_text(_DRIVER)
    exe = tmp_path / "drv"
    flags = ["-O2", "-fno-omit-frame-pointer", "-fsanitize-coverage=trace-pc-guard"]
    proc = subprocess.run(
        ["clang", *flags, "-include", _SHIM, "-o", str(exe), str(src), "-ldl"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return str(exe)


def _run(exe: str, keep_aslr: str) -> tuple[int, int]:
    """Run under a live SHM map; return (returncode, edges recorded)."""
    cov = ShmCoverage(size=65536)
    try:
        env = dict(
            os.environ,
            __AFL_SHM_ID=str(cov.shm_id),
            AFL_MAP_SIZE=str(cov.num_entries),
            FUZZER_KEEP_ASLR=keep_aslr,
        )
        proc = subprocess.run([exe, "abab"], env=env, capture_output=True, timeout=10)
        return proc.returncode, len(cov.get_edge_ids())
    finally:
        cov.cleanup()


def test_regression_ctx_use_relative_inline(tmp_path):
    exe = _build(tmp_path)
    syms = subprocess.run(["nm", exe], capture_output=True, text=True, check=True).stdout
    assert _HELPER not in syms, f"{_HELPER} left out of line"


@pytest.mark.parametrize("keep_aslr", ["0", "1", "1x", ""])
def test_ctx_use_relative_inline_env(tmp_path, keep_aslr):
    """Adversarial: every env form, incl. malformed, still runs and records edges."""
    rc, edges = _run(_build(tmp_path), keep_aslr)
    assert rc >= 0, f"signal {-rc} with FUZZER_KEEP_ASLR={keep_aslr!r}"
    assert edges > 1
