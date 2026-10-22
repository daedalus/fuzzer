"""cmin replayed with ASLR on, so caller-context edge ids were random per exec.

afl_shim.c folds a raw return address into every edge id. ``fuzz`` calls
``disable_aslr()`` first; ``minimize`` did not. Every replay under PIE+ASLR
reported fresh ids, set-cover saw each seed as unique, kept far too many and
dropped seeds whose edges only looked covered.

Measured on fuzzgoat: 3070 "edges" (521 real), 430 -> 376 files, re-replay of
the kept set 2912. With ASLR off: 501 edges, 430 -> 131, re-replay 501.
"""

import os
import shutil
import subprocess
import sys

import pytest

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# 'a' runs path_a; 'c' runs path_a then one more edge. Same call sites, so
# with stable ids 'a' is a strict subset of 'c' and set-cover drops it.
_TARGET = """
#include <stdio.h>
static void fire(uint32_t g) { __sanitizer_cov_trace_pc_guard(&g); }
static void path_a(void) { fire(10); fire(11); }
int main(void) {
    int ch = getchar();
    fire(1);
    if (ch == 'a')      { path_a(); }
    else if (ch == 'b') { fire(20); fire(21); }
    else if (ch == 'c') { path_a(); fire(30); }
    return 0;
}
"""

# Stable ids: 'a' is dropped.
_KEPT_STABLE = 2

# Runs in a child so the persona it clears or sets never leaks into pytest.
# INHERIT=1 sets ADDR_NO_RANDOMIZE before minimize (the known-good control).
_LAUNCHER = """
import ctypes, ctypes.util, os, sys
sys.path.insert(0, {src!r})
from fuzzer_tool.adapters.process import ADDR_NO_RANDOMIZE
libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
libc.personality.argtypes = [ctypes.c_ulong]
cur = libc.personality(0xFFFFFFFF)
if os.environ.get("INHERIT") == "1":
    libc.personality(cur | ADDR_NO_RANDOMIZE)
else:
    libc.personality(cur & ~ADDR_NO_RANDOMIZE)
from fuzzer_tool.services.minimize import minimize_corpus
kept, _ = minimize_corpus({target!r}, {corpus!r}, use_coverage=True, output_dir={out!r})
print(kept, bool(libc.personality(0xFFFFFFFF) & ADDR_NO_RANDOMIZE))
"""

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="personality() is Linux-only"
)


@pytest.fixture(scope="module")
def ctx_target(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("no clang")
    d = tmp_path_factory.mktemp("cmin_aslr")
    src = d / "t.c"
    src.write_text(_TARGET)
    exe = d / "t"
    cmd = ["clang", "-O0", "-fPIE", "-pie", "-fno-omit-frame-pointer", "-include", SHIM]
    r = subprocess.run([*cmd, "-o", str(exe), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"shim failed to build: {r.stderr[:300]}")
    return str(exe)


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in "abc":
        (corpus / f"id_{name}").write_bytes(name.encode())
    return str(corpus)


def _minimize(target, tmp_path, env):
    """Run minimize in a fresh launcher; return (kept, ASLR off afterwards)."""
    code = _LAUNCHER.format(
        src=_SRC, target=target, corpus=_corpus(tmp_path), out=str(tmp_path / "out")
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    kept, no_aslr = r.stdout.split()[-2:]
    return int(kept), no_aslr == "True"


@linux_only
def test_control_aslr_off_drops_subset(ctx_target, tmp_path):
    """Control: inherited ADDR_NO_RANDOMIZE gives the correct cover."""
    kept, _ = _minimize(ctx_target, tmp_path, {"INHERIT": "1", "FUZZER_KEEP_ASLR": "0"})
    assert kept == _KEPT_STABLE


@linux_only
def test_regression_minimize_disables_aslr(ctx_target, tmp_path):
    """Falsification: launched with ASLR on, minimize must still drop 'a'."""
    kept, no_aslr = _minimize(ctx_target, tmp_path, {"INHERIT": "0", "FUZZER_KEEP_ASLR": "0"})
    assert no_aslr, "minimize left ASLR on"
    assert kept == _KEPT_STABLE, f"kept {kept}: edge ids varied across replays"


@linux_only
def test_keep_aslr_opt_out_respected(ctx_target, tmp_path):
    """Adversarial: FUZZER_KEEP_ASLR=1 keeps ASLR; shim's base-relative ids hold."""
    kept, no_aslr = _minimize(ctx_target, tmp_path, {"INHERIT": "0", "FUZZER_KEEP_ASLR": "1"})
    assert not no_aslr
    assert kept == _KEPT_STABLE
