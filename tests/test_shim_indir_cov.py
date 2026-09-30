"""The shim must record (call site, callee) pairs for indirect calls.

``-fsanitize-coverage=indirect-calls`` makes clang call
``__sanitizer_cov_trace_pc_indir(callee)`` before every indirect call. Edge
coverage alone sees only the callee's entry block; the pair is a separate
synthetic channel (bit 31, like DATAFLOW/COMPCOV).

Real clang builds, like tests/test_shim_stack_depth.py.
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

SYNTH_BIT = 0x80000000
INDIR_MODE = "indirect-calls"

# Modes: a/b = site1 calls fa/fb, c = site2 calls fa, d = direct calls only,
# e = indirect call to a libc function (outside the target image).
_DRIVER = """
#include <stdlib.h>
#include <string.h>

typedef int (*fn_t)(int);

__attribute__((noinline)) static int fa(int x) { return x + 1; }
__attribute__((noinline)) static int fb(int x) { return x + 2; }

__attribute__((noinline)) static int site1(fn_t f, int x) {
    __asm__ volatile("" : "+r"(f));
    return f(x);
}

__attribute__((noinline)) static int site2(fn_t f, int x) {
    __asm__ volatile("" : "+r"(f));
    return f(x);
}

int main(int argc, char **argv) {
    if (argc < 2) return 0;
    switch (argv[1][0]) {
    case 'a': return site1(fa, 1) == 0x7fffffff;
    case 'b': return site1(fb, 1) == 0x7fffffff;
    case 'c': return site2(fa, 1) == 0x7fffffff;
    case 'd': return (fa(1) + fb(1)) == 0x7fffffff;
    case 'e': {
        int (*volatile ext)(int) = abs;
        return site1(ext, -3) == 0x7fffffff;
    }
    }
    return 0;
}
"""

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")


def _build(tmp_path, modes, *flags):
    src = tmp_path / "drv.c"
    src.write_text(_DRIVER)
    exe = tmp_path / f"drv_{abs(hash((modes, flags)))}"
    proc = subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            f"-fsanitize-coverage={modes}",
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


def _synth_ids(exe, mode):
    cov = ShmCoverage(size=65536)
    try:
        env = dict(
            os.environ,
            __AFL_SHM_ID=str(cov.shm_id),
            AFL_MAP_SIZE=str(cov.num_entries),
        )
        proc = subprocess.run([exe, mode], env=env, capture_output=True, timeout=10)
        assert proc.returncode == 0, f"target died: rc={proc.returncode}"
        return {i for i in cov.get_edge_ids() if i & SYNTH_BIT}
    finally:
        cov.cleanup()


@needs_clang
def test_indirect_call_records_synthetic_id(tmp_path):
    exe = _build(tmp_path, f"trace-pc-guard,{INDIR_MODE}")
    assert _synth_ids(exe, "a"), "indirect call left no synthetic id"


@needs_clang
def test_same_site_different_callee_differs(tmp_path):
    exe = _build(tmp_path, f"trace-pc-guard,{INDIR_MODE}")
    assert _synth_ids(exe, "a") != _synth_ids(exe, "b")


@needs_clang
def test_same_callee_different_site_differs(tmp_path):
    exe = _build(tmp_path, f"trace-pc-guard,{INDIR_MODE}")
    assert _synth_ids(exe, "a") != _synth_ids(exe, "c")


@needs_clang
def test_ids_stable_across_runs(tmp_path):
    """ASLR moves the image; ids must be base-relative."""
    exe = _build(tmp_path, f"trace-pc-guard,{INDIR_MODE}")
    assert _synth_ids(exe, "a") == _synth_ids(exe, "a")


@needs_clang
def test_direct_calls_leave_no_synthetic_id(tmp_path):
    """Falsification: ids must come from indirect calls, not from any call."""
    exe = _build(tmp_path, f"trace-pc-guard,{INDIR_MODE}")
    assert _synth_ids(exe, "d") == set()


@needs_clang
def test_build_without_mode_has_no_synthetic_id(tmp_path):
    """Falsification: the channel exists only when the build asks for it."""
    exe = _build(tmp_path, "trace-pc-guard")
    assert _synth_ids(exe, "a") == set()


@needs_clang
def test_callee_outside_image_is_ignored(tmp_path):
    """Adversarial: a libc callee has an ASLR-dependent offset, so it must not
    mint an id (it would differ every run and inflate the map)."""
    exe = _build(tmp_path, f"trace-pc-guard,{INDIR_MODE}")
    assert _synth_ids(exe, "e") == set()


@needs_clang
def test_indirect_calls_under_asan(tmp_path):
    """Adversarial: libasan ships weak stubs for the same symbol."""
    exe = _build(tmp_path, f"trace-pc-guard,{INDIR_MODE}", "-fsanitize=address")
    assert _synth_ids(exe, "a"), "callback lost to the sanitizer runtime's stub"
