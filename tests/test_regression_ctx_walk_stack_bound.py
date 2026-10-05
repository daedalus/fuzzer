"""Regression: __afl_get_caller_ctx() faulted on a junk saved frame pointer.

Crash seen fuzzing a frame-pointer-less ffmpeg build in --inprocess-direct mode:
SIGSEGV (si_code=SI_KERNEL, fault addr 0) inside __sanitizer_cov_trace_pc_guard,
rbp-chain value 0x800000010000 -- a general-purpose register value left in rbp by
an -fomit-frame-pointer caller (s337m_probe). The walk only bounded the hop length
(<= 4 MiB above the shim frame), so a value a few KiB above the shim frame but past
the stack end passed the check and the load of caller_fp[1] hit unmapped or
non-canonical memory.

This file also supersedes tests/test_regression_shim_canonical.py, which never
reached the callback (its main() carries no coverage instrumentation, so it passed
on the unfixed shim too). Case 2 below is that test's 0x800000010000 value, now
driven through __sanitizer_cov_trace_pc_guard with SHM attached.

The driver reproduces that exactly: an asm trampoline puts a junk value in rbp and
calls the coverage callback, so the callback's prologue saves it as the "caller FP".
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "fuzzer_tool", "adapters", "afl_shim.c",
)

pytestmark = [
    pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler"),
    pytest.mark.skipif(os.uname().machine != "x86_64", reason="x86-64 asm trampoline"),
]

_DRIVER = r"""
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
extern void call_with_rbp(uint32_t *guard, uintptr_t junk);
__asm__(
    ".text\n.globl call_with_rbp\ncall_with_rbp:\n"
    "  push %rbp\n  mov %rsi, %rbp\n  call __sanitizer_cov_trace_pc_guard\n"
    "  pop %rbp\n  ret\n");

static uintptr_t stack_hi(void) {
    pthread_attr_t a; void *lo; size_t sz;
    pthread_getattr_np(pthread_self(), &a);
    pthread_attr_getstack(&a, &lo, &sz);
    return (uintptr_t)lo + sz;
}

int main(int argc, char **argv) {
    uintptr_t hi = stack_hi();
    uintptr_t cases[] = {
        hi + 0x10000,        /* just past the stack end, inside the 4 MiB span */
        0x800000010000ULL,   /* the value from the real crash */
        hi - 8,              /* straddles the end: caller_fp[1] is past it */
        hi - 15,             /* misaligned */
        1, 0,
    };
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        uint32_t g = i + 1;
        call_with_rbp(&g, cases[i]);
    }
    puts("ok");
    return 0;
}
"""


def _build(tmp_path):
    src = tmp_path / "drv.c"
    src.write_text("#include <stdint.h>\n" + _DRIVER)
    exe = tmp_path / "drv"
    r = subprocess.run(
        ["gcc", "-O1", "-g", "-D__AFL_CTX_SENSITIVE=1", "-fno-omit-frame-pointer",
         "-include", SHIM, "-o", str(exe), str(src)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"shim failed to build: {r.stderr[:300]}")
    return str(exe)


@pytest.mark.parametrize("aslr", [False, True])
def test_junk_saved_fp_past_stack_end_does_not_fault(tmp_path, aslr):
    exe = _build(tmp_path)
    cmd = [exe] if aslr else ["setarch", "-R", exe]
    if not aslr and shutil.which("setarch") is None:
        pytest.skip("no setarch")
    # Without an attached SHM map __afl_area is NULL and the callback returns
    # before the frame walk, so the test would pass vacuously.
    shm = ShmCoverage(size=1024)
    try:
        env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": "1024"}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    finally:
        shm.cleanup()
    assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr[:300]}"
    assert "ok" in r.stdout
