"""Regression test for the canonical-address check in __afl_get_caller_ctx().

Before the fix, a garbage saved frame pointer in the x86-64 non-canonical
gap (e.g. 0x800000010000) passed the 4MiB distance check and caused SIGSEGV
on the second-hop dereference inside __afl_get_caller_ctx().  The fix adds
a canonical-address guard (cfp >> 47) before the second hop.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from tests.conftest import requires_clang

SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"


@requires_clang
def test_regression_noncanonical_frameptr_no_sigsegv(tmp_path):
    """A saved frame pointer in the non-canonical gap (0x800000010000)
    must not cause SIGSEGV in __afl_get_caller_ctx().  The shim should
    reject it and return 0 (no context) instead of crashing.

    Builds a standalone executable that corrupts its saved frame pointer
    before firing a coverage trace, then runs it as a subprocess.
    """
    src = tmp_path / "t.c"
    src.write_text(
        textwrap.dedent("""
        #include <stdint.h>
        #include <stddef.h>
        #include <unistd.h>
        /* Entry that fires trace_pc_guard via the shim's coverage path. */
        int main(void) {
            unsigned long *rbp = (unsigned long *)__builtin_frame_address(0);
            *rbp = 0x800000010000ul;  /* non-canonical: bits 63:47 != all-0/all-1 */
            asm volatile("" : : : "memory");  /* prevent dead-store elimination */
            /* Drain stdin so the fuzzer's run_target_fast path works. */
            char buf[4096];
            (void)!read(0, buf, sizeof(buf));
            return 0;
        }
    """)
    )
    exe = tmp_path / "t"
    cc = [
        "clang",
        "-O1",
        "-g",
        "-fno-omit-frame-pointer",
        "-include",
        str(SHIM),
        "-o",
        str(exe),
        str(src),
    ]
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    r = subprocess.run(
        [str(exe)],
        input=b"test",
        capture_output=True,
        timeout=10,
    )
    assert r.returncode == 0, f"SIGSEGV or crash: stdout={r.stdout!r} stderr={r.stderr!r}"
