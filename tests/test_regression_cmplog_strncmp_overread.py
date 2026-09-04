"""strncmp/strncasecmp cmplog logging must not read past the shorter operand.

The n a caller passes to strncmp is an upper bound, not a readable length:
the standard says the comparison stops at the first NUL or first mismatch, so

    char *p = malloc(2); p[0] = 'h'; p[1] = 'x';
    strncmp(p, "http:", 5);

is legal C -- real strncmp compares two bytes and returns. The interceptor
logged n bytes of both operands regardless, so it read three bytes past the
end of p. FFmpeg's url_find_protocol does exactly this shape against the
protocol table, and a campaign against ffmpeg_read surfaced it as an ASAN
heap-buffer-overflow whose top two frames were __afl_put_hexbytes and
__afl_cmplog_bytes -- a crash in our own instrumentation, reported as a
crash in the target.

strcmp above it was already careful (it measures with __afl_fb_len before
logging) and the memmem site already carries a comment about the same class
of unguarded read, so these two were the instances that were missed.

The bound has to be what the real call provably read -- every byte up to and
including the first NUL or mismatch -- not strlen: an operand with no NUL
inside n would over-read again while measuring.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

AFL_SHIM = Path("src/fuzzer_tool/adapters/afl_shim.c")

# Each buffer is sized exactly, so ASAN's redzone catches a read of byte n.
# The mismatch sits inside the valid region in every case, which is what
# makes each call legal and the interceptor's extra bytes a genuine overread.
_TARGET_C = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char *p = malloc(2);
    p[0] = 'h'; p[1] = 'x';
    volatile int r = strncmp(p, "http:", 5);

    char *q = malloc(3);
    q[0] = 'H'; q[1] = 'T'; q[2] = 'Z';
    r += strncasecmp(q, "http:", 5);

    /* Fully readable operands: the log must still cover all n bytes here,
     * so a fix that clamps too aggressively fails this half. */
    char *full = malloc(4);
    memcpy(full, "abcd", 4);
    r += strncmp(full, "abcZ", 4);

    printf("%d\\n", r);
    free(p); free(q); free(full);
    return 0;
}
"""


@requires_clang
def test_strncmp_cmplog_does_not_read_past_operand(tmp_path):
    src = tmp_path / "target.c"
    src.write_text(_TARGET_C)
    exe = tmp_path / "target"
    build = subprocess.run(
        [
            "clang",
            "-fsanitize=address",
            "-O1",
            "-g",
            "-fno-builtin-strncmp",
            "-fno-builtin-strncasecmp",
            "-D__AFL_CMPLOG=1",
            "-include",
            str(AFL_SHIM),
            "-o",
            str(exe),
            str(src),
            "-ldl",
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"overread target did not build: {build.stderr[-400:]}")

    out = tmp_path / "cmplog.out"
    proc = subprocess.run(
        [str(exe)],
        capture_output=True,
        text=True,
        env={"_CMPLOG_OUT": str(out), "PATH": "/usr/bin:/bin"},
        timeout=60,
    )

    assert "heap-buffer-overflow" not in proc.stderr, proc.stderr[-800:]
    assert "AddressSanitizer" not in proc.stderr, proc.stderr[-800:]
    assert proc.returncode == 0, proc.stderr[-400:]

    # The point of bounding rather than skipping: cmplog still gets the
    # operand bytes up to the mismatch, which is the part redqueen uses.
    records = [ln for ln in out.read_text().splitlines() if ln.startswith("CMP ")]
    assert records, "bounding the read must not silence cmplog entirely"
    # "hx" vs "ht" -- two bytes, the readable prefix of the 5-byte bound.
    assert any(r.startswith("CMP 6878 6874 ") for r in records), records
    # The fully readable compare still logs all four bytes.
    assert any(r.startswith("CMP 61626364 6162635a ") for r in records), records
