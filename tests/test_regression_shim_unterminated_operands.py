"""Regression: COMPCOV/CMPLOG string interceptors faulted on unterminated operands.

strcmp/strcasecmp/wcscmp/wcscasecmp/strpbrk/strspn/strcspn call the REAL libc
function first, then measured both operands with an unbounded strlen
(__afl_fb_len / __afl_fb_wcslen) to size the cmplog record. The real call may
legally return at the first mismatch (or first match), so an operand that is
unterminated and ends at a page boundary is fine for glibc -- but the unbounded
scan afterwards ran off the mapping and the SHIM faulted: a SIGSEGV with our frame
on top, reported as a target crash. --hail-mary forces cmplog and compcov on, so
every string-heavy target (ffmpeg) is exposed.

Same bug class as the frame-pointer walk in __afl_get_caller_ctx: instrumentation
dereferencing target memory it has not proven readable. The fix measures with
__afl_safe_len / __afl_safe_wcslen, bounded to the page holding s[0].

Each case first runs WITHOUT the shim: if libc itself faults the case is not a
valid probe and is skipped, so a pass here means the shim added nothing.
"""

import os
import shutil
import subprocess

import pytest

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "fuzzer_tool", "adapters", "afl_shim.c",
)

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")

_DRIVER = r"""
#define _GNU_SOURCE
#include <string.h>
#include <strings.h>
#include <wchar.h>
#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
/* Operand ends exactly at the last readable byte before a PROT_NONE page, with no NUL.
 * Every call below returns at the first mismatch/match, so libc never touches the guard page. */
int main(int argc, char **argv) {
    long ps = sysconf(_SC_PAGESIZE);
    char *m = mmap(NULL, 2 * ps, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    mprotect(m + ps, ps, PROT_NONE);
    char *buf = m + ps - 4;  memcpy(buf, "AAAA", 4);
    char *m2 = mmap(NULL, 2 * ps, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    mprotect(m2 + ps, ps, PROT_NONE);
    wchar_t *wbuf = (wchar_t *)(m2 + ps - 4 * sizeof(wchar_t));
    for (int i = 0; i < 4; i++) wbuf[i] = L'A';
    int (*volatile f_strcmp)(const char *, const char *) = strcmp;
    int (*volatile f_strcasecmp)(const char *, const char *) = strcasecmp;
    int (*volatile f_wcscmp)(const wchar_t *, const wchar_t *) = wcscmp;
    int (*volatile f_wcscasecmp)(const wchar_t *, const wchar_t *) = wcscasecmp;
    char *(*volatile f_strpbrk)(const char *, const char *) = strpbrk;
    size_t (*volatile f_strspn)(const char *, const char *) = strspn;
    size_t (*volatile f_strcspn)(const char *, const char *) = strcspn;
    const char *w = argv[1];
    long r = 0;
    if      (!strcmp(w, "strcmp"))      r = f_strcmp(buf, "Z");
    else if (!strcmp(w, "strcasecmp"))  r = f_strcasecmp(buf, "Z");
    else if (!strcmp(w, "wcscmp"))      r = f_wcscmp(wbuf, L"Z");
    else if (!strcmp(w, "wcscasecmp"))  r = f_wcscasecmp(wbuf, L"Z");
    else if (!strcmp(w, "strpbrk"))     r = f_strpbrk(buf, "A") != 0;   /* match at byte 0 */
    else if (!strcmp(w, "strspn"))      r = f_strspn(buf, "Z");         /* stops at byte 0 */
    else if (!strcmp(w, "strcspn"))     r = f_strcspn(buf, "A");        /* stops at byte 0 */
    printf("%s ok r=%ld\n", w, r);
    return 0;
}
"""

CASES = ["strcmp", "strcasecmp", "wcscmp", "wcscasecmp", "strpbrk", "strspn", "strcspn"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("unterminated")
    src = d / "drv.c"
    src.write_text(_DRIVER)
    native = d / "native"
    shim = d / "shim"
    r = subprocess.run(["gcc", "-O1", "-g", "-o", str(native), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"native driver failed to build: {r.stderr[:300]}")
    r = subprocess.run(
        ["gcc", "-O1", "-g", "-D__AFL_CMPLOG=1", "-D__AFL_CTX_SENSITIVE=0",
         "-include", SHIM, "-o", str(shim), str(src), "-ldl", "-lpthread"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"shim failed to build: {r.stderr[:300]}")
    return d, str(native), str(shim)


@pytest.mark.parametrize("case", CASES)
def test_unterminated_operand_at_page_edge(built, case):
    d, native, shim = built
    r = subprocess.run([native, case], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        pytest.skip(f"libc itself faults on {case} here (rc={r.returncode}); not a valid probe")
    env = {**os.environ, "_CMPLOG_OUT": str(d / "cmplog.out"), "__AFL_COMPCOV_LEVEL": "2"}
    r = subprocess.run([shim, case], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, f"{case}: rc={r.returncode} (139 = shim SIGSEGV) stderr={r.stderr[:200]}"
    assert f"{case} ok" in r.stdout
