"""Regression: cmplog-layer review findings in afl_shim.c.

1. Interceptors forwarded to libc via RTLD_NEXT, past ASAN's interceptors:
   an over-long memcmp / unterminated strcmp went unreported.
2. A FIFO flush wrote a partial record into a full pipe; the next flush
   glued the next record onto it.
3. __cmplog_reset / __cmplog_close truncated / closed a log fd number the
   target had taken over.
4. wcscmp / wcscasecmp logged without the terminator strcmp logs.
5. trace_switch skipped switches over MAX_SWITCH_CASES and recorded no
   site counts without $_CMPLOG_COUNTS.
6. A record dropped by the sink stayed in the dedup table: its repeats were
   suppressed until the next reset.
"""

import glob
import os
import re
import shutil
import subprocess

import pytest

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)
CC = "clang"
TIMEOUT = 60
SHIM_FLAGS = ["-D__AFL_CMPLOG=1", "-D__AFL_CTX_SENSITIVE=0", "-include", SHIM]
SHIM_LIBS = ["-ldl", "-lpthread"]
ASAN_REPORT = "ERROR: AddressSanitizer: heap-buffer-overflow"
LIBASAN_SO = "libasan.so.8"
CMP_LINE = re.compile(r"^CMP ([0-9a-f]+) ([0-9a-f]+) -?\d+ \d+( 0x[0-9a-f]+)?$")

pytestmark = pytest.mark.skipif(shutil.which(CC) is None, reason="no clang")


def _cc(*args: str) -> None:
    r = subprocess.run([CC, *args], capture_output=True, text=True, timeout=TIMEOUT)
    if r.returncode != 0:
        pytest.skip(f"build failed: {r.stderr[-400:]}")


def _cmp_records(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        m = CMP_LINE.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _shim_const(name: str) -> int:
    with open(SHIM) as f:
        m = re.search(rf"#define {name} (\d+)", f.read())
    assert m, name
    return int(m.group(1))


# ── 1. ASAN interceptors ────────────────────────────────────────────────

_ASAN_DRV = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int (*volatile f_memcmp)(const void *, const void *, size_t) = memcmp;
static int (*volatile f_strcmp)(const char *, const char *) = strcmp;
/* m: memcmp reads 8 past h. s: strcmp on unterminated h. i: in bounds. */
int run(int mode) {
    char *h = malloc(8);
    memcpy(h, "ABCDEFGH", 8);
    char other[64];
    memset(other, 'A', 63);
    other[63] = 0;
    memcpy(other, "ABCDEFGH", 8);
    int r;
    if (mode == 'm') r = f_memcmp(h, other, 16);
    else if (mode == 's') r = f_strcmp(h, other);
    else r = f_memcmp(h, "ABCDEFGX", 8);
    free(h);
    return r;
}
#ifndef AS_LIB
int main(int argc, char **argv) { printf("ok %d\n", run(argv[1][0])); return 0; }
#endif
"""

_HOST = r"""
#include <dlfcn.h>
#include <stdio.h>
/* direct_lite shape: non-ASAN host, target .so dlopen'd RTLD_LOCAL. */
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    int (*run)(int) = (int (*)(int))dlsym(h, "run");
    printf("ok %d\n", run(argv[2][0]));
    return 0;
}
"""

ASAN_CFLAGS = ["-O1", "-g", "-fsanitize=address", "-fno-builtin-memcmp", "-fno-builtin-strcmp"]


def _libasan_a() -> str:
    found = sorted(glob.glob("/usr/lib/gcc/x86_64-linux-gnu/*/libasan.a"))
    if not found:
        pytest.skip("no static libasan")
    return found[-1]


# Static runtime inside the executable: clang's own default link, and
# gcc's libasan.a linked whole (the layout build_targets.sh falls back to).
STATIC_RUNTIMES = ["clang", "libasan_a"]


def _static_asan_exe(d, name: str, shim: bool, runtime: str) -> str:
    obj, exe = str(d / f"{name}_{runtime}.o"), str(d / f"{name}_{runtime}")
    _cc(*ASAN_CFLAGS, *(SHIM_FLAGS if shim else []), "-c", "-o", obj, str(d / "drv.c"))
    if runtime == "clang":
        _cc("-fsanitize=address", "-o", exe, obj, *SHIM_LIBS)
        return exe
    _cc(
        "-o", exe, obj, "-rdynamic",
        "-Wl,--whole-archive", _libasan_a(), "-Wl,--no-whole-archive",
        "-ldl", "-lpthread", "-lrt", "-lm",
    )  # fmt: skip
    return exe


@pytest.fixture(scope="module")
def asan_bins(tmp_path_factory):
    d = tmp_path_factory.mktemp("asan_icpt")
    (d / "drv.c").write_text(_ASAN_DRV)
    (d / "host.c").write_text(_HOST)
    bins = {
        "host": str(d / "host"),
        "lib_plain": str(d / "plain.so"),
        "lib_shim": str(d / "shim.so"),
        "dyn": str(d / "dyn"),
        "preload": str(d / "preload.so"),
        "nosan": str(d / "nosan"),
    }
    _cc("-O1", "-o", bins["host"], str(d / "host.c"), "-ldl")
    # Linked apart: a NEEDED system libasan, as build_targets.sh does.
    lib = [*ASAN_CFLAGS, "-DAS_LIB", "-fPIC"]
    for name, extra in (("lib_plain", []), ("lib_shim", SHIM_FLAGS)):
        obj = str(d / f"{name}.o")
        _cc(*lib, *extra, "-c", "-o", obj, str(d / "drv.c"))
        _cc("-shared", "-Wl,-Bsymbolic", "-o", bins[name], obj, "-lasan", *SHIM_LIBS)
    # Dynamic runtime + preload-only shim, for the recursion probe.
    _cc(*ASAN_CFLAGS, "-c", "-o", str(d / "dyn.o"), str(d / "drv.c"))
    _cc("-o", bins["dyn"], str(d / "dyn.o"), "-lasan")
    _cc("-shared", "-fPIC", "-O2", "-D__AFL_PRELOAD_ONLY", "-o", bins["preload"], SHIM, "-ldl")
    _cc("-O1", "-g", *SHIM_FLAGS, "-o", bins["nosan"], str(d / "drv.c"), *SHIM_LIBS)
    return d, bins


def _run(argv, d, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    env.update({"_CMPLOG_OUT": str(d / "cmplog.out"), **(env_extra or {})})
    (d / "cmplog.out").unlink(missing_ok=True)
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=TIMEOUT)


def _log(d) -> str:
    p = d / "cmplog.out"
    return p.read_text(errors="replace") if p.exists() else ""


@pytest.mark.parametrize("runtime", STATIC_RUNTIMES)
@pytest.mark.parametrize("mode", ["m", "s"])
def test_regression_asan_sees_overread_static_runtime(asan_bins, mode, runtime):
    d, _ = asan_bins
    plain = _static_asan_exe(d, "plain", shim=False, runtime=runtime)
    control = _run([plain, mode], d)
    assert ASAN_REPORT in control.stderr, "control: ASAN must catch it without the shim"

    shim = _static_asan_exe(d, "shim", shim=True, runtime=runtime)
    r = _run([shim, mode], d)
    assert ASAN_REPORT in r.stderr, f"shim hid the overread from ASAN: rc={r.returncode}"


@pytest.mark.parametrize("mode", ["m", "s"])
def test_adversarial_asan_sees_overread_inprocess(asan_bins, mode):
    """direct_lite shape: the .so NEEDs libasan, so RTLD_NEXT already lands
    on the interceptor. Must keep doing so."""
    d, b = asan_bins
    pre = {"LD_PRELOAD": LIBASAN_SO}
    control = _run([b["host"], b["lib_plain"], mode], d, pre)
    assert ASAN_REPORT in control.stderr, "control: ASAN must catch it without the shim"

    r = _run([b["host"], b["lib_shim"], mode], d, pre)
    assert ASAN_REPORT in r.stderr, f"shim hid the overread from ASAN: rc={r.returncode}"


def test_falsification_inbounds_clean_and_logged(asan_bins):
    """In-bounds compare: no report, and the shim still records it."""
    d, _ = asan_bins
    want = b"ABCDEFGH".hex()
    r = _run([_static_asan_exe(d, "shim", shim=True, runtime="clang"), "i"], d)
    assert r.returncode == 0 and "AddressSanitizer" not in r.stderr
    assert any(a == want for a, _ in _cmp_records(_log(d)))


def test_adversarial_runtime_before_preload_shim_no_recursion(asan_bins):
    """libasan ahead of a preload shim: its REAL(memcmp) is the shim.

    Forwarding to the interceptor there would loop; must still return.
    """
    d, b = asan_bins
    pre = {"LD_PRELOAD": f"{LIBASAN_SO}:{b['preload']}"}
    r = _run([b["dyn"], "i"], d, pre)
    assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr[-300:]}"
    assert "ok" in r.stdout


def test_adversarial_no_sanitizer_falls_back(asan_bins):
    """No ASAN runtime: lookup misses, RTLD_NEXT path still works and logs."""
    d, b = asan_bins
    r = _run([b["nosan"], "i"], d)
    assert r.returncode == 0
    assert any(a == b"ABCDEFGH".hex() for a, _ in _cmp_records(_log(d)))


# ── 2 + 6. FIFO sink: split records, dropped records stay deduped ────────

_FIFO_DRV = r"""
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
void __tracecmp_flush(void);
static int (*volatile f_memcmp)(const void *, const void *, size_t) = memcmp;
static int rfd;

static void drain(const char *path) {
    FILE *out = fopen(path, "wb");
    char b[65536];
    ssize_t n;
    while ((n = read(rfd, b, sizeof b)) > 0) fwrite(b, 1, (size_t)n, out);
    fclose(out);
}

static void marker(void) { f_memcmp("XMARKER-0123456789", "YMARKER-0123456789", 18); }

/* argv: out1 out2 mode. mode d: overfill the pipe so the marker is dropped.
 * mode n: marker alone, nothing dropped. */
int main(int argc, char **argv) {
    rfd = open(getenv("_CMPLOG_OUT"), O_RDONLY | O_NONBLOCK);
    fcntl(rfd, F_SETPIPE_SZ, 65536);
    int cap = fcntl(rfd, F_GETPIPE_SZ);
    if (cap <= 0) cap = 65536;  /* regular-file sink */
    if (argv[3][0] == 'd') {
        unsigned char a[64], b[64];
        int fill = cap / 128 + 64;  /* records are > 128 bytes */
        for (int i = 0; i < fill; i++) {
            memset(a, 'a', sizeof a);
            memset(b, 'b', sizeof b);
            a[0] = b[0] = (unsigned char)i;
            a[1] = b[1] = (unsigned char)(i >> 8);
            f_memcmp(a, b, sizeof a);
        }
    }
    marker();
    __tracecmp_flush();
    drain(argv[1]);
    marker();
    __tracecmp_flush();
    drain(argv[2]);
    return 0;
}
"""

MARKER = b"XMARKER-0123456789".hex()


@pytest.fixture(scope="module")
def fifo_bin(tmp_path_factory):
    d = tmp_path_factory.mktemp("fifo_drv")
    (d / "drv.c").write_text(_FIFO_DRV)
    exe = str(d / "drv")
    _cc("-O1", "-g", "-fno-builtin-memcmp", *SHIM_FLAGS, "-o", exe, str(d / "drv.c"), *SHIM_LIBS)
    return exe


def _run_fifo(exe, tmp_path, mode):
    fifo = tmp_path / "sink.fifo"
    os.mkfifo(fifo)
    # A reader must exist when the shim opens its O_NONBLOCK write end.
    hold = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        env = {**os.environ, "_CMPLOG_OUT": str(fifo)}
        env.pop("LD_PRELOAD", None)
        out1, out2 = tmp_path / "p1", tmp_path / "p2"
        r = subprocess.run(
            [exe, str(out1), str(out2), mode], capture_output=True, env=env, timeout=TIMEOUT
        )
        assert r.returncode == 0, r.stderr[-300:]
        return out1.read_bytes(), out2.read_bytes()
    finally:
        os.close(hold)
        fifo.unlink()


def _assert_whole_records(data: bytes):
    assert data.endswith(b"\n"), f"record split at the pipe edge: ...{data[-40:]!r}"
    for line in data.decode("latin-1").splitlines():
        m = CMP_LINE.match(line)
        assert m and len(m.group(1)) == len(m.group(2)), f"garbled record: {line[:80]!r}"


def test_regression_fifo_full_pipe_no_split_record(fifo_bin, tmp_path):
    p1, p2 = _run_fifo(fifo_bin, tmp_path, "d")
    assert p1, "nothing reached the pipe"
    _assert_whole_records(p1)
    _assert_whole_records(p1 + p2)


def test_regression_dropped_record_not_deduped(fifo_bin, tmp_path):
    p1, p2 = _run_fifo(fifo_bin, tmp_path, "d")
    assert MARKER not in p1.decode("latin-1"), "probe invalid: marker was not dropped"
    assert any(a == MARKER for a, _ in _cmp_records(p2.decode("latin-1")))


def test_falsification_written_record_still_deduped(fifo_bin, tmp_path):
    """Nothing dropped: the repeat in the same scope stays suppressed."""
    p1, p2 = _run_fifo(fifo_bin, tmp_path, "n")
    assert any(a == MARKER for a, _ in _cmp_records(p1.decode("latin-1")))
    assert not any(a == MARKER for a, _ in _cmp_records(p2.decode("latin-1")))


def test_adversarial_regular_file_keeps_every_record(fifo_bin, tmp_path):
    """Regular-file sink is not chunked away: every filler arrives intact."""
    log = tmp_path / "log"
    env = {**os.environ, "_CMPLOG_OUT": str(log)}
    env.pop("LD_PRELOAD", None)
    # Driver reads its sink as a FIFO; on a regular file the drains just
    # copy the file, which is all this needs.
    r = subprocess.run(
        [fifo_bin, str(tmp_path / "p1"), str(tmp_path / "p2"), "d"],
        capture_output=True,
        env=env,
        timeout=TIMEOUT,
    )
    assert r.returncode == 0
    data = log.read_bytes()
    _assert_whole_records(data)
    fillers = [a for a, _ in _cmp_records(data.decode("latin-1")) if a != MARKER]
    assert len(fillers) == len(set(fillers)) and len(fillers) >= 64 * 1024 // 128


# ── 3. Stolen fd: reset / close ─────────────────────────────────────────

_STEAL_DRV = r"""
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
void __cmplog_reset(void);
void __cmplog_close(void);
void __tracecmp_flush(void);
static int (*volatile f_memcmp)(const void *, const void *, size_t) = memcmp;
#define MAX_FD 64
/* s: the target's file takes every fd number the shim could hold.
 * o: no theft; reset must still empty the log. */
int main(int argc, char **argv) {
    if (argv[2][0] == 'o') {
        f_memcmp("AAAA", "BBBB", 4);
        __tracecmp_flush();
        __cmplog_reset();
        return 0;
    }
    int mine = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (write(mine, "TARGET\n", 7) != 7) return 2;
    for (int fd = 3; fd < MAX_FD; fd++) if (fd != mine) dup2(mine, fd);
    __cmplog_reset();
    __cmplog_close();
    for (int fd = 3; fd < MAX_FD; fd++)
        if (fcntl(fd, F_GETFD) < 0) { printf("closed %d\n", fd); return 3; }
    return 0;
}
"""


@pytest.fixture(scope="module")
def steal_bin(tmp_path_factory):
    d = tmp_path_factory.mktemp("steal_drv")
    (d / "drv.c").write_text(_STEAL_DRV)
    exe = str(d / "drv")
    _cc("-O1", "-g", "-fno-builtin-memcmp", *SHIM_FLAGS, "-o", exe, str(d / "drv.c"), *SHIM_LIBS)
    return exe


def _run_steal(exe, tmp_path, mode):
    env = {**os.environ, "_CMPLOG_OUT": str(tmp_path / "cmplog.out")}
    env.pop("LD_PRELOAD", None)
    target_file = tmp_path / "target.dat"
    r = subprocess.run(
        [exe, str(target_file), mode], capture_output=True, text=True, env=env, timeout=TIMEOUT
    )
    return r, target_file


def test_regression_reset_close_spare_stolen_fd(steal_bin, tmp_path):
    r, target_file = _run_steal(steal_bin, tmp_path, "s")
    assert r.returncode == 0, f"shim closed the target's fd: {r.stdout}"
    assert target_file.read_bytes() == b"TARGET\n", "shim truncated the target's file"


def test_falsification_reset_truncates_own_log(steal_bin, tmp_path):
    r, _ = _run_steal(steal_bin, tmp_path, "o")
    assert r.returncode == 0
    assert (tmp_path / "cmplog.out").stat().st_size == 0


# ── 4. wcscmp terminator ────────────────────────────────────────────────

_WCS_DRV = r"""
#include <stdio.h>
#include <wchar.h>
static int (*volatile f_wcscmp)(const wchar_t *, const wchar_t *) = wcscmp;
static int (*volatile f_wcscasecmp)(const wchar_t *, const wchar_t *) = wcscasecmp;
int main(int argc, char **argv) {
    int (*f)(const wchar_t *, const wchar_t *) = argv[1][3] == 'c' ? f_wcscmp : f_wcscasecmp;
    if (argv[2][0] == 's') f(L"AB", L"ABC");
    else if (argv[2][0] == 'l') f(L"ABCDEFGHIJKLMNOPQRSTUVWX", L"ABCDEFGHIJKLMNOPQRSTUVWY");
    else f(L"AB", L"AB");
    return 0;
}
"""

WCHAR = 4
WENC = "utf-32-le"


@pytest.fixture(scope="module")
def wcs_bin(tmp_path_factory):
    d = tmp_path_factory.mktemp("wcs_drv")
    (d / "drv.c").write_text(_WCS_DRV)
    exe = str(d / "drv")
    _cc("-O1", "-g", "-fno-builtin", *SHIM_FLAGS, "-o", exe, str(d / "drv.c"), *SHIM_LIBS)
    return exe


def _wcs_records(exe, tmp_path, fn, case):
    log = tmp_path / f"{fn}{case}.out"
    env = {**os.environ, "_CMPLOG_OUT": str(log)}
    env.pop("LD_PRELOAD", None)
    r = subprocess.run([exe, fn, case], capture_output=True, env=env, timeout=TIMEOUT)
    assert r.returncode == 0
    return _cmp_records(log.read_text()) if log.exists() else []


@pytest.mark.parametrize("fn", ["wcscmp", "wcscasecmp"])
def test_regression_wcs_logs_terminator(wcs_bin, tmp_path, fn):
    recs = _wcs_records(wcs_bin, tmp_path, fn, "s")
    want_a = "AB\0".encode(WENC).hex()
    want_b = "ABC".encode(WENC).hex()
    assert (want_a, want_b) in recs, recs


@pytest.mark.parametrize("fn", ["wcscmp", "wcscasecmp"])
def test_falsification_wcs_long_capped(wcs_bin, tmp_path, fn):
    cap = _shim_const("CMPLOG_MAX_OPERAND")
    recs = _wcs_records(wcs_bin, tmp_path, fn, "l")
    want = "ABCDEFGHIJKLMNOPQRSTUVWX".encode(WENC)[:cap].hex()
    assert any(a == want for a, _ in recs), recs


@pytest.mark.parametrize("fn", ["wcscmp", "wcscasecmp"])
def test_adversarial_wcs_equal_not_logged(wcs_bin, tmp_path, fn):
    assert not any(
        a.startswith("AB".encode(WENC).hex()) for a, _ in _wcs_records(wcs_bin, tmp_path, fn, "e")
    )


# ── 5. trace_switch ─────────────────────────────────────────────────────

SWITCH_BASE = 1000


def _switch_src(n_cases: int) -> str:
    cases = "\n".join(f"    case {SWITCH_BASE + i}: return {i + 1};" for i in range(n_cases))
    return (
        "#include <stdlib.h>\n"
        f"int big(int v) {{\n  switch (v) {{\n{cases}\n    default: return 0;\n  }}\n}}\n"
        "int main(int argc, char **argv) { return big(atoi(argv[1])) ? 0 : 1; }\n"
    )


def _switch_bin(tmp_path_factory, n_cases):
    d = tmp_path_factory.mktemp(f"switch{n_cases}")
    (d / "drv.c").write_text(_switch_src(n_cases))
    exe = str(d / "drv")
    cov = "-fsanitize-coverage=trace-pc-guard,trace-cmp"
    obj = str(d / "drv.o")
    _cc("-O0", "-g", cov, *SHIM_FLAGS, "-c", "-o", obj, str(d / "drv.c"))
    _cc("-o", exe, obj, *SHIM_LIBS)
    return exe


@pytest.fixture(scope="module")
def big_switch(tmp_path_factory):
    n = _shim_const("MAX_SWITCH_CASES") + 44
    return _switch_bin(tmp_path_factory, n), n


@pytest.fixture(scope="module")
def small_switch(tmp_path_factory):
    return _switch_bin(tmp_path_factory, 4), 4


def _run_switch(exe, tmp_path, value, **env_files):
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    for var, name in env_files.items():
        env[var] = str(tmp_path / name)
    subprocess.run([exe, str(value)], capture_output=True, env=env, timeout=TIMEOUT)
    return {
        name: (tmp_path / name).read_text() if (tmp_path / name).exists() else ""
        for name in env_files.values()
    }


def _cnt(text: str, name: str) -> tuple[int, int]:
    fired = hit = 0
    for line in text.splitlines():
        p = line.split()
        if len(p) == 4 and p[0] == "CNT" and p[1] == name:
            fired, hit = fired + int(p[2]), hit + int(p[3])
    return fired, hit


def test_regression_big_switch_counted(big_switch, tmp_path):
    exe, n = big_switch
    out = _run_switch(exe, tmp_path, SWITCH_BASE + n - 1, _CMPLOG_COUNTS="counts")
    assert _cnt(out["counts"], "trace_switch") == (1, 1), "last case past the cap must match"


def test_regression_big_switch_records_clamped(big_switch, tmp_path):
    exe, n = big_switch
    cap = _shim_const("MAX_SWITCH_CASES")
    val = (5).to_bytes(4, "little").hex()
    cases = {(SWITCH_BASE + i).to_bytes(4, "little").hex() for i in range(n)}
    out = _run_switch(exe, tmp_path, 5, _CMPLOG_OUT="log")
    hits = [b for a, b in _cmp_records(out["log"]) if a == val and b in cases]
    assert len(hits) == min(n, cap), len(hits)


def test_regression_switch_sites_without_counts(small_switch, tmp_path):
    exe, _ = small_switch
    out = _run_switch(exe, tmp_path, SWITCH_BASE, _CMPLOG_SITE_COUNTS="sites")
    assert any(line.startswith("CNS trace_switch ") for line in out["sites"].splitlines())


def test_falsification_small_switch_default_unsatisfied(small_switch, tmp_path):
    exe, _ = small_switch
    out = _run_switch(exe, tmp_path, 5, _CMPLOG_COUNTS="counts")
    assert _cnt(out["counts"], "trace_switch") == (1, 0)


def test_adversarial_big_switch_default_unsatisfied(big_switch, tmp_path):
    exe, _ = big_switch
    out = _run_switch(exe, tmp_path, 5, _CMPLOG_COUNTS="counts")
    assert _cnt(out["counts"], "trace_switch") == (1, 0)
