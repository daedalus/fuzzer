"""Regressions from the afl_shim.c audit (2026-11-12): synthetic ids, LTL, cmplog.

  - state/LTL synthetic ids leaked bit 30 into prev_loc: the next real edge
    left VIRGIN_DENSE_MAX
  - LTL events fd: no O_CLOEXEC, a stolen fd number got the records
  - cmplog FIFO with no reader at init stayed off for good
  - a signal inside our own flush rewrote the chunks already written
  - a resolve nested in dlsym pinned the naive fallback for good
  - wcsncmp's n * sizeof(wchar_t) wrapped and lost the record

See docs/handover/handover_afl_shim_audit_2026-11-12.md.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import VIRGIN_DENSE_MAX, ShmCoverage
from fuzzer_tool.core.ltl import read_events
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

TIMEOUT_S = 30
CMPLOG = "-D__AFL_CMPLOG=1"
CMP_LINE = re.compile(r"^CMP ([0-9a-f]+) ([0-9a-f]+) (-?\d+) (\d+)")
LTL_RECORD = 12  # {id, offset, digest}


def _records(text: str) -> list[tuple[str, str, int]]:
    """(a hex, b hex, n) per CMP line."""
    out = []
    for line in text.splitlines():
        m = CMP_LINE.match(line)
        if m:
            out.append((m.group(1), m.group(2), int(m.group(4))))
    return out


def _run(exe: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    r = subprocess.run(
        [str(exe), *args], capture_output=True, text=True, timeout=TIMEOUT_S, env=_clean_env(**env)
    )
    assert r.returncode == 0, r.stderr[-600:]
    return r


# ── 1. Synthetic ids stay out of the real-edge id space ────────────────

_SYNTH_SRC = """
int main(int argc, char **argv) {
    (void)argc;
    __afl_map_edge(0x1100);
    switch (argv[1][0]) {
    case 's': __sfuzz_state(7, 3); break;
    case 'e': __fuzz_event(1); break;
    case 'E': __fuzz_event(2); break;
    }
    __afl_map_edge(0x1101);
    return 0;
}
"""


@pytest.fixture(scope="module")
def synth_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    return _build(d, _SYNTH_SRC, "-D__AFL_TRACE_FIRES=1", "-D__AFL_CTX_SENSITIVE=0")


def _last_fire(exe: Path, tmp: Path, mode: str) -> int:
    """Edge id of the real edge after *mode*'s event."""
    cov = ShmCoverage(size=4096)
    out = tmp / f"fires_{mode}"
    try:
        _run(
            exe, mode,
            __AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(cov.num_entries), __AFL_FIRES_OUT=str(out),
        )  # fmt: skip
    finally:
        cov.cleanup()
    return int(out.read_text().split()[-1])


@requires_clang
@pytest.mark.parametrize("mode", ["s", "e"])
def test_regression_synth_event_keeps_next_edge_dense(synth_exe, tmp_path, mode):
    assert _last_fire(synth_exe, tmp_path, mode) < VIRGIN_DENSE_MAX


@requires_clang
def test_falsification_no_event_edge_dense(synth_exe, tmp_path):
    """Control: the same real edge with no event in between."""
    assert _last_fire(synth_exe, tmp_path, "n") < VIRGIN_DENSE_MAX


@requires_clang
def test_adversarial_event_still_names_next_edge(synth_exe, tmp_path):
    """The event stays in the prev_loc chain: two events, two next-edge ids."""
    one = _last_fire(synth_exe, tmp_path, "e")
    two = _last_fire(synth_exe, tmp_path, "E")
    none = _last_fire(synth_exe, tmp_path, "n")
    assert len({one, two, none}) == 3


# ── 3. LTL events fd ───────────────────────────────────────────────────

_LTL_SRC = """
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#define FD_SCAN 1024

/* FD_CLOEXEC of the fd open on *path*, -1 when none is. */
static int cloexec_of(const char *path) {
    char want[PATH_MAX], link[64], got[PATH_MAX];
    if (!realpath(path, want)) return -1;
    for (int fd = 3; fd < FD_SCAN; fd++) {
        snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
        ssize_t n = readlink(link, got, sizeof got - 1);
        if (n <= 0) continue;
        got[n] = 0;
        if (!strcmp(got, want)) return (fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0;
    }
    return -1;
}

/* Close every fd and take the lowest number for a file of our own, then
 * start the next run: the shim checks its fd once per run, at the first event. */
static void steal(const char *mine) {
    for (int fd = 3; fd < FD_SCAN; fd++) close(fd);
    int fd = open(mine, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    __afl_exec_reset();
    __fuzz_event(2);
    close(fd);
}

/* argv: mode mine1 mine2. c: print cloexec. k: no steal. s: steal. S: steal twice. */
int main(int argc, char **argv) {
    (void)argc;
    __fuzz_event(1);
    char m = argv[1][0];
    if (m == 'c') { printf("%d\\n", cloexec_of(getenv("__LTL_EVENTS_OUT"))); return 0; }
    if (m == 'k') { __fuzz_event(2); return 0; }
    steal(argv[2]);
    if (m == 'S') steal(argv[3]);
    return 0;
}
"""


@pytest.fixture(scope="module")
def ltl_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("ltl_fd")
    return _build(d, _LTL_SRC)


def _ltl(exe: Path, tmp: Path, mode: str) -> tuple[subprocess.CompletedProcess, Path]:
    ev = tmp / "ev"
    r = _run(exe, mode, str(tmp / "mine1"), str(tmp / "mine2"), __LTL_EVENTS_OUT=str(ev))
    return r, ev


@requires_clang
def test_regression_ltl_fd_cloexec(ltl_exe, tmp_path):
    r, _ = _ltl(ltl_exe, tmp_path, "c")
    assert r.stdout.strip() == "1"


@requires_clang
def test_regression_ltl_stolen_fd_reopened(ltl_exe, tmp_path):
    _, ev = _ltl(ltl_exe, tmp_path, "s")
    assert [e.event for e in read_events(str(ev))] == [1, 2]
    assert (tmp_path / "mine1").read_bytes() == b""


@requires_clang
def test_falsification_ltl_no_steal(ltl_exe, tmp_path):
    _, ev = _ltl(ltl_exe, tmp_path, "k")
    assert [e.event for e in read_events(str(ev))] == [1, 2]


@requires_clang
def test_adversarial_ltl_stolen_twice(ltl_exe, tmp_path):
    """The reopened fd is stolen again: reopened again, both files clean."""
    _, ev = _ltl(ltl_exe, tmp_path, "S")
    assert ev.stat().st_size % LTL_RECORD == 0
    assert [e.event for e in read_events(str(ev))] == [1, 2, 2]
    assert (tmp_path / "mine1").read_bytes() == b""
    assert (tmp_path / "mine2").read_bytes() == b""


# ── 4. cmplog FIFO whose reader opens after the target started ──────────

_LATE_SRC = """
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#define MARK_LEN 17
void __cmplog_reset(void);
void __tracecmp_flush(void);
static int (*volatile f_memcmp)(const void *, const void *, size_t) = memcmp;

/* Aligned: the shim clamps operands at a page end, and a stack copy that
 * straddles one (ASLR) would be logged short. */
static char a[32] __attribute__((aligned(32))) = "XMARK?-0123456789";
static char b[32] __attribute__((aligned(32))) = "YMARK?-0123456789";

static void marker(char tag) {
    a[5] = b[5] = tag;
    f_memcmp(a, b, MARK_LEN);
}

static void drain(int rfd, const char *path) {
    FILE *out = fopen(path, "wb");
    char buf[65536];
    ssize_t n;
    while ((n = read(rfd, buf, sizeof buf)) > 0) fwrite(buf, 1, (size_t)n, out);
    fclose(out);
}

/* argv: out mode. r: reader, then reset. f: reader, then fork. n: never a reader. */
int main(int argc, char **argv) {
    (void)argc;
    char m = argv[2][0];
    marker('A');
    if (m == 'n') { __cmplog_reset(); marker('B'); __tracecmp_flush(); return 0; }
    int rfd = open(getenv("_CMPLOG_OUT"), O_RDONLY | O_NONBLOCK);
    if (m == 'f') {
        pid_t pid = fork();
        if (pid == 0) { marker('C'); exit(0); }
        waitpid(pid, NULL, 0);
    } else {
        __cmplog_reset();
        marker('B');
        __tracecmp_flush();
    }
    drain(rfd, argv[1]);
    return 0;
}
"""


def _mark(tag: str) -> str:
    return f"XMARK{tag}-0123456789".encode().hex()


@pytest.fixture(scope="module")
def late_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("late_fifo")
    return _build(d, _LATE_SRC, CMPLOG, "-fno-builtin-memcmp")


def _late(exe: Path, tmp: Path, mode: str, *, early_reader: bool = False) -> list[str]:
    fifo = tmp / "sink.fifo"
    os.mkfifo(fifo)
    hold = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK) if early_reader else -1
    out = tmp / "drained"
    try:
        _run(exe, str(out), mode, _CMPLOG_OUT=str(fifo))
    finally:
        if hold >= 0:
            os.close(hold)
        fifo.unlink()
    text = out.read_bytes().decode("latin-1") if out.exists() else ""
    return [a for a, _, _ in _records(text)]


@requires_clang
@pytest.mark.parametrize(("mode", "tag"), [("r", "B"), ("f", "C")])
def test_regression_cmplog_fifo_late_reader(late_exe, tmp_path, mode, tag):
    assert _mark(tag) in _late(late_exe, tmp_path, mode)


@requires_clang
def test_falsification_cmplog_fifo_early_reader(late_exe, tmp_path):
    """Control: reader open before start, every marker arrives."""
    got = _late(late_exe, tmp_path, "r", early_reader=True)
    assert _mark("A") in got
    assert _mark("B") in got


@requires_clang
def test_adversarial_cmplog_fifo_never_read(late_exe, tmp_path):
    """No reader ever: the retry must not block; nothing is drained."""
    assert _late(late_exe, tmp_path, "n") == []


# ── 5. Signal inside our own flush ─────────────────────────────────────

_NEST_SRC = """
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#define RECORDS 40
#define OPERAND 64
void __tracecmp_flush(void);
static int (*volatile f_memcmp)(const void *, const void *, size_t) = memcmp;
static volatile sig_atomic_t armed;

/* The shim's write: the signal lands after the bytes are out, before the
 * flush loop sees the count -- the window the audit found. */
ssize_t write(int fd, const void *b, size_t n) {
    ssize_t r = syscall(SYS_write, fd, b, n);
    const char *c = b;
    if (armed && n >= 4 && c[0] == 'C' && c[1] == 'M' && c[2] == 'P') {
        armed = 0;
        raise(SIGUSR1);
    }
    return r;
}

static void on_usr1(int sig) { (void)sig; __tracecmp_flush(); }

/* argv: mode out. s: signal in flush. q: quiet. out "-": file sink. */
int main(int argc, char **argv) {
    (void)argc;
    signal(SIGUSR1, on_usr1);
    int rfd = -1;
    if (strcmp(argv[2], "-")) rfd = open(getenv("_CMPLOG_OUT"), O_RDONLY | O_NONBLOCK);
    /* Aligned: a page-straddling stack copy would be logged short. */
    static unsigned char a[OPERAND] __attribute__((aligned(OPERAND)));
    static unsigned char b[OPERAND] __attribute__((aligned(OPERAND)));
    for (int i = 0; i < RECORDS; i++) {
        memset(a, 'a', sizeof a);
        memset(b, 'b', sizeof b);
        a[0] = b[0] = (unsigned char)i;
        f_memcmp(a, b, sizeof a);
    }
    armed = argv[1][0] == 's';
    __tracecmp_flush();
    if (rfd < 0) return 0;
    FILE *out = fopen(argv[2], "wb");
    char buf[65536];
    ssize_t n;
    while ((n = read(rfd, buf, sizeof buf)) > 0) fwrite(buf, 1, (size_t)n, out);
    fclose(out);
    return 0;
}
"""

NEST_RECORDS = 40
NEST_OPERAND_HEX = 2 * 64  # OPERAND bytes as hex


@pytest.fixture(scope="module")
def nest_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("nest")
    return _build(d, _NEST_SRC, CMPLOG, "-fno-builtin-memcmp")


def _nest_file(exe: Path, tmp: Path, mode: str) -> list[str]:
    log = tmp / "cmplog.txt"
    _run(exe, mode, "-", _CMPLOG_OUT=str(log))
    return [a for a, _, _ in _records(log.read_text(encoding="latin-1"))]


def _expected_heads() -> list[str]:
    """First operand byte of each record, in order."""
    return [f"{i:02x}" for i in range(NEST_RECORDS)]


def _heads(got: list[str]) -> list[str]:
    """First byte of each driver memcmp record (drops the argv strcmp)."""
    return [a[:2] for a in got if len(a) == NEST_OPERAND_HEX]


@requires_clang
def test_regression_nested_flush_no_rewrite(nest_exe, tmp_path):
    got = _nest_file(nest_exe, tmp_path, "s")
    assert _heads(got) == _expected_heads()


@requires_clang
def test_falsification_flush_without_signal(nest_exe, tmp_path):
    got = _nest_file(nest_exe, tmp_path, "q")
    assert _heads(got) == _expected_heads()


@requires_clang
def test_adversarial_nested_flush_fifo_chunks(nest_exe, tmp_path):
    """FIFO sink: PIPE_BUF chunks; the nested flush resumes at the next one."""
    fifo = tmp_path / "sink.fifo"
    os.mkfifo(fifo)
    hold = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    out = tmp_path / "drained"
    try:
        _run(nest_exe, "s", str(out), _CMPLOG_OUT=str(fifo))
    finally:
        os.close(hold)
        fifo.unlink()
    got = [a for a, _, _ in _records(out.read_bytes().decode("latin-1"))]
    assert len(out.read_bytes()) > os.pathconf("/", "PC_PIPE_BUF"), "probe: one chunk only"
    assert _heads(got) == _expected_heads()


# ── 6. Resolve nested inside dlsym ─────────────────────────────────────

_RESOLVE_SRC = """
#include <stdio.h>
#include <string.h>
static char *(*volatile f_strstr)(const char *, const char *) = strstr;

/* argv: n: nested in dlsym. o: outermost. Prints: unresolved-before,
 * fallback-after-call, fallback-after-next-lookup, result-offset. */
int main(int argc, char **argv) {
    (void)argc;
    int nested = argv[1][0] == 'n';
    int before = real_strstr == NULL;
    __afl_in_dlsym = nested;
    const char *h = "haystack";
    char *r = f_strstr(h, "st");
    __afl_in_dlsym = 0;
    int pinned = real_strstr == (afl_str_str_fn)__afl_fb_strstr;
    (void)__afl_next_sym("memmem");
    int still = real_strstr == (afl_str_str_fn)__afl_fb_strstr;
    printf("%d %d %d %ld\\n", before, pinned, still, (long)(r - h));
    return 0;
}
"""


@pytest.fixture(scope="module")
def resolve_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("resolve")
    return _build(d, _RESOLVE_SRC, CMPLOG, "-fno-builtin")


def _resolve(exe: Path, mode: str) -> list[int]:
    return [int(x) for x in _run(exe, mode).stdout.split()]


@requires_clang
def test_regression_nested_resolve_not_pinned(resolve_exe):
    before, _, still, _ = _resolve(resolve_exe, "n")
    assert before == 1, "probe: strstr resolved before main"
    assert still == 0


@requires_clang
def test_falsification_outer_resolve_is_real(resolve_exe):
    before, pinned, still, _ = _resolve(resolve_exe, "o")
    assert (before, pinned, still) == (1, 0, 0)


@requires_clang
def test_adversarial_nested_resolve_still_answers(resolve_exe):
    """While nested, the fallback runs and returns the right match."""
    *_, off = _resolve(resolve_exe, "n")
    assert off == "haystack".index("st")


# ── 7. wchar length overflow ───────────────────────────────────────────

_WCHAR_SRC = """
#include <stdlib.h>
#include <wchar.h>
#define WLEN 32
void __tracecmp_flush(void);
static int (*volatile f_wcsncmp)(const wchar_t *, const wchar_t *, size_t) = wcsncmp;

/* argv: n, decimal. Strings differ at index 2, so wcsncmp stops there. */
int main(int argc, char **argv) {
    (void)argc;
    /* Aligned: a page-straddling stack copy would be logged short. */
    static wchar_t a[WLEN] __attribute__((aligned(sizeof(wchar_t) * WLEN)));
    static wchar_t b[WLEN] __attribute__((aligned(sizeof(wchar_t) * WLEN)));
    for (int i = 0; i < WLEN; i++) a[i] = b[i] = L'a' + i % 26;
    a[WLEN - 1] = b[WLEN - 1] = 0;
    b[2] = L'Z';
    f_wcsncmp(a, b, strtoull(argv[1], NULL, 10));
    __tracecmp_flush();
    return 0;
}
"""

WCHAR_BYTES = 4  # sizeof(wchar_t) on Linux
MAX_OPERAND_HEX = 2 * 64  # CMPLOG_MAX_OPERAND bytes as hex


@pytest.fixture(scope="module")
def wchar_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("wchar")
    return _build(d, _WCHAR_SRC, CMPLOG)


def _wchar(exe: Path, tmp: Path, n: int) -> list[tuple[str, str, int]]:
    log = tmp / f"cmplog_{n}.txt"
    _run(exe, str(n), _CMPLOG_OUT=str(log))
    return _records(log.read_text(encoding="latin-1"))


@requires_clang
def test_regression_wcsncmp_huge_n_logged(wchar_exe, tmp_path):
    n = 1 << 62  # n * 4 wraps to 0
    recs = _wchar(wchar_exe, tmp_path, n)
    assert [(len(a), k) for a, _, k in recs] == [(MAX_OPERAND_HEX, MAX_OPERAND_HEX // 2)]


@requires_clang
def test_falsification_wcsncmp_small_n(wchar_exe, tmp_path):
    n = 8
    recs = _wchar(wchar_exe, tmp_path, n)
    assert [(len(a), k) for a, _, k in recs] == [(2 * n * WCHAR_BYTES, n * WCHAR_BYTES)]


@requires_clang
def test_adversarial_wcsncmp_wrap_to_one_wchar(wchar_exe, tmp_path):
    """n * 4 wraps to 4: the record must still be capped, not one wchar wide."""
    n = (1 << 62) + 1
    recs = _wchar(wchar_exe, tmp_path, n)
    assert [len(a) for a, _, _ in recs] == [MAX_OPERAND_HEX]
