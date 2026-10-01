"""Regressions from the afl_shim.c audit (2026-10-01).

Each test builds a tiny target against the real shim and checks one defect:

  - crash handler jumped to an unarmed sigjmp_buf outside __afl_guarded_call
    (SIGFPE reported as SIGSEGV, ASAN's SEGV report lost, SIGPIPE in a
    host Python process turned into a crash)
  - forkserver spun forever when SIGCHLD was ignored
  - cmplog record buffer had no cross-thread exclusion
  - strspn/strcspn logged solved comparisons and dropped unsolved ones
  - trace_switch logged every case at 8 bytes
  - segment sizes (AFL_MAP_SIZE, distance header) were never validated

Host-process cases run in a child interpreter: loading the shim installs
signal handlers, which must not leak into pytest.
"""

from __future__ import annotations

import os
import select
import signal
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

from fuzzer_tool.adapters.shm import DistanceTableShm, ShmCoverage
from tests.conftest import requires_clang

SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
FORKSRV_FD = 198
STATUS_TIMEOUT_S = 10
MAP_ENTRIES = 8192


def _build(tmp: Path, src: str, *flags: str, shared: bool = False, asan: bool = False) -> Path:
    """Compile *src* with the shim -included, link separately (no sancov runtime)."""
    c = tmp / "t.c"
    c.write_text(textwrap.dedent(src))
    obj = tmp / "t.o"
    out = tmp / ("t.so" if shared else "t")
    san = ["-fsanitize=address"] if asan else []
    pic = ["-fPIC"] if shared else []
    cc = [
        "clang", "-O1", "-g", "-fno-omit-frame-pointer", *san, *pic, *flags,
        "-include", str(SHIM), "-c", str(c), "-o", str(obj),
    ]  # fmt: skip
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    link = ["clang", *san, *(["-shared"] if shared else []), str(obj), "-o", str(out), "-ldl"]
    r = subprocess.run(link, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def _clean_env(**extra: str) -> dict[str, str]:
    """Environment without inherited shim/sanitizer knobs."""
    drop = ("__AFL_", "AFL_", "_CMPLOG", "ASAN_OPTIONS", "LD_PRELOAD", "FUZZER_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env.update(extra)
    return env


def _run_py(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        capture_output=True, text=True, timeout=60, env=_clean_env(),
    )  # fmt: skip


# ── Crash handler scope ────────────────────────────────────────────────


@requires_clang
def test_regression_sigfpe_not_reported_as_sigsegv(tmp_path):
    exe = _build(
        tmp_path,
        """
        int main(int argc, char **argv) { volatile int z = argc - 1; return 10 / z; }
        """,
        "-fsanitize-coverage=trace-pc-guard",
    )
    r = subprocess.run([str(exe)], capture_output=True, env=_clean_env())

    assert r.returncode == -signal.SIGFPE


@requires_clang
def test_regression_asan_segv_report_survives(tmp_path):
    exe = _build(
        tmp_path,
        """
        int main(int argc, char **argv) { volatile int *p = (int *)(long)(argc - 1); return *p; }
        """,
        "-fsanitize-coverage=trace-pc-guard",
        asan=True,
    )
    r = subprocess.run([str(exe)], capture_output=True, text=True, env=_clean_env())

    assert r.returncode != 0
    assert "AddressSanitizer: SEGV" in r.stderr


_SO_SRC = """
#include <stdint.h>
#include <stddef.h>
int ok_entry(const uint8_t *d, size_t n) { (void)d; return (int)n; }
int crash_entry(const uint8_t *d, size_t n) { (void)d; volatile int *p = (int *)(n - n); return *p; }
"""


@requires_clang
def test_regression_host_python_keeps_sigpipe_ignored(tmp_path):
    so = _build(tmp_path, _SO_SRC, "-fsanitize-coverage=trace-pc-guard", shared=True)
    r = _run_py(
        """
        import ctypes, os, sys
        ctypes.CDLL(sys.argv[1])
        rd, wr = os.pipe(); os.close(rd)
        try:
            os.write(wr, b"x")
        except BrokenPipeError:
            print("EPIPE")
        """,
        str(so),
    )

    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "EPIPE"


@requires_clang
def test_regression_stray_signal_goes_to_previous_handler(tmp_path):
    """Adversarial: a signal outside the guard reaches the old handler, and the
    guard still recovers afterwards (handlers re-armed)."""
    so = _build(tmp_path, _SO_SRC, "-fsanitize-coverage=trace-pc-guard", shared=True)
    r = _run_py(
        """
        import ctypes, os, signal, sys
        hits = []
        signal.signal(signal.SIGSEGV, lambda *a: hits.append(1))
        lib = ctypes.CDLL(sys.argv[1])
        g = lib.__afl_guarded_call
        g.restype = ctypes.c_int
        g.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        crash = ctypes.cast(lib.crash_entry, ctypes.c_void_p)
        ok = ctypes.cast(lib.ok_entry, ctypes.c_void_p)
        first = g(crash, b"x", 1)
        os.kill(os.getpid(), signal.SIGSEGV)   # stray: not inside the guard
        print(first, len(hits), g(ok, b"ab", 2), g(crash, b"x", 1))
        """,
        str(so),
    )

    assert r.returncode == 0, r.stderr
    sigsegv = int(signal.SIGSEGV)
    assert r.stdout.split() == [str(-sigsegv), "1", "2", str(-sigsegv)]


# ── Forkserver ─────────────────────────────────────────────────────────


class _Forkserver:
    """Drive the shim's in-target forkserver over AFL's fd pair."""

    def __init__(self, exe: Path, args: list[str], env: dict[str, str]):
        self._ctl_r, self._ctl_w = os.pipe()
        self._st_r, self._st_w = os.pipe()
        ctl_r, st_w = self._ctl_r, self._st_w

        def _wire():
            os.dup2(ctl_r, FORKSRV_FD)
            os.dup2(st_w, FORKSRV_FD + 1)

        self.proc = subprocess.Popen(
            [str(exe), *args], env={**env, "__AFL_FORKSRV": "1"},
            preexec_fn=_wire, close_fds=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        os.close(self._ctl_r)
        os.close(self._st_w)
        assert len(self._read4()) == 4, "no forkserver hello"

    def _read4(self) -> bytes:
        ready, _, _ = select.select([self._st_r], [], [], STATUS_TIMEOUT_S)
        if not ready:
            return b""
        return os.read(self._st_r, 4)

    def run_one(self) -> int | None:
        """One execution; wait status, or None if the server never answered."""
        os.write(self._ctl_w, b"\0\0\0\0")
        assert len(self._read4()) == 4, "no child pid"
        status = self._read4()
        if len(status) != 4:
            return None
        return struct.unpack("<i", status)[0]

    def close(self) -> None:
        os.close(self._ctl_w)
        os.close(self._st_r)
        self.proc.kill()
        self.proc.wait(timeout=STATUS_TIMEOUT_S)


@requires_clang
def test_regression_forkserver_survives_ignored_sigchld(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <signal.h>
        __attribute__((constructor(101))) static void early(void) { signal(SIGCHLD, SIG_IGN); }
        int main(void) { return 3; }
        """,
        "-fsanitize-coverage=trace-pc-guard",
    )
    fs = _Forkserver(exe, [], _clean_env())
    try:
        status = fs.run_one()
    finally:
        fs.close()

    assert status is not None, "forkserver never reported a wait status"
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 3


# ── cmplog record writer ───────────────────────────────────────────────


def _cmplog_records(exe: Path, tmp: Path) -> list[str]:
    log = tmp / "cmplog.txt"
    r = subprocess.run(
        [str(exe)], capture_output=True, text=True, env=_clean_env(_CMPLOG_OUT=str(log))
    )
    assert r.returncode == 0, r.stderr
    return log.read_text().splitlines() if log.exists() else []


@requires_clang
def test_regression_cmplog_writer_honours_lock(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <stdio.h>
        int main(void) {
            char a[8] = "AAAAAAA", b[8] = "BBBBBBB", z[8] = "ZZZZZZZ";
            __atomic_store_n(&__afl_cmplog_lock, 1, __ATOMIC_RELEASE);  /* another writer */
            volatile int r = memcmp(a, b, 8);
            __atomic_store_n(&__afl_cmplog_lock, 0, __ATOMIC_RELEASE);
            r += memcmp(a, z, 8);
            return r == 0;
        }
        """,
        "-D__AFL_CMPLOG=1",
        "-fno-builtin-memcmp",
    )
    recs = _cmplog_records(exe, tmp_path)

    assert not any("4242424242424200" in x for x in recs), recs
    assert any("5a5a5a5a5a5a5a00" in x for x in recs), recs


@requires_clang
def test_regression_strspn_logs_unsolved_only(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <string.h>
        int main(void) {
            volatile size_t r = strspn("zzzz", "abc");   /* 0: unsolved */
            r += strspn("abcq", "abc");                  /* 3: solved   */
            r += strcspn(":xy", ":");                    /* 0: unsolved */
            r += strcspn("xy:", ":");                    /* 2: solved   */
            return r == 0;
        }
        """,
        "-D__AFL_CMPLOG=1",
        "-fno-builtin",
    )
    recs = [x.split()[1:4] for x in _cmplog_records(exe, tmp_path)]

    assert ["7a7a7a", "616263", "-1"] in recs, recs
    assert ["3a", "3a", "-1"] in recs, recs
    assert not any(r[0] in ("616263", "78") for r in recs), recs


@requires_clang
def test_regression_trace_switch_uses_case_width(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <stdint.h>
        int main(void) {
            uint64_t c8[]  = {2, 8, 0x41, 0x42};
            uint64_t c32[] = {1, 32, 0x1234};
            uint64_t odd[] = {1, 7, 0x55};   /* adversarial: not a byte multiple */
            __sanitizer_cov_trace_switch(0x43, c8);
            __sanitizer_cov_trace_switch(0x1235, c32);
            __sanitizer_cov_trace_switch(0x56, odd);
            return 0;
        }
        """,
        "-D__AFL_CMPLOG=1",
    )
    recs = [x.split()[1:5] for x in _cmplog_records(exe, tmp_path)]

    assert ["43", "41", "1", "1"] in recs, recs
    assert ["35120000", "34120000", "1", "4"] in recs, recs
    assert ["5600000000000000", "5500000000000000", "1", "8"] in recs, recs


# ── Segment validation ─────────────────────────────────────────────────

_HEALTH_MAIN = """
#include <stdio.h>
int main(void) {
    uint64_t h[8] = {0};
    uint32_t n = __afl_shim_health(h, 8);
    for (uint32_t i = 0; i < n && i < 8; i++) printf("%llu ", (unsigned long long)h[i]);
    printf("\\n");
    return 0;
}
"""


@requires_clang
def test_regression_undersized_segment_refused(tmp_path):
    exe = _build(tmp_path, _HEALTH_MAIN, "-fsanitize-coverage=trace-pc-guard")
    shm = ShmCoverage(size=1024)
    try:
        r = subprocess.run(
            [str(exe)], capture_output=True, text=True,
            env=_clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES)),
        )  # fmt: skip
    finally:
        shm.cleanup()

    assert r.returncode == 0, r.stderr
    assert "too small" in r.stderr
    assert r.stdout.split()[0] == "0"  # not attached


@requires_clang
def test_regression_negative_map_size_refused(tmp_path):
    exe = _build(tmp_path, _HEALTH_MAIN, "-fsanitize-coverage=trace-pc-guard")
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = subprocess.run(
            [str(exe)], capture_output=True, text=True,
            env=_clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE="-5"),
        )  # fmt: skip
    finally:
        shm.cleanup()

    assert r.returncode == 0, r.stderr
    assert "AFL_MAP_SIZE" in r.stderr


@requires_clang
def test_regression_distance_header_overrun_refused(tmp_path):
    import ctypes

    exe = _build(tmp_path, _HEALTH_MAIN, "-D__AFL_DISTANCE_MODE", "-fsanitize-coverage=trace-pc")
    shm = ShmCoverage(size=MAP_ENTRIES)
    table = DistanceTableShm({0x1000: 1.0})
    ctypes.c_uint32.from_address(table._ptr).value = 1 << 30  # claims 16 GiB of entries
    try:
        r = subprocess.run(
            [str(exe)], capture_output=True, text=True,
            env=_clean_env(
                __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES),
                __AFL_DIST_SHM_ID=table.env_id,
            ),
        )  # fmt: skip
    finally:
        shm.cleanup()
        table.cleanup()

    assert r.returncode == 0, r.stderr
    fields = [int(x) for x in r.stdout.split()]
    assert fields[0] == 1  # edge table still attached
    assert fields[2] >= 1  # one segment refused
