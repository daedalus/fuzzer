"""Crash shapes the fuzzer used to be blind to.

1. A target that installs its own fatal-signal handler and exits 0/1: the
   shim now stays first in line, writes SHIM_FATAL_MARKER and chains.
3. --persistent pointed the target's stderr at /dev/null.
4. rc -2 was dropped before the sanitizer report was parsed.
5. "Sanitizer CHECK failed" (exit 1, no ERROR/SUMMARY header) read as clean.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.persistent_signal import PersistentRunner
from fuzzer_tool.adapters.process import (
    SHIM_FATAL_MARKER,
    shim_fatal_signal,
    stderr_crash_marker,
)
from fuzzer_tool.services.runner import TargetRunner

SHIM = Path(__file__).resolve().parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

ASAN_REPORT = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014 "
    "at pc 0x0 bp 0x0 sp 0x0\nWRITE of size 4 at 0x602000000014 thread T0\n"
    "    #0 0x4 in f /t.c:3:5\nSUMMARY: AddressSanitizer: heap-buffer-overflow /t.c:3:5 in f\n"
)


def _runner() -> TargetRunner:
    return TargetRunner(SimpleNamespace(extra_crash_codes=(), last_report=None))


class TestCheckFailed:  # 5
    def test_nonzero_exit_is_crash(self):
        err = '==1==AddressSanitizer CHECK failed: asan_rtl.cpp:1 "x" (0x0, 0x0)\n'
        assert stderr_crash_marker(1, err) is not None
        assert _runner().is_crash(1, err)

    def test_clean_exit_still_vouches(self):
        assert stderr_crash_marker(0, "Sanitizer CHECK failed") is None


class TestSentinelOrdering:  # 4
    def test_report_before_infra_failure_is_crash(self):
        r = _runner()
        assert r.is_crash(-2, ASAN_REPORT)
        assert r.f.last_report is not None

    def test_bare_sentinels_are_not_crashes(self):
        r = _runner()
        assert not r.is_crash(-2, "")
        assert not r.is_crash(-2, "exec timeout")
        assert not r.is_crash(-1, "timeout")

    def test_report_before_timeout_still_crash(self):
        assert _runner().is_crash(-1, ASAN_REPORT)


class TestShimMarker:  # 1 (Python side)
    def test_parse(self):
        assert shim_fatal_signal(f"x\n{SHIM_FATAL_MARKER}11\n") == 11
        assert shim_fatal_signal("nothing") is None
        assert shim_fatal_signal(SHIM_FATAL_MARKER + "\n") is None

    @pytest.mark.parametrize("rc", [0, 1, 255])
    def test_crash_whatever_the_exit_code(self, rc):
        assert _runner().is_crash(rc, f"{SHIM_FATAL_MARKER}11\n")

    def test_plain_clean_exit_is_not_a_crash(self):
        assert not _runner().is_crash(0, "")


_HANDLER_SRC = r"""
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
static void h(int s) { (void)s; _exit(%(code)d); }
int main(void) {
    char c = 0;
    if (read(0, &c, 1) < 1) return 0;
    signal(SIGSEGV, h);
    if (c == 'X') { volatile int *p = 0; *p = 1; }
    return 0;
}
"""


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    d = tmp_path_factory.mktemp("swallow")

    def _build(code: int) -> str:
        src, out = d / f"h{code}.c", d / f"h{code}"
        src.write_text(_HANDLER_SRC % {"code": code})
        r = subprocess.run(
            [
                "clang",
                "-O1",
                "-fsanitize-coverage=trace-pc-guard",
                "-include",
                str(SHIM),
                str(src),
                "-o",
                str(out),
                "-ldl",
                "-lpthread",
                "-lrt",
                "-lm",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            pytest.skip(f"build failed: {r.stderr[:300]}")
        return str(out)

    return _build


class TestSwallowedHandlerEndToEnd:  # 1
    @pytest.mark.parametrize("code", [0, 1])
    def test_shim_reports_fault_the_target_swallowed(self, build, code):
        exe = build(code)
        crash = subprocess.run([exe], input=b"X", capture_output=True, timeout=10)
        assert crash.returncode == code  # the target's handler really ran
        err = crash.stderr.decode(errors="replace")
        assert shim_fatal_signal(err) == 11
        assert _runner().is_crash(crash.returncode, err)

    def test_clean_input_writes_no_marker(self, build):
        exe = build(0)
        ok = subprocess.run([exe], input=b"Y", capture_output=True, timeout=10)
        assert ok.returncode == 0
        assert SHIM_FATAL_MARKER.encode() not in ok.stderr


_PERSIST_SRC = r"""
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <unistd.h>
static volatile int go;
static void on_usr1(int s) { (void)s; go = 1; }
int main(void) {
    unsigned char *shm = shmat(atoi(getenv("__AFL_SHM_ID")), 0, 0);
    signal(SIGUSR1, on_usr1);
    raise(SIGSTOP);  /* start() waits for this first stop */
    for (;;) {
        while (!go) pause();
        go = 0;
        unsigned n = *(unsigned *)shm;
        if (n && shm[8] == 'X') {
            fputs("==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n", stderr);
            _exit(1);
        }
        *(int *)(shm + 8 + n) = 0;
        fputs("iteration ok\n", stderr);
        raise(SIGSTOP);  /* iteration boundary; the runner SIGCONTs */
    }
}
"""


class TestPersistentStderr:  # 3
    def test_exit_after_report_surfaces_stderr(self, tmp_path):
        cc = shutil.which("gcc") or shutil.which("clang")
        if not cc:
            pytest.skip("no C compiler")
        src, exe = tmp_path / "p.c", tmp_path / "p"
        src.write_text(_PERSIST_SRC)
        r = subprocess.run([cc, "-O0", "-o", str(exe), str(src)], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"build failed: {r.stderr[:300]}")
        with PersistentRunner(str(exe), timeout=5) as pr:
            if not pr._started:
                pytest.skip("persistent target did not stop at start")
            # The first stop is start()'s; only a SIGCONT leaves it.
            os.kill(pr.pid, signal.SIGCONT)
            rc, err = pr.run_one(b"A")
            assert rc == 0 and "iteration ok" in err
            assert "ok" in err and "ERROR" not in err  # per-iteration, not cumulative
            rc, err = pr.run_one(b"X")
        assert rc == 1
        assert _runner().is_crash(rc, err)


class TestCrashingSeed:  # 6
    """A crashing seed is saved once and counted once.

    Calibration used to count it again without saving, so 1 execution
    reported 2 crashes. (Seeds are read from corpus/seeds/ only; a seed at
    the corpus root is ignored and replaced by the AAAAAAAA default.)
    """

    def test_seed_crash_saved_and_counted_once(self, build, tmp_path):
        exe = build(0)
        corpus, out = tmp_path / "corpus", tmp_path / "out"
        (corpus / "seeds").mkdir(parents=True)
        (corpus / "seeds" / "seed").write_bytes(b"X" + b"A" * 15)
        code = (
            "import sys; from fuzzer_tool.cli.commands import main; "
            "sys.argv = ['fuzzer-tool', 'fuzz', sys.argv[1], '-d', sys.argv[2], "
            "'-o', sys.argv[3], '--max-execs', '1']; main()"
        )
        r = subprocess.run(
            [sys.executable, "-c", code, exe, str(corpus), str(out)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert len(list(out.glob("*.bin"))) == 1, r.stdout[-800:]
        assert "crashes: 1" in r.stdout and "crashes: 2" not in r.stdout
