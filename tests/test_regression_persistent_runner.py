"""Regression: ``--persistent`` never ran an input and never recovered.

1. ``start()`` consumed the target's initial SIGSTOP but sent no SIGCONT, so
   the first SIGUSR1 stayed pending: exec 1 timed out, the rest returned -2.
2. A clean ``exit(0)`` read as "no event" (``status == 0``) and the next
   ``waitpid`` raised ``ChildProcessError``.
3. After a crash nothing restarted the target: every later exec returned -2.
"""

import shutil
import signal
import subprocess
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.persistent_signal import PersistentRunner
from fuzzer_tool.services.runner import TargetRunner

_TIMEOUT_S = 5
_OK = b"A"
_CRASH = b"X"
_EXIT = b"Q"

_SRC = r"""
#include <signal.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <unistd.h>
static volatile int go;
static void on_usr1(int s) { (void)s; go = 1; }
int main(void) {
    unsigned char *shm = shmat(atoi(getenv("__AFL_SHM_ID")), 0, 0);
    sigset_t block, wait;
    sigemptyset(&block);
    sigaddset(&block, SIGUSR1);
    sigprocmask(SIG_BLOCK, &block, &wait);
    signal(SIGUSR1, on_usr1);
    raise(SIGSTOP);
    for (;;) {
        while (!go) sigsuspend(&wait);
        go = 0;
        unsigned n = *(unsigned *)shm;
        if (n && shm[8] == 'Q') _exit(0);
        if (n && shm[8] == 'X') abort();
        *(int *)(shm + 8 + n) = 0;
        raise(SIGSTOP);
    }
}
"""


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    cc = shutil.which("clang")
    if not cc:
        pytest.skip("no clang")
    d = tmp_path_factory.mktemp("persist")
    src, out = d / "p.c", d / "p"
    src.write_text(_SRC)
    r = subprocess.run([cc, "-O0", "-o", str(out), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"build failed: {r.stderr[:300]}")
    return str(out)


@pytest.fixture
def runner(exe):
    with PersistentRunner(exe, timeout=_TIMEOUT_S) as pr:
        assert pr._started
        yield pr


def test_regression_first_exec_runs(runner):
    assert [runner.run_one(_OK)[0] for _ in range(3)] == [0, 0, 0]


def test_regression_clean_exit_is_reported(runner):
    assert runner.run_one(_OK)[0] == 0
    rc, err = runner.run_one(_EXIT)

    assert (rc, err) == (0, "target exited")
    assert not runner._started


def test_crash_is_still_a_crash(runner):
    """Falsification: the abort is reported, not masked by the restart logic."""
    assert runner.run_one(_CRASH)[0] == -signal.SIGABRT


def test_regression_restart_after_crash(runner):
    """Adversarial: crash, exit, crash - each next exec runs on a fresh target."""
    f = SimpleNamespace(_persistent_runner=runner, _perf_counters=None)
    run = TargetRunner(f)._run_other

    rcs = [run(d, None)[0] for d in (_CRASH, _OK, _EXIT, _OK, _CRASH, _OK)]

    assert rcs == [-signal.SIGABRT, 0, 0, 0, -signal.SIGABRT, 0]
