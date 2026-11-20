"""PR #449 review follow-ups: guarded abort() paths that still escaped.

1. abort() on a worker thread of a guarded entry returned (no frame there).
2. SIGABRT blocked when the guard was entered: raise() left it pending.
3. A handler installed past the shim's sigaction wrapper (sigset) caught the
   raise and returned into the target.
4. A lock held at the abort stayed held: every later input timed out.
5. __afl_abort_is_crash's lazy cache raced between first aborts.

Every in-process probe runs in a child interpreter: a regression here
kills or wedges the host, which must not be pytest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
ABORT_ENV = "__AFL_ABORT_CRASH"
_PAST_ABORT_RC = 7
_SIGABRT_RC = -6
_TIMEOUT_RC = -1

_SO_SRC = rf"""
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <stdlib.h>
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int counter;
static void *aborter(void *arg) {{ (void)arg; abort(); return NULL; }}
static void swallow(int sig) {{ (void)sig; }}
int fuzz_shm_run(const unsigned char *d, size_t n) {{
    if (!n) return 0;
    switch (d[0]) {{
    case 'W': {{                          /* abort on a worker thread */
        pthread_t t;
        pthread_create(&t, NULL, aborter, NULL);
        pthread_join(t, NULL);
        return {_PAST_ABORT_RC};
    }}
    case 'H':                              /* handler past the wrapper */
        sigset(SIGABRT, swallow);
        abort();
        return {_PAST_ABORT_RC};
    case 'L':                              /* abort holding a lock */
        pthread_mutex_lock(&lock);
        abort();
        pthread_mutex_unlock(&lock);
        return {_PAST_ABORT_RC};
    case 'A':
        abort();
        return {_PAST_ABORT_RC};
    case 'C':                              /* global state across inputs */
        return ++counter;
    default:
        pthread_mutex_lock(&lock);
        pthread_mutex_unlock(&lock);
        return 0;
    }}
}}
"""

# argv: so, restore mode, block ("block"|"-"), inputs...; prints one rc per line.
_PROBE = """
import signal, sys
from fuzzer_tool.adapters.inprocess import InProcessRunner, StateRestore
so, mode, block, *inputs = sys.argv[1:]
r = InProcessRunner(so, function_name="fuzz_shm_run", direct_lite=True, timeout=1.0,
                    restore=StateRestore(mode))
if block == "block":
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGABRT})
for d in inputs:
    rc, _ = r.run_one(d.encode())
    print(rc, flush=True)
r.stop()
"""


def _clang(tmp_path: Path, name: str, *flags: str) -> Path:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / f"{name}.c"
    src.write_text(_SO_SRC)
    out = tmp_path / f"{name}.so"
    r = subprocess.run(
        ["clang", "-O1", "-D_GNU_SOURCE", *flags, "-shared", "-fPIC"]
        + ["-include", str(SHIM), "-o", str(out), str(src), "-lpthread"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return out


@pytest.fixture
def so(tmp_path) -> Path:
    return _clang(tmp_path, "t")


def _probe(so: Path, *inputs: str, mode: str = "off", block: str = "-") -> list[int]:
    env = {k: v for k, v in os.environ.items() if k != ABORT_ENV}
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(so), mode, block, *inputs],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    return [int(x) for x in proc.stdout.split()]


# ── 1. worker-thread abort ──────────────────────────────────────────────


def test_regression_worker_abort_is_crash(so):
    assert _probe(so, "W", "q") == [_SIGABRT_RC, 0]


def test_main_thread_abort_unchanged(so):
    """Falsification: the owner thread's abort was already a crash."""
    assert _probe(so, "A", "q") == [_SIGABRT_RC, 0]


# ── 2. SIGABRT blocked on entry ─────────────────────────────────────────


def test_regression_abort_blocked_on_entry(so):
    # The host enters the guard with SIGABRT blocked; the mask the guard
    # saved comes back on landing, so nothing stays pending for the host.
    assert _probe(so, "A", "q", block="block") == [_SIGABRT_RC, 0]


# ── 3. handler installed past the wrapper ───────────────────────────────


def test_regression_abort_displaced_handler(so):
    assert _probe(so, "H", "q") == [_SIGABRT_RC, 0]


# ── 4. lock held at the abort ───────────────────────────────────────────


def test_regression_abort_lock_released(so):
    # Globals come back after an abandoned call: the static mutex is free.
    assert _probe(so, "L", "q", "q") == [_SIGABRT_RC, 0, 0]


def test_held_lock_times_out_without_abandon(so):
    """Control: a lock taken by a plain crash path stays held without the
    restore -- proved with globals restore, which must agree."""
    assert _probe(so, "L", "q", mode="globals") == [_SIGABRT_RC, 0]


def test_state_kept_across_clean_inputs(so):
    """Adversarial: off mode must not reset globals after normal inputs."""
    assert _probe(so, "C", "C", "C") == [1, 2, 3]


def test_state_reset_only_after_abandon(so):
    """Adversarial: the restore lands only after the abandoned call."""
    assert _probe(so, "C", "C", "A", "C") == [1, 2, _SIGABRT_RC, 1]


# ── 5. abort mode cache under concurrent first aborts ───────────────────

_RACE_SRC = r"""
#include <pthread.h>
#include <stdlib.h>
static void *first_abort(void *arg) { (void)arg; abort(); return NULL; }
int main(void) {
    pthread_t t[4];
    for (int i = 0; i < 4; i++) pthread_create(&t[i], NULL, first_abort, NULL);
    for (int i = 0; i < 4; i++) pthread_join(t[i], NULL);
    return 0;
}
"""


def test_regression_abort_mode_cache_race_free(tmp_path):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "race.c"
    src.write_text(_RACE_SRC)
    exe = tmp_path / "race"
    r = subprocess.run(
        ["clang", "-O1", "-g", "-fsanitize=thread", "-include", str(SHIM)]
        + ["-o", str(exe), str(src), "-lpthread"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"no TSAN runtime: {r.stderr[-200:]}")
    env = {k: v for k, v in os.environ.items() if k != ABORT_ENV}
    env["TSAN_OPTIONS"] = "halt_on_error=0:exitcode=0"

    run = subprocess.run([str(exe)], capture_output=True, text=True, env=env, timeout=60)

    assert "__afl_abort_is_crash" not in run.stderr, run.stderr[-1500:]
    assert run.stderr.count("[shim] abort() intercepted") == 4
