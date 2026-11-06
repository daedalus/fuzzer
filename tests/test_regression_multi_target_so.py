"""Multi-target .so runs: every target runs in-process on its own SHM map.

Bug: one InProcessRunner was built for target 0 on the global map. Target
switches only changed ``Fuzzer.target``, so every exec ran target 0 and the
per-target maps the fuzzer read stayed empty: 0 edges for every target.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

# Writes a marker file named after the target on its first call: proof it ran.
_SO_SRC = """
#include <stdio.h>
#include <stddef.h>
static int ran;
int fuzz_shm_run(const unsigned char *d, size_t n) {
    if (!ran) { FILE *f = fopen("ran_%s", "w"); if (f) fclose(f); ran = 1; }
    if (n > 1 && d[0] == '%s') return d[1] == 'Z';
    return 0;
}
"""


def _so(tmp: Path, name: str) -> Path:
    sub = tmp / name
    sub.mkdir()
    return _build(sub, _SO_SRC % (name, name[0]), "-fsanitize-coverage=trace-pc-guard", shared=True)


def _fuzz(tmp: Path, *targets: Path) -> subprocess.CompletedProcess:
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "seed").write_bytes(b"seed")
    return subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", *map(str, targets), "-d", str(corpus),
         "-o", str(tmp / "crashes"), "--max-execs", "40", "--target-schedule", "round-robin"],
        capture_output=True, text=True, timeout=180, cwd=tmp, env=_clean_env(),
    )  # fmt: skip


@requires_clang
def test_regression_multi_target_so_coverage(tmp_path):
    """Falsification: each target's map records edges at the self-test."""
    r = _fuzz(tmp_path, _so(tmp_path, "alpha"), _so(tmp_path, "beta"))

    assert r.returncode == 0, r.stderr[-2000:]
    assert "recorded no edges" not in r.stdout
    assert "Shim self-test: 2 target(s) OK" in r.stdout


@requires_clang
def test_regression_multi_target_so_runs_each(tmp_path):
    """Adversarial: coverage alone could come from target 0; each must run."""
    r = _fuzz(tmp_path, _so(tmp_path, "alpha"), _so(tmp_path, "beta"))

    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "ran_alpha").exists()
    assert (tmp_path / "ran_beta").exists()


# Module loaded first; the last loaded one owns the process's signal handlers.
_CRASH_SRC = """
#include <stddef.h>
int fuzz_shm_run(const unsigned char *d, size_t n) {
    volatile int *p = 0;
    if (n) *p = d[0];
    return 0;
}
"""

_HANG_SRC = """
#include <stddef.h>
int fuzz_shm_run(const unsigned char *d, size_t n) {
    volatile size_t i = 0;
    while (n) i++;
    return (int)i + d[0];
}
"""


def _so_src(tmp: Path, name: str, src: str) -> Path:
    sub = tmp / name
    sub.mkdir()
    return _build(sub, src, "-fsanitize-coverage=trace-pc-guard", shared=True)


def _crash_files(tmp: Path) -> list[Path]:
    return [p for p in (tmp / "crashes").rglob("*") if p.is_file() and p.suffix != ".json"]


@requires_clang
def test_regression_multi_target_so_crash_in_first(tmp_path):
    """Falsification: a crash in a module that does not own the handlers is recovered."""
    first = _so_src(tmp_path, "crasher", _CRASH_SRC)
    r = _fuzz(tmp_path, first, _so(tmp_path, "beta"))

    assert r.returncode == 0, r.stderr[-2000:]
    assert _crash_files(tmp_path)


@requires_clang
def test_regression_multi_target_so_hang_in_first(tmp_path):
    """Adversarial: a hang in the first module times out instead of wedging the run."""
    first = _so_src(tmp_path, "hanger", _HANG_SRC)
    r = _fuzz(tmp_path, first, _so(tmp_path, "beta"))

    assert r.returncode == 0, r.stderr[-2000:]
    assert "timeouts: 0 " not in r.stdout.rsplit("execs:", 1)[-1]


# Load C, A, B; then A and B re-arm in turn, each saving the other as the
# previous owner: B.old = A, A.old = B. A crash in C reached B (no guard
# frame), was handed to A (none), back to B: a re-fault loop, 100% CPU.
_CYCLE_DRIVER = """
import ctypes, sys
SIGSEGV = 11
fields = (ctypes.c_uint64 * 64)()
c, a, b = (ctypes.CDLL(p) for p in sys.argv[1:4])
buf = ctypes.create_string_buffer(b"x")
for lib in (a, b):
    lib.__afl_shim_health(fields, 64)
    lib.__afl_guarded_call(lib.fuzz_shm_run, buf, 1)
    getattr(lib, "__afl_signals_claim", lambda: None)()
for lib in (a, b):
    getattr(lib, "__afl_signals_release", lambda: None)()
getattr(c, "__afl_signals_claim", lambda: None)()
print("rc", c.__afl_guarded_call(c.fuzz_shm_run, buf, 1) == -SIGSEGV, flush=True)
"""


@requires_clang
def test_regression_multi_target_so_handler_cycle(tmp_path):
    """Falsification: a crash after modules re-arm in turn is recovered, not looped."""
    libs = [
        _so_src(tmp_path, "c", _CRASH_SRC),
        _so(tmp_path, "alpha"),
        _so(tmp_path, "beta"),
    ]
    try:
        r = subprocess.run(
            [sys.executable, "-c", _CYCLE_DRIVER, *map(str, libs)],
            capture_output=True, text=True, timeout=30, cwd=tmp_path, env=_clean_env(),
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        raise AssertionError("crash handler cycle: re-fault loop") from None

    assert "rc True" in r.stdout, r.stderr[-2000:]


class _FakeRunner:
    def __init__(self, name: str, log: list[str]):
        self.name, self.log = name, log

    def claim_signals(self) -> None:
        self.log.append(f"claim {self.name}")

    def release_signals(self) -> None:
        self.log.append(f"release {self.name}")


def _switching_fuzzer(log: list[str]):
    from types import MappingProxyType

    from fuzzer_tool.services.fuzzer import Fuzzer

    f = Fuzzer.__new__(Fuzzer)
    runners = {t: _FakeRunner(t, log) for t in ("a", "b")}
    f._target_runners = MappingProxyType(runners)
    f.target, f._inprocess_runner = "a", runners["a"]
    return f, runners


def test_regression_set_target_hands_over_signals():
    """Falsification: a switch releases the old target, then claims the new one."""
    log: list[str] = []
    f, runners = _switching_fuzzer(log)

    f._set_target("b")

    assert log == ["release a", "claim b"]
    assert f._inprocess_runner is runners["b"]


def test_regression_set_target_same_target_no_syscalls():
    """Adversarial: re-selecting the active target must not re-install handlers."""
    log: list[str] = []
    f, _ = _switching_fuzzer(log)

    f._set_target("a")
    f._set_target("unknown")

    assert log == []
    assert f.target == "unknown"
