"""Regression: a guard timeout inside an ASAN report deadlocked shutdown.

Recover-mode ASAN holds its thread-registry lock while it prints a report.
A report outlasting the guard budget (slow symbolization) took the timer
tick's siglongjmp out of it with the lock held; the next thread exit (the
cmplog reader at fuzzer shutdown) blocked in ASAN's FinishThread forever.
Measured: fuzzgoat_read_asan.so, seed 2, 200k execs, hung after the run's
only timeout.

Fix: the timeout handler defers while ``__asan_report_present()``; the next
tick after the report ends the call.
"""

from __future__ import annotations

import enum
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fuzzer_tool.cli.ldpreload_wrapper import _resolve_asan
from tests.conftest import requires_clang

SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
HANG_TIMEOUT_S = 60
GUARD_TIMEOUT_RC = -1
TIMEOUT_US = 200_000
SYMBOLIZER_SLEEP_S = 1  # > TIMEOUT_US: the budget expires mid-report


class Runtime(enum.Enum):
    SLOW_REPORT = "clang runtime + stalling symbolizer"
    PRELOADED = "the GCC libasan the fuzzer preloads, own symbolizer"


_SRC = """
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
int report_then_spin(const uint8_t *d, size_t n) {
    char *p = malloc(4);
    volatile int v = p[n + 8];        /* recover-mode heap-buffer-overflow */
    for (;;) v++;                     /* then a real hang */
    return v;
}
int spin(const uint8_t *d, size_t n) { volatile size_t x = n; for (;;) x++; return 0; }
"""

_DRIVER = """
import ctypes, sys, threading
lib = ctypes.CDLL(sys.argv[1])
gt = lib.__afl_guarded_call_timeout
gt.restype = ctypes.c_int
gt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
rc = gt(ctypes.cast(getattr(lib, sys.argv[2]), ctypes.c_void_p), b"x", 1, int(sys.argv[3]))
th = threading.Thread(target=lambda: None)
th.start()
th.join()
print("rc", rc, "joined", flush=True)
"""


def _clang_asan_rt() -> Path | None:
    r = subprocess.run(["clang", "-print-resource-dir"], capture_output=True, text=True)
    rt = Path(r.stdout.strip()) / "lib/linux/libclang_rt.asan-x86_64.so"
    return rt if r.returncode == 0 and rt.exists() else None


def _build(tmp: Path, link_rt: str) -> Path:
    c = tmp / "t.c"
    c.write_text(textwrap.dedent(_SRC))
    obj, so = tmp / "t.o", tmp / "t.so"
    asan = ["-fsanitize=address", "-fsanitize-recover=address"]
    cc = ["clang", "-O1", "-g", "-fPIC", *asan, "-include", str(SHIM), "-c", str(c), "-o", str(obj)]
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    link = ["clang", "-shared", "-fsanitize=address", link_rt, str(obj), "-o", str(so), "-ldl"]
    r = subprocess.run(link, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return so


def _slow_symbolizer(tmp: Path) -> Path:
    # ASAN accepts only known symbolizer names; the script just stalls.
    sym = tmp / "bin" / "llvm-symbolizer"
    sym.parent.mkdir()
    sym.write_text(f"#!/bin/sh\nsleep {SYMBOLIZER_SLEEP_S}\n")
    sym.chmod(0o755)
    return sym


def _run(tmp: Path, entry: str, runtime: Runtime = Runtime.SLOW_REPORT):
    """Guarded call of *entry*, then a thread start/join; None on a hang."""
    slow_report = runtime is Runtime.SLOW_REPORT
    rt = _clang_asan_rt() if slow_report else _resolve_asan()
    if rt is None:
        pytest.skip("ASAN runtime not installed")
    so = _build(tmp, "-shared-libasan" if slow_report else "-lasan")
    drop = ("__AFL_", "AFL_", "ASAN_OPTIONS", "LD_PRELOAD", "FUZZER_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["LD_PRELOAD"] = str(rt)
    env["ASAN_OPTIONS"] = "halt_on_error=0:abort_on_error=0:verify_asan_link_order=0:detect_leaks=0"
    if slow_report:
        env["ASAN_OPTIONS"] += f":external_symbolizer_path={_slow_symbolizer(tmp)}"
    cmd = [sys.executable, "-c", textwrap.dedent(_DRIVER), str(so), entry, str(TIMEOUT_US)]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=HANG_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return None


@requires_clang
def test_regression_timeout_in_asan_report(tmp_path):
    r = _run(tmp_path, "report_then_spin")

    assert r is not None, "thread join deadlocked after a timeout inside an ASAN report"
    assert f"rc {GUARD_TIMEOUT_RC} joined" in r.stdout, r.stderr[-500:]
    assert "heap-buffer-overflow" in r.stderr


@requires_clang
def test_timeout_without_report_still_fires(tmp_path):
    # Falsification: deferral must not disable timeouts when no report runs.
    r = _run(tmp_path, "spin")

    assert r is not None
    assert f"rc {GUARD_TIMEOUT_RC} joined" in r.stdout, r.stderr[-500:]


@requires_clang
def test_timeout_after_report_on_preloaded_libasan(tmp_path):
    # Adversarial: the libasan the fuzzer preloads must clear the report flag
    # once a recover-mode report ends, or every later hang is never cut.
    r = _run(tmp_path, "report_then_spin", Runtime.PRELOADED)

    assert r is not None, "report flag stayed set: the hang after the report was never cut"
    assert f"rc {GUARD_TIMEOUT_RC} joined" in r.stdout, r.stderr[-500:]
