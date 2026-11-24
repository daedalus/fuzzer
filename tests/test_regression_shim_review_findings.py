"""Regressions for the findings in the afl_shim.c review report
(docs/../afl_shim_code_review, 2026-11-24, on top of 93cca99).

  F1 (P1) the wcscasecmp fallback took three parameters while its slot is a
      two-argument typedef -- calling it through the slot reads the third
      argument register as n, and when it holds 0 the fallback reports
      "strings equal" without comparing anything. Pinned by the source
      arity check (the register value is build-specific) and by a
      forced-fallback semantics test: a preload lib that fails dlsym for
      wcscasecmp drives the shim onto the fallback path, which no other
      test reaches (both existing wcscasecmp tests resolve libc).
  F2 (P2) a guard escape between __ltl_open's CAS and its fd store (the
      open(2) is interruptible) left the sink in LTL_FD_OPENING forever:
      every later event span the 4096-yield loop and was dropped. The
      landing path now resets it, as the fork child already does.
  F4 (P3) __afl_env_num was dead code in default builds and AFL_MAP_SIZE
      re-implemented the digits-only parse inline; one capped helper now
      serves both, and the warning-free build tests pin it.
  F5 (P3) an explicitly requested write guard that could not be armed
      failed silently; now one stderr line, like every other loud failure.
  F6 (P4) build warnings in three of four configurations: dead
      __afl_fb_wcslen, six edge-only forward declarations compiled in the
      preload build, and gcc's -Wmissing-attributes noise on the
      afl_cmp_* aliases (matching those attributes would be a lie: ours
      record comparisons and are not pure).

F3 of the report (strtol prefix parses) was fixed and tested in the
previous commit (test_regression_shim_channel_gates.py).
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from tests.conftest import requires_clang, requires_gcc
from tests.test_regression_shim_audit import _build, _clean_env

REPO = Path(__file__).resolve().parents[1]
AFL_SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

MAP_ENTRIES = 8192
HANG_TIMEOUT_S = 30


# ── F1: the fallback must match its slot's arity ───────────────────────


@requires_clang
class TestWcscasecmpFallback:
    def test_regression_wcscasecmp_fallback_arity(self):
        """Two-arg fallback, delegating unbounded like __afl_fb_wcscmp.

        Called through the 2-arg slot, a 3-param body reads the third
        argument register as n (C11 6.5.2.2p9); n == 0 makes every
        comparison report "equal".
        """
        src = AFL_SHIM.read_text()
        assert not re.search(
            r"__afl_fb_wcscasecmp\(const wchar_t \*a, const wchar_t \*b, size_t n\)", src
        ), "fallback must take the slot's two arguments, not three"
        assert re.search(
            r"__afl_fb_wcscasecmp\(const wchar_t \*a, const wchar_t \*b\) \{\s*"
            r"for \(;;\) \{\s*"
            r"wchar_t x = __afl_fb_wlower\(\*a\+\+\), y = __afl_fb_wlower\(\*b\+\+\);",
            src,
        ), "fallback must run unbounded on wlower, terminated by the NUL pair"

    def test_regression_wcscasecmp_fallback_semantics(self, tmp_path):
        """Force the fallback path and check the compare results.

        A preload library fails dlsym("wcscasecmp") and passes everything
        else through, so __AFL_RESOLVE installs the fallback -- the path
        no other wcscasecmp test exercises (libc always resolves there).
        """
        fake = tmp_path / "fake_dlsym.so"
        fake_src = tmp_path / "fake_dlsym.c"
        fake_src.write_text(
            textwrap.dedent(
                """
                #define _GNU_SOURCE
                #include <dlfcn.h>
                #include <string.h>
                static void *(*real_dlsym)(void *, const char *);
                __attribute__((constructor)) static void init(void) {
                    real_dlsym = (void *(*)(void *, const char *))
                        dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
                }
                void *dlsym(void *handle, const char *name) {
                    if (!real_dlsym) init();
                    if (strcmp(name, "wcscasecmp") == 0) return NULL;
                    return real_dlsym ? real_dlsym(handle, name) : NULL;
                }
                """
            )
        )
        r = subprocess.run(
            ["clang", "-O1", "-shared", "-fPIC", "-o", str(fake), str(fake_src), "-ldl"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 0, r.stderr

        probe = tmp_path / "wcs_probe"
        probe_src = tmp_path / "wcs_probe.c"
        probe_src.write_text(
            textwrap.dedent(
                """
                #include <wchar.h>
                #include <stdio.h>
                int main(void) {
                    printf("r1=%d r2=%d r3=%d\\n",
                           wcscasecmp(L"AAAA", L"BBBB"), wcscasecmp(L"BBBB", L"AAAA"),
                           wcscasecmp(L"aaaa", L"AAAA"));
                    return 0;
                }
                """
            )
        )
        r = subprocess.run(
            [
                "clang",
                "-O1",
                "-D__AFL_CMPLOG=1",
                "-fno-builtin-wcscasecmp",
                "-include",
                str(AFL_SHIM),
                "-o",
                str(probe),
                str(probe_src),
                "-ldl",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, r.stderr[-600:]

        run = subprocess.run(
            [str(probe)], capture_output=True, text=True, timeout=60,
            env=_clean_env(LD_PRELOAD=str(fake)),
        )  # fmt: skip
        assert run.returncode == 0, run.stderr[-500:]
        assert run.stdout.strip() == "r1=-1 r2=1 r3=0"


# ── F2: an escape must not wedge the LTL sink in OPENING ───────────────

_LTL_SO = """
#include <stdint.h>
#include <stddef.h>
extern void __fuzz_event_at(unsigned, unsigned);
int emit_and_spin(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    __fuzz_event_at(0xBEEF, 0);   /* first event: the open(2) blocks */
    volatile size_t x = 0;
    for (;;) x++;
}
int emit_once(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    __fuzz_event_at(0xBEEF, 0);
    return 3;
}
"""

_LTL_DRIVER = """
import ctypes, os, sys
lib = ctypes.CDLL(sys.argv[1])
g = lib.__afl_guarded_call
g.restype = ctypes.c_int
g.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
gt = lib.__afl_guarded_call_timeout
gt.restype = ctypes.c_int
gt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
def fn(name):
    return ctypes.cast(getattr(lib, name), ctypes.c_void_p)

# 1. A timed call whose first event blocks opening the sink FIFO: the
#    budget runs out inside open(2), the guard escapes, and (pre-fix) the
#    fd stays LTL_FD_OPENING forever.
print(gt(fn("emit_and_spin"), b"x", 1, 2000), flush=True)

# 2. A reader exists now, so a recovered sink opens and records at once.
rd = os.open(sys.argv[2], os.O_RDONLY | os.O_NONBLOCK)
print(g(fn("emit_once"), b"y", 1), flush=True)
try:
    data = os.read(rd, 12)
except BlockingIOError:
    data = b""
print(data.hex(), flush=True)
os.close(rd)
"""


@requires_clang
def test_regression_ltl_opening_unwedged_after_escape(tmp_path):
    so = _build(tmp_path, _LTL_SO, shared=True)
    fifo = tmp_path / "events.fifo"
    os.mkfifo(fifo)

    script = textwrap.dedent(_LTL_DRIVER)
    try:
        r = subprocess.run(
            [os.sys.executable, "-c", script, str(so), str(fifo)],
            capture_output=True,
            text=True,
            timeout=HANG_TIMEOUT_S,
            env=_clean_env(__LTL_EVENTS_OUT=str(fifo)),
        )
    except subprocess.TimeoutExpired:
        r = None

    assert r is not None and r.returncode == 0, r and (r.returncode, r.stderr[-500:])
    tokens = r.stdout.split()
    assert tokens[:2] == ["-1", "3"], (r.stdout, r.stderr[-500:])
    assert len(tokens) == 3 and tokens[2] == "efbe00000000000000000000", (
        "the event after the escape was dropped: the LTL sink stayed wedged in OPENING"
    )


# ── F4/F6: the four documented build modes must be warning-free ────────

_MODES = {
    "default": [],
    "cmplog": ["-D__AFL_CMPLOG=1"],
    "preload": ["-D__AFL_PRELOAD_ONLY"],
    "private-heap": ["-D__AFL_PRIVATE_HEAP=1"],
}

_TRIVIAL_MAIN = "int main(void) { return 0; }\n"


def _build_clean(
    tmp_path, cc: str, mode_flags: list[str], name: str
) -> subprocess.CompletedProcess:
    src = tmp_path / f"{name}.c"
    src.write_text(_TRIVIAL_MAIN)
    out = tmp_path / name
    return subprocess.run(
        [cc, "-O1", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Werror",
         *mode_flags, "-include", str(AFL_SHIM), "-o", str(out), str(src), "-ldl"],
        capture_output=True, text=True, timeout=180,
    )  # fmt: skip


@requires_clang
@pytest.mark.parametrize("mode", sorted(_MODES))
def test_regression_shim_builds_warning_free_clang(tmp_path, mode):
    r = _build_clean(tmp_path, "clang", _MODES[mode], f"warn_clang_{mode}")
    assert r.returncode == 0, r.stderr[-800:]


@requires_gcc
@pytest.mark.parametrize("mode", sorted(_MODES))
def test_regression_shim_builds_warning_free_gcc(tmp_path, mode):
    r = _build_clean(tmp_path, "gcc", _MODES[mode], f"warn_gcc_{mode}")
    assert r.returncode == 0, r.stderr[-800:]


# ── F5: a requested write guard that cannot arm must say so ────────────

_ARM_FAIL_MARK = "could not be armed"


def _run_with_wguard(tmp_path, wguard: str) -> str:
    exe = _build(tmp_path, "int main(void) { return 0; }\n")
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = subprocess.run(
            [str(exe)], capture_output=True, text=True, timeout=60,
            env=_clean_env(
                __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES), __AFL_WGUARD=wguard
            ),
        )  # fmt: skip
    finally:
        shm.cleanup()
    assert r.returncode == 0, r.stderr[-500:]
    return r.stderr


@requires_clang
class TestWguardArmDiagnostic:
    def test_regression_wguard_unknown_mode_loud(self, tmp_path):
        """An explicit value that arms nothing must not fail silently."""
        assert _ARM_FAIL_MARK in _run_with_wguard(tmp_path, "banana"), (
            "__AFL_WGUARD=banana armed nothing and said nothing"
        )

    def test_regression_wguard_pkey_failure_loud(self, tmp_path):
        """No PKU on this host: pkey arming fails, and must say so."""
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.syscall(330, 0, 0) >= 0:  # SYS_pkey_alloc
            pytest.skip("pkey_alloc works here; the failure path cannot be driven")
        assert _ARM_FAIL_MARK in _run_with_wguard(tmp_path, "pkey"), (
            "pkey arming failed and the run continued unguarded in silence"
        )

    def test_control_wguard_mprotect_arms_quietly(self, tmp_path):
        """Arming successfully must stay silent (the diagnostic is for failures)."""
        assert _ARM_FAIL_MARK not in _run_with_wguard(tmp_path, "mprotect")
