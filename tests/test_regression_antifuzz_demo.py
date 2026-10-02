"""antifuzz_demo.c: benchmarkable bug, in-process safety, evade-shim scope.

Plain clang builds (no ASAN runtime needed): these check the target's
control flow, not the overflow report.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

DEMO_SRC = Path(__file__).parent.parent / "targets" / "antifuzz_demo.c"
MAGIC = b"crsh"
ALL_GATES_OFF = {"AF_COVERAGE": "0", "AF_CRASH": "0", "AF_SPEED": "0", "AF_PTRACE": "0"}

# Models afl_shim.c's cmplog layer 1 (libc memcmp interposition): logs
# each memcmp operand pair to stderr as "CMP <a> <b>".
_MEMCMP_HOOK = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
int memcmp(const void *a, const void *b, size_t n) {
    static int (*real)(const void *, const void *, size_t);
    if (!real) real = dlsym(RTLD_NEXT, "memcmp");
    fprintf(stderr, "CMP %.*s %.*s\n", (int)n, (const char *)a, (int)n, (const char *)b);
    return real(a, b, n);
}
"""

# Calls fuzz_shm_run twice in one process, as direct_lite does, then
# reports the process's tracer pid.
_INPROC_DRIVER = """
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
for _ in range(2):
    lib.fuzz_shm_run(b"zzzz", 4)
status = open("/proc/self/status").read()
tracer = next(l for l in status.splitlines() if l.startswith("TracerPid:"))
print(tracer.split()[1])
"""


def _clang() -> str:
    cc = shutil.which("clang")
    if cc is None:
        pytest.skip("no clang")
    return cc


@pytest.fixture(scope="module")
def build(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("antifuzz_demo")
    cc = _clang()

    exe, so, hook = d / "demo", d / "demo.so", d / "hook.so"
    hook_src = d / "hook.c"
    hook_src.write_text(_MEMCMP_HOOK)

    common = [cc, "-O0", "-fno-builtin"]
    subprocess.run([*common, "-o", str(exe), str(DEMO_SRC)], check=True, capture_output=True)
    subprocess.run(
        [*common, "-shared", "-fPIC", "-o", str(so), str(DEMO_SRC)], check=True, capture_output=True
    )
    subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(hook), str(hook_src), "-ldl"],
        check=True,
        capture_output=True,
    )
    return {"exe": exe, "so": so, "hook": hook}


def _run_exe(build: dict[str, Path], data: bytes, extra: dict[str, str]) -> str:
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    env.update(ALL_GATES_OFF, LD_PRELOAD=str(build["hook"]), **extra)
    res = subprocess.run([str(build["exe"])], input=data, capture_output=True, timeout=10, env=env)
    return res.stderr.decode(errors="replace")


def test_regression_demo_magic_reachable_via_cmplog(build) -> None:
    # AF_HASHCMP=0: the magic compare reaches memcmp, so cmplog sees "crsh".
    stderr = _run_exe(build, b"zzzz", {"AF_HASHCMP": "0"})

    assert f"zzzz {MAGIC.decode()}" in stderr


def test_demo_hashcmp_hides_magic(build) -> None:
    # Falsification: default gate keeps the hash compare; no operand leaks.
    stderr = _run_exe(build, b"zzzz", {})

    assert MAGIC.decode() not in stderr


def test_regression_demo_inprocess_no_self_trace(build) -> None:
    # Adversarial: all gates on (default) in a long-lived host process.
    env = {k: v for k, v in os.environ.items() if not k.startswith("AF_")}
    env["AF_SPEED"] = "0"  # keep the test fast; delay is not under test

    res = subprocess.run(
        [sys.executable, "-c", _INPROC_DRIVER, str(build["so"])],
        capture_output=True,
        timeout=20,
        env=env,
    )

    assert res.returncode == 0, res.stderr.decode()
    assert res.stdout.strip() == b"0", "fuzz_shm_run made the host process traced"


def _evade_fuzzer(inprocess: object) -> Fuzzer:
    f = Fuzzer.__new__(Fuzzer)
    f._antifuzz_evade = True
    f._inprocess_runner = inprocess
    return f


def test_regression_evade_skipped_inprocess(monkeypatch, capsys) -> None:
    # LD_PRELOAD cannot reach an already-running host: say so, leave env alone.
    monkeypatch.setenv("LD_PRELOAD", "keep.so")

    _evade_fuzzer(object())._install_antifuzz_evade()

    assert os.environ["LD_PRELOAD"] == "keep.so"
    assert "in-process" in capsys.readouterr().out


def test_evade_installed_out_of_process(monkeypatch) -> None:
    # Falsification: subprocess modes still get the preload.
    monkeypatch.setattr(
        "fuzzer_tool.adapters.evade_shim.evade_ld_preload", lambda cur: f"evade.so:{cur}"
    )
    monkeypatch.setenv("LD_PRELOAD", "keep.so")

    _evade_fuzzer(None)._install_antifuzz_evade()

    assert os.environ["LD_PRELOAD"] == "evade.so:keep.so"


def test_regression_evade_refreshes_env_cache(monkeypatch) -> None:
    # A _clean_env(None) snapshot taken before install must not hide the shim.
    from fuzzer_tool.adapters import process

    monkeypatch.setattr(
        "fuzzer_tool.adapters.evade_shim.evade_ld_preload", lambda cur: f"evade.so:{cur}"
    )
    monkeypatch.setenv("LD_PRELOAD", "keep.so")
    monkeypatch.setattr(process, "_clean_env_cache", None)
    process._clean_env(None)

    _evade_fuzzer(None)._install_antifuzz_evade()

    assert process._clean_env(None)["LD_PRELOAD"] == "evade.so:keep.so"
