"""AntiFuzz-evasion preload shim (antifuzz_evade.c).

Compiles a probe that reproduces AntiFuzz's self-ptrace check (§4.2) and a
measured sleep (§4.3), then runs it with and without the preload to prove the
shim neutralises both while leaving the real calls intact when opted out.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from fuzzer_tool.adapters.evade_shim import build_evade_shim, evade_ld_preload

ADAPTERS = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters"
SHIM_SRC = ADAPTERS / "antifuzz_evade.c"

# Mirrors AntiFuzz: TRACEME failing means "being traced" -> exit 42.
# Then sleep(2); print elapsed ms so the caller can see it was skipped.
_PROBE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <sys/ptrace.h>
#include <time.h>
#include <unistd.h>
int main(void) {
    if (ptrace(PTRACE_TRACEME, 0, NULL, NULL) == -1) return 42;
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    sleep(2);
    clock_gettime(CLOCK_MONOTONIC, &b);
    long ms = (b.tv_sec - a.tv_sec) * 1000 + (b.tv_nsec - a.tv_nsec) / 1000000;
    printf("%ld\n", ms);
    return 0;
}
"""


def _cc() -> str:
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler")
    return cc


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("evade")
    src = d / "probe.c"
    src.write_text(_PROBE)
    out = d / "probe"
    subprocess.run([_cc(), "-O0", "-o", str(out), str(src)], check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def shim() -> str:
    _cc()  # ensure a compiler exists before building
    so = build_evade_shim()
    if so is None:
        pytest.skip("evade shim did not build")
    return so


def _run(probe: Path, preload: str | None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    if preload:
        env["LD_PRELOAD"] = preload

    # Real self-ptrace from the forked probe: if the shim fails to fake it,
    # the probe is genuinely traced by nothing, so rc=0 without a tracer.
    return subprocess.run([str(probe)], capture_output=True, timeout=10, env=env)


def test_regression_self_ptrace_and_sleep_evaded(probe: Path, shim: str) -> None:
    # With the shim: TRACEME is faked (rc 0, not 42) and sleep returns at once.
    result = _run(probe, shim)

    assert result.returncode == 0, result.stderr.decode()
    assert int(result.stdout.strip()) < 500, "sleep(2) was not neutralised"


def test_sleep_opt_out_restores_real_wait(probe: Path, shim: str) -> None:
    # Falsification: ANTIFUZZ_EVADE_SLEEP=0 must let the real sleep run.
    env = os.environ.copy()
    env["LD_PRELOAD"] = shim
    env["ANTIFUZZ_EVADE_SLEEP"] = "0"
    env["ANTIFUZZ_EVADE_PTRACE"] = "1"  # keep ptrace faked so rc stays 0

    start = time.monotonic()
    result = subprocess.run([str(probe)], capture_output=True, timeout=10, env=env)
    elapsed = time.monotonic() - start

    assert result.returncode == 0
    assert elapsed >= 1.5, "real sleep should run when opted out"


def test_ptrace_opt_out_exposes_check(probe: Path, shim: str) -> None:
    # Falsification: with ptrace evasion off, the self-check sees no tracer
    # and still succeeds (rc 0) — proving rc 42 only comes from a real tracer,
    # i.e. the evaded case is doing real work, not a no-op probe.
    env = os.environ.copy()
    env["LD_PRELOAD"] = shim
    env["ANTIFUZZ_EVADE_PTRACE"] = "0"
    env["ANTIFUZZ_EVADE_SLEEP"] = "1"

    result = subprocess.run([str(probe)], capture_output=True, timeout=10, env=env)

    assert result.returncode == 0


def test_evade_ld_preload_prepends_once() -> None:
    so = build_evade_shim()
    if so is None:
        pytest.skip("evade shim did not build")

    assert evade_ld_preload(None) == so
    assert evade_ld_preload("x.so").split(":")[0] == so
    # Idempotent: already present -> unchanged.
    assert evade_ld_preload(f"{so}:x.so") == f"{so}:x.so"
