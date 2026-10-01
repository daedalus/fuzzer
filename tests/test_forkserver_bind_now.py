"""The forkserver target must start with LD_BIND_NOW=1.

With lazy binding every forked child resolves the PLT entries it touches
again, since the parent never touched them. Binding once before the fork
server loop means every child inherits resolved slots (AFL does the same;
lcamtuf, "Fuzzing random programs without execve()").

Only the forkserver child gets it: the fork+exec fallback would pay eager
binding of every symbol on each exec.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.adapters.shm import ShmCoverage

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(_ROOT, "src", "fuzzer_tool", "adapters", "afl_shim.c")

# Reports the variable on stderr each run, so the assertion reads what the
# executing child actually saw.
_TARGET = r"""
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    const char *v = getenv("LD_BIND_NOW");
    fprintf(stderr, "BIND=[%s]\n", v ? v : "<unset>");
    return 0;
}
"""


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    d = tmp_path_factory.mktemp("bindnow")
    src = d / "t.c"
    src.write_text(_TARGET)
    exe = d / "t"
    r = subprocess.run(
        ["clang", "-O0", "-include", SHIM, "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(exe)


def _run(target: str, extra_env: dict[str, str]) -> tuple[str, str]:
    """(exec mode, child's stderr) for one execution."""
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    shm = ShmCoverage(size=8192)
    env = {"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size), **extra_env}
    r = ForkserverRunner(target, timeout=2.0, env=env)
    try:
        if not r.start():
            pytest.skip("forkserver failed to start")
        _rc, stderr = r.run_one(b"x")
        return r.exec_mode, stderr
    finally:
        r.stop()
        shm.cleanup()


@pytest.fixture
def no_bind_now(monkeypatch):
    monkeypatch.delenv("LD_BIND_NOW", raising=False)


def test_forkserver_child_binds_now(target, no_bind_now):
    mode, stderr = _run(target, {})
    assert mode == "forkserver"
    assert "BIND=[1]" in stderr


def test_user_value_is_kept(target, no_bind_now):
    # Adversarial: an explicit caller choice must not be overwritten.
    mode, stderr = _run(target, {"LD_BIND_NOW": "custom"})
    assert mode == "forkserver"
    assert "BIND=[custom]" in stderr


def test_loader_itself_stays_lazy(no_bind_now):
    # Falsification: setting it in the loader's own environment would also
    # reach the fork+exec fallback. The runner must not inject it there.
    r = ForkserverRunner("/bin/true", timeout=1.0, env={})
    assert "LD_BIND_NOW" not in r.env_overrides
