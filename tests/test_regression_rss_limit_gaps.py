"""Regression: --rss-limit-mb gaps (file-input race, persistent .so).

File input: RLIMIT_AS was set by prlimit after posix_spawn returned; the
input file exists before spawn, so the target could run (constructor, first
allocation) unbounded. Measured 1 in 2000 spawns. The limit is now set
before exec. Persistent .so: the per-call child ran unbounded.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader

_LIMIT_MB = 64
_UNLIMITED = 43

# Reports, before main, whether RLIMIT_AS is already in place.
_PROBE = r"""
#include <sys/resource.h>
#include <unistd.h>
__attribute__((constructor(101))) static void early(void) {
    struct rlimit rl;
    getrlimit(RLIMIT_AS, &rl);
    if (rl.rlim_cur == RLIM_INFINITY) _exit(43);
}
int main(void) { return 0; }
"""

_HOG_SO = r"""
#include <stdlib.h>
#include <string.h>
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {
    if (!n || d[0] != 'M') return 0;
    char *p = malloc((size_t)512 << 20);
    if (!p) abort();
    memset(p, 1, (size_t)512 << 20);
    return p[n] == 1 ? 0 : 1;
}
"""


def _clang(tmp_path, src: str, out: str, *extra: str) -> str:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    c = tmp_path / "src.c"
    c.write_text(src)
    dst = tmp_path / out
    subprocess.run(["clang", "-O0", *extra, "-o", str(dst), str(c)], check=True)
    return str(dst)


@pytest.fixture
def probe(tmp_path) -> str:
    return _clang(tmp_path, _PROBE, "probe")


@pytest.fixture
def limit():
    process.set_rss_limit(_LIMIT_MB)
    yield
    process.set_rss_limit(0)


@pytest.fixture
def slow_parent(monkeypatch):
    # Widen the spawn -> prlimit window that measured 1/2000 into a certainty.
    real = process._track

    def _slow(pid: int) -> None:
        real(pid)
        time.sleep(0.05)

    monkeypatch.setattr(process, "_track", _slow)


def test_regression_file_input_limited_before_exec(probe, limit, slow_parent) -> None:
    rc, _, _ = process.run_target_fast(probe, b"x", timeout=10.0)

    assert rc == 0


def test_probe_sees_no_limit_when_off(probe, slow_parent) -> None:
    # Falsification: the probe does detect an unlimited start.
    rc, _, _ = process.run_target_fast(probe, b"x", timeout=10.0)

    assert rc == _UNLIMITED


def test_target_path_not_shell_parsed(tmp_path, probe, limit) -> None:
    # Adversarial: the pre-exec wrapper must not evaluate the path.
    odd = tmp_path / "t $(touch pwned) ;x"
    shutil.copy(probe, odd)
    rc, _, _ = process.run_target_fast(str(odd), b"x", timeout=10.0, env={"PWD": str(tmp_path)})

    assert rc == 0
    assert not (tmp_path / "pwned").exists()


@pytest.fixture
def hog_loader(tmp_path, request, monkeypatch):
    so = _clang(tmp_path, _HOG_SO, "hog.so", "-shared", "-fPIC")
    if request.param:
        monkeypatch.setenv(process.RSS_LIMIT_ENV, str(_LIMIT_MB))
    else:
        monkeypatch.delenv(process.RSS_LIMIT_ENV, raising=False)
    ldr = PersistentLoader(so, timeout=10.0)
    if not ldr.start():
        ldr.stop()
        pytest.skip("persistent loader failed to start")
    yield ldr
    ldr.stop()


@pytest.mark.parametrize("hog_loader", [True], indirect=True)
def test_regression_persistent_so_limited(hog_loader) -> None:
    rc, _ = hog_loader.run_one(b"M")

    assert process.signal_number(rc) == 6  # malloc failed -> abort()


@pytest.mark.parametrize("hog_loader", [False], indirect=True)
def test_persistent_so_unlimited_when_off(hog_loader) -> None:
    # Falsification: same input allocates without the flag.
    assert hog_loader.run_one(b"M")[0] == 0


@pytest.mark.parametrize("hog_loader", [True], indirect=True)
def test_persistent_small_input_unaffected(hog_loader) -> None:
    # Edge: the budget sits on top of the loader's own mapping.
    assert hog_loader.run_one(b"q")[0] == 0
