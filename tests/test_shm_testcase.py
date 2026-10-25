"""SHM testcase delivery: ``__AFL_FUZZ_TEST_CASE_BUF`` / ``_LEN``.

The loader wrote every input to a file (lseek, write, ftruncate, lseek) and
the target read it back. With ``__AFL_SHM_FUZZ=1`` the loader writes
``{u32 len; u8 data[]}`` into a SysV segment named by ``__AFL_SHM_FUZZ_ID``
and the shim's macros read it in place. The shim also defines
``__AFL_HAVE_MANUAL_CONTROL``, which turns on the AFL++ ``main()`` already in
four in-tree harnesses; without a segment their macros read ``argv[1]`` or
stdin, as the old ``main()`` did, so crash replay keeps working.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled, mode_env
from fuzzer_tool.core.elf import AflMode, detect_afl_modes

REPO = Path(__file__).resolve().parents[1]
SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
TIMEOUT = 5.0
LOOP_MAX = 5
# Over the segment capacity (1 MiB): delivery truncates, never overruns.
HUGE = (1 << 20) + 4096

# Echo what each iteration received: length and a digest of the bytes.
HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned sum(const unsigned char *b, int n) {
    unsigned s = 0;
    for (int i = 0; i < n; i++) s = s * 31 + b[i];
    return s;
}

#ifdef __AFL_HAVE_MANUAL_CONTROL
int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
    while (__AFL_LOOP(LOOP_MAX)) {
        int len = __AFL_FUZZ_TEST_CASE_LEN;
        fprintf(stderr, "len=%d sum=%u\n", len, sum(buf, len));
        if (len == 3 && memcmp(buf, "OOB", 3) == 0) {
            volatile unsigned char c = buf[len];   /* one past the input */
            (void)c;
        }
    }
    return 0;
}
#else
int main(void) { fprintf(stderr, "no manual control\n"); return 0; }
#endif
"""


def _sum(data: bytes) -> int:
    s = 0
    for b in data:
        s = (s * 31 + b) & 0xFFFFFFFF
    return s


def _build(d: Path, name: str, *flags: str) -> Path:
    src = d / "h.c"
    src.write_text(HARNESS)
    exe = d / name
    cmd = [
        "clang", "-O0", "-g", "-fsanitize-coverage=trace-pc-guard", *flags,
        f"-DLOOP_MAX={LOOP_MAX}", "-include", str(SHIM), "-o", str(exe), str(src),
    ]  # fmt: skip
    built = subprocess.run(cmd, capture_output=True, text=True)
    if built.returncode != 0:
        pytest.skip(f"harness did not build: {built.stderr[-300:]}")
    return exe


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    return _build(tmp_path_factory.mktemp("shmfuzz"), "h")


@pytest.fixture(scope="module")
def asan_exe(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    return _build(tmp_path_factory.mktemp("shmfuzz_asan"), "h_asan", "-fsanitize=address")


@pytest.fixture
def fsrv():
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    made = []

    def make(target: Path, env: dict[str, str] | None = None) -> ForkserverRunner:
        r = ForkserverRunner(
            str(target), timeout=TIMEOUT, env=mode_env(str(target)) if env is None else env
        )
        made.append(r)
        if not r.start():
            pytest.skip("forkserver failed to start")
        return r

    yield make
    for r in made:
        r.stop()


def _echo(err: str) -> tuple[int, int]:
    m = re.search(r"len=(\d+) sum=(\d+)", err)
    assert m, f"no echo in stderr: {err!r}"
    return int(m.group(1)), int(m.group(2))


def test_detects_shm_testcase(exe):
    assert AflMode.SHM_TESTCASE in detect_afl_modes(str(exe))
    assert mode_env(str(exe)).get("__AFL_SHM_FUZZ") == "1"


def test_regression_shm_testcase_delivery(exe, fsrv):
    """Each input arrives intact via the segment, longer and shorter in turn."""
    r = fsrv(exe)
    assert "shmfuzz" in r.capabilities
    for data in (b"hello world", b"ab", b"x" * 5000, b"", b"tail"):
        rc, err = r.run_one(data)
        assert rc == 0
        assert _echo(err) == (len(data), _sum(data)), data[:16]


def test_file_delivery_without_env(exe, fsrv):
    """Falsification: no __AFL_SHM_FUZZ, the macros read the staged input."""
    r = fsrv(exe, env={"__AFL_PERSISTENT": "1"})
    assert "shmfuzz" not in r.capabilities
    for data in (b"hello world", b"ab"):
        assert _echo(r.run_one(data)[1]) == (len(data), _sum(data))


def test_oversized_input_truncated(exe, fsrv):
    """Adversarial: an input past capacity is cut, not written past the end."""
    r = fsrv(exe)
    rc, err = r.run_one(b"z" * HUGE)
    assert rc == 0
    n, _ = _echo(err)
    assert 0 < n < HUGE
    assert _echo(r.run_one(b"ok")[1]) == (2, _sum(b"ok"))


def test_standalone_reads_argv_file(exe, tmp_path):
    """Crash replay: `target file` still reads the file."""
    f = tmp_path / "in"
    f.write_bytes(b"from file")
    done = subprocess.run(
        [str(exe), str(f)], capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    assert _echo(done.stderr) == (9, _sum(b"from file"))


def test_standalone_reads_stdin(exe):
    done = subprocess.run([str(exe)], input=b"from stdin", capture_output=True)
    assert _echo(done.stderr.decode()) == (10, _sum(b"from stdin"))


@pytest.mark.parametrize("via_shm", [True, False], ids=["shm", "standalone"])
def test_read_past_input_is_reported(asan_exe, fsrv, tmp_path, via_shm):
    """Adversarial: the fixed buffer must not hide a one-byte overread from ASAN."""
    if via_shm:
        rc, err = fsrv(asan_exe).run_one(b"OOB")
    else:
        f = tmp_path / "oob"
        f.write_bytes(b"OOB")
        done = subprocess.run(
            [str(asan_exe), str(f)],
            capture_output=True,
            text=True,
            env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
        )
        rc, err = done.returncode, done.stderr
    assert rc != 0
    assert "AddressSanitizer" in err
