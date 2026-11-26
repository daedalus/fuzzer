"""PR #468 review: tmin/replay in-process gaps.

- Spawn-only options (-F, -A, -c) are ignored in-process: warn.
- Direct + globals restore still shares libc/heap state between candidates:
  the final check runs in a fresh process (loader), so the minimum stands
  on its own.
- Direct runners for one .so share one lock: they share its globals.
- `_tmin_backend` keeps executables on spawn.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

TARGET_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    for (size_t i = 0; i < len; i++)
        if (buf[i] == 'x') abort();
    return 0;
}
"""

# 'a' arms state in libc's environment, outside the .so's globals: the
# snapshot restore cannot undo it. An armed 'x' aborts.
ARM_VAR = "FUZZER_TMIN_REVIEW_ARMED"
LIBC_STATE_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (buf[i] == 'a') setenv("@VAR@", "1", 1);
        if (buf[i] == 'x' && getenv("@VAR@")) abort();
    }
    return 0;
}
""".replace("@VAR@", ARM_VAR)
LIBC_STATE_INPUT = b"zzazzxzz"

CRASH_INPUT = b"aaxbb"
OK = 0


@pytest.fixture(autouse=True)
def _restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
    os.unsetenv(ARM_VAR)  # set by the target in C, unknown to os.environ


def _build(tmp_path: Path, src: str, name: str = "t.so") -> str:
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    d = tmp_path / "so"
    d.mkdir(exist_ok=True)
    c = d / "t.c"
    c.write_text(src)
    so = d / name
    cmd = ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(c), str(SHIM)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"build failed: {proc.stderr[-400:]}")
    return str(so)


def _tmin(monkeypatch, tmp_path, target, data, *flags) -> tuple[int, Path]:
    inp = tmp_path / "crash.bin"
    inp.write_bytes(data)
    out = tmp_path / "min.bin"
    argv = ["fuzzer-tool", "tmin", target, str(inp), "-O", str(out), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    return commands.main(), out


class TestSpawnOnlyWarning:
    @pytest.mark.parametrize(
        ("flags", "named"),
        [(("-F",), "-F"), (("-c",), "-c"), (("-A", "{file}"), "-A")],
    )
    def test_warns_in_process(self, monkeypatch, tmp_path, capsys, flags, named):
        so = _build(tmp_path, TARGET_SRC)
        rc, _ = _tmin(
            monkeypatch, tmp_path, so, CRASH_INPUT, "--inprocess-func", "fuzz_test", *flags
        )
        assert rc == OK
        err = capsys.readouterr().err
        assert "ignored" in err
        assert named in err

    def test_quiet_on_spawn(self, monkeypatch, tmp_path, capsys):
        # Falsification: an executable uses -F; no warning.
        exe = shutil.which("true")
        assert exe is not None
        _tmin(monkeypatch, tmp_path, exe, CRASH_INPUT, "-F")
        assert "ignored" not in capsys.readouterr().err

    def test_replay_warns(self, monkeypatch, tmp_path, capsys):
        so = _build(tmp_path, TARGET_SRC)
        inp = tmp_path / "c.bin"
        inp.write_bytes(CRASH_INPUT)
        argv = ["fuzzer-tool", "replay", so, str(inp), "-F", "--inprocess-func", "fuzz_test"]
        monkeypatch.setattr(sys, "argv", argv)
        commands.main()
        assert "-F" in capsys.readouterr().err


class TestFreshVerify:
    def test_regression_libc_state(self, monkeypatch, tmp_path):
        # Direct + globals: the reducer accepts a lone "x" (libc state leaks);
        # the fresh-process check rejects it, so the original is returned.
        so = _build(tmp_path, LIBC_STATE_SRC)
        flags = ("--inprocess-func", "fuzz_test", "--inprocess-direct")
        rc, out = _tmin(monkeypatch, tmp_path, so, LIBC_STATE_INPUT, *flags)
        assert rc == OK
        assert out.read_bytes() == LIBC_STATE_INPUT

    def test_stateless_still_minimizes(self, monkeypatch, tmp_path):
        # Control: a stateless target's minimum survives the fresh check.
        so = _build(tmp_path, TARGET_SRC)
        flags = ("--inprocess-func", "fuzz_test", "--inprocess-direct")
        rc, out = _tmin(monkeypatch, tmp_path, so, CRASH_INPUT, *flags)
        assert rc == OK
        assert out.read_bytes() == b"x"


class TestTargetLock:
    def test_same_target_same_lock(self, tmp_path):
        a = tmp_path / "a.so"
        a.write_bytes(b"")
        link = tmp_path / "link.so"
        link.symlink_to(a)
        assert commands._target_lock(str(a)) is commands._target_lock(str(link))

    def test_other_target_other_lock(self, tmp_path):
        # Falsification: distinct files never share a lock.
        a, b = tmp_path / "a.so", tmp_path / "b.so"
        a.write_bytes(b"")
        b.write_bytes(b"")
        assert commands._target_lock(str(a)) is not commands._target_lock(str(b))


def test_backend_spawns_executable(monkeypatch, tmp_path):
    # Executable + in-process flags: _tmin_backend itself must pick spawn.
    seen = []
    monkeypatch.setattr(commands, "cmd_tmin", lambda a: seen.append(a) or 0)
    exe = shutil.which("true")
    assert exe is not None
    argv = ["fuzzer-tool", "tmin", exe, "c.bin", "--inprocess", "--inprocess-direct"]
    monkeypatch.setattr(sys, "argv", argv)
    commands.main()
    backend = commands._tmin_backend(seen[-1])
    assert backend.run is None
    assert backend.verify is None
