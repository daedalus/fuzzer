"""`tmin` picks its .so backend with replay's flags (no `-N`).

  neither                         subprocess loader (a .so cannot be exec'd)
  --inprocess                     subprocess loader (crash-isolated)
  --inprocess-direct              direct_lite ctypes call
  --inprocess --inprocess-direct  direct ctypes call

An executable stays on spawn. The shim binds `__afl_area` at dlopen, so
every run builds its own `.so`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.services.minimize import ExecMode

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# Aborts when any byte is CRASH_BYTE: the 1-byte input is the minimum.
CRASH_BYTE = b"x"
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

CRASH_INPUT = b"aaaa" + CRASH_BYTE + b"bbbb"
BENIGN_INPUT = b"aaaabbbb"
OK = 0
FAILED = 1

IN_PROCESS_MODES = {
    ExecMode.LOADER: ("--inprocess",),
    ExecMode.DIRECT_LITE: ("--inprocess-direct",),
    ExecMode.DIRECT: ("--inprocess", "--inprocess-direct"),
}


@pytest.fixture(autouse=True)
def _restore_env():
    """Direct modes write os.environ (input path, SHM id); undo it."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def build_so(tmp_path):
    """Factory: a fresh shim-linked target `.so` per call."""
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    count = 0

    def build() -> str:
        nonlocal count
        count += 1
        d = tmp_path / f"so{count}"
        d.mkdir()
        src = d / "t.c"
        src.write_text(TARGET_SRC)
        so = d / "t.so"
        cmd = ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"target failed to build: {proc.stderr[-400:]}")
        return str(so)

    return build


def _main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "tmin", *argv])
    return commands.main()


def _input(tmp_path: Path, data: bytes) -> str:
    p = tmp_path / f"in_{data.hex()}.bin"
    p.write_bytes(data)
    return str(p)


class TestFlags:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        """Run the real parser; capture the Namespace cmd_tmin gets."""
        self.seen = []
        monkeypatch.setattr(commands, "cmd_tmin", lambda a: self.seen.append(a) or 0)
        self.monkeypatch = monkeypatch

    def _args(self, *extra: str):
        assert _main(self.monkeypatch, "t.so", "crash.bin", *extra) == 0
        return self.seen[-1]

    def test_defaults_off(self):
        a = self._args()
        assert (a.inprocess, a.inprocess_direct, a.inprocess_func) == (False, False, None)

    def test_flags_parse(self):
        a = self._args("--inprocess", "--inprocess-direct", "--inprocess-func", "fuzz_test")
        assert (a.inprocess, a.inprocess_direct, a.inprocess_func) == (True, True, "fuzz_test")

    def test_no_repeat_flag(self):
        # Adversarial: -N is replay-only; tmin must reject it.
        with pytest.raises(SystemExit):
            self._args("-N", "3")


@pytest.mark.parametrize(
    "flags", [(), *IN_PROCESS_MODES.values()], ids=["default", *(m.value for m in IN_PROCESS_MODES)]
)
class TestMinimize:
    def _tmin(self, monkeypatch, tmp_path, so, data, flags):
        out = tmp_path / "min.bin"
        argv = (so, _input(tmp_path, data), "-O", str(out), "--inprocess-func", "fuzz_test")
        return _main(monkeypatch, *argv, *flags), out

    def test_minimizes_so(self, monkeypatch, tmp_path, build_so, flags):
        rc, out = self._tmin(monkeypatch, tmp_path, build_so(), CRASH_INPUT, flags)
        assert rc == OK
        assert out.read_bytes() == CRASH_BYTE

    def test_benign_not_reproduced(self, monkeypatch, tmp_path, build_so, flags):
        # Falsification: a clean input has nothing to minimize.
        rc, out = self._tmin(monkeypatch, tmp_path, build_so(), BENIGN_INPUT, flags)
        assert rc == FAILED
        assert not out.exists()

    def test_control_two_runs_agree(self, monkeypatch, tmp_path, build_so, flags):
        """The oracle against itself: the same input minimizes the same way."""
        outs = []
        for name in "ab":
            d = tmp_path / name
            d.mkdir()
            rc, out = self._tmin(monkeypatch, d, build_so(), CRASH_INPUT, flags)
            assert rc == OK
            outs.append(out.read_bytes())
        assert outs[0] == outs[1]


@pytest.mark.parametrize("mode", [ExecMode.DIRECT_LITE, ExecMode.DIRECT], ids=lambda m: m.value)
def test_missing_func(monkeypatch, tmp_path, build_so, mode, capsys):
    # Adversarial: a wrong entry point in direct mode is a clean error.
    argv = (build_so(), _input(tmp_path, CRASH_INPUT), "--inprocess-func", "no_such_function")
    assert _main(monkeypatch, *argv, *IN_PROCESS_MODES[mode]) == FAILED
    assert "no_such_function" in capsys.readouterr().err


def test_executable_spawns(monkeypatch, tmp_path):
    # An executable ignores the flags: spawn, as replay does.
    seen = []
    monkeypatch.setattr(commands, "cmd_tmin", lambda a: seen.append(a) or 0)
    exe = shutil.which("true")
    assert exe is not None
    _main(monkeypatch, exe, "crash.bin", "--inprocess", "--inprocess-direct")
    assert commands._replay_exec_mode(seen[-1]) is ExecMode.SPAWN


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        (["tmin", "t.so", "crash.bin"], "t.so"),
        (["tmin", "--inprocess-direct"], None),  # adversarial: option, not a path
    ],
)
def test_wrapper_preload(argv, want):
    from fuzzer_tool.cli.ldpreload_wrapper import _fuzz_target

    assert _fuzz_target(argv) == want


# 'a' arms a global; an armed 'x' aborts. Alone, "x" is clean: "ax" is minimal
# only when each candidate starts from load-time globals.
STATEFUL_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
static int armed;
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (buf[i] == 'a') armed = 1;
        if (buf[i] == 'x' && armed) abort();
    }
    return 0;
}
"""
STATEFUL_INPUT = b"zzazzxzz"
STATEFUL_MIN = b"ax"


class TestSnapshotRestore:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        self.seen = []
        self.monkeypatch = monkeypatch

    def _args(self, *extra: str):
        self.monkeypatch.setattr(commands, "cmd_tmin", lambda a: self.seen.append(a) or 0)
        assert _main(self.monkeypatch, "t.so", "crash.bin", *extra) == 0
        return self.seen[-1].snapshot_restore

    @pytest.mark.parametrize(
        ("flags", "want"),
        [
            ((), "off"),
            (("--snapshot-restore", "globals"), "globals"),
            (("--snapshot-restore", "globals", "--no-snapshot-restore"), "off"),
        ],
    )
    def test_flag(self, flags, want):
        assert self._args(*flags) == want

    def test_rejects_bad_value(self):
        # Adversarial: only StateRestore values parse.
        with pytest.raises(SystemExit):
            self._args("--snapshot-restore", "heap")

    def _tmin(self, tmp_path, build_so, *flags):
        so = tmp_path / "so"
        so.mkdir()
        src = so / "t.c"
        src.write_text(STATEFUL_SRC)
        lib = so / "t.so"
        cmd = ["clang", "-O1", "-fPIC", "-shared", "-o", str(lib), str(src), str(SHIM)]
        if shutil.which("clang") is None or subprocess.run(cmd).returncode != 0:
            pytest.skip("clang build failed")
        out = tmp_path / "min.bin"
        argv = (str(lib), _input(tmp_path, STATEFUL_INPUT), "-O", str(out))
        rc = _main(self.monkeypatch, *argv, "--inprocess-func", "fuzz_test", *flags)
        assert rc == OK
        return out.read_bytes()

    @pytest.mark.parametrize("mode", [ExecMode.DIRECT_LITE, ExecMode.DIRECT], ids=lambda m: m.value)
    def test_globals_isolates(self, tmp_path, build_so, mode):
        flags = (*IN_PROCESS_MODES[mode], "--snapshot-restore", "globals")
        assert self._tmin(tmp_path, build_so, *flags) == STATEFUL_MIN

    def test_loader_isolates(self, tmp_path, build_so):
        # Control: the loader forks per run, so it agrees with globals restore.
        assert self._tmin(tmp_path, build_so, "--inprocess") == STATEFUL_MIN

    def test_off_leaks_state(self, tmp_path, build_so):
        # Falsification: without restore the armed global leaks into later
        # candidates and a lone "x" passes; a crash then resets globals, the
        # final re-check rejects "x", and tmin falls back to the original.
        got = self._tmin(tmp_path, build_so, "--inprocess-direct")
        assert got == STATEFUL_INPUT
