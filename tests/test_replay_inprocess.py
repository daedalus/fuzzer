"""`replay` picks its .so backend from the same flags as `fuzz`/`minimize`.

  neither                         subprocess loader (a .so cannot be exec'd)
  --inprocess                     subprocess loader (crash-isolated)
  --inprocess-direct              direct_lite ctypes call
  --inprocess --inprocess-direct  direct ctypes call

An executable stays on spawn whatever the flags. The shim binds
`__afl_area` at dlopen, so every direct run builds its own `.so`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.cli.commands import _replay_exec_mode
from fuzzer_tool.services.minimize import ExecMode

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# 'x' aborts; anything else returns cleanly.
TARGET_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    if (len && buf[0] == 'x') abort();
    return 0;
}
"""

CRASH = b"x"
BENIGN = b"a"
REPRODUCED = 0
NOT_REPRODUCED = 1

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
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "replay", *argv])
    return commands.main()


def _input(tmp_path: Path, data: bytes) -> str:
    p = tmp_path / f"in_{data.hex() or 'empty'}.bin"
    p.write_bytes(data)
    return str(p)


class TestFlags:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        """Run the real parser; capture the Namespace cmd_replay gets."""
        self.seen = []
        monkeypatch.setattr(commands, "cmd_replay", lambda a: self.seen.append(a) or 0)
        self.monkeypatch = monkeypatch

    def _args(self, target: str, *extra: str):
        assert _main(self.monkeypatch, target, "crash.bin", *extra) == 0
        return self.seen[-1]

    def test_defaults_off(self):
        a = self._args("t.so")
        assert (a.inprocess, a.inprocess_direct) == (False, False)
        assert a.inprocess_func == "LLVMFuzzerTestOneInput"

    @pytest.mark.parametrize(
        ("flags", "want"),
        [((), ExecMode.LOADER), *((f, m) for m, f in IN_PROCESS_MODES.items())],
    )
    def test_so_mode_table(self, flags, want):
        assert _replay_exec_mode(self._args("t.so", *flags)) is want

    @pytest.mark.parametrize("flags", [(), *IN_PROCESS_MODES.values()])
    def test_executable_stays_spawn(self, flags):
        # Adversarial: an ET_EXEC cannot be dlopen'd, whatever the flags say.
        exe = shutil.which("true")
        assert exe is not None
        assert _replay_exec_mode(self._args(exe, *flags)) is ExecMode.SPAWN


@pytest.mark.parametrize("mode", list(IN_PROCESS_MODES), ids=lambda m: m.value)
class TestReplay:
    def _replay(self, monkeypatch, tmp_path, so, data, mode, *extra):
        flags = (*IN_PROCESS_MODES[mode], "--inprocess-func", "fuzz_test", *extra)
        return _main(monkeypatch, so, _input(tmp_path, data), *flags)

    def test_crash_reproduced(self, monkeypatch, tmp_path, build_so, mode):
        assert self._replay(monkeypatch, tmp_path, build_so(), CRASH, mode) == REPRODUCED

    def test_benign_not_reproduced(self, monkeypatch, tmp_path, build_so, mode):
        # Falsification: a clean input must not read as a crash.
        assert self._replay(monkeypatch, tmp_path, build_so(), BENIGN, mode) == NOT_REPRODUCED

    def test_control_two_runs_agree(self, monkeypatch, tmp_path, build_so, mode):
        """The oracle against itself: the same input gives the same verdict."""
        runs = [self._replay(monkeypatch, tmp_path, build_so(), CRASH, mode) for _ in "ab"]
        assert runs[0] == runs[1]

    def test_missing_function_fails_cleanly(self, monkeypatch, tmp_path, build_so, mode, capsys):
        # Adversarial: a wrong --inprocess-func is an error, never "reproduced".
        so = build_so()
        rc = _main(
            monkeypatch,
            so,
            _input(tmp_path, CRASH),
            *IN_PROCESS_MODES[mode],
            "--inprocess-func",
            "no_such_function",
        )
        # Direct modes raise at load; the loader falls back to per-call and exits 1.
        out = capsys.readouterr()
        assert rc == NOT_REPRODUCED
        assert "Crash reproduced" not in out.out
