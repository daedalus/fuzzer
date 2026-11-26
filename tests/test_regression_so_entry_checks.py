"""In-process `.so` runs check the target's exports before loading it.

- A missing entry point is a clear error in every mode. The loader used to
  fall back to a per-call child that died, read as "not reproduced".
- Direct modes contain crashes and hangs only through the shim's
  `__afl_guarded_call`. Without it a crash killed fuzzer-tool, so a guard-less
  `.so` now falls back to the loader with a warning.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.elf import exported_names

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
GUARD = "__afl_guarded_call"

# 'x' aborts. `abort` is an undefined import, never an export.
TARGET_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    if (len && buf[0] == 'x') abort();
    return 0;
}
"""
# No fuzz_* entry at all: the probe falls back to a name that is not there.
NO_ENTRY_SRC = r"""
__attribute__((visibility("default"))) int other_fn(void) { return 0; }
"""

CRASH = b"x"
REPRODUCED = 0
FAILED = 1


@pytest.fixture(autouse=True)
def _restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _build(tmp_path: Path, name: str, src: str, shim: bool) -> str:
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    d = tmp_path / name
    d.mkdir()
    c = d / "t.c"
    c.write_text(src)
    so = d / "t.so"
    cmd = ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(c)]
    if shim:
        cmd.append(str(SHIM))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"build failed: {proc.stderr[-400:]}")
    return str(so)


def _input(tmp_path: Path) -> str:
    p = tmp_path / "crash.bin"
    p.write_bytes(CRASH)
    return str(p)


class TestExportedNames:
    def test_defined_exports(self, tmp_path):
        names = exported_names(_build(tmp_path, "s", TARGET_SRC, shim=True))
        assert {"fuzz_test", GUARD} <= names

    def test_imports_excluded(self, tmp_path):
        # Falsification: an undefined import (abort) is not an export.
        names = exported_names(_build(tmp_path, "b", TARGET_SRC, shim=False))
        assert "fuzz_test" in names
        assert "abort" not in names
        assert GUARD not in names

    @pytest.mark.parametrize("blob", [b"", b"not an elf", b"\x7fELF\x02\x01\x01" + b"\0" * 9])
    def test_garbage_is_empty(self, tmp_path, blob):
        # Adversarial: non-ELF and truncated headers give no names, no exception.
        p = tmp_path / "junk.so"
        p.write_bytes(blob)
        assert exported_names(str(p)) == frozenset()

    def test_missing_file_is_empty(self, tmp_path):
        assert exported_names(str(tmp_path / "absent.so")) == frozenset()


def _main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", *argv])
    return commands.main()


@pytest.mark.parametrize("cmd", ["replay", "tmin"])
@pytest.mark.parametrize(
    "flags",
    [("--inprocess",), ("--inprocess-direct",), ("--inprocess", "--inprocess-direct")],
    ids=["loader", "direct_lite", "direct"],
)
def test_regression_missing_entry(monkeypatch, tmp_path, capsys, cmd, flags):
    so = _build(tmp_path, "s", TARGET_SRC, shim=True)
    argv = (cmd, so, _input(tmp_path), "--inprocess-func", "no_such_function", *flags)
    assert _main(monkeypatch, *argv) == FAILED
    err = capsys.readouterr().err
    assert "no_such_function" in err
    assert "not exported" in err


@pytest.mark.parametrize("cmd", ["replay", "probe-size"])
def test_regression_probe_finds_none(monkeypatch, tmp_path, capsys, cmd):
    # Adversarial: no fuzz_* export; the probe's fallback must fail loudly.
    so = _build(tmp_path, "n", NO_ENTRY_SRC, shim=False)
    assert _main(monkeypatch, cmd, so, _input(tmp_path)) == FAILED
    assert "not exported" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags",
    [("--inprocess-direct",), ("--inprocess", "--inprocess-direct")],
    ids=["lite", "direct"],
)
def test_regression_unguarded_direct(tmp_path, flags):
    # A fresh process: without the fallback, the abort kills the host.
    so = _build(tmp_path, "b", TARGET_SRC, shim=False)
    argv = [sys.executable, "-m", "fuzzer_tool", "replay", so, _input(tmp_path)]
    proc = subprocess.run(
        [*argv, "--inprocess-func", "fuzz_test", *flags],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == REPRODUCED, proc.stdout + proc.stderr
    assert "Crash reproduced" in proc.stdout
    assert GUARD in proc.stderr  # the fallback warning names the missing guard


def test_guarded_direct_kept(monkeypatch, tmp_path, capsys):
    # Control: a shim-linked .so keeps its direct mode (no fallback warning).
    so = _build(tmp_path, "s", TARGET_SRC, shim=True)
    argv = ("replay", so, _input(tmp_path), "--inprocess-func", "fuzz_test")
    assert _main(monkeypatch, *argv, "--inprocess-direct") == REPRODUCED
    assert GUARD not in capsys.readouterr().err
