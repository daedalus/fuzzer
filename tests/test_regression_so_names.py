"""Versioned shared objects (`libx.so.1`) load in-process; a `.so` needs only read access.

`.so` detection was a suffix check, so `libx.so.1` was spawned as a program
(and died). `_validate_target` required the execute bit, which dlopen never
needs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.elf import is_so_path

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
CRASH_INPUT = b"aaxbb"
MINIMUM = b"x"
OK = 0
READ_ONLY = 0o644


@pytest.fixture(autouse=True)
def _restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize(
    "path",
    ["a.so", "libx.so.1", "libx.so.1.2.3", "/d/X.SO", "a.dylib", "a.dll", "LIB.SO.2"],
)
def test_so_names(path):
    assert is_so_path(path)
    assert commands._is_shared_object(path)


@pytest.mark.parametrize(
    "path",
    ["a.so.x", "a.sox", "a.solib", "a.py", "pkg.mod:fn", "so", "a.so.", "libso.1", "a.so.1x"],
)
def test_not_so_names(path):
    # Falsification / adversarial: near-misses are not shared objects.
    assert not is_so_path(path)


def _build(tmp_path: Path, name: str, mode: int) -> str:
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    src = tmp_path / "t.c"
    src.write_text(TARGET_SRC)
    so = tmp_path / name
    cmd = ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"build failed: {proc.stderr[-400:]}")
    so.chmod(mode)
    return str(so)


def _tmin(monkeypatch, tmp_path, so, *flags) -> tuple[int, Path]:
    inp = tmp_path / "crash.bin"
    inp.write_bytes(CRASH_INPUT)
    out = tmp_path / "min.bin"
    argv = ["fuzzer-tool", "tmin", so, str(inp), "-O", str(out), "--inprocess-func", "fuzz_test"]
    monkeypatch.setattr(sys, "argv", [*argv, *flags])
    return commands.main(), out


@pytest.mark.parametrize("flags", [(), ("--inprocess-direct",)], ids=["loader", "direct_lite"])
def test_regression_versioned_so(monkeypatch, tmp_path, flags):
    so = _build(tmp_path, "libx.so.1", 0o755)
    rc, out = _tmin(monkeypatch, tmp_path, so, *flags)
    assert rc == OK
    assert out.read_bytes() == MINIMUM


def test_regression_readable_so(monkeypatch, tmp_path):
    so = _build(tmp_path, "t.so", READ_ONLY)
    assert not os.access(so, os.X_OK)
    rc, out = _tmin(monkeypatch, tmp_path, so)
    assert rc == OK
    assert out.read_bytes() == MINIMUM


def test_unexecutable_program_rejected(monkeypatch, tmp_path):
    # Control: a non-.so still needs the execute bit.
    script = tmp_path / "t.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(READ_ONLY)
    with pytest.raises(SystemExit):
        _tmin(monkeypatch, tmp_path, str(script))
