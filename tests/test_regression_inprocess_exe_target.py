"""--inprocess/--inprocess-direct on an executable: run it in exec mode.

Both in-process loaders dlopen the target, which an executable cannot be.
--hail-mary forces in-process mode, so pointing it at ffmpeg_read_<ver>_asan
raised instead of fuzzing the executable the normal way.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.elf import is_elf_executable
from fuzzer_tool.services.fuzzer import Fuzzer

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang required")

SRC = "int fuzz_shm_run(const unsigned char *d, unsigned long n) { return 0; }\n"
MAIN = "int main(void) { return 0; }\n"


def _cc(tmp_path: Path, name: str, src: str, *flags: str) -> Path:
    c = tmp_path / f"{name}.c"
    c.write_text(src)
    out = tmp_path / name
    subprocess.run(["clang", *flags, "-o", str(out), str(c)], check=True)
    return out


def _fuzzer(tmp_path: Path, target: Path, direct: bool) -> Fuzzer:
    return Fuzzer(
        target=str(target),
        corpus_dir=str(tmp_path / "corpus"),
        crashes_dir=str(tmp_path / "crashes"),
        inprocess=True,
        inprocess_direct=direct,
        max_len=64,
        timeout=1,
        mutations_per_input=1,
        cmplog=False,
    )


@pytest.mark.parametrize("direct", [True, False])
def test_regression_pie_exe_uses_exec_mode(tmp_path, direct):
    """Falsification: PIE exe + in-process flags -> no in-process runner, target unchanged."""
    exe = _cc(tmp_path, "t", SRC + MAIN, "-fPIE", "-pie")

    f = _fuzzer(tmp_path, exe, direct)

    assert f.target == str(exe)
    assert f._inprocess_runner is None


def test_regression_so_keeps_inprocess(tmp_path):
    """Adversarial: a shared object (ET_DYN, no PT_INTERP) stays in direct mode."""
    so = _cc(tmp_path, "t.so", SRC, "-shared", "-fPIC")

    f = _fuzzer(tmp_path, so, direct=True)

    assert f._inprocess_runner is not None
    assert f._inprocess_runner.direct


def test_is_elf_executable_kinds(tmp_path):
    """PIE and non-PIE executables detected; .so and non-ELF are not."""
    pie = _cc(tmp_path, "pie", SRC + MAIN, "-fPIE", "-pie")
    nopie = _cc(tmp_path, "nopie", SRC + MAIN, "-fno-PIE", "-no-pie")
    so = _cc(tmp_path, "lib.so", SRC, "-shared", "-fPIC")
    text = tmp_path / "x.sh"
    text.write_text("#!/bin/sh\n")

    assert is_elf_executable(str(pie))
    assert is_elf_executable(str(nopie))
    assert not is_elf_executable(str(so))
    assert not is_elf_executable(str(text))
    assert not is_elf_executable(str(tmp_path / "missing"))


def test_is_elf_executable_truncated(tmp_path):
    """Adversarial: ELF header whose program headers lie past EOF -> not executable, no raise."""
    so = _cc(tmp_path, "lib.so", SRC, "-shared", "-fPIC")
    cut = tmp_path / "cut"
    cut.write_bytes(so.read_bytes()[:64])

    assert not is_elf_executable(str(cut))
