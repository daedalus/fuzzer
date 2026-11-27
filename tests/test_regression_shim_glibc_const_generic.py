"""Regression: the shim failed to compile on glibc's C23 const-generic macros.

Newer glibc (string.h, ``__glibc_const_generic``) defines ``memchr``,
``strstr`` and ``strpbrk`` as function-like macros under ``_GNU_SOURCE``. The
shim defines those functions, so the definitions macro-expanded into
``_Generic(...)`` and every shim build failed on the CI runner (127 tests).
The local glibc has no such macros, so ``-D`` stands in for them here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# Overrides the shim defines, and C23 const-generic names it does not.
DEFINED = ("memchr", "strstr", "strpbrk")
UNDEFINED = ("strchr", "strrchr")
ARGS = {"memchr": "S,C,N", "strstr": "H,N", "strpbrk": "S,A", "strchr": "S,C", "strrchr": "S,C"}


def _glibc_header(tmp_path: Path, names) -> Path:
    """string.h, then non-identity macros: glibc defines them after the prototypes."""
    lines = ["#define _GNU_SOURCE 1", "#include <string.h>"]
    for n in names:
        args = ARGS[n]
        lines.append(f"#define {n}({args}) ((void *){n}({args}))")
    hdr = tmp_path / "glibc_c23.h"
    hdr.write_text("\n".join(lines) + "\n")
    return hdr


def _compile(tmp_path: Path, names) -> subprocess.CompletedProcess:
    src = tmp_path / "t.c"
    src.write_text("int main(void) { return 0; }\n")
    cmd = [
        "clang", "-O1", "-D__AFL_CTX_SENSITIVE=0", "-D__AFL_CMPLOG=1",
        "-fsanitize-coverage=trace-pc-guard,trace-cmp",
        "-include", str(_glibc_header(tmp_path, names)),
        "-include", str(SHIM), "-c", str(src), "-o", str(tmp_path / "t.o"),
    ]  # fmt: skip
    return subprocess.run(cmd, capture_output=True, text=True)


@requires_clang
def test_regression_shim_glibc_const_generic(tmp_path):
    """The shim builds when its overridden names are function-like macros."""
    r = _compile(tmp_path, DEFINED)
    assert r.returncode == 0, r.stderr[-2000:]


@requires_clang
def test_overrides_still_exported(tmp_path):
    """Falsification: the fix keeps the interposers, not just the build."""
    assert _compile(tmp_path, DEFINED).returncode == 0
    syms = subprocess.run(["nm", str(tmp_path / "t.o")], capture_output=True, text=True).stdout
    defined = {line.split()[-1] for line in syms.splitlines() if " T " in line}
    assert set(DEFINED) <= defined


@requires_clang
@pytest.mark.parametrize("names", [UNDEFINED, DEFINED + UNDEFINED])
def test_unrelated_macros_untouched(tmp_path, names):
    """Adversarial: macros for names the shim only calls still compile."""
    r = _compile(tmp_path, names)
    assert r.returncode == 0, r.stderr[-2000:]
