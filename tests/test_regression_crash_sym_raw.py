"""Regression: sanitizer .so SEGVs caught by the shim shared one ``signal:11``.

In-process / persistent, the shim's guard catches the fault before the
sanitizer reports, so there is no report. The crash record that names the
site was off for sanitizer targets because symbolizing every crash is slow;
``__AFL_CRASH_SYM_RAW=1`` records PCs only, so the record is now always on.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.filesystem import classify_crash
from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader
from fuzzer_tool.core.crash_symbols import CrashSymbolSink, crash_pcs
from fuzzer_tool.services import fuzzer as fuzzer_mod

_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
RAW_ENV = "__AFL_CRASH_SYM_RAW"
SINK_ENV = "__AFL_CRASH_SYM_OUT"
_RT_DIR = Path("/usr/lib/llvm-18/lib/clang/18/lib/linux")
_RT = "libclang_rt.ubsan_standalone-x86_64.so"

_SO_SRC = r"""
#include <stddef.h>
__attribute__((noinline)) static void site_a(volatile int *p) { *p = 1; }
__attribute__((noinline)) static void site_b(volatile int *p) { p[1] = 2; }
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {
    if (n && d[0] == 'A') site_a(0);
    if (n && d[0] == 'B') site_b(0);
    return 0;
}
"""


@pytest.fixture
def ubsan_so(tmp_path, monkeypatch) -> str:
    """A UBSAN .so: a sanitizer runtime with a symbolizer, loadable by dlopen."""
    if not shutil.which("clang") or not (_RT_DIR / _RT).exists():
        pytest.skip("clang / shared UBSAN runtime not installed")
    src = tmp_path / "t.c"
    src.write_text(_SO_SRC)
    out = tmp_path / "t.so"
    cmd = [
        "clang", "-O1", "-g", "-fsanitize=undefined", "-shared-libsan",
        "-fsanitize-coverage=trace-pc-guard", "-include", str(SHIM),
        "-shared", "-fPIC", "-o", str(out), str(src),
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(_RT_DIR))
    return str(out)


def _records(tmp_path, so: str, monkeypatch, raw: bool, inputs: list[bytes]):
    sink = CrashSymbolSink(str(tmp_path / "sym.log"))
    monkeypatch.setenv(SINK_ENV, sink.path)
    if raw:
        monkeypatch.setenv(RAW_ENV, "1")
    else:
        monkeypatch.delenv(RAW_ENV, raising=False)

    ldr = PersistentLoader(so, timeout=5.0)
    out = []
    try:
        assert ldr.start()
        for data in inputs:
            rc, _ = ldr.run_one(data)
            out.append((rc, sink.drain()))
    finally:
        ldr.stop()
    return out


def test_regression_raw_record_not_symbolized(tmp_path, ubsan_so, monkeypatch) -> None:
    ((_, sym),) = _records(tmp_path, ubsan_so, monkeypatch, True, [b"A"])

    assert sym is not None and sym.pc
    assert not sym.symbolized


def test_symbolized_without_raw(tmp_path, ubsan_so, monkeypatch) -> None:
    # Falsification: --crash-symbolize keeps resolving names.
    ((_, sym),) = _records(tmp_path, ubsan_so, monkeypatch, False, [b"A"])

    assert sym is not None and sym.symbolized


def test_regression_sanitizer_so_sites_split(tmp_path, ubsan_so, monkeypatch) -> None:
    # No report reaches stderr: the PCs alone tell the two sites apart.
    recs = _records(tmp_path, ubsan_so, monkeypatch, True, [b"A", b"A", b"B"])
    sigs = [
        classify_crash(b"x", rc, "", set(), {}, crash_pcs=crash_pcs(s)).signature for rc, s in recs
    ]

    assert sigs[0].startswith("signal:11@")
    assert sigs[0] == sigs[1] != sigs[2]


@pytest.mark.parametrize(("symbolize", "raw"), [(False, "1"), (True, None)])
def test_fuzzer_exports_sym_mode(monkeypatch, symbolize: bool, raw: str | None) -> None:
    monkeypatch.setenv(RAW_ENV, "x")  # recorded, so the test restores the environment
    monkeypatch.delenv(RAW_ENV)
    fuzzer_mod._export_sym_mode(
        fuzzer_mod.SymMode.SYMBOLIZE if symbolize else fuzzer_mod.SymMode.RAW
    )

    assert fuzzer_mod.os.environ.get(RAW_ENV) == raw
