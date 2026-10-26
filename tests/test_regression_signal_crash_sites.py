"""Regression: every no-sanitizer SIGSEGV shared one ``signal:11`` bucket.

Without a sanitizer report (or a ptrace fault address) the signature was the
signal number alone, so distinct crash sites deduplicated into one. The shim
records the faulting PC and a backtrace (``$__AFL_CRASH_SYM_OUT``); their page
offsets (ASLR-stable) now name the site.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.adapters.filesystem import classify_crash
from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader
from fuzzer_tool.core.crash_symbols import CrashSymbol, CrashSymbolSink, SymFrame, crash_pcs
from fuzzer_tool.services import fuzzer as fuzzer_mod

SEGV = -int(signal.SIGSEGV)
_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"


def _sig(pcs) -> str:
    return classify_crash(b"x", SEGV, "", set(), {}, crash_pcs=pcs).signature


def test_regression_crash_sites_split_signal_signature() -> None:
    assert _sig((0x401111, 0x402222)) != _sig((0x401333, 0x402222))


def test_crash_site_signature_aslr_stable() -> None:
    # Falsification: the same site at another load base is one bug.
    base = 0x7F0000000000
    assert _sig((0x55D000401111, 0x55D000402222)) == _sig((base + 0x1111, base + 0x2222))


def test_no_pcs_keeps_signal_signature() -> None:
    # Edge: no shim record, no fault address: unchanged.
    assert _sig(None) == "signal:11"
    assert classify_crash(b"x", SEGV, "", set(), {}, fault_addr=0).signature == "signal:11@0x0"


def test_sanitizer_report_wins_over_pcs() -> None:
    # Edge: a report already names the bug.
    asan = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n    #0 0x4 in f\n"
    v = classify_crash(b"x", 1, asan, set(), {}, crash_pcs=(0x401111,))

    assert v.signature == "AddressSanitizer:heap-use-after-free@f"


def test_crash_pcs_from_shim_record() -> None:
    # Inlined frames repeat a PC; one entry per distinct return address.
    sym = CrashSymbol(signal=11, pc=0x401111)
    sym.frames = [SymFrame(pc=0x401111), SymFrame(pc=0x401111), SymFrame(pc=0x402222)]

    assert crash_pcs(sym) == (0x401111, 0x402222)
    assert crash_pcs(CrashSymbol(signal=11, pc=0x401111)) == (0x401111,)
    assert crash_pcs(None) is None


_PLAIN, _SAN = fuzzer_mod.TargetRuntime.PLAIN, fuzzer_mod.TargetRuntime.SANITIZER


@pytest.mark.parametrize(
    ("symbolize", "runtime", "want"),
    [(False, _PLAIN, True), (False, _SAN, False), (True, _SAN, True), (True, _PLAIN, True)],
)
def test_sym_sink_wanted(symbolize: bool, runtime, want: bool) -> None:
    # On by default only where no sanitizer runtime would symbolize per crash.
    assert fuzzer_mod._sym_sink_wanted(symbolize, runtime) is want


# ── End to end: shim-linked targets with two crash sites ───────────────

_SITES = r"""
#include <stddef.h>
#include <unistd.h>
__attribute__((noinline)) static void site_a(volatile int *p) { *p = 1; }
__attribute__((noinline)) static void site_b(volatile int *p) { p[1] = 2; }
static void dispatch(char c) {
    if (c == 'A') site_a(0);
    if (c == 'B') site_b(0);
}
"""
_SO_SRC = (
    _SITES
    + r"""
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {
    if (n) dispatch((char)d[0]);
    return 0;
}
"""
)
_EXE_SRC = (
    _SITES
    + r"""
int main(void) {
    char c = 0;
    if (read(0, &c, 1) == 1) dispatch(c);
    return 0;
}
"""
)


def _build(tmp_path, src_text: str, out_name: str, *extra: str) -> str:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / f"{out_name}.c"
    src.write_text(src_text)
    out = tmp_path / out_name
    cmd = ["clang", "-O1", "-fsanitize-coverage=trace-pc-guard", "-include", str(SHIM)]
    r = subprocess.run([*cmd, *extra, "-o", str(out), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(out)


@pytest.fixture
def so(tmp_path) -> str:
    return _build(tmp_path, _SO_SRC, "t.so", "-shared", "-fPIC")


@pytest.fixture
def loader(tmp_path, so, monkeypatch):
    sink = CrashSymbolSink(str(tmp_path / "sym.log"))
    monkeypatch.setenv("__AFL_CRASH_SYM_OUT", sink.path)  # the loader inherits it
    ldr = PersistentLoader(so, timeout=5.0)
    if not ldr.start():
        ldr.stop()
        pytest.skip("persistent loader failed to start")
    yield ldr, sink
    ldr.stop()


def _sig_of(ldr, sink, data: bytes) -> str:
    rc, _ = ldr.run_one(data)
    return classify_crash(data, rc, "", set(), {}, crash_pcs=crash_pcs(sink.drain())).signature


def test_regression_so_two_sites_two_signatures(loader) -> None:
    ldr, sink = loader
    a1, a2, b = _sig_of(ldr, sink, b"A"), _sig_of(ldr, sink, b"A"), _sig_of(ldr, sink, b"B")

    assert a1.startswith("signal:11@")
    assert a1 == a2
    assert a1 != b


def test_plain_so_detected_plain(so) -> None:
    # The shim's own __sanitizer_cov_* and weak __asan_* are not a runtime.
    assert fuzzer_mod._target_runtime(so) is _PLAIN


def test_sancov_exe_has_runtime_report(tmp_path) -> None:
    # Edge: clang links the UBSAN runtime into sancov executables; its SEGV
    # report already names the site, and the sink stays off for them.
    exe = _build(tmp_path, _EXE_SRC, "t")
    sigs = []
    for data in (b"A", b"B"):
        rc, stderr, _ = process.run_target_stdin(exe, data, 5.0, env=process.environ_copy())
        sigs.append(classify_crash(data, rc, stderr, set(), {}).signature)

    assert fuzzer_mod._target_runtime(exe) is _SAN
    assert sigs[0] != sigs[1]
    assert all(s.startswith("UndefinedBehaviorSanitizer:SEGV@") for s in sigs)
