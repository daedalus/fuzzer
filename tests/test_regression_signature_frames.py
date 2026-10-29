"""Regression: crash signatures split one bug by harness and runtime frames.

Signatures took the first 6 crash-stack frames. On shallow stacks that ran
into the harness (``fuzz_shm_run``, ``__afl_guard_run``) and the Python /
libffi frames under it, whose offsets vary per environment; allocator
interceptors (``free``) took a slot. Now: runtime frames skipped, harness
boundary ends the stack, 3 frames kept. PC-only sites skip libc frames.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.filesystem import classify_crash
from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader
from fuzzer_tool.core.crash_symbols import CrashSymbol, CrashSymbolSink, SymFrame, crash_pcs
from fuzzer_tool.core.sanitizer import SanitizerReport

_ROOT = Path(__file__).resolve().parent.parent
SHIM = _ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
ABRT = -int(signal.SIGABRT)

_UAF = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
_BAD_FREE = "==1==ERROR: AddressSanitizer: attempting free on address which was not malloc()-ed\n"


def _report(header: str, frames: list[str]) -> str:
    body = "".join(f"    #{i} 0x{0x401000 + i:x} in {f}\n" for i, f in enumerate(frames))
    return header + body


def _sig(header: str, frames: list[str]) -> str:
    r = SanitizerReport.parse(_report(header, frames))
    assert r is not None
    return r.signature


def test_regression_harness_frames_not_in_signature() -> None:
    via_ctypes = [
        "json_parse_ex",
        "json_parse",
        "fuzz_shm_run",
        "__afl_guard_run",
        "libffi.so.8+0x7b15",
    ]
    via_loader = ["json_parse_ex", "json_parse", "LLVMFuzzerTestOneInput", "main"]

    assert _sig(_UAF, via_ctypes) == _sig(_UAF, via_loader)
    assert _sig(_UAF, via_ctypes) == "AddressSanitizer:heap-use-after-free@json_parse_ex@json_parse"


def test_regression_interceptor_skipped() -> None:
    sig = _sig(_BAD_FREE, ["free", "json_value_free_ex", "fuzz_shm_run", "__afl_guard_run"])

    assert sig == "AddressSanitizer:bad-free@json_value_free_ex"


def test_signature_keeps_three_frames() -> None:
    deep = ["a", "b", "c", "d", "e"]

    assert _sig(_UAF, deep) == _sig(_UAF, [*deep[:3], "x", "y"])
    assert _sig(_UAF, deep).endswith("@a@b@c")


def test_distinct_sites_still_split() -> None:
    # Falsification: another crash frame or caller is another bug.
    base = ["json_parse_ex", "json_parse", "fuzz_shm_run"]

    assert _sig(_UAF, base) != _sig(_UAF, ["new_value", *base[1:]])
    assert _sig(_UAF, base) != _sig(_UAF, ["json_parse_ex", "parse_trailer", "fuzz_shm_run"])


def test_crash_in_harness_frame_kept() -> None:
    # Edge: the bug is in the wrapper itself; it is the site.
    assert _sig(_UAF, ["fuzz_shm_run", "__afl_guard_run"]) == (
        "AddressSanitizer:heap-use-after-free@fuzz_shm_run"
    )


def test_all_runtime_stack_keeps_first_frame() -> None:
    # Edge: nothing but runtime frames: the innermost still names something.
    assert _sig(_BAD_FREE, ["free", "__libc_start_main"]) == "AddressSanitizer:bad-free@free"


def test_lookalike_names_not_skipped() -> None:
    # Adversarial: names that only resemble runtime frames are target code.
    frames = ["free_list", "libcrypto.so.3+0x11d6", "malloc_wrapper"]

    assert _sig(_UAF, frames).endswith("@free_list@libcrypto.so.3+0x11d6@malloc_wrapper")


def test_unsymbolized_libc_frame_skipped() -> None:
    frames = ["libc.so.6+0x9ec0c", "target.so+0x11d6", "target.so+0x1250"]

    assert _sig(_UAF, frames) == (
        "AddressSanitizer:heap-use-after-free@target.so+0x11d6@target.so+0x1250"
    )


# ── PC-only sites (shim record) ─────────────────────────────────────────


def _sym(*frames: tuple[int, str]) -> CrashSymbol:
    sym = CrashSymbol(signal=6, pc=frames[0][0])
    sym.frames = [SymFrame(pc=pc, module=mod) for pc, mod in frames]
    return sym


def test_regression_libc_pcs_skipped() -> None:
    libc = "/usr/lib/x86_64-linux-gnu/libc.so.6"
    sym = _sym((0x7F0000001C0C, libc), (0x7F000000227E, libc), (0x55D000401111, "/t/t.so"))

    assert crash_pcs(sym) == (0x55D000401111,)


def test_pcs_without_modules_unchanged() -> None:
    # Falsification: records without module info (old shim) keep every PC.
    sym = _sym((0x401111, ""), (0x402222, ""))

    assert crash_pcs(sym) == (0x401111, 0x402222)


def test_all_libc_pcs_kept() -> None:
    # Edge: a crash wholly inside libc keeps its PCs rather than none.
    libc = "/lib/libc.so.6"
    sym = _sym((0x7F0000001C0C, libc), (0x7F000000227E, libc))

    assert crash_pcs(sym) == (0x7F0000001C0C, 0x7F000000227E)


def test_pc_site_signature_three_frames() -> None:
    pcs = (0x401111, 0x402222, 0x403333, 0x404444)

    sig = classify_crash(b"x", ABRT, "", set(), {}, crash_pcs=pcs).signature
    assert sig == "signal:6@0x111@0x222@0x333"


# ── End to end: glibc aborts from two call sites ────────────────────────

_SO_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
__attribute__((noinline)) static void site_a(char *p) { free(p); free(p); }
__attribute__((noinline)) static void site_b(char *p) { free(p + 16); }
int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n) {
    char *p = malloc(64);
    if (n && d[0] == 'A') site_a(p);
    if (n && d[0] == 'B') site_b(p);
    return 0;
}
"""


@pytest.fixture
def loader(tmp_path, monkeypatch):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    src = tmp_path / "t.c"
    src.write_text(_SO_SRC)
    so = tmp_path / "t.so"
    cmd = ["clang", "-O1", "-fsanitize-coverage=trace-pc-guard", "-include", str(SHIM)]
    r = subprocess.run(
        [*cmd, "-shared", "-fPIC", "-o", str(so), str(src)], capture_output=True, text=True
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")

    sink = CrashSymbolSink(str(tmp_path / "sym.log"))
    monkeypatch.setenv("__AFL_CRASH_SYM_OUT", sink.path)
    monkeypatch.setenv("__AFL_CRASH_SYM_RAW", "1")
    ldr = PersistentLoader(str(so), timeout=5.0)
    if not ldr.start():
        ldr.stop()
        pytest.skip("persistent loader failed to start")
    yield ldr, sink
    ldr.stop()


def test_regression_glibc_abort_sites_split(loader) -> None:
    ldr, sink = loader
    recs = []
    for data in (b"A", b"B"):
        rc, _ = ldr.run_one(data)
        recs.append((rc, sink.drain()))

    (rc_a, sym_a), (rc_b, sym_b) = recs
    assert sym_a is not None and sym_b is not None
    assert any("libc" in f.module for f in sym_a.frames)  # the shim names modules
    assert not sym_a.symbolized

    pcs_a, pcs_b = crash_pcs(sym_a), crash_pcs(sym_b)
    assert pcs_a and pcs_b
    libc_pcs = {f.pc for f in sym_a.frames + sym_b.frames if "libc.so" in f.module}
    assert not libc_pcs & set(pcs_a + pcs_b)

    sig_a = classify_crash(b"A", rc_a, "", set(), {}, crash_pcs=pcs_a).signature
    sig_b = classify_crash(b"B", rc_b, "", set(), {}, crash_pcs=pcs_b).signature
    assert sig_a != sig_b


def test_shim_module_offset(loader) -> None:
    # The record's module offset is the PC minus the object's load base.
    ldr, sink = loader
    ldr.run_one(b"A")
    sym = sink.drain()

    assert sym is not None
    target = [f for f in sym.frames if f.module.endswith("t.so")]
    assert target and all(0 < f.offset < f.pc for f in target)
    assert len({f.pc - f.offset for f in target}) == 1


# ── Fuzzy match on 3-frame signatures ───────────────────────────────────


def _known(header: str, frames: list[str]) -> dict[str, int]:
    return {_sig(header, frames): 1}


def test_inlined_frame_folds() -> None:
    known = _known(_UAF, ["parse_chunk", "read_block", "decode"])
    v = classify_crash(
        b"i", 1, _report(_UAF, ["parse_chunk", "parse_chunk_inl", "read_block"]), set(), known
    )

    assert not v.novel


def test_other_third_frame_not_folded() -> None:
    # Adversarial: dropping the last frame is truncation, not inlining.
    a = ["p", "q", "decode_image_scanline_with_filter"]
    b = ["p", "q", "parse_container_metadata_block"]
    v = classify_crash(b"o", 1, _report(_UAF, b), set(), _known(_UAF, a))

    assert v.novel


def test_prefix_stack_not_folded() -> None:
    # Adversarial: a shallower stack (called from the harness) is another caller.
    a = ["parse_container_metadata", "read_block_header"]
    v = classify_crash(
        b"p", 1, _report(_UAF, a), set(), _known(_UAF, [*a, "decode_scanline_filter"])
    )

    assert v.novel
