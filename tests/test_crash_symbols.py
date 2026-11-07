"""Shim-side crash symbolization: parser, sink, hydration, and the real shim."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core import crash_symbols as cs
from fuzzer_tool.core.crash_metadata import CrashMetadata

SHIM = Path(cs.__file__).resolve().parents[1] / "adapters" / "afl_shim.c"


def test_parse_symbolized_line():
    sym = cs.parse_line("SYM 11 0x55f3183229ff 0x20 0x55f3183229ff:boom|/src/h.c|5|/bin/h|0x49fe")
    assert sym and sym.symbolized and sym.signal == 11
    assert sym.pc == 0x55F3183229FF and sym.fault_addr == 0x20
    fr = sym.frames[0]
    assert (fr.function, fr.file, fr.line, fr.offset) == ("boom", "/src/h.c", 5, 0x49FE)
    assert sym.frame_strings()[0] == "0x55f3183229ff in boom /src/h.c:5 (h+0x49fe)"


def test_parse_raw_pc_when_no_symbolizer():
    sym = cs.parse_line("SYM 6 0x1000 0x0 0x1000:-@0x2000:-")
    assert sym and not sym.symbolized
    assert sym.frame_strings() == ["0x1000", "0x2000"]


def test_parse_backtrace_inlined_and_libc_null_file():
    sym = cs.parse_line(
        "SYM 6 0x7f00 0x4d5 0x7f00:pthread_kill|<null>|0|/lib/libc.so.6|0x9ec0b"
        "@0x4011:inner|a.c|3|m|0x1;outer|a.c|9|m|0x2@0x4022:main|a.c|20|m|0x9"
    )
    assert [f.function for f in sym.frames] == ["pthread_kill", "inner", "outer", "main"]
    assert sym.frames[0].file == "" and sym.frames[1].pc == 0x4011
    assert sym.frames[0].render().startswith("0x7f00 in pthread_kill (libc.so.6+0x9ec0b)")


@pytest.mark.parametrize(
    "bad", ["", "garbage", "SYM x y z", "SYM 11 zz 0x0 -", "SYM 11", "SYM 11 0x1 0x0 nocolon"]
)
def test_parse_rejects_malformed(bad):
    assert cs.parse_line(bad) is None


def test_sink_drain_takes_newest_and_truncates():
    sink = cs.CrashSymbolSink()
    try:
        assert sink.drain() is None
        Path(sink.path).write_text(
            "SYM 11 0x1 0x0 0x1:-\nnoise\nSYM 6 0x2 0x0 0x2:f|a.c|1|m|0x0\n", encoding="utf-8"
        )
        sym = sink.drain()
        assert sym and sym.pc == 2
        assert Path(sink.path).read_text() == ""
        assert sink.drain() is None  # truncated: the next crash can't see this one
    finally:
        sink.close()
    assert not os.path.exists(sink.path)


def test_hydrate_fills_gaps_only():
    sym = cs.parse_line("SYM 11 0xabc 0x20 0xabc:boom|a.c|5|m|0x4")
    meta = CrashMetadata()
    assert cs.hydrate(meta, sym)
    assert meta.frames and meta.rip == 0xABC and meta.fault_addr == "0x20"
    assert meta.shim_symbol["frames"][0]["function"] == "boom"
    txt = meta.format_sidecar()
    assert "shim symbolization" in txt and "#0 0xabc boom a.c:5" in txt
    assert meta.to_dict()["shim_symbol"]["pc"] == "0xabc"

    # Another source already set these: hydration must not overwrite them.
    meta2 = CrashMetadata(frames=["asan frame"], rip=0x1, fault_addr="0x99")
    cs.hydrate(meta2, sym)
    assert (meta2.frames, meta2.rip, meta2.fault_addr) == (["asan frame"], 0x1, "0x99")
    assert cs.hydrate(CrashMetadata(), None) is False


def test_sink_enable_sets_env_and_restores(monkeypatch):
    monkeypatch.delenv(cs.ENV_VAR, raising=False)
    sink = cs.CrashSymbolSink()
    try:
        sink.enable()
        assert os.environ[cs.ENV_VAR] == sink.path
    finally:
        sink.close()
        os.environ.pop(cs.ENV_VAR, None)


_HARNESS = r"""
#include <stdint.h>
#include <stddef.h>
int __afl_guarded_call(int (*)(const uint8_t*,size_t), const uint8_t*, size_t);
__attribute__((noinline)) int boom(const uint8_t *d, size_t n){ volatile int *p=(int*)0x10; return *p+(int)n; }
int main(void){ uint8_t b[1]={0}; return __afl_guarded_call(boom,b,1) == -11 ? 0 : 3; }
"""


def _build(tmp_path, *flags, src=_HARNESS):
    cc = shutil.which("clang") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler")
    (tmp_path / "h.c").write_text(src)
    exe = tmp_path / "h"
    r = subprocess.run(
        [
            cc,
            "-g",
            "-O0",
            *flags,
            "-o",
            str(exe),
            str(tmp_path / "h.c"),
            str(SHIM),
            "-ldl",
            "-lpthread",
            "-lrt",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"shim did not build here: {r.stderr[-300:]}")
    return exe


def _asan_runtime(exe) -> str | None:
    """Shared ASAN runtime *exe* links, or None when it is static (clang).

    A shared runtime must come first in the initial library list or ASAN
    aborts at startup; an inherited LD_PRELOAD pushes it back.
    """
    ldd = shutil.which("ldd")
    if not ldd:
        return None
    out = subprocess.run([ldd, str(exe)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        # "libasan.so.8 => /usr/lib/x86_64-linux-gnu/libasan.so.8 (0x...)"
        name, _, rest = line.partition("=>")
        if "asan" in name and rest.split():
            return rest.split()[0]
    return None


def _run(exe, out, **env):
    e = {**os.environ, **env}
    e.pop(cs.ENV_VAR, None)
    if out is not None:
        e[cs.ENV_VAR] = str(out)

    # Only the sanitizer runs ask for ASAN; an inherited ASAN_OPTIONS must
    # not preload it into the no-sanitizer harness.
    libasan = _asan_runtime(exe) if "ASAN_OPTIONS" in env else None
    if libasan:
        existing = e.get("LD_PRELOAD", "")
        e["LD_PRELOAD"] = libasan + ((":" + existing) if existing else "")
    return subprocess.run([str(exe)], env=e, capture_output=True, text=True)


def test_shim_raw_pc_without_sanitizer_runtime(tmp_path):
    exe = _build(tmp_path)
    out = tmp_path / "sym.log"
    assert _run(exe, out).returncode == 0
    sym = cs.parse_line(out.read_text().splitlines()[-1])
    assert sym and sym.signal == 11 and sym.fault_addr == 0x10 and sym.pc and not sym.symbolized


def test_shim_is_inert_without_env(tmp_path):
    exe = _build(tmp_path)
    out = tmp_path / "sym.log"
    assert _run(exe, None).returncode == 0
    assert not out.exists()


def test_shim_symbolizes_with_asan_runtime(tmp_path):
    exe = _build(tmp_path, "-fsanitize=address", "-fno-omit-frame-pointer")
    out = tmp_path / "sym.log"
    assert _run(exe, out, ASAN_OPTIONS="handle_segv=0:symbolize=1").returncode == 0
    sym = cs.parse_line(out.read_text().splitlines()[-1])
    assert sym and sym.symbolized
    assert sym.frames[0].function == "boom" and sym.frames[0].file.endswith("h.c")
    assert sym.frames[0].line > 0


# Fault inside an always_inline callee: one PC, two source frames.
_INLINED_HARNESS = r"""
#include <stdint.h>
#include <stddef.h>
int __afl_guarded_call(int (*)(const uint8_t*,size_t), const uint8_t*, size_t);
static inline __attribute__((always_inline)) int leaf(size_t n){ volatile int *p=(int*)0x10; return *p+(int)n; }
__attribute__((noinline)) int boom(const uint8_t *d, size_t n){ return leaf(n); }
int main(void){ uint8_t b[1]={0}; return __afl_guarded_call(boom,b,1) == -11 ? 0 : 3; }
"""


def test_regression_shim_keeps_inlined_frames(tmp_path):
    """__sanitizer_symbolize_pc NUL-separates inlined frames; the shim
    kept only the first, so fuzzgoat's default_free hid json_value_free_ex."""
    exe = _build(tmp_path, "-fsanitize=address", "-fno-omit-frame-pointer", src=_INLINED_HARNESS)
    out = tmp_path / "sym.log"
    assert _run(exe, out, ASAN_OPTIONS="handle_segv=0:symbolize=1").returncode == 0
    sym = cs.parse_line(out.read_text().splitlines()[-1])
    assert sym and sym.symbolized

    at_pc = [f.function for f in sym.frames if f.pc == sym.frames[0].pc]
    assert at_pc[:2] == ["leaf", "boom"]


def test_shim_no_phantom_inlined_frames(tmp_path):
    """Adversarial: the zeroed tail must not turn into extra frames."""
    exe = _build(tmp_path, "-fsanitize=address", "-fno-omit-frame-pointer")
    out = tmp_path / "sym.log"
    assert _run(exe, out, ASAN_OPTIONS="handle_segv=0:symbolize=1").returncode == 0
    sym = cs.parse_line(out.read_text().splitlines()[-1])
    assert sym and [f.function for f in sym.frames if f.pc == sym.frames[0].pc] == ["boom"]


def _harness_line(needle: str, src: str = _HARNESS) -> int:
    """1-based line of *needle* in the harness source as written to h.c."""
    return next(i for i, ln in enumerate(src.splitlines(), 1) if needle in ln)


@pytest.mark.parametrize("pie", ["-pie", "-no-pie"])
def test_regression_raw_frames_resolved_from_dwarf(tmp_path, pie):
    """No sanitizer runtime: the shim writes module+offset only. hydrate names
    the frame from the module's own DWARF and symtab (fuzzgoat_read_nosan
    sidecars stayed raw though llvm-symbolizer could name every frame)."""
    exe = _build(tmp_path, pie, "-fPIE" if pie == "-pie" else "-fno-pie")
    out = tmp_path / "sym.log"
    assert _run(exe, out).returncode == 0
    sym = cs.parse_line(out.read_text().splitlines()[-1])
    assert sym and not sym.symbolized

    meta = CrashMetadata()
    assert cs.hydrate(meta, sym)
    top = sym.frames[0]
    assert (top.function, os.path.basename(top.file)) == ("boom", "h.c")
    assert top.line == _harness_line("volatile int *p")
    assert "in boom" in meta.frames[0]
    assert any(f.function == "main" for f in sym.frames[1:])


def test_raw_frames_unresolvable_module_left_raw(tmp_path):
    """Adversarial: a stripped / missing module leaves the record untouched."""
    sym = cs.CrashSymbol(
        signal=11,
        pc=0x1234,
        frames=[
            cs.SymFrame(pc=0x1234, module=str(tmp_path / "gone.so"), offset=0x234),
            cs.SymFrame(pc=0x5678, module="/bin/true", offset=0x10),
        ],
    )
    before = [f.render() for f in sym.frames]
    assert cs.hydrate(CrashMetadata(), sym)
    assert [f.render() for f in sym.frames] == before and not sym.symbolized


def _shim_title(meta: CrashMetadata) -> str:
    return next(ln for ln in meta.format_sidecar().splitlines() if ln.startswith("=== shim"))


def test_regression_shim_title_names_dwarf_resolver():
    """Names from the module's DWARF are not 'sanitizer runtime'."""
    meta = CrashMetadata()
    sym = cs.CrashSymbol(signal=11, pc=0x10, frames=[cs.SymFrame(pc=0x10, function="boom")])
    sym.symbolized, sym.resolver = True, cs.Resolver.DWARF
    cs.hydrate(meta, sym)
    assert "DWARF" in _shim_title(meta) and "sanitizer runtime" not in _shim_title(meta)


def test_regression_shim_title_flags_secondary_fault():
    """In-process ASAN (halt_on_error=0) reports, runs on, then faults: the
    shim record is that later fault (fuzzgoat UAF reported at :643, SEGV at :987)."""
    meta = CrashMetadata(
        sanitizer="AddressSanitizer",
        error_type="heap-use-after-free",
        raw_stderr="==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x5 at pc 0x99 bp 0x1 sp 0x2\n",
    )
    sym = cs.CrashSymbol(
        signal=11, pc=0x10, symbolized=True, frames=[cs.SymFrame(pc=0x10, function="f")]
    )
    cs.hydrate(meta, sym)
    assert "later fault" in _shim_title(meta)


def test_shim_title_plain_signal_crash_not_flagged():
    """Adversarial: without a sanitizer report the shim record is the crash itself."""
    meta = CrashMetadata()
    cs.hydrate(meta, cs.CrashSymbol(signal=11, pc=0x10, frames=[cs.SymFrame(pc=0x10)]))
    assert "later fault" not in _shim_title(meta)


def test_regression_crash_pcs_skip_interpreter_frames():
    """In-process (.so) records unwind through python3 / libffi; those PCs
    named the fuzzer, not the target, and leaked into short signatures."""
    import sys

    py = os.path.realpath(sys.executable)
    sym = cs.CrashSymbol(
        signal=11,
        pc=0x7F0010,
        frames=[
            cs.SymFrame(pc=0x7F0010, module="/b/t.so"),
            cs.SymFrame(pc=0x7F0020, module="/b/t.so"),
            cs.SymFrame(pc=0xDA9052, module="/usr/lib/libffi.so.8"),
            cs.SymFrame(pc=0x11FCBE9, module=py),
        ],
    )
    assert cs.crash_pcs(sym) == (0x7F0010, 0x7F0020)


def test_crash_pcs_target_exe_kept():
    """Adversarial: a forkserver target's own executable is not the interpreter."""
    sym = cs.CrashSymbol(
        signal=11,
        pc=0x10,
        frames=[cs.SymFrame(pc=0x10, module="/b/target"), cs.SymFrame(pc=0x20, module="/b/target")],
    )
    assert cs.crash_pcs(sym) == (0x10, 0x20)


def test_regression_shim_title_same_fault_not_flagged():
    """Forkserver ASAN SEGV: the report and the shim saw one signal at one pc."""
    meta = CrashMetadata(
        sanitizer="AddressSanitizer",
        error_type="SEGV",
        raw_stderr="==1==ERROR: AddressSanitizer: SEGV on unknown address 0x5 (pc 0x10 bp 0x0 sp 0x2 T0)\n",
    )
    cs.hydrate(
        meta,
        cs.CrashSymbol(
            signal=11, pc=0x10, symbolized=True, frames=[cs.SymFrame(pc=0x10, function="f")]
        ),
    )
    assert "later fault" not in _shim_title(meta)


def test_resolve_raw_skips_interpreter(monkeypatch):
    """In-process: python3's own frames are neither parsed nor named."""
    import sys

    from fuzzer_tool.core import pc_resolver

    asked: list[str] = []
    monkeypatch.setattr(pc_resolver, "resolver_for", lambda m: asked.append(m))
    py = os.path.realpath(sys.executable)
    sym = cs.CrashSymbol(
        signal=11,
        pc=0x10,
        frames=[
            cs.SymFrame(pc=0x10, module="/t/x.so", offset=0x10),
            cs.SymFrame(pc=0x20, module=py, offset=0x20),
        ],
    )
    cs.resolve_raw(sym)
    assert asked == ["/t/x.so"]
