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


@pytest.mark.parametrize("bad", ["", "garbage", "SYM x y z", "SYM 11 zz 0x0 -", "SYM 11", "SYM 11 0x1 0x0 nocolon"])
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


def _build(tmp_path, *flags):
    cc = shutil.which("gcc") or shutil.which("clang")
    if not cc:
        pytest.skip("no C compiler")
    (tmp_path / "h.c").write_text(_HARNESS)
    exe = tmp_path / "h"
    r = subprocess.run(
        [cc, "-g", "-O0", *flags, "-o", str(exe), str(tmp_path / "h.c"), str(SHIM), "-ldl", "-lpthread", "-lrt"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"shim did not build here: {r.stderr[-300:]}")
    return exe


def _run(exe, out, **env):
    e = {**os.environ, **env}
    e.pop(cs.ENV_VAR, None)
    if out is not None:
        e[cs.ENV_VAR] = str(out)
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
