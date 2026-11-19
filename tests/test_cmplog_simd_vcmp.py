"""SIMD trace-cmp: vector ``icmp`` reaches the cmplog stream.

SanitizerCoverage's trace-cmp skips vector compares, so a magic checked with
``_mm_cmpeq_epi8`` never reaches cmplog. ``tools/llvm/vcmp_pass.cpp`` adds a
callback ahead of each vector icmp; ``afl_shim.c`` logs it:

    splat constant  ->  one integer record per lane   CMP 42 61 ... 1 0x<pc>
    anything else   ->  one byte-pair record          CMP 4142..50 6161..61 ...
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

ROOT = Path(__file__).parent.parent
AFL_SHIM = ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
PASS_SRC = ROOT / "tools" / "llvm" / "vcmp_pass.cpp"

MAGIC = b"ABCDEFGHIJKLMNOP"
SPLAT_BYTE = 0x42
WIDE_LANES = 64  # <64 x i8>: CMPLOG_MAX_OPERAND, the widest record the shim writes
HUGE_LANES = 128  # <128 x i8>: past the cap, must clamp
Q_SPLAT = 0x1122334455667788
NOCOV_MAGIC = b"zyxwvutsrqponmlk"

_TARGET_C = r"""
#include <emmintrin.h>
#include <stdint.h>
#include <string.h>
void __tracecmp_flush(void);
typedef unsigned char v64 __attribute__((vector_size(64)));
typedef unsigned char v128 __attribute__((vector_size(128)));
typedef uint64_t v2q __attribute__((vector_size(16)));

__attribute__((noinline)) int magic(const unsigned char *p) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i m = _mm_loadu_si128((const __m128i *)"ABCDEFGHIJKLMNOP");
    return _mm_movemask_epi8(_mm_cmpeq_epi8(v, m)) == 0xFFFF;
}
__attribute__((noinline)) int splat(const unsigned char *p) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    return _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_set1_epi8(0x42)));
}
__attribute__((noinline)) int wide(const unsigned char *p) {
    v64 v, k;
    memcpy(&v, p, sizeof v);
    for (int i = 0; i < 64; i++) k[i] = (unsigned char)(0x80 + i);
    v64 r = v == k;
    return r[0] + r[63];
}
__attribute__((noinline)) int huge(const unsigned char *p) {
    v128 v, k;
    memcpy(&v, p, sizeof v);
    for (int i = 0; i < 128; i++) k[i] = (unsigned char)i;
    v128 r = v == k;
    return r[0] + r[127];
}
__attribute__((noinline)) int quad(const unsigned char *p) {
    v2q v;
    memcpy(&v, p, sizeof v);
    v2q r = v == (v2q){0x1122334455667788ull, 0x1122334455667788ull};
    return (int)(r[0] + r[1]);
}
__attribute__((noinline, no_sanitize("coverage"))) int nocov(const unsigned char *p) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m128i m = _mm_loadu_si128((const __m128i *)"zyxwvutsrqponmlk");
    return _mm_movemask_epi8(_mm_cmpeq_epi8(v, m));
}
__attribute__((noinline)) int solved(const unsigned char *p) {
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    return _mm_movemask_epi8(_mm_cmpeq_epi8(v, v));
}

int main(int argc, char **argv) {
    unsigned char buf[128];
    for (int i = 0; i < 128; i++) buf[i] = (unsigned char)(0x61 + i % 16);
    volatile int sink = 0;
    sink += magic(buf); sink += splat(buf); sink += wide(buf); sink += huge(buf);
    sink += quad(buf); sink += nocov(buf); sink += solved(buf);
    (void)sink; (void)argc; (void)argv;
    __tracecmp_flush();
    return 0;
}
"""

# The input every function reads: 'a'..'p' repeated.
_INPUT = bytes(0x61 + i % 16 for i in range(HUGE_LANES))


def _llvm_config() -> str | None:
    for name in ("llvm-config", "llvm-config-18"):
        if shutil.which(name):
            return name
    return None


def _build_pass(out: Path) -> Path:
    cfg = _llvm_config()
    if cfg is None:
        pytest.skip("llvm-config not installed")
    cxxflags = subprocess.run([cfg, "--cxxflags"], capture_output=True, text=True).stdout.split()
    so = out / "vcmp_pass.so"
    proc = subprocess.run(
        ["clang++", *cxxflags, "-shared", "-fPIC", str(PASS_SRC), "-o", str(so)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"vcmp pass did not build (LLVM dev headers?): {proc.stderr[-400:]}")
    return so


def _build_target(d: Path, plugin: Path | None) -> Path:
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / ("target_vcmp" if plugin else "target_plain")
    extra = [f"-fpass-plugin={plugin}"] if plugin else []
    proc = subprocess.run(
        [
            "clang",
            "-O2",
            *extra,
            "-fsanitize-coverage=trace-pc-guard,trace-cmp",
            "-D__AFL_CMPLOG=1",
            "-include",
            str(AFL_SHIM),
            "-o",
            str(exe),
            str(src),
            "-ldl",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return exe


@pytest.fixture(scope="module")
def targets(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("cmplog_simd_vcmp")
    plugin = _build_pass(d)
    return {"plugin": plugin, "vcmp": _build_target(d, plugin), "plain": _build_target(d, None)}


def _run(exe: Path, tmp_path: Path) -> tuple[list[list[str]], str]:
    records = tmp_path / f"{exe.name}.cmplog"
    counts = tmp_path / f"{exe.name}.counts"
    env = dict(os.environ, _CMPLOG_OUT=str(records), _CMPLOG_COUNTS=str(counts))
    run = subprocess.run([str(exe)], capture_output=True, text=True, env=env, timeout=60)
    assert run.returncode == 0, run.stderr
    lines = [ln.split() for ln in records.read_text().splitlines()]
    return [p for p in lines if p and p[0] == "CMP"], counts.read_text()


def _pairs(recs: list[list[str]]) -> set[tuple[str, str]]:
    return {(p[1], p[2]) for p in recs}


def _count(counts: str, name: str) -> int:
    return sum(
        int(p[2]) for p in (ln.split() for ln in counts.splitlines()) if p[:2] == ["CNT", name]
    )


@requires_clang
class TestSimdVcmp:
    def test_intrinsic_magic_logged(self, targets, tmp_path):
        recs, _ = _run(targets["vcmp"], tmp_path)
        assert (MAGIC.hex(), _INPUT[:16].hex()) in _pairs(recs)

    def test_plain_build_misses_magic(self, targets, tmp_path):
        """Falsification: without the pass, trace-cmp never sees the vector magic."""
        recs, _ = _run(targets["plain"], tmp_path)
        assert not any(MAGIC.hex() in p[1] + p[2] for p in recs)

    def test_splat_logged_per_lane(self, targets, tmp_path):
        recs, _ = _run(targets["vcmp"], tmp_path)
        want = {(f"{SPLAT_BYTE:02x}", f"{b:02x}") for b in _INPUT[:16]}
        assert want <= _pairs(recs)

    def test_quad_splat_lane_width(self, targets, tmp_path):
        """Adversarial: <2 x i64> splat logs 8-byte lanes, not 16 bytes."""
        recs, _ = _run(targets["vcmp"], tmp_path)
        const = Q_SPLAT.to_bytes(8, "little").hex()
        lanes = {_INPUT[:8].hex(), _INPUT[8:16].hex()}
        assert {(const, ln) for ln in lanes} <= _pairs(recs)

    def test_wide_vector_full_record(self, targets, tmp_path):
        recs, _ = _run(targets["vcmp"], tmp_path)
        const = bytes(0x80 + i for i in range(WIDE_LANES)).hex()
        assert (const, _INPUT[:WIDE_LANES].hex()) in _pairs(recs)

    def test_huge_vector_clamped(self, targets, tmp_path):
        """Adversarial: a 128-byte vector clamps to the 64-byte record cap."""
        recs, _ = _run(targets["vcmp"], tmp_path)
        assert (bytes(range(WIDE_LANES)).hex(), _INPUT[:WIDE_LANES].hex()) in _pairs(recs)
        assert all(len(p[1]) <= 2 * WIDE_LANES for p in recs)

    def test_nocov_function_skipped(self, targets, tmp_path):
        """Adversarial: no_sanitize("coverage") (the shim's own __AFL_NO_COV) is not instrumented."""
        recs, _ = _run(targets["vcmp"], tmp_path)
        assert not any(NOCOV_MAGIC.hex() in p[1] + p[2] for p in recs)

    def test_solved_compare_dropped(self, targets, tmp_path):
        """Adversarial: equal vectors are a solved compare; no pair to replace."""
        recs, _ = _run(targets["vcmp"], tmp_path)
        assert (_INPUT[:16].hex(), _INPUT[:16].hex()) not in _pairs(recs)

    def test_links_without_cmplog_shim(self, targets, tmp_path):
        """Adversarial: a TU built with the pass but no cmplog shim (zlib_read_tracecmp.so)
        still links and runs: the pass's weak no-op stubs stand in."""
        exe = tmp_path / "noshim"
        src = tmp_path / "noshim.c"
        src.write_text(_TARGET_C.replace("__tracecmp_flush();", ""))
        build = subprocess.run(
            ["clang", "-O2", f"-fpass-plugin={targets['plugin']}", "-o", str(exe), str(src)],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr[-2000:]
        assert subprocess.run([str(exe)], timeout=60).returncode == 0

    def test_counted_with_scalar_width(self, targets, tmp_path):
        """Vector compares count under the lane-width trace-cmp name (no new segment slot)."""
        _, plain = _run(targets["plain"], tmp_path)
        _, vcmp = _run(targets["vcmp"], tmp_path)
        assert _count(vcmp, "trace_const_cmp1") > _count(plain, "trace_const_cmp1")
        assert _count(vcmp, "trace_const_cmp8") > _count(plain, "trace_const_cmp8")


# ── tools/build_targets.sh wiring ────────────────────────────────────

SCRIPT = ROOT / "tools" / "build_targets.sh"
_SCRIPT_FUNCS = ("sancov_cmp_modes", "sancov_trace_flags", "build_vcmp_pass")
_FALLBACK = "-fno-vectorize -fno-slp-vectorize"


def _func(text: str, name: str) -> str:
    start = text.index(f"\n{name}() {{") + 1
    return text[start : text.index("\n}\n", start) + 3]


def _trace_flags(tmp_path: Path, path: str) -> str:
    """sancov_trace_flags after build_vcmp_pass, run from the repo root."""
    text = SCRIPT.read_text()
    consts = "\n".join(
        ln for ln in text.splitlines() if ln.startswith(("VCMP_PASS_SRC=", "VCMP_FALLBACK="))
    )
    prog = "\n".join(
        [
            "ok() { :; }; warn() { :; }",
            f"TARGETS={tmp_path}; BUILD_LOG={tmp_path}/build.log",
            consts,
            *(_func(text, f) for f in _SCRIPT_FUNCS),
            "build_vcmp_pass; sancov_trace_flags",
        ]
    )
    env = dict(os.environ, PATH=path)
    run = subprocess.run(
        ["bash", "-c", prog], capture_output=True, text=True, cwd=ROOT, env=env, timeout=300
    )
    assert run.returncode == 0, run.stderr
    return run.stdout.strip()


@requires_clang
class TestBuildScript:
    def test_trace_flags_load_plugin(self, tmp_path):
        if _llvm_config() is None:
            pytest.skip("llvm-config not installed")
        flags = _trace_flags(tmp_path, os.environ["PATH"])
        assert f"-fpass-plugin={tmp_path}/vcmp_pass.so" in flags
        assert (tmp_path / "vcmp_pass.so").is_file()

    def test_fallback_without_llvm_config(self, tmp_path):
        """Adversarial: no llvm-config on PATH -> scalarize, never a dangling plugin flag."""
        bindir = tmp_path / "bin"
        bindir.mkdir()
        for tool in ("clang", "clang++", "cut", "bash"):
            (bindir / tool).symlink_to(shutil.which(tool))
        flags = _trace_flags(tmp_path, str(bindir))
        assert _FALLBACK in flags
        assert "-fpass-plugin" not in flags

    def test_both_tracecmp_builds_call_it(self):
        text = SCRIPT.read_text()
        for fn in ("build_tracecmp_targets", "build_vendored_tracecmp_targets"):
            body = _func(text, fn)
            assert body.index("build_vcmp_pass") < body.index("TRACE_FLAGS=$(sancov_trace_flags)")
