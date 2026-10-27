"""Call-context collapse detector (afl_shim.c) and its build-script companions.

CTX hashes the return address one frame above the instrumented function, read
by walking saved frame pointers. Instrumented code built without them (gcc and
clang drop them at -O2 by default) does not fail the walk, it degrades:

  constant  rbp still holds a caller's frame, so every edge gets the same
            context. Valid-looking ids, no crash, no log line.
  junk      rbp is a general register, the walk fails a bounds check, and
            every edge gets context 0.

Measured with a 150-function chain on gcc 13, clang 18 and zig 0.17: the
detector fires on both shapes and stays silent with frame pointers, with CTX
off, and below the edge threshold. The build-script tests pin the wiring of
the three companions: FFmpeg's frame-pointer flag, --no-sibling-calls, and the
zig -O0 UBSan quirk.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from fuzzer_tool.core.shim_health import Attach, ShimField, ShimWatch, health_issues
from tests.test_regression_shim_audit import _clean_env

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "src/fuzzer_tool/adapters/afl_shim.c"
BUILD_SCRIPT = ROOT / "tools" / "build_targets.sh"

X86_64 = platform.machine() in ("x86_64", "AMD64")

COMPILERS = [
    pytest.param(("clang", "-fsanitize-coverage=trace-pc-guard"), id="clang-guard"),
    pytest.param(("gcc", "-fsanitize-coverage=trace-pc"), id="gcc-trace-pc"),
]

TAG_LINE = "__afl_shim: call-context looks collapsed"


# ── Python side ────────────────────────────────────────────────────────


def _counters(**fields: int) -> list[int]:
    out = [0] * len(ShimField)
    out[ShimField.ATTACHED] = 1
    for name, value in fields.items():
        out[ShimField[name]] = value
    return out


def test_shimfield_indices_are_contiguous():
    assert [int(f) for f in ShimField] == list(range(len(ShimField)))


def test_collapsed_counter_yields_one_actionable_issue():
    issues = health_issues(_counters(CTX_COLLAPSED=1), Attach.EXPECTED)
    assert len(issues) == 1
    assert "-fno-omit-frame-pointer" in issues[0]
    assert "collapsed" in issues[0]


def test_healthy_counters_yield_no_collapse_issue():
    assert health_issues(_counters(CTX_RA_CHANGES=400, CTX_BAILS=3), Attach.EXPECTED) == []


def test_older_shim_without_the_field_is_not_flagged():
    older = _counters()[: ShimField.CTX_RA_CHANGES]
    assert health_issues(older, Attach.EXPECTED) == []


def test_watch_reports_collapse_once():
    watch = ShimWatch()
    bad = _counters(CTX_COLLAPSED=1)
    assert len(watch.counters(bad, Attach.EXPECTED)) == 1
    assert watch.counters(bad, Attach.EXPECTED) == []


def test_watch_dedups_the_stderr_line():
    watch = ShimWatch()
    line = f"{TAG_LINE}: x\n"
    assert watch.stderr(line) == [line.strip()]
    assert watch.stderr(line) == []


# ── C side: build a harness + a generated target, read the counters ────

_MAIN = r"""
#include <stdio.h>
#include <stdlib.h>
void run(void);
int main(void) {
    uint32_t map = 4096;
    uint8_t *g = calloc(1, 32 + (size_t)map * 8 + 16);
    __afl_area        = (struct __afl_entry *)(g + 32);
    __afl_map_size    = map;
    __afl_stack_depth = (uint32_t *)g;
    __afl_gen_word    = (uint32_t *)(g + 4);
    __afl_path_hash   = (uint64_t *)(g + 8);
    __afl_edge_count  = (uint64_t *)(g + 16);
    __afl_dropped     = (uint64_t *)(g + 24);
    run();
    uint64_t h[32] = {0};
    uint32_t n = __afl_shim_health(h, 32);
    printf("%u %d %d %d %llu %llu %llu\n", n,
           (int)__AFL_HEALTH_CTX_RA_CHANGES, (int)__AFL_HEALTH_CTX_COLLAPSED,
           (int)__AFL_HEALTH_CTX_BAILS,
           (unsigned long long)h[__AFL_HEALTH_CTX_RA_CHANGES],
           (unsigned long long)h[__AFL_HEALTH_CTX_COLLAPSED],
           (unsigned long long)h[__AFL_HEALTH_CTX_BAILS]);
    return 0;
}
"""


def _target_source(n_funcs: int, *, junk_rbp: bool) -> str:
    """n_funcs noinline functions called from distinct sites, no live values.

    Constants as arguments keep ``run`` from using any callee-saved register,
    so with frame pointers omitted rbp still holds main's frame (constant
    shape). *junk_rbp* loads rbp with a small integer first (junk shape).
    """
    parts = ["volatile int sink;"]
    for i in range(n_funcs):
        parts.append(
            f"__attribute__((noinline)) int f{i}(int x)"
            f" {{ if (x & 1) sink += {i}; else sink -= {i}; return sink; }}"
        )
    pre = '__asm__ volatile("mov $0x1234, %%rbp" ::: "rbp");' if junk_rbp else ""
    calls = " ".join(f"f{i}(3);" for i in range(n_funcs))
    parts.append(f"void run(void) {{ {pre} {calls} }}")
    return "\n".join(parts)


def _run_harness(
    tmp: Path,
    cc: str,
    cov: str,
    *,
    target_fp: str,
    n_funcs: int = 150,
    junk_rbp: bool = False,
    shim_defs: tuple[str, ...] = (),
):
    (tmp / "main.c").write_text(textwrap.dedent(_MAIN))
    (tmp / "target.c").write_text(_target_source(n_funcs, junk_rbp=junk_rbp))
    t_obj, m_obj, exe = tmp / "target.o", tmp / "main.o", tmp / "prog"
    r = subprocess.run(
        [cc, "-O2", target_fp, cov, "-w", "-c", str(tmp / "target.c"), "-o", str(t_obj)],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [cc, "-O2", "-fno-omit-frame-pointer", *shim_defs, "-w", "-include", str(SHIM),
         "-c", str(tmp / "main.c"), "-o", str(m_obj)],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [cc, str(m_obj), str(t_obj), "-o", str(exe), "-ldl", "-lpthread", "-lrt"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    prog = subprocess.run(
        [str(exe)], capture_output=True, text=True, env=_clean_env(), check=True, timeout=60
    )
    n, i_ra, i_coll, i_bail, ra, coll, bails = (int(x) for x in prog.stdout.split())
    return {
        "n": n, "idx": (i_ra, i_coll, i_bail),
        "ra_changes": ra, "collapsed": coll, "bails": bails, "stderr": prog.stderr,
    }  # fmt: skip


@pytest.fixture(params=COMPILERS)
def toolchain(request):
    cc, cov = request.param
    if shutil.which(cc) is None:
        pytest.skip(f"{cc} not installed")
    return cc, cov


def test_c_layout_matches_python_field_indices(tmp_path, toolchain):
    cc, cov = toolchain
    r = _run_harness(tmp_path, cc, cov, target_fp="-fno-omit-frame-pointer")
    assert r["n"] == len(ShimField)
    assert r["idx"] == (
        int(ShimField.CTX_RA_CHANGES),
        int(ShimField.CTX_COLLAPSED),
        int(ShimField.CTX_BAILS),
    )


def test_frame_pointers_keep_context_and_stay_silent(tmp_path, toolchain):
    cc, cov = toolchain
    r = _run_harness(tmp_path, cc, cov, target_fp="-fno-omit-frame-pointer")
    assert r["collapsed"] == 0
    assert r["ra_changes"] > 100, "each call site should be a distinct context"
    assert TAG_LINE not in r["stderr"]


def test_omitted_frame_pointers_give_constant_context_and_are_flagged(tmp_path, toolchain):
    cc, cov = toolchain
    r = _run_harness(tmp_path, cc, cov, target_fp="-fomit-frame-pointer")
    assert r["collapsed"] == 1
    assert r["ra_changes"] == 1
    assert r["stderr"].count(TAG_LINE) == 1, "one line per process"


@pytest.mark.skipif(not X86_64, reason="loads rbp with inline asm")
def test_junk_rbp_gives_zero_context_and_is_flagged(tmp_path, toolchain):
    cc, cov = toolchain
    r = _run_harness(tmp_path, cc, cov, target_fp="-fomit-frame-pointer", junk_rbp=True)
    assert r["collapsed"] == 1
    # run()'s own entry guard fires before the asm loads rbp, so one walk
    # succeeds; every callee after it fails the bounds check.
    assert r["ra_changes"] <= 1
    assert r["bails"] > 100
    assert r["stderr"].count(TAG_LINE) == 1


def test_small_targets_are_not_judged(tmp_path, toolchain):
    cc, cov = toolchain
    r = _run_harness(tmp_path, cc, cov, target_fp="-fomit-frame-pointer", n_funcs=5)
    assert r["collapsed"] == 0
    assert TAG_LINE not in r["stderr"]


def test_context_off_is_not_flagged(tmp_path, toolchain):
    cc, cov = toolchain
    r = _run_harness(
        tmp_path, cc, cov, target_fp="-fomit-frame-pointer",
        shim_defs=("-D__AFL_CTX_SENSITIVE=0",),
    )  # fmt: skip
    assert r["collapsed"] == 0
    assert TAG_LINE not in r["stderr"]


# ── Warnings the cleanup removed stay removed ──────────────────────────


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not installed")
def test_shim_compiles_without_warnings_under_gcc(tmp_path):
    src = ROOT / "tools" / "shim-micro-benchmark.c"
    r = subprocess.run(
        ["gcc", "-O2", "-Wall", "-Wextra", "-include", str(SHIM), "-c", str(src),
         "-o", str(tmp_path / "b.o")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "warning" not in r.stderr, r.stderr


# ── build_targets.sh wiring ────────────────────────────────────────────


@pytest.fixture(scope="module")
def script_text() -> str:
    if not BUILD_SCRIPT.exists():
        pytest.skip("build_targets.sh not present")
    return BUILD_SCRIPT.read_text()


def _bash(snippet: str, **env: str):
    return subprocess.run(
        ["bash", "-c", snippet], capture_output=True, text=True, env=_clean_env(**env)
    )


def _extract_fn(text: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert m, f"{name} not found in build_targets.sh"
    return m.group(0)


def _warn_check(
    script_text: str, tmp_path: Path, config_mak: str | None
) -> subprocess.CompletedProcess:
    mak = tmp_path / "config.mak"
    if config_mak is not None:
        mak.write_text(config_mak)
    prelude = "YELLOW=''\nNC=''\n" + 'warn() { echo -e "  WARN: $1"; }\n'
    fn = _extract_fn(script_text, "warn_if_frame_pointer_omitted")
    return _bash(f'{prelude}{fn}\nwarn_if_frame_pointer_omitted "{mak}" "ffmpeg"; echo rc=$?')


def test_ffmpeg_configure_gets_the_frame_pointer_flag(script_text):
    m = re.search(r'^\s*COV_FLAGS="\$COV_FLAGS \$FRAME_POINTER \$DEBUG_CFLAGS"', script_text, re.M)
    assert m, "FFmpeg COV_FLAGS must carry FRAME_POINTER (and stay in COV_FLAGS for cfg_stamp)"


def test_ffmpeg_configure_result_is_checked(script_text):
    assert (
        "warn_if_frame_pointer_omitted"
        in script_text.split("vendored FFmpeg${label}: make")[0][-400:]
    )


def test_frame_pointer_check_is_quiet_when_ours_wins(script_text, tmp_path):
    r = _warn_check(
        script_text, tmp_path, "CFLAGS=-O3 -fomit-frame-pointer -fno-omit-frame-pointer -g\n"
    )
    assert r.returncode == 0, r.stderr
    assert "WARN" not in r.stdout
    assert "rc=0" in r.stdout


def test_frame_pointer_check_warns_when_configure_appends_omit(script_text, tmp_path):
    r = _warn_check(
        script_text, tmp_path, "CFLAGS=-fno-omit-frame-pointer -O3 -fomit-frame-pointer\n"
    )
    assert "WARN" in r.stdout and "-fomit-frame-pointer" in r.stdout
    assert "rc=0" in r.stdout, "a warning must never fail the build"


def test_frame_pointer_check_warns_when_flag_is_missing(script_text, tmp_path):
    r = _warn_check(script_text, tmp_path, "CFLAGS=-O3 -g\n")
    assert "no frame-pointer flag" in r.stdout


def test_frame_pointer_check_ignores_a_missing_file(script_text, tmp_path):
    r = _warn_check(script_text, tmp_path, None)
    assert r.stdout.strip() == "rc=0"


def test_no_sibling_calls_flag_is_wired_after_frame_pointer(script_text):
    fp_def = script_text.index('FRAME_POINTER="-fno-omit-frame-pointer"')
    parse = script_text.index('[ "$arg" = "--no-sibling-calls" ] && NO_SIBLING_CALLS=1')
    apply_ = script_text.index('FRAME_POINTER="$FRAME_POINTER -fno-optimize-sibling-calls"')
    assert fp_def < parse < apply_


def test_no_sibling_calls_is_off_by_default_and_env_enabled(script_text):
    assert 'NO_SIBLING_CALLS="${NO_SIBLING_CALLS:-0}"' in script_text
    assert "--no-sibling-calls" in script_text.split("set -e")[0], "documented in the usage header"


def _flag(cc: str) -> str:
    text = BUILD_SCRIPT.read_text()
    m = re.search(r"^GCC_TRACE_PC=.*?^cov_flag_for_cc\(\) \{.*?^\}", text, re.S | re.M)
    assert m
    r = _bash(f'WITH_CMPLOG=0\n{m.group(0)}\ncov_flag_for_cc "{cc}"')
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.mark.parametrize("cc", ["zig cc", "/opt/zig/zig cc"])
def test_zig_gets_guard_coverage_without_ubsan(script_text, cc):
    flag = _flag(cc)
    assert "-fsanitize-coverage=trace-pc-guard" in flag
    assert "-fno-sanitize=undefined" in flag


def test_clang_flag_is_unchanged(script_text):
    assert _flag("clang") == "-fsanitize-coverage=trace-pc-guard"


def test_a_path_that_merely_contains_zig_is_not_zig(script_text):
    assert "-fno-sanitize=undefined" not in _flag("/home/zigfan/bin/clang")
