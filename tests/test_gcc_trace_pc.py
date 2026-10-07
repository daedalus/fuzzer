"""gcc ``-fsanitize-coverage=trace-pc`` against the real shim.

gcc has no trace-pc-guard, so trace-pc is its only automatic edge mode. Before
this was supported a gcc trace-pc target built, linked and then segfaulted at
startup: gcc (unlike clang) does not skip functions named ``__sanitizer_cov_*``
and the shim marked almost none of its own code, so
``__sanitizer_cov_trace_pc`` was instrumented and called itself until the stack
ran out.

Covers the four moving parts: the shim no longer instruments itself, the edge
callback works with the distance channel off, the Python side recognises a
section-less trace-pc binary, and ``build_targets.sh`` picks the right flag per
compiler. Clang tests pin the invariant the gcc work must not disturb.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.elf import (
    estimate_map_size_detail,
    sancov_guard_status,
    trace_pc_call_sites,
)
from fuzzer_tool.services.fuzzer import _detect_distance
from tests.conftest import requires_clang

REPO = Path(__file__).parent.parent
SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
BUILD_SCRIPT = REPO / "tools" / "build_targets.sh"
MAP_ENTRIES = 8192
TRACE_PC = "-fsanitize-coverage=trace-pc"


def _gcc_major() -> int:
    gcc = shutil.which("gcc")
    if not gcc:
        return 0
    out = subprocess.run([gcc, "-dumpversion"], capture_output=True, text=True).stdout
    m = re.match(r"(\d+)", out.strip())
    return int(m.group(1)) if m else 0


# no_sanitize_coverage (what the shim excludes its own code with) is gcc >= 12.
requires_gcc12 = pytest.mark.skipif(_gcc_major() < 12, reason="needs gcc >= 12")

# noinline arms: at -O2 an inline if/else folds into a branchless select, and
# trace-pc then has nothing distinct to report for the two inputs.
_BRANCHY = r"""
#include <stdint.h>
#include <string.h>
#include <unistd.h>
volatile int sink;
__attribute__((noinline)) static void hit_x(void) { sink += 7; }
__attribute__((noinline)) static void hit_other(void) { sink -= 1; }
__attribute__((noinline)) static int entry(const uint8_t *d, size_t n) {
    if (n && d[0] == 'X') hit_x();
    else hit_other();
    return 0;
}
int main(int argc, char **argv) {
    const char *in = argc > 1 ? argv[1] : "";
    __afl_guarded_call(entry, (const uint8_t *)in, strlen(in));
    _exit(0);
}
"""

_PLAIN = "int f(int x){return x>3;} int main(int c,char**v){(void)v;return f(c)>5;}\n"


def _build(tmp_path, name, cc, *flags, source=_BRANCHY, opt="-O1"):
    src = tmp_path / f"{name}.c"
    src.write_text(source)
    exe = tmp_path / name
    r = subprocess.run(
        [cc, opt, "-g", *flags, "-include", str(SHIM), "-o", str(exe), str(src), "-ldl"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-800:]
    return str(exe)


def _edges(exe, *args):
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(MAP_ENTRIES))
        env.pop("LD_PRELOAD", None)
        r = subprocess.run([exe, *args], env=env, capture_output=True, timeout=30)
        return r.returncode, cov.get_edge_ids()
    finally:
        cov.cleanup()


def _instrumented_functions(exe):
    """Names of functions in *exe* that contain a call to the trace-pc callback."""
    dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", exe], capture_output=True, text=True).stdout
    names, cur = set(), None
    for line in dis.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            cur = m.group(1)
        elif cur and re.search(r"call.*<__sanitizer_cov_trace_pc>", line):
            names.add(cur)
    return names


# Functions that are the target's own code (and so should be instrumented), or
# compiler-generated static-init stubs that belong to no shim source line.
_TARGET_OWN = {"main", "entry", "hit_x", "hit_other"}


def _is_shim(name):
    return name not in _TARGET_OWN and not re.match(r"_sub_[DI]_", name)


@requires_gcc12
class TestGccTracePc:
    @pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2", "-O3"])
    def test_runs_instead_of_recursing(self, tmp_path, opt):
        """Regression: this segfaulted (rc 139) before the shim excluded itself."""
        exe = _build(tmp_path, "t", "gcc", TRACE_PC, opt=opt)
        rc, edges = _edges(exe, "X")
        assert rc == 0
        assert edges

    @pytest.mark.parametrize(
        "extra",
        [
            [],
            ["-D__AFL_DISTANCE_MODE=0"],
            ["-fsanitize-coverage=trace-pc,trace-cmp", "-D__AFL_CMPLOG=1"],
            ["-D__AFL_NGRAM_K=2"],
            ["-fsanitize=address"],
        ],
    )
    def test_no_shim_function_is_instrumented(self, tmp_path, extra):
        """The structural guarantee behind the crash fix, across shim configs.

        Any shim function left unmarked is a latent recursion or re-entrancy
        hazard under gcc even when this particular binary happens to survive.
        """
        flags = [TRACE_PC, *extra] if not any(f.startswith("-fsanitize-coverage") for f in extra) else extra
        exe = _build(tmp_path, "t", "gcc", *flags)
        offenders = {n for n in _instrumented_functions(exe) if _is_shim(n)}
        assert not offenders, sorted(offenders)

    def test_edges_depend_on_input(self, tmp_path):
        """Falsification: a callback that records nothing yields identical maps."""
        exe = _build(tmp_path, "t", "gcc", TRACE_PC)
        _, taken = _edges(exe, "X")
        _, skipped = _edges(exe, "Y")
        assert taken and skipped
        assert taken != skipped

    def test_edge_ids_are_stable_across_runs(self, tmp_path):
        """PC-keyed ids must survive ASLR (key = pc - load base)."""
        exe = _build(tmp_path, "t", "gcc", TRACE_PC, "-D__AFL_CTX_SENSITIVE=0")
        _, a = _edges(exe, "X")
        _, b = _edges(exe, "X")
        assert a == b

    def test_edge_ids_fit_the_dense_virgin_map(self, tmp_path):
        from fuzzer_tool.adapters.shm import VIRGIN_DENSE_MAX

        exe = _build(tmp_path, "t", "gcc", TRACE_PC)
        _, edges = _edges(exe, "X")
        assert max(edges) < VIRGIN_DENSE_MAX

    def test_distance_off_still_has_edges(self, tmp_path):
        """trace-pc used to be defined only under __AFL_DISTANCE_MODE."""
        exe = _build(tmp_path, "t", "gcc", TRACE_PC, "-D__AFL_DISTANCE_MODE=0")
        rc, edges = _edges(exe, "X")
        assert rc == 0
        assert edges
        assert not _detect_distance(exe)

    def test_distance_on_is_still_detected(self, tmp_path):
        exe = _build(tmp_path, "t", "gcc", TRACE_PC)
        assert _detect_distance(exe)


@requires_gcc12
class TestDetection:
    def test_trace_pc_binary_counts_as_instrumented(self, tmp_path):
        exe = _build(tmp_path, "t", "gcc", TRACE_PC)
        assert (trace_pc_call_sites(exe) or 0) > 0
        assert sancov_guard_status(exe) == "present"

    def test_uninstrumented_gcc_binary_is_absent(self, tmp_path):
        """The shim alone adds no call sites under gcc, so this is a clean 0."""
        exe = _build(tmp_path, "t", "gcc", source=_PLAIN)
        assert trace_pc_call_sites(exe) == 0
        assert sancov_guard_status(exe) == "absent"

    def test_map_size_provenance(self, tmp_path):
        exe = _build(tmp_path, "t", "gcc", TRACE_PC)
        est = estimate_map_size_detail(exe)
        assert est.source == "trace_pc_calls"
        assert est.blocks == trace_pc_call_sites(exe)
        assert not est.exact  # call sites, not a guard array

    def test_stripped_binary_is_unknown_not_zero(self, tmp_path):
        exe = _build(tmp_path, "t", "gcc", TRACE_PC)
        subprocess.run(["strip", exe], check=True)
        assert trace_pc_call_sites(exe) is None


class TestClangUnchanged:
    """What the gcc work must not disturb."""

    @requires_clang
    def test_guard_build_is_not_reported_as_trace_pc(self, tmp_path):
        exe = _build(tmp_path, "t", "clang", "-fsanitize-coverage=trace-pc-guard")
        assert sancov_guard_status(exe) == "present"
        assert estimate_map_size_detail(exe).source == "sancov_guards"

    @requires_clang
    def test_clang_trace_pc_with_distance_off_links(self, tmp_path):
        """Failed to link before: the callback lived inside the distance gate."""
        exe = _build(tmp_path, "t", "clang", TRACE_PC, "-D__AFL_DISTANCE_MODE=0")
        rc, edges = _edges(exe, "X")
        assert rc == 0
        assert edges

    @requires_clang
    def test_clang_trace_pc_edge_ids_are_unmixed(self, tmp_path):
        """Clang keeps ``key >> 1`` ids; mixing is gcc-only (ngram/distance builds)."""
        exe = _build(tmp_path, "t", "clang", TRACE_PC, "-D__AFL_CTX_SENSITIVE=0")
        _, a = _edges(exe, "X")
        _, b = _edges(exe, "Y")
        assert a and b and a != b


def _flag(cc, *, sancov=None, cmplog="0", gcc_trace_pc=None):
    """Run cov_flag_for_cc from build_targets.sh in isolation."""
    text = BUILD_SCRIPT.read_text()
    m = re.search(r"^GCC_TRACE_PC=.*?^cov_flag_for_cc\(\) \{.*?^\}", text, re.S | re.M)
    assert m, "cov_flag_for_cc not found in build_targets.sh"
    env = dict(os.environ)
    env.pop("GCC_TRACE_PC", None)
    if gcc_trace_pc is not None:
        env["GCC_TRACE_PC"] = gcc_trace_pc
    prelude = f"WITH_CMPLOG={cmplog}\n"
    if sancov is not None:
        prelude += f'SANCOV_FLAG="{sancov}"\n'
    r = subprocess.run(
        ["bash", "-c", f'{prelude}{m.group(0)}\ncov_flag_for_cc "{cc}"'],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


class TestBuildScriptFlag:
    def test_clang_gets_guard_by_default(self):
        assert _flag("clang") == "-fsanitize-coverage=trace-pc-guard"

    def test_clang_honours_sancov_choice(self):
        assert _flag("clang", sancov="-fsanitize-coverage=inline-8bit-counters") == (
            "-fsanitize-coverage=inline-8bit-counters"
        )

    def test_clang_ignores_gcc_switch(self):
        assert _flag("clang", gcc_trace_pc="0") == "-fsanitize-coverage=trace-pc-guard"

    @requires_gcc12
    def test_gcc_gets_trace_pc_not_guard(self):
        # SANCOV_FLAG is always set to the guard flag by the script; gcc must not see it.
        flag = _flag("gcc", sancov="-fsanitize-coverage=trace-pc-guard")
        assert flag == TRACE_PC

    @requires_gcc12
    def test_gcc_cmplog_adds_trace_cmp(self):
        assert _flag("gcc", cmplog="1") == "-fsanitize-coverage=trace-pc,trace-cmp"

    @requires_gcc12
    def test_gcc_opt_out(self):
        assert _flag("gcc", gcc_trace_pc="0") == ""

    def test_gcc_flag_is_accepted_by_gcc(self, tmp_path):
        if _gcc_major() < 12:
            pytest.skip("needs gcc >= 12")
        src = tmp_path / "t.c"
        src.write_text(_PLAIN)
        for cmplog in ("0", "1"):
            flag = _flag("gcc", cmplog=cmplog)
            r = subprocess.run(["gcc", "-c", flag, str(src), "-o", str(tmp_path / "t.o")], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
