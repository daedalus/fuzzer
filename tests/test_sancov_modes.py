"""SanitizerCoverage modes beyond trace-pc-guard, against the real shim.

Before this, a target built with inline-8bit-counters, inline-bool-flag,
pc-table or trace-loads/trace-stores and ``-include afl_shim.c`` did not
link: the shim defined none of their runtime callbacks.

Objects are compiled with ``-fsanitize-coverage`` and linked without it, so
clang does not pull in a sanitizer runtime (absent from minimal toolchains).
``-D__AFL_CTX_SENSITIVE=0`` keeps guard edge ids ASLR-independent, so two
runs of one binary can be compared id-for-id.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.elf import estimate_map_size_detail, sancov_guard_status
from tests.conftest import requires_clang

SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
MAP_ENTRIES = 8192

COUNTERS = "inline-8bit-counters"
BOOLS = "inline-bool-flag"
DATAFLOW = "trace-pc-guard,trace-loads,trace-stores"
INDIR = "indirect-calls"

# Branch on argv[1][0]; 'X' reaches a block no other input reaches.
_BRANCHY = r"""
#include <stdint.h>
#include <string.h>
#include <unistd.h>
volatile int sink;
/* noinline arms: at -O2 an inline if/else folds into a branchless select. */
__attribute__((noinline)) static void hit_x(void) { sink += 7; }
__attribute__((noinline)) static void hit_other(void) { sink -= 1; }
__attribute__((noinline)) static int entry(const uint8_t *d, size_t n) {
    if (n && d[0] == 'X') hit_x();
    else hit_other();
    if (n && d[0] == 'C') *(volatile int *)0 = 1;
    return 0;
}
int main(int argc, char **argv) {
    const char *in = argc > 1 ? argv[1] : "";
    __afl_guarded_call(entry, (const uint8_t *)in, strlen(in));
    _exit(0); /* no destructors: only the guarded-call fold can report */
}
"""

# Same control flow for every index; only the loaded address differs.
_LOADS = r"""
#include <stdlib.h>
int table[64];
int main(int argc, char **argv) {
    int i = argc > 1 ? atoi(argv[1]) : 0;
    __afl_map_edge(0x1100); /* inlined shim-state access, as harness wrappers do */
#ifdef USE_STACK
    int local[64] = {0};
    volatile int *p = &local[i & 63];
#else
    volatile int *p = &table[i & 63];
#endif
    *p = i;
    return *p == -1;
}
"""


@pytest.fixture(params=["-O0", "-O1", "-O2"])
def opt(request):
    """Inlining and the vector fold both change with the optimisation level."""
    return request.param


def _build(tmp_path, name, source, mode, *defines, opt="-O1"):
    src = tmp_path / f"{name}.c"
    src.write_text(source)
    obj = tmp_path / f"{name}.o"
    exe = tmp_path / name

    # Compile with the coverage flag, link without it (see module docstring).
    cc = [
        "clang",
        opt,
        "-fno-omit-frame-pointer",
        "-D__AFL_CTX_SENSITIVE=0",
        *defines,
        f"-fsanitize-coverage={mode}",
        "-include",
        str(SHIM),
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]

    r = subprocess.run(["clang", str(obj), "-o", str(exe), "-ldl"], capture_output=True, text=True)
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


@requires_clang
class TestLinks:
    @pytest.mark.parametrize(
        "mode",
        [COUNTERS, BOOLS, f"{COUNTERS},pc-table", "trace-pc-guard,pc-table", DATAFLOW],
    )
    def test_mode_links_and_runs(self, tmp_path, mode, opt):
        exe = _build(tmp_path, "t", _LOADS, mode, opt=opt)
        rc, edges = _edges(exe, "3")
        assert rc == 0
        assert edges


@requires_clang
class TestInlineFold:
    @pytest.mark.parametrize("mode", [COUNTERS, BOOLS])
    def test_branch_changes_blocks(self, tmp_path, mode, opt):
        """Falsification: an unfolded map is identical for both inputs."""
        exe = _build(tmp_path, "b", _BRANCHY, mode, opt=opt)
        _, taken = _edges(exe, "X")
        _, skipped = _edges(exe, "Y")
        assert taken and skipped
        assert taken != skipped

    @pytest.mark.parametrize("mode", [COUNTERS, BOOLS])
    def test_crash_still_folds(self, tmp_path, mode, opt):
        """Adversarial: the crash path siglongjmps past the normal return."""
        exe = _build(tmp_path, "c", _BRANCHY, mode, opt=opt)
        _, crashed = _edges(exe, "C")
        _, clean = _edges(exe, "Y")
        assert crashed - clean

    def test_fold_clears_counters(self, tmp_path, opt):
        """Adversarial: stale counts would credit old blocks to the next exec."""
        # Walks the shim's own region table (same TU). Declaring the
        # __start___sancov_cntrs bounds here would shadow clang's hidden
        # copies and hand the module ctor a null range. Uninstrumented, so
        # the walk cannot tick counters itself. 255 = nothing registered.
        counter = (
            '__attribute__((no_sanitize("coverage"))) static int live(void) {\n'
            "    if (!__afl_sancov_nregions) return 255;\n"
            "    int n = 0;\n"
            "    for (uint32_t r = 0; r < __afl_sancov_nregions; r++)\n"
            "        for (uint8_t *c = __afl_sancov_regions[r].start;\n"
            "             c < __afl_sancov_regions[r].stop; c++)\n"
            "            n += *c != 0;\n"
            "    return n;\n"
            "}\n"
            "int main("
        )
        src = _BRANCHY.replace("int main(", counter).replace("_exit(0);", "_exit(live());")
        exe = _build(tmp_path, "z", src, COUNTERS, opt=opt)
        rc, _ = _edges(exe, "X")
        assert rc == 0


def _calls_in(exe, fn):
    """Call targets inside one function's disassembly (objdump, local symbols)."""
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", f"--disassemble={fn}", exe],
        capture_output=True,
        text=True,
    ).stdout
    return [line.split("<", 1)[-1].rstrip(">") for line in out.splitlines() if "\tcall" in line]


@requires_clang
class TestFoldSignalSafety:
    def test_fold_calls_no_libc(self, tmp_path, opt):
        """The crash handler runs the fold, so it must not call memset or friends.

        Adversarial: a crash inside malloc followed by a libc call from the
        handler can deadlock. Only the fold's own helpers may appear.
        """
        exe = _build(tmp_path, "s", _BRANCHY, COUNTERS, opt=opt)
        calls = _calls_in(exe, "__afl_sancov_fold") + _calls_in(exe, "__afl_sancov_fold_region")
        assert all(c.startswith("__afl_sancov_") for c in calls), calls


@requires_clang
class TestDataflow:
    def test_global_offset_is_a_feature(self, tmp_path, opt):
        """Falsification: control flow is identical, only the offset differs."""
        exe = _build(tmp_path, "g", _LOADS, DATAFLOW, opt=opt)
        _, a = _edges(exe, "3")
        _, b = _edges(exe, "9")
        assert a != b

    def test_same_offset_is_stable(self, tmp_path, opt):
        """Control: two runs on one input must agree, or the check above is noise."""
        exe = _build(tmp_path, "s", _LOADS, DATAFLOW, opt=opt)
        _, a = _edges(exe, "3")
        _, b = _edges(exe, "3")
        assert a == b

    def test_stack_addresses_are_ignored(self, tmp_path, opt):
        """Adversarial: stack/heap addresses move with ASLR; they must mint nothing."""
        exe = _build(tmp_path, "l", _LOADS, DATAFLOW, "-DUSE_STACK", opt=opt)
        _, a = _edges(exe, "3")
        _, b = _edges(exe, "9")
        assert a == b

    def test_shim_state_is_not_a_feature(self, tmp_path, opt):
        """Adversarial: the shim's own globals share the TU, so they are in range.

        With only stack accesses in the target, a data-flow build must record
        exactly the guard-only build's edges; any extra id is shim bookkeeping.
        """
        flow = _build(tmp_path, "f", _LOADS, DATAFLOW, "-DUSE_STACK", opt=opt)
        plain = _build(tmp_path, "p", _LOADS, "trace-pc-guard", "-DUSE_STACK", opt=opt)
        _, with_flow = _edges(flow, "3")
        _, without = _edges(plain, "3")
        assert with_flow == without


@requires_clang
class TestElfRecognition:
    def test_bool_flag_is_instrumented(self, tmp_path):
        exe = _build(tmp_path, "e", _LOADS, BOOLS)
        assert sancov_guard_status(exe) == "present"

    def test_bool_flag_block_count_is_exact(self, tmp_path):
        exe = _build(tmp_path, "m", _LOADS, BOOLS)
        est = estimate_map_size_detail(exe)
        assert est.source == "sancov_bools"
        assert est.exact


# ── tools/build_targets.sh: --sancov=MODES ─────────────────────────────

BUILD_SCRIPT = Path(__file__).parent.parent / "tools" / "build_targets.sh"


def _bash_fn(name):
    """One function's source, cut from the build script by its column-0 braces."""
    text = BUILD_SCRIPT.read_text()
    start = text.index(f"{name}() {{")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def _run_fns(body, *names):
    """Run *body* in bash after loading the named functions from the build script."""
    script = "\n".join([*(_bash_fn(n) for n in names), body])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _validate(modes):
    return _run_fns(
        f"validate_sancov_modes '{modes}'", "sancov_cmp_modes", "validate_sancov_modes"
    ).returncode


class TestBuildScriptModes:
    @pytest.mark.parametrize(
        "modes",
        [
            "trace-pc-guard",
            f"{COUNTERS},pc-table",
            f"{BOOLS},trace-loads",
            f"trace-pc-guard,{INDIR}",
            "trace-pc-guard,trace-cmp,trace-div,trace-gep",
            f"{COUNTERS},trace-cmp",
        ],
    )
    def test_supported_modes_pass(self, modes):
        assert _validate(modes) == 0

    @pytest.mark.parametrize(
        "modes", ["pc-table,trace-loads", INDIR, "trace-cmp", "trace-cmp,trace-div,trace-gep"]
    )
    def test_no_edge_mode_rejected(self, modes):
        """Falsification: pc-table/loads/indirect-calls/trace-cmp alone record no edges."""
        assert _validate(modes) != 0

    @pytest.mark.parametrize("modes", ["trace-lods", "trace-pc-guard,stack-depth", ""])
    def test_unsupported_mode_rejected(self, modes):
        """Adversarial: a typo or a mode the shim lacks callbacks for fails the link late."""
        assert _validate(modes) != 0

    def test_flag_parsed(self):
        text = BUILD_SCRIPT.read_text()
        assert "--sancov=*)" in text
        assert 'validate_sancov_modes "$SANCOV_MODES"' in text

    def test_indir_flag_parsed(self):
        assert "--indir-cov" in BUILD_SCRIPT.read_text()

    @pytest.mark.parametrize(
        ("modes", "want"),
        [
            ("trace-pc-guard", 1),
            (f"trace-pc-guard,{INDIR},trace-loads", 1),
            ("trace-pc-guard,trace-cmp", 0),
            ("trace-pc-guard,trace-div", 0),
            (f"{COUNTERS},trace-gep", 0),
        ],
    )
    def test_needs_cmplog_only_for_cmp_modes(self, modes, want):
        """A cmp/div/gep mode needs the shim's cmplog layer; nothing else does."""
        r = _run_fns(f"sancov_needs_cmplog '{modes}'", "sancov_cmp_modes", "sancov_needs_cmplog")
        assert r.returncode == want

    def test_needs_cmplog_not_matched_by_prefix(self):
        """Adversarial: a longer name containing a cmp token is not that token."""
        r = _run_fns(
            "sancov_needs_cmplog 'trace-pc-guard,xtrace-cmp'",
            "sancov_cmp_modes",
            "sancov_needs_cmplog",
        )
        assert r.returncode == 1

    def test_vendored_trace_flags_unchanged(self):
        """Control: deriving the flag from the table must not move a single byte,
        or every vendored object would rebuild and saved edge ids would shift."""
        r = _run_fns("sancov_trace_flags", "sancov_cmp_modes", "sancov_trace_flags")
        assert r.stdout.strip() == (
            "-fsanitize-coverage=trace-cmp,trace-div,trace-gep,trace-pc-guard"
        )

    def test_no_hardcoded_cmp_flag_left(self):
        """One place per AGENTS.md rule 1: builds must call sancov_trace_flags."""
        text = BUILD_SCRIPT.read_text()
        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        assert not [ln for ln in code if "trace-cmp,trace-div,trace-gep" in ln]
        assert text.count("TRACE_FLAGS=$(sancov_trace_flags)") == 2

    @pytest.mark.parametrize(
        ("flags", "modes", "cmplog", "want"),
        [
            ("-fsanitize=memory", "trace-pc-guard,trace-cmp", 1, 0),
            ("-fsanitize=thread", "trace-pc-guard,trace-div", 1, 0),
            # Default behaviour unchanged: no cmp mode -> MSAN/TSAN stay excluded.
            ("-fsanitize=memory", "trace-pc-guard", 1, 1),
            ("-fsanitize=thread", "trace-pc-guard", 1, 1),
            # Non-MSAN/TSAN builds are unaffected by the modes.
            ("-fsanitize=address", "trace-pc-guard", 1, 0),
            ("", "trace-pc-guard", 1, 0),
            # WITH_CMPLOG off always wins.
            ("-fsanitize=memory", "trace-pc-guard,trace-cmp", 0, 1),
            ("", "trace-pc-guard,trace-cmp", 0, 1),
        ],
    )
    def test_build_wants_cmplog(self, flags, modes, cmplog, want):
        """MSAN/TSAN executables get the cmplog layer only when a cmp mode is asked
        for: otherwise the sanitizer runtime's weak no-op stubs silently win."""
        r = _run_fns(
            f"WITH_CMPLOG={cmplog}; SANCOV_MODES='{modes}'; build_wants_cmplog '{flags}'",
            "sancov_cmp_modes",
            "sancov_needs_cmplog",
            "build_wants_cmplog",
        )
        assert r.returncode == want

    def test_cmp_mode_turns_cmplog_on(self):
        """The build script must set WITH_CMPLOG for a --sancov cmp mode, even if
        it was off, so the shim callbacks the objects reference get compiled in."""
        r = _run_fns(
            'WITH_CMPLOG=0; SANCOV_MODES="trace-pc-guard,trace-cmp"\n'
            'sancov_needs_cmplog "$SANCOV_MODES" && WITH_CMPLOG=1; echo $WITH_CMPLOG',
            "sancov_cmp_modes",
            "sancov_needs_cmplog",
        )
        assert r.stdout.strip() == "1"
        assert 'sancov_needs_cmplog "$SANCOV_MODES" && WITH_CMPLOG=1' in BUILD_SCRIPT.read_text()

    @pytest.mark.parametrize(
        ("given", "want"),
        [
            ("trace-pc-guard", f"trace-pc-guard,{INDIR}"),
            (f"{COUNTERS},pc-table", f"{COUNTERS},pc-table,{INDIR}"),
            (f"trace-pc-guard,{INDIR}", f"trace-pc-guard,{INDIR}"),
        ],
    )
    def test_indir_mode_added_once(self, given, want):
        """--indir-cov extends whatever --sancov chose, and never twice."""
        script = f"{_bash_fn('add_indir_mode')}\nadd_indir_mode '{given}'"
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert out.stdout.strip() == want

    def test_indir_mode_not_matched_by_prefix(self):
        """Adversarial: a longer mode name containing the token is not the token."""
        script = f"{_bash_fn('add_indir_mode')}\nadd_indir_mode 'trace-pc-guard,x{INDIR}'"
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert out.stdout.strip().endswith(f",{INDIR}")
        assert out.stdout.strip().count(INDIR) == 2

    @requires_clang
    def test_object_follows_modes(self, tmp_path):
        """compile_fuzzgoat_object must honour $SANCOV_FLAG, not hardcode guards."""
        vendor = tmp_path / "vendor" / "fuzzgoat"
        vendor.mkdir(parents=True)
        (vendor / "fuzzgoat.c").write_text("int f(int x){return x>3?x*2:x-1;}\n")
        suffix = f"_sancovtest{os.getpid()}"
        obj = Path(f"/tmp/fuzzgoat{suffix}.o")
        script = (
            f"VENDOR={tmp_path / 'vendor'}\nBUILD_LOG={tmp_path / 'log'}\n"
            f"SANCOV_FLAG=-fsanitize-coverage={COUNTERS}\n"
            f"{_bash_fn('compile_fuzzgoat_object')}\ncompile_fuzzgoat_object {suffix} '' clang"
        )
        try:
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            sections = subprocess.run(
                ["readelf", "-S", str(obj)], capture_output=True, text=True
            ).stdout
            assert "__sancov_cntrs" in sections
            assert "__sancov_guards" not in sections
        finally:
            obj.unlink(missing_ok=True)

    @requires_clang
    @pytest.mark.parametrize("mode", [COUNTERS, BOOLS])
    def test_verify_accepts_inline_modes(self, tmp_path, mode):
        """verify_sancov must not warn "ZERO edges" on a counters/bools .so."""
        src = tmp_path / "t.c"
        src.write_text(
            "int fuzz_shm_run(const unsigned char *d, unsigned long n){return n && d[0];}\n"
        )
        obj = tmp_path / "t.o"
        so = tmp_path / "t.so"
        cc = ["clang", "-fPIC", f"-fsanitize-coverage={mode}", "-c", str(src), "-o", str(obj)]
        subprocess.run(cc, check=True)
        subprocess.run(["clang", "-shared", str(obj), "-o", str(so)], check=True)

        script = (
            'ok() { echo "OK: $1"; }\nwarn() { echo "WARN: $1"; }\n'
            f"WITH_CLANG_SCOV=1\nTARGETS={tmp_path}\n{_bash_fn('verify_sancov')}\nverify_sancov"
        )
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout
        assert "OK: 1 .so targets" in out
        assert "WARN" not in out
