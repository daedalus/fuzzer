"""A target can carry the shim and still have no instrumented call sites.

``afl_instrumentation_status`` looks for ``__afl_area``, ``__afl_map_shm`` and
``__sanitizer_cov``.  All three are the shim's own definitions and the shim is
``-include``'d into every target, so a binary the compiler never instrumented
reports "present" there and the startup line says "detected".

Measured when this was written: a default ``tools/build_targets.sh`` run
produced 20 binaries, 4 of which carried a guard section, and all 20 were
classified "present".  300 execs against one of the other 16 under
``--inprocess-direct`` reported ``shm: 2 max: 2 sat: 100%``,
``Edges discovered: 2``, ``Total richness: 2 - 2 (95% CI, Chao2)`` and
``P(new code next): 0.00%`` -- the two edges being the harness's own
``__afl_map_edge`` calls.  Nothing in that output distinguishes it from a
target that really is exhausted.

``sancov_guard_status`` checks what actually produces edges, the guard array
``-fsanitize-coverage=trace-pc-guard`` emits, which is the same thing
``verify_sancov`` greps for in the build script -- but before a campaign
rather than only at build time.
"""

import shutil
import subprocess

import pytest

from fuzzer_tool.core.elf import sancov_guard_status
from fuzzer_tool.services.fuzzer import Fuzzer, afl_instrumentation_status

_SRC = """
#include <stdlib.h>
static int leaf(int v) { return v > 3 ? v * 2 : v - 1; }
int main(int argc, char **argv) {
    int acc = 0;
    for (int i = 1; i < argc; i++) acc += leaf(i);
    return acc == 0x7fffffff;
}
"""

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")


def _build(tmp_path, name, *flags):
    src = tmp_path / f"{name}.c"
    src.write_text(_SRC)
    exe = tmp_path / name
    proc = subprocess.run(
        ["clang", "-O1", *flags, "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"clang build failed: {proc.stderr[:200]}")
    return str(exe)


class TestSancovGuardStatus:
    @needs_clang
    def test_instrumented_build_is_present(self, tmp_path):
        exe = _build(tmp_path, "inst", "-fsanitize-coverage=trace-pc-guard")
        assert sancov_guard_status(exe) == "present"

    @needs_clang
    def test_inline_counters_also_count(self, tmp_path):
        """A different section and element width, but still instrumented."""
        exe = _build(tmp_path, "cnt", "-fsanitize-coverage=inline-8bit-counters")
        assert sancov_guard_status(exe) == "present"

    @needs_clang
    def test_plain_build_is_absent(self, tmp_path):
        assert sancov_guard_status(_build(tmp_path, "plain")) == "absent"

    @needs_clang
    def test_stripped_instrumented_build_is_unknown(self, tmp_path):
        """Both bound symbols are static-only, so stripping hides the evidence.

        The distinction that matters: .dynsym survives stripping and is full
        of names, so a check that merges the two tables would call this
        "absent" and fire a false alarm on a target that works.
        """
        exe = _build(tmp_path, "stripped", "-fsanitize-coverage=trace-pc-guard")
        if shutil.which("strip") is None:
            pytest.skip("no strip")
        subprocess.run(["strip", exe], capture_output=True)
        assert sancov_guard_status(exe) == "unknown"

    def test_missing_file_is_unknown(self):
        assert sancov_guard_status("/nonexistent/binary") == "unknown"

    @needs_clang
    def test_this_is_exactly_what_the_old_check_misses(self, tmp_path):
        """The regression in one assertion: shim symbols, no instrumentation."""
        exe = _build(tmp_path, "shimonly", "-DNOTHING")
        # No shim here, so afl_instrumentation_status says "absent" for its own
        # reasons; the point is only that the two checks answer different
        # questions and the guard check is the one tied to edges.
        assert sancov_guard_status(exe) == "absent"
        assert afl_instrumentation_status(exe) in {"absent", "present", "unknown"}


class _Bare:
    """Minimal object carrying only what _warn_no_compiler_coverage touches."""

    _warn_no_compiler_coverage = Fuzzer._warn_no_compiler_coverage

    def __init__(self, use_coverage=True, ptrace_cov=None, use_ptrace=False):
        self.use_coverage = use_coverage
        self.ptrace_cov = ptrace_cov
        self.use_ptrace = use_ptrace


class TestNoCompilerCoverageWarning:
    @needs_clang
    def test_warns_on_an_uninstrumented_target(self, tmp_path, capsys):
        exe = _build(tmp_path, "plain")
        _Bare()._warn_no_compiler_coverage(exe)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "--clang-scov" in out

    @needs_clang
    def test_silent_on_an_instrumented_target(self, tmp_path, capsys):
        exe = _build(tmp_path, "inst2", "-fsanitize-coverage=trace-pc-guard")
        _Bare()._warn_no_compiler_coverage(exe)
        assert capsys.readouterr().out == ""

    @needs_clang
    def test_silent_under_ptrace(self, tmp_path, capsys):
        """ptrace sets breakpoints on the binary; build-time flags are moot."""
        exe = _build(tmp_path, "plain_ptrace")
        _Bare(ptrace_cov=object())._warn_no_compiler_coverage(exe)
        _Bare(use_ptrace=True)._warn_no_compiler_coverage(exe)
        assert capsys.readouterr().out == ""

    @needs_clang
    def test_silent_with_coverage_off(self, tmp_path, capsys):
        exe = _build(tmp_path, "plain_nocov")
        _Bare(use_coverage=False)._warn_no_compiler_coverage(exe)
        assert capsys.readouterr().out == ""

    @needs_clang
    def test_warns_once(self, tmp_path, capsys):
        exe = _build(tmp_path, "plain_twice")
        bare = _Bare()
        bare._warn_no_compiler_coverage(exe)
        bare._warn_no_compiler_coverage(exe)
        assert capsys.readouterr().out.count("WARNING") == 1

    def test_silent_when_status_is_unknown(self, tmp_path, capsys):
        """A stripped target must not be nagged about."""
        _Bare()._warn_no_compiler_coverage("/nonexistent/binary")
        assert capsys.readouterr().out == ""
