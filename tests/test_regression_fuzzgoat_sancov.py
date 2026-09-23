"""fuzzgoat's library object must carry compiler-inserted coverage.

``fuzzgoat_read.c`` is a thin wrapper; the parser lives in the separately
compiled ``fuzzgoat.c`` object linked in as ``/tmp/fuzzgoat.o``.  Coverage of
``json_parse_ex`` therefore depends entirely on how
``compile_fuzzgoat_object`` compiles that object -- nothing else in the build
instrumented it.

Measured: an in-process campaign against ``fuzzgoat_read_noasan.so`` never
escaped ``shm: 5`` (the wrapper's own ``__afl_map_edge`` ceiling) while the
PIE-based collector saw 179-196 edge ids on the same corpus.  ``nm`` told the
story: the .so had 88 ``sanitizer_cov_trace_pc`` references total and *zero*
inside ``json_parse_ex``, while the PIE had 849 / 198.  The .so's object was
compiled without any ``-fsanitize-coverage`` flag of its own -- unlike the
grep objects, which add ``cov_flag`` inside ``compile_grep_objects`` -- and
through a shared `/tmp/fuzzgoat.o` path that every suffix pass clobbers.

This asserts the same repair structurally (cov_flag present, per-suffix
object path) and behaviorally (a clang compile of the vendored source emits
``sanitizer_cov_trace_pc`` references).
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"

pytestmark = pytest.mark.skipif(not BUILD_SCRIPT.exists(), reason="build_targets.sh not present")

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")


def _extract_fuzzgoat_compile() -> str:
    """Pull compile_fuzzgoat_object out of the build script."""
    text = BUILD_SCRIPT.read_text()
    start = text.index("compile_fuzzgoat_object() {")
    end = text.index("# ── Build a target")
    body = text[start:end]
    assert "compile_fuzzgoat_object() {" in body
    return body


class TestObjectCarriesCoverage:
    def test_compile_line_has_its_own_cov_flag(self):
        """The function must not rely on callers passing -fsanitize-coverage.

        The default ASAN / No-ASAN passes invoke it with empty extra_cflags,
        so no caller-provided flag can be the mechanism.  grep objects already
        add cov_flag inside their compiler (compile_grep_objects); fuzzgoat
        must mirror that.
        """
        body = _extract_fuzzgoat_compile()
        assert "cov_flag" in body
        assert "-fsanitize-coverage=trace-pc-guard" in body
        # gcc has no trace-pc-guard; the case exists in compile_grep_objects.
        assert 'case "$cc" in' in body

    def test_object_path_is_per_suffix(self):
        body = _extract_fuzzgoat_compile()
        assert "${suffix}.o" in body
        # grep objects write /tmp/grep_$(basename "$src")${suffix}.o and every
        # pass keys its object on its own suffix; a shared single path lets
        # the last pass to run clobber what earlier links consumed.
        assert 'local suffix="$1"' in body

    def test_link_sites_consume_the_suffixed_object(self):
        """Every place that links the object must reference the same suffix.

        If the object is now per-suffix but a link site still says
        /tmp/fuzzgoat.o, the pass compiles one variant and links another.
        The wrapper functions pass their own suffix in as $1, so the link
        key is `${1}.o` there; the ngram pass keys on an empty suffix and is
        the one site that legitimately links the plain /tmp/fuzzgoat.o.
        """
        for line in BUILD_SCRIPT.read_text().splitlines():
            if "fuzzgoat_read.c" not in line or "/tmp/fuzzgoat" not in line:
                continue
            if "build_ngram_flavor" in line:
                assert "/tmp/fuzzgoat.o" in line, f"ngram site: {line}"
                continue
            assert "${suffix}.o" in line or "${1}.o" in line, f"link site not per-suffix: {line}"


class TestActualCompile:
    @needs_clang
    def _compile_fixture(self, tmp_path, suffix) -> str:
        vendor = tmp_path / "vendor"
        fuzz_src = vendor / "fuzzgoat"
        fuzz_src.mkdir(parents=True, exist_ok=True)
        (fuzz_src / "fuzzgoat.c").write_text(
            "static int branchy(int x){return x>3?x*2:x-1;}\n"
            "int json_parse(const char*b,unsigned n){int a=0;"
            "for(unsigned i=0;i<n;i++)a+=branchy(b[i]);return a;}\n"
        )
        build_log = tmp_path / "build.log"
        script = textwrap.dedent(
            f"""
            VENDOR={vendor}
            DEFAULT_CC="clang"
            BUILD_LOG={build_log}
            {_extract_fuzzgoat_compile()}
            compile_fuzzgoat_object "{suffix}" "" clang
            """
        )
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        return f"/tmp/fuzzgoat{suffix}.o"

    def test_compiled_object_references_trace_pc(self, tmp_path):
        """A clang compile of the object must be instrumented.

        Before the repair the function took no suffix, so this call treated
        ``_tst`` as flags, invoked the empty compiler, and produced no object
        with trace-pc references.
        """
        obj = self._compile_fixture(tmp_path, "_tst")
        try:
            proc = subprocess.run(["nm", obj], capture_output=True, text=True, timeout=60)
            assert proc.returncode == 0, proc.stderr
            assert "sanitizer_cov_trace_pc" in proc.stdout, proc.stdout
        finally:
            Path(obj).unlink(missing_ok=True)

    def test_distinct_suffixes_get_distinct_objects(self, tmp_path):
        """Each pass's object must land on its own path, not a shared one."""
        objs = [self._compile_fixture(tmp_path, suffix) for suffix in ("_asan", "")]
        try:
            paths = [Path(o) for o in objs]
            assert all(p.exists() for p in paths)
            assert paths[0] != paths[1]
        finally:
            for p in objs:
                Path(p).unlink(missing_ok=True)
