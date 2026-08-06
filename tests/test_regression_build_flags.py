"""Regression tests: build_targets.sh flag handling.

--clang-scov was completely non-functional — full automatic edge coverage
was unreachable — because of three separate defects:

  1. gated on [ "$OPTS" = "--clang-scov" ], an exact single-argument match,
     so "--fast --clang-scov" silently skipped the whole block;
  2. used `local` at top-level script scope, which is a bash error, so
     SCOV_CC / SCOV_FLAGS were never set even when the block did run;
  3. build_simple_targets took only 3 positional parameters, so SCOV_FLAGS
     never reached the compiler even after 1 and 2 were fixed.

Each failed silently — the build reported success and simply produced
uninstrumented binaries. These tests pin the script-level structure so the
regressions cannot reappear unnoticed. Compilation itself is covered by
the sanitizer/coverage tests elsewhere; here we assert the wiring.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "tools" / "build_targets.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="build_targets.sh not present")


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def test_script_is_syntactically_valid():
    """`local` outside a function is a syntax-level error bash only reports
    at runtime — parse the whole script to catch that class of bug."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr}"


def test_no_local_outside_functions(script_text: str):
    """Regression: `local SCOV_CC=...` sat at top-level scope, so the
    assignment silently failed with 'local: can only be used in a function'.

    Every function in this script is defined at column 0 as `name() {` and
    closed by a `}` at column 0, so track those markers rather than counting
    braces (which appear inside strings and ${...} expansions).
    """
    in_function = False
    offenders = []
    for lineno, line in enumerate(script_text.splitlines(), 1):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", line):
            in_function = True
            continue
        if in_function and line.startswith("}"):
            in_function = False
            continue
        if not in_function and line.strip().startswith("local "):
            offenders.append((lineno, line.strip()))
    assert not offenders, (
        f"`local` used outside a function (fails at runtime with "
        f"'local: can only be used in a function'): {offenders}"
    )


def test_clang_scov_uses_parsed_flag_not_exact_arg_match(script_text: str):
    """Regression: gating on `$OPTS = "--clang-scov"` meant the flag only
    worked as the sole argument. It must use the parsed WITH_CLANG_SCOV."""
    assert '"$OPTS" = "--clang-scov"' not in script_text, (
        "clang-scov gated on exact argument match — breaks when combined with any other flag"
    )
    assert re.search(r'\[\s*"\$WITH_CLANG_SCOV"\s*-eq\s*1\s*\]', script_text), (
        "clang-scov block should be gated on the parsed WITH_CLANG_SCOV flag"
    )


def test_parsed_flags_are_actually_consumed(script_text: str):
    """Every WITH_* flag the parser sets must be read somewhere else.

    A flag that is parsed and displayed in the feature matrix but never
    consumed is a no-op — exactly what --clang-scov was.
    """
    parsed = set(re.findall(r'\[ "\$arg" = "--[a-z-]+" \] && (WITH_[A-Z_]+)=1', script_text))
    assert parsed, "no WITH_* flags found — parser structure changed?"

    unconsumed = []
    for flag in parsed:
        # Uses beyond: initialization, the parser line, and the matrix display.
        uses = re.findall(rf"\$(?:\{{)?{flag}(?:\}})?", script_text)
        gating = re.findall(rf'\[\s*"\$\{{?{flag}}}?"\s*-eq\s*1\s*\]', script_text)
        if not gating and len(uses) < 2:
            unconsumed.append(flag)
    assert not unconsumed, f"flags parsed but never used to gate a build: {unconsumed}"


def test_build_simple_targets_accepts_compiler_and_cflags(script_text: str):
    """Regression: the function took only (suffix, flags, label), so the
    scov flags could not be threaded through to the compiler."""
    m = re.search(
        r"build_simple_targets\(\)\s*\{\s*\n\s*local ([^\n]+)",
        script_text,
    )
    assert m, "build_simple_targets signature not found"
    params = m.group(1)
    assert "cc=" in params, "build_simple_targets must accept a compiler override"
    assert "extra_cflags=" in params, "build_simple_targets must accept extra cflags"


def test_clang_scov_passes_flags_to_simple_targets(script_text: str):
    """The call sites must actually forward SCOV_CC/SCOV_FLAGS."""
    calls = re.findall(r"build_simple_targets [^\n]*Clang-scov[^\n]*", script_text)
    assert calls, "no Clang-scov build_simple_targets calls found"
    for call in calls:
        assert "$SCOV_CC" in call and "$SCOV_FLAGS" in call, (
            f"Clang-scov call does not forward the scov compiler/flags: {call}"
        )


@pytest.mark.parametrize("flag,var", [("--msan", "WITH_MSAN"), ("--tsan", "WITH_TSAN")])
def test_sanitizer_flags_parsed_and_gate_a_build(script_text: str, flag, var):
    assert f'[ "$arg" = "{flag}" ] && {var}=1' in script_text, f"{flag} not parsed"
    assert re.search(rf'\[\s*"\${var}"\s*-eq\s*1\s*\]', script_text), (
        f"{var} does not gate a build block"
    )


def test_msan_skips_uninstrumented_system_lib_targets(script_text: str):
    """MSAN reports false positives against uninstrumented libpng/libz/
    libjpeg, so those targets must be excluded from MSAN builds."""
    m = re.search(r"build_sanitizer_targets\(\)\s*\{(.*?)\n\}", script_text, re.S)
    assert m, "build_sanitizer_targets not found"
    body = m.group(1)
    assert 'if [ "$label" != "MSAN" ]' in body, (
        "MSAN must skip targets linking uninstrumented system libraries"
    )


def test_default_compiler_prefers_clang(script_text: str):
    """clang is the only compiler that can produce full automatic edge
    coverage here, so it must remain the default when available."""
    m = re.search(r"_pick_cc\(\)\s*\{(.*?)\n\}", script_text, re.S)
    assert m, "_pick_cc not found"
    body = m.group(1)
    first_branch = body.split("else")[0]
    assert "clang" in first_branch, "clang should be preferred when available"
