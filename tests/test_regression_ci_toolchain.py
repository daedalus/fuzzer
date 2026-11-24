"""CI must install and assert the ASAN runtime (2026-11-24 review).

The workflow installed clang but not ``libclang-rt-dev`` (compiler-rt):
Ubuntu's clang omits the ASAN runtime, so 38 ASAN-gated tests — 20 in
test_mutant_build, 10 in test_mutant_kill, 8 in the CLI mutants tests —
skipped silently on every CI run while the verify step reported a clean
toolchain, because it probed for clang and z3 but never linked a
``-fsanitize=address`` program. The Dockerfile had learned this lesson
already; the workflow had not.

This gate pins both halves textually (no YAML dependency): the package in
the install step and the runtime probe in the verify step.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CI = REPO / ".github" / "workflows" / "ci.yml"


def _ci_text() -> str:
    assert CI.is_file(), "ci.yml moved — update this test's path"
    return CI.read_text()


def test_ci_installs_the_asan_runtime_package():
    text = _ci_text()
    assert "libclang-rt-dev" in text, (
        "ci.yml no longer installs libclang-rt-dev: Ubuntu's clang omits "
        "compiler-rt, and every -fsanitize=address link (38 gated tests) "
        "skips silently without it"
    )


def test_ci_verifies_the_asan_runtime_not_just_clang():
    text = _ci_text()
    verify_section = text.split("Verify the toolchain", 1)[-1]
    assert "-fsanitize=address" in verify_section, (
        "the verify step no longer links a -fsanitize=address probe — a "
        "missing ASAN runtime is a silent skip, not a detected failure"
    )
    assert "ASAN" in verify_section, "the verify step should name the ASAN runtime it probes for"
