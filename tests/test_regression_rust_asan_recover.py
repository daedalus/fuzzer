"""Regression: Rust ASAN targets built without -fsanitize-recover=address.

build_rust_target.sh passed a literal -fsanitize=address to its _asan
variants and -Z sanitizer=address to rustc, never the recover flags.
direct_lite's halt_on_error=0 is ignored by non-recover check sites, so
the first report Die()d inside the fuzzer -- the C-target bug
test_regression_direct_lite_asan_recover.py pins, still open for Rust.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).parent.parent / "tools"
RUST_SCRIPT = TOOLS / "build_rust_target.sh"
C_SCRIPT = TOOLS / "build_targets.sh"
ASAN_FLAGS = TOOLS / "lib" / "asan_flags.sh"

RECOVER = "-fsanitize-recover=address"
RUST_RECOVER = "-Z sanitizer-recover=address"


@pytest.fixture(scope="module")
def rust_text() -> str:
    return RUST_SCRIPT.read_text()


def _sourced(var: str) -> str:
    """*var* as a shell sees it after sourcing the shared file."""
    r = subprocess.run(
        ["bash", "-c", f'. "{ASAN_FLAGS}" && printf %s "${var}"'],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return r.stdout


def test_regression_rust_asan_variants_use_shared_flags(rust_text):
    variants = re.findall(r'build_variant "_asan" ("[^"]*")', rust_text)

    assert variants, "no _asan variant found"
    assert set(variants) == {'"$ASAN_CFLAGS"'}


def test_regression_rust_sanitizer_recovers(rust_text):
    """Every ASAN RUSTFLAGS takes the shared, recover-mode rustc flags."""
    rustflags = [
        line for line in rust_text.splitlines()
        if "RUSTFLAGS=" in line and ("-Z sanitizer" in line or "RUST_ASAN_FLAGS" in line)
    ]  # fmt: skip

    assert rustflags
    assert all("$RUST_ASAN_FLAGS" in line and "-Z sanitizer" not in line for line in rustflags)
    assert RUST_RECOVER in _sourced("RUST_ASAN_FLAGS")


@pytest.mark.parametrize("script", [RUST_SCRIPT, C_SCRIPT], ids=lambda p: p.name)
def test_both_build_scripts_source_one_definition(script):
    """Falsification: a second ASAN_CFLAGS literal can drift from the shared one."""
    text = script.read_text()

    assert "tools/lib/asan_flags.sh" in text
    assert not re.search(r"^ASAN_CFLAGS=", text, re.M)


def test_shared_asan_cflags_recover():
    flags = _sourced("ASAN_CFLAGS").split()

    assert "-fsanitize=address" in flags
    assert RECOVER in flags


def test_no_literal_asan_flag_reaches_a_rust_variant(rust_text):
    """Adversarial: a hand-typed -fsanitize=address bypasses recover mode."""
    offenders = [
        line.strip()
        for line in rust_text.splitlines()
        if "build_variant" in line and '"-fsanitize=address"' in line
    ]
    assert offenders == []
