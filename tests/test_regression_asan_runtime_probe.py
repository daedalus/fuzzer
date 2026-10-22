"""ASAN runtime preflight in tools/build_targets.sh.

Without clang's ASAN runtime (``libclang-rt-<ver>-dev``) every ASAN link
failed separately: 30 "cannot find libclang_rt.asan*.a" errors buried in the
build log, nothing saying which package was missing. The probe turns that
into one warning naming the fix, skips the ASAN variants, and counts one
failure so the exit status still reports it.

The function is extracted and sourced; a fake compiler stands in for clang.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"

pytestmark = pytest.mark.skipif(not BUILD_SCRIPT.exists(), reason="build_targets.sh not present")


def _extract() -> str:
    text = BUILD_SCRIPT.read_text()
    start = text.index("# ── ASAN runtime preflight")
    end = text.index("# ── end ASAN runtime preflight")
    body = text[start:end]
    assert "check_asan_runtime()" in body, "extraction range is stale"
    return body


def _run(tmp_path: Path, cc_exit: int) -> list[str]:
    cc = tmp_path / "fakecc"
    cc.write_text(f"#!/bin/sh\nexit {cc_exit}\n")
    cc.chmod(0o755)
    script = textwrap.dedent(
        f"""
        warn() {{ echo "WARN: $1"; }}
        BUILD_FAILURES=0
        BUILD_ASAN=1
        ASAN_CFLAGS="-fsanitize=address"
        ASAN_PROBE_CC="{cc}"
        """
    )
    script += _extract() + '\ncheck_asan_runtime\necho "$BUILD_ASAN $BUILD_FAILURES"\n'
    proc = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()


def test_regression_missing_runtime_skips_asan_and_counts_failure(tmp_path):
    out = _run(tmp_path, cc_exit=1)

    assert out[-1] == "0 1"
    assert any("libclang-rt" in line for line in out[:-1])


def test_runtime_present_keeps_asan(tmp_path):
    """Falsification: a working link leaves the build untouched and silent."""
    out = _run(tmp_path, cc_exit=0)

    assert out == ["1 0"]


def test_asan_not_requested_skips_probe(tmp_path):
    """Adversarial: --nosan builds must not pay for, or fail on, the probe."""
    cc = tmp_path / "fakecc"
    cc.write_text("#!/bin/sh\necho PROBED >&2\nexit 1\n")
    cc.chmod(0o755)
    script = (
        'warn() { echo "WARN: $1"; }\nBUILD_FAILURES=0\nBUILD_ASAN=0\n'
        f'ASAN_CFLAGS="-fsanitize=address"\nASAN_PROBE_CC="{cc}"\n'
        + _extract()
        + '\ncheck_asan_runtime\necho "$BUILD_ASAN $BUILD_FAILURES"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )

    assert proc.stdout.strip() == "0 0"
    assert "PROBED" not in proc.stderr
