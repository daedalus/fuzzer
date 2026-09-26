"""Regression: vendor_ffmpeg.sh masked configure/make failures.

Both steps pipe through `| tail -5` without `pipefail`, so the pipeline's
status was tail's (0). A failed configure fell through to make, make to
verify, and the only error printed was "MISSING: libavformat.a" -- hiding
the real cause (e.g. clang missing compiler-rt in the Docker image).

A fake FFmpeg tree whose `configure` is a stub drives each failure without
network or a real build.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "tools" / "vendor_ffmpeg.sh"

pytestmark = [
    pytest.mark.skipif(not SCRIPT.exists(), reason="vendor_ffmpeg.sh not present"),
    pytest.mark.skipif(shutil.which("clang") is None, reason="clang required"),
]

CONFIGURE_FAIL = "#!/bin/sh\necho 'C compiler test failed.'\nexit 1\n"
CONFIGURE_OK = "#!/bin/sh\nexit 0\n"


def _run(tmp_path: Path, configure: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run vendor_ffmpeg.sh against a fake tree whose configure is *configure*."""
    tree = tmp_path / "ffmpeg"
    tree.mkdir()
    stub = tree / "configure"
    stub.write_text(configure)
    stub.chmod(0o755)

    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "FUZZ_VENDOR_ROOT": str(tmp_path / "vendor"),
        "FFMPEG_DIR": str(tree),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), "--nosan", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_regression_configure_failure_stops(tmp_path):
    """Falsification: a failing configure must abort before make."""
    r = _run(tmp_path, CONFIGURE_FAIL, "--minimal")

    assert r.returncode != 0
    assert "configure failed" in r.stdout
    assert "[4/5]" not in r.stdout


def test_regression_make_failure_stops(tmp_path):
    """Adversarial: configure passes, make has no Makefile -- must abort at make, not verify."""
    r = _run(tmp_path, CONFIGURE_OK, "--minimal")

    assert r.returncode != 0
    assert "make failed" in r.stdout
    assert "[5/5]" not in r.stdout


@pytest.mark.parametrize(("args", "label"), [((), "(nosan)"), (("--minimal",), "(nosan, minimal)")])
def test_regression_minimal_label(tmp_path, args, label):
    """`${MINIMAL:+...}` fired for MINIMAL=0, labelling full builds minimal."""
    r = _run(tmp_path, CONFIGURE_FAIL, *args)

    assert f"Configuring FFmpeg {label}" in r.stdout
