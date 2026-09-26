"""Regression: the Docker image could not build an ffmpeg campaign.

Ubuntu's `clang` package ships without compiler-rt, so every
`-fsanitize=*` / `-fsanitize-coverage` link failed ("cannot find
libclang_rt.ubsan_standalone") and vendor_ffmpeg.sh's configure aborted.
`curl` was missing too, so the primary source fetch always failed.

The image build itself verifies the ASAN link; these tests pin the wiring.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).parent.parent / "Dockerfile"

pytestmark = pytest.mark.skipif(not DOCKERFILE.exists(), reason="Dockerfile not present")

_FROM_RE = re.compile(r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?", re.M | re.I)


@pytest.fixture(scope="module")
def text() -> str:
    return DOCKERFILE.read_text()


def test_regression_compiler_rt_installed(text):
    """Falsification: sanitizer runtimes and curl must be installed."""
    assert "libclang-rt-dev" in text
    assert re.search(r"^\s+curl\s*\\$", text, re.M)


def test_regression_build_verifies_asan_link(text):
    """The image must fail to build when an ASAN link fails."""
    assert "-fsanitize=address" in text


def test_regression_ffmpeg_stage_builds_harness(text):
    stages = [alias for _, alias in _FROM_RE.findall(text)]
    assert "ffmpeg" in stages

    stage = text.split("AS ffmpeg", 1)[1].split("\nFROM ", 1)[0]
    assert "tools/build_ffmpeg_ready.sh" in stage
    assert "--inprocess-func fuzz_ffmpeg" in stage


def test_regression_default_stage_is_not_ffmpeg(text):
    """Adversarial: a plain `docker build .` must not pay the FFmpeg build."""
    last_base, last_alias = _FROM_RE.findall(text)[-1]
    assert last_alias.lower() != "ffmpeg"
    assert last_base.lower() != "ffmpeg"
