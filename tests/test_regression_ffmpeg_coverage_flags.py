"""Regression: the linked FFmpeg build dropped reachable code.

build_targets.sh configured with --disable-parsers --disable-bsfs, left
zlib/bzlib/lzma to host autodetect, and kept asm on when nasm existed --
each one removes instrumented code from the edge map. The shared flag set
lives in tools/lib/ffmpeg_config.sh.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
LIB = ROOT / "tools" / "lib" / "ffmpeg_config.sh"
BUILD = ROOT / "tools" / "build_targets.sh"
VENDOR = ROOT / "tools" / "vendor_ffmpeg.sh"

DEP_FLAGS = ("--enable-zlib", "--enable-bzlib", "--enable-lzma", "--enable-libxml2")


def _flags(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Source the lib and print ffmpeg_feature_flags."""
    return subprocess.run(
        ["bash", "-c", f'. "{LIB}" && ffmpeg_feature_flags'],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=30,
    )


def _fake_tool(tmp_path: Path, name: str) -> Path:
    """A tool that always fails: every probe through it reports 'absent'."""
    tool = tmp_path / name
    tool.write_text("#!/bin/sh\nexit 1\n")
    tool.chmod(0o755)
    return tool


def test_regression_parsers_bsfs_not_disabled():
    """The sancov configure line must keep parsers and bsfs."""
    text = BUILD.read_text()

    assert "--disable-parsers" not in text
    assert "--disable-bsfs" not in text


def test_regression_shared_flags_used_and_stamped():
    """Both scripts source the lib; the stamp covers the feature flags."""
    build = BUILD.read_text()

    assert "tools/lib/ffmpeg_config.sh" in build
    assert "lib/ffmpeg_config.sh" in VENDOR.read_text()
    assert '"$FFMPEG_FEATURE_FLAGS"' in build


def test_asm_forced_off():
    """Asm is uninstrumented: always --disable-asm, nasm or not."""
    r = subprocess.run(
        ["bash", "-c", f'. "{LIB}" && echo "$FFMPEG_ASM_FLAG"'],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert r.stdout.strip() == "--disable-asm"


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang required")
def test_present_deps_enabled(tmp_path):
    """Falsification: every dep the host can link is enabled explicitly."""
    have_xml = subprocess.run(["pkg-config", "--exists", "libxml-2.0"]).returncode == 0
    r = _flags({})
    flags = r.stdout.split()

    assert r.returncode == 0
    assert "--disable-autodetect" in flags
    for dep, lib in (("zlib", "-lz"), ("bzlib", "-lbz2"), ("lzma", "-llzma")):
        links = (
            subprocess.run(
                ["clang", "-x", "c", "-", lib, "-o", os.devnull],
                input="int main(void){return 0;}",
                text=True,
                capture_output=True,
            ).returncode
            == 0
        )
        assert (f"--enable-{dep}" in flags) == links
    assert ("--enable-libxml2" in flags) == have_xml


def test_absent_deps_dropped(tmp_path):
    """Adversarial: unlinkable deps are dropped with a warning, never enabled.

    configure aborts on an explicit --enable-X it cannot satisfy, so an
    absent dep must cost that feature, not the whole build.
    """
    cc = _fake_tool(tmp_path, "cc")
    _fake_tool(tmp_path, "pkg-config")
    r = _flags({"FFMPEG_PROBE_CC": str(cc), "PATH": f"{tmp_path}:{os.environ['PATH']}"})
    flags = r.stdout.split()

    assert r.returncode == 0
    assert flags == ["--disable-autodetect"]
    for dep in DEP_FLAGS:
        assert dep.removeprefix("--enable-") in r.stderr
