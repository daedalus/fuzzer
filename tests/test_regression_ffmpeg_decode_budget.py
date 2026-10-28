"""``targets/ffmpeg_read.c`` bounds decode work by pixels, not frame count.

The driver decoded up to 500 frames per call whatever their size, so cost grew
with resolution x frames. Measured against FFmpeg 6.1 on a 2 s clip: 320x240
6 ms/call, 720p 39 ms, 1080p 81 ms. Capping at 8 frames took 720p to 6.6 ms.
It also called ``malloc_trim(0)`` on every call (~1.3 ms of ~5 ms) and set no
``max_pixels``, so a fuzzed header could demand a huge frame.

Three properties, one test group each: a pixel budget ends the decode loop,
``malloc_trim`` runs every N calls, and decoders are opened with ``max_pixels``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "ffmpeg_read.c"


@pytest.fixture(scope="module")
def src() -> str:
    return TARGET.read_text()


def _define(src: str, name: str) -> str:
    m = re.search(rf"^#\s*define\s+{name}\s+(.+)$", src, re.M)
    assert m, f"{name} is not defined in the ffmpeg target"
    return m.group(1)


def test_pixel_budget_constant_exists(src: str) -> None:
    assert _define(src, "FUZZ_PIXEL_BUDGET")


def test_decode_loop_stops_on_pixel_budget(src: str) -> None:
    """The loop must compare spent pixels with the budget and break."""
    loop = src[src.index("av_read_frame(fmt_ctx") :]
    assert re.search(r"FUZZ_PIXEL_BUDGET", loop), "budget never checked in loop"


def test_frame_cap_is_kept(src: str) -> None:
    """Falsification: the budget must not replace the packet/frame caps, which
    still bound audio and tiny-frame streams that add no pixels."""
    assert re.search(r"total_packets\s*>\s*\d+\s*\|\|\s*total_frames\s*>\s*\d+", src)


def test_budget_guarded_against_nonpositive(src: str) -> None:
    """Adversarial: -D...=0 would end every decode after one frame silently."""
    for name in ("FUZZ_PIXEL_BUDGET", "FUZZ_TRIM_EVERY", "FUZZ_MAX_FRAME_PIXELS"):
        assert re.search(rf"#if\s+{name}\s*<\s*1\b", src), f"no #error guard for {name}"


def _pixels_fn(src: str) -> str:
    m = re.search(
        r"static\s+(?:inline\s+)?int64_t\s+fuzz_frame_pixels\([^)]*\)\s*\{.*?\n\}", src, re.S
    )
    assert m, "fuzz_frame_pixels() not found"
    return m.group(0)


@pytest.mark.skipif(shutil.which("clang") is None, reason="needs clang")
def test_frame_pixels_behaviour(src: str, tmp_path: Path) -> None:
    """Rule 49: compile the helper alone. Overflow and bad sizes must not wrap."""
    prog = tmp_path / "t.c"
    prog.write_text(
        "#include <stdint.h>\n#include <stdio.h>\n#include <limits.h>\n"
        + _pixels_fn(src)
        + "\nint main(void){\n"
        '  printf("%lld %lld %lld %lld %lld\\n",\n'
        "    (long long)fuzz_frame_pixels(1920,1080),\n"
        "    (long long)fuzz_frame_pixels(0,1080),\n"
        "    (long long)fuzz_frame_pixels(-5,100),\n"
        "    (long long)fuzz_frame_pixels(INT_MAX,INT_MAX),\n"
        "    (long long)fuzz_frame_pixels(1,1));\n  return 0;}\n"
    )
    exe = tmp_path / "t"
    subprocess.run(["clang", "-O1", "-o", str(exe), str(prog)], check=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split()
    assert out == ["2073600", "0", "0", str((2**31 - 1) ** 2), "1"]


def test_trim_is_not_unconditional(src: str) -> None:
    """malloc_trim(0) on every call cost ~25% of a small-seed execution."""
    calls = list(re.finditer(r"^\s*malloc_trim\(0\);", src, re.M))
    assert calls, "malloc_trim call not found"
    for m in calls:
        before = src[max(0, m.start() - 200) : m.start()]
        assert "FUZZ_TRIM_EVERY" in before, "malloc_trim must be gated by FUZZ_TRIM_EVERY"


def test_trim_gate_uses_modulus(src: str) -> None:
    assert re.search(r"%\s*FUZZ_TRIM_EVERY", src)


def test_max_pixels_set_before_open(src: str) -> None:
    set_at = src.find("max_pixels")
    open_at = src.find("avcodec_open2(slot->ctx")
    assert set_at != -1, "decoder context never gets max_pixels"
    assert set_at < open_at, "max_pixels must be set before avcodec_open2"
