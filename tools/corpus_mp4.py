#!/usr/bin/env python3
"""Generate a small MP4/MOV/3GP seed corpus with the local ffmpeg (no download).

An empty-corpus GStreamer run reached ~80% of a real MP4's qtdemux edges and
never built a sample table; these seeds start campaigns past that gap. One
tiny file per container x codec x layout: avc1, hvc1, mp4v, h263, mjpeg,
aac, alac, ac3, mp3, opus, lpcm; faststart, fragmented, edit list.

Usage:
    python tools/corpus_mp4.py [--out DIR]

Seeds go to <out>/seeds/ (the layout load_corpus reads), named mp4_<variant>.
Variants whose encoder this ffmpeg lacks are skipped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Make the `tools` namespace package importable when run as a script
# (cwd-independent), as in tools/gen_ffmpeg_gst_seeds.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.gen_ffmpeg_gst_seeds import write_seeds  # noqa: E402

DEFAULT_OUT = os.path.expanduser("~/fuzzing/corpus_mp4")
PREFIX = "mp4_"
MAX_SEED_BYTES = 16384  # well under the fuzzer's -m on any campaign here
FFMPEG_TIMEOUT_S = 60
VIDEO = ("-f", "lavfi", "-i", "testsrc=size=16x16:rate=2:duration=1")
H263_VIDEO = ("-f", "lavfi", "-i", "testsrc=size=128x96:rate=2:duration=1")  # h263 sizes only
AUDIO = ("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.1")


@dataclass(frozen=True)
class Variant:
    """One seed: container format, codecs (None = no stream), extra muxer args."""

    name: str
    fmt: str
    vcodec: str | None = None
    acodec: str | None = None
    extra: tuple[str, ...] = field(default_factory=tuple)


VARIANTS: tuple[Variant, ...] = (
    Variant("mp4v", "mp4", vcodec="mpeg4"),
    Variant("avc1", "mp4", vcodec="libx264"),
    Variant("hvc1", "mp4", vcodec="libx265", extra=("-tag:v", "hvc1")),
    Variant("avc1_aac", "mp4", vcodec="libx264", acodec="aac"),
    Variant("aac", "mp4", acodec="aac"),
    Variant("ac3", "mp4", acodec="ac3"),
    Variant("mp3", "mp4", acodec="libmp3lame"),
    Variant("opus", "mp4", acodec="libopus"),
    Variant("faststart", "mp4", vcodec="libx264", acodec="aac", extra=("-movflags", "+faststart")),
    Variant(
        "frag",
        "mp4",
        vcodec="libx264",
        extra=("-movflags", "frag_keyframe+empty_moov+default_base_moof"),
    ),
    Variant(
        "editlist", "mp4", vcodec="mpeg4", acodec="aac", extra=("-use_editlist", "1", "-ss", "0.05")
    ),
    Variant("mov_mp4v", "mov", vcodec="mpeg4"),
    Variant("mov_mjpeg", "mov", vcodec="mjpeg"),
    Variant("mov_alac", "mov", acodec="alac"),
    Variant("mov_lpcm", "mov", acodec="pcm_s16be"),
    Variant("mov_avc1_aac", "mov", vcodec="libx264", acodec="aac"),
    Variant("3gp_h263_aac", "3gp", vcodec="h263", acodec="aac"),
)


def needed_encoders() -> frozenset[str]:
    """Every encoder the variant table uses."""
    return frozenset(c for v in VARIANTS for c in (v.vcodec, v.acodec) if c)


def _encoders() -> frozenset[str]:
    """Encoder names this ffmpeg offers; empty if ffmpeg is missing."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_S,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    return frozenset(parts[1] for line in out.splitlines() if len(parts := line.split()) > 1)


def _args(v: Variant, out: Path) -> list[str]:
    """ffmpeg command for *v*: inputs, one frame / 0.1 s, codecs, muxer flags."""
    cmd = ["ffmpeg", "-loglevel", "error", "-y"]
    if v.vcodec:
        cmd += H263_VIDEO if v.vcodec == "h263" else VIDEO
    if v.acodec:
        cmd += AUDIO

    if v.vcodec:
        cmd += ["-c:v", v.vcodec, "-frames:v", "1"]
    if v.acodec:
        cmd += ["-c:a", v.acodec, "-ac", "1"]
    return [*cmd, *v.extra, "-f", v.fmt, str(out)]


def _encode(v: Variant, out: Path) -> bytes | None:
    """Encode *v* to *out*; its bytes, or None when ffmpeg fails."""
    try:
        done = subprocess.run(
            _args(v, out), capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0 or not out.exists():
        return None
    return out.read_bytes()


def generate(seeds_dir: Path) -> list[Path]:
    """Write every available, bounded variant to *seeds_dir*; their paths."""
    seeds_dir.mkdir(parents=True, exist_ok=True)
    have = _encoders()
    written: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        for v in VARIANTS:
            if not all(c in have for c in (v.vcodec, v.acodec) if c):
                print(f"skip {v.name}: encoder missing")
                continue

            data = _encode(v, Path(tmp) / f"{v.name}.{v.fmt}")
            if not data or len(data) > MAX_SEED_BYTES:
                print(f"skip {v.name}: encode failed or > {MAX_SEED_BYTES} B")
                continue

            write_seeds(str(seeds_dir), {v.name: lambda d=data: d}, prefix=PREFIX)
            written.append(seeds_dir / f"{PREFIX}{v.name}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="corpus root; seeds go to <out>/seeds/ (default: %(default)s)",
    )
    args = ap.parse_args()

    written = generate(Path(args.out) / "seeds")
    print(f"[*] {len(written)} MP4-family seeds in {args.out}/seeds/")


if __name__ == "__main__":
    main()
