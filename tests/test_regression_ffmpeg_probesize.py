"""``targets/ffmpeg_read.c`` bounds find_stream_info by a small probesize.

FFmpeg's wav demuxer asks to probe every PCM_S16LE stream (``wavdec.c``
``handle_stream_probing``): it buffers up to 32 packets (128 KB) and re-runs
every demuxer's probe at each power-of-two size and at the end. The MP3 probe
scans every byte offset, so one wav seed spent 8.4 of 8.7 ms/call in
``avformat_find_stream_info`` (vendored 9.0.2, sancov). The probe ends early
once ``raw_packet_buffer_size >= probesize``, so a 64 KiB ``probesize`` cut it
to 3.0 ms and 32 KiB to 1.3 ms. Container detection uses ``format_probesize``,
a separate field, which must stay at its default.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "ffmpeg_read.c"


@pytest.fixture(scope="module")
def src() -> str:
    return TARGET.read_text()


def test_probesize_macro_defined_small(src: str) -> None:
    m = re.search(r"^#\s*define\s+FFMPEG_PROBESIZE\s+\((\d+)\s*\*\s*1024\)", src, re.M)
    assert m, "FFMPEG_PROBESIZE must be defined as (N * 1024)"
    assert 1 <= int(m.group(1)) <= 128, (
        "above 128 KiB the wav probe buffer cap (32 packets) wins and nothing is gained"
    )


def test_probesize_overridable_and_guarded(src: str) -> None:
    assert re.search(r"#ifndef\s+FFMPEG_PROBESIZE", src)
    assert re.search(r"#if\s+FFMPEG_PROBESIZE\s*<\s*1\b", src), (
        "a zero probesize makes every stream_info fail"
    )


def test_probesize_applied_to_context(src: str) -> None:
    assert re.search(r"fmt_ctx->probesize\s*=\s*FFMPEG_PROBESIZE\s*;", src)
    assert not re.search(r"fmt_ctx->probesize\s*=\s*1\s*\*\s*1024\s*\*\s*1024", src), (
        "old 1 MB literal left behind"
    )


def test_format_probesize_untouched(src: str) -> None:
    """Falsification: container detection must keep its default window, or
    large-header formats stop being recognised at all."""
    assert not re.search(r"format_probesize\s*=", src)
