"""``targets/ffmpeg_read.c`` must pass a filename to ``avformat_open_input``.

The harness fed ffmpeg through an in-memory ``AVIOContext`` and passed
``NULL`` as the url, so ``AVProbeData.filename`` was empty on every single
execution. Demuxer selection is by content probe *and* filename, and a
demuxer that is gated on the extension is then unreachable no matter what
bytes the campaign produces. Measured against the vendored FFmpeg 9.0.1:

*   27 demuxers have no ``read_probe`` at all -- ``g722``, ``g729``, ``sbc``,
    ``rawvideo``, the ``s16le``/``u8``/``alaw`` raw family. The extension is
    their only selector, so they had never executed a single time.
*   172 of 355 demuxers declare extensions and only 8 declare a mime type,
    so the filename is the only hint that reaches the rest.
*   ``hls`` is the motivating case: ``hls_probe`` returns 0 unless the
    filename matches ``m3u8``/``m3u`` even when the playlist tags are
    present, so every playlist input was rejected before ``read_header``.

The failure mode is silent. The campaign runs, reports edges, and finds
crashes -- it simply never touches a whole class of demuxers, and nothing in
the output says so.

Three properties are load-bearing and each has its own test:

1.  A filename hint reaches ``avformat_open_input``.
2.  The hint is not derived from the input bytes. Content-derived selection
    would let a one-byte mutation change which demuxer runs, destabilising
    the edge signal the scheduler reads. ``FUZZ_FFMPEG_EXT`` pins an
    extension per campaign (the idiom upstream uses, one
    ``target_dem_fuzzer`` per demuxer) and the sniff table stays tiny.
3.  Sub-resource I/O stays blocked. Once a playlist demuxer parses segment
    URLs it will try to open them relative to our url, through the file
    protocol. An empty ``protocol_whitelist`` keeps the target hermetic;
    without it an input could pull an arbitrary local path into the run and
    coverage would depend on the filesystem rather than on the bytes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "ffmpeg_read.c"


@pytest.fixture(scope="module")
def target_src() -> str:
    return TARGET.read_text()


def test_open_input_receives_a_filename(target_src: str) -> None:
    """A NULL url here silently removes every extension-gated demuxer."""
    calls = re.findall(r"avformat_open_input\(([^;]*)\)", target_src, re.S)
    assert calls, "no avformat_open_input call found in the ffmpeg target"
    for args in calls:
        url_arg = args.split(",")[1].strip()
        assert url_arg != "NULL", (
            "avformat_open_input must receive a filename, not NULL: a NULL url "
            "leaves AVProbeData.filename empty and makes every demuxer that is "
            "gated on an extension (27 have no read_probe at all) unreachable."
        )


def test_hint_is_not_derived_from_input_bytes(target_src: str) -> None:
    """The extension must come from the environment or a fixed table.

    A tail/footer protocol was tried once (285d0fa) and reverted. Beyond that
    history, any content-derived extension makes demuxer choice a function of
    the mutated bytes, so a single flipped bit moves the input to a different
    demuxer and the edge signal stops being comparable across executions.
    """
    assert "FUZZ_FFMPEG_EXT" in target_src, (
        "the per-campaign extension override is how the headerless raw "
        "formats (g722, s16le, rawvideo) are reached -- they have no magic "
        "bytes, so no sniff can select them"
    )
    build_url = re.search(
        r"fuzz_build_url\(char \*out.*?\n\}", target_src, re.S
    )
    assert build_url, "fuzz_build_url() not found"
    body = build_url.group(0)
    assert "fuzz_ext_override()" in body and "fuzz_sniff_ext(" in body, (
        "the hint must come from the env override or the magic table only"
    )


def test_sniff_table_stays_small(target_src: str) -> None:
    """Every entry silently redirects which demuxer an input reaches."""
    sniff = re.search(r"fuzz_sniff_ext\(.*?\n\}", target_src, re.S)
    assert sniff, "fuzz_sniff_ext() not found"
    entries = re.findall(r"memcmp\(buf,", sniff.group(0))
    assert entries, "the sniff table has no entries"
    assert len(entries) <= 8, (
        "keep the magic table small and auditable: an over-eager match moves "
        "inputs off demuxers they currently reach, which loses coverage "
        "without any visible symptom"
    )


def test_sub_resource_io_is_blocked(target_src: str) -> None:
    """A parsed playlist must not be able to open local files."""
    assert re.search(
        r'av_dict_set\(&opts,\s*"protocol_whitelist",\s*""', target_src
    ), (
        "protocol_whitelist must be empty: once the filename hint is live a "
        "playlist demuxer opens the segments it parsed, relative to our url "
        "and through the file protocol"
    )


def test_playlist_reload_loops_are_bounded(target_src: str) -> None:
    """Unbounded reloads cost 10 s per input and report as a timeout.

    Measured on the vendored 9.0.1 with a three-line playlist: 10026 ms with
    the stock ``max_reload`` default, 8 ms with it pinned to 0. The interrupt
    callback does not cover this -- the wait is an ``av_usleep()`` the
    demuxer never polls through.
    """
    assert re.search(r'av_dict_set\(&opts,\s*"max_reload",\s*"0"', target_src)
    assert re.search(
        r'av_dict_set\(&opts,\s*"m3u8_hold_counters",\s*"0"', target_src
    )


def test_interrupt_callback_is_armed_per_execution(target_src: str) -> None:
    """A stale budget would leave later executions with no escape hatch."""
    assert "fuzz_interrupt_cb" in target_src
    assert re.search(
        r"fuzz_interrupt_budget\s*=\s*FUZZ_INTERRUPT_BUDGET", target_src
    ), "the budget must be reset on every open, not just initialised once"
