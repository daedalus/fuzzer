"""Regression tests for tools/gen_ffmpeg_gst_seeds.py.

The dual-target pack writes FFmpeg seeds (``ff_`` prefix, reused verbatim from
``gen_ffmpeg_91_seeds``) and GStreamer seeds (``gs_`` prefix). The FFmpeg
builders are covered by ``test_regression_ffmpeg_91_seeds``; here we guard the
GStreamer half.

The one property that makes a GStreamer seed useful is *routing*: the target
(``targets/gstreamer_read.c``) type-finds the input and only then feeds the
matching demuxer. A seed whose leading bytes miss its format's typefind magic
falls through to ``kFallback[buf[0] % 5]`` and lands on the wrong demuxer with a
synthesised caps, so its crafted sub-structure is never parsed. Each test below
re-derives the route from the bytes (independently of the builder) and asserts
it reaches the intended demuxer, then checks the crafted sub-structure.
"""

from __future__ import annotations

import io
import math
import os
import struct

import pytest
import tools.gen_ffmpeg_gst_seeds as gen
from tools.gen_ffmpeg_gst_seeds import GSTREAMER_SEEDS, write_seeds

# --- Routing model: mirrors targets/gstreamer_read.c ------------------------
# typefind magic -> caps name, caps name -> demuxer element, and the byte[0]%5
# fallback table. These are the GStreamer facts the target relies on; the test
# encodes them independently of how the generator builds its bytes.
EBML_MAGIC = b"\x1a\x45\xdf\xa3"
QT_BRANDS = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"uuid"}

MATROSKADEMUX = "matroskademux"
QTDEMUX = "qtdemux"
WAVPARSE = "wavparse"
AVIDEMUX = "avidemux"
FLVDEMUX = "flvdemux"

FALLBACK = (QTDEMUX, MATROSKADEMUX, WAVPARSE, AVIDEMUX, FLVDEMUX)

# Intended demuxer per seed, inferred from the format in the seed name — the
# human-side oracle, independent of the generator's byte layout.
GS_INTENDED = {
    "matroska_duration_inf.mkv": MATROSKADEMUX,
    "matroska_video_dims.mkv": MATROSKADEMUX,
    "matroska_codecprivate.mkv": MATROSKADEMUX,
    "quicktime_mvhd.mp4": QTDEMUX,
    "wav_missing_fmt.wav": WAVPARSE,
    "avi_bad_avih.avi": AVIDEMUX,
    "flv_nan_duration.flv": FLVDEMUX,
}


def typefind(data: bytes) -> str | None:
    """Caps name GStreamer's typefinders assign, for the formats we emit."""
    if data[:4] == EBML_MAGIC:
        return "video/x-matroska"
    if data[:4] == b"RIFF":
        if data[8:12] == b"WAVE":
            return "audio/x-wav"
        if data[8:12] == b"AVI ":
            return "video/x-msvideo"
    if data[:3] == b"FLV":
        return "video/x-flv"
    if data[4:8] in QT_BRANDS:
        return "video/quicktime"
    return None


CAPS_TO_ELEM = {
    "video/quicktime": QTDEMUX,
    "video/x-matroska": MATROSKADEMUX,
    "video/webm": MATROSKADEMUX,
    "audio/x-wav": WAVPARSE,
    "video/x-msvideo": AVIDEMUX,
    "video/x-flv": FLVDEMUX,
}


def route(data: bytes) -> str:
    """Demuxer the target picks: typefind match, else byte[0]%5 fallback."""
    name = typefind(data)
    if name:
        return CAPS_TO_ELEM[name]
    return FALLBACK[data[0] % 5]


# EBML / Matroska element ids (spec-level, shared with FFmpeg's matroskadec).
def ebml_vint(p: io.BytesIO) -> int:
    first = p.read(1)[0]
    length = 1
    mask = 0x80
    while not first & mask:
        length += 1
        mask >>= 1
    value = first & (mask - 1)
    for _ in range(length - 1):
        value = (value << 8) | p.read(1)[0]
    return value


class TestRouting:
    def test_each_seed_reaches_intended_demuxer(self):
        """Falsification: every gs_ seed must type-find to its format's demuxer.

        The committed-then-fixed bug: matroska seeds lacking the EBML magic fell
        to ``FALLBACK[0] == qtdemux``. This asserts the magic routes correctly.
        """
        for name, build in GSTREAMER_SEEDS.items():
            assert name in GS_INTENDED, f"no intended demuxer declared for {name}"
            assert route(build()) == GS_INTENDED[name], name

    def test_route_model_has_teeth(self):
        """Control (Hard Rule 46): the route model must misroute broken input.

        Strip a matroska seed's EBML magic and it must NOT still resolve to
        matroskademux -- otherwise the routing assertion above is vacuous.
        """
        good = GSTREAMER_SEEDS["matroska_duration_inf.mkv"]()
        assert route(good) == MATROSKADEMUX

        broken = b"\x00\x00\x00\x15" + good[4:]  # old be32-length prefix, no magic
        assert route(broken) != MATROSKADEMUX
        assert route(broken) == FALLBACK[broken[0] % 5]

    def test_no_accidental_qt_brand_at_offset4(self):
        """Adversarial: a non-qt seed must not carry a qt brand at bytes 4..8.

        A stray ``ftyp``/``moov`` there would hijack typefind to qtdemux.
        """
        for name, build in GSTREAMER_SEEDS.items():
            if GS_INTENDED[name] == QTDEMUX:
                continue
            assert build()[4:8] not in QT_BRANDS, name


class TestMatroskaSeeds:
    def test_duration_inf_double(self):
        """Info.Duration is a non-finite double -> parse_info float handling."""
        data = GSTREAMER_SEEDS["matroska_duration_inf.mkv"]()
        assert data[:4] == EBML_MAGIC
        assert b"\x15\x49\xa9\x66" in data  # Info id
        off = data.index(b"\x44\x89")  # Duration id (2-byte)
        p = io.BytesIO(data[off + 2 :])
        assert ebml_vint(p) == 8
        d = struct.unpack(">d", p.read(8))[0]
        # Falsification: a finite double would take the normal path.
        assert math.isinf(d) or math.isnan(d)

    def test_video_dims_huge(self):
        """PixelWidth*PixelHeight overflows -> video track size computation."""
        data = GSTREAMER_SEEDS["matroska_video_dims.mkv"]()
        assert data[:4] == EBML_MAGIC
        assert b"\x16\x54\xae\x6b" in data  # Tracks id
        w_off = data.index(b"\xb0")  # PixelWidth id
        p = io.BytesIO(data[w_off + 1 :])
        n = ebml_vint(p)
        width = int.from_bytes(p.read(n), "big")
        # Falsification: a sane width wouldn't stress the size math.
        assert width >= 0xFFFF

    def test_codecprivate_claims_more_than_present(self):
        """CodecPrivate size vint exceeds the bytes present -> truncation path."""
        data = GSTREAMER_SEEDS["matroska_codecprivate.mkv"]()
        assert data[:4] == EBML_MAGIC
        cp_off = data.index(b"\x63\xa2")  # CodecPrivate id (2-byte)
        p = io.BytesIO(data[cp_off + 2 :])
        declared = ebml_vint(p)
        remaining = len(data) - (cp_off + 2) - (p.tell())
        # Falsification: the declared length must outrun the actual payload.
        assert declared > remaining


class TestQuickTimeSeed:
    def test_ftyp_at_offset4_and_moov(self):
        """ftyp at byte 4 (typefind) and a moov box for qtdemux to parse."""
        data = GSTREAMER_SEEDS["quicktime_mvhd.mp4"]()
        # Falsification: a prefix byte would shift ftyp off offset 4 (old bug).
        assert data[4:8] == b"ftyp"
        size = struct.unpack(">I", data[0:4])[0]
        assert size >= 8 and size % 4 == 0
        assert b"moov" in data
        assert b"mvhd" in data

    def test_no_leading_prefix_byte(self):
        """Adversarial: byte 0 is part of the box size, not a routing selector."""
        data = GSTREAMER_SEEDS["quicktime_mvhd.mp4"]()
        assert data[0] == 0x00  # high byte of a small 32-bit box size


class TestWavSeed:
    def test_riff_wave_without_fmt(self):
        """RIFF/WAVE present (routes to wavparse); fmt chunk absent (error path)."""
        data = GSTREAMER_SEEDS["wav_missing_fmt.wav"]()
        assert data[0:4] == b"RIFF"
        assert data[8:12] == b"WAVE"
        # Falsification: a fmt chunk would take the normal negotiation path.
        assert b"fmt " not in data


class TestAviSeed:
    def test_riff_avi_container_with_avih(self):
        """RIFF/AVI container (routes to avidemux) carrying a malformed avih."""
        data = GSTREAMER_SEEDS["avi_bad_avih.avi"]()
        assert data[0:4] == b"RIFF"
        # Falsification: the old seed used bare b"AVI"; typefind needs "AVI ".
        assert data[8:12] == b"AVI "
        assert b"hdrl" in data
        assert b"avih" in data


class TestFlvSeed:
    def test_flv_magic_and_nan_duration(self):
        """FLV magic (routes to flvdemux) with a NaN AMF duration double."""
        data = GSTREAMER_SEEDS["flv_nan_duration.flv"]()
        assert data[0:3] == b"FLV"
        assert b"duration" in data
        nan_off = data.index(b"duration") + len("duration") + 1
        num = struct.unpack(">d", data[nan_off : nan_off + 8])[0]
        # Falsification: a finite duration wouldn't exercise the NaN path.
        assert math.isnan(num)


def _stray(dir_path):
    """Temp-staging leftovers that load_corpus must never see."""
    return [n for n in os.listdir(dir_path) if n.endswith(".part") or ".part." in n]


class TestAtomicPublish:
    def test_both_prefixes_published(self, tmp_path):
        """Falsification: --target both writes every ff_ and gs_ seed, no temp."""
        from tools.gen_ffmpeg_gst_seeds import FFMPEG_SEEDS

        seeds = tmp_path / "seeds"
        seeds.mkdir()

        write_seeds(str(seeds), FFMPEG_SEEDS, prefix="ff_")
        write_seeds(str(seeds), GSTREAMER_SEEDS, prefix="gs_")

        for name, build in FFMPEG_SEEDS.items():
            assert (seeds / f"ff_{name}").read_bytes() == build()
        for name, build in GSTREAMER_SEEDS.items():
            assert (seeds / f"gs_{name}").read_bytes() == build()

        expected = {f"ff_{n}" for n in FFMPEG_SEEDS} | {f"gs_{n}" for n in GSTREAMER_SEEDS}
        assert set(os.listdir(seeds)) == expected
        assert _stray(seeds) == []
        assert _stray(tmp_path) == []

    def test_interrupt_leaves_no_temp(self, tmp_path, monkeypatch):
        """Adversarial: a failed publish leaves no temp and no partial in seeds/."""
        seeds = tmp_path / "seeds"
        seeds.mkdir()

        real_replace = os.replace
        calls = {"n": 0}

        def flaky_replace(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("boom")
            return real_replace(src, dst)

        monkeypatch.setattr(gen.os, "replace", flaky_replace)

        with pytest.raises(OSError, match="boom"):
            write_seeds(str(seeds), GSTREAMER_SEEDS, prefix="gs_")

        names = list(GSTREAMER_SEEDS)
        assert (seeds / f"gs_{names[0]}").exists()
        assert not (seeds / f"gs_{names[1]}").exists()
        assert _stray(seeds) == []
        assert _stray(tmp_path) == []
