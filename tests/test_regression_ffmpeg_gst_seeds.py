"""Regression tests for tools/gen_ffmpeg_gst_seeds.py.

The cross-library pack writes FFmpeg seeds (``ff_`` prefix, reused verbatim
from ``gen_ffmpeg_91_seeds``), GStreamer seeds (``gs_``) and one
``dual_`` file that hits both libraries. The FFmpeg builders are covered by
``test_regression_ffmpeg_91_seeds``; here we guard the GStreamer half.

Two properties make a GStreamer seed useful:

*routing* -- the target (``targets/gstreamer_read.c``) type-finds the input
and only then feeds the matching demuxer. A seed whose leading bytes miss
its format's typefind magic falls through to ``kFallback[buf[0] % 5]`` and
lands on the wrong demuxer with a synthesised caps, so its crafted
sub-structure is never parsed.

*reachability* -- each seed targets a demuxer path GStreamer 1.28.0 ships
unguarded and a later 1.28.x hardened. Each test re-parses the produced
bytes exactly as the demuxer reads them (independently of the builder) and
asserts the unguarded path is reached, so a silent offset drift in the
generator fails here rather than producing a seed that no longer lands on
the bug.
"""

from __future__ import annotations

import bz2
import io
import os
import random
import struct

import pytest
import tools.gen_ffmpeg_gst_seeds as gen
from tools.gen_ffmpeg_gst_seeds import BZ2_RNG_SEED, DUAL_SEEDS, GSTREAMER_SEEDS, write_seeds

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

# Intended demuxer per seed, inferred from the format in the seed name -- the
# human-side oracle, independent of the generator's byte layout.
GS_INTENDED = {
    "wav_adtl_null_deref.wav": WAVPARSE,
    "wav_blockalign_zero_take.wav": WAVPARSE,
    "avi_vprp_zero_div.avi": AVIDEMUX,
    "mkv_flac_priv_overread.mkv": MATROSKADEMUX,
    "mkv_bz2_avail_out_overflow.mkv": MATROSKADEMUX,
    "mkv_lzo_getlen_overflow.mkv": MATROSKADEMUX,
    "webm_duration_lzo.webm": MATROSKADEMUX,
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


def simple_block_payload(data: bytes) -> bytes:
    """First SimpleBlock's payload, located via its Cluster id.

    A bare ``data.index(b"\xa3")`` matches the last byte of the EBML magic
    (``1a 45 df a3``); the search must start at the Cluster element.
    """
    cluster_off = data.index(b"\x1f\x43\xb6\x75")  # Cluster id
    blk_off = data.index(b"\xa3", cluster_off)  # SimpleBlock id
    p = io.BytesIO(data[blk_off + 1 :])
    size = ebml_vint(p)
    return p.read(size)[4:]  # track vint + int16 timecode + flags


class TestRouting:
    def test_each_seed_reaches_intended_demuxer(self):
        """Falsification: every gs_/dual_ seed must type-find to its demuxer.

        The committed-then-fixed bug: matroska seeds lacking the EBML magic
        fell to ``FALLBACK[0] == qtdemux``. This asserts the magic routes
        correctly.
        """
        for name, build in {**GSTREAMER_SEEDS, **DUAL_SEEDS}.items():
            assert name in GS_INTENDED, f"no intended demuxer declared for {name}"
            assert route(build()) == GS_INTENDED[name], name

    def test_route_model_has_teeth(self):
        """Control (Hard Rule 46): the route model must misroute broken input.

        Strip a matroska seed's EBML magic and it must NOT still resolve to
        matroskademux -- otherwise the routing assertion above is vacuous.
        """
        good = GSTREAMER_SEEDS["mkv_flac_priv_overread.mkv"]()
        assert route(good) == MATROSKADEMUX

        broken = b"\x00\x00\x00\x15" + good[4:]  # old be32-length prefix, no magic
        assert route(broken) != MATROSKADEMUX
        assert route(broken) == FALLBACK[broken[0] % 5]

    def test_no_accidental_qt_brand_at_offset4(self):
        """Adversarial: a non-qt seed must not carry a qt brand at bytes 4..8.

        A stray ``ftyp``/``moov`` there would hijack typefind to qtdemux.
        """
        for name, build in {**GSTREAMER_SEEDS, **DUAL_SEEDS}.items():
            if GS_INTENDED[name] == QTDEMUX:
                continue
            assert build()[4:8] not in QT_BRANDS, name


class TestMatroskaSeeds:
    def test_flac_priv_declared_len_overreads_payload(self):
        """The STREAMINFO header declares 4 bytes more than the payload holds.

        1.28.0 checks ``off + len > codec_data_size`` but copies ``len + 4``:
        the declared length must sit exactly at the private-data end so the
        check passes while the copy reads past it.
        """
        data = GSTREAMER_SEEDS["mkv_flac_priv_overread.mkv"]()
        assert data[:4] == EBML_MAGIC
        cp_off = data.index(b"\x63\xa2")  # CodecPrivate id (2-byte)
        p = io.BytesIO(data[cp_off + 2 :])
        declared = ebml_vint(p)
        priv = p.read(declared)
        # Falsification: a declared len within the payload would not overread.
        assert priv[:4] == b"fLaC"
        # Last block of the block header: not-last flag, type 0 (STREAMINFO),
        # then the 24-bit length.
        block_len = int.from_bytes(priv[5:8], "big")
        assert block_len == len(priv) - 8 + gen.FLAC_OVERREAD
        assert block_len + 4 > len(priv) - 8

    def test_bz2_block_decompresses_larger_than_input(self):
        """The bz2 block must end the first pass output-full with input left.

        The buggy grow loop only iterates when BZ2_bzDecompress returns
        BZ_OK with the output full; that needs decompressed > compressed.
        """
        data = GSTREAMER_SEEDS["mkv_bz2_avail_out_overflow.mkv"]()
        payload = simple_block_payload(data)
        raw = bz2.decompress(payload)
        # Falsification: a well-compressing block would finish in one pass.
        assert len(raw) > len(payload)
        rng = random.Random(BZ2_RNG_SEED)
        assert raw == bytes(rng.randrange(4) for _ in range(len(raw)))

    def test_lzo_zero_run_wraps_int32(self):
        """The length-run of zeros must overflow get_len's int counter.

        1.28.0's ``cnt += 255`` runs until the zeros end; only a run past
        2**31/255 bytes wraps cnt negative. The trailing 0x01 ends the run.
        """
        data = GSTREAMER_SEEDS["mkv_lzo_getlen_overflow.mkv"]()
        payload = simple_block_payload(data)
        # Falsification: a short run stays positive and takes the clamps.
        assert payload[: payload.index(b"\x01")] == b"\x00" * gen.LZO_ZERO_RUN
        assert gen.LZO_ZERO_RUN * 255 > gen.INT32_MAX
        assert payload.endswith(b"\x01")


class TestWavSeed:
    def test_adtl_list_declares_absent_body(self):
        """LIST/adtl whose declared size leaves no room for any sub-chunk.

        1.28.0 flushes the 12-byte header and maps ``size - 4`` bytes; with
        size == 12 the 8 mapped body bytes never arrive -> NULL map -> the
        deref in gst_wavparse_adtl_chunk.
        """
        data = GSTREAMER_SEEDS["wav_adtl_null_deref.wav"]()
        assert data[0:4] == b"RIFF"
        assert data[8:12] == b"WAVE"
        # Falsification: a body-carrying LIST would map real bytes.
        adtl_off = data.index(b"adtl")
        list_size = struct.unpack("<I", data[adtl_off - 4 : adtl_off])[0]
        assert list_size == gen.WAV_ADTL_LIST_SIZE
        assert len(data) == adtl_off + 4  # nothing after the "adtl" fourcc


class TestWavBlockalignSeed:
    def test_fmt_triggers_the_round_down_to_zero(self):
        """blockalign must exceed both the 40 ms budget and the 4096 floor.

        wavparse computes max_buf_size = MAX(0.04 * bps, 4096) then applies
        ``max_buf_size -= max_buf_size % blockalign``; only a blockalign
        above BOTH values rounds the buffer size down to 0, which is what
        sends desired = 0 into gst_adapter_take_buffer.
        """
        data = GSTREAMER_SEEDS["wav_blockalign_zero_take.wav"]()
        assert data[0:4] == b"RIFF"
        assert data[8:12] == b"WAVE"
        fmt_off = data.index(b"fmt ") + 8
        tag, ch, rate, av_bps, blockalign, bits = struct.unpack(
            "<HHIIHH", data[fmt_off : fmt_off + 16]
        )
        # Falsification: PCM rejects an inflated blockalign at fmt sanity
        # (blockalign > channels * ceil(bits/8) -> invalid_blockalign), so
        # the seed must carry a non-PCM tag.
        assert tag != 1
        assert tag == gen.WAV_FMT_VOXWARE_AC8
        budget = gen.GST_BUF_TIME_MS / 1000 * av_bps
        assert blockalign > max(budget, gen.GST_MAX_BUFFER_SIZE)
        # The default fmt branch also requires av_bps <= blockalign * rate.
        assert av_bps <= blockalign * rate

    def test_declared_data_beats_the_dataleft_bail(self):
        """Declared data size must reach blockalign or wavparse bails first.

        ``dataleft < blockalign`` jumps to found_eos before the adapter
        take; the declared size must exceed blockalign while the actual
        payload stays tiny.
        """
        data = GSTREAMER_SEEDS["wav_blockalign_zero_take.wav"]()
        data_off = data.index(b"data")
        declared = struct.unpack("<I", data[data_off + 4 : data_off + 8])[0]
        assert declared >= gen.WAV_BLOCKALIGN
        assert len(data) - (data_off + 8) == gen.WAV_DATA_PRESENT


class TestAviSeed:
    def test_vprp_carries_zero_fields_and_fixed_prefix(self):
        """vprp sized exactly to offsetof(field_info) with fields == 0.

        1.28.0 computes ``(size - offsetof(field_info)) / vprp->fields``;
        the 36-byte body passes its size check and divides by zero.
        """
        data = GSTREAMER_SEEDS["avi_vprp_zero_div.avi"]()
        assert data[0:4] == b"RIFF"
        # Falsification: "AVI" alone (old bug) does not type-find to avidemux.
        assert data[8:12] == b"AVI "
        for marker in (b"hdrl", b"avih", b"strl", b"strh", b"strf", b"movi"):
            assert marker in data, marker
        vprp_off = data.index(b"vprp")
        size = struct.unpack("<I", data[vprp_off + 4 : vprp_off + 8])[0]
        body = data[vprp_off + 8 : vprp_off + 8 + size]
        assert size == 36  # exactly offsetof(gst_riff_vprp, field_info)
        # Falsification: fields != 0 takes the division.
        assert struct.unpack("<I", body[32:36])[0] == 0


class TestDualSeed:
    def test_webm_carries_both_legs(self):
        """One file, both libraries: Duration 1e18 (FFmpeg) + LZO run (gst).

        Falsification: either half alone would crash only one target.
        """
        data = DUAL_SEEDS["webm_duration_lzo.webm"]()
        assert data[:4] == EBML_MAGIC
        off = data.index(b"\x44\x89")  # Duration id (2-byte)
        p = io.BytesIO(data[off + 2 :])
        assert ebml_vint(p) == 8
        d = struct.unpack(">d", p.read(8))[0]
        # 1e18 * 1e6 (time scale) * 1000 / 1e6 = 1e21 > INT64_MAX.
        assert d == gen.WEBM_DURATION
        assert d * 1e6 * 1000 / 1e6 > 2**63 - 1
        # LZO leg: same zero-run payload as the gst lzo seed.
        payload = simple_block_payload(data)
        assert payload[: gen.LZO_ZERO_RUN] == b"\x00" * gen.LZO_ZERO_RUN

    def test_webm_doctype_routes_to_matroskademux(self):
        """Adversarial: doctype webm must still type-find to matroskademux."""
        data = DUAL_SEEDS["webm_duration_lzo.webm"]()
        assert b"webm" in data
        assert route(data) == MATROSKADEMUX


class TestFfImport:
    def test_ff_group_is_the_91_pack_verbatim(self):
        """The ff_ group imports the 91 pack's builders -- no copy, no drift."""
        from tools.gen_ffmpeg_91_seeds import SEEDS

        assert gen.FFMPEG_SEEDS is SEEDS


def _stray(dir_path):
    """Temp-staging leftovers that load_corpus must never see."""
    return [n for n in os.listdir(dir_path) if n.endswith(".part") or ".part." in n]


class TestAtomicPublish:
    def test_all_prefixes_published(self, tmp_path):
        """Falsification: --target both writes every ff_, gs_ and dual_ seed."""
        from tools.gen_ffmpeg_gst_seeds import FFMPEG_SEEDS

        seeds = tmp_path / "seeds"
        seeds.mkdir()

        write_seeds(str(seeds), FFMPEG_SEEDS, prefix="ff_")
        write_seeds(str(seeds), GSTREAMER_SEEDS, prefix="gs_")
        write_seeds(str(seeds), DUAL_SEEDS, prefix="dual_")

        for group, prefix in (
            (FFMPEG_SEEDS, "ff_"),
            (GSTREAMER_SEEDS, "gs_"),
            (DUAL_SEEDS, "dual_"),
        ):
            for name, build in group.items():
                assert (seeds / f"{prefix}{name}").read_bytes() == build()

        expected = (
            {f"ff_{n}" for n in FFMPEG_SEEDS}
            | {f"gs_{n}" for n in GSTREAMER_SEEDS}
            | {f"dual_{n}" for n in DUAL_SEEDS}
        )
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
