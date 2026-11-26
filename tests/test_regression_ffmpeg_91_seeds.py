"""Regression tests for tools/gen_ffmpeg_91_seeds.py.

The seeds target demuxer paths FFmpeg 9.0.2 ships unguarded and 9.1 hardened.
Each test re-parses the produced bytes exactly as the 9.0.2 demuxer reads them
(independently of the builder) and asserts the unguarded arithmetic is reached,
so a silent offset drift in the generator fails here rather than producing a
seed that no longer lands on the bug.
"""

from __future__ import annotations

import io
import math
import os
import struct

import pytest
import tools.gen_ffmpeg_91_seeds as gen
from tools.gen_ffmpeg_91_seeds import (
    BEXT_FIXED_SIZE,
    INT64_MAX,
    SEEDS,
    caf_pakt_overflow,
    flv_amf_nan_duration,
    rf64_dataend_overflow,
    rpl_idx_overflow,
    sbg_nan_cast,
    wav_bext_bigalloc,
    webm_duration_cast,
    write_seeds,
)

# version field offset inside the 602-byte BEXT prologue (wav_parse_bext_tag)
BEXT_VERSION_OFF = 346

# matroska_read_header defaults the time scale to 1e6 when Info has no
# TimecodeScale element; the duration product uses it and AV_TIME_BASE.
MK_DEFAULT_TIME_SCALE = 1_000_000
AV_TIME_BASE = 1_000_000


def ebml_id(p, n=4):
    """Read one EBML element id (length depends on the id's vint width)."""
    return p.read(n)


def ebml_vint(p):
    """Read one EBML size vint: leading zeros in the first byte mark the length."""
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


class TestCafPaktOverflow:
    def test_nb_frames_overflow(self):
        """read_pakt_chunk does `nb_frames += priming`; the add must overflow int64."""
        p = io.BytesIO(caf_pakt_overflow())

        assert p.read(4) == b"caff"
        assert struct.unpack(">H", p.read(2))[0] == 1  # version
        p.read(2)  # flags
        assert p.read(4) == b"desc"
        assert struct.unpack(">Q", p.read(8))[0] == 32  # desc size the probe checks
        p.read(32)  # read_desc_chunk consumes exactly 32 bytes

        assert p.read(4) == b"pakt"
        struct.unpack(">Q", p.read(8))[0]  # chunk size
        num_packets = struct.unpack(">q", p.read(8))[0]
        nb_frames = struct.unpack(">q", p.read(8))[0]
        priming = struct.unpack(">I", p.read(4))[0]

        # num_packets must pass `num_packets < 0 || INT32_MAX/sizeof(AVIndexEntry) < n`
        assert 0 <= num_packets < (2**31 - 1) // 24
        assert nb_frames == INT64_MAX
        # Falsification: the unguarded add must exceed the signed int64 range.
        assert nb_frames + priming > INT64_MAX

    def test_priming_nonzero(self):
        """Adversarial: priming==0 would make the add a no-op; it must be positive."""
        p = io.BytesIO(caf_pakt_overflow())
        p.read(20 + 32)  # magic/version/flags + desc tag/size + desc payload
        p.read(4 + 8 + 8 + 8)  # pakt tag, size, num_packets, nb_frames
        priming = struct.unpack(">I", p.read(4))[0]
        assert priming > 0


class TestRplIdxOverflow:
    def test_offset_on_index_row(self):
        """The seek target must equal the header length so the index row is read."""
        data = rpl_idx_overflow()
        lines = data.split(b"\n")

        assert lines[0] == b"ARMovie"
        assert int(lines[4]) == 0  # video_format = 0
        assert int(lines[8]) != 0  # fps non-zero, else read_fps sets error
        assert int(lines[9]) != 0  # audio_format non-zero -> audio stream
        assert int(lines[11]) > 0  # channels > 0

        catalog_off = int(lines[17])
        header = b"\n".join(lines[:20]) + b"\n"
        # Falsification: a drifted offset would seek away from the crafted row.
        assert catalog_off == len(header)

    def test_offset_size_overflows(self):
        """Adversarial: offset+video_size (pre-9.1-guard) must overflow int64."""
        data = rpl_idx_overflow()
        catalog_off = int(data.split(b"\n")[17])
        row = data[catalog_off:].split(b"\n")[0].decode()
        offset, video_size, _audio = (int(x) for x in row.replace(";", ",").split(","))
        assert offset == INT64_MAX
        assert offset + video_size > INT64_MAX


class TestSbgNanCast:
    def test_nan_bypasses_range_check(self):
        """scale_double's `m < INT_MIN || m >= INT_MAX` must not catch the NaN."""
        text = sbg_nan_cast().decode()
        name, _, token = text.rstrip("\n").partition(": ")

        # lex_double == strtod: 'nan' parses to a float, not a parse error.
        d = float(token)
        assert math.isnan(d)
        # Falsification: both comparisons are False for NaN, so `*r = m` runs.
        m = d * (1 << 16)
        assert not (m < -(2**31) or m >= 2**31 - 1)
        # 9.1's added isnan(m) is the only check that catches it.
        assert math.isnan(m)

    def test_grammar_reaches_synth_def(self):
        """Adversarial: the line must parse as a named sine-channel def.

        parse_synth_channel tries '-', pink, bell, mix, spin before sine;
        the token must miss all fixed prefixes so lex_double('nan') is
        reached, and the trailing LF satisfies parse_synth_def's lex_line_end.
        """
        data = sbg_nan_cast()
        assert data.endswith(b"\n")

        line = data.rstrip(b"\n")
        name, colon, rest = line.partition(b":")
        assert name == b"a"  # lex_name: [A-Za-z0-9_-]+
        assert colon == b":"
        token = rest.lstrip(b" ")
        assert token == b"nan"
        for prefix in (b"-", b"pink", b"bell", b"mix", b"spin"):
            assert not token.startswith(prefix)


class TestFlvAmfNanDuration:
    def test_onmetadata_nan_duration(self):
        """amf_parse_object assigns `num_val * AV_TIME_BASE` to int64 duration.

        num_val is av_int2double of the raw AMF double; NaN stays NaN through
        the multiply, so the store is a float-cast-overflow.
        """
        data = flv_amf_nan_duration()
        p = io.BytesIO(data)

        assert p.read(3) == b"FLV"
        assert p.read(1)[0] < 5  # version
        p.read(1)  # flags
        offset = struct.unpack(">I", p.read(4))[0]
        assert offset == 9
        assert data[5] == 0  # probe: high byte of the offset must be 0
        # probe: offset + 100 < buf_size
        assert offset + 100 < len(data)
        assert struct.unpack(">I", p.read(4))[0] == 0  # PreviousTagSize0

        tag_type = p.read(1)[0]
        tag_size = int.from_bytes(p.read(3), "big")
        assert tag_type == 18  # FLV_TAG_TYPE_META
        assert tag_size > 18  # flv_read_metabody gate: size > 13 + 1 + 4
        p.read(4)  # dts (24+8)
        p.read(3)  # stream id
        tag_end = p.tell() + tag_size

        # first AMF object: string "onMetaData"
        assert p.read(1)[0] == 0x02
        assert p.read(2 + len("onMetaData")) == b"\x00\x0aonMetaData"
        # second AMF object: ECMA array holding "duration" -> NaN double
        assert p.read(1)[0] == 0x08
        p.read(4)  # max array index
        key_len = struct.unpack(">H", p.read(2))[0]
        key = p.read(key_len)
        assert key == b"duration"
        assert p.tell() < tag_end - 2  # MIXEDARRAY loop bound
        assert p.read(1)[0] == 0x00  # AMF_DATA_TYPE_NUMBER
        num_val = struct.unpack(">d", p.read(8))[0]

        assert math.isnan(num_val)
        # Falsification: NaN survives the scale, the int64 store overflows.
        assert math.isnan(num_val * AV_TIME_BASE)
        # the value bytes lie inside the tag the loop bound is derived from
        assert p.tell() <= tag_end

    def test_amf_end_marker_present(self):
        """Adversarial: a missing END marker would abort the array walk early."""
        data = flv_amf_nan_duration()
        tag_size = int.from_bytes(data[14:17], "big")
        body_end = 24 + tag_size
        assert body_end <= len(data)
        # AMF_END_OF_OBJECT terminates the ECMA array after the NaN value
        assert data[body_end - 3 : body_end] == b"\x00\x00\x09"


class TestWebmDurationCast:
    def test_duration_overflows_int64(self):
        """`duration * time_scale * 1000 / AV_TIME_BASE` must exceed INT64_MAX.

        9.0.2's only guard is isnan(duration); a finite huge value slips
        through and the double->int64 store is a float-cast-overflow.
        """
        p = io.BytesIO(webm_duration_cast())

        assert ebml_id(p) == b"\x1a\x45\xdf\xa3"  # EBML header
        header_len = ebml_vint(p)
        header = p.read(header_len)
        assert b"webm" in header  # matroska_probe wants a known doctype

        assert ebml_id(p) == b"\x18\x53\x80\x67"  # Segment
        seg_len = ebml_vint(p)
        seg = io.BytesIO(p.read(seg_len))

        assert ebml_id(seg) == b"\x15\x49\xa9\x66"  # Info
        info_len = ebml_vint(seg)
        info = io.BytesIO(seg.read(info_len))

        assert ebml_id(info, 2) == b"\x44\x89"  # Duration (2-byte id)
        assert ebml_vint(info) == 8
        d = struct.unpack(">d", info.read(8))[0]

        assert math.isfinite(d) and not math.isnan(d)
        product = d * MK_DEFAULT_TIME_SCALE * 1000 / AV_TIME_BASE
        # Falsification: finite, but beyond the signed int64 range.
        assert math.isfinite(product)
        assert product > INT64_MAX

    def test_cluster_stops_header_parse(self):
        """Adversarial: without the Cluster id the header parse bails first.

        ebml_parse only returns 1 (EBML_STOP) at a Cluster; anything else
        sends matroska_read_header into matroska_resync, which returns EOF
        before the duration line is ever reached.
        """
        data = webm_duration_cast()
        assert b"\x1f\x43\xb6\x75" in data  # MATROSKA_ID_CLUSTER
        # the Cluster id is the last element: parse ends there, not mid-info
        assert data.endswith(b"\x1f\x43\xb6\x75")


class TestRf64DataendOverflow:
    def test_data_end_plus_one_overflows(self):
        """`wav->data_end + (data_size & 1)` must overflow signed int64.

        ds64's data_size saturates av_sat_add64 to INT64_MAX; the odd value
        then makes next_tag_ofs INT64_MAX + 1 (signed overflow, UBSAN).
        """
        data = rf64_dataend_overflow()
        p = io.BytesIO(data)

        assert p.read(4) == b"RF64"
        p.read(4)  # riff size, ignored
        assert p.read(4) == b"WAVE"
        assert p.read(4) == b"ds64"
        ds64_size = struct.unpack("<I", p.read(4))[0]
        assert ds64_size >= 24
        p.read(8)  # riff size
        data_size = struct.unpack("<q", p.read(8))[0]
        sample_count = struct.unpack("<q", p.read(8))[0]

        assert data_size >= 0 and sample_count >= 0  # the only 9.0.2 checks
        assert data_size == INT64_MAX
        # av_sat_add64(tell, INT64_MAX) saturates to INT64_MAX
        data_end = min(p.tell() + data_size, INT64_MAX)
        assert data_end == INT64_MAX
        # Falsification: the unguarded add leaves the int64 range.
        assert data_size & 1 == 1
        assert data_end + (data_size & 1) > INT64_MAX

    def test_fmt_before_data_and_probe(self):
        """Adversarial: non-seekable input needs fmt parsed before data.

        wav_probe demands >32 bytes and ds64 at offset 12; the data tag must
        carry size 0xFFFFFFFF so the rf64 branch (not the size branch) runs.
        """
        data = rf64_dataend_overflow()
        assert len(data) > 32
        assert data[8:12] == b"WAVE"
        assert data[12:16] == b"ds64"

        fmt_off = data.index(b"fmt ")
        data_off = data.index(b"data")
        assert fmt_off < data_off
        fmt = data[fmt_off + 8 : fmt_off + 8 + 16]
        tag, ch, rate, _brate, align, bits = struct.unpack("<HHIIHH", fmt)
        assert (tag, ch, rate, align, bits) == (1, 1, 44100, 2, 16)  # valid PCM
        assert struct.unpack("<I", data[data_off + 4 : data_off + 8])[0] == 0xFFFFFFFF


class TestWavBextBigalloc:
    def test_large_history_alloc(self):
        """bext size - 602 is the coding_history allocation; it must be large."""
        p = io.BytesIO(wav_bext_bigalloc())

        assert p.read(4) == b"RIFF"
        struct.unpack("<I", p.read(4))[0]  # riff size
        assert p.read(4) == b"WAVE"
        assert p.read(4) == b"bext"
        bext_size = struct.unpack("<I", p.read(4))[0]
        body = p.read()

        # The 602 fixed bytes must be present so the prologue reads succeed and
        # control reaches av_malloc(size - 602 + 1).
        assert len(body) >= BEXT_FIXED_SIZE
        version = struct.unpack("<H", body[BEXT_VERSION_OFF : BEXT_VERSION_OFF + 2])[0]
        assert version == 0  # version < 1 skips UMID -> exactly 602 fixed bytes

        history = bext_size - BEXT_FIXED_SIZE
        # Falsification: history must exceed 9.1's 1 MiB cap to be 9.0.2-only.
        assert history > 1024 * 1024

    def test_probe_header_valid(self):
        """Adversarial: wav_probe needs >32 bytes and RIFF..WAVE, else no demux."""
        data = wav_bext_bigalloc()
        assert len(data) > 32
        assert data[0:4] == b"RIFF"
        assert data[8:12] == b"WAVE"


def _stray(dir_path):
    """Temp-staging leftovers that load_corpus must never see."""
    return [n for n in os.listdir(dir_path) if n.endswith(".part") or ".part." in n]


class TestAtomicPublish:
    def test_publishes_clean(self, tmp_path):
        """Falsification: every seed lands in seeds/ with exact bytes, no temp left.

        A temp staged inside seeds/ (the scanned tree) would show up here as a
        .part leftover or an extra file; load_corpus would then ingest it.
        """
        seeds = tmp_path / "seeds"
        seeds.mkdir()

        write_seeds(str(seeds))

        for name, build in SEEDS.items():
            assert (seeds / name).read_bytes() == build()

        # Exactly the published seeds in seeds/, and no temp in it or its parent.
        assert sorted(os.listdir(seeds)) == sorted(SEEDS)
        assert _stray(seeds) == []
        assert _stray(tmp_path) == []

    def test_interrupt_leaves_no_temp(self, tmp_path, monkeypatch):
        """Adversarial: a failed publish leaves no temp and no partial in seeds/.

        os.replace raises on the 2nd seed; the first stays published, the failed
        temp is cleaned up, and nothing half-written sits where load_corpus scans.
        """
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
            write_seeds(str(seeds))

        names = list(SEEDS)
        # First seed published; the one that failed and the rest are absent.
        assert (seeds / names[0]).exists()
        assert not (seeds / names[1]).exists()
        # No staged temp survives, in the scanned dir or its parent.
        assert _stray(seeds) == []
        assert _stray(tmp_path) == []

    def test_regression_tmp_symlink(self, tmp_path):
        """Adversarial: planted temp-name symlinks must not be written through."""
        victim = tmp_path / "victim"
        victim.write_bytes(b"keep")
        seeds = tmp_path / "seeds"
        seeds.mkdir()

        # Plant at the fixed names a non-exclusive writer would open.
        for name in SEEDS:
            (seeds / f"{name}.tmp").symlink_to(victim)
            (tmp_path / f".{name}.part").symlink_to(victim)

        write_seeds(str(seeds))

        assert victim.read_bytes() == b"keep"
        for name, build in SEEDS.items():
            assert (seeds / name).read_bytes() == build()
