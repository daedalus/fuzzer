"""Regression tests for tools/gen_ffmpeg_91_seeds.py.

The seeds target demuxer paths FFmpeg 9.0.2 ships unguarded and 9.1 hardened.
Each test re-parses the produced bytes exactly as the 9.0.2 demuxer reads them
(independently of the builder) and asserts the unguarded arithmetic is reached,
so a silent offset drift in the generator fails here rather than producing a
seed that no longer lands on the bug.
"""

from __future__ import annotations

import io
import os
import struct

import pytest
from tools import gen_ffmpeg_91_seeds as gen
from tools.gen_ffmpeg_91_seeds import (
    BEXT_FIXED_SIZE,
    INT64_MAX,
    SEEDS,
    caf_pakt_overflow,
    rpl_idx_overflow,
    wav_bext_bigalloc,
    write_seeds,
)

# version field offset inside the 602-byte BEXT prologue (wav_parse_bext_tag)
BEXT_VERSION_OFF = 346


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


class TestWriteSeeds:
    def test_publishes_exact_seeds(self, tmp_path):
        """Falsification: every seed lands with builder bytes, no temp left."""
        write_seeds(str(tmp_path))

        assert sorted(os.listdir(tmp_path)) == sorted(SEEDS)
        for name, build in SEEDS.items():
            assert (tmp_path / name).read_bytes() == build()

    def test_regression_tmp_symlink(self, tmp_path):
        """Adversarial: a planted `<seed>.tmp` symlink must not be written through."""
        victim = tmp_path / "victim"
        victim.write_bytes(b"keep")
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        for name in SEEDS:
            (seeds / f"{name}.tmp").symlink_to(victim)

        write_seeds(str(seeds))

        assert victim.read_bytes() == b"keep"

    def test_regression_replace_fails(self, tmp_path, monkeypatch):
        """Adversarial: a failed rename leaves neither seed nor temp behind."""

        def boom(_src, _dst):
            raise OSError("replace failed")

        monkeypatch.setattr(gen.os, "replace", boom)

        with pytest.raises(OSError):
            write_seeds(str(tmp_path))

        assert os.listdir(tmp_path) == []
