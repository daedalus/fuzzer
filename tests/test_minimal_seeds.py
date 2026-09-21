"""Cold-start seeds must decode with the format's own parser (P2-3, generators handover).

Before this change every constant seed in ``SeedPicker._format_aware_seed`` was
rejected by a real decoder (Pillow / zlib / gzip).
"""

from __future__ import annotations

import gzip
import io
import struct
import zlib

import pytest

from fuzzer_tool.core.minimal_seeds import (
    MINIMAL_SEEDS,
    minimal_bmp,
    minimal_gif,
    minimal_gzip,
    minimal_jpeg,
    minimal_png,
    minimal_zlib,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.seed_picker import SeedPicker


def _picker(sig, max_len=4096):
    class _F:
        _rng = RandPool(seed=1)
        _profile = type("P", (), {"format_signature": sig})()

    _F.max_len = max_len
    return SeedPicker(_F())


class TestStdlibValidity:
    def test_png_structure(self):
        b = minimal_png()
        assert b.startswith(b"\x89PNG\r\n\x1a\n")
        pos, kinds, idat = 8, [], b""
        while pos < len(b):
            (ln,) = struct.unpack(">I", b[pos : pos + 4])
            kind, data = b[pos + 4 : pos + 8], b[pos + 8 : pos + 8 + ln]
            (crc,) = struct.unpack(">I", b[pos + 8 + ln : pos + 12 + ln])
            assert crc == zlib.crc32(kind + data)
            kinds.append(kind)
            if kind == b"IHDR":
                assert ln == 13
                assert struct.unpack(">IIBBBBB", data) == (1, 1, 8, 2, 0, 0, 0)
            if kind == b"IDAT":
                idat += data
            pos += 12 + ln
        assert kinds == [b"IHDR", b"IDAT", b"IEND"]
        assert zlib.decompress(idat) == b"\x00\x00\x00\x00"

    def test_jpeg_markers(self):
        b = minimal_jpeg()
        assert b[:2] == b"\xff\xd8" and b[-2:] == b"\xff\xd9"
        for marker in (b"\xff\xdb", b"\xff\xc0", b"\xff\xc4", b"\xff\xda"):
            assert marker in b

    def test_gif_trailer_and_header(self):
        b = minimal_gif()
        assert b[:6] == b"GIF89a" and b[-1:] == b";"

    def test_bmp_declared_sizes_match(self):
        b = minimal_bmp()
        size, _, _, offset = struct.unpack("<IHHI", b[2:14])
        assert size == len(b)
        assert offset == 54
        assert struct.unpack("<I", b[14:18])[0] == 40

    def test_zlib_roundtrip(self):
        assert zlib.decompress(minimal_zlib()) == b"\x00"

    def test_gzip_roundtrip_and_no_zlib_wrapper(self):
        b = minimal_gzip()
        assert gzip.decompress(b) == b"\x00"
        # raw deflate: the body must not start with a zlib CMF/FLG pair
        assert b[10:12] != b"\x78\x9c"


class TestRealDecoders:
    @pytest.mark.parametrize("name", ["png", "jpeg", "gif", "bmp"])
    def test_pillow_decodes(self, name):
        Image = pytest.importorskip("PIL.Image")
        im = Image.open(io.BytesIO(MINIMAL_SEEDS[name]()))
        im.load()
        assert im.size == (1, 1)


class TestSeedPickerDispatch:
    @pytest.mark.parametrize("sig", sorted(MINIMAL_SEEDS))
    def test_constant_signatures_return_the_valid_seed(self, sig):
        assert _picker(sig)._format_aware_seed() == MINIMAL_SEEDS[sig]()

    def test_riff_uses_generator_and_parses(self):
        from fuzzer_tool.core.mutations.riff import parse_riff_chunks

        seed = _picker("riff")._format_aware_seed()
        assert seed[:4] == b"RIFF"
        assert parse_riff_chunks(seed) is not None

    @pytest.mark.parametrize("sig", ["webp", "webm", "zip", "protobuf"])
    def test_generator_signatures_still_produce_bytes(self, sig):
        out = _picker(sig)._format_aware_seed()
        assert isinstance(out, bytes) and len(out) > 0

    @pytest.mark.parametrize("sig", [None, "elf", "unrecognized-format"])
    def test_unknown_signature_falls_back_to_short_random_buffer(self, sig):
        out = _picker(sig)._format_aware_seed()
        assert 4 <= len(out) <= 64

    def test_max_len_clamps_constant_seeds(self):
        assert len(_picker("jpeg", max_len=16)._format_aware_seed()) == 16

    def test_short_max_len_unknown_format_does_not_raise(self):
        assert len(_picker("unrecognized-format", max_len=2)._format_aware_seed()) <= 2
