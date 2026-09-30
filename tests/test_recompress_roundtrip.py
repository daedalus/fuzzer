"""Tests for recompress_lz4 and recompress_idat (core/mutations/recompress).

Frames and PNGs are built and checked by test-local readers using xxhash,
binascii.crc32 and zlib -- not by the module under test -- so a checksum the
module forgets to repair fails here.
"""

import binascii
import struct
import zlib

import pytest
import xxhash

from fuzzer_tool.core.mutations.recompress import recompress_idat, recompress_lz4
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_MAGIC = 0x184D2204
_FLG_VERSION = 0x40
_FLG_B_INDEP = 0x20
_FLG_C_SIZE = 0x08
_FLG_C_CHECKSUM = 0x04
_BD_64K = 0x40
_UNCOMPRESSED = 0x80000000
_BLOCK_64K = 1 << 16
_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# _mutate_plain script: op 0 (bit flip) at index i, bit b.
_FLIP = 0
_STORE_COMPRESSED = 0
_STORE_RAW = 1


def _flip(i, bit, *tail):
    return ScriptedRng(randints=[_FLIP, i, bit, *tail])


# ── LZ4 reference ──────────────────────────────────────────────────────


def _hc(desc):
    return (xxhash.xxh32(desc).intdigest() >> 8) & 0xFF


def _frame(content, flg=_FLG_VERSION | _FLG_B_INDEP | _FLG_C_CHECKSUM):
    desc = bytes((flg, _BD_64K))
    if flg & _FLG_C_SIZE:
        desc += struct.pack("<Q", len(content))
    out = b"\x00" + struct.pack("<I", _MAGIC) + desc + bytes((_hc(desc),))
    if content:
        out += struct.pack("<I", len(content) | _UNCOMPRESSED) + content
    out += struct.pack("<I", 0)
    if flg & _FLG_C_CHECKSUM:
        out += struct.pack("<I", xxhash.xxh32(content).intdigest())
    return out


def _literal_block(src):
    """Decode a literal-only LZ4 block (the only form recompress emits)."""
    lit = src[0] >> 4
    i = 1
    if lit == 15:
        while True:
            lit += src[i]
            i += 1
            if src[i - 1] != 255:
                break
    assert i + lit == len(src), "block holds more than one literal run"
    return src[i:]


def _read_frame(data):
    """Decode, asserting HC, content size and content checksum all hold."""
    assert data[0] & 1 == 0
    assert struct.unpack_from("<I", data, 1)[0] == _MAGIC
    flg = data[5]
    assert data[6] == _BD_64K
    pos = 7
    size = None
    if flg & _FLG_C_SIZE:
        size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
    assert data[pos] == _hc(data[5:pos]), "header checksum not repaired"
    pos += 1

    content = b""
    blocks = 0
    while True:
        word = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if word == 0:
            break
        n = word & ~_UNCOMPRESSED
        assert n <= _BLOCK_64K
        blk = data[pos : pos + n]
        pos += n
        content += blk if word & _UNCOMPRESSED else _literal_block(blk)
        blocks += 1

    if flg & _FLG_C_CHECKSUM:
        assert struct.unpack_from("<I", data, pos)[0] == xxhash.xxh32(content).intdigest()
    if size is not None:
        assert size == len(content), "content size not repaired"
    return content, blocks


class TestRecompressLz4:
    def test_flip_in_plaintext_raw_blocks(self):
        out = recompress_lz4(_frame(b"hello world"), 4096, rng=_flip(0, 0, _STORE_RAW))
        assert _read_frame(out)[0] == b"iello world"

    def test_flip_in_plaintext_compressed_blocks(self):
        out = recompress_lz4(_frame(b"hello world"), 4096, rng=_flip(1, 1, _STORE_COMPRESSED))
        assert _read_frame(out)[0] == b"hgllo world"

    def test_content_size_is_repaired(self):
        src = _frame(b"abcdef", flg=_FLG_VERSION | _FLG_C_SIZE | _FLG_C_CHECKSUM)
        # op 3 = delete a span: start 0, length 2.
        out = recompress_lz4(src, 4096, rng=ScriptedRng(randints=[3, 0, 2, _STORE_RAW]))
        assert _read_frame(out)[0] == b"cdef"

    def test_splits_blocks_at_block_max(self):
        content = bytes(_BLOCK_64K + 100)
        out = recompress_lz4(_frame(content), 1 << 20, rng=_flip(0, 0, _STORE_COMPRESSED))
        plain, blocks = _read_frame(out)
        assert plain == b"\x01" + content[1:]
        assert blocks == 2

    # Falsification: nothing to round-trip means decline.
    def test_regression_not_lz4_declines(self):
        assert recompress_lz4(b"hello world", 4096, rng=ScriptedRng()) is None

    def test_empty_content_declines(self):
        assert recompress_lz4(_frame(b""), 4096, rng=ScriptedRng()) is None

    def test_undecodable_block_declines(self):
        bad = bytearray(_frame(b"xx"))
        bad[11] &= 0x7F  # drop the uncompressed bit: "xx" is not a valid block
        assert recompress_lz4(bytes(bad), 4096, rng=ScriptedRng()) is None

    @pytest.mark.parametrize("max_len", [1, 8, 20, 64, 4096])
    def test_adversarial_never_exceeds_max_len(self, max_len):
        src = _frame(bytes(range(256)) * 4)
        for seed in range(30):
            out = recompress_lz4(src, max_len, rng=RandPool(seed=seed))
            assert out is None or len(out) <= max_len
            if out is not None:
                _read_frame(out)


# ── PNG IDAT reference ─────────────────────────────────────────────────


def _chunk(kind, data):
    crc = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


_IHDR = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
_IEND = _chunk(b"IEND", b"")


def _png(*idat_payloads, extra=b""):
    return _PNG_SIG + _IHDR + extra + b"".join(_chunk(b"IDAT", p) for p in idat_payloads) + _IEND


def _read_png(data):
    """Return (chunk types, inflated IDAT) asserting every CRC is valid."""
    assert data[:8] == _PNG_SIG
    pos, kinds, idat = 8, [], b""
    while pos < len(data):
        n = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        crc = struct.unpack_from(">I", data, pos + 8 + n)[0]
        assert crc == binascii.crc32(kind + body) & 0xFFFFFFFF, kind
        kinds.append(kind)
        if kind == b"IDAT":
            idat += body
        pos += 12 + n
    return kinds, zlib.decompress(idat)


class TestRecompressIdat:
    def test_flip_in_scanline(self):
        out = recompress_idat(_png(zlib.compress(b"\x00\x7f")), 4096, rng=_flip(1, 0))
        kinds, plain = _read_png(out)
        assert plain == b"\x00\x7e"
        assert kinds == [b"IHDR", b"IDAT", b"IEND"]

    def test_multi_idat_merges_into_one_in_place(self):
        z = zlib.compress(b"\x00\x10")
        text = _chunk(b"tEXt", b"k\x00v")
        out = recompress_idat(_png(z[:3], z[3:], extra=text), 4096, rng=_flip(0, 0))
        kinds, plain = _read_png(out)
        assert plain == b"\x01\x10"
        assert kinds == [b"IHDR", b"tEXt", b"IDAT", b"IEND"]

    # Falsification: no inflatable IDAT means decline.
    def test_regression_no_idat_declines(self):
        assert recompress_idat(_PNG_SIG + _IHDR + _IEND, 4096, rng=ScriptedRng()) is None

    def test_garbage_idat_declines(self):
        assert recompress_idat(_png(b"\x00\x01\x02"), 4096, rng=ScriptedRng()) is None

    def test_not_png_declines(self):
        assert recompress_idat(zlib.compress(b"x"), 4096, rng=ScriptedRng()) is None

    @pytest.mark.parametrize("max_len", [1, 8, 60, 4096])
    def test_adversarial_never_exceeds_max_len(self, max_len):
        src = _png(zlib.compress(bytes(range(256)) * 8))
        for seed in range(30):
            out = recompress_idat(src, max_len, rng=RandPool(seed=seed))
            assert out is None or len(out) <= max_len
            if out is not None:
                _read_png(out)
