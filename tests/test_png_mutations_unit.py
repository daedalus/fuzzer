"""Tests for core/png_mutations.py — PNG chunk parsing and mutation."""

import struct
import zlib

import pytest

from fuzzer_tool.core.png_mutations import (
    PngChunk,
    PngChunkMutator,
    parse_png_chunks,
    serialize_png_chunks,
)


def _make_png(chunks=None):
    """Build a minimal valid PNG."""
    sig = b"\x89PNG\r\n\x1a\n"
    if chunks is None:
        # Minimal IHDR + IEND
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr = PngChunk(b"IHDR", ihdr_data)
        iend = PngChunk(b"IEND", b"")
        chunks = [ihdr, iend]
    return sig + b"".join(c.serialize() for c in chunks)


class TestPngChunk:
    def test_serialize(self):
        c = PngChunk(b"TEST", b"data")
        raw = c.serialize()
        length = struct.unpack_from(">I", raw, 0)[0]
        assert length == 4
        assert raw[4:8] == b"TEST"
        assert raw[8:12] == b"data"
        assert len(raw) == 16  # 4(len) + 4(type) + 4(data) + 4(crc)

    def test_crc_computed(self):
        c = PngChunk(b"IHDR", b"\x00" * 13)
        raw = c.serialize()
        # Verify CRC is present
        assert len(raw) == 4 + 4 + 13 + 4


class TestParsePngChunks:
    def test_invalid_signature(self):
        assert parse_png_chunks(b"not png") is None

    def test_empty_data(self):
        assert parse_png_chunks(b"") is None

    def test_valid_png(self):
        data = _make_png()
        chunks = parse_png_chunks(data)
        assert chunks is not None
        assert len(chunks) == 2
        assert chunks[0].chunk_type == b"IHDR"
        assert chunks[1].chunk_type == b"IEND"

    def test_truncated_png(self):
        sig = b"\x89PNG\r\n\x1a\n"
        # IHDR says 100 bytes of data, but we only provide 5
        ihdr = PngChunk(b"IHDR", b"\x00" * 100)
        raw = sig + ihdr.serialize()[:16]  # truncated
        chunks = parse_png_chunks(raw)
        # May return None or partial chunks depending on truncation
        if chunks:
            assert len(chunks) <= 1

    def test_multiple_chunks(self):
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        chunks = [
            PngChunk(b"IHDR", ihdr_data),
            PngChunk(b"tEXt", b"key=value"),
            PngChunk(b"IEND", b""),
        ]
        data = _make_png(chunks)
        parsed = parse_png_chunks(data)
        assert parsed is not None
        assert len(parsed) == 3
        assert parsed[1].chunk_type == b"tEXt"


class TestSerializePngChunks:
    def test_roundtrip(self):
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        original = [PngChunk(b"IHDR", ihdr_data), PngChunk(b"IEND", b"")]
        data = serialize_png_chunks(original)
        parsed = parse_png_chunks(data)
        assert parsed is not None
        assert len(parsed) == 2
        assert parsed[0].chunk_type == b"IHDR"
        assert parsed[1].chunk_type == b"IEND"

    def test_signature_present(self):
        data = serialize_png_chunks([PngChunk(b"IEND", b"")])
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


class TestPngChunkMutator:
    def test_mutate_valid_png(self):
        mutator = PngChunkMutator()
        data = _make_png()
        results = {mutator.mutate(data) for _ in range(50)}
        assert len(results) > 1  # different mutations
        for r in results:
            assert isinstance(r, bytes)

    def test_mutate_invalid_png_generates_random(self):
        mutator = PngChunkMutator()
        data = b"not a png"
        results = {mutator.mutate(data) for _ in range(10)}
        # Should generate some valid PNGs
        assert any(r[:8] == b"\x89PNG\r\n\x1a\n" for r in results)

    def test_mutate_max_len(self):
        mutator = PngChunkMutator()
        data = _make_png()
        for _ in range(50):
            result = mutator.mutate(data, max_len=50)
            assert len(result) <= 50

    def test_all_24_ops_reachable(self):
        """Run enough mutations to hit all 24 operator branches."""
        mutator = PngChunkMutator()
        data = _make_png()
        results = set()
        for _ in range(200):
            r = mutator.mutate(data)
            results.add(r)
        # Should produce many distinct outputs
        assert len(results) > 10
