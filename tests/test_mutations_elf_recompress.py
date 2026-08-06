"""Tests for the ELF and compressed-stream round-trip mutators."""

import random
import struct
import zlib

import pytest

from fuzzer_tool.core.mutations.elf import ElfMutator, parse_elf_header, sniff_elf
from fuzzer_tool.core.mutations.recompress import (
    deflate_gzip,
    inflate_gzip,
    inflate_zlib,
    recompress_gzip,
    recompress_zlib,
    sniff_gzip,
    sniff_zlib,
)
from fuzzer_tool.core.operator_registry import REGISTRY

PLAIN = b"the quick brown fox jumps over the lazy dog " * 64


def _rng():
    return random.Random(1234)


def _minimal_elf64(shnum=3, phnum=2, shentsize=64, phentsize=56):
    """Build a small but structurally valid ELF64 little-endian file."""
    phoff = 64
    shoff = phoff + phnum * phentsize
    total = shoff + shnum * shentsize
    buf = bytearray(total + 256)
    buf[0:4] = b"\x7fELF"
    buf[4] = 2  # ELFCLASS64
    buf[5] = 1  # ELFDATA2LSB
    buf[6] = 1  # EV_CURRENT
    struct.pack_into("<H", buf, 16, 2)  # e_type = ET_EXEC
    struct.pack_into("<H", buf, 18, 0x3E)  # e_machine = x86-64
    struct.pack_into("<I", buf, 20, 1)  # e_version
    struct.pack_into("<Q", buf, 32, phoff)
    struct.pack_into("<Q", buf, 40, shoff)
    struct.pack_into("<H", buf, 52, 64)  # e_ehsize
    struct.pack_into("<H", buf, 54, phentsize)
    struct.pack_into("<H", buf, 56, phnum)
    struct.pack_into("<H", buf, 58, shentsize)
    struct.pack_into("<H", buf, 60, shnum)
    struct.pack_into("<H", buf, 62, shnum - 1)  # e_shstrndx
    return bytes(buf)


# ── Compressed round-trip ──────────────────────────────────────────────


class TestRecompressSniffers:
    def test_sniff_zlib_accepts_real_stream(self):
        assert sniff_zlib(zlib.compress(PLAIN))

    def test_sniff_gzip_accepts_real_member(self):
        assert sniff_gzip(deflate_gzip(PLAIN))

    @pytest.mark.parametrize("blob", [b"", b"GIF89a", b"\x7fELF" + b"\x00" * 60, b"PK\x03\x04"])
    def test_sniffers_reject_other_formats(self, blob):
        assert not sniff_zlib(blob)
        assert not sniff_gzip(blob)

    def test_zlib_sniffer_rejects_bad_fcheck(self):
        """CMF/FLG must be a multiple of 31; a corrupted FLG must not pass."""
        data = bytearray(zlib.compress(PLAIN))
        data[1] ^= 0x01
        assert not sniff_zlib(bytes(data))


class TestRecompressRoundTrip:
    def test_zlib_output_still_inflates(self):
        """The whole point: the mutated stream must remain decompressible."""
        out = recompress_zlib(zlib.compress(PLAIN), max_len=1 << 20, rng=_rng())
        assert out is not None
        assert zlib.decompress(out)

    def test_gzip_output_still_inflates(self):
        out = recompress_gzip(deflate_gzip(PLAIN), max_len=1 << 20, rng=_rng())
        assert out is not None
        assert zlib.decompress(out, 15 | 16)

    def test_zlib_plaintext_actually_changes(self):
        """A round trip that returns the original payload did no work."""
        src = zlib.compress(PLAIN)
        r = _rng()
        assert any(
            zlib.decompress(recompress_zlib(src, max_len=1 << 20, rng=r)) != PLAIN
            for _ in range(20)
        )

    def test_gzip_trailer_is_recomputed(self):
        """CRC-32 and ISIZE must match the mutated payload, not the original."""
        out = recompress_gzip(deflate_gzip(PLAIN), max_len=1 << 20, rng=_rng())
        payload = zlib.decompress(out, 15 | 16)
        crc, isize = struct.unpack("<II", out[-8:])
        assert crc == zlib.crc32(payload) & 0xFFFFFFFF
        assert isize == len(payload) & 0xFFFFFFFF

    def test_non_compressed_input_returns_none(self):
        """Callers rely on None to fall through to another operator."""
        assert recompress_zlib(b"not a zlib stream at all", rng=_rng()) is None
        assert recompress_gzip(b"not a gzip member at all", rng=_rng()) is None

    def test_respects_max_len_and_stays_valid(self):
        """Output is bounded by trimming plaintext, never by truncating output."""
        src = zlib.compress(bytes(random.Random(9).randbytes(200_000)))
        out = recompress_zlib(src, max_len=4096, rng=_rng())
        assert out is not None
        assert len(out) <= 4096
        assert zlib.decompress(out) is not None  # not corrupted by truncation

    def test_inflate_is_bounded_on_compression_bomb(self):
        """A bomb must cost a bounded amount of work, not exhaust memory."""
        bomb = zlib.compress(b"\x00" * (32 << 20))
        plain = inflate_zlib(bomb)
        assert plain is not None
        assert len(plain) <= (1 << 22)

    def test_oversized_compressed_input_is_skipped(self):
        assert inflate_zlib(b"\x78\x9c" + b"\x00" * (1 << 21)) is None

    def test_truncated_stream_keeps_partial_output(self):
        """Corpora are full of partially-valid files; salvage what inflates."""
        src = zlib.compress(PLAIN)
        assert inflate_zlib(src[: len(src) // 2]) is not None

    def test_gzip_inflate_rejects_zlib_stream(self):
        assert inflate_gzip(zlib.compress(PLAIN)) is None


# ── ELF ────────────────────────────────────────────────────────────────


class TestElfParsing:
    def test_sniff_accepts_valid_elf(self):
        assert sniff_elf(_minimal_elf64())

    @pytest.mark.parametrize(
        "blob",
        [b"", b"\x7fELF", b"GIF89a" + b"\x00" * 64, b"\x7fELX" + b"\x00" * 64],
    )
    def test_sniff_rejects_non_elf(self, blob):
        assert not sniff_elf(blob)

    def test_sniff_rejects_bad_class_byte(self):
        data = bytearray(_minimal_elf64())
        data[4] = 7  # not ELFCLASS32/64
        assert not sniff_elf(bytes(data))

    def test_parse_header_reads_table_geometry(self):
        is64, endian, shoff, shnum, shentsize, phoff, phnum, phentsize = parse_elf_header(
            _minimal_elf64(shnum=3, phnum=2)
        )
        assert is64 is True
        assert endian == "<"
        assert (shnum, shentsize) == (3, 64)
        assert (phoff, phnum, phentsize) == (64, 2, 56)
        assert shoff == 64 + 2 * 56

    def test_parse_header_rejects_non_elf(self):
        assert parse_elf_header(b"nope" * 32) is None

    def test_parse_real_system_binary(self):
        """Guard the layout tables against a genuine toolchain-produced ELF."""
        with open("/bin/sh", "rb") as fh:
            data = fh.read()
        hdr = parse_elf_header(data)
        assert hdr is not None
        is64, _endian, shoff, shnum, shentsize, _phoff, _phnum, _phentsize = hdr
        assert is64 is True
        assert shoff + shnum * shentsize <= len(data)


class TestElfMutator:
    def test_non_elf_input_passes_through_untouched(self):
        blob = b"GIF89a" * 10
        assert ElfMutator().mutate(blob, rng=_rng()) == blob

    def test_mutation_changes_the_file(self):
        src = _minimal_elf64()
        m, r = ElfMutator(), _rng()
        assert any(m.mutate(src, max_len=len(src), rng=r) != src for _ in range(40))

    def test_mutation_preserves_length(self):
        """In-place patching must not grow or shrink the buffer."""
        src = _minimal_elf64()
        m, r = ElfMutator(), _rng()
        for _ in range(50):
            assert len(m.mutate(src, max_len=len(src), rng=r)) == len(src)

    def test_respects_max_len(self):
        src = _minimal_elf64()
        assert len(ElfMutator().mutate(src, max_len=128, rng=_rng())) <= 128

    def test_all_ops_survive_missing_tables(self):
        """shoff/phoff of zero must fall back, not raise or corrupt memory."""
        data = bytearray(_minimal_elf64())
        struct.pack_into("<Q", data, 32, 0)  # e_phoff
        struct.pack_into("<Q", data, 40, 0)  # e_shoff
        src = bytes(data)
        m, r = ElfMutator(), _rng()
        for _ in range(100):
            assert len(m.mutate(src, max_len=len(src), rng=r)) == len(src)

    def test_absurd_table_counts_do_not_hang_or_raise(self):
        """e_shnum/e_phnum are attacker-controlled; slot picking must clamp."""
        data = bytearray(_minimal_elf64())
        struct.pack_into("<H", data, 56, 0xFFFF)  # e_phnum
        struct.pack_into("<H", data, 60, 0xFFFF)  # e_shnum
        src = bytes(data)
        m, r = ElfMutator(), _rng()
        for _ in range(100):
            assert len(m.mutate(src, max_len=len(src), rng=r)) == len(src)

    def test_big_endian_elf32_is_handled(self):
        data = bytearray(64 + 8 * 32 + 128)
        data[0:4] = b"\x7fELF"
        data[4] = 1  # ELFCLASS32
        data[5] = 2  # ELFDATA2MSB
        struct.pack_into(">I", data, 32, 52)  # e_shoff
        struct.pack_into(">H", data, 46, 40)  # e_shentsize
        struct.pack_into(">H", data, 48, 4)  # e_shnum
        src = bytes(data)
        hdr = parse_elf_header(src)
        assert hdr is not None and hdr[1] == ">"
        m, r = ElfMutator(), _rng()
        for _ in range(50):
            assert len(m.mutate(src, max_len=len(src), rng=r)) == len(src)

    def test_reaches_multiple_distinct_operations(self):
        """A dispatcher stuck on one op would still pass the tests above."""
        src = _minimal_elf64()
        m, r = ElfMutator(), _rng()
        variants = {m.mutate(src, max_len=len(src), rng=r) for _ in range(300)}
        assert len(variants) > 20

    def test_deterministic_for_a_fixed_seed(self):
        """Reproducibility for a fixed seed is a hard requirement here."""
        src = _minimal_elf64()
        a = [ElfMutator().mutate(src, max_len=len(src), rng=random.Random(5)) for _ in range(5)]
        b = [ElfMutator().mutate(src, max_len=len(src), rng=random.Random(5)) for _ in range(5)]
        assert a == b


# ── Registry wiring ────────────────────────────────────────────────────


class TestRegistration:
    @pytest.mark.parametrize("name", ["elf_chunk_mutate", "recompress_zlib", "recompress_gzip"])
    def test_operator_is_registered_as_format(self, name):
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == "format"

    @pytest.mark.parametrize("name", ["elf_chunk_mutate", "recompress_zlib", "recompress_gzip"])
    def test_operator_has_a_handler(self, name):
        """Registering without a handler is a bug the dispatcher must catch."""
        from fuzzer_tool.services.operators import OperatorEngine

        assert hasattr(OperatorEngine, f"_op_{name}")

    @pytest.mark.parametrize(
        ("name", "sample"),
        [
            ("elf_chunk_mutate", _minimal_elf64()),
            ("recompress_zlib", zlib.compress(PLAIN)),
            ("recompress_gzip", deflate_gzip(PLAIN)),
        ],
    )
    def test_sniffer_gate_admits_matching_input(self, name, sample):
        assert name in REGISTRY.available(None, sample)
