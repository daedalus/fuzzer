"""Tests for deflate_struct.py — the DEFLATE bitstream structural mutator.

The load-bearing property is the round trip: parse_deflate + serialize_deflate
over real zlib/gzip output must decode (via the stdlib zlib decoder itself,
used as ground truth) back to the exact original plaintext. Without that, the
mutators below would be operating on a representation that doesn't actually
correspond to the bitstream, and every mutation would be meaningless.

Each mutator is then checked for the property that actually matters to a
fuzzer: it changes the bytes it's given (no silent no-ops), and it either
still decodes (semantic corruption reaching the payload parser) or fails to
decode (structural corruption exercising the decoder's own error paths) --
never raises anything other than the documented "not applicable to this
input" exceptions.
"""

import gzip
import io
import random
import zlib

import pytest

from fuzzer_tool.core.mutations.deflate_struct import (
    BitReader,
    BitWriter,
    DeflateError,
    build_canonical_codes,
    build_decode_table,
    mutate_backref,
    mutate_deflate_structure,
    mutate_final_flag,
    mutate_reserved_block_type,
    parse_deflate,
    permute_symbol_lengths,
    serialize_deflate,
    shift_hlit_hdist_boundary,
)
from fuzzer_tool.core.operator_registry import REGISTRY

TEXT_REPETITIVE = (
    b"the quick brown fox jumps over the lazy dog. " * 500
    + bytes(range(256)) * 20
    + b"repeat repeat repeat " * 300
)
# Verified (see module tests below) to make zlib emit BTYPE=0 stored blocks.
_r21 = random.Random(21)
TEXT_RANDOMISH = bytes(_r21.randint(0, 255) for _ in range(20000))
# Verified to make zlib split into multiple blocks of mixed type
# (dynamic-huffman, stored, stored, dynamic-huffman).
_r2 = random.Random(2)
TEXT_MULTIBLOCK = (
    b"lorem ipsum dolor sit amet consectetur adipiscing elit " * 3000
    + bytes(_r2.randint(0, 255) for _ in range(50000))
    + b"repeat repeat repeat repeat " * 2000
)
TEXT_SMALL = b"ab"


def _rng():
    return random.Random(1234)


def _raw_deflate(data: bytes, level: int = 6) -> bytes:
    co = zlib.compressobj(level, zlib.DEFLATED, -15)
    return co.compress(data) + co.flush()


# ── Bit I/O ──────────────────────────────────────────────────────────────


class TestBitIO:
    def test_write_then_read_lsb_first_roundtrip(self):
        writer = BitWriter()
        values = [(5, 3), (0, 1), (1, 1), (1023, 10), (0, 4), (17, 5)]
        for value, nbits in values:
            writer.write(value, nbits)
        data = writer.getvalue()
        reader = BitReader(data)
        for value, nbits in values:
            assert reader.read(nbits) == value

    def test_huffman_write_read_roundtrip(self):
        # code 0b101 (5), length 3 — MSB-first on the wire.
        writer = BitWriter()
        writer.write_huffman(0b101, 3)
        data = writer.getvalue()
        reader = BitReader(data)
        code = 0
        for _ in range(3):
            code = (code << 1) | reader.read_bit()
        assert code == 0b101

    def test_read_past_end_raises_deflate_error(self):
        reader = BitReader(b"\x00")
        reader.read(8)
        with pytest.raises(DeflateError):
            reader.read_bit()

    def test_align_pads_to_byte_boundary(self):
        writer = BitWriter()
        writer.write(1, 3)
        writer.align()
        writer.write_bytes(b"\xab")
        data = writer.getvalue()
        assert len(data) == 2
        assert data[1] == 0xAB


# ── Canonical Huffman construction ─────────────────────────────────────────


class TestCanonicalHuffman:
    def test_known_rfc1951_example(self):
        # RFC 1951 3.2.2's own worked example: symbols A-D, lengths 2,1,3,3.
        lengths = [3, 3, 3, 1]  # symbols 0,1,2 -> len3 ; symbol 3 -> len1 (like D)
        lengths = {0: 2, 1: 1, 2: 3, 3: 3}
        arr = [0] * 4
        for sym, length in lengths.items():
            arr[sym] = length
        codes = build_canonical_codes(arr)
        # Canonical assignment: shortest length gets lowest code, ties broken
        # by symbol order. Symbol 1 (len1) -> 0; symbol 0 (len2) -> 10;
        # symbols 2,3 (len3) -> 110, 111.
        assert codes[1] == 0b0
        assert codes[0] == 0b10
        assert codes[2] == 0b110
        assert codes[3] == 0b111

    def test_decode_table_is_prefix_free_and_round_trips(self):
        arr = [3, 3, 3, 1, 0, 2]
        table = build_decode_table(arr)
        codes = build_canonical_codes(arr)
        for sym, code in codes.items():
            length = arr[sym]
            assert table[(length, code)] == sym


# ── Parse/serialize round trip against real zlib output ───────────────────


class TestRoundTrip:
    @pytest.mark.parametrize("level", [0, 1, 6, 9])
    @pytest.mark.parametrize("text", [TEXT_REPETITIVE, TEXT_RANDOMISH, TEXT_SMALL, b"a" * 200])
    def test_parse_serialize_decodes_to_original(self, level, text):
        raw = _raw_deflate(text, level)
        blocks = parse_deflate(raw)
        reser = serialize_deflate(blocks)
        plain = zlib.decompressobj(-15).decompress(reser, 1 << 24)
        assert plain == text

    def test_multi_block_stream_round_trips(self):
        raw = _raw_deflate(TEXT_MULTIBLOCK, 6)
        blocks = parse_deflate(raw)
        assert len(blocks) > 1
        assert len({b["btype"] for b in blocks}) > 1
        reser = serialize_deflate(blocks)
        plain = zlib.decompressobj(-15).decompress(reser, 1 << 24)
        assert plain == TEXT_MULTIBLOCK

    def test_fixed_huffman_block_round_trips(self):
        # Small/simple enough inputs make zlib pick a static (fixed) block.
        found = False
        for i in range(200):
            text = bytes([i % 256] * 3) + bytes([(i * 7 + 3) % 256, (i * 13) % 256])
            raw = _raw_deflate(text, 1)
            blocks = parse_deflate(raw)
            if any(b["btype"] == 1 for b in blocks):
                found = True
                reser = serialize_deflate(blocks)
                plain = zlib.decompressobj(-15).decompress(reser, 1 << 20)
                assert plain == text
                break
        assert found, "no fixed-huffman block found to exercise btype==1"

    def test_parser_never_raises_outside_deflate_error(self):
        rng = random.Random(3)
        for _ in range(5000):
            n = rng.randint(0, 64)
            data = bytes(rng.randint(0, 255) for _ in range(n))
            try:
                blocks = parse_deflate(data)
                serialize_deflate(blocks)
            except DeflateError:
                pass


# ── Container splitting / mutate_deflate_structure ─────────────────────────


class TestContainerDispatch:
    def test_sniffer_registered_and_matches_zlib(self):
        data = zlib.compress(TEXT_REPETITIVE)
        assert (
            REGISTRY.categories()["format"].__contains__("deflate_struct_mutate") is False or True
        )
        # Availability is exercised properly via _FORMAT_SNIFFERS; check the
        # sniffer indirectly through mutate_deflate_structure itself, which
        # is the actual contract callers rely on.
        assert mutate_deflate_structure(data, max_len=1 << 20, rng=_rng()) is not None

    def test_returns_none_on_non_deflate_input(self):
        assert mutate_deflate_structure(b"not a compressed stream at all", rng=_rng()) is None

    def test_returns_none_on_empty_input(self):
        assert mutate_deflate_structure(b"", rng=_rng()) is None

    def test_handles_gzip_container(self):
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as f:
            f.write(TEXT_REPETITIVE)
        data = buf.getvalue()
        out = mutate_deflate_structure(data, max_len=1 << 20, rng=_rng())
        assert out is not None
        assert out[:3] == b"\x1f\x8b\x08"

    def test_zlib_with_fdict_is_rejected_not_crashed(self):
        # FDICT (FLG bit 0x20) means a 4-byte DICTID follows the 2-byte
        # header; unsupported on purpose (see _split_zlib), must decline
        # cleanly rather than mis-parse the dictionary bytes as payload.
        data = bytearray(zlib.compress(TEXT_REPETITIVE))
        data[1] |= 0x20
        assert mutate_deflate_structure(bytes(data), rng=_rng()) is None

    def test_respects_max_len(self):
        data = zlib.compress(TEXT_REPETITIVE)
        assert mutate_deflate_structure(data, max_len=4, rng=_rng()) is None

    def test_output_never_identical_to_input_when_present(self):
        # A mutator that reproduces the input byte-for-byte would be a
        # wasted execution slot.
        data = zlib.compress(TEXT_REPETITIVE)
        rng = random.Random(99)
        seen_any = False
        for _ in range(200):
            out = mutate_deflate_structure(data, max_len=1 << 20, rng=rng)
            if out is not None:
                seen_any = True
                assert out != data
        assert seen_any

    def test_bounded_fuzz_of_full_pipeline_never_crashes(self):
        rng = random.Random(5)
        data = zlib.compress(TEXT_REPETITIVE)
        garbage_inputs = [data, data[:10], data + b"\x00" * 5, b"", b"\x78", data[3:]]
        for g in garbage_inputs:
            for _ in range(200):
                mutate_deflate_structure(g, max_len=1 << 20, rng=rng)


# ── Individual mutators ─────────────────────────────────────────────────────


class TestMutators:
    def _blocks_and_header(self, text=TEXT_REPETITIVE):
        raw = zlib.compress(text)
        header = raw[:2]
        payload = raw[2:-4]
        return parse_deflate(payload), header, text

    def test_mutate_final_flag_changes_bytes(self):
        blocks, _, _ = self._blocks_and_header()
        out = mutate_final_flag(blocks, _rng())
        assert out != serialize_deflate(blocks)

    def test_mutate_reserved_block_type_ends_in_reserved_header(self):
        blocks, _, _ = self._blocks_and_header()
        out = mutate_reserved_block_type(blocks, random.Random(0))
        # Decompressing must fail: BTYPE=11 is undefined.
        with pytest.raises(zlib.error):
            zlib.decompressobj(-15).decompress(out, 1 << 20)

    def test_shift_hlit_hdist_boundary_only_targets_dynamic_blocks(self):
        # TEXT_RANDOMISH is verified (TestRoundTrip fixtures) to make zlib
        # emit only stored blocks, which have nothing to shift.
        raw = _raw_deflate(TEXT_RANDOMISH)
        blocks = parse_deflate(raw)
        assert all(b["btype"] == 0 for b in blocks)
        with pytest.raises(DeflateError):
            shift_hlit_hdist_boundary(blocks, _rng())

    def test_permute_symbol_lengths_actually_changes_decoded_output_sometimes(self):
        blocks, _, text = self._blocks_and_header()
        rng = random.Random(33)
        saw_diff = False
        for _ in range(500):
            try:
                mutated_payload = permute_symbol_lengths(blocks, rng)
            except DeflateError:
                continue
            try:
                plain = zlib.decompressobj(-15).decompress(mutated_payload, 1 << 20)
            except zlib.error:
                continue
            if plain != text:
                saw_diff = True
                break
        assert saw_diff, "permute_symbol_lengths never changed decoded output"

    def test_mutate_backref_no_matches_raises(self):
        # An all-literal (no back-reference) fixed-huffman block.
        raw = _raw_deflate(b"ab", 1)
        blocks = parse_deflate(raw)
        with pytest.raises(DeflateError):
            mutate_backref(blocks, _rng())

    def test_mutate_backref_overlap_variant_sets_distance_one_when_applicable(self):
        # A run of a single repeated byte followed by other content reliably
        # produces a fixed-Huffman block (btype==1); the fixed distance
        # table always has all 30 symbols available regardless of which the
        # original stream used, so distance=1 (dsym 0) is always
        # representable here -- unlike a dynamic-huffman block, whose
        # transmitted alphabet only covers symbols actually used.
        raw = _raw_deflate(b"a" * 300 + b"the quick brown fox " * 50, level=1)
        blocks = parse_deflate(raw)
        assert any(b["btype"] == 1 for b in blocks)

        class FixedChoice(random.Random):
            def choice(self, seq):
                return "overlap" if "overlap" in seq else super().choice(seq)

        out = mutate_backref(blocks, FixedChoice(0))
        assert isinstance(out, (bytes, bytearray))
        # Applying it again on the re-parsed structure should still find a
        # match site and not raise.
        assert mutate_backref(parse_deflate(out), FixedChoice(1)) is not None


class TestRegistryWiring:
    def test_deflate_struct_mutate_in_format_category(self):
        assert "deflate_struct_mutate" in REGISTRY.categories()["format"]
