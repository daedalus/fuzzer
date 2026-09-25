"""Tests for the structure-aware RAR mutator (`rar_chunk_mutate`)."""

from __future__ import annotations

import struct
import time
import zlib

from fuzzer_tool.core.mutations.rar import (
    _VINT_EDGES,
    RAR4_SIG,
    RAR5_SIG,
    RarMutator,
    _rar4_header,
    _rar5_header,
    decode_vint,
    encode_vint,
    parse_rar,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_PAYLOAD = b"hello"
_RAR5_FILE = 2
_RAR5_END = 5
_RAR4_FILE = 0x74


def _gen(version: int) -> bytes:
    """Deterministic generator output: 5-byte payload, version 5 or 4."""
    rng = ScriptedRng(randints=[len(_PAYLOAD), 1 if version == 5 else 0], randbytes=[_PAYLOAD])
    return RarMutator()._generate_random_rar(max_len=4096, rng=rng)


def _crc5_ok(data: bytes) -> bool:
    arc = parse_rar(data)
    return all(
        struct.unpack_from("<I", data, b.start)[0] == zlib.crc32(data[b.start + 4 : b.end])
        for b in arc.blocks
    )


def _crc4_ok(data: bytes) -> bool:
    arc = parse_rar(data)
    return all(
        struct.unpack_from("<H", data, b.start)[0] == zlib.crc32(data[b.start + 2 : b.end]) & 0xFFFF
        for b in arc.blocks
    )


def _mutate(data: bytes, **script) -> bytes:
    return RarMutator().mutate(data, max_len=4096, rng=ScriptedRng(**script))


class TestVint:
    def test_round_trip_edges(self):
        for v in _VINT_EDGES:
            enc = encode_vint(v)
            assert decode_vint(enc, 0, len(enc)) == (v, len(enc))

    def test_overlong_keeps_value(self):
        enc = encode_vint(1, 10)
        assert len(enc) == 10
        assert decode_vint(enc, 0, len(enc)) == (1, 10)

    def test_eleven_bytes_rejected(self):
        enc = encode_vint(1, 11)
        assert decode_vint(enc, 0, len(enc)) is None

    def test_truncated_rejected(self):
        assert decode_vint(b"\x80\x80", 0, 2) is None


class TestParse:
    def test_rar5_generator_parses(self):
        data = _gen(5)
        assert data.startswith(RAR5_SIG)
        assert [b.htype for b in parse_rar(data).blocks] == [1, _RAR5_FILE, _RAR5_END]
        assert _crc5_ok(data)

    def test_rar4_generator_parses(self):
        data = _gen(4)
        assert data.startswith(RAR4_SIG)
        assert [b.htype for b in parse_rar(data).blocks] == [0x73, _RAR4_FILE, 0x7B]
        assert _crc4_ok(data)

    def test_rar5_data_area_is_payload(self):
        data = _gen(5)
        blk = parse_rar(data).blocks[1]
        assert data[blk.end : blk.data_end] == _PAYLOAD

    def test_rejects_non_rar(self):
        for d in (b"", b"Rar!", b"PK\x03\x04" + bytes(20), b"Rar!\x1a\x07\x02" + bytes(20)):
            assert parse_rar(d) is None


class TestRar5Branches:
    def test_hdr_field_rewrites_and_repairs(self):
        data = _gen(5)
        edge = 6  # _VINT_EDGES index -> 0xFFFFFFFF
        out = _mutate(data, randints=[0], choice_idxs=[1, 0, edge, 0])
        assert parse_rar(out).blocks[1].htype == _VINT_EDGES[edge]
        assert _crc5_ok(out)

    def test_data_size_boundary(self):
        data = _gen(5)
        blk = parse_rar(data).blocks[1]
        remaining = len(data) - blk.end
        out = _mutate(data, randints=[1], choice_idxs=[0, 2])  # remaining + 1
        new = parse_rar(out).blocks[1]
        s, e = new.data_slice
        assert decode_vint(out, s, e)[0] == remaining + 1
        assert _crc5_ok(out)

    def test_file_field_unpacked_size(self):
        data = _gen(5)
        edge = _VINT_EDGES.index(1 << 63)
        out = _mutate(data, randints=[2], choice_idxs=[0, 1, edge, 0])
        blk = parse_rar(out).blocks[1]
        _, p = decode_vint(out, blk.rest, blk.end)  # file flags
        assert decode_vint(out, p, blk.end)[0] == 1 << 63
        assert _crc5_ok(out)

    def test_extra_record_keeps_extra_size_consistent(self):
        rec = encode_vint(2) + encode_vint(7) + b"x"  # size=2 covers type+data
        body = encode_vint(_RAR5_FILE) + encode_vint(1) + encode_vint(len(rec)) + bytes(3) + rec
        data = RAR5_SIG + _rar5_header(body)
        edge = _VINT_EDGES.index(0x4000)
        out = _mutate(data, randints=[3], choice_idxs=[0, 1, edge, 0])
        blk = parse_rar(out).blocks[0]
        assert blk.extra_size == len(encode_vint(0x4000)) + 2
        assert decode_vint(out, blk.end - blk.extra_size + 1, blk.end)[0] == 0x4000
        assert _crc5_ok(out)

    def test_corrupt_crc_flips_only_crc_byte(self):
        data = _gen(5)
        blk = parse_rar(data).blocks[0]
        out = _mutate(data, randints=[4, 0x5A], choice_idxs=[0])
        assert out[blk.start] == data[blk.start] ^ 0x5A
        assert out[: blk.start] + out[blk.start + 1 :] == data[: blk.start] + data[blk.start + 1 :]

    def test_block_drop_end_header(self):
        data = _gen(5)
        out = _mutate(data, randints=[5, 1, 2])
        assert [b.htype for b in parse_rar(out).blocks] == [1, _RAR5_FILE]

    def test_block_dup_and_swap(self):
        data = _gen(5)
        dup = _mutate(data, randints=[5, 0, 1])
        assert [b.htype for b in parse_rar(dup).blocks] == [1, 2, 2, 5]
        swap = _mutate(data, randints=[5, 2, 0, 2])  # swap 0 with (0 + 2) % 3
        assert [b.htype for b in parse_rar(swap).blocks] == [5, 2, 1]

    def test_insert_header(self):
        data = _gen(5)
        out = _mutate(data, randints=[6, 0], choice_idxs=[2])  # _RAR5 end header first
        assert [b.htype for b in parse_rar(out).blocks] == [5, 1, 2, 5]
        assert _crc5_ok(out)

    def test_generate_branch(self):
        out = _mutate(_gen(5), randints=[7, len(_PAYLOAD), 0], randbytes=[_PAYLOAD])
        assert out == _gen(4)


class TestRar4Branches:
    def test_hdr_field_repairs_crc16(self):
        data = _gen(4)
        out = _mutate(data, randints=[0], choice_idxs=[1, 1, 1])  # file hdr flags = 1
        assert struct.unpack_from("<H", out, parse_rar(out).blocks[1].start + 3)[0] == 1
        assert _crc4_ok(out)

    def test_file_field_method(self):
        data = _gen(4)
        blk = parse_rar(data).blocks[1]
        edge = _VINT_EDGES.index(0x7F)
        out = _mutate(data, randints=[2], choice_idxs=[0, 4, edge])  # method byte
        assert out[blk.start + 25] == 0x7F
        assert _crc4_ok(out)

    def test_data_size_sets_long_block(self):
        data = _gen(4)
        blk = parse_rar(data).blocks[0]  # main header has no ADD_SIZE
        out = _mutate(data, randints=[1], choice_idxs=[0, 0])
        assert struct.unpack_from("<H", out, blk.start + 3)[0] & 0x8000
        assert _crc4_ok(out)


class TestFalsification:
    def test_repair_is_load_bearing(self):
        """Field edits keep CRCs valid; a raw edit without repair would not."""
        data = _gen(5)
        out = _mutate(data, randints=[0], choice_idxs=[1, 1, 0, 0])  # flags = 0
        assert out != data
        assert _crc5_ok(out)
        raw = bytearray(out)
        blk = parse_rar(out).blocks[1]
        raw[blk.end - 1] ^= 1
        assert not _crc5_ok(bytes(raw))


class TestAdversarial:
    def test_every_truncation_is_safe(self):
        mut = RarMutator(seed=3)
        for data in (_gen(5), _gen(4)):
            for n in range(len(data) + 1):
                out = mut.mutate(data[:n], max_len=64, rng=RandPool(seed=n))
                assert len(out) <= 64

    def test_hostile_sizes(self):
        mut = RarMutator(seed=1)
        huge = RAR5_SIG + b"\x00" * 4 + encode_vint((1 << 64) - 1) + b"\x01\x00"
        zero = RAR5_SIG + b"\x00" * 4 + b"\x00" * 8
        runaway = RAR5_SIG + b"\x00" * 4 + b"\xff" * 64
        for d in (huge, zero, runaway):
            for seed in range(32):
                assert len(mut.mutate(d, max_len=128, rng=RandPool(seed=seed))) <= 128

    def test_many_tiny_headers_bounded(self):
        tiny = _rar4_header(0x7A, 0, b"")
        data = RAR4_SIG + tiny * 10000
        start = time.perf_counter()
        arc = parse_rar(data)
        assert len(arc.blocks) <= 4096
        out = RarMutator(seed=2).mutate(data, max_len=1 << 20, rng=RandPool(seed=2))
        assert len(out) <= 1 << 20
        assert time.perf_counter() - start < 2.0
