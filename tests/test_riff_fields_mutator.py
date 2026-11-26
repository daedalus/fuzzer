"""Tests for the AVI / WAV payload-field mutators (`riff_fields`)."""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.field_poke import W_EDGES
from fuzzer_tool.core.mutations.riff_fields import (
    DATA_SIZE_EDGES,
    DEPTH_MAX,
    FORMAT_EDGES,
    LONGS_EDGES,
    MAGICS,
    SUBFORMAT_EDGES,
    mutate_avi,
    mutate_wav,
    walk_riff,
)
from fuzzer_tool.core.operator_registry import format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool

from .support.scripted_rng import ScriptedRng

CHUNK_HDR = 8
U32_MAX = 0xFFFFFFFF


class _Riff:
    """RIFF builder recording each chunk's absolute payload offset."""

    def __init__(self, form: bytes, magic: bytes = b"RIFF"):
        self.magic = magic
        self.form = form
        self.offs: dict[bytes, int] = {}

    def chunk(self, fcc: bytes, payload: bytes, size: int | None = None) -> bytes:
        size = len(payload) if size is None else size
        return fcc + struct.pack("<I", size) + payload + b"\x00" * (len(payload) & 1)

    def lst(self, kind: bytes, *kids: bytes) -> bytes:
        return self.chunk(b"LIST", kind + b"".join(kids))

    def build(self, *chunks: bytes) -> bytes:
        body = self.form + b"".join(chunks)
        data = self.magic + struct.pack("<I", len(body)) + body
        for c in walk_riff(data):
            self.offs.setdefault(c.fourcc, c.off)
        return data


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _strh(kind: bytes) -> bytes:
    # type handler flags prio lang init | scale rate start length bufsize quality samplesize | rect
    return kind + b"H264" + bytes(12) + struct.pack("<7I", 1, 25, 0, 10, 0, 0, 0) + bytes(8)


BIH = struct.pack("<IiiHHIIiiII", 40, 64, 48, 1, 24, 0, 0, 0, 0, 0, 0)
WFX_PCM = struct.pack("<HHIIHH", 1, 2, 44100, 176400, 4, 16)
GUID_TAIL = bytes.fromhex("000010008000 00aa00389b71".replace(" ", ""))
WFX_EXT = (
    struct.pack("<HHIIHHHHI", 0xFFFE, 2, 44100, 176400, 4, 16, 22, 16, 3)
    + struct.pack("<I", 1)
    + GUID_TAIL
)
INDX = struct.pack("<HBBIIII", 4, 0, 0, 1, 0x63643030, 0, 0) + bytes(16)


def _avi(kind: bytes = b"vids", strf: bytes = BIH) -> tuple[bytes, _Riff]:
    r = _Riff(b"AVI ")
    data = r.build(
        r.lst(
            b"hdrl",
            r.chunk(b"avih", struct.pack("<14I", 40000, 0, 0, 0, 10, 0, 1, 0, 64, 48, 0, 0, 0, 0)),
            r.lst(
                b"strl",
                r.chunk(b"strh", _strh(kind)),
                r.chunk(b"strf", strf),
                r.chunk(b"indx", INDX),
            ),
        ),
        r.lst(b"movi", r.chunk(b"00dc", b"abcd")),
        r.chunk(b"idx1", struct.pack("<4sIII", b"00dc", 0x10, 4, 4) * 2),
    )
    return data, r


def _wav(fmt: bytes = WFX_EXT) -> tuple[bytes, _Riff]:
    r = _Riff(b"WAVE")
    data = r.build(
        r.chunk(b"fmt ", fmt),
        r.chunk(b"fact", struct.pack("<I", 2)),
        r.chunk(b"data", bytes(8)),
        r.chunk(b"cue ", struct.pack("<I", 1) + struct.pack("<II4sIII", 1, 2, b"data", 0, 0, 2)),
    )
    return data, r


class TestWalk:
    def test_order_is_depth_first(self):
        data, _ = _avi()
        fccs = [c.fourcc for c in walk_riff(data)]
        assert fccs == [
            b"LIST",
            b"avih",
            b"LIST",
            b"strh",
            b"strf",
            b"indx",
            b"LIST",
            b"00dc",
            b"idx1",
        ]

    def test_payload_offsets(self):
        data, r = _avi()
        assert data[r.offs[b"avih"] - CHUNK_HDR : r.offs[b"avih"] - 4] == b"avih"

    # Adversarial: a size past EOF is clipped to the bytes present.
    def test_oversized_chunk_clipped(self):
        r = _Riff(b"WAVE")
        data = r.build(r.chunk(b"data", b"xy", size=U32_MAX))
        (c,) = walk_riff(data)
        assert (c.fourcc, c.size) == (b"data", 2)

    # Adversarial: LIST nesting past DEPTH_MAX stops descending, never recurses.
    def test_deep_nesting_bounded(self):
        r = _Riff(b"AVI ")
        inner = b""
        for _ in range(DEPTH_MAX * 4):
            inner = r.lst(b"hdrl", inner)
        data = r.build(inner)
        assert len(walk_riff(data)) <= DEPTH_MAX + 1

    def test_truncated_header(self):
        assert walk_riff(b"RIFF\x00\x00\x00\x00AVI LIS") == []


class TestAvi:
    def test_strh_rate(self):
        data, r = _avi()
        # sites: avih, strh, strf, indx, idx1 -> strh; fields: type handler scale rate
        out = mutate_avi(data, rng=ScriptedRng(randints=[1, 3], choice_idxs=[3]))
        assert _u32(out, r.offs[b"strh"] + 24) == W_EDGES[4][3]
        assert len(out) == len(data)

    def test_vids_strf_is_bitmapinfo(self):
        data, r = _avi()
        out = mutate_avi(data, rng=ScriptedRng(randints=[2, 0], choice_idxs=[2]))
        assert _u32(out, r.offs[b"strf"] + 4) == W_EDGES[4][2]

    def test_auds_strf_is_waveformat(self):
        data, r = _avi(b"auds", WFX_PCM)
        idx = FORMAT_EDGES.index(0xFFFE)
        out = mutate_avi(data, rng=ScriptedRng(randints=[2, 0], choice_idxs=[idx]))
        assert _u16(out, r.offs[b"strf"]) == 0xFFFE

    def test_indx_longs_per_entry(self):
        data, r = _avi()
        idx = LONGS_EDGES.index(0x4001)
        out = mutate_avi(data, rng=ScriptedRng(randints=[3, 0], choice_idxs=[idx]))
        assert _u16(out, r.offs[b"indx"]) == 0x4001

    def test_idx1_entry_stride(self):
        data, r = _avi()
        # site idx1, entry 1, field offset (id flags offset size)
        out = mutate_avi(data, rng=ScriptedRng(randints=[4, 1, 2], choice_idxs=[4]))
        assert _u32(out, r.offs[b"idx1"] + 16 + 8) == W_EDGES[4][4]
        assert out[: r.offs[b"idx1"] + 24] == data[: r.offs[b"idx1"] + 24]

    # Falsification: writing the value already present declines.
    def test_unchanged_declines(self):
        data, _ = _avi()
        # strh start is 0 == W4_EDGES[0]
        assert mutate_avi(data, rng=ScriptedRng(randints=[1, 4], choice_idxs=[0])) is None

    def test_not_avi(self):
        data, _ = _wav()
        assert mutate_avi(data, rng=ScriptedRng()) is None

    # Adversarial: every truncation never raises and keeps length (seeded draws).
    def test_truncations_never_raise(self):
        data, _ = _avi()
        for cut in range(len(data) + 1):
            out = mutate_avi(data[:cut], rng=RandPool(seed=cut))
            assert out is None or len(out) == cut


class TestWav:
    def test_magic_rf64(self):
        data, _ = _wav()
        out = mutate_wav(
            data, rng=ScriptedRng(randints=[0, 0], choice_idxs=[MAGICS.index(b"RF64")])
        )
        assert out[:4] == b"RF64"
        assert format_gate_matches("wav_field_mutate", out)

    def test_extensible_subformat(self):
        data, r = _wav()
        idx = SUBFORMAT_EDGES.index(0x92)
        out = mutate_wav(data, rng=ScriptedRng(randints=[1, 9], choice_idxs=[idx]))
        assert _u32(out, r.offs[b"fmt "] + 24) == 0x92

    def test_data_size_in_header(self):
        data, r = _wav()
        idx = DATA_SIZE_EDGES.index(U32_MAX)
        out = mutate_wav(data, rng=ScriptedRng(randints=[3, 0], choice_idxs=[idx]))
        assert _u32(out, r.offs[b"data"] - 4) == U32_MAX

    def test_cue_position(self):
        data, r = _wav()
        out = mutate_wav(data, rng=ScriptedRng(randints=[4, 2], choice_idxs=[4]))
        assert _u32(out, r.offs[b"cue "] + 8) == U32_MAX

    # Adversarial: a 16-byte PCM fmt exposes only the six base fields.
    def test_short_fmt_hides_extension(self):
        data, r = _wav(WFX_PCM)
        out = mutate_wav(data, rng=ScriptedRng(randints=[1, 5], choice_idxs=[2]))
        assert _u16(out, r.offs[b"fmt "] + 14) == W_EDGES[2][2]

    def test_not_wav(self):
        data, _ = _avi()
        assert mutate_wav(data, rng=ScriptedRng()) is None

    def test_truncations_never_raise(self):
        data, _ = _wav()
        for cut in range(len(data) + 1):
            out = mutate_wav(data[:cut], rng=RandPool(seed=cut))
            assert out is None or len(out) == cut


class TestRegistry:
    def test_sniffers(self):
        avi, _ = _avi()
        wav, _ = _wav()
        assert format_gate_matches("avi_field_mutate", avi)
        assert not format_gate_matches("avi_field_mutate", wav)
        assert format_gate_matches("wav_field_mutate", wav)
        assert not format_gate_matches("wav_field_mutate", avi)
        assert not format_gate_matches("wav_field_mutate", b"RIFF\x00\x00\x00\x00WEBP")
