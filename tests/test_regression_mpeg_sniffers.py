"""Regression: MPEG-family sniffers claim the right streams.

- `nal_chunk_mutate` claimed MPEG-1/2 video, MPEG-PS and MPEG-4 Part 2
  (`00 00 01 B3`/`BA`/`B6`) because it matched any start code.
- `mp3_chunk_mutate` / `adts_chunk_mutate` missed files opening with an
  ID3v2 tag, so real `.mp3` files got flat-byte mutation only.
- `isobmff_chunk_mutate` missed fMP4 segments and QuickTime files whose
  first box is not `ftyp`.
"""

from __future__ import annotations

import random

from fuzzer_tool.core.mutations.adts import AdtsMutator
from fuzzer_tool.core.mutations.id3 import id3v2_len
from fuzzer_tool.core.mutations.mp3 import Mp3Mutator
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

ID3_HEADER_LEN = 10
ID3_FOOTER_FLAG = 0x10


def _id3(body_len: int, flags: int = 0) -> bytes:
    """ID3v2.4 tag with a syncsafe size; footer appended when flagged."""
    size = bytes((body_len >> s) & 0x7F for s in (21, 14, 7, 0))
    tag = b"ID3\x04\x00" + bytes([flags]) + size + b"\x00" * body_len
    if flags & ID3_FOOTER_FLAG:
        tag += b"3DI\x04\x00" + bytes([flags]) + size
    return tag


def _mp3() -> bytes:
    rngs = (random.Random(1), random.Random(2))
    return b"".join(Mp3Mutator()._generate_random_mp3(rng=r) for r in rngs)


def _adts() -> bytes:
    rngs = (random.Random(1), random.Random(2))
    return b"".join(AdtsMutator()._generate_random_adts(rng=r) for r in rngs)


def _box(kind: bytes, body: bytes = b"") -> bytes:
    return (8 + len(body)).to_bytes(4, "big") + kind + body


class TestNalSniffer:
    def test_regression_mpeg2_sequence_header_not_nal(self):
        assert not format_gate_matches("nal_chunk_mutate", b"\x00\x00\x01\xb3" + bytes(16))

    def test_regression_mpeg_ps_pack_not_nal(self):
        assert not format_gate_matches("nal_chunk_mutate", b"\x00\x00\x01\xba" + bytes(16))

    def test_regression_mpeg4_vop_not_nal(self):
        data = b"\x00\x00\x00\x01\xb6" + bytes(16)
        assert not format_gate_matches("nal_chunk_mutate", data)

    def test_h264_sps_still_nal(self):
        assert format_gate_matches("nal_chunk_mutate", b"\x00\x00\x00\x01\x67" + bytes(16))

    def test_hevc_vps_still_nal(self):
        assert format_gate_matches("nal_chunk_mutate", b"\x00\x00\x01\x40\x01" + bytes(16))

    def test_bare_start_code_not_nal(self):
        # Adversarial: no header byte after the start code.
        assert not format_gate_matches("nal_chunk_mutate", b"\x00\x00\x01")
        assert not format_gate_matches("nal_chunk_mutate", b"\x00\x00\x00\x01")


class TestId3Len:
    def test_plain_tag(self):
        assert id3v2_len(_id3(32) + b"x") == ID3_HEADER_LEN + 32

    def test_footer_counted(self):
        assert id3v2_len(_id3(32, ID3_FOOTER_FLAG)) == 2 * ID3_HEADER_LEN + 32

    def test_no_tag(self):
        assert id3v2_len(b"\xff\xfb\x90\x00") == 0

    def test_adversarial_headers(self):
        # Non-syncsafe size byte, truncated tag, short header: all "no tag".
        bad_size = b"ID3\x04\x00\x00\x00\x00\x00\x80" + bytes(200)
        truncated = _id3(64)[:40]
        for data in (bad_size, truncated, b"ID3", b"ID3\x04\x00\x00\x00"):
            assert id3v2_len(data) == 0


class TestId3Sniffing:
    def test_regression_id3_mp3_sniffed(self):
        assert format_gate_matches("mp3_chunk_mutate", _id3(64) + _mp3())

    def test_regression_id3_adts_sniffed(self):
        assert format_gate_matches("adts_chunk_mutate", _id3(64) + _adts())

    def test_id3_without_audio_not_sniffed(self):
        # Falsification: a tag followed by non-audio must not match either.
        data = _id3(64) + b"\x89PNG\r\n\x1a\n" + bytes(16)
        assert not format_gate_matches("mp3_chunk_mutate", data)
        assert not format_gate_matches("adts_chunk_mutate", data)

    def test_id3_adts_not_mp3(self):
        assert not format_gate_matches("mp3_chunk_mutate", _id3(64) + _adts())

    def test_mp3_mutate_keeps_tag(self):
        # Op 4 rewrites frame 0's channel_mode; the tag must pass through intact.
        tag, audio = _id3(64), _mp3()
        channel_op, mode = 4, 2
        rng = ScriptedRng(randints=[channel_op, mode], choice_idxs=[0])
        out = Mp3Mutator().mutate(tag + audio, max_len=1 << 16, rng=rng)

        expected = bytearray(audio)
        expected[3] = (expected[3] & 0x3F) | (mode << 6)
        assert out == tag + bytes(expected)

    def test_adts_mutate_keeps_tag(self):
        tag, audio = _id3(64), _adts()
        out = AdtsMutator().mutate(tag + audio, max_len=1 << 16, rng=random.Random(4))
        assert out[: len(tag)] == tag


class TestIsobmffSniffer:
    def test_regression_non_ftyp_heads_sniffed(self):
        for kind in (b"styp", b"moof", b"moov", b"mdat", b"free", b"wide", b"sidx"):
            data = _box(kind, bytes(8)) + _box(b"mdat", bytes(8))
            assert format_gate_matches("isobmff_chunk_mutate", data), kind

    def test_ftyp_still_sniffed(self):
        assert format_gate_matches("isobmff_chunk_mutate", _box(b"ftyp", b"isom" + bytes(4)))

    def test_unknown_fourcc_not_sniffed(self):
        # Falsification: arbitrary ASCII at offset 4 is not a box head.
        assert not format_gate_matches("isobmff_chunk_mutate", _box(b"abcd", bytes(8)))
        assert not format_gate_matches("isobmff_chunk_mutate", b"\x00\x00\x00\x10moo")


def _engine(rng):
    """OperatorEngine with only the ctx the format handlers read."""
    from types import SimpleNamespace

    from fuzzer_tool.services.operators import OperatorEngine

    engine = object.__new__(OperatorEngine)
    engine._ctx_cache = SimpleNamespace(_rng=rng, max_len=1 << 16)
    return engine


class TestId3Handlers:
    """The `_op_*` handlers pre-parsed the whole buffer, so an ID3-prefixed
    seed fell through to the generator and was replaced."""

    def test_regression_mp3_handler_keeps_id3_seed(self):
        tag, audio = _id3(64), _mp3()
        channel_op, mode = 4, 2
        rng = ScriptedRng(randints=[channel_op, mode], choice_idxs=[0])
        out = _engine(rng)._op_mp3_chunk_mutate(bytearray(tag + audio), 0, b"")

        expected = bytearray(audio)
        expected[3] = (expected[3] & 0x3F) | (mode << 6)
        assert bytes(out) == tag + bytes(expected)

    def test_regression_adts_handler_keeps_id3_seed(self):
        tag, audio = _id3(64), _adts()
        out = _engine(random.Random(4))._op_adts_chunk_mutate(bytearray(tag + audio), 0, b"")
        assert bytes(out[: len(tag)]) == tag
