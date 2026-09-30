"""Tests for core/mutations/covering_array_walk.py (ISO-BMFF tkhd, WAV, AVI, WebP VP8/VP8L, ZIP CD/EOCD/entry)."""

from __future__ import annotations

import io
import random
import struct
import wave
import zipfile

import pytest

from fuzzer_tool.core import covering_array as ca
from fuzzer_tool.core import field_spec
from fuzzer_tool.core.mutations import covering_array_walk as w
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY


class _Rng:
    def __init__(self, seed: int = 7):
        self._r = random.Random(seed)

    def randint(self, a, b):
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)


# --------------------------------------------------------------------------
# Samples built independently of the code under test.


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _tkhd(version: int = 0) -> bytes:
    matrix = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
    flags = b"\x00\x00\x07"
    if version == 0:
        # version/flags, creation, modification, track_id, reserved, duration, 8 reserved
        head = struct.pack(">B3sIIIII8x", 0, flags, 1, 2, 1, 0, 1000)
    else:
        head = struct.pack(">B3sQQIIQ8x", 1, flags, 1, 2, 1, 0, 1000)
    tail = struct.pack(">hhhh", 3, 4, 0x0100, 0) + matrix + struct.pack(">II", 640 << 16, 480 << 16)
    return _box(b"tkhd", head + tail)


def _mp4(version: int = 0) -> bytes:
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2")
    moov = _box(b"moov", _box(b"mvhd", b"\x00" * 100) + _box(b"trak", _tkhd(version)))
    return ftyp + moov


def _riff(form: bytes, body: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + form + body


def _chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(b"\x01\x02\x03\x04" * 50)
    return buf.getvalue()


def _avi() -> bytes:
    avih = struct.pack("<14I", 33333, 1000, 0, 0x10, 240, 0, 1, 65536, 320, 200, 0, 0, 0, 0)[:56]
    hdrl = b"hdrl" + _chunk(b"avih", avih)
    return _riff(b"AVI ", _chunk(b"LIST", hdrl) + _chunk(b"LIST", b"movi"))


def _vp8() -> bytes:
    # key frame, version 0, shown, first partition 16 bytes; 640x480.
    tag = (0 | (0 << 1) | (1 << 4) | (16 << 5)).to_bytes(3, "little")
    frame = tag + b"\x9d\x01\x2a" + struct.pack("<HH", 640, 480) + b"\x00" * 8
    return _riff(b"WEBP", _chunk(b"VP8 ", frame))


def _vp8l() -> bytes:
    packed = (640 - 1) | ((480 - 1) << 14) | (1 << 28)
    return _riff(b"WEBP", _chunk(b"VP8L", b"\x2f" + struct.pack("<I", packed) + b"\x00" * 6))


def _zip(names=("a.txt",), comment: bytes = b"") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, n.encode() * 40)
        z.comment = comment
    return buf.getvalue()


def _zip2() -> bytes:
    return _zip(("a.txt", "bb.txt"))


def _zip3() -> bytes:
    return _zip(("a.txt", "bb.txt", "ccc.txt"), comment=b"hi")


CASES = [
    ("covering_array_isobmff_tkhd", w.IsobmffTkhdCoveringArrayMutator, _mp4),
    ("covering_array_wav", w.WavCoveringArrayMutator, _wav),
    ("covering_array_avi", w.AviCoveringArrayMutator, _avi),
    ("covering_array_webp_vp8", w.WebpVp8CoveringArrayMutator, _vp8),
    ("covering_array_webp_vp8l", w.WebpVp8lCoveringArrayMutator, _vp8l),
    ("covering_array_zip_eocd", w.ZipEocdCoveringArrayMutator, _zip),
    ("covering_array_zip_cd", w.ZipCdCoveringArrayMutator, _zip),
    ("covering_array_zip_entry", w.ZipEntryCoveringArrayMutator, _zip2),
]
IDS = [c[0] for c in CASES]


def _read(m, data) -> dict[str, int]:
    fields = m.fields_for(data)
    assert fields is not None
    row = field_spec.read_row(data, fields)
    return dict(zip([f.name for f in fields], row, strict=True))


# --------------------------------------------------------------------------
# Offsets verified against independent parsers / known inputs.


class TestOffsets:
    def test_wav_matches_wave_module(self):
        data = _wav()
        got = _read(w.WavCoveringArrayMutator(), data)
        with wave.open(io.BytesIO(data)) as f:
            assert got["channels"] == f.getnchannels() == 2
            assert got["sample_rate"] == f.getframerate() == 44100
            assert got["bits"] == f.getsampwidth() * 8 == 16
        assert got["audio_format"] == 1
        assert got["block_align"] == 4
        assert got["byte_rate"] == 44100 * 4
        assert got["fmt_size"] == 16
        assert got["data_size"] == 200
        assert got["riff_size"] == len(data) - 8

    @pytest.mark.parametrize("version", [0, 1])
    def test_tkhd_fields_match_constructed_box(self, version):
        got = _read(w.IsobmffTkhdCoveringArrayMutator(), _mp4(version))
        assert got["version"] == version
        assert got["flags"] == 7
        assert got["track_id"] == 1
        assert got["layer"] == 3
        assert got["volume"] == 0x0100
        assert got["width"] == 640 << 16
        assert got["height"] == 480 << 16

    @pytest.mark.parametrize("version", [0, 1])
    def test_tkhd_duration_and_track_id_for_both_layouts(self, version):
        got = _read(w.IsobmffTkhdCoveringArrayMutator(), _mp4(version))
        assert got["duration"] == 1000
        assert got["track_id"] == 1

    def test_tkhd_v1_fields_sit_where_v0_would_be_wrong(self):
        # Falsification: reading a v1 box with the v0 table must give wrong values.
        data = _mp4(1)
        base = w.find_box(data, (b"moov", b"trak", b"tkhd"))[0]
        assert int.from_bytes(data[base + 76 : base + 80], "big") != 640 << 16

    def test_avi_fields_match_constructed_header(self):
        got = _read(w.AviCoveringArrayMutator(), _avi())
        assert got["microsec"] == 33333
        assert got["flags"] == 0x10
        assert got["total_frames"] == 240
        assert got["streams"] == 1
        assert got["buffer"] == 65536
        assert got["width"] == 320
        assert got["height"] == 200
        assert got["avih_size"] == 56

    def test_vp8_fields(self):
        got = _read(w.WebpVp8CoveringArrayMutator(), _vp8())
        assert got["start_code"] == 0x9D012A
        assert got["width"] == 640
        assert got["height"] == 480
        assert got["frame_tag"] == (1 << 4) | (16 << 5)

    def test_vp8l_fields(self):
        got = _read(w.WebpVp8lCoveringArrayMutator(), _vp8l())
        assert got["signature"] == 0x2F
        assert got["packed"] == 639 | (479 << 14) | (1 << 28)

    def test_zip_eocd_matches_struct_parse(self):
        data = _zip3()
        at = data.rfind(b"PK\x05\x06")
        sig, disk, cd_disk, n_disk, n_total, cd_size, cd_off, clen = struct.unpack(
            "<4sHHHHIIH", data[at : at + 22]
        )
        got = _read(w.ZipEocdCoveringArrayMutator(), data)
        assert (got["disk"], got["cd_disk"], got["entries_disk"], got["entries_total"]) == (
            disk,
            cd_disk,
            n_disk,
            n_total,
        )
        assert (got["cd_size"], got["cd_offset"], got["comment_len"]) == (cd_size, cd_off, clen)
        assert n_total == 3
        assert clen == 2

    def test_zip_cd_matches_zipfile(self):
        data = _zip()
        info = zipfile.ZipFile(io.BytesIO(data)).infolist()[0]
        got = _read(w.ZipCdCoveringArrayMutator(), data)
        assert got["method"] == info.compress_type
        assert got["comp_size"] == info.compress_size
        assert got["file_size"] == info.file_size
        assert got["name_len"] == len(info.filename)
        assert got["local_offset"] == info.header_offset
        assert got["flags"] == info.flag_bits
        assert got["int_attr"] == info.internal_attr
        assert got["ext_attr"] == info.external_attr

    def test_zip_entry_targets_second_local_header(self):
        data = _zip2()
        infos = zipfile.ZipFile(io.BytesIO(data)).infolist()
        got = _read(w.ZipEntryCoveringArrayMutator(), data)
        assert got["comp_size"] == infos[1].compress_size
        assert got["file_size"] == infos[1].file_size
        assert got["name_len"] == len(infos[1].filename)
        assert got["method"] == infos[1].compress_type

    def test_zip_entry_walks_to_the_last_with_rng(self):
        data = _zip3()
        infos = zipfile.ZipFile(io.BytesIO(data)).infolist()
        seen = set()
        for seed in range(30):
            f = w.ZipEntryCoveringArrayMutator().fields_for(data, _Rng(seed))
            seen.add(f[0].offset - 4)
        assert seen == {infos[1].header_offset, infos[2].header_offset}

    def test_zip_cd_random_entry_reaches_every_entry(self):
        data = _zip3()
        seen = set()
        for seed in range(40):
            seen.add(w.ZipCdCoveringArrayMutator().fields_for(data, _Rng(seed))[0].offset)
        assert len(seen) == 3


# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "cls", "make"), CASES, ids=IDS)
class TestPerOperator:
    def test_registered_in_format_category(self, name, cls, make):
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == "format"

    def test_available_on_valid_sample(self, name, cls, make):
        assert cls().is_available(MutationContext(), make())

    def test_unavailable_on_empty_and_garbage(self, name, cls, make):
        m = cls()
        assert not m.is_available(MutationContext(), b"")
        assert not m.is_available(MutationContext(), b"\xff" * 64)
        assert m.mutate(b"", _Rng()) is None
        assert m.mutate(b"\x00" * 64, _Rng()) is None

    def test_unavailable_on_truncated_header(self, name, cls, make):
        assert not cls().is_available(MutationContext(), make()[:14])

    def test_values_fit_their_fields(self, name, cls, make):
        fields = cls().fields_for(make())
        assert fields is not None
        for f in fields:
            assert all(0 <= v <= f.max_value for v in f.values), f.name

    def test_fields_do_not_overlap(self, name, cls, make):
        fields = sorted(cls().fields_for(make()), key=lambda f: f.offset)
        for a, b in zip(fields, fields[1:], strict=False):
            assert a.end <= b.offset, (a.name, b.name)

    def test_only_declared_fields_change(self, name, cls, make):
        data = make()
        fields = cls().fields_for(data)
        touched = {i for f in fields for i in range(f.offset, f.end)}
        m = cls()
        for _ in range(60):
            out = m.mutate(data, _Rng())
            if out is None:
                continue
            assert len(out) == len(data)
            assert all(
                a == b for i, (a, b) in enumerate(zip(data, out, strict=True)) if i not in touched
            )

    def test_sweep_is_pairwise_complete(self, name, cls, make):
        # Falsification: a full round-robin sweep must equal a verified array.
        data = make()
        m = cls()
        rng = _Rng()
        fields = m.fields_for(data)
        first = m.mutate(data, rng)
        assert first is not None
        n_rows = len(m._rows)  # noqa: SLF001
        rows = [field_spec.read_row(first, fields)]
        for _ in range(n_rows - 1):
            out = m.mutate(data, rng)
            rows.append(field_spec.read_row(out if out is not None else data, fields))
        assert ca.verify_coverage(rows, m.value_sets(), t=2)

    def test_max_len_truncates(self, name, cls, make):
        out = cls().mutate(make(), _Rng(), max_len=12)
        assert out is not None
        assert len(out) <= 12

    def test_never_returns_input_unchanged(self, name, cls, make):
        data = make()
        m = cls()
        for _ in range(80):
            assert m.mutate(data, _Rng()) != data


# --------------------------------------------------------------------------
# Gates are format-specific (an operator must not fire on its neighbours).


class TestGating:
    def test_wav_not_on_avi_or_webp(self):
        m = w.WavCoveringArrayMutator()
        assert not m.is_available(MutationContext(), _avi())
        assert not m.is_available(MutationContext(), _vp8())

    def test_avi_not_on_wav(self):
        assert not w.AviCoveringArrayMutator().is_available(MutationContext(), _wav())

    def test_vp8_and_vp8l_are_mutually_exclusive(self):
        a, b = w.WebpVp8CoveringArrayMutator(), w.WebpVp8lCoveringArrayMutator()
        assert a.is_available(MutationContext(), _vp8())
        assert not a.is_available(MutationContext(), _vp8l())
        assert b.is_available(MutationContext(), _vp8l())
        assert not b.is_available(MutationContext(), _vp8())

    def test_vp8_found_behind_a_vp8x_chunk(self):
        vp8x = _chunk(b"VP8X", bytes([0x10, 0, 0, 0]) + b"\x00" * 6)
        inner = _chunk(b"VP8 ", _vp8()[20:])
        data = _riff(b"WEBP", vp8x + inner)
        assert w.WebpVp8CoveringArrayMutator().is_available(MutationContext(), data)

    def test_tkhd_needs_a_track(self):
        no_trak = _box(b"ftyp", b"isom\x00\x00\x02\x00isom") + _box(
            b"moov", _box(b"mvhd", b"\x00" * 100)
        )
        assert not w.IsobmffTkhdCoveringArrayMutator().is_available(MutationContext(), no_trak)

    def test_tkhd_works_without_ftyp(self):
        # QuickTime files may omit ftyp entirely.
        data = _box(b"moov", _box(b"trak", _tkhd()))
        assert w.IsobmffTkhdCoveringArrayMutator().is_available(MutationContext(), data)

    def test_tkhd_skips_a_trak_without_tkhd(self):
        data = _box(b"moov", _box(b"trak", _box(b"free", b"\x00" * 8)) + _box(b"trak", _tkhd()))
        assert w.IsobmffTkhdCoveringArrayMutator().is_available(MutationContext(), data)

    def test_riff_odd_sized_chunk_is_padded_to_even(self):
        # An odd-sized chunk before fmt: the walker must skip its pad byte, or
        # every later offset is off by one (survived an earlier mutation check).
        plain = _wav()
        fmt_and_data = plain[12:]
        data = _riff(b"WAVE", _chunk(b"junk", b"abc") + fmt_and_data)
        got = _read(w.WavCoveringArrayMutator(), data)
        assert got["sample_rate"] == 44100
        assert got["channels"] == 2
        assert got["data_size"] == 200

    def test_wav_needs_a_data_chunk(self):
        data = _riff(b"WAVE", _chunk(b"fmt ", struct.pack("<HHIIHH", 1, 2, 44100, 176400, 4, 16)))
        assert not w.WavCoveringArrayMutator().is_available(MutationContext(), data)

    def test_zip_entry_needs_two_entries(self):
        assert not w.ZipEntryCoveringArrayMutator().is_available(MutationContext(), _zip())
        assert w.ZipEntryCoveringArrayMutator().is_available(MutationContext(), _zip2())

    def test_zip_cd_and_eocd_need_a_central_directory(self):
        lfh_only = _zip()[: _zip().find(b"PK\x01\x02")]
        assert not w.ZipCdCoveringArrayMutator().is_available(MutationContext(), lfh_only)
        assert not w.ZipEocdCoveringArrayMutator().is_available(MutationContext(), lfh_only)

    def test_empty_zip_has_eocd_but_no_cd_entry(self):
        empty = _zip(names=())
        assert w.ZipEocdCoveringArrayMutator().is_available(MutationContext(), empty)
        assert not w.ZipCdCoveringArrayMutator().is_available(MutationContext(), empty)

    def test_last_eocd_signature_wins_like_backwards_scanning_parsers(self):
        fake = b"PK\x05\x06" + b"\x00" * 18
        data = _zip(comment=fake)
        assert w.zip_eocd_offset(data) == data.rfind(b"PK\x05\x06")
        assert w.zip_eocd_offset(data) > data.find(b"PK\x05\x06")


# --------------------------------------------------------------------------
# Adversarial: every walker must terminate and never raise on damaged input.


@pytest.mark.parametrize(("name", "cls", "make"), CASES, ids=IDS)
class TestRobustness:
    def test_random_damage_never_raises(self, name, cls, make):
        rng = random.Random(1234)
        base = make()
        m = cls()
        for _ in range(300):
            data = bytearray(base)
            for _ in range(rng.randint(1, 8)):
                data[rng.randrange(len(data))] = rng.randrange(256)
            d = bytes(data)
            m.is_available(MutationContext(), d)
            m.mutate(d, _Rng(rng.randrange(1000)))

    def test_every_prefix_is_safe(self, name, cls, make):
        data = make()
        m = cls()
        for n in range(len(data)):
            m.is_available(MutationContext(), data[:n])
            m.mutate(data[:n], _Rng())


class TestWalkBounds:
    def test_box_size_loop_terminates(self):
        # size 8, repeated 100k times: must stop at the node cap, not run 100k iterations.
        data = (b"\x00\x00\x00\x08free") * 100_000
        assert sum(1 for _ in w._boxes(data, 0, len(data))) == w._MAX_NODES  # noqa: SLF001

    def test_box_smaller_than_header_stops(self):
        data = b"\x00\x00\x00\x04abcd" + b"\x00" * 32
        assert list(w._boxes(data, 0, len(data))) == []  # noqa: SLF001

    def test_box_size_zero_runs_to_end(self):
        data = b"\x00\x00\x00\x00mdat" + b"\x01" * 20
        assert list(w._boxes(data, 0, len(data))) == [(b"mdat", 8, len(data))]  # noqa: SLF001

    def test_largesize_box(self):
        data = b"\x00\x00\x00\x01mdat" + struct.pack(">Q", 24) + b"\x02" * 8
        assert list(w._boxes(data, 0, len(data))) == [(b"mdat", 16, 24)]  # noqa: SLF001

    def test_box_running_past_end_is_clamped(self):
        data = b"\x7f\xff\xff\xffmdat" + b"\x03" * 10
        assert list(w._boxes(data, 0, len(data))) == [(b"mdat", 8, len(data))]  # noqa: SLF001

    def test_riff_chunk_loop_terminates(self):
        data = b"RIFF\x00\x00\x00\x00WAVE" + (b"junk\x00\x00\x00\x00") * 100_000
        assert sum(1 for _ in w._riff_chunks(data, 12, len(data))) == w._MAX_NODES  # noqa: SLF001

    def test_riff_huge_chunk_size_ends_walk(self):
        data = b"RIFF\x00\x00\x00\x00WAVE" + b"fmt \xff\xff\xff\xff" + b"\x00" * 16
        assert len(list(w._riff_chunks(data, 12, len(data)))) == 1  # noqa: SLF001

    def test_zip_cd_chain_is_capped(self):
        many = _zip(names=tuple(f"f{i}" for i in range(80)))
        assert len(w.zip_cd_entries(many)) == w._MAX_ZIP_ENTRIES  # noqa: SLF001

    def test_zip_cd_offset_out_of_range(self):
        data = bytearray(_zip())
        at = data.rfind(b"PK\x05\x06")
        data[at + 16 : at + 20] = struct.pack("<I", 0xFFFFFFFF)
        assert w.zip_cd_entries(bytes(data)) == []

    def test_zip_local_offsets_reject_bad_pointers(self):
        data = bytearray(_zip2())
        cd = w.zip_cd_entries(bytes(data))
        data[cd[1] + 42 : cd[1] + 46] = struct.pack("<I", 3)
        assert len(w.zip_local_offsets(bytes(data))) == 1
