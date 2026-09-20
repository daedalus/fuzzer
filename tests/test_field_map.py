"""Field map: names the byte ranges of a known container format.

Three groups, per AGENTS.md rule 23:

* the maps of valid files (what the walkers are for);
* falsification: claims that would be false if the map trusted the file's
  own length fields, or depended on field *values* rather than layout;
* adversarial: truncation at every length, chunk floods, hostile sizes,
  runaway nesting. A walker runs on the attacker-controlled crashing input,
  so it must terminate, stay in bounds and never raise.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib

import pytest

from fuzzer_tool.core.field_map import (
    MAX_FIELDS,
    Endian,
    FieldKind,
    FieldMap,
    map_fields,
    span_int,
    span_repr,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
ZIP_DATE = (2020, 1, 1, 0, 0, 0)
U32_MAX = 0xFFFFFFFF


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(ctype + body)
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", crc)


def _png(width: int = 16, height: int = 8, idat: bytes = b"\x78\x9c\x03\x00\x00\x00\x00\x01"):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        PNG_MAGIC + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
    )


def _gzip(name: bytes = b"a.txt") -> bytes:
    payload = b"hello gzip " * 4
    comp = zlib.compressobj(wbits=-15)
    body = comp.compress(payload) + comp.flush()
    header = b"\x1f\x8b\x08\x08" + struct.pack("<I", 0) + b"\x00\x03" + name + b"\x00"
    return header + body + struct.pack("<II", zlib.crc32(payload), len(payload))


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("a.txt", date_time=ZIP_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, b"hello zip " * 8)
    return buf.getvalue()


def _riff_chunk(cid: bytes, body: bytes) -> bytes:
    pad = b"\x00" if len(body) % 2 else b""
    return cid + struct.pack("<I", len(body)) + body + pad


def _riff(*chunks: bytes, form: bytes = b"WAVE") -> bytes:
    body = form + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _wav() -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 8000, 1, 8)
    return _riff(_riff_chunk(b"fmt ", fmt), _riff_chunk(b"data", b"\x01\x02\x03"))


VALID = {
    "png": _png,
    "gzip": _gzip,
    "zip": _zip,
    "riff": _wav,
}


def _well_formed(fm: FieldMap, data: bytes) -> None:
    """Spans are non-empty, sorted, disjoint and inside the input."""
    end = 0
    for sp in fm.spans:
        assert sp.width > 0, sp
        assert sp.offset >= end, sp
        assert sp.offset + sp.width <= len(data), sp
        end = sp.offset + sp.width
    assert len(fm.spans) <= MAX_FIELDS


def _by_name(fm: FieldMap) -> dict:
    return {sp.name: sp for sp in fm.spans}


class TestValidFiles:
    @pytest.mark.parametrize("fmt", sorted(VALID))
    def test_format_detected(self, fmt):
        assert map_fields(VALID[fmt]()).fmt == fmt

    @pytest.mark.parametrize("fmt", sorted(VALID))
    def test_spans_cover_every_byte(self, fmt):
        data = VALID[fmt]()
        fm = map_fields(data)

        _well_formed(fm, data)
        assert sum(sp.width for sp in fm.spans) == len(data)

    @pytest.mark.parametrize("fmt", sorted(VALID))
    def test_names_are_unique(self, fmt):
        names = [sp.name for sp in map_fields(VALID[fmt]()).spans]
        assert len(names) == len(set(names))

    def test_png_ihdr_fields(self):
        data = _png(width=16, height=8)
        spans = _by_name(map_fields(data))

        width = spans["IHDR[0].width"]
        assert (width.offset, width.width) == (16, 4)
        assert width.kind is FieldKind.VALUE
        assert width.endian is Endian.BIG
        assert span_int(data, width) == 16
        assert span_int(data, spans["IHDR[0].height"]) == 8
        assert spans["IHDR[0].crc"].kind is FieldKind.CRC
        assert spans["IEND[2].length"].kind is FieldKind.LENGTH
        assert span_int(data, spans["IEND[2].length"]) == 0

    def test_gzip_name_field(self):
        data = _gzip(b"a.txt")
        spans = _by_name(map_fields(data))

        fname = spans["fname"]
        assert (fname.offset, fname.width) == (10, len(b"a.txt") + 1)
        assert spans["flags"].kind is FieldKind.FLAGS
        assert spans["crc32"].kind is FieldKind.CRC
        assert span_int(data, spans["isize"]) == len(b"hello gzip " * 4)

    def test_zip_offsets_point_at_real_records(self):
        data = _zip()
        spans = _by_name(map_fields(data))

        # The EOCD's central-directory offset must equal where the central
        # directory really is: the map and the file agree.
        assert spans["eocd.cd_offset"].kind is FieldKind.OFFSET
        assert span_int(data, spans["eocd.cd_offset"]) == data.find(b"PK\x01\x02")
        assert span_int(data, spans["cdh[0].local_offset"]) == 0
        assert span_int(data, spans["lfh[0].comp_size"]) == spans["lfh[0].data"].width

    def test_riff_chunks_and_pad_byte(self):
        data = _wav()
        spans = _by_name(map_fields(data))

        assert spans["riff.id"].kind is FieldKind.MAGIC
        assert spans["fmt[0].size"].endian is Endian.LITTLE
        # "data" holds 3 bytes: odd, so RIFF pads it to even.
        assert spans["data[1].data"].width == 3
        assert spans["data[1].pad"].kind is FieldKind.PADDING

    def test_riff_list_children_are_nested(self):
        info = b"INFO" + _riff_chunk(b"ISFT", b"tool\x00")
        data = _riff(_riff_chunk(b"LIST", info))
        fm = map_fields(data)

        _well_formed(fm, data)
        assert "LIST[0]/ISFT[0].data" in _by_name(fm)


class TestValueRendering:
    def test_int_field_renders_fixed_width_hex(self):
        data = _png(width=16)
        sp = _by_name(map_fields(data))["IHDR[0].width"]
        assert span_repr(data, sp) == "0x00000010"

    def test_wide_field_has_no_int_value(self):
        data = _zip()
        sp = _by_name(map_fields(data))["lfh[0].data"]
        assert span_int(data, sp) is None

    def test_long_data_is_capped(self):
        data = _png(idat=b"\xaa" * 200)
        sp = _by_name(map_fields(data))["IDAT[1].data"]
        assert len(span_repr(data, sp)) < 64


class TestFalsification:
    def test_map_follows_bytes_not_claimed_lengths(self):
        # IDAT claims 4 GiB. Only the bytes that exist may be mapped.
        data = bytearray(_png())
        idat_len_off = 8 + 12 + 13
        data[idat_len_off : idat_len_off + 4] = struct.pack(">I", U32_MAX)
        data = bytes(data)
        fm = map_fields(data)
        spans = _by_name(fm)

        _well_formed(fm, data)
        assert spans["IDAT[1].data"].offset + spans["IDAT[1].data"].width == len(data)
        assert "IDAT[1].crc" not in spans

    def test_layout_is_independent_of_field_values(self):
        a = map_fields(_png(width=16, height=8)).spans
        b = map_fields(_png(width=U32_MAX, height=0)).spans
        assert a == b

    @pytest.mark.parametrize("junk", [b"", b"hello world", b"\x89PNG", b"RIFF", b"\x1f", b"PK"])
    def test_non_formats_are_unmapped(self, junk):
        assert map_fields(junk) == FieldMap("", [])

    def test_magic_only_input_maps_the_magic(self):
        fm = map_fields(PNG_MAGIC)
        assert fm.fmt == "png"
        assert [sp.name for sp in fm.spans] == ["signature"]


class TestAdversarial:
    @pytest.mark.parametrize("fmt", sorted(VALID))
    def test_every_truncation_is_well_formed(self, fmt):
        data = VALID[fmt]()
        for cut in range(len(data) + 1):
            prefix = data[:cut]
            _well_formed(map_fields(prefix), prefix)

    def test_chunk_flood_is_capped(self):
        data = PNG_MAGIC + _png_chunk(b"tEXt", b"") * 10_000
        fm = map_fields(data)

        _well_formed(fm, data)
        assert len(fm.spans) == MAX_FIELDS

    def test_zip_hostile_sizes(self):
        lfh = b"PK\x03\x04" + b"\x00" * 14 + struct.pack("<IIHH", U32_MAX, U32_MAX, 0xFFFF, 0xFFFF)
        data = lfh + b"tail"
        _well_formed(map_fields(data), data)

    def test_zip_data_descriptor_without_size_stops_at_next_record(self):
        # flags bit 3 set and comp_size 0: size is unknown until the
        # descriptor; the walker must resync on the next signature.
        lfh = b"PK\x03\x04" + struct.pack("<HHHHHIIIHH", 20, 8, 0, 0, 0, 0, 0, 0, 0, 0)
        data = lfh + b"payload" + b"PK\x07\x08" + b"\x00" * 12
        fm = map_fields(data)
        spans = _by_name(fm)

        _well_formed(fm, data)
        assert spans["lfh[0].data"].width == len(b"payload")
        assert "dd[0].signature" in spans

    def test_gzip_unterminated_name_covers_the_rest(self):
        data = b"\x1f\x8b\x08\x08" + b"\x00" * 6 + b"no-terminator" * 20
        fm = map_fields(data)

        _well_formed(fm, data)
        assert sum(sp.width for sp in fm.spans) == len(data)

    def test_riff_size_field_at_max(self):
        data = b"RIFF" + struct.pack("<I", U32_MAX) + b"WAVE" + b"data" + struct.pack("<I", U32_MAX)
        _well_formed(map_fields(data), data)

    def test_riff_nesting_is_bounded(self):
        body = b"x"
        for _ in range(50):
            body = _riff_chunk(b"LIST", b"INFO" + body)
        data = _riff(body)
        fm = map_fields(data)

        _well_formed(fm, data)
        assert len(fm.spans) <= MAX_FIELDS

    @pytest.mark.parametrize(
        "magic", [PNG_MAGIC, b"\x1f\x8b\x08\x1f", b"PK\x03\x04", b"RIFF\xff\xff\xff\xffWAVE"]
    )
    @pytest.mark.parametrize("fill", [b"\xff", b"\x00", bytes(range(256))])
    def test_hostile_body_after_magic(self, magic, fill):
        data = magic + fill * 4
        _well_formed(map_fields(data), data)
