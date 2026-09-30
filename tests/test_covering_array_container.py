"""Tests for core/mutations/covering_array_container.py (RIFF VP8X, ISO-BMFF ftyp, ZIP LFH)."""

from __future__ import annotations

import io
import random
import struct
import zipfile

import pytest

from fuzzer_tool.core import covering_array as ca
from fuzzer_tool.core import field_spec
from fuzzer_tool.core.mutations import covering_array_container as cac
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY


class _Rng:
    def __init__(self, seed: int = 7):
        self._r = random.Random(seed)

    def randint(self, a, b):
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)


def _webp() -> bytes:
    vp8x = b"VP8X" + struct.pack("<I", 10) + bytes([0x10, 0, 0, 0]) + (99).to_bytes(3, "little")
    vp8x += (49).to_bytes(3, "little")
    body = b"WEBP" + vp8x + b"VP8 " + struct.pack("<I", 4) + b"\x00" * 4
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _mp4() -> bytes:
    ftyp = b"ftyp" + b"isom" + struct.pack(">I", 512) + b"isomiso2"
    return struct.pack(">I", 8 + len(ftyp) - 4) + ftyp + b"\x00\x00\x00\x08free"


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.txt", b"hello world" * 8)
    return buf.getvalue()


CASES = [
    ("covering_array_webp", cac.WebpCoveringArrayMutator, cac.WEBP_FIELDS, _webp),
    ("covering_array_isobmff", cac.IsobmffCoveringArrayMutator, cac.ISOBMFF_FIELDS, _mp4),
    ("covering_array_zip", cac.ZipCoveringArrayMutator, cac.ZIP_FIELDS, _zip),
]
IDS = [c[0] for c in CASES]


@pytest.mark.parametrize(("name", "cls", "fields", "make"), CASES, ids=IDS)
class TestPerFormat:
    def test_registered_as_format(self, name, cls, fields, make):
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == "format"

    def test_available_on_valid_only(self, name, cls, fields, make):
        m = cls()
        assert m.is_available(MutationContext(), make())
        assert not m.is_available(MutationContext(), b"")
        assert not m.is_available(MutationContext(), b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)

    def test_declines_truncated_header(self, name, cls, fields, make):
        # Adversarial: magic present but header cut inside the last field.
        m = cls()
        data = make()
        cut = max(f.end for f in fields) - 1
        assert not m.is_available(MutationContext(), data[:cut])
        assert m.mutate(data[:cut], _Rng()) is None

    def test_declines_wrong_magic(self, name, cls, fields, make):
        assert cls().mutate(b"nope" * 20, _Rng()) is None

    def test_fields_do_not_overlap_and_fit_inside_header(self, name, cls, fields, make):
        ordered = sorted(fields, key=lambda f: f.offset)
        for a, b in zip(ordered, ordered[1:], strict=False):
            assert a.end <= b.offset
        assert all(f.values and len(set(f.values)) == len(f.values) for f in fields)
        assert all(0 <= v <= f.max_value for f in fields for v in f.values)

    def test_touches_only_declared_fields(self, name, cls, fields, make):
        data = make()
        m = cls()
        for _ in range(len(fields) * 4):
            out = m.mutate(data, _Rng())
            if out is None:
                continue
            assert len(out) == len(data)
            covered = {i for f in fields for i in range(f.offset, f.end)}
            diff = {i for i in range(len(data)) if out[i] != data[i]}
            assert diff <= covered

    def test_max_len_truncates(self, name, cls, fields, make):
        out = cls().mutate(make(), _Rng(), max_len=max(f.end for f in fields))
        assert out is not None and len(out) <= max(f.end for f in fields)

    def test_full_sweep_is_pairwise_covering(self, name, cls, fields, make):
        # Falsification: replaying every row must verify as a t=2 array.
        m = cls()
        data = make()
        rng = _Rng()
        first = m.mutate(data, rng)
        assert first is not None
        n_rows = len(m._rows)  # noqa: SLF001
        rows = [field_spec.read_row(first, fields)]
        for _ in range(n_rows - 1):
            out = m.mutate(data, rng)
            rows.append(field_spec.read_row(out if out is not None else data, fields))
        assert ca.verify_coverage(rows, field_spec.value_sets(list(fields)), t=2)

    def test_row_equal_to_input_returns_none(self, name, cls, fields, make):
        m = cls()
        data = make()
        m.mutate(data, _Rng())
        same = field_spec.apply_row(data, fields, m._rows[0])  # noqa: SLF001
        m2 = cls()
        m2._rows = m._rows  # noqa: SLF001
        assert m2.mutate(same, _Rng()) is None

    def test_same_seed_same_sequence(self, name, cls, fields, make):
        data = make()
        a, b = cls(), cls()
        ra, rb = _Rng(3), _Rng(3)
        assert [a.mutate(data, ra) for _ in range(5)] == [b.mutate(data, rb) for _ in range(5)]


def _named(data: bytes, fields) -> dict[str, int]:
    names = [f.name for f in fields]
    return dict(zip(names, field_spec.read_row(data, fields), strict=True))


class TestOffsets:
    """Pin offsets against the real containers, not against our own constants."""

    def test_webp_offsets(self):
        d = _webp()
        row = _named(d, cac.WEBP_FIELDS)
        assert row["chunk_size"] == 10
        assert row["flags"] == 0x10
        assert row["width"] == 99
        assert row["height"] == 49

    def test_isobmff_offsets(self):
        d = _mp4()
        row = _named(d, cac.ISOBMFF_FIELDS)
        assert row["major"] == int.from_bytes(b"isom", "big")
        assert row["minor"] == 512
        assert row["size"] == 24

    def test_zip_offsets(self):
        d = _zip()
        row = _named(d, cac.ZIP_FIELDS)
        zi = zipfile.ZipFile(io.BytesIO(d)).infolist()[0]
        assert row["method"] == zi.compress_type
        assert row["comp_size"] == zi.compress_size
        assert row["file_size"] == zi.file_size
        assert row["name_len"] == len(b"a.txt")


class TestGateIsSpecific:
    def test_riff_without_vp8x_is_declined(self):
        wav = b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt " + b"\x00" * 40
        assert not cac.WebpCoveringArrayMutator().is_available(MutationContext(), wav)

    def test_mov_without_ftyp_is_declined(self):
        moov = struct.pack(">I", 16) + b"moov" + b"\x00" * 32
        assert not cac.IsobmffCoveringArrayMutator().is_available(MutationContext(), moov)

    def test_zip_end_record_only_is_declined(self):
        empty = b"PK\x05\x06" + b"\x00" * 18
        assert not cac.ZipCoveringArrayMutator().is_available(MutationContext(), empty)
