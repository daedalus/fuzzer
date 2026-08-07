"""Tests for simultaneous satisfaction of coupled derived fields.

The property under test throughout: after ``repair``, *every* field holds at
once. Repairing fields one at a time does not achieve this — fixing a length
changes bytes a checksum covers, so the checksum goes stale again.
"""

import binascii
import random
import struct
import zlib

import pytest

from fuzzer_tool.core.field_constraints import (
    CHECKSUM_CRC32,
    CHECKSUM_SUM,
    CONSTANT,
    LENGTH,
    Field,
    _solve_self_referential_crc,
    _solve_self_referential_sum,
    check,
    compute_field,
    dependency_order,
    gzip_fields,
    png_fields,
    repair,
    satisfied,
    solve_coupled,
)
from fuzzer_tool.core.operator_registry import REGISTRY


def _chunk(ctype: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + ctype
        + payload
        + struct.pack(">I", binascii.crc32(ctype + payload) & 0xFFFFFFFF)
    )


def _png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(b"\x00" * 20))
        + _chunk(b"IEND", b"")
    )


def _len_crc_layout(payload: bytes) -> tuple[bytes, list[Field]]:
    """[len(4)][payload][crc(4)] with the CRC covering the length field."""
    buf = struct.pack(">I", 0) + payload + b"\x00" * 4
    fields = [
        Field(LENGTH, offset=0, width=4, span=(4, 4 + len(payload)), name="len"),
        Field(
            CHECKSUM_CRC32,
            offset=4 + len(payload),
            width=4,
            span=(0, 4 + len(payload)),
            name="crc",
        ),
    ]
    return buf, fields


class TestFieldModel:
    def test_length_field_computes_span_size(self):
        field = Field(LENGTH, offset=0, width=4, span=(4, 20))
        assert compute_field(field, b"\x00" * 32) == 16

    def test_length_adjust_is_applied(self):
        """Some formats count a header the span excludes."""
        field = Field(LENGTH, offset=0, width=4, span=(4, 20), adjust=4)
        assert compute_field(field, b"\x00" * 32) == 20

    def test_crc_field_computes_over_span(self):
        data = b"HEAD" + b"payload!" + b"\x00" * 4
        field = Field(CHECKSUM_CRC32, offset=12, width=4, span=(4, 12))
        assert compute_field(field, data) == binascii.crc32(b"payload!") & 0xFFFFFFFF

    def test_sum_field_wraps_at_width(self):
        field = Field(CHECKSUM_SUM, offset=0, width=1, span=(1, 5))
        assert compute_field(field, b"\x00\xff\xff\x01\x01") == (0xFF + 0xFF + 1 + 1) & 0xFF

    def test_constant_field(self):
        field = Field(CONSTANT, offset=0, width=4, value=0xCAFEBABE)
        assert compute_field(field, b"\x00" * 8) == 0xCAFEBABE

    def test_span_containment_detection(self):
        outer = Field(CHECKSUM_CRC32, offset=20, width=4, span=(0, 20))
        inner = Field(LENGTH, offset=0, width=4, span=(4, 20))
        assert outer.span_contains_field(inner)
        assert not inner.span_contains_field(outer)

    def test_self_reference_detection(self):
        covering = Field(CHECKSUM_CRC32, offset=8, width=4, span=(0, 16))
        excluded = Field(CHECKSUM_CRC32, offset=12, width=4, span=(0, 12))
        assert covering.is_self_referential()
        assert not excluded.is_self_referential()


class TestDependencyOrder:
    def test_contained_field_is_written_first(self):
        _, fields = _len_crc_layout(b"payload")
        ordered, cyclic = dependency_order(fields)
        assert [f.name for f in ordered] == ["len", "crc"]
        assert cyclic == []

    def test_self_referential_field_is_reported_cyclic(self):
        field = Field(CHECKSUM_CRC32, offset=8, width=4, span=(0, 16), name="self")
        ordered, cyclic = dependency_order([field])
        assert ordered == []
        assert [f.name for f in cyclic] == ["self"]

    def test_mutual_containment_is_cyclic(self):
        a = Field(CHECKSUM_CRC32, offset=0, width=4, span=(0, 16), name="a")
        b = Field(CHECKSUM_CRC32, offset=8, width=4, span=(0, 16), name="b")
        _, cyclic = dependency_order([a, b])
        assert {f.name for f in cyclic} == {"a", "b"}

    def test_independent_fields_all_orderable(self):
        fields = [
            Field(LENGTH, offset=0, width=4, span=(8, 20), name="a"),
            Field(LENGTH, offset=4, width=4, span=(20, 32), name="b"),
        ]
        ordered, cyclic = dependency_order(fields)
        assert len(ordered) == 2
        assert cyclic == []

    def test_empty_input(self):
        assert dependency_order([]) == ([], [])


class TestSimultaneousRepair:
    """The core claim: coupled fields end up all correct at once."""

    def test_length_and_crc_repaired_together(self):
        payload = b"MUTATED-PAYLOAD-XYZ"
        buf, fields = _len_crc_layout(payload)
        assert len(check(fields, buf)) == 2  # both stale

        fixed = repair(fields, buf)
        assert fixed is not None
        assert satisfied(fields, fixed)
        assert struct.unpack(">I", fixed[0:4])[0] == len(payload)
        assert struct.unpack(">I", fixed[-4:])[0] == binascii.crc32(fixed[:-4]) & 0xFFFFFFFF

    def test_sequential_repair_would_leave_the_checksum_stale(self):
        """Why ordering matters: writing the CRC before the length breaks it."""
        payload = b"MUTATED-PAYLOAD-XYZ"
        buf, fields = _len_crc_layout(payload)
        length_field, crc_field = fields

        out = bytearray(buf)
        # Deliberately wrong order: CRC first, then length.
        out[crc_field.offset : crc_field.end] = struct.pack(
            ">I", compute_field(crc_field, bytes(out))
        )
        out[length_field.offset : length_field.end] = struct.pack(
            ">I", compute_field(length_field, bytes(out))
        )
        assert not satisfied(fields, bytes(out))

    def test_valid_png_is_already_satisfied(self):
        png = _png()
        assert satisfied(png_fields(png), png)

    def test_png_payload_mutation_is_repaired(self):
        png = bytearray(_png())
        idat_payload = 8 + 12 + 13 + 8
        png[idat_payload + 2] ^= 0xFF
        raw = bytes(png)
        fields = png_fields(raw)
        assert check(fields, raw)

        fixed = repair(fields, raw)
        assert fixed is not None
        assert satisfied(png_fields(fixed), fixed)

    def test_repair_preserves_length(self):
        png = _png()
        fixed = repair(png_fields(png), png)
        assert len(fixed) == len(png)

    def test_empty_field_list_is_a_passthrough(self):
        assert repair([], b"anything") == b"anything"

    def test_field_past_end_of_data_fails_cleanly(self):
        field = Field(LENGTH, offset=100, width=4, span=(0, 10))
        assert repair([field], b"short") is None


class TestSelfReferentialCrc:
    """CRC-32 is affine over GF(2), so a field inside its own span can be
    solved in closed form instead of searched — 32 probes and a Gaussian
    elimination, no solver."""

    def _buf(self, payload: bytes) -> tuple[bytearray, Field]:
        buf = bytearray(payload + b"\x00" * 4)
        field = Field(CHECKSUM_CRC32, offset=len(buf) - 4, width=4, span=(0, len(buf)))
        return buf, field

    def test_fixpoint_is_reached(self):
        """Not every buffer admits a fixpoint — the linear system can be
        inconsistent — so search for one that does and verify it closes.
        """
        for i in range(64):
            buf, field = self._buf(b"PAYLOAD-DATA-HERE-%02d" % i)
            value = _solve_self_referential_crc(bytes(buf), field)
            if value is None:
                continue
            buf[field.offset : field.end] = value.to_bytes(4, "big")
            assert binascii.crc32(bytes(buf)) & 0xFFFFFFFF == value
            return
        pytest.fail("no fixpoint found across 64 payloads; the algebra is broken")

    @pytest.mark.parametrize("big_endian", [True, False])
    def test_both_byte_orders(self, big_endian):
        buf = bytearray(b"SOME-PAYLOAD" + b"\x00" * 4)
        field = Field(
            CHECKSUM_CRC32,
            offset=len(buf) - 4,
            width=4,
            span=(0, len(buf)),
            big_endian=big_endian,
        )
        value = _solve_self_referential_crc(bytes(buf), field)
        if value is None:
            pytest.skip("no fixpoint exists for this buffer")
        order = "big" if big_endian else "little"
        buf[field.offset : field.end] = value.to_bytes(4, order)
        assert binascii.crc32(bytes(buf)) & 0xFFFFFFFF == int.from_bytes(
            buf[field.offset : field.end], order
        )

    def test_never_returns_a_wrong_value(self):
        """Randomised: a returned value must always be a true fixpoint.

        Not every buffer admits one — the linear system can be inconsistent —
        and None is the correct answer there. What must never happen is a
        confidently wrong value.
        """
        rng = random.Random(0)
        solved = wrong = none = 0
        for _ in range(120):
            payload = bytes(rng.randrange(256) for _ in range(rng.randint(8, 60)))
            buf, field = self._buf(payload)
            value = _solve_self_referential_crc(bytes(buf), field)
            if value is None:
                none += 1
                continue
            buf[field.offset : field.end] = value.to_bytes(4, "big")
            if binascii.crc32(bytes(buf)) & 0xFFFFFFFF == value:
                solved += 1
            else:
                wrong += 1
        assert wrong == 0
        assert solved > 0, "solver never succeeded; the algebra is broken"

    def test_non_32_bit_field_is_declined(self):
        buf = bytearray(b"data" + b"\x00\x00")
        field = Field(CHECKSUM_CRC32, offset=4, width=2, span=(0, len(buf)))
        assert _solve_self_referential_crc(bytes(buf), field) is None

    def test_repair_routes_self_referential_fields(self):
        buf, field = self._buf(b"PAYLOAD-DATA-HERE")
        fixed = repair([field], bytes(buf))
        if fixed is not None:
            assert satisfied([field], fixed)


class TestSelfReferentialSum:
    def test_fixpoint_is_reached(self):
        """A width-1 self-sum needs ``v == (rest + v) mod 256``, i.e. rest
        must be a multiple of 256 — solvable only for some payloads.
        """
        for i in range(512):
            buf = bytearray(b"ABCDEFGH" + bytes([i & 0xFF]) + b"\x00")
            field = Field(CHECKSUM_SUM, offset=9, width=1, span=(0, 10))
            value = _solve_self_referential_sum(bytes(buf), field)
            if value is None:
                continue
            buf[9] = value
            assert sum(buf) & 0xFF == buf[9]
            return
        pytest.fail("no self-sum fixpoint found across 512 payloads")

    def test_wide_sum_field(self):
        buf = bytearray(b"PAYLOAD" + b"\x00\x00")
        field = Field(CHECKSUM_SUM, offset=7, width=2, span=(0, 9))
        value = _solve_self_referential_sum(bytes(buf), field)
        if value is not None:
            buf[7:9] = value.to_bytes(2, "big")
            assert sum(buf) & 0xFFFF == int.from_bytes(buf[7:9], "big")


class TestCoupledFields:
    """Fields constrained against each other have no recomputation order
    that satisfies both — this is where a solver is required."""

    def test_sum_constraint_between_two_fields(self):
        z3 = pytest.importorskip("z3")
        assert z3 is not None
        data = b"\x00" * 8
        fields = [
            Field(LENGTH, offset=0, width=2, name="a"),
            Field(LENGTH, offset=2, width=2, name="b"),
        ]
        out = solve_coupled(fields, data, [("sum_eq", 0, 1, 512)])
        assert out is not None
        total = int.from_bytes(out[0:2], "big") + int.from_bytes(out[2:4], "big")
        assert total == 512

    def test_ordering_constraint(self):
        pytest.importorskip("z3")
        data = b"\x00" * 8
        fields = [
            Field(LENGTH, offset=0, width=2, name="a"),
            Field(LENGTH, offset=2, width=2, name="b"),
        ]
        out = solve_coupled(fields, data, [("lt", 0, 1, 0)])
        assert out is not None
        assert int.from_bytes(out[0:2], "big") < int.from_bytes(out[2:4], "big")

    def test_unsatisfiable_returns_none(self):
        pytest.importorskip("z3")
        data = b"\x00" * 8
        fields = [
            Field(CONSTANT, offset=0, width=2, value=100, name="a"),
            Field(CONSTANT, offset=2, width=2, value=100, name="b"),
        ]
        assert solve_coupled(fields, data, [("sum_eq", 0, 1, 999)]) is None


class TestPngExtraction:
    def test_finds_two_fields_per_chunk(self):
        fields = png_fields(_png())
        assert len(fields) == 6  # IHDR, IDAT, IEND
        assert sum(1 for f in fields if f.kind == LENGTH) == 3
        assert sum(1 for f in fields if f.kind == CHECKSUM_CRC32) == 3

    def test_non_png_returns_nothing(self):
        assert png_fields(b"GIF89a" + b"\x00" * 32) == []
        assert png_fields(b"") == []

    def test_truncated_png_stops_cleanly(self):
        assert png_fields(_png()[:20]) is not None

    def test_absurd_chunk_length_does_not_overrun(self):
        """Chunk lengths are attacker-controlled."""
        bad = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 0xFFFFFFF) + b"IHDR" + b"\x00" * 8
        assert png_fields(bad) == []

    def test_crc_span_excludes_the_length_field(self):
        """PNG CRC covers type+data but not the length — the two are coupled
        through the data span, not by containment of the CRC field."""
        fields = png_fields(_png())
        length_field = fields[0]
        crc_field = fields[1]
        assert crc_field.span[0] == length_field.offset + 4


class TestGzipExtraction:
    def test_locates_trailer_fields(self):
        blob = b"\x1f\x8b\x08" + b"\x00" * 20
        fields = gzip_fields(blob)
        assert len(fields) == 2
        assert all(not f.big_endian for f in fields)

    def test_non_gzip_returns_nothing(self):
        assert gzip_fields(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20) == []


class TestOperatorRegistration:
    def test_registered_as_format_operator(self):
        assert "field_repair" in REGISTRY.names()
        assert REGISTRY.category_of("field_repair") == "format"

    def test_handler_exists(self):
        from fuzzer_tool.services.operators import OperatorEngine

        assert hasattr(OperatorEngine, "_op_field_repair")

    def test_sniffer_admits_png(self):
        assert "field_repair" in REGISTRY.available(None, _png())

    def test_sniffer_rejects_non_png(self):
        class _Fuzzer:
            dictionary = None
            _rand_pool = random.Random(1)

        assert "field_repair" not in REGISTRY.available(_Fuzzer(), b"plain text input")
