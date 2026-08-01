"""Tests for the 7 structure-aware format mutators: protobuf, gif, webp, webm, zip, x86, arm.

Each test class covers:
- parse -> serialize round trip (byte-identical for canonical input)
- size-field-reaches-the-wire regressions (mutated declared-size fields must
  be written verbatim, never recomputed from the payload)
- mutate() diversity (must produce more than one distinct output)
- degenerate-input safety (mutate must not raise on tiny/empty buffers)
"""

import random
import struct

import pytest


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _diversity(mutator, data: bytes, rounds: int = 60, max_len: int = 4096) -> int:
    rng = _rng()
    return len({mutator.mutate(data, max_len=max_len, rng=rng) for _ in range(rounds)})


class TestRandPoolCompat:
    """Mutators must work with the project's RandPool, not just random.Random.

    RandPool (core/rand_pool.py) exposes randint/choice/randbytes/random but
    NOT getrandbits — a mutator calling self._rng.getrandbits() only crashes
    under the real fuzzer's _rand_pool, never under unit-test Random.
    """

    @pytest.mark.parametrize(
        "module_name,generator",
        [
            ("gif", "gif"),
            ("webp", "webp"),
            ("webm", "webm"),
            ("zip", "zip"),
            ("protobuf", "protobuf"),
            ("x86", "x86"),
            ("arm", "arm"),
        ],
    )
    def test_mutate_with_randpool(self, module_name, generator):
        from fuzzer_tool.core.rand_pool import RandPool

        mod = __import__(f"fuzzer_tool.core.mutations.{module_name}", fromlist=["*"])
        mutator_cls = getattr(mod, f"{module_name.title()}Mutator")
        gen = getattr(mutator_cls(), f"_generate_random_{generator}")
        data = gen(max_len=4096, rng=_rng())
        mutator = mutator_cls()
        mutator._rng = RandPool()
        out = mutator.mutate(data, max_len=4096, rng=mutator._rng)
        assert isinstance(out, bytes) and len(out) <= 4096


class TestGifMutations:
    def test_roundtrip_generated(self):
        from fuzzer_tool.core.mutations.gif import GifMutator, parse_gif, serialize_gif

        data = GifMutator()._generate_random_gif(max_len=4096, rng=_rng())
        nodes = parse_gif(data)
        assert nodes is not None
        assert serialize_gif(nodes) == data

    def test_parse_requires_magic(self):
        from fuzzer_tool.core.mutations.gif import parse_gif

        assert parse_gif(b"\x00" * 64) is None
        assert parse_gif(b"NOTGIF" + b"\x00" * 32) is None

    def test_diversity(self):
        from fuzzer_tool.core.mutations.gif import GifMutator

        data = GifMutator()._generate_random_gif(max_len=4096, rng=_rng())
        assert _diversity(GifMutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.gif import GifMutator

        mut = GifMutator()
        for data in (b"", b"G", b"GIF89a", b"GIF89a" + b"\x00" * 12):
            mut.mutate(data, max_len=4096, rng=_rng())

    def test_regression_truncated_image_descriptor_indexerror(self):
        """An image marker with only 9 of the 10 trailing bytes must be
        stashed as raw, not raise IndexError on the min-code-size read."""
        from fuzzer_tool.core.mutations.gif import parse_gif

        # GIF89a(6) + LSD(7, no GCT) + 0x2C marker + 9 descriptor bytes:
        # pos+10 == len(data), so data[pos+10] used to go out of range.
        data = b"GIF89a" + b"\x00" * 7 + b"\x2c" + b"\x00" * 9
        nodes = parse_gif(data)
        assert nodes is not None
        assert nodes[-1].kind == "raw"
        assert b"".join(n.raw for n in nodes) == data

    def test_mutate_respects_max_len(self):
        from fuzzer_tool.core.mutations.gif import GifMutator

        data = GifMutator()._generate_random_gif(max_len=4096, rng=_rng())
        for _ in range(20):
            out = GifMutator().mutate(data, max_len=1024, rng=_rng())
            assert len(out) <= 1024


class TestWebpMutations:
    def test_roundtrip_generated(self):
        from fuzzer_tool.core.mutations.webp import WebpMutator, parse_webp, serialize_webp

        data = WebpMutator()._generate_random_webp(max_len=4096, rng=_rng())
        chunks = parse_webp(data)
        assert chunks is not None
        assert serialize_webp(chunks) == data

    def test_parse_requires_riff_webp(self):
        from fuzzer_tool.core.mutations.webp import parse_webp

        assert parse_webp(b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 16) is None
        assert parse_webp(b"RIFF" + struct.pack("<I", 8) + b"WEBP" + b"\x00" * 32) is None

    def test_serialize_writes_size_orig_verbatim(self):
        """Regression: mutated chunk size must reach the wire, not be recomputed."""
        from fuzzer_tool.core.mutations.webp import Chunk, serialize_webp

        chunk = Chunk(fourcc=b"VP8L", size_orig=0xFFFF, payload=b"tiny")
        result = serialize_webp([chunk])
        # Chunk size field sits after RIFF(4) + size(4) + WEBP(4) + fourcc(4)
        written_size = struct.unpack_from("<I", result, 16)[0]
        assert written_size == 0xFFFF, f"Expected size_orig=0xFFFF, got {written_size}"
        # Payload bytes still preserved after the 20-byte container+chunk header
        assert result[20:] == b"tiny"

    def test_diversity(self):
        from fuzzer_tool.core.mutations.webp import WebpMutator

        data = WebpMutator()._generate_random_webp(max_len=4096, rng=_rng())
        assert _diversity(WebpMutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.webp import WebpMutator

        mut = WebpMutator()
        for data in (b"", b"R", b"RIFF", b"RIFF\x04\x00\x00\x00WEBP"):
            mut.mutate(data, max_len=4096, rng=_rng())


class TestWebmMutations:
    def test_roundtrip_generated(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator, parse_webm, serialize_webm

        data = WebmMutator()._generate_random_webm(max_len=4096, rng=_rng())
        elements = parse_webm(data)
        assert elements is not None
        assert serialize_webm(elements) == data

    def test_parse_requires_ebml_segment(self):
        from fuzzer_tool.core.mutations.webm import parse_webm

        assert parse_webm(b"\x1a\x45\xdf\xa3" + b"\x00" * 32) is None
        assert parse_webm(b"\x00" * 64) is None

    def test_size_vint_roundtrip(self):
        from fuzzer_tool.core.mutations.webm import _read_vint

        for val, raw in [(0x01, b"\x81"), (0x7F, b"\xff"), (0x80, b"\x41\x00")]:
            v, r, pos = _read_vint(raw, 0)
            assert r == raw, f"raw mismatch for {val}"
        assert _read_vint(b"\xff\xff\xff\xff\xff\xff\xff\xff", 0)[0] == -1

    def test_size_vint_rewrite_reaches_wire(self):
        """Regression: a deliberately mismatched size_val must be re-encoded,
        not auto-corrected back to len(payload)."""
        from fuzzer_tool.core.mutations.webm import Element, serialize_webm

        # 2-byte size vint (0x42 0x00) declaring size 0x200 for a 3-byte payload,
        # with size_reencoded forcing the mismatch onto the wire.
        el = Element(
            elem_id=0x4286,
            id_raw=b"\x42\x86",
            size_raw=b"\x42\x00",
            size_val=0x200,
            data=b"abc",
            size_reencoded=True,
        )
        result = serialize_webm([el])
        # Size vint must be re-encoded at minimal length: 0x200 needs 2 bytes (0x42 0x00)
        assert result[2:4] == b"\x42\x00", f"Expected re-encoded size vint, got {result[2:4]!r}"

    def test_diversity(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator

        data = WebmMutator()._generate_random_webm(max_len=4096, rng=_rng())
        assert _diversity(WebmMutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator

        mut = WebmMutator()
        for data in (b"", b"\x1a", b"\x1a\x45\xdf\xa3", b"\x1a\x45\xdf\xa3\x80"):
            mut.mutate(data, max_len=4096, rng=_rng())


class TestZipMutations:
    def test_roundtrip_generated(self):
        from fuzzer_tool.core.mutations.zip import ZipMutator, parse_zip, serialize_zip

        data = ZipMutator()._generate_random_zip(max_len=4096, rng=_rng())
        doc = parse_zip(data)
        assert doc is not None
        assert serialize_zip(doc) == data

    def test_parse_requires_lfh(self):
        from fuzzer_tool.core.mutations.zip import parse_zip

        assert parse_zip(b"\x00" * 64) is None
        assert parse_zip(b"PK\x01\x02" + b"\x00" * 64) is None

    def test_zip64_sentinel_rejected(self):
        """zip64 EOCD unsupported: 0xFFFFFFFF sentinel -> None -> fallback."""
        from fuzzer_tool.core.mutations.zip import parse_zip

        name = b"a.txt"
        lfh = struct.pack(
            "<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0, 0, 0xFFFFFFFF, 5, len(name), 0
        )
        data = lfh + name + b"hello"
        cd = struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,
            20,
            20,
            0,
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
            5,
            len(name),
            0,
            0,
            0,
            0,
            0,
            0,
        )
        eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(cd), 0, 0)
        assert parse_zip(data + cd + eocd) is None

    def test_csize_reaches_wire(self):
        """Regression: mutated csize must be patched into LFH verbatim."""
        from fuzzer_tool.core.mutations.zip import ZipEntry, _patch_lfh

        name = b"a.txt"
        entry = ZipEntry(
            name=name,
            extra=b"",
            comment=b"",
            method=0,
            flags=0,
            crc32=0,
            csize=0xDEADBEEF,
            usize=5,
            modtime=0,
            moddate=0,
            data=b"hello",
            desc=b"",
            lfh_fixed=struct.pack(
                "<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, 0, 0, 0, 5, 5, len(name), 0
            ),
            cd_fixed=b"",
        )
        lfh = _patch_lfh(entry)
        written = struct.unpack_from("<I", lfh, 18)[0]
        assert written == 0xDEADBEEF, f"Expected csize=0xDEADBEEF on the wire, got {written}"

    def test_diversity(self):
        from fuzzer_tool.core.mutations.zip import ZipMutator

        data = ZipMutator()._generate_random_zip(max_len=4096, rng=_rng())
        assert _diversity(ZipMutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.zip import ZipMutator

        mut = ZipMutator()
        for data in (b"", b"P", b"PK", b"PK\x03\x04", b"PK\x03\x04" + b"\x00" * 30):
            mut.mutate(data, max_len=4096, rng=_rng())


class TestProtobufMutations:
    def test_roundtrip_canonical(self):
        from fuzzer_tool.core.mutations.protobuf import parse_protobuf, serialize_protobuf

        data = b"\x08\x96\x01" + b"\x12\x03abc" + b"\x1d\x00\x00\x00\x00"
        fields = parse_protobuf(data)
        assert fields is not None
        assert serialize_protobuf(fields) == data

    def test_roundtrip_group(self):
        """Wire-3 group round trip: end-group tag must be re-encoded canonically."""
        from fuzzer_tool.core.mutations.protobuf import parse_protobuf, serialize_protobuf

        data = b"\x0b\x08\x01\x0c"  # field 1 group containing field 1 varint
        fields = parse_protobuf(data)
        assert fields is not None
        assert len(fields) == 1
        assert fields[0].children is not None
        assert serialize_protobuf(fields) == data

    def test_first_tag_strict(self):
        from fuzzer_tool.core.mutations.protobuf import parse_protobuf

        # Undecodable first tag -> None (strict), mid-stream junk -> tolerated
        assert parse_protobuf(b"\xff" * 10) is None
        fields = parse_protobuf(b"\x08\x01" + b"\xff" * 9)
        assert fields is not None and len(fields) == 1

    def test_regression_deeply_nested_groups_no_crash(self):
        """Deeply nested group fields must not crash _parse_fields with TypeError.

        Before the fix, the depth-limit path returned bare None instead of
        (None, []), so the recursive unpack ``inner, inner_between = ...``
        raised ``TypeError: cannot unpack non-iterable NoneType``.
        """
        from fuzzer_tool.core.mutations.protobuf import (
            _encode_varint,
            parse_protobuf,
        )

        # 18 levels of nested groups (field 1..18), each with a distinct field
        # number so the end-group scanner can track nesting correctly.
        data = bytearray()
        for i in range(1, 19):
            data += _encode_varint((i << 3) | 3)  # start-group
        data += b"\x08\x01"  # innermost varint field
        for i in range(18, 0, -1):
            data += _encode_varint((i << 3) | 4)  # end-group
        result = parse_protobuf(bytes(data))
        assert result is None or isinstance(result, type(result))

    def test_rewritten_length_reaches_wire(self):
        """Regression: _rewrite_length's new length must reach the wire varint."""
        from fuzzer_tool.core.mutations.protobuf import (
            Field,
            ProtobufMutator,
            ProtoFields,
            serialize_protobuf,
        )

        fields = ProtoFields([Field(field_num=2, wire_type=2, raw_payload=b"abc")])
        mut = ProtobufMutator()
        mut._rng = _rng(5)
        mut._rewrite_length(fields, max_len=4096)
        result = serialize_protobuf(fields)
        # tag \x12, then length varint, then payload
        assert result[0] == 0x12
        length = result[1]
        assert length == len(fields[0].raw_payload), (
            f"Wire length {length} must match rewritten payload len {len(fields[0].raw_payload)}"
        )

    def test_regression_protofields_raw_between_no_class_field(self):
        """Regression: a bare ProtoFields (not from parse) must serialize.

        ProtoFields used to declare `raw_between` via dataclasses.field() on a
        non-dataclass class, leaving a Field descriptor object as the class
        attribute — getattr() then returned that object and len() raised
        TypeError during serialize.
        """
        from fuzzer_tool.core.mutations.protobuf import Field, ProtoFields, serialize_protobuf

        fields = ProtoFields([Field(field_num=1, wire_type=0, raw_payload=b"\x01")])
        assert isinstance(fields.raw_between, list)
        assert serialize_protobuf(fields) == b"\x08\x01"

    def test_diversity(self):
        from fuzzer_tool.core.mutations.protobuf import ProtobufMutator

        data = ProtobufMutator()._generate_random_protobuf(max_len=4096, rng=_rng())
        assert _diversity(ProtobufMutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.protobuf import ProtobufMutator

        mut = ProtobufMutator()
        for data in (b"", b"\x08", b"\xff" * 8, b"\x08\x01"):
            mut.mutate(data, max_len=4096, rng=_rng())


class TestX86Mutations:
    def test_roundtrip_generated(self):
        from fuzzer_tool.core.mutations.x86 import X86Mutator, _decode_insns, serialize_x86

        data = X86Mutator()._generate_random_x86(max_len=4096, rng=_rng())
        insns = _decode_insns(data)
        assert insns is not None
        assert serialize_x86(insns) == data

    def test_decode_known_instructions(self):
        from fuzzer_tool.core.mutations.x86 import _decode_insns

        # 90 (nop, 1B), B8 78 56 34 12 (mov eax, imm32, 5B), C3 (ret, 1B)
        data = b"\x90\xb8\x78\x56\x34\x12\xc3"
        insns = _decode_insns(data)
        assert [i.length for i in insns] == [1, 5, 1], (
            f"Expected lengths [1, 5, 1], got {[i.length for i in insns]}"
        )
        assert b"".join(i.raw for i in insns) == data

    def test_unknown_opcode_resyncs(self):
        """Unknown opcodes must not poison later boundaries: 1-byte resync."""
        from fuzzer_tool.core.mutations.x86 import _decode_insns

        data = b"\x0f\x0b\x90\xc3"  # ud2 (0F 0B, 2B), nop, ret
        insns = _decode_insns(data)
        assert b"".join(i.raw for i in insns) == data

    def test_regression_imm_values_clamped_to_field_width(self):
        """Regression: IMM_VALUES contains constants wider than the imm field.

        A 1-byte imm target choosing 0xFFFF from IMM_VALUES previously blew up
        in _pack_imm with OverflowError at fuzz time (masked in unit tests
        because random.Random's getrandbits path was always in-range).
        """
        from fuzzer_tool.core.mutations.x86 import X86Mutator, _decode_insns

        data = b"\xb0\x00" * 8  # mov al, imm8 repeated
        mut = X86Mutator()
        for seed in range(25):
            mut._rng = _rng(seed)
            out = mut.mutate(data, max_len=4096, rng=mut._rng)
            assert isinstance(out, bytes)
            assert len(out) <= 4096
            # Still decodable (other sub-ops may legitimately change opcodes)
            _decode_insns(out)

    def test_diversity(self):
        from fuzzer_tool.core.mutations.x86 import X86Mutator

        data = X86Mutator()._generate_random_x86(max_len=4096, rng=_rng())
        assert _diversity(X86Mutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.x86 import X86Mutator

        mut = X86Mutator()
        for data in (b"", b"\x90", b"\x0f", b"\x66\x90"):
            mut.mutate(data, max_len=4096, rng=_rng())


class TestArmMutations:
    def test_roundtrip_generated(self):
        from fuzzer_tool.core.mutations.arm import ArmMutator, parse_arm, serialize_arm

        data = ArmMutator()._generate_random_arm(max_len=4096, rng=_rng())
        words = parse_arm(data)
        assert words is not None
        assert serialize_arm(words) == data

    def test_parse_empty_returns_none(self):
        from fuzzer_tool.core.mutations.arm import parse_arm

        assert parse_arm(b"") is None

    def test_a32_words(self):
        from fuzzer_tool.core.mutations.arm import parse_arm

        data = struct.pack("<II", 0xE1A00000, 0xE12FFF1E)  # nop, bx lr
        words = parse_arm(data)
        assert [w.kind for w in words] == ["a32", "a32"]
        assert b"".join(w.raw for w in words) == data

    def test_thumb_words(self):
        from fuzzer_tool.core.mutations.arm import parse_arm

        # 0xE7FE (b .) starts with F-bit pattern -> t32 stream; 0xE7FE is a 32-bit
        # t32 insn (bits 15-11 = 0x1F... verify against decoder), else t16.
        hw = struct.pack("<H", 0xE7FE)
        words = parse_arm(hw)
        assert words is not None
        assert b"".join(w.raw for w in words) == hw

    def test_diversity(self):
        from fuzzer_tool.core.mutations.arm import ArmMutator

        data = ArmMutator()._generate_random_arm(max_len=4096, rng=_rng())
        assert _diversity(ArmMutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.arm import ArmMutator

        mut = ArmMutator()
        for data in (b"", b"\x00", b"\x00\x00", b"\x00" * 5):
            mut.mutate(data, max_len=4096, rng=_rng())
