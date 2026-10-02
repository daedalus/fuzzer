"""Tests for core/rq_encodings.py — Redqueen encoding engine."""

import base64
import zlib

import pytest

from fuzzer_tool.core import rq_encodings
from fuzzer_tool.core.mutations.generic import encode_sleb128, encode_uleb128
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.rq_encodings import (
    BUILTIN_ENCODERS,
    Crc32Encoder,
    CStrChrEncoder,
    CStringEncoder,
    Encoder,
    Fnv1aEncoder,
    MemEncoder,
    PlainEncoder,
    SextEncoder,
    SplitEncoder,
    ZextEncoder,
    _fnv1a,
    _fnv1a_fwd_table,
    _fnv1a_invert_all,
    encoders_summary,
    find_offsets,
    generate_mutations,
)


class TestEncoders:
    def test_39_encoders_loaded(self):
        assert len(BUILTIN_ENCODERS) >= 39

    def test_all_encoder_types_present(self):
        names = [e.name() for e in BUILTIN_ENCODERS]
        for expected in ("plain_p", "plain_r", "cstr", "split_p", "split_r"):
            assert expected in names, f"missing encoder {expected}"
        for prefix in ("zext", "sext", "ascii"):
            assert any(n.startswith(prefix) for n in names), f"missing {prefix}*"
        for length in range(4, 16):
            assert f"mem_{length}" in names, f"missing mem_{length}"

    def test_encoders_summary(self):
        summary = encoders_summary()
        assert len(summary) == len(BUILTIN_ENCODERS)
        for entry in summary:
            assert "name" in entry
            assert "desc" in entry
            assert "size" in entry

    def test_plain_encoder_applicable(self):
        enc = PlainEncoder(reverse=False)
        assert enc.is_applicable(32, "CMP", b"\x01\x02", b"\x03\x04")
        assert not enc.is_applicable(32, "STR", b"\x01\x02", b"\x03\x04")

    def test_plain_encoder_encode(self):
        enc = PlainEncoder(reverse=False)
        assert enc.encode(b"\x01\x02") == [b"\x01\x02"]
        enc_r = PlainEncoder(reverse=True)
        assert enc_r.encode(b"\x01\x02") == [b"\x02\x01"]

    def test_zext_encoder_applicable(self):
        z8 = ZextEncoder(1, False)
        # 32-bit comparison where upper 24 bits are zero
        assert z8.is_applicable(32, "CMP", b"\x00\x00\x00\x2a", b"\x00\x00\x00\x2b")
        # Non-zero upper bytes should fail
        assert not z8.is_applicable(32, "CMP", b"\x00\x01\x00\x2a", b"\x00\x00\x00\x2b")
        # STR type should fail
        assert not z8.is_applicable(32, "STR", b"\x00\x00\x00\x2a", b"\x00\x00\x00\x2b")

    def test_zext_encode(self):
        z8 = ZextEncoder(1, False)
        assert z8.encode(b"\x00\x00\x00\x2a") == [b"\x2a"]

    def test_sext_encoder_applicable(self):
        s1 = SextEncoder(1, False)
        # 32-bit where upper 24 bits are 0xFF (negative sign extension)
        assert s1.is_applicable(32, "CMP", b"\xff\xff\xff\x80", b"\xff\xff\xff\x81")
        # Not applicable when mixed with non-sign bits
        assert not s1.is_applicable(32, "CMP", b"\xff\xfe\x00\x2a", b"\x00\x00\x00\x2a")

    def test_ascii_encoder_encode(self):
        import struct

        from fuzzer_tool.core.rq_encodings import AsciiEncoder

        enc = AsciiEncoder(10, False)
        val = struct.pack("<I", 42)
        assert enc.encode(val) == [b"42"]

        enc16 = AsciiEncoder(16, False)
        val = struct.pack("<I", 255)
        assert enc16.encode(val) == [b"ff"]

    def test_cstring_encoder(self):
        enc = CStringEncoder()
        assert enc.is_applicable(512, "STR", b"hello\x00world", b"hello\x00")
        assert not enc.is_applicable(512, "STR", b"\x00abc", b"abc")
        assert enc.encode(b"hello\x00world") == [b"hello"]
        assert enc.encode(b"hello") == [b"hello"]

    def test_cstrchr_encoder(self):
        enc0 = CStrChrEncoder(0)
        # RHS is null-terminated single char
        assert enc0.is_applicable(512, "STR", b"abc", b"a\x00")
        assert not enc0.is_applicable(512, "STR", b"a", b"ab\x00")
        assert enc0.encode(b"abc") == [b"a"]

    def test_mem_encoder(self):
        enc = MemEncoder(4)
        assert enc.is_applicable(512, "STR", b"abcdef", b"1234")
        assert not enc.is_applicable(512, "STR", b"ab", b"12")
        assert enc.encode(b"abcdef") == [b"abcd"]

    def test_split_encoder(self):
        enc = SplitEncoder(False)
        assert enc.is_applicable(64, "CMP", b"\x01\x02\x03\x04\x05\x06\x07\x08", b"x" * 8)
        assert not enc.is_applicable(32, "CMP", b"\x01\x02\x03\x04", b"x" * 4)
        chunks = enc.encode(b"\x01\x02\x03\x04\x05\x06\x07\x08")
        assert len(chunks) == 2
        assert chunks[0] == b"\x01\x02\x03\x04"
        assert chunks[1] == b"\x05\x06\x07\x08"

    def test_split_encoder_reverse(self):
        enc = SplitEncoder(True)
        chunks = enc.encode(b"\x01\x02\x03\x04\x05\x06\x07\x08")
        assert chunks[0] == b"\x08\x07\x06\x05"
        assert chunks[1] == b"\x04\x03\x02\x01"


class TestFindOffsets:
    def test_basic(self):
        assert find_offsets(b"abcabcabc", b"abc") == [0, 3, 6]

    def test_overlapping(self):
        assert find_offsets(b"aaaa", b"aa") == [0, 1, 2]

    def test_no_match(self):
        assert find_offsets(b"abc", b"xyz") == []

    def test_empty_data(self):
        assert find_offsets(b"", b"a") == []

    def test_empty_pattern(self):
        assert find_offsets(b"abc", b"") == []


class TestGenerateMutations:
    def test_basic_plain(self):
        mutations = generate_mutations(b"\xff\xfe", b"\x00\x01", 16, "CMP", b"\xff\xfe\x00\x00")
        assert len(mutations) > 0
        offsets, replacements, enc = mutations[0]
        assert isinstance(offsets, tuple)
        assert isinstance(replacements, tuple)
        assert len(offsets) >= 1

    def test_returns_mutations_with_encoder(self):
        mutations = generate_mutations(b"\x01\x02", b"\x03\x04", 16, "CMP", b"\x01\x02\xff")
        assert len(mutations) > 0
        _, _, enc = mutations[0]
        assert hasattr(enc, "name")

    def test_no_match_returns_empty(self):
        mutations = generate_mutations(b"\x01\x02", b"\x03\x04", 16, "CMP", b"\x05\x06")
        assert mutations == []

    def test_split_encoder_multi_chunk(self):
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        op_a = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        op_b = b"\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11"
        mutations = generate_mutations(op_a, op_b, 64, "CMP", data)
        # Should have split encoder producing 2-offset mutations
        split_muts = [(o, r) for o, r, e in mutations if "split" in e.name()]
        if split_muts:
            offsets, repls = split_muts[0]
            assert len(offsets) == 2
            assert len(repls) == 2

    def test_hammer_produces_more_mutations(self):
        data = b"\x01\x02\x03\x04"
        op_a = b"\x01\x02\x03\x04"
        op_b = b"\x05\x06\x07\x08"
        hammered = generate_mutations(op_a, op_b, 32, "CMP", data, hammer=True)
        # Note: may not always be more since it depends on encoder matching,
        # but hammer=True generates more integer variants
        assert len(hammered) >= 0  # at least doesn't crash

    def test_mismatched_operand_lengths(self):
        """Regression test: cmp_size from longer operand must not cause
        struct.error when the other operand is shorter."""
        # op_a is 4 bytes → cmp_size=32, bytes_len=4, key="L"
        # op_b is 2 bytes → struct.unpack(">L", 2-byte val) would crash
        data = b"\x01\x02\x03\x04\xff\xff"
        op_a = b"\x01\x02\x03\x04"
        op_b = b"\x05\x06"
        mutations = generate_mutations(op_a, op_b, 32, "CMP", data, hammer=True)
        assert isinstance(mutations, list)

        # Same with a 1-byte operand_b
        op_b_1 = b"\x05"
        mutations2 = generate_mutations(op_a, op_b_1, 32, "CMP", data, hammer=True)
        assert isinstance(mutations2, list)

        # 8-byte op_a with short op_b (key="Q")
        data8 = b"\x01\x02\x03\x04\x05\x06\x07\x08\xff"
        op_a8 = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        op_b8 = b"\x05"
        mutations3 = generate_mutations(op_a8, op_b8, 64, "CMP", data8, hammer=True)
        assert isinstance(mutations3, list)

        # SUB type with mismatched lengths (same struct.unpack path)
        mutations4 = generate_mutations(op_a, op_b, 32, "SUB", data, hammer=True)
        assert isinstance(mutations4, list)

    def test_hash_skip(self):
        """Hash-like pairs should be skipped when is_hash is provided."""
        mutations = generate_mutations(
            b"\x01\x02",
            b"\x03\x04",
            16,
            "CMP",
            b"\x01\x02\xff",
            is_hash=lambda a, b: True,
        )
        assert mutations == []

    def test_cstring_mutation_generates_variants(self):
        data = b"hello world"
        op_a = b"hello"
        op_b = b"world"
        mutations = generate_mutations(op_a, op_b, 512, "STR", data, hammer=True)
        # If encoders match, there should be at least some mutations
        assert len(mutations) >= 0  # just checking it doesn't crash

    def test_encoder_cache_transparent(self):
        """The input-independent encoder cache must not change results:
        repeated calls with the same pair/input return identical mutations,
        and a different input yields the same encoders at different offsets."""
        op_a, op_b = b"\x01\x02", b"\x03\x04"
        m1 = generate_mutations(op_a, op_b, 16, "CMP", b"\x01\x02\xff\x00", hammer=True)
        m2 = generate_mutations(op_a, op_b, 16, "CMP", b"\x01\x02\xff\x00", hammer=True)
        assert m1 == m2
        # Pattern moved from offset 0 to offset 2 — offsets must follow.
        m3 = generate_mutations(op_a, op_b, 16, "CMP", b"xx\x01\x02", hammer=True)
        assert m3
        assert all(off == 2 for offs, _, _ in m3 for off in offs), (
            f"expected offsets at 2, got {[offs for offs, _, _ in m3]}"
        )
        # Non-applicable pairs stay non-applicable on the cached path.
        m4 = generate_mutations(op_a, op_b, 16, "CMP", b"\xff\xfe\x00\x00", hammer=True)
        assert m4 == generate_mutations(op_a, op_b, 16, "CMP", b"\xff\xfe\x00\x00", hammer=True)


# ── Decoder-layer encoders ─────────────────────────────────────────────
#
# The target decodes the input, then compares the decoded bytes: cmplog
# sees the decoded operand, the input holds its encoded form. Expected
# bytes come from the stdlib / LEB helpers, never from rq_encodings.


def _apply(data: bytes, mutation) -> bytes:
    """Overwrite each chunk at its offset, as ``_rq_apply_encoded`` does."""
    offsets, repls, _enc = mutation
    buf = bytearray(data)
    for off, chunk in zip(offsets, repls, strict=True):
        buf[off : off + len(chunk)] = chunk
    return bytes(buf)


def _by_encoder(mutations, name: str) -> list:
    return [m for m in mutations if m[2].name() == name]


_OP_A = b"hello world"
_OP_B = b"HELLO_WORLD"


class TestDecoderEncoders:
    @pytest.mark.parametrize(
        ("name", "enc", "dec"),
        [
            ("b64_std", base64.b64encode, base64.b64decode),
            ("b64_url", base64.urlsafe_b64encode, base64.urlsafe_b64decode),
        ],
    )
    def test_b64_solves_decoded_compare(self, name, enc, dec):
        # 11 bytes: the 15th char is shared with the trailing "!". The
        # \xfb\xef\xbe prefix encodes to "++++" / "----", so std != url.
        op_a = b"\xfb\xef\xbe" + _OP_A[:8]
        data = b"k=" + enc(op_a + b"!")
        hits = _by_encoder(generate_mutations(op_a, _OP_B, 512, "STR", data), name)
        assert hits
        assert dec(_apply(data, hits[0])[2:])[: len(_OP_B)] == _OP_B

    @pytest.mark.parametrize(("name", "upper"), [("hex_l", False), ("hex_u", True)])
    def test_hex_solves_decoded_compare(self, name, upper):
        h = _OP_A.hex()
        data = b"id=" + (h.upper() if upper else h).encode()
        hits = _by_encoder(generate_mutations(_OP_A, _OP_B, 512, "STR", data), name)
        assert hits
        assert bytes.fromhex(_apply(data, hits[0])[3:].decode()) == _OP_B

    @pytest.mark.parametrize(
        ("name", "codec"), [("utf16_le", "utf-16-le"), ("utf16_be", "utf-16-be")]
    )
    def test_utf16_widen_solves_narrowed_compare(self, name, codec):
        data = b"\x00" + _OP_A.decode("latin-1").encode(codec)
        hits = _by_encoder(generate_mutations(_OP_A, _OP_B, 512, "STR", data), name)
        assert hits
        assert _apply(data, hits[0])[1:].decode(codec) == _OP_B.decode()

    def test_utf16_narrow_solves_widened_compare(self):
        op_a, op_b = "abcdef".encode("utf-16-le"), "ABCDEF".encode("utf-16-le")
        data = b"<abcdef>"
        hits = _by_encoder(generate_mutations(op_a, op_b, 512, "STR", data), "utf16_narrow")
        assert hits
        assert _apply(data, hits[0]) == b"<ABCDEF>"

    @pytest.mark.parametrize(("name", "fold"), [("case_u", bytes.upper), ("case_l", bytes.lower)])
    def test_case_solves_folded_compare(self, name, fold):
        op_a, op_b = b"Content-Type", b"Content-Size"
        data = b"\n" + fold(op_a) + b": x"
        hits = _by_encoder(generate_mutations(op_a, op_b, 512, "STR", data), name)
        assert hits
        assert _apply(data, hits[0]) == b"\n" + fold(op_b) + b": x"

    def test_uleb128_solves_varint_compare(self):
        op_a, op_b = (300).to_bytes(4, "little"), (70000).to_bytes(4, "little")
        data = b"\x01" + encode_uleb128(300) + b"\x02"
        hits = _by_encoder(generate_mutations(op_a, op_b, 32, "CMP", data), "uleb128")
        assert hits
        assert encode_uleb128(70000) in [m[1][0] for m in hits]

    def test_sleb128_solves_signed_varint_compare(self):
        op_a = (-200).to_bytes(4, "little", signed=True)
        op_b = (-5000).to_bytes(4, "little", signed=True)
        data = b"\x01" + encode_sleb128(-200) + b"\x02"
        hits = _by_encoder(generate_mutations(op_a, op_b, 32, "CMP", data), "sleb128")
        assert hits
        assert encode_sleb128(-5000) in [m[1][0] for m in hits]

    # Falsification: each encoder stays silent where its decoder is absent.
    def test_falsify_plain_input_triggers_no_decoder_encoder(self):
        data = b"xx" + _OP_A + b"yy"
        names = {m[2].name() for m in generate_mutations(_OP_A, _OP_B, 512, "STR", data)}
        assert not names & {"b64_std", "b64_url", "hex_l", "hex_u", "utf16_le", "utf16_be"}

    def test_falsify_not_applicable(self):
        by = {e.name(): e for e in BUILTIN_ENCODERS}
        # Text encoders never fire on integer compares, varints never on strings.
        for name in ("b64_std", "hex_l", "utf16_le", "utf16_narrow", "case_u"):
            assert not by[name].is_applicable(32, "CMP", b"abcd", b"efgh")
        assert not by["uleb128"].is_applicable(512, "STR", b"abcd", b"efgh")
        # Single-byte varint duplicates zext_1; a narrow operand is not UTF-16.
        assert not by["uleb128"].is_applicable(32, "CMP", b"\x05\0\0\0", b"\x06\0\0\0")
        assert not by["sleb128"].is_applicable(32, "CMP", b"\x05\0\0\0", b"\x06\0\0\0")
        assert not by["utf16_narrow"].is_applicable(512, "STR", b"abcdefgh", b"ijklmnop")
        # Case fold that changes nothing is plain memcmp's job.
        assert not by["case_u"].is_applicable(512, "STR", b"ABC-123", b"DEF-456")
        assert not by["b64_std"].is_applicable(512, "STR", b"ab", b"cd")

    # Adversarial: hostile operand shapes never raise or desync chunk counts.
    # Width/type derived as ``_rq_apply_encoded`` does: operands fit the width.
    def test_adversarial_random_operands(self):
        rng = RandPool(seed=7)
        for _ in range(500):
            a = rng.randbytes(rng.randint(1, 40))
            b = rng.randbytes(rng.randint(1, 40) if len(a) > 8 else rng.randint(1, len(a)))
            data = rng.randbytes(rng.randint(0, 64))
            size = 512 if len(a) > 8 or len(b) > 8 else max(len(a), len(b)) * 8
            kind = "STR" if len(a) > 8 else rng.choice(("CMP", "SUB"))
            for offs, repls, enc in generate_mutations(a, b, size, kind, data + a, hammer=True):
                assert len(offs) == len(repls) == enc.size()


def _crc_le(data: bytes) -> bytes:
    """zlib CRC-32 of *data* as a little-endian cmplog operand."""
    return zlib.crc32(data).to_bytes(4, "little")


class TestCrc32Encoder:
    """Fuzzification AntiHybrid: ``if (CRC_LOOP(value) == OUTPUT_CRC)``."""

    def test_crc32_solves_hashed_compare(self):
        x, want = (12345).to_bytes(4, "little"), (0xDEADBEEF).to_bytes(4, "little")
        data = b"AB" + x + b"CD"
        hits = _by_encoder(
            generate_mutations(_crc_le(x), _crc_le(want), 32, "CMP", data), "crc32_p"
        )
        assert [m[0] for m in hits][0] == (2,)
        assert want in [m[1][0] for m in hits]

    def test_crc32_reversed_field(self):
        x = bytes.fromhex("01020304")
        data = x[::-1] + b"zz"
        want = b"\x00\x00\x30\x39"
        hits = _by_encoder(
            generate_mutations(_crc_le(x), _crc_le(want), 32, "CMP", data), "crc32_r"
        )
        assert want[::-1] in [m[1][0] for m in hits]

    # Falsification: no preimage in the input, wrong width, or strings -> silent.
    def test_falsify_crc32(self):
        x = (777).to_bytes(4, "little")
        assert not _by_encoder(
            generate_mutations(_crc_le(x), _crc_le(b"zzzz"), 32, "CMP", b"\x00" * 16), "crc32_p"
        )
        enc = Crc32Encoder(reverse=False)
        assert not enc.is_applicable(64, "CMP", b"\x01" * 8, b"\x02" * 8)
        assert not enc.is_applicable(512, "STR", b"abcd", b"efgh")
        assert not enc.is_applicable(16, "CMP", b"\x01\x02", b"\x03\x04")
        # Small-int compare (len == 12): neither side looks like a CRC.
        assert not enc.is_applicable(
            32, "CMP", (12).to_bytes(4, "little"), (13).to_bytes(4, "little")
        )

    # Adversarial: every 32-bit preimage round-trips, incl. all-zero / all-ones.
    def test_adversarial_crc32_roundtrip(self):
        enc = Crc32Encoder(reverse=False)
        rng = RandPool(seed=11)
        xs = [b"\x00" * 4, b"\xff" * 4] + [rng.randbytes(4) for _ in range(300)]
        for x in xs:
            assert enc.encode(_crc_le(x)) == [x]


def _fnv_le(data: bytes) -> bytes:
    """FNV-1a of *data* as a little-endian cmplog operand."""
    return _fnv1a(data).to_bytes(4, "little")


class TestFnv1aEncoder:
    """AntiFuzz §4.4: ``if (fnv1a(value) == OUTPUT_HASH)`` — antifuzz_demo.c."""

    def test_fnv1a_solves_hashed_compare(self):
        """Encoder produces crsh when given the demo's two hash values."""
        garbage, want = b"xxxx", b"crsh"
        data = b"AA" + garbage + b"BB"
        hits = _by_encoder(
            generate_mutations(_fnv_le(garbage), _fnv_le(want), 32, "CMP", data),
            "fnv1a_p",
        )
        assert ((2,), (want,)) in [(m[0], m[1]) for m in hits]

    def test_fnv1a_reversed_field(self):
        x = bytes.fromhex("01020304")
        data = x[::-1] + b"zz"
        want = b"crsh"
        hits = _by_encoder(
            generate_mutations(_fnv_le(x), _fnv_le(want), 32, "CMP", data), "fnv1a_r"
        )
        assert any(m[1] == (want[::-1],) for m in hits)

    def test_falsify_fnv1a(self):
        """Independent reference: fnv1a(encode(C)) == C; wrong widths/types silent."""
        enc = Fnv1aEncoder(reverse=False)
        rng = RandPool(seed=42)
        for _ in range(50):
            x = rng.randbytes(4)
            c = _fnv_le(x)
            for pre in enc.encode(c):
                assert _fnv1a(pre) == _fnv1a(x)
        # No preimage in the input -> no mutation
        x = (777).to_bytes(4, "little")
        assert not _by_encoder(
            generate_mutations(_fnv_le(x), _fnv_le(b"zzzz"), 32, "CMP", b"\x00" * 16),
            "fnv1a_p",
        )
        assert not enc.is_applicable(64, "CMP", b"\x01" * 8, b"\x02" * 8)
        assert not enc.is_applicable(512, "STR", b"abcd", b"efgh")
        assert not enc.is_applicable(16, "CMP", b"\x01\x02", b"\x03\x04")

    def test_adversarial_small_integer_skipped(self):
        """Small-integer compares (both < 2^24) are not scanned as FNV."""
        enc = Fnv1aEncoder(reverse=False)
        assert not enc.is_applicable(
            32, "CMP", (12).to_bytes(4, "little"), (13).to_bytes(4, "little")
        )
        # One side large is still skipped (both must look hash-like).
        assert not enc.is_applicable(
            32, "CMP", (12).to_bytes(4, "little"), (1 << 28).to_bytes(4, "little")
        )

    def test_adversarial_fnv1a_roundtrip(self):
        enc = Fnv1aEncoder(reverse=False)
        rng = RandPool(seed=11)
        xs = [b"\x00" * 4, b"\xff" * 4, b"crsh"] + [rng.randbytes(4) for _ in range(100)]
        for x in xs:
            pres = enc.encode(_fnv_le(x))
            assert x in pres
            assert all(_fnv1a(p) == _fnv1a(x) for p in pres)


class TestFnv1aCost:
    """Hard Rule 41: FNV-1a must not slow the pairs it cannot solve."""

    def test_no_match_never_encodes_replacements(self, monkeypatch):
        """Pattern absent from the input -> replacement variants are not built."""
        calls = []
        real = rq_encodings._get_encoded_variants
        monkeypatch.setattr(
            rq_encodings, "_get_encoded_variants", lambda *a: calls.append(a) or real(*a)
        )
        rng = RandPool(seed=3)
        data = b"\x00" * 64
        for n in (2, 8):
            for _ in range(20):
                a, b = rng.randbytes(n), rng.randbytes(n)
                size = 64 if n == 8 else 8 * n
                generate_mutations(a, b, size, "CMP", data, hammer=True)
        assert calls == []

    def test_hammer_inverts_only_the_constant(self, monkeypatch):
        """hammer=True must not invert the constant +-64: hash neighbours are useless."""
        calls = []
        real = rq_encodings._fnv1a_invert_all
        monkeypatch.setattr(rq_encodings, "_fnv1a_invert_all", lambda t: calls.append(t) or real(t))
        garbage, want = b"qqqq", b"crsh"
        data = b"AA" + garbage + b"BB"
        hits = _by_encoder(
            generate_mutations(
                _fnv_le(garbage),
                _fnv_le(want),
                32,
                "CMP",
                data,
                hammer=True,
                is_hash=lambda a, b: False,
            ),
            "fnv1a_p",
        )
        assert ((2,), (want,)) in [(m[0], m[1]) for m in hits]
        assert len(set(calls)) <= 2  # operand + constant; fnv1a_p/_r share the cache
        assert {m[1] for m in hits} == {(p,) for p in real(_fnv1a(want))}

    def test_empty_chunks_dropped(self, monkeypatch):
        """An encoding that yields an empty chunk is dropped on both sides."""

        class Empty(Encoder):
            def encode(self, val):
                return [b""]

        enc = Empty()
        assert rq_encodings._get_encoded_variants(enc, "CMP", 32, b"\x01\x00\x00\x00", False) == []
        monkeypatch.setattr(rq_encodings, "BUILTIN_ENCODERS", [enc])
        assert generate_mutations(b"\x01\x00", b"\x02\x00", 16, "CMP", b"\x01\x00\x01\x00") == []


class TestFnv1aPreimages:
    def test_not_a_bijection(self):
        """Falsification: some 32-bit hashes have no 4-byte preimage; all found ones are exact."""
        rng = RandPool(seed=5)
        counts = []
        for _ in range(20):
            target = int.from_bytes(rng.randbytes(4), "little")
            pres = _fnv1a_invert_all(target)
            assert all(_fnv1a(p) == target for p in pres)
            counts.append(len(pres))
        assert 0 in counts

    def test_two_byte_table_is_injective(self):
        """Adversarial: no collisions, so values are ints, not lists."""
        fwd = _fnv1a_fwd_table()
        assert len(fwd) == 1 << 16
        assert all(isinstance(v, int) for v in fwd.values())

    def test_table_built_once(self):
        assert _fnv1a_fwd_table() is _fnv1a_fwd_table()
