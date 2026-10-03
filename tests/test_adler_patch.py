"""Tests for the format-aware Adler-32 patcher (core/mutations/recompress).

Streams and PNGs are built and checked by test-local code (zlib.adler32,
binascii.crc32, zlib.decompress), not by the module under test, so a
trailer or chunk CRC the patcher forgets to repair fails here.
"""

import binascii
import struct
import zlib

import pytest

from fuzzer_tool.core.analyzers.analyzer_checksum_learner import ChecksumLearner
from fuzzer_tool.core.int_checksum import (
    ADLER32,
    FLETCHER16,
    SUM32,
    clear_active_int_model,
)
from fuzzer_tool.core.mutations.recompress import patch_adler
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_ADLER_LEN = 4
_ZLIB_HDR_LEN = 2
_ROWS = 6
_ROW = b"\x00" + bytes(range(1, 9))
_PLAIN = _ROW * _ROWS


@pytest.fixture(autouse=True)
def _reset_active_model():
    clear_active_int_model()
    yield
    clear_active_int_model()


# ── reference builders ─────────────────────────────────────────────────


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _ihdr() -> bytes:
    return _chunk(b"IHDR", struct.pack(">IIBBBBB", 8, _ROWS, 8, 0, 0, 0, 0))


def _png(idats: list[bytes], extra: bytes = b"") -> bytes:
    body = b"".join(_chunk(b"IDAT", d) for d in idats)
    return _PNG_SIG + _ihdr() + extra + body + _chunk(b"IEND", b"")


def _stale(stream: bytes) -> bytes:
    """Same stream with its Adler-32 trailer replaced by a wrong value."""
    return stream[:-_ADLER_LEN] + b"\xde\xad\xbe\xef"


def _walk(png: bytes):
    """Yield (type, data, crc_ok) for every complete chunk."""
    pos = len(_PNG_SIG)
    while pos + 12 <= len(png):
        (length,) = struct.unpack_from(">I", png, pos)
        kind = png[pos + 4 : pos + 8]
        data = png[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack_from(">I", png, pos + 8 + length)
        yield kind, data, crc == binascii.crc32(kind + data) & 0xFFFFFFFF
        pos += 12 + length


def _idat_stream(png: bytes) -> bytes:
    return b"".join(d for k, d, _ in _walk(png) if k == b"IDAT")


# ── bare zlib ──────────────────────────────────────────────────────────


class TestBareZlib:
    def test_repairs_stale_trailer(self):
        good = zlib.compress(_PLAIN)

        out = patch_adler(_stale(good), ADLER32)

        assert out == good

    def test_trailer_covers_plaintext_not_compressed_bytes(self):
        stream = _stale(zlib.compress(_PLAIN))

        out = patch_adler(stream, ADLER32)

        assert out[-_ADLER_LEN:] == struct.pack(">I", zlib.adler32(_PLAIN))
        assert out[-_ADLER_LEN:] != struct.pack(">I", zlib.adler32(stream[:-_ADLER_LEN]))

    def test_plaintext_edit_in_stored_block_is_repaired(self):
        # Level 0 = stored blocks: plaintext sits verbatim in the stream, so a
        # byte-level mutation keeps DEFLATE valid and only the trailer is stale.
        good = zlib.compress(_PLAIN, 0)
        at = good.index(_PLAIN)
        mutated = bytearray(good)
        mutated[at + 3] ^= 0xFF

        out = patch_adler(bytes(mutated), ADLER32)

        plain = bytearray(_PLAIN)
        plain[3] ^= 0xFF
        assert zlib.decompress(out) == bytes(plain)

    def test_idempotent_on_valid_stream(self):
        good = zlib.compress(_PLAIN)

        assert patch_adler(good, ADLER32) == good

    def test_trailing_bytes_after_stream_are_kept(self):
        good = zlib.compress(_PLAIN)

        out = patch_adler(_stale(good) + b"TAIL", ADLER32)

        assert out == good + b"TAIL"

    def test_fdict_header_is_skipped(self):
        co = zlib.compressobj(zdict=b"dictionary")
        good = co.compress(_PLAIN) + co.flush()

        out = patch_adler(_stale(good), ADLER32)

        assert out == good


# ── PNG IDAT ───────────────────────────────────────────────────────────


class TestPngIdat:
    def test_repairs_adler_and_idat_crc(self):
        good = _png([zlib.compress(_PLAIN)])
        bad = _png([_stale(zlib.compress(_PLAIN))])
        assert bad != good

        out = patch_adler(bad, ADLER32)

        assert out == good
        assert all(ok for _, _, ok in _walk(out))
        assert zlib.decompress(_idat_stream(out)) == _PLAIN

    def test_other_chunks_stay_byte_identical(self):
        # A deliberately wrong CRC on a non-IDAT chunk must survive: the
        # patcher edits only what it repairs.
        extra = _chunk(b"tEXt", b"k\x00v")[:-4] + b"\x00\x00\x00\x00"
        bad = _png([_stale(zlib.compress(_PLAIN))], extra=extra)

        out = patch_adler(bad, ADLER32)

        assert extra in out
        assert out.startswith(_PNG_SIG + _ihdr() + extra)
        assert out.endswith(_chunk(b"IEND", b""))

    @pytest.mark.parametrize("cut", [1, 2, 3])
    def test_trailer_split_across_idat_chunks(self, cut):
        stream = _stale(zlib.compress(_PLAIN))
        parts = [stream[:-cut], stream[-cut:]]
        good_parts = [zlib.compress(_PLAIN)[:-cut], zlib.compress(_PLAIN)[-cut:]]

        out = patch_adler(_png(parts), ADLER32)

        assert out == _png(good_parts)
        assert all(ok for _, _, ok in _walk(out))

    def test_iend_crc_not_overwritten(self):
        # Regression: the generic trailing-field patch wrote the Adler of
        # buf[:-4] over the IEND CRC.
        bad = _png([_stale(zlib.compress(_PLAIN))])

        out = patch_adler(bad, ADLER32)

        assert out[-4:] == bad[-4:]

    def test_png_without_idat_is_returned_unchanged(self):
        bare = _PNG_SIG + _ihdr() + _chunk(b"IEND", b"")

        assert patch_adler(bare, ADLER32) == bare


# ── falsification ──────────────────────────────────────────────────────


class TestFalsification:
    @pytest.mark.parametrize("model", [FLETCHER16, SUM32])
    def test_non_adler_model_is_not_applied(self, model):
        # A recovered Fletcher-16 says nothing about a zlib trailer; writing
        # it (or Adler) there would be the family substitution the learner
        # forbids.
        assert patch_adler(_stale(zlib.compress(_PLAIN)), model) is None
        assert patch_adler(_png([_stale(zlib.compress(_PLAIN))]), model) is None

    def test_patch_differs_from_generic_trailing_write(self):
        stream = _stale(zlib.compress(_PLAIN))
        generic = stream[:-4] + struct.pack(">I", zlib.adler32(stream[:-4]))

        assert patch_adler(stream, ADLER32) != generic

    def test_non_container_is_not_claimed(self):
        assert patch_adler(b"ABCDEFGH" * 8, ADLER32) is None
        assert patch_adler(b"", ADLER32) is None


# ── adversarial ────────────────────────────────────────────────────────


class TestAdversarial:
    def test_truncated_stream_is_not_claimed(self):
        stream = zlib.compress(_PLAIN)

        assert patch_adler(stream[: len(stream) // 2], ADLER32) is None

    def test_missing_trailer_is_not_claimed(self):
        stream = zlib.compress(_PLAIN)

        assert patch_adler(stream[:-_ADLER_LEN], ADLER32) is None
        assert patch_adler(stream[:-1], ADLER32) is None

    def test_valid_header_garbage_body(self):
        assert patch_adler(b"\x78\x9c" + b"\xff" * 64, ADLER32) is None

    def test_inflate_bomb_is_bounded_and_not_patched(self):
        bomb = zlib.compress(b"\x00" * (16 << 20), 9)

        assert patch_adler(_stale(bomb), ADLER32) is None

    def test_png_truncated_idat_is_returned_unchanged(self):
        bad = _png([_stale(zlib.compress(_PLAIN))])
        cut = bad[: bad.index(b"IDAT") + 10]

        assert patch_adler(cut, ADLER32) == cut

    def test_png_corrupt_idat_body_is_returned_unchanged(self):
        bad = _png([b"\xff" * 40])

        assert patch_adler(bad, ADLER32) == bad

    def test_png_length_field_beyond_file(self):
        bad = bytearray(_png([_stale(zlib.compress(_PLAIN))]))
        at = bad.index(b"IDAT") - 4
        bad[at : at + 4] = b"\xff\xff\xff\xff"

        out = patch_adler(bytes(bad), ADLER32)

        assert out in (None, bytes(bad))

    def test_never_changes_length(self):
        for stream in (zlib.compress(_PLAIN), _stale(zlib.compress(_PLAIN))):
            out = patch_adler(stream, ADLER32)
            assert out is None or len(out) == len(stream)


# ── wiring: _op_crc_learn ──────────────────────────────────────────────


class TestCrcLearnWiring:
    def _engine(self, model=ADLER32):
        f = make_minimal_fuzzer(0x5EED)
        learner = ChecksumLearner(f)
        learner._set_int_model(model)
        assert learner.ensure_poly() is None
        f.checksum_learner = learner
        return OperatorEngine(f)

    def test_png_gets_adler_repaired(self):
        engine = self._engine()
        good = _png([zlib.compress(_PLAIN)])
        buf = bytearray(_png([_stale(zlib.compress(_PLAIN))]))

        engine._op_crc_learn(buf, 0, bytes(buf))

        assert bytes(buf) == good

    def test_bare_zlib_gets_adler_repaired(self):
        engine = self._engine()
        buf = bytearray(_stale(zlib.compress(_PLAIN)))

        engine._op_crc_learn(buf, 0, bytes(buf))

        assert bytes(buf) == zlib.compress(_PLAIN)

    def test_non_container_keeps_generic_trailing_patch(self):
        engine = self._engine()
        buf = bytearray(b"HDR:payload-bytes....")

        engine._op_crc_learn(buf, 0, bytes(buf))

        expected = struct.pack(">I", zlib.adler32(b"HDR:payload-bytes...."[:-4]))
        # Adler's two halves are independent of word order, so compare via
        # the learner's own model rather than assuming layout here.
        assert bytes(buf[:-4]) == b"HDR:payload-bytes...."[:-4]
        assert bytes(buf[-4:]) == expected

    def test_non_adler_model_keeps_generic_trailing_patch(self):
        engine = self._engine(FLETCHER16)
        stream = _stale(zlib.compress(_PLAIN))
        buf = bytearray(stream)

        engine._op_crc_learn(buf, 0, stream)

        assert bytes(buf[:-2]) == stream[:-2]
