"""Format-aware integer-checksum patch: the zlib Adler-32 slot, not the last bytes.

``crc_learn`` under an integer model used to write the checksum over the
whole buffer into its last ``nbytes``. On a PNG that is the IEND CRC — a
CRC-32 field by spec — and the stale Adler-32 inside IDAT stayed stale.
``patch_adler`` locates the trailer by raw-inflating the stream and writes
``adler32(plaintext)`` there, re-CRCing only the IDATs it touched.
Complements ``tests/test_adler_patch.py`` with an independent chunk parser.
"""

from __future__ import annotations

import struct
import zlib
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.analyzers.analyzer_checksum_learner import ChecksumLearner
from fuzzer_tool.core.int_checksum import (
    ADLER32,
    FLETCHER16,
    clear_active_int_model,
    eval_model,
)
from fuzzer_tool.core.mutations.recompress import patch_adler
from fuzzer_tool.services.operators import OperatorEngine

PLAIN = b"the quick brown fox jumps over the lazy dog " * 40
PNG_SIG = b"\x89PNG\r\n\x1a\n"
ADLER_BYTES = 4
STALE = b"\x00\x00\x00\x00"


@pytest.fixture(autouse=True)
def _reset_active_model():
    clear_active_int_model()
    yield
    clear_active_int_model()


def _stale(stream: bytes) -> bytes:
    """*stream* with its Adler-32 zeroed: inflates raw, fails zlib's check."""
    return stream[:-ADLER_BYTES] + STALE


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _png(stream: bytes, cut: int) -> bytes:
    """PNG carrying *stream* split into two IDATs at *cut*."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        PNG_SIG
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", stream[:cut])
        + _chunk(b"IDAT", stream[cut:])
        + _chunk(b"IEND", b"")
    )


def _chunks(png: bytes) -> list[tuple[bytes, bytes, int]]:
    """(type, data, stored crc) per chunk, parsed independently of the code under test."""
    out, pos = [], len(PNG_SIG)
    while pos < len(png):
        (length,) = struct.unpack_from(">I", png, pos)
        ctype = png[pos + 4 : pos + 8]
        data = png[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack_from(">I", png, pos + 8 + length)
        out.append((ctype, data, crc))
        pos += 12 + length
    return out


def _idat_stream(png: bytes) -> bytes:
    return b"".join(d for t, d, _ in _chunks(png) if t == b"IDAT")


# ── PNG IDAT ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("back", [1, 2, 3, ADLER_BYTES, 9])
def test_regression_idat_trailer_across_chunks(back):
    """Trailer straddling the IDAT boundary at every offset is still patched."""
    good = zlib.compress(PLAIN)
    src = _png(_stale(good), len(good) - back)

    out = patch_adler(src, ADLER32)

    assert zlib.decompress(_idat_stream(out)) == PLAIN
    for ctype, data, crc in _chunks(out):
        assert crc == zlib.crc32(ctype + data) & 0xFFFFFFFF, ctype
    assert [len(d) for _, d, _ in _chunks(out)] == [len(d) for _, d, _ in _chunks(src)]


# ── adversarial ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not zlib at all",
        zlib.compress(PLAIN)[:-2],  # trailer cut short
        zlib.compress(PLAIN)[: len(zlib.compress(PLAIN)) // 2],  # deflate unterminated
        b"\x78\xbb" + zlib.compress(PLAIN)[2:],  # FDICT set, dict id missing
    ],
)
def test_regression_unpatchable_stream_declines(data):
    assert patch_adler(data, ADLER32) is None


@pytest.mark.parametrize(
    "png",
    [
        PNG_SIG + _chunk(b"IEND", b""),  # no IDAT
        PNG_SIG + b"\x00\x00\x10\x00IDAT",  # truncated chunk
    ],
)
def test_regression_unpatchable_png_unchanged(png):
    assert patch_adler(png, ADLER32) == png


# ── crc_learn wiring ───────────────────────────────────────────────────


class _FakeFuzzer:
    def __init__(self):
        self._cmplog = None
        self._op_declines = {}


def _crc_learn(model, data: bytes) -> bytes:
    """Run ``OperatorEngine._op_crc_learn`` on a minimal engine with *model* installed."""
    f = _FakeFuzzer()
    learner = ChecksumLearner(f)
    learner._set_int_model(model)
    eng = SimpleNamespace(f=f, ctx=SimpleNamespace(checksum_learner=learner, _rng=None))
    eng._try_format_int_patch = lambda buf, m: OperatorEngine._try_format_int_patch(eng, buf, m)

    buf = bytearray(data)
    OperatorEngine._op_crc_learn(eng, buf, 0, data)
    return bytes(buf)


def test_regression_crc_learn_keeps_iend_crc():
    good = zlib.compress(PLAIN)
    out = _crc_learn(ADLER32, _png(_stale(good), len(good) - 2))

    assert zlib.decompress(_idat_stream(out)) == PLAIN
    assert _chunks(out)[-1][2] == zlib.crc32(b"IEND") & 0xFFFFFFFF


def test_regression_crc_learn_narrow_model_skips_zlib_slot():
    """A 2-byte model cannot be an Adler-32 slot: generic patch, trailer width 2."""
    good = zlib.compress(PLAIN)
    nbytes = FLETCHER16.nbytes

    out = _crc_learn(FLETCHER16, good)

    want = eval_model(FLETCHER16, good[:-nbytes]).to_bytes(nbytes, "big")
    assert out == good[:-nbytes] + want
