"""Tests for the structure-aware LZ4 frame mutator (core/mutations/lz4.py).

Input layout mirrors targets/lz4_read.c: byte 0 is the mode selector
(even -> frame decode), the LZ4 frame starts at offset 1.

Every branch is driven by ScriptedRng (Hard Rule 39). Expected checksums
are derived independently: the C ``xxhash`` module when installed, and the
system liblz4 (ctypes) as a decoder oracle when present.
"""

import ctypes
import ctypes.util
import struct

import pytest

from fuzzer_tool.core.mutations import lz4 as mod
from fuzzer_tool.core.mutations.lz4 import (
    BLOCK_SIZE_MASK,
    BLOCK_UNCOMPRESSED,
    FLG_B_CHECKSUM,
    FLG_B_INDEP,
    FLG_BITS,
    FLG_C_CHECKSUM,
    FLG_C_SIZE,
    FLG_VERSION,
    LZ4_MAGIC,
    SKIPPABLE_MAGIC_BASE,
    BlockEdit,
    ChecksumTarget,
    Lz4Mutator,
    Lz4Op,
    SkipPlace,
    parse_lz4,
    serialize_lz4,
    xxh32,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

xxhash = pytest.importorskip("xxhash")

_MAGIC_BYTES = struct.pack("<I", LZ4_MAGIC)
_MODE = 0x02
_HDR_START = 1 + len(_MAGIC_BYTES)  # FLG offset
_STORED = 1
_COMPRESSED = 0


# ── independent oracles ────────────────────────────────────────────────────


def _ref_xxh32(data: bytes) -> int:
    return xxhash.xxh32_intdigest(data)


def _ref_hc(desc: bytes) -> int:
    return (_ref_xxh32(desc) >> 8) & 0xFF


def _liblz4():
    name = ctypes.util.find_library("lz4")
    if not name:
        return None
    lib = ctypes.CDLL(name)
    lib.LZ4F_createDecompressionContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.LZ4F_createDecompressionContext.restype = ctypes.c_size_t
    lib.LZ4F_freeDecompressionContext.argtypes = [ctypes.c_void_p]
    lib.LZ4F_decompress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib.LZ4F_decompress.restype = ctypes.c_size_t
    lib.LZ4F_isError.argtypes = [ctypes.c_size_t]
    lib.LZ4F_isError.restype = ctypes.c_uint
    return lib


_LIB = _liblz4()
_LZ4F_VERSION = 100
_OUT_CAP = 1 << 20


def _lz4f_decode(frame: bytes) -> bytes | None:
    """Decode one frame with system liblz4; None on any error/incomplete."""
    dctx = ctypes.c_void_p()
    assert not _LIB.LZ4F_isError(
        _LIB.LZ4F_createDecompressionContext(ctypes.byref(dctx), _LZ4F_VERSION)
    )
    out = ctypes.create_string_buffer(_OUT_CAP)
    src = ctypes.create_string_buffer(frame, len(frame))
    dst_size = ctypes.c_size_t(_OUT_CAP)
    src_size = ctypes.c_size_t(len(frame))
    ret = _LIB.LZ4F_decompress(dctx, out, ctypes.byref(dst_size), src, ctypes.byref(src_size), None)
    _LIB.LZ4F_freeDecompressionContext(dctx)
    if _LIB.LZ4F_isError(ret) or ret != 0:
        return None
    return out.raw[: dst_size.value]


needs_lib = pytest.mark.skipif(_LIB is None, reason="system liblz4 not found")


# ── fixtures ───────────────────────────────────────────────────────────────


def _gen(flg_opts: int, bd: int, blocks: list[tuple[bytes, int]]) -> bytes:
    """Drive the generator: [(payload, stored_flag), ...]."""
    # Draw order: opts, bd, block count, then per block: length, stored flag.
    randints = [flg_opts, bd, len(blocks)]
    blobs = []
    for payload, stored in blocks:
        randints += [len(payload), stored]
        blobs.append(payload)
    return Lz4Mutator()._generate_random_lz4(
        max_len=1 << 20, rng=ScriptedRng(randints=randints, randbytes=blobs)
    )


_P1 = bytes(range(20))  # >= 15 -> extended literal length when compressed
_P2 = b"hello lz4 frame"
_CONTENT = _P1 + _P2
_ALL_OPTS = FLG_B_INDEP | FLG_B_CHECKSUM | FLG_C_SIZE | FLG_C_CHECKSUM


def _sample(opts: int = _ALL_OPTS) -> bytes:
    return b"\x00" + _gen(opts, 4, [(_P1, _COMPRESSED), (_P2, _STORED)])[1:]


def _mut(data: bytes, **script) -> bytes:
    return Lz4Mutator().mutate(data, max_len=1 << 20, rng=ScriptedRng(**script))


def _desc_len(flg: int) -> int:
    return 2 + (8 if flg & FLG_C_SIZE else 0) + (4 if flg & mod.FLG_DICT_ID else 0)


def _assert_hc(out: bytes) -> None:
    flg = out[_HDR_START]
    n = _desc_len(flg)
    desc = out[_HDR_START : _HDR_START + n]
    assert out[_HDR_START + n] == _ref_hc(desc)


# ── xxh32 ──────────────────────────────────────────────────────────────────


_VECTORS = [
    (b"", 0x02CC5D05),
    (b"abc", 0x32D153FF),
    (b"Nobody inspects the spammish repetition", 0xE2293B2F),
]


@pytest.mark.parametrize("fn", [xxh32, mod._xxh32_py])
@pytest.mark.parametrize(("data", "want"), _VECTORS)
def test_xxh32_known_vectors(fn, data, want):
    assert fn(data) == want


def test_xxh32_py_matches_c_all_tail_lengths():
    """Every stripe/tail combination 0..80 bytes, plus a nonzero seed."""
    blob = bytes((i * 131 + 7) & 0xFF for i in range(80))
    for n in range(len(blob) + 1):
        assert mod._xxh32_py(blob[:n]) == _ref_xxh32(blob[:n])
    assert mod._xxh32_py(blob, 0x9747B28C) == xxhash.xxh32_intdigest(blob, seed=0x9747B28C)


# ── parse / generate ───────────────────────────────────────────────────────


def test_parse_accepts_generated():
    frame = parse_lz4(_sample())
    assert frame is not None
    assert [b.data for b in frame.blocks][1] == _P2
    assert frame.end_mark


def test_serialize_round_trips_generated():
    data = _sample()
    assert serialize_lz4(parse_lz4(data)) == data


@pytest.mark.parametrize(
    "bad",
    [
        b"\x00" + b"\x05\x22\x4d\x18" + b"\x60\x40\x82" + b"\x00" * 4,  # wrong magic
        b"\x01" + _MAGIC_BYTES + b"\x60\x40\x82" + b"\x00" * 4,  # odd mode -> raw block
        b"\x00" + _MAGIC_BYTES + b"\x60",  # short
        b"",
    ],
)
def test_parse_rejects(bad):
    assert parse_lz4(bad) is None


def test_generated_hc_correct():
    _assert_hc(_sample())
    _assert_hc(_sample(0))


def test_generated_mode_byte_even_and_magic():
    data = _gen(0, 7, [(_P2, _STORED)])
    assert data[0] == 0
    assert data[1:5] == _MAGIC_BYTES


@needs_lib
@pytest.mark.parametrize("opts", [0, _ALL_OPTS, FLG_B_CHECKSUM, FLG_C_CHECKSUM | FLG_C_SIZE])
def test_generated_decodes_with_liblz4(opts):
    assert _lz4f_decode(_sample(opts)[1:]) == _CONTENT


def test_generator_respects_max_len():
    rng = ScriptedRng(randints=[_ALL_OPTS, 4, 1, len(_P1), _STORED], randbytes=[_P1])
    assert len(Lz4Mutator()._generate_random_lz4(max_len=9, rng=rng)) == 9


def test_mutate_nonmatching_input_generates():
    rng = ScriptedRng(randints=[0, 4, 1, len(_P2), _STORED], randbytes=[_P2])
    out = Lz4Mutator().mutate(b"garbage", max_len=4096, rng=rng)
    assert parse_lz4(out) is not None


# ── branch: FLG bit flip + HC repair ──────────────────────────────────────


def test_flg_flip_block_checksum_on_repairs_everything():
    data = _sample(FLG_B_INDEP)
    out = _mut(data, randints=[Lz4Op.FLG], choice_idxs=[FLG_BITS.index(FLG_B_CHECKSUM)])
    assert out[_HDR_START] == FLG_VERSION | FLG_B_INDEP | FLG_B_CHECKSUM
    _assert_hc(out)
    frame = parse_lz4(out)
    assert [b.checksum for b in frame.blocks] == [_ref_xxh32(b.data) for b in frame.blocks]
    if _LIB is not None:
        assert _lz4f_decode(out[1:]) == _CONTENT


def test_flg_flip_version_bit_keeps_hc_valid():
    data = _sample()
    out = _mut(data, randints=[Lz4Op.FLG], choice_idxs=[FLG_BITS.index(FLG_VERSION)])
    assert out[_HDR_START] == data[_HDR_START] ^ FLG_VERSION
    _assert_hc(out)


# ── branch: BD block-max ──────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0, 3, 7])
def test_bd_block_max_set_and_hc_repaired(value):
    out = _mut(_sample(), randints=[Lz4Op.BD, value])
    assert (out[_HDR_START + 1] >> mod.BD_SHIFT) & mod.BD_VALUE_MASK == value
    _assert_hc(out)


# ── branch: content size ──────────────────────────────────────────────────

_U64 = (1 << 64) - 1


@pytest.mark.parametrize(
    ("idx", "want"),
    [
        (0, 0),
        (1, len(_CONTENT)),
        (2, len(_CONTENT) - 1),
        (3, len(_CONTENT) + 1),
        (4, 1 << 32),
        (5, _U64),
    ],
)
def test_content_size_boundaries(idx, want):
    out = _mut(_sample(FLG_B_INDEP), randints=[Lz4Op.CSIZE], choice_idxs=[idx])
    assert out[_HDR_START] & FLG_C_SIZE
    assert struct.unpack_from("<Q", out, _HDR_START + 2)[0] == want
    _assert_hc(out)


@needs_lib
def test_content_size_true_value_still_decodes():
    out = _mut(_sample(FLG_B_INDEP), randints=[Lz4Op.CSIZE], choice_idxs=[1])
    assert _lz4f_decode(out[1:]) == _CONTENT


# ── branch: block size field ──────────────────────────────────────────────


def _block_words(out: bytes) -> list[int]:
    return [b.word for b in parse_lz4(out).blocks]


@pytest.mark.parametrize(
    ("idx", "size_fn"),
    [
        (0, lambda n, bmax: 0),
        (1, lambda n, bmax: n - 1),
        (2, lambda n, bmax: n + 1),
        (3, lambda n, bmax: bmax),
        (4, lambda n, bmax: bmax + 1),
        (5, lambda n, bmax: BLOCK_SIZE_MASK),
    ],
)
def test_block_size_boundaries(idx, size_fn):
    data = _sample(0)
    out = _mut(data, randints=[Lz4Op.BSIZE, 1], choice_idxs=[idx])
    bmax = 1 << 16  # BD=4 -> 64 KiB
    word = struct.unpack_from("<I", out, _HDR_START + 3 + 4 + len(parse_lz4(data).blocks[0].data))[
        0
    ]
    assert word == BLOCK_UNCOMPRESSED | (size_fn(len(_P2), bmax) & BLOCK_SIZE_MASK)


def test_block_size_toggle_uncompressed_bit():
    data = _sample(0)
    before = _block_words(data)[0]
    out = _mut(data, randints=[Lz4Op.BSIZE, 0], choice_idxs=[6])
    assert struct.unpack_from("<I", out, _HDR_START + 3)[0] == before ^ BLOCK_UNCOMPRESSED


# ── branch: checksum repair / corrupt ─────────────────────────────────────

_REPAIR = 0.0
_CORRUPT = 0.99


def test_block_checksum_repair():
    out = _mut(
        _sample(FLG_B_INDEP), randints=[Lz4Op.CHECKSUM, ChecksumTarget.BLOCK, 0], randoms=[_REPAIR]
    )
    frame = parse_lz4(out)
    assert out[_HDR_START] & FLG_B_CHECKSUM
    assert [b.checksum for b in frame.blocks] == [_ref_xxh32(b.data) for b in frame.blocks]
    _assert_hc(out)


def test_block_checksum_repair_fixes_bad_input():
    data = bytearray(_sample())
    frame = parse_lz4(bytes(data))
    cs_off = _HDR_START + 2 + 8 + 1 + 4 + len(frame.blocks[0].data)
    data[cs_off] ^= 0xFF
    out = _mut(bytes(data), randints=[Lz4Op.CHECKSUM, ChecksumTarget.BLOCK, 0], randoms=[_REPAIR])
    assert out == _sample()


def test_block_checksum_corrupt():
    bit = 5
    out = _mut(
        _sample(), randints=[Lz4Op.CHECKSUM, ChecksumTarget.BLOCK, 1, bit], randoms=[_CORRUPT]
    )
    blk = parse_lz4(out).blocks[1]
    assert blk.checksum == _ref_xxh32(_P2) ^ (1 << bit)


def test_content_checksum_corrupt():
    bit = 3
    out = _mut(
        _sample(), randints=[Lz4Op.CHECKSUM, ChecksumTarget.CONTENT, bit], randoms=[_CORRUPT]
    )
    assert struct.unpack_from("<I", out, len(out) - 4)[0] == _ref_xxh32(_CONTENT) ^ (1 << bit)


def test_content_checksum_repair_enables_flag():
    out = _mut(
        _sample(FLG_B_INDEP), randints=[Lz4Op.CHECKSUM, ChecksumTarget.CONTENT], randoms=[_REPAIR]
    )
    assert out[_HDR_START] & FLG_C_CHECKSUM
    assert struct.unpack_from("<I", out, len(out) - 4)[0] == _ref_xxh32(_CONTENT)
    _assert_hc(out)


def test_header_checksum_corrupt():
    bit = 2
    data = _sample()
    out = _mut(data, randints=[Lz4Op.CHECKSUM, ChecksumTarget.HEADER, bit], randoms=[_CORRUPT])
    hc_off = _HDR_START + _desc_len(data[_HDR_START])
    assert out[hc_off] == data[hc_off] ^ (1 << bit)


# ── branch: block list edits ──────────────────────────────────────────────


@needs_lib
def test_insert_block_repairs_content_checksum_and_size():
    new = b"XYZ"
    out = _mut(_sample(), randints=[Lz4Op.BLOCKS, BlockEdit.INSERT, len(new), 1], randbytes=[new])
    want = _P1 + new + _P2
    assert _lz4f_decode(out[1:]) == want
    assert struct.unpack_from("<Q", out, _HDR_START + 2)[0] == len(want)


@needs_lib
def test_duplicate_block():
    out = _mut(_sample(), randints=[Lz4Op.BLOCKS, BlockEdit.DUP, 1])
    assert _lz4f_decode(out[1:]) == _CONTENT + _P2


@needs_lib
def test_drop_block():
    out = _mut(_sample(), randints=[Lz4Op.BLOCKS, BlockEdit.DROP, 0])
    assert _lz4f_decode(out[1:]) == _P2


def test_drop_end_mark():
    data = _sample(FLG_B_INDEP)
    out = _mut(data, randints=[Lz4Op.BLOCKS, BlockEdit.DROP_END])
    assert out == data[: -len(struct.pack("<I", 0))]
    assert not parse_lz4(out).end_mark


# ── branch: skippable frame ───────────────────────────────────────────────


def test_skippable_prepend_truncated_size():
    payload = b"\xaa" * 6
    nibble = 0xF
    data = _sample()
    out = _mut(
        data,
        randints=[Lz4Op.SKIPPABLE, nibble, len(payload), SkipPlace.PREPEND],
        randbytes=[payload],
        choice_idxs=[1],  # declared = len + 1 -> truncated
    )
    skip = struct.pack("<II", SKIPPABLE_MAGIC_BASE + nibble, len(payload) + 1) + payload
    assert out == data[:1] + skip + data[1:]


def test_skippable_append_keeps_frame_parseable():
    payload = b"\x01\x02"
    data = _sample()
    out = _mut(
        data,
        randints=[Lz4Op.SKIPPABLE, 0, len(payload), SkipPlace.APPEND],
        randbytes=[payload],
        choice_idxs=[0],
    )
    assert out == data + struct.pack("<II", SKIPPABLE_MAGIC_BASE, len(payload)) + payload
    assert parse_lz4(out) is not None


# ── falsification / adversarial ───────────────────────────────────────────


@needs_lib
def test_falsification_repair_decodes_corrupt_rejects():
    """Repair must be accepted by real liblz4 and corrupt rejected; a mutator
    that computed checksums wrong, or ignored the corrupt path, fails one side.
    Also: mode byte parity preserved and output differs from input."""
    data = bytes([_MODE]) + _sample(FLG_B_INDEP)[1:]
    ok = _mut(data, randints=[Lz4Op.CHECKSUM, ChecksumTarget.CONTENT], randoms=[_REPAIR])
    bad = _mut(data, randints=[Lz4Op.CHECKSUM, ChecksumTarget.CONTENT, 0], randoms=[_CORRUPT])
    assert ok != data and bad != data
    assert ok[0] == bad[0] == _MODE
    assert _lz4f_decode(ok[1:]) == _CONTENT
    assert _lz4f_decode(bad[1:]) is None


def _hostile_inputs() -> list[bytes]:
    pool = RandPool(seed=7)
    head = b"\x00" + _MAGIC_BYTES
    good = _sample()
    inputs = [good[:n] for n in range(len(good))]
    inputs.append(head + bytes([FLG_VERSION, 0x40]) + b"\x00" + struct.pack("<I", BLOCK_SIZE_MASK))
    inputs.append(head + bytes([0xFF, 0xFF]) + b"\xff" * 16)
    inputs.append(
        head + bytes([FLG_VERSION, 0x40, 0]) + struct.pack("<I", BLOCK_UNCOMPRESSED) * 20000
    )
    # compressed block of max-length match tokens: decoder must stay bounded
    bomb = b"\x1f" + b"A" + b"\x01\x00" + b"\xff" * 4000
    inputs.append(
        head
        + bytes([FLG_VERSION | FLG_C_CHECKSUM | FLG_C_SIZE, 0x40])
        + bytes(9)
        + struct.pack("<I", len(bomb))
        + bomb
        + bytes(4)
    )
    inputs += [head + pool.randbytes(pool.randint(0, 200)) for _ in range(200)]
    return inputs


@pytest.mark.parametrize("max_len", [0, 16, 4096])
def test_adversarial_hostile_inputs_never_raise(max_len):
    pool = RandPool(seed=11)
    mut = Lz4Mutator(seed=3)
    for data in _hostile_inputs():
        out = mut.mutate(data, max_len=max_len, rng=pool)
        assert len(out) <= max_len
        if max_len and parse_lz4(data) is not None:
            assert out[0] == data[0]
