"""Tests for the ZFS uberblock / nvlist mutator (`zfs_label_mutate`)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.zfs import (
    NV_SIZE_EDGES,
    U64_EDGES,
    UB_VERSIONS,
    Op,
    find_slots,
    mutate_zfs,
    nv_pairs,
    sniff_zfs,
)
from fuzzer_tool.core.rand_pool import RandPool

from .support import zfs_build as zb
from .support.scripted_rng import ScriptedRng

FRAG = zb.fragment()
UB_OFF = zb.SLOT
FIRST_PAIR = 12  # 4-byte stream header + nvl_version + nvl_nvflag
LABEL = 256 * 1024
PHYS_OFF = 16 * 1024
PHYS_END = 128 * 1024


def _u64(data: bytes, off: int, fmt: str = "<") -> int:
    return struct.unpack_from(fmt + "Q", data, off)[0]


def _i32(data: bytes, off: int) -> int:
    return struct.unpack_from(">i", data, off)[0]


def _slot(data: bytes, off: int = UB_OFF, size: int = zb.SLOT) -> bytes:
    return data[off : off + size]


def _rng(op: Op, randints=(), choices=()) -> ScriptedRng:
    """Op drawn by choice() over list(Op) (every op is live on FRAG)."""
    return ScriptedRng(randints=list(randints), choice_idxs=[int(op), *choices])


class TestOracle:
    """Hard Rule 46: the checksum oracle must pass its own control first."""

    def test_control_fixture_verifies(self):
        assert zb.cksum_ok(_slot(FRAG), UB_OFF, "<")
        assert zb.cksum_ok(zb.uberblock(">", offset=0), 0, ">")

    def test_control_rejects_planted_flip(self):
        slot = bytearray(_slot(FRAG))
        slot[100] ^= 1
        assert not zb.cksum_ok(bytes(slot), UB_OFF, "<")

    def test_control_rejects_wrong_verifier_offset(self):
        assert not zb.cksum_ok(_slot(FRAG), 0, "<")


class TestSniff:
    def test_fragment(self):
        assert sniff_zfs(FRAG)

    def test_lone_uberblock_both_orders(self):
        assert sniff_zfs(zb.uberblock("<"))
        assert sniff_zfs(zb.uberblock(">"))

    def test_lone_nvlist(self):
        assert sniff_zfs(zb.label_nvlist())

    @pytest.mark.parametrize("data", [b"", bytes(4096), b"\x0c\xb1\xba" * 500])
    def test_rejects(self, data):
        assert not sniff_zfs(data)

    def test_rejects_unaligned_magic(self):
        assert not sniff_zfs(b"\x00" * 8 + zb.uberblock("<"))


class TestParse:
    def test_find_slots(self):
        (slot,) = find_slots(FRAG)
        assert (slot.off, slot.fmt, slot.size, slot.sealed) == (UB_OFF, "<", zb.SLOT, True)

    def test_unsealed_slot(self):
        (slot,) = find_slots(zb.uberblock(">", eck=False))
        assert (slot.fmt, slot.sealed) == (">", False)

    def test_4k_slot_trailer(self):
        """ashift 12 pools use 4 KiB slots; the trailer sits at the slot end."""
        size = 4096
        slot = bytearray(zb.uberblock("<", eck=False) + bytes(size - zb.SLOT))
        struct.pack_into("<Q", slot, size - zb.ECK_LEN, zb.ZEC_MAGIC)
        slot[-32:] = zb.label_cksum(slot, 0, "<")
        (found,) = find_slots(bytes(slot))
        assert (found.size, found.sealed) == (size, True)

    def test_nested_pairs_walked(self):
        names = [p.name for p in nv_pairs(FRAG)]
        assert names == [b"version", b"name", b"pool_guid", b"vdev_tree", b"type", b"id"]


class TestUberblockOps:
    def test_version_set_and_resealed(self):
        out = mutate_zfs(FRAG, rng=_rng(Op.UB_VERSION, [0], [0]))
        assert _u64(out, UB_OFF + 8) == UB_VERSIONS[0]
        assert zb.cksum_ok(_slot(out), UB_OFF, "<")

    def test_big_endian_txg(self):
        data = zb.uberblock(">", txg=7, offset=0)
        pool = [op for op in Op if op.value <= Op.BP_BIRTH or op is Op.ECK_BREAK]
        rng = ScriptedRng(randints=[0], choice_idxs=[pool.index(Op.UB_TXG), len(U64_EDGES)])
        out = mutate_zfs(data, rng=rng)
        assert _u64(out, 16, ">") == 6  # regress: txg - 1
        assert zb.cksum_ok(out, 0, ">")

    def test_prop_field_lsize(self):
        before = _u64(FRAG, UB_OFF + 40 + 48)
        out = mutate_zfs(FRAG, rng=_rng(Op.BP_PROP, [0, 0], [0]))
        after = _u64(out, UB_OFF + 40 + 48)
        assert after & 0xFFFF == 0
        assert after >> 16 == before >> 16
        assert zb.cksum_ok(_slot(out), UB_OFF, "<")

    def test_same_value_still_changes(self):
        """UB_VERSIONS holds 5000, the fixture's own version."""
        k = UB_VERSIONS.index(5000)
        out = mutate_zfs(FRAG, rng=_rng(Op.UB_VERSION, [0], [k]))
        assert _u64(out, UB_OFF + 8) == 5000 ^ 1


class TestNvlistOps:
    def test_encode_size(self):
        out = mutate_zfs(FRAG, rng=_rng(Op.NV_SIZE, [0, 0], [0]))
        assert _i32(out, FIRST_PAIR) == NV_SIZE_EDGES[0]

    def test_strlen_targets_string_pairs(self):
        name = next(p for p in nv_pairs(FRAG) if p.name == b"name")
        out = mutate_zfs(FRAG, rng=_rng(Op.NV_STRLEN, [0], [0]))
        assert struct.unpack_from(">I", out, name.value_off)[0] != 4
        assert out[: name.value_off] == FRAG[: name.value_off]

    def test_terminator_removed(self):
        out = mutate_zfs(FRAG, rng=_rng(Op.NV_TERM, [0], [0]))
        assert out != FRAG
        assert out.count(bytes(8)) <= FRAG.count(bytes(8))

    def test_label_phys_resealed(self):
        """nvlist at label+16K with a trailer at label+128K: reseal over that region."""
        img = bytearray(LABEL)
        nv = zb.label_nvlist()
        img[PHYS_OFF : PHYS_OFF + len(nv)] = nv
        struct.pack_into("<Q", img, PHYS_END - zb.ECK_LEN, zb.ZEC_MAGIC)
        img[PHYS_END - 32 : PHYS_END] = zb.label_cksum(img[PHYS_OFF:PHYS_END], PHYS_OFF, "<")
        assert zb.cksum_ok(bytes(img[PHYS_OFF:PHYS_END]), PHYS_OFF, "<")

        pool = [op for op in Op if Op.NV_SIZE <= op <= Op.NV_TERM or op is Op.ECK_BREAK]
        rng = ScriptedRng(randints=[0, 0], choice_idxs=[pool.index(Op.NV_SIZE), 0])
        out = mutate_zfs(bytes(img), max_len=LABEL, rng=rng)
        assert _i32(out, PHYS_OFF + FIRST_PAIR) == NV_SIZE_EDGES[0]
        assert zb.cksum_ok(out[PHYS_OFF:PHYS_END], PHYS_OFF, "<")


class TestFalsification:
    def test_eck_break_fails_verification(self):
        out = mutate_zfs(FRAG, rng=_rng(Op.ECK_BREAK, [0, 5]))
        assert out[:UB_OFF] == FRAG[:UB_OFF]
        assert not zb.cksum_ok(_slot(out), UB_OFF, "<")

    def test_unsealed_slot_offers_no_break(self):
        """Last pool entry is BP_BIRTH, not ECK_BREAK, without a trailer."""
        data = zb.uberblock("<", eck=False)
        pool = [op for op in Op if op.value <= Op.BP_BIRTH]
        rng = ScriptedRng(randints=[0, 0], choice_idxs=[len(pool) - 1, 0])
        out = mutate_zfs(data, rng=rng)
        assert _u64(out, 40 + 72) == U64_EDGES[0]


@pytest.mark.parametrize("op", list(Op))
def test_every_op_changes_fragment(op):
    rng = ScriptedRng(randints=[0] * 8, choice_idxs=[int(op), *[0] * 8])
    out = mutate_zfs(FRAG, rng=rng)
    assert out is not None
    assert out != FRAG
    assert len(out) == len(FRAG)


def test_max_len_truncates():
    out = mutate_zfs(FRAG, max_len=100, rng=_rng(Op.UB_GUID, [0], [0]))
    assert len(out) == 100


@pytest.mark.parametrize("data", [b"", b"\x01", bytes(4096), b"hello world" * 50])
def test_declines_non_zfs(data):
    assert mutate_zfs(data, rng=RandPool(seed=1)) is None


def test_adversarial_inputs_never_raise():
    """Truncations, random corruptions and hostile nvlist sizes stay bounded."""
    rng = RandPool(seed=0xF5)
    hostile = [FRAG[:n] for n in range(0, len(FRAG), 37)]
    for _ in range(200):
        buf = bytearray(FRAG)
        for _ in range(rng.randint(1, 16)):
            buf[rng.randint(0, len(buf) - 1)] = rng.randint(0, 255)
        hostile.append(bytes(buf))
    # Encode sizes that point backwards, at themselves, and past EOF.
    for esz in (0, 1, 8, -8, 0x7FFFFFFF):
        buf = bytearray(FRAG)
        struct.pack_into(">i", buf, FIRST_PAIR, esz)
        hostile.append(bytes(buf))
    # Self-nesting: every pair claims to be an embedded nvlist.
    deep = bytearray(FRAG)
    for p in nv_pairs(FRAG):
        struct.pack_into(">i", deep, p.type_off, zb.DT_NVLIST)
    hostile.append(bytes(deep))

    for data in hostile:
        sniff_zfs(data)
        out = mutate_zfs(data, max_len=1500, rng=rng)
        assert out is None or len(out) <= 1500
