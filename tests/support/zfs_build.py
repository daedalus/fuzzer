"""Pure-Python ZFS fragments for mutator tests: uberblock slots and XDR nvlists.

Independent of ``core/mutations/zfs.py``: layouts and the embedded label
checksum are re-derived here from the OpenZFS on-disk format.
"""

from __future__ import annotations

import hashlib
import struct

UB_MAGIC = 0x00BAB10C
ZEC_MAGIC = 0x0210DA7AB10C7A11
MMP_MAGIC = 0xA11CEA11
SLOT = 1024
ECK_LEN = 40  # zec_magic u64 + 4 x u64 checksum

# data_type_t values used by the fixtures.
DT_UINT64 = 8
DT_STRING = 9
DT_NVLIST = 19


def label_cksum(region: bytes, offset: int, fmt: str) -> bytes:
    """32 checksum bytes: SHA-256 with the verifier {offset,0,0,0} in place."""
    probe = bytearray(region)
    eck = len(probe) - ECK_LEN
    struct.pack_into(fmt + "4Q", probe, eck + 8, offset, 0, 0, 0)
    digest = hashlib.sha256(probe).digest()
    words = [int.from_bytes(digest[i : i + 8], "big") for i in range(0, 32, 8)]
    return struct.pack(fmt + "4Q", *words)


def cksum_ok(region: bytes, offset: int, fmt: str) -> bool:
    """True when the region's stored trailer checksum verifies."""
    return region[-32:] == label_cksum(region, offset, fmt)


def uberblock(fmt: str = "<", txg: int = 7, offset: int = 0, eck: bool = True) -> bytes:
    """One 1 KiB uberblock slot written in byte order *fmt*."""
    slot = bytearray(SLOT)
    struct.pack_into(fmt + "5Q", slot, 0, UB_MAGIC, 5000, txg, 0x1234, 1_700_000_000)

    # ub_rootbp: one DVA (vdev 0, asize 8 sectors, offset 0x40), then blk_prop.
    byteorder = 1 if fmt == "<" else 0
    prop = 7 | (7 << 16) | (2 << 32) | (7 << 40) | (11 << 48) | (byteorder << 63)
    struct.pack_into(fmt + "QQ", slot, 40, 8, 0x40)
    struct.pack_into(fmt + "Q", slot, 40 + 48, prop)
    struct.pack_into(fmt + "3Q", slot, 40 + 72, txg, txg, 3)
    struct.pack_into(fmt + "4Q", slot, 40 + 96, 1, 2, 3, 4)

    struct.pack_into(fmt + "QQ", slot, 168, 5000, MMP_MAGIC)
    if eck:
        struct.pack_into(fmt + "Q", slot, SLOT - ECK_LEN, ZEC_MAGIC)
        slot[-32:] = label_cksum(slot, offset, fmt)
    return bytes(slot)


def _xstr(s: str) -> bytes:
    raw = s.encode()
    return struct.pack(">I", len(raw)) + raw + bytes(-len(raw) % 4)


def _pair(name: str, dtype: int, nelem: int, value: bytes) -> bytes:
    body = _xstr(name) + struct.pack(">ii", dtype, nelem) + value
    esz = 8 + len(body)
    return struct.pack(">ii", esz, esz + 16) + body


def u64(name: str, v: int) -> bytes:
    return _pair(name, DT_UINT64, 1, struct.pack(">Q", v))


def string(name: str, s: str) -> bytes:
    return _pair(name, DT_STRING, 1, _xstr(s))


def nvlist_body(*pairs: bytes) -> bytes:
    """nvl_version 0, NV_UNIQUE_NAME, pairs, zero terminator."""
    return struct.pack(">ii", 0, 1) + b"".join(pairs) + bytes(8)


def nested(name: str, *pairs: bytes) -> bytes:
    """An NVLIST pair; the embedded list follows the pair header."""
    return _pair(name, DT_NVLIST, 1, b"") + nvlist_body(*pairs)


def nvlist(*pairs: bytes) -> bytes:
    """XDR-packed nvlist with its 4-byte stream header."""
    return b"\x01\x01\x00\x00" + nvlist_body(*pairs)


def label_nvlist() -> bytes:
    return nvlist(
        u64("version", 5000),
        string("name", "tank"),
        u64("pool_guid", 0xDEADBEEF),
        nested("vdev_tree", string("type", "disk"), u64("id", 0)),
    )


def fragment() -> bytes:
    """nvlist padded to 1 KiB, then one sealed LE uberblock at offset 1024."""
    nv = label_nvlist()
    return nv + bytes(SLOT - len(nv)) + uberblock("<", offset=SLOT)
