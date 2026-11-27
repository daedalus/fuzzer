"""Minimal TCP/IP packet builders with valid checksums (tcpip mutator tests)."""

from __future__ import annotations

import struct
from enum import Enum

TCP, UDP, ICMP, ICMP6 = 6, 17, 1, 58
HOPOPT = 0

SRC4 = bytes([192, 0, 2, 1])
DST4 = bytes([198, 51, 100, 2])
SRC6 = bytes.fromhex("20010db8000000000000000000000001")
DST6 = bytes.fromhex("20010db8000000000000000000000002")

_L4_CSUM_OFF = {TCP: 16, UDP: 6, ICMP: 2, ICMP6: 2}


def _csum(data: bytes) -> int:
    """RFC 1071 one's-complement checksum, plain loop."""
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) | data[i + 1]
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def tcp_seg(payload: bytes = b"hello", options: bytes = b"", flags: int = 0x18) -> bytes:
    """TCP segment, checksum 0; *options* must be a multiple of 4 bytes."""
    doff = (20 + len(options)) // 4
    hdr = struct.pack(">HHIIBBHHH", 1234, 80, 1000, 2000, doff << 4, flags, 8192, 0, 0)
    return hdr + options + payload


def udp_dgram(payload: bytes = b"query") -> bytes:
    return struct.pack(">HHHH", 5353, 53, 8 + len(payload), 0) + payload


def icmp_msg(typ: int = 8, code: int = 0, payload: bytes = b"ping") -> bytes:
    return struct.pack(">BBHI", typ, code, 0, 0x00010001) + payload


def hbh(nh: int) -> bytes:
    """8-byte hop-by-hop header: next header *nh*, one PadN option."""
    return bytes([nh, 0, 1, 4, 0, 0, 0, 0])


def _fill(l4: bytes, proto: int, pseudo: bytes) -> bytes:
    off = _L4_CSUM_OFF.get(proto)
    if off is None:
        return l4
    c = _csum(pseudo + l4)
    if proto == UDP and c == 0:
        c = 0xFFFF
    return l4[:off] + struct.pack(">H", c) + l4[off + 2 :]


def ipv4(proto: int, l4: bytes, options: bytes = b"", ttl: int = 64) -> bytes:
    ihl = 5 + len(options) // 4
    total = ihl * 4 + len(l4)
    hdr = struct.pack(">BBHHHBBH4s4s", 0x40 | ihl, 0, total, 0x1234, 0x4000, ttl, proto, 0, SRC4, DST4)
    hdr += options
    hdr = hdr[:10] + struct.pack(">H", _csum(hdr)) + hdr[12:]
    pseudo = b"" if proto == ICMP else SRC4 + DST4 + struct.pack(">BBH", 0, proto, len(l4))
    return hdr + _fill(l4, proto, pseudo)


def ipv6(proto: int, l4: bytes, ext: bytes = b"", first_nh: int = HOPOPT) -> bytes:
    nh = first_nh if ext else proto
    hdr = struct.pack(">IHBB16s16s", 0x60000000, len(ext) + len(l4), nh, 64, SRC6, DST6)
    pseudo = SRC6 + DST6 + struct.pack(">IxxxB", len(l4), proto)
    return hdr + ext + _fill(l4, proto, pseudo)


class Tag(Enum):
    NONE = 0
    VLAN = 1


MAC_DST = bytes.fromhex("020000000001")
MAC_SRC = bytes.fromhex("020000000002")


def ether(ip: bytes, tag: Tag = Tag.NONE, dst: bytes = MAC_DST) -> bytes:
    etype = 0x0800 if ip[0] >> 4 == 4 else 0x86DD
    vlan = struct.pack(">HH", 0x8100, 7) if tag is Tag.VLAN else b""
    return dst + MAC_SRC + vlan + struct.pack(">H", etype) + ip


#: (magic as stored, struct byte order) for the four classic pcap variants.
PCAP_MAGICS = (
    (b"\xa1\xb2\xc3\xd4", ">"),
    (b"\xd4\xc3\xb2\xa1", "<"),
    (b"\xa1\xb2\x3c\x4d", ">"),
    (b"\x4d\x3c\xb2\xa1", "<"),
)


def pcap(frames: list[bytes], linktype: int = 1, variant: int = 1) -> bytes:
    """Classic libpcap file; *variant* indexes PCAP_MAGICS."""
    magic, order = PCAP_MAGICS[variant]
    out = magic + struct.pack(order + "HHiIII", 2, 4, 0, 0, 65535, linktype)
    for f in frames:
        out += struct.pack(order + "IIII", 0, 0, len(f), len(f)) + f
    return out
