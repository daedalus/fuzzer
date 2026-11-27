"""TCP/IP packet mutator: IPv4/IPv6 + TCP/UDP/ICMP header edits.

Containers locate IP packets (bounded walks):

  raw IP    | IPv4 / IPv6 header ...
  Ethernet  | dst 6 | src 6 | [8100 tci] | type 0800/86DD | IP ...
  libpcap   | hdr 24 (magic, linktype @20) | rec 16 (incl_len @8) | frame | rec ...

  IPv4  0 ver|ihl  1 tos  2 totlen  4 id  6 flags|frag  8 ttl  9 proto  10 csum
        12 src  16 dst  20 [options]
  IPv6  0 ver|tc|flow  4 plen  6 nh  7 hop  8 src  24 dst  40 [ext: nh, len, ..]
  TCP   0 sport 2 dport 4 seq 8 ack 12 doff 13 flags 14 win 16 csum 18 urg 20 [opts]
  UDP   0 sport 2 dport 4 len 6 csum 8 payload
  ICMP  0 type 1 code 2 csum 4 rest 8 [quoted IP header, error types]

Why each edit matters to a parser (lwIP, picoTCP, smoltcp, Wireshark/tcpdump
dissectors, netfilter):
- IHL / data offset / ext hdr len / option len drive pointer arithmetic:
  out-of-bounds reads and underflow when they are < minimum or past the buffer.
- total length / payload length / UDP length that disagree with the bytes
  present hit truncation and length-underflow paths.
- DF+MF, fragment offset 0x1FFF and fragment ext headers hit reassembly.
- TCP flag combos (SYN+FIN, null, Xmas) and URG past data hit state machines.
- LSRR/SSRR/RR/timestamp pointers are rewritten in place by forwarders.
- type-0 routing, looping ext chains and hop-by-hop not first hit chain walks.
- ICMP error quotes are re-parsed as IP: a truncated quote hits nested parsing.

Checksums (RFC 1071) are repaired after every edit over the lengths the
mutated packet claims, clamped to the bytes present, so the edit passes
validation. BREAK_CSUM flips bit 0 of one repaired checksum (off by one,
never the +0/-0 alias); ZERO_UDP_CSUM stores 0 ("none" over IPv4, illegal
over IPv6). Every edit is length-preserving: pcap incl_len/orig_len stay valid.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

# Containers.
ETH_HDR = 14
ETHERTYPE_OFF = 12
VLAN_TAG = 4
ETH_VLAN = 0x8100
ETHERTYPE_VERSION = {0x0800: 4, 0x86DD: 6}

PCAP_HDR = 24
PCAP_REC = 16
PCAP_LINKTYPE_OFF = 20
PCAP_INCL_OFF = 8
PCAP_ORDER = {
    b"\xa1\xb2\xc3\xd4": ">",
    b"\xa1\xb2\x3c\x4d": ">",
    b"\xd4\xc3\xb2\xa1": "<",
    b"\x4d\x3c\xb2\xa1": "<",
}
LINK_ETHER, LINK_RAW, LINK_IPV4, LINK_IPV6 = 1, 101, 228, 229
LINKTYPE_MASK = 0xFFFF
MAX_PACKETS = 4096
MAX_RECORDS = 65536
MAX_EXT_HDRS = 8

# IP headers.
V4_MIN = 20
V6_HDR = 40
V4_TOTLEN_OFF, V4_ID_OFF, V4_FRAG_OFF = 2, 4, 6
V4_TTL_OFF, V4_PROTO_OFF, V4_CSUM_OFF = 8, 9, 10
V4_ADDRS = slice(12, 20)
V6_PLEN_OFF, V6_NH_OFF, V6_HOP_OFF = 4, 6, 7
V6_ADDRS = slice(8, 40)
V6_EXT_MIN = 8
FLOW_MASK = 0xFFFFF

TCP, UDP, ICMP, ICMP6 = 6, 17, 1, 58
HOPOPT, ROUTING, FRAGMENT, AH, DSTOPTS, NO_NEXT = 0, 43, 44, 51, 60, 59
EXT_HDRS = frozenset({HOPOPT, ROUTING, FRAGMENT, AH, DSTOPTS})
#: Protocols a real header names; text bytes (>= 0x20) almost never match.
KNOWN_PROTOS = frozenset(
    {HOPOPT, ICMP, 2, 4, TCP, UDP, 41, ROUTING, FRAGMENT, 47, 50, AH, ICMP6, NO_NEXT, DSTOPTS, 132}
)

# Transport headers.
L4_CSUM_OFF = {TCP: 16, UDP: 6, ICMP: 2, ICMP6: 2}
L4_MIN = {TCP: 20, UDP: 8, ICMP: 8, ICMP6: 8}
TCP_MIN = 20
TCP_SEQ_OFF, TCP_DOFF_OFF, TCP_FLAGS_OFF, TCP_WIN_OFF, TCP_URG_OFF = 4, 12, 13, 14, 18
TCP_URG = 0x20
UDP_HDR = 8
UDP_LEN_OFF = 4
ICMP_QUOTE_OFF = 8
CSUM_NONE = 0  # UDP over IPv4: no checksum
CSUM_ZERO = 0xFFFF  # computed 0 stored as -0
U16_MASK = 0xFFFF
_FOLD = 0xFFFF  # 2**16 == 1 (mod 0xFFFF): word sum == big int mod 0xFFFF
#: Bytes from which numpy's word sum beats the big-int modulo (crossover ~700).
NUMPY_MIN_LEN = 768

# Edge tables.
IHL_EDGES = (0, 1, 4, 5, 6, 15)
DOFF_EDGES = IHL_EDGES
TOS_EDGES = (0x00, 0x01, 0x02, 0x03, 0xB8, 0xFC, 0xFF)  # ECN 01/10/11, EF, all
U16_EDGES = (0, 1, 0x7FFF, 0x8000, 0xFFFF)
U32_EDGES = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
#: flags|offset: none, MF, DF, DF+MF, evil bit, max offset, MF+max, MF+8, DF+8, all.
FRAG_EDGES = (0x0000, 0x2000, 0x4000, 0x6000, 0x8000, 0x1FFF, 0x3FFF, 0x2001, 0x4001, 0xFFFF)
TTL_EDGES = (0, 1, 255)
PROTO_EDGES = (TCP, UDP, ICMP, ICMP6, 41, 47, 0, 4, 50, 132, 255)
WINDOW_EDGES = (0, 1, 0xFFFF)
#: SYN+FIN, null, Xmas, SYN+RST, URG, SYN+ACK, SYN+ECE+CWR, RST+FIN, all, all but ECE/CWR.
TCP_FLAG_EDGES = (0x03, 0x00, 0x29, 0x06, 0x20, 0x12, 0xC2, 0x05, 0xFF, 0x3F)
BYTE_EDGES = (0x00, 0x01, 0x7F, 0x80, 0xFF)
PORT_EDGES = (0, 53, 67, 68, 123, 443, 500, 1900, 4789, 5353, 0xFFFF)
FLOW_EDGES = (0, 1, 0x80000, 0xFFFFF)
V6_NH_EDGES = (HOPOPT, ROUTING, FRAGMENT, DSTOPTS, NO_NEXT, TCP, UDP, ICMP6, ICMP, 41, AH, 135, 255)

#: IPv4 options, written at offset 20 and cut to the option area.
IP_OPTIONS = (
    b"\x83\x00",  # LSRR len 0
    b"\x83\x01",  # LSRR len 1
    b"\x83\xff\x04",  # LSRR len overruns header
    b"\x89\x07\x03",  # SSRR pointer < 4
    b"\x07\x07\xff",  # RR pointer past option
    b"\x07\x03\x04",  # RR with no room for one address
    b"\x44\x04\x05",  # timestamp len 4, pointer 5
    b"\x44\x0c\x0d\x0f",  # timestamp pointer past end, invalid flag 15
    b"\x44\x28\x05\x03",  # prespecified timestamps, len 40
    b"\x00",  # EOL first
    b"\x01" * 40,  # NOPs only, no EOL
    b"\x94\x04\x00\x00",  # router alert
    b"\x82\x0b",  # security len 11
)
#: TCP options, written at offset 20 of the segment and cut to the option area.
TCP_OPTIONS = (
    b"\x02\x03\x05",  # MSS len 3
    b"\x02\xff",  # MSS len overrun
    b"\x02\x04\x00\x00",  # MSS 0
    b"\x03\x03\x0f",  # window scale 15 (> 14)
    b"\x03\x03\xff",  # window scale 255
    b"\x04\x03",  # SACK-permitted len 3
    b"\x05\x0c" + bytes(10),  # SACK len 12: not 2 + 8k
    b"\x05\x06\x00\x00\x00\x01",  # SACK len 6: half a block
    b"\x05\x22" + bytes(32),  # SACK 4 blocks, past the option area
    b"\x08\x09" + bytes(7),  # timestamps len 9
    b"\x08\x0a" + b"\xff" * 8,  # timestamps max
    b"\x02\x00",  # len 0: infinite loop bait
    b"\x04\x01",  # len 1
    b"\x01" * 40,  # NOPs only, no EOL
    b"\x22\xff",  # fast open len overrun
    b"\x1e\x02",  # MPTCP len 2
)
ORIG_NH = -1  # ext template inner next header: the packet's original one
#: (header type, inner next header, 7 bytes after it) written at offset 40.
V6_EXT_HDRS = (
    (HOPOPT, ORIG_NH, bytes([0, 1, 4, 0, 0, 0, 0])),  # valid, PadN
    (HOPOPT, ORIG_NH, bytes([0xFF, 1, 4, 0, 0, 0, 0])),  # hdr ext len overrun
    (HOPOPT, HOPOPT, bytes([0, 1, 4, 0, 0, 0, 0])),  # hop-by-hop not first
    (HOPOPT, ORIG_NH, bytes([0, 0xC2, 4, 0xFF, 0xFF, 0xFF, 0xFF])),  # jumbo 0xFFFFFFFF
    (HOPOPT, ORIG_NH, bytes([0, 0x05, 2, 0, 0, 0, 0])),  # router alert
    (ROUTING, ORIG_NH, bytes([0, 0, 1, 0, 0, 0, 0])),  # type 0, no addresses
    (ROUTING, ORIG_NH, bytes([2, 0, 0xFF, 0, 0, 0, 0])),  # type 0, segleft 255
    (ROUTING, ROUTING, bytes([0, 4, 1, 0, 0, 0, 0])),  # SRH looping
    (FRAGMENT, ORIG_NH, bytes([0, 0xFF, 0xF9, 0, 0, 0, 1])),  # max offset + M
    (FRAGMENT, ORIG_NH, bytes([0, 0, 1, 0, 0, 0, 1])),  # first fragment
    (FRAGMENT, FRAGMENT, bytes([0, 0, 0, 0, 0, 0, 2])),  # nested fragment
    (DSTOPTS, ORIG_NH, bytes([0xFF, 1, 4, 0, 0, 0, 0])),  # hdr ext len overrun
    (DSTOPTS, ORIG_NH, bytes([0, 0x80, 0xFF, 0, 0, 0, 0])),  # option len overrun
    (DSTOPTS, HOPOPT, bytes([0, 1, 4, 0, 0, 0, 0])),  # hop-by-hop after dest opts
)
ICMP4_TYPES = (
    (0, 0), (3, 0), (3, 3), (3, 4), (3, 15), (5, 1), (8, 0),
    (11, 0), (11, 1), (12, 0), (12, 2), (13, 0), (17, 0), (255, 255),
)  # fmt: skip
ICMP6_TYPES = (
    (1, 0), (1, 4), (2, 0), (3, 0), (4, 0), (4, 1), (128, 0), (129, 0),
    (130, 0), (133, 0), (134, 0), (135, 0), (136, 0), (137, 0), (143, 0), (255, 255),
)  # fmt: skip
ICMP4_ERR = (3, 3)  # port unreachable
ICMP6_ERR = (1, 4)  # port unreachable
#: Quoted bytes the outer length keeps: inside / at the end of v4 and v6 headers.
QUOTE_CUTS = (0, 1, 8, 19, 20, 28, 39, 40, 48)


class Op(IntEnum):
    """Packet edit. Values are stable ids; the draw is a choice over packet_ops()."""

    V4_IHL = 0
    V4_TOS = 1
    V4_TOTLEN = 2
    V4_ID = 3
    V4_FRAG = 4
    V4_TTL = 5
    V4_PROTO = 6
    V4_OPTION = 7
    V6_PLEN = 8
    V6_NH = 9
    V6_HOPLIM = 10
    V6_FLOW = 11
    V6_EXT = 12
    TCP_DOFF = 13
    TCP_FLAGS = 14
    TCP_SEQACK = 15
    TCP_WINDOW = 16
    TCP_OPTION = 17
    PAYLOAD = 18
    UDP_LEN = 19
    UDP_PORT = 20
    ICMP_TYPE = 21
    ICMP_QUOTE = 22
    BREAK_CSUM = 23
    ZERO_UDP_CSUM = 24


class Site(IntEnum):
    """Checksum BREAK_CSUM may corrupt."""

    IP = 0
    L4 = 1


V4_FIELD_OPS = (Op.V4_IHL, Op.V4_TOS, Op.V4_TOTLEN, Op.V4_ID, Op.V4_FRAG, Op.V4_TTL, Op.V4_PROTO)
V6_FIELD_OPS = (Op.V6_PLEN, Op.V6_NH, Op.V6_HOPLIM, Op.V6_FLOW)
TCP_FIELD_OPS = (Op.TCP_DOFF, Op.TCP_FLAGS, Op.TCP_SEQACK, Op.TCP_WINDOW)


@dataclass(frozen=True)
class _Layout:
    """Where things are in one IP packet, as a receiver would compute them."""

    version: int
    hdr: int  # IPv4 checksummed header (IHL, 20 when IHL is invalid); 40 for IPv6
    l4: int  # transport offset; -1 when the header chain leaves the packet
    end: int  # transport end: claimed length clamped to the bytes present
    proto: int

    @property
    def span(self) -> int:
        return self.end - self.l4


def inet_checksum(data: bytes) -> int:
    """RFC 1071 checksum to store; a region holding a valid one sums to 0.

    Short buffers: one big-int modulo replaces the word loop, since
    2**16 == 1 (mod 0xFFFF) makes the folded word sum the value mod 0xFFFF.
    Long buffers: numpy sums the big-endian words. Either way a nonzero sum
    folding to a multiple of 0xFFFF is 0xFFFF (-0), not 0.
    """
    if len(data) & 1:
        data = bytes(data) + b"\x00"
    if len(data) >= NUMPY_MIN_LEN:
        value = int(np.frombuffer(data, dtype=">u2").sum(dtype=np.uint64))
    else:
        value = int.from_bytes(data, "big")

    total = value % _FOLD
    if total == 0 and value:
        total = _FOLD
    return ~total & U16_MASK


def _u16(data: bytes | bytearray, off: int) -> int:
    return (data[off] << 8) | data[off + 1]


def _put16(pkt: bytearray, off: int, value: int) -> None:
    struct.pack_into(">H", pkt, off, value & U16_MASK)


# --- Locating packets ------------------------------------------------------


def _ip_basic(data: bytes, s: int, e: int) -> int:
    """IP version (4/6) when [s, e) holds a minimal header, else 0."""
    if e - s < V4_MIN:
        return 0
    ver = data[s] >> 4
    if ver == 4:
        ihl = (data[s] & 0xF) * 4
        return 4 if V4_MIN <= ihl <= e - s else 0
    if ver == 6 and e - s >= V6_HDR:
        return 6
    return 0


def _find_raw(data: bytes, s: int, e: int) -> int:
    return s if _ip_basic(data, s, e) else -1


def _find_v4(data: bytes, s: int, e: int) -> int:
    return s if _ip_basic(data, s, e) == 4 else -1


def _find_v6(data: bytes, s: int, e: int) -> int:
    return s if _ip_basic(data, s, e) == 6 else -1


def _eth_ip(data: bytes, s: int, e: int) -> int:
    """IP offset in an Ethernet II frame at [s, e) (one 802.1Q tag), else -1."""
    off = s + ETHERTYPE_OFF
    if off + 2 > e:
        return -1

    etype = _u16(data, off)
    if etype == ETH_VLAN:
        off += VLAN_TAG
        if off + 2 > e:
            return -1
        etype = _u16(data, off)

    want = ETHERTYPE_VERSION.get(etype)
    ip = off + 2
    return ip if want and _ip_basic(data, ip, e) == want else -1


_LINK_FINDERS: dict[int, Callable[[bytes, int, int], int]] = {
    LINK_ETHER: _eth_ip,
    LINK_RAW: _find_raw,
    LINK_IPV4: _find_v4,
    LINK_IPV6: _find_v6,
}


def _pcap_order(data: bytes) -> str:
    """Struct byte order of a classic pcap file, or '' if not one."""
    if len(data) < PCAP_HDR:
        return ""
    return PCAP_ORDER.get(data[:4], "")


def _pcap_link(data: bytes, order: str) -> Callable[[bytes, int, int], int] | None:
    link = struct.unpack_from(order + "I", data, PCAP_LINKTYPE_OFF)[0] & LINKTYPE_MASK
    return _LINK_FINDERS.get(link)


def _pcap_packets(data: bytes, order: str) -> list[tuple[int, int]]:
    """IP spans of pcap records; incl_len past EOF ends the walk."""
    find = _pcap_link(data, order)
    if find is None:
        return []

    n = len(data)
    fmt = order + "I"
    out: list[tuple[int, int]] = []
    pos = PCAP_HDR
    for _ in range(MAX_RECORDS):
        if pos + PCAP_REC > n or len(out) >= MAX_PACKETS:
            break
        body = pos + PCAP_REC
        incl = struct.unpack_from(fmt, data, pos + PCAP_INCL_OFF)[0]
        if incl > n - body:
            break
        ip = find(data, body, body + incl)
        if ip >= 0:
            out.append((ip, body + incl))
        pos = body + incl
    return out


def _v4_plausible(data: bytes, n: int) -> bool:
    """Raw IPv4: valid header checksum, or sane total length and protocol."""
    ihl = (data[0] & 0xF) * 4
    if inet_checksum(data[:ihl]) == 0:
        return True
    tot = _u16(data, V4_TOTLEN_OFF)
    return ihl <= tot <= n and data[V4_PROTO_OFF] in KNOWN_PROTOS


def _raw_strict(data: bytes) -> bool:
    """Whole buffer is one IP packet (filters text that starts 0x4X/0x6X)."""
    n = len(data)
    ver = _ip_basic(data, 0, n)
    if ver == 4:
        return _v4_plausible(data, n)
    if ver == 6:
        return V6_HDR + _u16(data, V6_PLEN_OFF) <= n and data[V6_NH_OFF] in KNOWN_PROTOS
    return False


def locate_packets(data: bytes) -> list[tuple[int, int]]:
    """(start, end) of every IP packet: pcap records, an Ethernet frame or raw IP."""
    order = _pcap_order(data)
    if order:
        return _pcap_packets(data, order)

    n = len(data)
    ip = _eth_ip(data, 0, n)
    if ip >= 0:
        return [(ip, n)]
    return [(0, n)] if _raw_strict(data) else []


def sniff_tcpip(data: bytes) -> bool:
    """pcap with a supported linktype, an Ethernet IP frame, or raw IP."""
    order = _pcap_order(data)
    if order:
        return _pcap_link(data, order) is not None
    return _eth_ip(data, 0, len(data)) >= 0 or _raw_strict(data)


# --- Layout ----------------------------------------------------------------


def _layout4(pkt: bytearray | bytes, n: int) -> _Layout:
    ihl = (pkt[0] & 0xF) * 4
    proto = pkt[V4_PROTO_OFF]
    if not V4_MIN <= ihl <= n:
        return _Layout(4, V4_MIN, -1, n, proto)

    tot = _u16(pkt, V4_TOTLEN_OFF)
    end = tot if ihl <= tot <= n else n
    return _Layout(4, ihl, ihl, end, proto)


def _ext_step(nh: int, hdr_len: int) -> int:
    if nh == FRAGMENT:
        return V6_EXT_MIN
    if nh == AH:
        return (hdr_len + 2) * 4
    return (hdr_len + 1) * 8


def _layout6(pkt: bytearray | bytes, n: int) -> _Layout:
    """Walk at most MAX_EXT_HDRS extension headers to the transport header."""
    plen = _u16(pkt, V6_PLEN_OFF)
    end = V6_HDR + plen if 0 < plen <= n - V6_HDR else n
    nh, off = pkt[V6_NH_OFF], V6_HDR

    for _ in range(MAX_EXT_HDRS):
        if nh not in EXT_HDRS:
            return _Layout(6, V6_HDR, off if off <= end else -1, end, nh)
        if off + 2 > end:
            break
        nh, off = pkt[off], off + _ext_step(nh, pkt[off + 1])
    return _Layout(6, V6_HDR, -1, end, nh)


def _layout(pkt: bytearray | bytes) -> _Layout | None:
    n = len(pkt)
    if n < V4_MIN:
        return None
    ver = pkt[0] >> 4
    if ver == 4:
        return _layout4(pkt, n)
    if ver == 6 and n >= V6_HDR:
        return _layout6(pkt, n)
    return None


def _l4_ready(lay: _Layout) -> bool:
    """Transport header present in full, with a known checksum."""
    need = L4_MIN.get(lay.proto)
    return need is not None and lay.l4 >= 0 and lay.span >= need


def _payload(pkt: bytearray | bytes, lay: _Layout) -> tuple[int, int] | None:
    """Transport payload span, or None when empty or unlocatable."""
    if lay.proto == UDP:
        start = lay.l4 + UDP_HDR
    else:
        doff = (pkt[lay.l4 + TCP_DOFF_OFF] >> 4) * 4
        if doff < TCP_MIN:
            return None
        start = lay.l4 + doff
    return (start, lay.end) if start < lay.end else None


# --- Applicable ops --------------------------------------------------------


def _sites(lay: _Layout) -> tuple[Site, ...]:
    ip = (Site.IP,) if lay.version == 4 else ()
    return ip + ((Site.L4,) if _l4_ready(lay) else ())


def _ip_ops(pkt: bytes, lay: _Layout) -> list[Op]:
    if lay.version == 4:
        opt = [Op.V4_OPTION] if lay.hdr > V4_MIN and lay.l4 >= 0 else []
        return [*V4_FIELD_OPS, *opt]
    ext = [Op.V6_EXT] if len(pkt) >= V6_HDR + V6_EXT_MIN else []
    return [*V6_FIELD_OPS, *ext]


def _tcp_ops(pkt: bytes, lay: _Layout) -> list[Op]:
    ops = list(TCP_FIELD_OPS)
    doff = (pkt[lay.l4 + TCP_DOFF_OFF] >> 4) * 4
    if TCP_MIN < doff <= lay.span:
        ops.append(Op.TCP_OPTION)
    if _payload(pkt, lay):
        ops.append(Op.PAYLOAD)
    return ops


def _udp_ops(pkt: bytes, lay: _Layout) -> list[Op]:
    ops = [Op.UDP_LEN, Op.UDP_PORT]
    if _payload(pkt, lay):
        ops.append(Op.PAYLOAD)
    if _u16(pkt, lay.l4 + L4_CSUM_OFF[UDP]) != CSUM_NONE:
        ops.append(Op.ZERO_UDP_CSUM)
    return ops


def _l4_ops(pkt: bytes, lay: _Layout) -> list[Op]:
    if not _l4_ready(lay):
        return []
    if lay.proto == TCP:
        return _tcp_ops(pkt, lay)
    if lay.proto == UDP:
        return _udp_ops(pkt, lay)
    return [Op.ICMP_TYPE, Op.ICMP_QUOTE]


def _ops_for(pkt: bytes, lay: _Layout) -> tuple[Op, ...]:
    ops = _ip_ops(pkt, lay) + _l4_ops(pkt, lay)
    if _sites(lay):
        ops.append(Op.BREAK_CSUM)
    return tuple(sorted(ops))


def packet_ops(pkt: bytes) -> tuple[Op, ...]:
    """Ops applicable to one IP packet, in Op order (the choice() domain)."""
    lay = _layout(pkt)
    return () if lay is None else _ops_for(pkt, lay)


# --- Checksum repair -------------------------------------------------------


def _pseudo(pkt: bytearray, lay: _Layout, length: int) -> bytes:
    if lay.proto == ICMP:
        return b""
    if lay.version == 4:
        return bytes(pkt[V4_ADDRS]) + struct.pack(">BBH", 0, lay.proto, length & U16_MASK)
    return bytes(pkt[V6_ADDRS]) + struct.pack(">IxxxB", length, lay.proto)


def _store_l4(pkt: bytearray, lay: _Layout) -> None:
    off = lay.l4 + L4_CSUM_OFF[lay.proto]
    _put16(pkt, off, 0)
    c = inet_checksum(_pseudo(pkt, lay, lay.span) + bytes(pkt[lay.l4 : lay.end]))
    if lay.proto == UDP and c == CSUM_NONE:
        c = CSUM_ZERO
    _put16(pkt, off, c)


def _repair(pkt: bytearray) -> _Layout | None:
    """Recompute the IPv4 header and transport checksums in place."""
    lay = _layout(pkt)
    if lay is None:
        return None

    if lay.version == 4:
        _put16(pkt, V4_CSUM_OFF, 0)
        _put16(pkt, V4_CSUM_OFF, inet_checksum(bytes(pkt[: lay.hdr])))
    if _l4_ready(lay):
        _store_l4(pkt, lay)
    return lay


# --- Edits -----------------------------------------------------------------


def _pick(rng: RandPool, table: tuple, cur: object) -> object:
    """Drawn edge, or the next one in the table when it equals *cur*."""
    first = rng.choice(table)
    start = table.index(first)
    k = len(table)
    for i in range(k):
        cand = table[(start + i) % k]
        if cand != cur:
            return cand
    return first


def _write_opt(
    pkt: bytearray, off: int, area: int, table: tuple[bytes, ...], rng: RandPool
) -> None:
    """Overwrite an option area's head with a template cut to the area."""
    cur = bytes(pkt[off : off + area])
    cands = tuple(t[:area] + cur[len(t) :] for t in table)
    pkt[off : off + area] = _pick(rng, cands, cur)


def _v4_ihl(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    pkt[0] = 0x40 | _pick(rng, IHL_EDGES, pkt[0] & 0xF)


def _v4_tos(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    pkt[1] = _pick(rng, TOS_EDGES, pkt[1])


def _v4_totlen(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    n = len(pkt)
    edges = tuple(v & U16_MASK for v in (0, lay.hdr - 1, n - 1, n + 1, U16_MASK))
    _put16(pkt, V4_TOTLEN_OFF, _pick(rng, edges, _u16(pkt, V4_TOTLEN_OFF)))


def _v4_id(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    _put16(pkt, V4_ID_OFF, _pick(rng, U16_EDGES, _u16(pkt, V4_ID_OFF)))


def _v4_frag(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    _put16(pkt, V4_FRAG_OFF, _pick(rng, FRAG_EDGES, _u16(pkt, V4_FRAG_OFF)))


def _v4_ttl(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    pkt[V4_TTL_OFF] = _pick(rng, TTL_EDGES, pkt[V4_TTL_OFF])


def _v4_proto(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    pkt[V4_PROTO_OFF] = _pick(rng, PROTO_EDGES, pkt[V4_PROTO_OFF])


def _v4_option(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    _write_opt(pkt, V4_MIN, lay.hdr - V4_MIN, IP_OPTIONS, rng)


def _v6_plen(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    have = len(pkt) - V6_HDR
    edges = tuple(v & U16_MASK for v in (0, 1, have - 1, have + 1, U16_MASK))
    _put16(pkt, V6_PLEN_OFF, _pick(rng, edges, _u16(pkt, V6_PLEN_OFF)))


def _v6_nh(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    pkt[V6_NH_OFF] = _pick(rng, V6_NH_EDGES, pkt[V6_NH_OFF])


def _v6_hop(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    pkt[V6_HOP_OFF] = _pick(rng, TTL_EDGES, pkt[V6_HOP_OFF])


def _v6_flow(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    word = struct.unpack_from(">I", pkt, 0)[0]
    flow = _pick(rng, FLOW_EDGES, word & FLOW_MASK)
    struct.pack_into(">I", pkt, 0, (word & ~FLOW_MASK) | flow)


def _v6_ext(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    """Insert one extension header at 40, over the start of the payload."""
    orig = pkt[V6_NH_OFF]
    cur = (orig, bytes(pkt[V6_HDR : V6_HDR + V6_EXT_MIN]))
    cands = tuple(
        (typ, bytes([orig if inner == ORIG_NH else inner]) + tail)
        for typ, inner, tail in V6_EXT_HDRS
    )
    typ, body = _pick(rng, cands, cur)
    pkt[V6_NH_OFF] = typ
    pkt[V6_HDR : V6_HDR + V6_EXT_MIN] = body


def _tcp_doff(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    off = lay.l4 + TCP_DOFF_OFF
    pkt[off] = (pkt[off] & 0x0F) | (_pick(rng, DOFF_EDGES, pkt[off] >> 4) << 4)


def _tcp_flags(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    """Flag combo; with URG, the urgent pointer lands one past the data."""
    off = lay.l4 + TCP_FLAGS_OFF
    pkt[off] = _pick(rng, TCP_FLAG_EDGES, pkt[off])
    if not pkt[off] & TCP_URG:
        return
    span = _payload(pkt, lay)
    data_len = span[1] - span[0] if span else 0
    _put16(pkt, lay.l4 + TCP_URG_OFF, data_len + 1)


def _tcp_seqack(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    off = lay.l4 + TCP_SEQ_OFF + 4 * rng.randint(0, 1)
    cur = struct.unpack_from(">I", pkt, off)[0]
    struct.pack_into(">I", pkt, off, _pick(rng, U32_EDGES, cur))


def _tcp_window(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    off = lay.l4 + TCP_WIN_OFF
    _put16(pkt, off, _pick(rng, WINDOW_EDGES, _u16(pkt, off)))


def _tcp_option(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    doff = (pkt[lay.l4 + TCP_DOFF_OFF] >> 4) * 4
    _write_opt(pkt, lay.l4 + TCP_MIN, doff - TCP_MIN, TCP_OPTIONS, rng)


def _payload_edit(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    start, end = _payload(pkt, lay) or (0, 0)
    if start >= end:
        return
    pos = start + rng.randint(0, end - start - 1)
    pkt[pos] = _pick(rng, BYTE_EDGES, pkt[pos])


def _udp_len(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    span = lay.span
    edges = tuple(v & U16_MASK for v in (0, UDP_HDR - 1, UDP_HDR, span - 1, span + 1, U16_MASK))
    off = lay.l4 + UDP_LEN_OFF
    _put16(pkt, off, _pick(rng, edges, _u16(pkt, off)))


def _udp_port(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    off = lay.l4 + 2 * rng.randint(0, 1)
    _put16(pkt, off, _pick(rng, PORT_EDGES, _u16(pkt, off)))


def _icmp_type(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    table = ICMP6_TYPES if lay.proto == ICMP6 else ICMP4_TYPES
    l4 = lay.l4
    pkt[l4], pkt[l4 + 1] = _pick(rng, table, (pkt[l4], pkt[l4 + 1]))


def _icmp_quote(pkt: bytearray, lay: _Layout, rng: RandPool) -> None:
    """Error message quoting this packet's own IP header, outer length cut.

    The outer length keeps QUOTE_CUTS[k] quoted bytes, so the receiver sees a
    quote truncated inside (or exactly at the end of) the inner header.
    """
    l4 = lay.l4
    pkt[l4], pkt[l4 + 1] = ICMP6_ERR if lay.proto == ICMP6 else ICMP4_ERR
    pkt[l4 + 4 : l4 + ICMP_QUOTE_OFF] = bytes(4)

    quote = l4 + ICMP_QUOTE_OFF
    hdr = bytes(pkt[: lay.hdr])
    room = len(pkt) - quote
    pkt[quote : quote + min(room, len(hdr))] = hdr[:room]

    claimed = quote + rng.choice(QUOTE_CUTS)
    if lay.version == 4:
        _put16(pkt, V4_TOTLEN_OFF, claimed)
        return
    _put16(pkt, V6_PLEN_OFF, claimed - V6_HDR)


def _break_csum(pkt: bytearray, _lay: _Layout, rng: RandPool) -> None:
    """Repair, then flip bit 0 of one checksum (+-1: never the +0/-0 alias)."""
    lay = _repair(pkt)
    if lay is None:
        return
    site = rng.choice(_sites(lay))
    off = V4_CSUM_OFF if site is Site.IP else lay.l4 + L4_CSUM_OFF[lay.proto]
    pkt[off + 1] ^= 1


def _zero_udp(pkt: bytearray, lay: _Layout, _rng: RandPool) -> None:
    _repair(pkt)
    _put16(pkt, lay.l4 + L4_CSUM_OFF[UDP], CSUM_NONE)


_EDITS: dict[Op, Callable[[bytearray, _Layout, RandPool], None]] = {
    Op.V4_IHL: _v4_ihl,
    Op.V4_TOS: _v4_tos,
    Op.V4_TOTLEN: _v4_totlen,
    Op.V4_ID: _v4_id,
    Op.V4_FRAG: _v4_frag,
    Op.V4_TTL: _v4_ttl,
    Op.V4_PROTO: _v4_proto,
    Op.V4_OPTION: _v4_option,
    Op.V6_PLEN: _v6_plen,
    Op.V6_NH: _v6_nh,
    Op.V6_HOPLIM: _v6_hop,
    Op.V6_FLOW: _v6_flow,
    Op.V6_EXT: _v6_ext,
    Op.TCP_DOFF: _tcp_doff,
    Op.TCP_FLAGS: _tcp_flags,
    Op.TCP_SEQACK: _tcp_seqack,
    Op.TCP_WINDOW: _tcp_window,
    Op.TCP_OPTION: _tcp_option,
    Op.PAYLOAD: _payload_edit,
    Op.UDP_LEN: _udp_len,
    Op.UDP_PORT: _udp_port,
    Op.ICMP_TYPE: _icmp_type,
    Op.ICMP_QUOTE: _icmp_quote,
    Op.BREAK_CSUM: _break_csum,
    Op.ZERO_UDP_CSUM: _zero_udp,
}
_SELF_REPAIR = frozenset({Op.BREAK_CSUM, Op.ZERO_UDP_CSUM})


def mutate_tcpip(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One length-preserving header edit on one packet; None if declined.

    Draw order: randint packet index, choice over packet_ops(packet), then
    the op's own draws (field selector randint / payload offset randint,
    then the choice from its edge table; BREAK_CSUM: choice of Site).
    """
    spans = locate_packets(data)
    if not spans:
        return None
    start, end = spans[rng.randint(0, len(spans) - 1)]

    pkt = bytearray(data[start:end])
    lay = _layout(pkt)
    if lay is None:
        return None
    ops = _ops_for(bytes(pkt), lay)
    if not ops:
        return None

    op = rng.choice(ops)
    _EDITS[op](pkt, lay, rng)
    if op not in _SELF_REPAIR:
        _repair(pkt)

    out = data[:start] + bytes(pkt) + data[end:]
    if out == data:
        return None
    return out[:max_len]
