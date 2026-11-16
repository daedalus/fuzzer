"""Packet captures → message lists, for multi-message seeds (VirtFuzz port).

Reads classic pcap and pcapng; yields each packet's captured bytes with the
link pseudo-header stripped, so a seed carries what the target parses::

    LINKTYPE 201 (HCI H4 + phdr)   [dir u32][H4 packet]   -> H4 packet
    LINKTYPE 127 (radiotap)        [radiotap][802.11]     -> 802.11 frame
    anything else                  kept as captured

``frame`` joins packets with length prefixes that ``seq_mutate.split_msgs``
splits back, so sequence ops and ``MsgPass`` see one packet per message.

Captures are untrusted: a malformed header or block ends the walk; a bad
pseudo-header drops that packet only. ``buf`` may be an ``mmap``.
"""

import struct
from collections.abc import Iterator
from enum import Enum

# Classic pcap magics (microsecond / nanosecond timestamps).
_PCAP_MAGICS = (0xA1B2C3D4, 0xA1B23C4D)
_PCAP_HDR = 24
_PCAP_REC = 16
_PCAP_LINK_OFF = 20

# pcapng block types and section byte-order magic.
_NG_SHB = 0x0A0D0D0A
_NG_IDB = 1
_NG_SPB = 3
_NG_EPB = 6
_NG_BOM = 0x1A2B3C4D
_NG_MIN_BLOCK = 12
_NG_EPB_HDR = 20  # iface, ts hi, ts lo, cap len, orig len
_NG_SPB_HDR = 4  # orig len

_LINK_HCI_H4_PHDR = 201
_LINK_RADIOTAP = 127
_HCI_PHDR_LEN = 4
_RADIOTAP_MIN = 4  # version, pad, le16 length


class Framing(Enum):
    """Length prefix written before each message (struct format)."""

    U32BE = ">I"
    U32LE = "<I"
    U16BE = ">H"
    U16LE = "<H"


def _strip(link: int, pkt: bytes) -> bytes | None:
    """Drop the link pseudo-header; None when it does not fit the packet."""
    if link == _LINK_HCI_H4_PHDR:
        return pkt[_HCI_PHDR_LEN:] if len(pkt) >= _HCI_PHDR_LEN else None

    if link == _LINK_RADIOTAP:
        if len(pkt) < _RADIOTAP_MIN:
            return None
        (n,) = struct.unpack_from("<H", pkt, 2)
        return pkt[n:] if _RADIOTAP_MIN <= n <= len(pkt) else None

    return pkt


def _pcap_endian(buf) -> str | None:
    """Byte order of a classic pcap header, or None."""
    if len(buf) < _PCAP_HDR:
        return None
    for endian in ("<", ">"):
        if struct.unpack_from(endian + "I", buf, 0)[0] in _PCAP_MAGICS:
            return endian
    return None


def _read_pcap(buf, endian: str) -> Iterator[bytes]:
    """Records of a classic pcap; a record past the end stops the walk."""
    (link,) = struct.unpack_from(endian + "I", buf, _PCAP_LINK_OFF)
    rec = struct.Struct(endian + "8xII")
    pos, n = _PCAP_HDR, len(buf)
    while pos + _PCAP_REC <= n:
        incl, _orig = rec.unpack_from(buf, pos)
        start = pos + _PCAP_REC
        end = start + incl
        if end > n:
            return
        pos = end

        pkt = _strip(link, bytes(buf[start:end]))
        if pkt:
            yield pkt


def _ng_endian(buf, pos: int) -> str | None:
    """Byte order of the section header block at *pos*, or None."""
    if pos + _NG_MIN_BLOCK > len(buf):
        return None
    for endian in ("<", ">"):
        if struct.unpack_from(endian + "I", buf, pos + 8)[0] == _NG_BOM:
            return endian
    return None


def _ng_packet(buf, btype: int, body: int, end: int, endian: str) -> tuple[int, bytes] | None:
    """``(interface, bytes)`` of an EPB/SPB body, None when malformed."""
    if btype == _NG_SPB:
        if body + _NG_SPB_HDR > end:
            return None
        (orig,) = struct.unpack_from(endian + "I", buf, body)
        data = body + _NG_SPB_HDR
        return 0, bytes(buf[data : data + min(orig, end - data)])

    if body + _NG_EPB_HDR > end:
        return None
    iface, _hi, _lo, cap, _orig = struct.unpack_from(endian + "IIIII", buf, body)
    data = body + _NG_EPB_HDR
    if data + cap > end:
        return None
    return iface, bytes(buf[data : data + cap])


def _read_pcapng(buf) -> Iterator[bytes]:
    """Packets of every section; a malformed block stops the walk."""
    pos, n = 0, len(buf)
    endian = "<"
    links: list[int] = []
    while pos + _NG_MIN_BLOCK <= n:
        btype = struct.unpack_from(endian + "I", buf, pos)[0]

        # Each section header resets byte order and interfaces.
        if btype == _NG_SHB:
            endian = _ng_endian(buf, pos) or ""
            if not endian:
                return
            links = []

        (total,) = struct.unpack_from(endian + "I", buf, pos + 4)
        if total < _NG_MIN_BLOCK or total % 4 or pos + total > n:
            return
        body, end = pos + 8, pos + total - 4
        pos += total

        if btype == _NG_IDB and body + 2 <= end:
            links.append(struct.unpack_from(endian + "H", buf, body)[0])
            continue
        if btype not in (_NG_EPB, _NG_SPB):
            continue

        got = _ng_packet(buf, btype, body, end, endian)
        if got is None:
            return
        iface, raw = got
        if iface >= len(links):
            continue
        pkt = _strip(links[iface], raw)
        if pkt:
            yield pkt


def read_packets(buf) -> Iterator[bytes]:
    """Non-empty packet payloads of a pcap/pcapng capture; nothing otherwise."""
    endian = _pcap_endian(buf)
    if endian:
        return _read_pcap(buf, endian)
    if len(buf) >= 4 and struct.unpack_from("<I", buf, 0)[0] == _NG_SHB:
        return _read_pcapng(buf)
    return iter(())


def frame(msgs: list[bytes], framing: Framing) -> bytes:
    """Length-prefix and join *msgs*; a message too long for the prefix is dropped.

    Example: ``frame([b"ab", b"c"], Framing.U16BE) == b"\\0\\2ab\\0\\1c"``.
    """
    pack = struct.Struct(framing.value)
    cap = 1 << (8 * pack.size)
    return b"".join(pack.pack(len(m)) + m for m in msgs if len(m) < cap)
