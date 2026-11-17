"""core/pcap.py: packet captures (pcap, pcapng) → message lists for seeds.

Captures are built here byte by byte from the format specs, independent of
the parser under test.
"""

import struct

from fuzzer_tool.core.mutations.seq_mutate import split_msgs
from fuzzer_tool.core.pcap import Framing, frame, read_packets

_ETHERNET = 1
_HCI_H4_PHDR = 201
_RADIOTAP = 127
_PCAP_US = 0xA1B2C3D4
_PCAP_NS = 0xA1B23C4D


def _pcap(packets, link=_ETHERNET, endian="<", magic=_PCAP_US, snaplen=65535) -> bytes:
    """Classic pcap: 24-byte global header, 16-byte record headers."""
    out = struct.pack(endian + "IHHiIII", magic, 2, 4, 0, 0, snaplen, link)
    for p in packets:
        out += struct.pack(endian + "IIII", 0, 0, len(p), len(p)) + p
    return out


def _pad4(b: bytes) -> bytes:
    return b + b"\x00" * (-len(b) % 4)


def _block(btype: int, body: bytes, endian="<") -> bytes:
    """pcapng block: type, total length, padded body, total length."""
    body = _pad4(body)
    total = 12 + len(body)
    return struct.pack(endian + "II", btype, total) + body + struct.pack(endian + "I", total)


def _shb(endian="<") -> bytes:
    return _block(0x0A0D0D0A, struct.pack(endian + "IHHq", 0x1A2B3C4D, 1, 0, -1), endian)


def _idb(link: int, endian="<") -> bytes:
    return _block(1, struct.pack(endian + "HHI", link, 0, 0), endian)


def _epb(payload: bytes, iface=0, endian="<") -> bytes:
    hdr = struct.pack(endian + "IIIII", iface, 0, 0, len(payload), len(payload))
    return _block(6, hdr + payload, endian)


def _spb(payload: bytes, endian="<") -> bytes:
    return _block(3, struct.pack(endian + "I", len(payload)) + payload, endian)


class TestClassicPcap:
    def test_little_endian_us(self):
        assert list(read_packets(_pcap([b"abc", b"defg"]))) == [b"abc", b"defg"]

    def test_big_endian_ns(self):
        blob = _pcap([b"xy", b"z"], endian=">", magic=_PCAP_NS)
        assert list(read_packets(blob)) == [b"xy", b"z"]

    def test_hci_phdr_stripped(self):
        # LINKTYPE 201 carries a 4-byte direction pseudo-header.
        blob = _pcap([b"\x00\x00\x00\x01" + b"\x01\x03\x0c\x00"], link=_HCI_H4_PHDR)
        assert list(read_packets(blob)) == [b"\x01\x03\x0c\x00"]

    def test_radiotap_stripped(self):
        # Radiotap: version, pad, le16 length (header included), present flags.
        rt = struct.pack("<BBHI", 0, 0, 8, 0)
        blob = _pcap([rt + b"\x80\x00frame"], link=_RADIOTAP)
        assert list(read_packets(blob)) == [b"\x80\x00frame"]

    def test_truncated_record_stops(self):
        # Adversarial: the last record claims more bytes than the file holds.
        blob = _pcap([b"ok", b"cut-short"])[:-3]
        assert list(read_packets(blob)) == [b"ok"]

    def test_huge_incl_len_stops(self):
        blob = _pcap([b"ok"]) + struct.pack("<IIII", 0, 0, 0xFFFFFFFF, 1) + b"x"
        assert list(read_packets(blob)) == [b"ok"]

    def test_oversized_radiotap_skipped(self):
        # Adversarial: radiotap length past the packet end drops the packet only.
        bad = struct.pack("<BBHI", 0, 0, 999, 0)
        good = struct.pack("<BBHI", 0, 0, 8, 0) + b"F"
        assert list(read_packets(_pcap([bad, good], link=_RADIOTAP))) == [b"F"]

    def test_not_a_capture(self):
        assert list(read_packets(b"\x89PNG\r\n\x1a\n" + bytes(32))) == []
        assert list(read_packets(b"")) == []


class TestPcapng:
    def test_epb_and_spb(self):
        blob = _shb() + _idb(_ETHERNET) + _epb(b"one") + _spb(b"two!")
        assert list(read_packets(blob)) == [b"one", b"two!"]

    def test_big_endian_section(self):
        blob = _shb(">") + _idb(_ETHERNET, ">") + _epb(b"be", endian=">")
        assert list(read_packets(blob)) == [b"be"]

    def test_per_interface_link_type(self):
        blob = (
            _shb()
            + _idb(_ETHERNET)
            + _idb(_HCI_H4_PHDR)
            + _epb(b"eth", 0)
            + _epb(b"\x00\x00\x00\x00hci", 1)
        )
        assert list(read_packets(blob)) == [b"eth", b"hci"]

    def test_unknown_interface_skipped(self):
        blob = _shb() + _idb(_ETHERNET) + _epb(b"ghost", iface=7) + _epb(b"real")
        assert list(read_packets(blob)) == [b"real"]

    def test_bad_block_length_stops(self):
        # Adversarial: total length below the 12-byte minimum ends the walk.
        bad = struct.pack("<II", 6, 4)
        blob = _shb() + _idb(_ETHERNET) + _epb(b"ok") + bad + _epb(b"never")
        assert list(read_packets(blob)) == [b"ok"]

    def test_cap_len_past_block_stops(self):
        hdr = struct.pack("<IIIII", 0, 0, 0, 500, 500)
        blob = _shb() + _idb(_ETHERNET) + _epb(b"ok") + _block(6, hdr + b"short")
        assert list(read_packets(blob)) == [b"ok"]


class TestFrame:
    def test_round_trip_split_msgs(self):
        # Falsification: seq_mutate recovers exactly the packets as messages.
        pkts = [b"alpha", b"be", b"gamma!"]
        msgs = split_msgs(frame(pkts, Framing.U32BE))
        assert [m[4:] for m in msgs] == pkts

    def test_u16le_layout(self):
        assert frame([b"ab", b"c"], Framing.U16LE) == b"\x02\x00ab\x01\x00c"

    def test_oversize_dropped_for_width(self):
        # Adversarial: a 64 KiB packet cannot carry a u16 length.
        assert frame([b"x" * 0x10000, b"ok"], Framing.U16BE) == b"\x00\x02ok"
