"""Tests for the TCP/IP packet mutator (``tcpip``)."""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.tcpip import (
    BYTE_EDGES,
    DOFF_EDGES,
    FLOW_EDGES,
    FRAG_EDGES,
    ICMP4_TYPES,
    ICMP6_TYPES,
    IHL_EDGES,
    IP_OPTIONS,
    MAX_PACKETS,
    PORT_EDGES,
    PROTO_EDGES,
    QUOTE_CUTS,
    TCP_FLAG_EDGES,
    TCP_OPTIONS,
    TOS_EDGES,
    TTL_EDGES,
    U16_EDGES,
    U32_EDGES,
    V6_EXT_HDRS,
    V6_NH_EDGES,
    WINDOW_EDGES,
    Op,
    Site,
    inet_checksum,
    locate_packets,
    mutate_tcpip,
    packet_ops,
    sniff_tcpip,
)
from fuzzer_tool.core.rand_pool import RandPool

from .support import tcpip_build as tb
from .support.scripted_rng import ScriptedRng

IP_OPT = bytes([1, 1, 1, 0])  # NOP NOP NOP EOL
TCP_OPT = bytes([2, 4, 0x05, 0xB4, 1, 1, 1, 0])  # MSS 1460, NOPs, EOL

V4TCP = tb.ipv4(tb.TCP, tb.tcp_seg(b"hello", TCP_OPT), options=IP_OPT)  # l4 24, opts 44, payload 52
V4UDP = tb.ipv4(tb.UDP, tb.udp_dgram(b"query"))
V4ICMP = tb.ipv4(tb.ICMP, tb.icmp_msg(payload=bytes(range(40))))
V6TCP = tb.ipv6(tb.TCP, tb.tcp_seg(b"hello", TCP_OPT))
V6UDP = tb.ipv6(tb.UDP, tb.udp_dgram(b"query"))
V6ICMP = tb.ipv6(tb.ICMP6, tb.icmp_msg(128, 0, bytes(range(48))))
V6HBH = tb.ipv6(tb.TCP, tb.tcp_seg(b"hi"), ext=tb.hbh(tb.TCP))

FIXTURES = [V4TCP, V4UDP, V4ICMP, V6TCP, V6UDP, V6ICMP, V6HBH]


def _oracle(data: bytes) -> int:
    """RFC 1071 by hand: fold the 16-bit big-endian word sum, complement."""
    padded = data + b"\x00" * (len(data) % 2)
    total = 0
    for i in range(0, len(padded), 2):
        total += padded[i] * 256 + padded[i + 1]
    while total > 0xFFFF:
        total = (total & 0xFFFF) + (total >> 16)
    return 0xFFFF - total


def _ip_ok(pkt: bytes, hdr: int | None = None) -> bool:
    hdr = (pkt[0] & 0xF) * 4 if hdr is None else hdr
    return _oracle(pkt[:hdr]) == 0


def _l4_ok(pkt: bytes, l4: int, end: int, proto: int) -> bool:
    seg = pkt[l4:end]
    if proto == tb.ICMP:
        return _oracle(seg) == 0
    if pkt[0] >> 4 == 4:
        pseudo = pkt[12:20] + struct.pack(">BBH", 0, proto, len(seg))
    else:
        pseudo = pkt[8:40] + struct.pack(">IxxxB", len(seg), proto)
    return _oracle(pseudo + seg) == 0


def _u16(b: bytes, off: int) -> int:
    return struct.unpack_from(">H", b, off)[0]


def _rng(op: Op, pkt: bytes, randints=(), choices=(), idx: int = 0) -> ScriptedRng:
    """Scripted draws: packet index, op (index into packet_ops), op draws."""
    ops = packet_ops(pkt)
    return ScriptedRng(randints=[idx, *randints], choice_idxs=[ops.index(op), *choices])


def _mut(pkt: bytes, op: Op, randints=(), choices=()) -> bytes:
    out = mutate_tcpip(pkt, rng=_rng(op, pkt, randints, choices))
    assert out is not None
    assert len(out) == len(pkt)
    return out


class TestChecksum:
    # Control: the oracle reproduces the RFC 1071 section 3 worked example.
    def test_oracle_rfc_example(self):
        assert _oracle(bytes.fromhex("0001f203f4f5f6f7")) == 0xFFFF - 0xDDF2

    # Control + falsification: fixtures verify, a planted bad checksum does not.
    @pytest.mark.parametrize("pkt", [V4TCP, V4UDP, V4ICMP])
    def test_oracle_rejects_planted_bad(self, pkt):
        assert _ip_ok(pkt)
        bad = bytearray(pkt)
        bad[11] ^= 1
        assert not _ip_ok(bytes(bad))

    def test_fixture_l4_verifies(self):
        assert _l4_ok(V4TCP, 24, len(V4TCP), tb.TCP)
        assert _l4_ok(V6UDP, 40, len(V6UDP), tb.UDP)
        assert _l4_ok(V6HBH, 48, len(V6HBH), tb.TCP)
        assert not _l4_ok(V4TCP, 24, len(V4TCP) - 1, tb.TCP)

    def test_matches_oracle_all_lengths(self):
        rng = RandPool(seed=11)
        for n in range(0, 1501, 7):
            blob = rng.randbytes(n)
            assert inet_checksum(blob) == _oracle(blob)

    @pytest.mark.parametrize("blob", [b"", bytes(20), b"\xff" * 20, b"\xff\xff", b"\x01"])
    def test_edge_buffers(self, blob):
        assert inet_checksum(blob) == _oracle(blob)


class TestLocate:
    def test_raw(self):
        assert locate_packets(V4TCP) == [(0, len(V4TCP))]
        assert locate_packets(V6TCP) == [(0, len(V6TCP))]

    def test_ether_and_vlan(self):
        f = tb.ether(V4UDP)
        assert locate_packets(f) == [(14, len(f))]
        f = tb.ether(V6TCP, tb.Tag.VLAN)
        assert locate_packets(f) == [(18, len(f))]

    # A dst MAC starting with 0x45 must not be read as raw IPv4.
    def test_ether_wins_over_raw_lookalike(self):
        f = tb.ether(V4UDP, dst=bytes.fromhex("450000000001"))
        assert locate_packets(f) == [(14, len(f))]

    @pytest.mark.parametrize("variant", range(4))
    def test_pcap_variants(self, variant):
        data = tb.pcap([tb.ether(V4TCP), tb.ether(V6UDP)], variant=variant)
        first = 24 + 16 + 14
        second = first + len(V4TCP) + 16 + 14
        assert locate_packets(data) == [(first, first + len(V4TCP)), (second, second + len(V6UDP))]

    @pytest.mark.parametrize(("link", "pkt", "found"), [(101, V6TCP, 1), (228, V4UDP, 1), (229, V6UDP, 1), (228, V6UDP, 0)])
    def test_pcap_linktypes(self, link, pkt, found):
        assert len(locate_packets(tb.pcap([pkt], linktype=link))) == found

    # Adversarial: incl_len past EOF ends the walk.
    def test_pcap_incl_len_past_eof(self):
        data = bytearray(tb.pcap([V4UDP, V4UDP], linktype=101))
        second = 24 + 16 + len(V4UDP)
        struct.pack_into("<I", data, second + 8, 0xFFFFFFFF)
        assert locate_packets(bytes(data)) == [(40, 40 + len(V4UDP))]

    # Adversarial: packet count stays bounded.
    def test_pcap_bounded(self):
        data = tb.pcap([V4UDP] * (MAX_PACKETS + 50), linktype=101)
        assert len(locate_packets(data)) == MAX_PACKETS


class TestSniff:
    @pytest.mark.parametrize(
        "data",
        [V4TCP, V6TCP, V6HBH, tb.ether(V4UDP), tb.ether(V6UDP, tb.Tag.VLAN), tb.pcap([V4UDP], linktype=228)],
    )
    def test_true(self, data):
        assert sniff_tcpip(data)

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            bytes(64),
            b"Hello, world! " * 8,
            b"lowercase text that is long enough to be a header " * 4,
            V4TCP[:19],
            tb.pcap([V4UDP], linktype=105),
            b"\x4f" + bytes(19),  # IHL 60 on a 20-byte buffer
        ],
    )
    def test_false(self, data):
        assert not sniff_tcpip(data)


class TestIPv4:
    def test_ihl_shrink_relocates_l4(self):
        out = _mut(V4TCP, Op.V4_IHL, choices=[IHL_EDGES.index(5)])
        assert out[0] == 0x45
        assert _ip_ok(out)
        assert _l4_ok(out, 20, len(out), tb.TCP)

    def test_ihl_past_buffer_repairs_20_bytes(self):
        out = _mut(V4TCP, Op.V4_IHL, choices=[IHL_EDGES.index(15)])
        assert out[0] & 0xF == 15
        assert _ip_ok(out, 20)
        assert out[24:] == V4TCP[24:]

    def test_tos(self):
        out = _mut(V4TCP, Op.V4_TOS, choices=[2])
        assert out[1] == TOS_EDGES[2]
        assert _ip_ok(out)

    # Collision: the drawn edge equals the current value, the next edge is used.
    def test_pick_skips_current_value(self):
        assert V4TCP[1] == TOS_EDGES[0]
        out = _mut(V4TCP, Op.V4_TOS, choices=[0])
        assert out[1] == TOS_EDGES[1]

    def test_totlen_short_reclamps_l4(self):
        n = len(V4TCP)
        out = _mut(V4TCP, Op.V4_TOTLEN, choices=[2])  # n - 1
        assert _u16(out, 2) == n - 1
        assert _ip_ok(out)
        assert _l4_ok(out, 24, n - 1, tb.TCP)

    def test_totlen_over_buffer(self):
        n = len(V4TCP)
        out = _mut(V4TCP, Op.V4_TOTLEN, choices=[3])  # n + 1
        assert _u16(out, 2) == n + 1
        assert _ip_ok(out)
        assert _l4_ok(out, 24, n, tb.TCP)

    def test_id(self):
        out = _mut(V4TCP, Op.V4_ID, choices=[4])
        assert _u16(out, 4) == U16_EDGES[4]
        assert _ip_ok(out)

    def test_frag(self):
        out = _mut(V4TCP, Op.V4_FRAG, choices=[FRAG_EDGES.index(0x3FFF)])
        assert _u16(out, 6) == 0x3FFF
        assert _ip_ok(out)

    def test_ttl(self):
        out = _mut(V4UDP, Op.V4_TTL, choices=[0])
        assert out[8] == TTL_EDGES[0]
        assert _ip_ok(out)
        assert out[20:] == V4UDP[20:]

    def test_proto_swap_repairs_new_l4(self):
        out = _mut(V4TCP, Op.V4_PROTO, choices=[PROTO_EDGES.index(tb.UDP)])
        assert out[9] == tb.UDP
        assert _ip_ok(out)
        assert _l4_ok(out, 24, len(out), tb.UDP)

    def test_proto_unknown_leaves_l4(self):
        out = _mut(V4TCP, Op.V4_PROTO, choices=[PROTO_EDGES.index(255)])
        assert out[24:] == V4TCP[24:]
        assert _ip_ok(out)

    def test_option(self):
        k = 2  # LSRR len overrun
        out = _mut(V4TCP, Op.V4_OPTION, choices=[k])
        assert out[20:24] == IP_OPTIONS[k][:3] + V4TCP[23:24]
        assert _ip_ok(out)

    def test_option_absent_without_space(self):
        assert Op.V4_OPTION not in packet_ops(V4UDP)


class TestIPv6:
    def test_plen_short(self):
        n = len(V6TCP)
        out = _mut(V6TCP, Op.V6_PLEN, choices=[2])  # present - 1
        assert _u16(out, 4) == n - 40 - 1
        assert _l4_ok(out, 40, n - 1, tb.TCP)

    def test_plen_zero_uses_buffer(self):
        out = _mut(V6TCP, Op.V6_PLEN, choices=[0])
        assert _u16(out, 4) == 0
        assert _l4_ok(out, 40, len(out), tb.TCP)

    def test_nh_swap(self):
        out = _mut(V6TCP, Op.V6_NH, choices=[V6_NH_EDGES.index(tb.UDP)])
        assert out[6] == tb.UDP
        assert _l4_ok(out, 40, len(out), tb.UDP)

    def test_hoplimit(self):
        out = _mut(V6UDP, Op.V6_HOPLIM, choices=[1])
        assert out[7] == TTL_EDGES[1]
        assert out[8:] == V6UDP[8:]

    def test_flow_keeps_version(self):
        out = _mut(V6UDP, Op.V6_FLOW, choices=[3])
        word = struct.unpack_from(">I", out, 0)[0]
        assert word >> 28 == 6
        assert word & 0xFFFFF == FLOW_EDGES[3]

    def test_ext_valid_hbh_chains_to_l4(self):
        k = 0  # well-formed hop-by-hop
        typ, inner, tail = V6_EXT_HDRS[k]
        out = _mut(V6TCP, Op.V6_EXT, choices=[k])
        assert out[6] == typ
        assert out[40] == tb.TCP
        assert out[41:48] == tail
        assert _l4_ok(out, 48, len(out), tb.TCP)

    def test_ext_overrun_skips_l4(self):
        k = next(i for i, t in enumerate(V6_EXT_HDRS) if t[2][0] == 0xFF)
        out = _mut(V6TCP, Op.V6_EXT, choices=[k])
        assert out[41] == 0xFF
        assert out[48:] == V6TCP[48:]

    def test_existing_chain_is_walked(self):
        out = _mut(V6HBH, Op.V6_HOPLIM, choices=[0])
        assert _l4_ok(out, 48, len(out), tb.TCP)


class TestTCP:
    def test_doff(self):
        out = _mut(V4TCP, Op.TCP_DOFF, choices=[DOFF_EDGES.index(15)])
        assert out[24 + 12] >> 4 == 15
        assert _l4_ok(out, 24, len(out), tb.TCP)

    def test_xmas_sets_urg_ptr_past_data(self):
        out = _mut(V4TCP, Op.TCP_FLAGS, choices=[TCP_FLAG_EDGES.index(0x29)])
        assert out[24 + 13] == 0x29
        assert _u16(out, 24 + 18) == len(b"hello") + 1
        assert _l4_ok(out, 24, len(out), tb.TCP)

    def test_ack(self):
        out = _mut(V6TCP, Op.TCP_SEQACK, randints=[1], choices=[4])
        assert struct.unpack_from(">I", out, 40 + 8)[0] == U32_EDGES[4]
        assert struct.unpack_from(">I", out, 40 + 4)[0] == 1000
        assert _l4_ok(out, 40, len(out), tb.TCP)

    def test_window(self):
        out = _mut(V4TCP, Op.TCP_WINDOW, choices=[0])
        assert _u16(out, 24 + 14) == WINDOW_EDGES[0]
        assert _l4_ok(out, 24, len(out), tb.TCP)

    def test_option(self):
        k = 6  # SACK with a length that is not 2 + 8k
        out = _mut(V4TCP, Op.TCP_OPTION, choices=[k])
        tmpl = TCP_OPTIONS[k][:8]
        assert out[44 : 44 + len(tmpl)] == tmpl
        assert _l4_ok(out, 24, len(out), tb.TCP)

    def test_payload(self):
        out = _mut(V4TCP, Op.PAYLOAD, randints=[2], choices=[0])
        assert out[52 + 2] == BYTE_EDGES[0]
        assert _l4_ok(out, 24, len(out), tb.TCP)


class TestUDP:
    def test_len(self):
        out = _mut(V4UDP, Op.UDP_LEN, choices=[1])  # 7
        assert _u16(out, 24) == 7
        assert _l4_ok(out, 20, len(out), tb.UDP)

    def test_port(self):
        out = _mut(V6UDP, Op.UDP_PORT, randints=[0], choices=[PORT_EDGES.index(4789)])
        assert _u16(out, 40) == 4789
        assert _l4_ok(out, 40, len(out), tb.UDP)

    def test_zero_checksum(self):
        out = _mut(V4UDP, Op.ZERO_UDP_CSUM)
        assert _u16(out, 26) == 0
        assert _ip_ok(out)

    def test_zero_absent_when_already_zero(self):
        pkt = bytearray(V4UDP)
        pkt[26:28] = b"\x00\x00"
        assert Op.ZERO_UDP_CSUM not in packet_ops(bytes(pkt))

    # A computed UDP checksum of 0 is stored as 0xFFFF (0 means "none").
    def test_computed_zero_stored_as_ffff(self):
        pkt = bytearray(tb.ipv4(tb.UDP, tb.udp_dgram(b"ab\x00\x00")))
        pkt[26:28] = b"\x00\x00"
        filler = _oracle(pkt[12:20] + struct.pack(">BBH", 0, tb.UDP, 12) + pkt[20:])
        pkt[30:32] = struct.pack(">H", filler)
        out = _mut(bytes(pkt), Op.V4_TTL, choices=[0])
        assert _u16(out, 26) == 0xFFFF
        assert _l4_ok(out, 20, len(out), tb.UDP)


class TestICMP:
    def test_type_v4(self):
        out = _mut(V4ICMP, Op.ICMP_TYPE, choices=[3])
        assert tuple(out[20:22]) == ICMP4_TYPES[3]
        assert _l4_ok(out, 20, len(out), tb.ICMP)

    def test_type_v6(self):
        out = _mut(V6ICMP, Op.ICMP_TYPE, choices=[1])
        assert tuple(out[40:42]) == ICMP6_TYPES[1]
        assert _l4_ok(out, 40, len(out), tb.ICMP6)

    def test_quote_v4_truncated(self):
        cut = QUOTE_CUTS.index(28)
        out = _mut(V4ICMP, Op.ICMP_QUOTE, choices=[cut])
        assert out[20] == 3
        assert out[28:48] == V4ICMP[:20]
        assert _u16(out, 2) == 20 + 8 + 28
        assert _ip_ok(out)
        assert _l4_ok(out, 20, 56, tb.ICMP)

    def test_quote_v6(self):
        out = _mut(V6ICMP, Op.ICMP_QUOTE, choices=[QUOTE_CUTS.index(39)])
        assert out[40] == 1
        assert out[48:88] == V6ICMP[:40]
        assert _u16(out, 4) == 8 + 39
        assert _l4_ok(out, 40, 40 + 8 + 39, tb.ICMP6)


class TestBreak:
    # Falsification: exactly the chosen checksum fails; flipping it back verifies.
    def test_break_ip(self):
        sites = (Site.IP, Site.L4)
        out = _mut(V4TCP, Op.BREAK_CSUM, choices=[sites.index(Site.IP)])
        assert not _ip_ok(out)
        assert _l4_ok(out, 24, len(out), tb.TCP)
        fixed = bytearray(out)
        fixed[11] ^= 1
        assert _ip_ok(bytes(fixed))

    def test_break_l4(self):
        out = _mut(V4TCP, Op.BREAK_CSUM, choices=[1])
        assert _ip_ok(out)
        assert not _l4_ok(out, 24, len(out), tb.TCP)
        fixed = bytearray(out)
        fixed[24 + 17] ^= 1
        assert _l4_ok(bytes(fixed), 24, len(out), tb.TCP)

    def test_break_v6_only_l4(self):
        out = _mut(V6UDP, Op.BREAK_CSUM, choices=[0])
        assert not _l4_ok(out, 40, len(out), tb.UDP)


class TestContainers:
    def test_pcap_record_edit_in_place(self):
        data = tb.pcap([tb.ether(V4UDP), tb.ether(V4TCP), tb.ether(V6UDP)])
        spans = locate_packets(data)
        s, e = spans[1]
        out = mutate_tcpip(data, rng=_rng(Op.V4_TTL, data[s:e], choices=[1], idx=1))
        assert len(out) == len(data)
        assert out[:s] == data[:s] and out[e:] == data[e:]
        assert out[s + 8] == TTL_EDGES[1]
        assert _ip_ok(out[s:e])

    def test_ether_edit(self):
        f = tb.ether(V4UDP, tb.Tag.VLAN)
        out = mutate_tcpip(f, rng=_rng(Op.UDP_PORT, V4UDP, randints=[1], choices=[1]))
        assert out[:18] == f[:18]
        assert _u16(out, 18 + 22) == PORT_EDGES[1]
        assert _l4_ok(out[18:], 20, len(V4UDP), tb.UDP)


class TestContract:
    def test_declines_non_ip(self):
        assert mutate_tcpip(b"hello" * 20, rng=ScriptedRng()) is None

    def test_pcap_without_packets_declines(self):
        assert mutate_tcpip(tb.pcap([]), rng=ScriptedRng()) is None

    def test_max_len(self):
        out = mutate_tcpip(V4TCP, max_len=10, rng=_rng(Op.V4_TTL, V4TCP, choices=[0]))
        assert out == _mut(V4TCP, Op.V4_TTL, choices=[0])[:10]

    @pytest.mark.parametrize("pkt", FIXTURES)
    def test_every_op_changes_input(self, pkt):
        ops = packet_ops(pkt)
        assert Op.BREAK_CSUM in ops
        for i, _op in enumerate(ops):
            out = mutate_tcpip(pkt, rng=ScriptedRng(randints=[0, 0, 0], choice_idxs=[i, 0, 0]))
            assert out is not None and out != pkt

    def test_every_op_covered(self):
        seen = set()
        for pkt in FIXTURES:
            seen.update(packet_ops(pkt))
        assert seen == set(Op)

    # Adversarial: hostile containers/headers never raise and stay bounded.
    def test_hostile_inputs(self):
        loop = bytearray(tb.ipv6(tb.TCP, tb.tcp_seg(b"x" * 64), ext=tb.hbh(0) * 6))
        hostile = [bytes(loop), b"\x4f" + bytes(59), b"\x45" + bytes(19), b"\x60" + bytes(39)]
        for pkt in FIXTURES:
            hostile += [pkt[:cut] for cut in range(len(pkt) + 1)]
            hostile += [tb.ether(pkt)[:cut] for cut in range(0, len(pkt) + 14, 5)]
        cap = bytearray(tb.pcap([tb.ether(V4TCP)] * 3))
        hostile += [bytes(cap[:cut]) for cut in range(0, len(cap), 3)]
        gen = RandPool(seed=5)
        hostile += [bytes([0x45]) + gen.randbytes(gen.randint(19, 200)) for _ in range(200)]
        hostile += [bytes([0x60]) + gen.randbytes(gen.randint(39, 200)) for _ in range(200)]
        hostile += [bytes(cap[:24]) + gen.randbytes(gen.randint(0, 300)) for _ in range(200)]

        rng = RandPool(seed=9)
        for data in hostile:
            sniff_tcpip(data)
            for _ in range(4):
                out = mutate_tcpip(data, max_len=64, rng=rng)
                assert out is None or (len(out) <= 64 and out != data[:64] or len(data) > 64)
