"""Tests for the Intel PT packet decoder and the PT coverage map.

The packet encodings are taken from the Intel SDM (Vol 3C, "Intel Processor
Trace") and cross-checked against the two reference decoders: the kernel's
``tools/perf/util/intel-pt-decoder/intel-pt-pkt-decoder.c`` and libipt.  Every
stream here is hand-built, so the whole decoder is exercised without any
``intel_pt`` PMU on the host.
"""

import random

import pytest

from fuzzer_tool.core.intel_pt import (
    PSB,
    PtCoverage,
    PtIpDecoder,
    PtMapMode,
    PtType,
    calc_ip,
    iter_packets,
)

# ── stream builders ────────────────────────────────────────────────────

PAD = b"\x00"


def ip_pkt(opcode: int, ipc: int, payload: bytes = b"") -> bytes:
    """An IP packet: opcode in bits[4:0], IPBytes in bits[7:5]."""
    return bytes([(ipc << 5) | opcode]) + payload


def tip(ipc: int, payload: bytes = b"") -> bytes:
    return ip_pkt(0x0D, ipc, payload)


def fup(ipc: int, payload: bytes = b"") -> bytes:
    return ip_pkt(0x1D, ipc, payload)


def pge(ipc: int, payload: bytes = b"") -> bytes:
    return ip_pkt(0x11, ipc, payload)


def pgd(ipc: int, payload: bytes = b"") -> bytes:
    return ip_pkt(0x01, ipc, payload)


def tip6(ip: int) -> bytes:
    """TIP with a full 8-byte IP (IPBytes=6)."""
    return tip(6, ip.to_bytes(8, "little"))


def types(buf: bytes) -> list[PtType]:
    return [p.type for p in iter_packets(buf)]


# ── packet framing ─────────────────────────────────────────────────────


def test_pad_is_one_byte():
    pkts = list(iter_packets(PAD * 3))
    assert [p.type for p in pkts] == [PtType.PAD] * 3
    assert [p.size for p in pkts] == [1, 1, 1]


def test_psb_is_sixteen_bytes():
    pkts = list(iter_packets(PSB))
    assert len(pkts) == 1
    assert pkts[0].type is PtType.PSB
    assert pkts[0].size == 16


def test_malformed_psb_is_bad():
    broken = bytearray(PSB)
    broken[8] = 0x03
    assert list(iter_packets(bytes(broken)))[0].type is PtType.BAD


@pytest.mark.parametrize(
    ("ipc", "size"),
    [(0, 1), (1, 3), (2, 5), (3, 7), (4, 7), (6, 9)],
)
def test_ip_packet_sizes(ipc, size):
    buf = tip(ipc, b"\xaa" * (size - 1))
    pkt = next(iter(iter_packets(buf)))
    assert pkt.type is PtType.TIP
    assert pkt.size == size
    assert pkt.count == ipc


@pytest.mark.parametrize("ipc", [5, 7])
def test_reserved_ip_compression_is_bad(ipc):
    assert list(iter_packets(tip(ipc, b"\xaa" * 8)))[0].type is PtType.BAD


def test_ip_packet_opcodes():
    buf = tip6(0x1000) + fup(1, b"\x02\x00") + pge(6, (0).to_bytes(8, "little")) + pgd(0)
    assert types(buf) == [PtType.TIP, PtType.FUP, PtType.TIP_PGE, PtType.TIP_PGD]


def test_short_tnt_count_is_bits_below_stop_bit():
    # Bit 0 is the opcode; the highest set bit is the stop bit and every bit
    # between them is one branch result.
    # 0b1000_0000 -> stop bit at 7, six results.
    # 0b0100_0110 -> stop bit at 6, five results.
    assert list(iter_packets(b"\x80"))[0].count == 6
    pkt = list(iter_packets(b"\x46"))[0]
    assert pkt.type is PtType.TNT
    assert pkt.count == 5
    assert pkt.size == 1


def test_long_tnt_count_and_size():
    # Stop bit in the top payload byte -> 47 branch results is the maximum.
    payload = bytes([0x02, 0xA3]) + b"\x00" * 5 + b"\x80"
    pkt = list(iter_packets(payload))[0]
    assert pkt.type is PtType.TNT
    assert pkt.size == 8
    assert pkt.count == 47


@pytest.mark.parametrize(
    ("buf", "size"),
    [
        (b"\x03", 1),  # CYC, no extension
        (b"\x07\x02", 2),  # CYC, one extension byte
        (b"\x07\x03\x02", 3),  # CYC, two extension bytes
    ],
)
def test_cyc_is_variable_length(buf, size):
    pkt = list(iter_packets(buf))[0]
    assert pkt.type is PtType.CYC
    assert pkt.size == size


@pytest.mark.parametrize(
    ("raw", "expected", "size"),
    [
        (b"\x02\x83", PtType.TRACESTOP, 2),
        (b"\x02\xf3", PtType.OVF, 2),
        (b"\x02\x23", PtType.PSBEND, 2),
        (b"\x02\x62", PtType.EXSTOP, 2),
        (b"\x02\xe2", PtType.EXSTOP, 2),
        (b"\x02\x33", PtType.BEP, 2),
        (b"\x02\xb3", PtType.BEP, 2),
        (b"\x02\x43" + b"\x00" * 6, PtType.PIP, 8),
        (b"\x02\x03\x11\x00", PtType.CBR, 4),
        (b"\x02\xc8" + b"\x00" * 5, PtType.VMCS, 7),
        (b"\x02\x73" + b"\x00" * 5, PtType.TMA, 7),
        (b"\x02\xc3\x88" + b"\x00" * 8, PtType.MNT, 11),
        (b"\x02\xc2" + b"\x00" * 8, PtType.MWAIT, 10),
        (b"\x02\x22\x00\x00", PtType.PWRE, 4),
        (b"\x02\xa2" + b"\x00" * 5, PtType.PWRX, 7),
        (b"\x02\x13\x00\x00", PtType.CFE, 4),
        (b"\x02\x53" + b"\x00" * 9, PtType.EVD, 11),
        (b"\x99\x01", PtType.MODE_EXEC, 2),
        (b"\x99\x20", PtType.MODE_TSX, 2),
        (b"\x19" + b"\x00" * 7, PtType.TSC, 8),
        (b"\x59\x01", PtType.MTC, 2),
    ],
)
def test_fixed_layout_packets(raw, expected, size):
    pkt = list(iter_packets(raw))[0]
    assert pkt.type is expected
    assert pkt.size == size


@pytest.mark.parametrize(("ptw_count", "size"), [(0, 6), (1, 10)])
def test_ptwrite_payload_sizes(ptw_count, size):
    raw = bytes([0x02, 0x12 | (ptw_count << 5)]) + b"\x00" * (size - 2)
    pkt = list(iter_packets(raw))[0]
    assert pkt.type is PtType.PTWRITE
    assert pkt.size == size


def test_bip_length_follows_bbp_size_bit():
    # BBP with the size bit set selects 4-byte block items, clear selects 8.
    four = b"\x02\x63\x80" + b"\x04" + b"\x00" * 4
    eight = b"\x02\x63\x00" + b"\x04" + b"\x00" * 8
    assert [(p.type, p.size) for p in iter_packets(four)][1] == (PtType.BIP, 5)
    assert [(p.type, p.size) for p in iter_packets(eight)][1] == (PtType.BIP, 9)


def test_bip_context_ends_at_bep():
    """After BEP a 0bxxxxx100 byte reverts to being a short TNT, so it must be
    consumed as one byte and not as a five-byte block item."""
    buf = b"\x02\x63\x80" + b"\x02\x33" + b"\x04" + b"\x00" * 4
    pkts = [(p.type, p.size) for p in iter_packets(buf)]
    assert pkts[:3] == [(PtType.BBP, 3), (PtType.BEP, 2), (PtType.TNT, 1)]


def test_truncated_packet_is_not_yielded():
    """A packet split across the end of the buffer must be left alone, not
    guessed at — the next AUX read continues the stream."""
    it = iter_packets(tip6(0x4000)[:5])
    assert list(it) == []


def test_truncated_offset_reports_undecoded_tail():
    decoder = PtIpDecoder()
    trace = tip6(0x4000)
    decoder.decode(trace[:5])
    assert decoder.pending == 5


def test_bad_packet_resyncs_to_next_psb():
    """Adversarial: a byte that decodes to nothing must not swallow the rest
    of the stream, and must not loop forever."""
    buf = b"\x05\x05\x05" + PSB + tip6(0x2000)
    pkts = list(iter_packets(buf))
    assert [p.type for p in pkts] == [PtType.BAD, PtType.PSB, PtType.TIP]
    assert pkts[0].size == 3  # the whole unsynchronized run, not one byte


# ── IP reconstruction ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ipc", "payload", "last", "expected"),
    [
        (1, 0xBEEF, 0x00007FFF12340000, 0x00007FFF1234BEEF),  # update low 16
        (2, 0xDEADBEEF, 0x00007FFF12345678, 0x00007FFFDEADBEEF),  # update low 32
        (3, 0x00007FFF1234, 0xFFFFFFFFFFFFFFFF, 0x00007FFF1234),  # sign-extend 48
        (3, 0x800000000000, 0, 0xFFFF800000000000),  # sign-extend 48, negative
        (4, 0x00007FFF1234, 0xFFFF000000000000, 0xFFFF00007FFF1234),  # update low 48
        (6, 0x00007FFF12345678, 0, 0x00007FFF12345678),  # full
    ],
)
def test_calc_ip(ipc, payload, last, expected):
    assert calc_ip(ipc, payload, last) == expected


def test_suppressed_ip_leaves_last_ip_untouched():
    assert calc_ip(0, 0, 0x1234) == 0x1234


def test_tip_ips_decoded_end_to_end():
    trace = PSB + b"\x02\x23" + tip6(0x401000) + tip(1, (0x2000).to_bytes(2, "little"))
    assert PtIpDecoder().decode(trace) == [0x401000, 0x402000]


def test_fup_updates_last_ip_but_is_not_reported():
    """Falsification: only TIP targets are block entries.  If FUP leaked into
    the output the list would have three entries, and if FUP did not update
    last_ip the compressed TIP would decode against 0."""
    trace = PSB + fup(6, (0x500000).to_bytes(8, "little")) + tip(1, (0x1234).to_bytes(2, "little"))
    assert PtIpDecoder().decode(trace) == [0x501234]


def test_compressed_ip_dropped_without_a_reference():
    """A partial IP with no preceding full IP reconstructs against zero, which
    is a fabricated address — it must be dropped, not reported."""
    assert PtIpDecoder().decode(tip(1, (0x1234).to_bytes(2, "little"))) == []


@pytest.mark.parametrize("resync", [b"\x02\xf3", PSB])
def test_overflow_and_psb_reset_the_reference_ip(resync):
    trace = tip6(0x401000) + resync + tip(1, (0x1234).to_bytes(2, "little"))
    assert PtIpDecoder().decode(trace) == [0x401000]


def test_decode_is_resumable_across_reads():
    """The AUX buffer is read in chunks; a packet straddling a chunk boundary
    must decode once the tail is carried into the next call."""
    trace = PSB + tip6(0x401000) + tip6(0x402000)
    decoder = PtIpDecoder()
    cut = len(PSB) + 4
    assert decoder.decode(trace[:cut]) == []
    assert decoder.decode(trace[cut:]) == [0x401000, 0x402000]


def test_cutoff_addr_drops_dynamic_code():
    decoder = PtIpDecoder(cutoff_addr=0x500000)
    trace = tip6(0x401000) + tip6(0x7FFF00000000)
    assert decoder.decode(trace) == [0x401000]


def test_decoder_survives_random_bytes():
    """Adversarial: arbitrary bytes must terminate and never raise."""
    rng = random.Random(1234)
    for _ in range(200):
        blob = bytes(rng.getrandbits(8) for _ in range(256))
        PtIpDecoder().decode(blob)


# ── coverage map ───────────────────────────────────────────────────────


MAP = 65536
# Distinct mod MAP, so a collision cannot pass for an ordering effect.
BLOCKS = (0x401000, 0x402000, 0x403000)


def _trace(*ips: int) -> bytes:
    return PSB + b"".join(tip6(ip) for ip in ips)


def test_block_map_is_order_insensitive():
    a = PtCoverage(map_size=MAP)
    b = PtCoverage(map_size=MAP)
    a.ingest(_trace(*BLOCKS))
    b.ingest(_trace(BLOCKS[2], BLOCKS[0], BLOCKS[1]))
    assert bytes(a.edge_map) == bytes(b.edge_map)
    assert a.total_edges == 3


def test_edge_map_is_order_sensitive():
    """Falsification of the mode split: block mode hashes one bit per IP, so
    the two orderings above collide.  Edge mode folds the predecessor in, so
    they must not."""
    a = PtCoverage(map_size=MAP, mode=PtMapMode.EDGE)
    b = PtCoverage(map_size=MAP, mode=PtMapMode.EDGE)
    a.ingest(_trace(*BLOCKS))
    b.ingest(_trace(BLOCKS[2], BLOCKS[0], BLOCKS[1]))
    assert bytes(a.edge_map) != bytes(b.edge_map)


def test_self_loop_does_not_collapse_to_bucket_zero():
    cov = PtCoverage(map_size=MAP, mode=PtMapMode.EDGE)
    cov.ingest(_trace(0x401234, 0x401234))
    assert cov.edge_map[0] == 0


def test_reset_clears_the_predecessor():
    """Two runs of the same input must produce the same map, which only holds
    if the predecessor chain is cleared between them."""
    cov = PtCoverage(map_size=MAP, mode=PtMapMode.EDGE)
    cov.ingest(_trace(BLOCKS[0], BLOCKS[1]))
    first = bytes(cov.edge_map)

    cov.reset_edge_map()
    cov.edge_map[:] = bytes(len(cov.edge_map))
    cov.ingest(_trace(BLOCKS[0], BLOCKS[1]))
    assert bytes(cov.edge_map) == first


def test_new_coverage_reported_once():
    cov = PtCoverage(map_size=MAP)
    cov.ingest(_trace(BLOCKS[0]))
    assert cov.is_new_coverage()
    assert not cov.is_new_coverage()


def test_base_address_makes_pie_addresses_stable():
    """Under ASLR the same block lands at a different absolute address every
    run; hashing must use the load-relative address."""
    a = PtCoverage(map_size=MAP)
    a.set_base(0x555500000000)
    a.ingest(_trace(0x555500001000))

    b = PtCoverage(map_size=MAP)
    b.set_base(0x7F0000000000)
    b.ingest(_trace(0x7F0000001000))
    assert bytes(a.edge_map) == bytes(b.edge_map)


def test_stats_counts_packets_and_ips():
    cov = PtCoverage(map_size=MAP)
    cov.ingest(_trace(BLOCKS[0], BLOCKS[1]))
    stats = cov.stats
    assert stats["pt_blocks"] == 2
    assert stats["pt_bytes"] == len(_trace(BLOCKS[0], BLOCKS[1]))


# ── regressions found by differential testing ──────────────────────────
#
# The four below were caught by running this decoder against the kernel's
# ``intel-pt-pkt-decoder.c`` on ~9,000 generated streams.  Three are the same
# class of bug: C truncates a shifted payload at 64 bits and Python does not.


def test_regression_short_tnt_payload_is_64_bit():
    pkt = list(iter_packets(b"\xfe"))[0]
    assert pkt.payload == 0xFC00000000000000
    assert pkt.payload <= 0xFFFFFFFFFFFFFFFF


def test_regression_long_tnt_payload_is_64_bit():
    raw = bytes([0x02, 0xA3]) + b"\xff" * 6
    assert list(iter_packets(raw))[0].payload <= 0xFFFFFFFFFFFFFFFF


def test_regression_cyc_payload_is_64_bit():
    """Ten CYC bytes carry 68 payload bits; the top four are dropped."""
    raw = bytes([0xFF]) + b"\xff" * 8 + b"\xfe"
    pkt = list(iter_packets(raw))[0]
    assert pkt.size == 10
    assert pkt.payload <= 0xFFFFFFFFFFFFFFFF


def test_regression_vmcs_reports_payload_length():
    assert list(iter_packets(b"\x02\xc8" + b"\x00" * 5))[0].count == 5


def test_regression_cyc_extension_flag_is_bit_zero():
    """Bit 2 continues the header byte, but bit 0 continues every byte after
    it.  Reading bit 2 on extension bytes ended the packet early and left the
    rest of the payload to be decoded as packets."""
    two = list(iter_packets(b"\x07\x03\x02"))
    assert [(p.type, p.size) for p in two] == [(PtType.CYC, 3)]
