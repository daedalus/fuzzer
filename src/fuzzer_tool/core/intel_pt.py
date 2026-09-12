"""Intel Processor Trace packet decoder and coverage map.

Intel PT emits a compressed control-flow trace into a hardware buffer with no
instrumentation in the target at all, so it gives coverage on binaries we
cannot rebuild — the case ``ptrace_coverage.py`` currently covers with
breakpoints, at roughly two orders of magnitude more overhead.

Scope is deliberately the *packet* level, not instruction-flow reconstruction:

    TIP / TIP.PGE / TIP.PGD / FUP packets carry a target IP.  Taking the
    targets of TIP packets alone gives one entry per indirect branch, call and
    return — enough for a block bitmap, and the only PT mode with a published
    stability baseline (honggfuzz ``--linux_perf_ipt_block``; PTrix, AsiaCCS
    '19).  Full reconstruction needs the decoded binary image to follow the
    conditional-branch (TNT) bits, which is what pulls in libipt.

What that costs, stated plainly: conditional branches are *not* visible here.
A TIP target is the destination of an indirect transfer, so consecutive TIP
targets are not adjacent basic blocks and ``PtMapMode.EDGE`` records path
fragments, not CFG edges.  Block mode is the default for that reason.

Packet encodings follow the Intel SDM Vol 3C ("Intel Processor Trace") and are
cross-checked against the kernel's ``intel-pt-pkt-decoder.c`` and libipt.
Unlike perf's decoder this one does not fold trailing PAD bytes into the
preceding packet's size; PAD is a no-op here, so it is yielded on its own.
"""

import logging
from enum import Enum, IntEnum
from typing import NamedTuple

from fuzzer_tool.core.count_class import classify_counts

log = logging.getLogger(__name__)

# PSB is the only self-synchronizing pattern in the stream: the decoder
# restarts here after a bad byte, and it is the longest packet (16 bytes),
# so no legitimate packet can need more than that many bytes to complete.
PSB = b"\x02\x82" * 8
MAX_PACKET_BYTES = len(PSB)

DEFAULT_MAP_SIZE = 65536


class PtType(IntEnum):
    """Packet kinds the decoder distinguishes."""

    BAD = 0
    PAD = 1
    TNT = 2
    TIP = 3
    TIP_PGE = 4
    TIP_PGD = 5
    FUP = 6
    PSB_PKT = 7
    PSBEND = 8
    OVF = 9
    TRACESTOP = 10
    CBR = 11
    TSC = 12
    TMA = 13
    MTC = 14
    CYC = 15
    MODE_EXEC = 16
    MODE_TSX = 17
    PIP = 18
    VMCS = 19
    MNT = 20
    PTWRITE = 21
    EXSTOP = 22
    MWAIT = 23
    PWRE = 24
    PWRX = 25
    BBP = 26
    BIP = 27
    BEP = 28
    CFE = 29
    EVD = 30


# PSB collides with the module-level constant name only in the enum namespace;
# expose the packet type under the name callers expect.
PtType.PSB = PtType.PSB_PKT


class PtCtx(Enum):
    """Block-item context.  BBP switches the meaning of ``0bxxxxx100`` bytes
    to 4- or 8-byte BIP packets until the matching BEP."""

    NONE = 0
    BLK_4 = 4
    BLK_8 = 8


class PtPacket(NamedTuple):
    type: PtType
    offset: int
    size: int
    payload: int
    count: int


class PtMapMode(Enum):
    """How a decoded IP becomes a bitmap index.

    BLOCK — one bit per masked IP (honggfuzz's model).
    EDGE  — IP folded with the previous IP, AFL-style.  See the module
            docstring: these are path fragments, not CFG edges.
    """

    BLOCK = "block"
    EDGE = "edge"


_BAD = PtPacket(PtType.BAD, 0, 1, 0, 0)

# First-byte opcode (bits[4:0]) for the four IP-bearing packets.
_IP_OPCODES = {
    0x0D: PtType.TIP,
    0x11: PtType.TIP_PGE,
    0x01: PtType.TIP_PGD,
    0x1D: PtType.FUP,
}
_IP_TYPES = frozenset(_IP_OPCODES.values())

# IPBytes -> total packet size including the opcode byte.  3 and 4 are both
# six payload bytes and differ only in how they are extended; 5 and 7 are
# reserved.
_IP_SIZES = {0: 1, 1: 3, 2: 5, 3: 7, 4: 7, 6: 9}

_U64 = 0xFFFFFFFFFFFFFFFF
_SHORT_TNT_MAX = 6  # branch results in a one-byte TNT
_LONG_TNT_MAX = 47  # branch results in an eight-byte TNT
_CYC_MAX_BYTES = 10

_TIMING_OPCODE = 0x19  # shared bits[4:0] of MODE, TSC and MTC
_PTW_OPCODE = 0x12  # bits[4:0] of the second byte
_BIP_TAG = 0x04  # bits[2:0] of a block-item byte
_EXT_PREFIX = 0x02

# Second byte -> (type, size) for extended packets with no payload we use.
_EXT_SIMPLE = {
    0x83: (PtType.TRACESTOP, 2),
    0xF3: (PtType.OVF, 2),
    0x23: (PtType.PSBEND, 2),
    0x62: (PtType.EXSTOP, 2),
    0xE2: (PtType.EXSTOP, 2),
    0x33: (PtType.BEP, 2),
    0xB3: (PtType.BEP, 2),
}

# Second byte -> (type, size, payload offset, payload length).  ``count`` is
# reported as the payload length only where the reference decoder does so.
_EXT_PAYLOAD = {
    0x43: (PtType.PIP, 8, 2, 6),
    0x03: (PtType.CBR, 4, 2, 2),
    0xC8: (PtType.VMCS, 7, 2, 5),
    0xC2: (PtType.MWAIT, 10, 2, 8),
    0x22: (PtType.PWRE, 4, 2, 2),
    0xA2: (PtType.PWRX, 7, 2, 5),
}

# Packet kinds that end a BBP block-item run.  Everything else leaves the
# context alone; mirrors intel_pt_upd_pkt_ctx().
_CTX_CLEARING = frozenset(
    {
        PtType.TNT,
        PtType.TIP,
        PtType.TIP_PGD,
        PtType.TIP_PGE,
        PtType.MODE_EXEC,
        PtType.MODE_TSX,
        PtType.PIP,
        PtType.OVF,
        PtType.VMCS,
        PtType.TRACESTOP,
        PtType.PSB,
        PtType.PSBEND,
        PtType.PTWRITE,
        PtType.MWAIT,
        PtType.BEP,
        PtType.CFE,
        PtType.EVD,
    }
)


def _le(buf: bytes, start: int, length: int) -> int:
    return int.from_bytes(buf[start : start + length], "little")


def _fits(buf: bytes, pos: int, size: int) -> bool:
    return pos + size <= len(buf)


def calc_ip(ipc: int, payload: int, last_ip: int) -> int:
    """Reconstruct a full IP from a compressed one.

    IPBytes 1/2/4 replace the low 16/32/48 bits of the previous IP; 3 is a
    sign-extended 48-bit address; 6 is the address in full; 0 means the IP was
    suppressed, so the previous one stands.
    """
    if ipc == 1:
        return (last_ip & ~0xFFFF) | (payload & 0xFFFF)

    if ipc == 2:
        return (last_ip & ~0xFFFFFFFF) | (payload & 0xFFFFFFFF)

    if ipc == 3:
        ip = payload & 0xFFFFFFFFFFFF
        return ip | 0xFFFF000000000000 if ip & 0x800000000000 else ip

    if ipc == 4:
        return (last_ip & ~0xFFFFFFFFFFFF) | (payload & 0xFFFFFFFFFFFF)

    if ipc == 6:
        return payload

    return last_ip


def _parse_ip(buf: bytes, pos: int, ptype: PtType) -> PtPacket | None:
    ipc = buf[pos] >> 5
    size = _IP_SIZES.get(ipc)
    if size is None:
        return _BAD

    if not _fits(buf, pos, size):
        return None

    return PtPacket(ptype, pos, size, _le(buf, pos + 1, size - 1), ipc)


def _parse_short_tnt(buf: bytes, pos: int) -> PtPacket:
    """One byte: a stop bit, then one result per bit below it."""
    byte = buf[pos]
    count = _SHORT_TNT_MAX
    while count and not byte & 0x80:
        byte = (byte << 1) & 0xFF
        count -= 1

    # Left-justify the results like the 64-bit form: the stop bit shifts out.
    return PtPacket(PtType.TNT, pos, 1, (byte << 57) & _U64, count)


def _parse_long_tnt(buf: bytes, pos: int) -> PtPacket | None:
    if not _fits(buf, pos, 8):
        return None

    payload = _le(buf, pos, 8)
    count = _LONG_TNT_MAX
    while count and not payload & (1 << 63):
        payload = (payload << 1) & _U64
        count -= 1

    return PtPacket(PtType.TNT, pos, 8, (payload << 1) & _U64, count)


def _parse_cyc(buf: bytes, pos: int) -> PtPacket | None:
    """Variable length: bit 2 of each byte continues the payload."""
    payload = buf[pos] >> 3
    size = 1
    shift = 5
    # The flag is bit 2 of the header byte and bit 0 of each extension byte.
    more = buf[pos] >> 2 & 1
    while more:
        if size >= _CYC_MAX_BYTES:
            return _BAD

        if not _fits(buf, pos, size + 1):
            return None

        byte = buf[pos + size]
        payload |= (byte >> 1) << shift
        more = byte & 1
        size += 1
        shift += 7

    # A maximal CYC carries 68 payload bits; the top ones are dropped.
    return PtPacket(PtType.CYC, pos, size, payload & _U64, 0)


def _parse_timing(buf: bytes, pos: int) -> PtPacket | None:
    """MODE (0x99), TSC (0x19) and MTC (0x59) share opcode bits[4:0]."""
    byte = buf[pos]
    if byte == 0x19:
        return PtPacket(PtType.TSC, pos, 8, _le(buf, pos + 1, 7), 0) if _fits(buf, pos, 8) else None

    if byte == 0x59:
        return PtPacket(PtType.MTC, pos, 2, buf[pos + 1], 0) if _fits(buf, pos, 2) else None

    if byte != 0x99:
        return _BAD

    if not _fits(buf, pos, 2):
        return None

    return _parse_mode(buf, pos)


_MODE_EXEC_BITS = {0: 16, 1: 64, 2: 32}


def _parse_mode(buf: bytes, pos: int) -> PtPacket:
    leaf = buf[pos + 1] >> 5
    if leaf == 0:
        width = _MODE_EXEC_BITS.get(buf[pos + 1] & 3)
        if width is None:
            return _BAD

        return PtPacket(PtType.MODE_EXEC, pos, 2, width, buf[pos + 1])

    if leaf == 1 and buf[pos + 1] & 3 != 3:
        return PtPacket(PtType.MODE_TSX, pos, 2, buf[pos + 1] & 3, 0)

    return _BAD


def _parse_psb(buf: bytes, pos: int) -> PtPacket | None:
    if not _fits(buf, pos, len(PSB)):
        return None

    if buf[pos : pos + len(PSB)] != PSB:
        return _BAD

    return PtPacket(PtType.PSB, pos, len(PSB), 0, 0)


def _parse_tma(buf: bytes, pos: int) -> PtPacket | None:
    if not _fits(buf, pos, 7):
        return None

    ctc = buf[pos + 2] | buf[pos + 3] << 8
    fc = buf[pos + 5] | (buf[pos + 6] & 1) << 8
    return PtPacket(PtType.TMA, pos, 7, ctc, fc)


def _parse_3byte(buf: bytes, pos: int) -> PtPacket | None:
    """Only MNT lives behind the three-byte header."""
    if not _fits(buf, pos, 3):
        return None

    if buf[pos + 2] != 0x88:
        return _BAD

    if not _fits(buf, pos, 11):
        return None

    return PtPacket(PtType.MNT, pos, 11, _le(buf, pos + 3, 8), 0)


def _parse_ptw(buf: bytes, pos: int) -> PtPacket | None:
    sizes = {0: 6, 1: 10}
    count = buf[pos + 1] >> 5 & 3
    size = sizes.get(count)
    if size is None:
        return _BAD

    if not _fits(buf, pos, size):
        return None

    return PtPacket(PtType.PTWRITE, pos, size, _le(buf, pos + 2, size - 2), count)


def _parse_bbp(buf: bytes, pos: int) -> PtPacket | None:
    if not _fits(buf, pos, 3):
        return None

    # Bit 7 of the payload byte selects 4-byte block items over 8-byte ones.
    return PtPacket(PtType.BBP, pos, 3, buf[pos + 2] & 0x1F, buf[pos + 2] >> 7)


def _parse_cfe(buf: bytes, pos: int) -> PtPacket | None:
    if not _fits(buf, pos, 4):
        return None

    return PtPacket(PtType.CFE, pos, 4, buf[pos + 3], buf[pos + 2] & 0x1F)


def _parse_evd(buf: bytes, pos: int) -> PtPacket | None:
    if not _fits(buf, pos, 11):
        return None

    return PtPacket(PtType.EVD, pos, 11, _le(buf, pos + 3, 8), buf[pos + 2] & 0x3F)


_EXT_CUSTOM = {
    0xA3: _parse_long_tnt,
    0x82: _parse_psb,
    0x73: _parse_tma,
    0xC3: _parse_3byte,
    0x63: _parse_bbp,
    0x13: _parse_cfe,
    0x53: _parse_evd,
}


def _parse_ext(buf: bytes, pos: int) -> PtPacket | None:
    """Packets prefixed with 0x02; the second byte selects the kind."""
    if not _fits(buf, pos, 2):
        return None

    second = buf[pos + 1]
    if second & 0x1F == _PTW_OPCODE:
        return _parse_ptw(buf, pos)

    simple = _EXT_SIMPLE.get(second)
    if simple is not None:
        return PtPacket(simple[0], pos, simple[1], 0, 0)

    payload = _EXT_PAYLOAD.get(second)
    if payload is not None:
        ptype, size, off, length = payload
        if not _fits(buf, pos, size):
            return None

        count = length if ptype is PtType.VMCS else 0
        return PtPacket(ptype, pos, size, _le(buf, pos + off, length), count)

    custom = _EXT_CUSTOM.get(second)
    if custom is not None:
        return custom(buf, pos)

    return _BAD


def _parse_bip(buf: bytes, pos: int, ctx: PtCtx) -> PtPacket | None:
    size = 1 + ctx.value
    if not _fits(buf, pos, size):
        return None

    return PtPacket(PtType.BIP, pos, size, _le(buf, pos + 1, ctx.value), buf[pos] >> 3)


def _parse_one(buf: bytes, pos: int, ctx: PtCtx) -> PtPacket | None:
    """Decode the packet at *pos*.

    Returns None when the buffer ends mid-packet (the caller carries the tail
    into the next read) and a BAD packet when the bytes decode to nothing.
    """
    byte = buf[pos]
    if ctx is not PtCtx.NONE and byte & 0x07 == _BIP_TAG:
        return _parse_bip(buf, pos, ctx)

    if not byte & 1:
        if byte == 0x00:
            return PtPacket(PtType.PAD, pos, 1, 0, 0)

        if byte == _EXT_PREFIX:
            return _parse_ext(buf, pos)

        return _parse_short_tnt(buf, pos)

    if byte & 2:
        return _parse_cyc(buf, pos)

    ptype = _IP_OPCODES.get(byte & 0x1F)
    if ptype is not None:
        return _parse_ip(buf, pos, ptype)

    if byte & 0x1F == _TIMING_OPCODE:
        return _parse_timing(buf, pos)

    return _BAD


def _next_ctx(pkt: PtPacket, ctx: PtCtx) -> PtCtx:
    if pkt.type is PtType.BBP:
        return PtCtx.BLK_4 if pkt.count else PtCtx.BLK_8

    return PtCtx.NONE if pkt.type in _CTX_CLEARING else ctx


def _resync(buf: bytes, pos: int) -> int:
    """Distance from *pos* to the next PSB, or to the end of the buffer.

    A bad byte means the stream is out of step and only PSB re-establishes it.
    Advancing byte by byte instead would invent packets out of payload bytes.
    """
    nxt = buf.find(PSB, pos + 1)
    return (len(buf) if nxt < 0 else nxt) - pos


def iter_packets(buf: bytes, start: int = 0):
    """Yield packets from *buf*.

    A packet straddling the end of the buffer is not yielded, so
    ``offset + size`` of the last packet is exactly the consumed prefix.
    """
    pos = start
    ctx = PtCtx.NONE
    while pos < len(buf):
        pkt = _parse_one(buf, pos, ctx)
        if pkt is None:
            return

        if pkt.type is PtType.BAD:
            skip = _resync(buf, pos)
            yield PtPacket(PtType.BAD, pos, skip, 0, 0)
            pos += skip
            ctx = PtCtx.NONE
            continue

        yield pkt._replace(offset=pos)
        ctx = _next_ctx(pkt, ctx)
        pos += pkt.size


class PtIpDecoder:
    """Turns a raw PT byte stream into the IPs of its TIP packets.

    Feed it whatever the AUX buffer gave you; a packet split across two reads
    is carried over. IP compression is stateful, so a compressed IP arriving
    with no reference (start of stream, after OVF or PSB) is dropped rather
    than reconstructed against zero — a fabricated address would hash into the
    map as if it were real coverage.

    Args:
        cutoff_addr: drop IPs at or above this address.  Dynamic libraries map
            high and change address every run, so their blocks are noise
            unless the map is keyed on a load base.
    """

    def __init__(self, cutoff_addr: int | None = None):
        self.cutoff_addr = cutoff_addr
        self.pending = 0
        self.packets = 0
        self.dropped_bytes = 0
        self._last_ip = 0
        self._have_ip = False
        self._tail = b""

    def reset(self) -> None:
        """Forget stream state.  Required between executions: the reference IP
        of the previous run would otherwise complete a compressed IP in this
        one."""
        self._last_ip = 0
        self._have_ip = False
        self._tail = b""
        self.pending = 0

    def decode(self, chunk: bytes) -> list[int]:
        """Return the TIP target IPs in *chunk*."""
        buf = self._tail + chunk if self._tail else chunk
        ips: list[int] = []
        consumed = 0

        for pkt in iter_packets(buf):
            consumed = pkt.offset + pkt.size
            self.packets += 1
            if pkt.type is PtType.BAD:
                self.dropped_bytes += pkt.size
                continue

            ip = self._track_ip(pkt)
            if ip is not None:
                ips.append(ip)

        self._keep_tail(buf, consumed)
        return ips

    def _track_ip(self, pkt: PtPacket) -> int | None:
        """Update the reference IP; return the IP if it is a block entry."""
        if pkt.type in (PtType.OVF, PtType.PSB):
            self._last_ip = 0
            self._have_ip = False
            return None

        if pkt.type not in _IP_TYPES or not pkt.count:
            return None

        # 3 and 6 carry the whole address; the rest need a reference.
        if pkt.count not in (3, 6) and not self._have_ip:
            return None

        self._last_ip = calc_ip(pkt.count, pkt.payload, self._last_ip)
        self._have_ip = True
        if pkt.type is not PtType.TIP:
            return None

        if self.cutoff_addr is not None and self._last_ip >= self.cutoff_addr:
            return None

        return self._last_ip

    def _keep_tail(self, buf: bytes, consumed: int) -> None:
        tail = buf[consumed:]
        # Nothing legitimate needs more than a PSB's worth of bytes to
        # complete, so a longer tail is garbage, not a split packet.
        if len(tail) > MAX_PACKET_BYTES:
            self.dropped_bytes += len(tail)
            tail = b""

        self._tail = tail
        self.pending = len(tail)


class PtCoverage:
    """Coverage map fed by Intel PT rather than by instrumentation.

    Mirrors the surface ``PtraceCoverage`` exposes (``edge_map``,
    ``reset_edge_map``, ``is_new_coverage``) so the two closed-source coverage
    backends are interchangeable at the call site.
    """

    def __init__(
        self,
        map_size: int = DEFAULT_MAP_SIZE,
        mode: PtMapMode = PtMapMode.BLOCK,
        cutoff_addr: int | None = None,
    ):
        self.map_size = map_size
        self.mode = mode
        self.edge_map = bytearray(map_size)
        self.prev_location = 0
        self.total_edges = 0
        self.cumulative_edges = 0
        self.total_blocks = 0
        self.total_bytes = 0
        self._base_address: int | None = None
        self._decoder = PtIpDecoder(cutoff_addr=cutoff_addr)
        self._map_snapshot = classify_counts(bytes(self.edge_map))

    def set_base(self, base: int | None) -> None:
        """Hash addresses relative to *base*.

        PIE binaries land somewhere new on every exec, so absolute IPs would
        make the map unreproducible run to run — the same reason
        ``PtraceCoverage.record_edge`` subtracts the load base.
        """
        self._base_address = base

    def ingest(self, raw: bytes) -> int:
        """Decode *raw* into the map; return how many entries are new."""
        self.total_bytes += len(raw)
        new = 0
        for ip in self._decoder.decode(raw):
            self.total_blocks += 1
            new += self.record_ip(ip)

        return new

    def record_ip(self, ip: int) -> bool:
        rel = ip - self._base_address if self._base_address else ip
        bucket = self._bucket(rel)
        if self.mode is PtMapMode.EDGE:
            # Shifted like the C shim's __afl_prev_loc: without it a block
            # reached twice in a row would XOR to bucket 0.
            self.prev_location = rel >> 1

        if self.edge_map[bucket]:
            return False

        self.edge_map[bucket] = 1
        self.total_edges += 1
        self.cumulative_edges += 1
        return True

    def _bucket(self, rel: int) -> int:
        if self.mode is PtMapMode.EDGE:
            return (rel ^ self.prev_location) % self.map_size

        return rel % self.map_size

    def reset_edge_map(self) -> None:
        """Clear per-execution state.  The map itself is cleared by the caller
        that owns it, exactly as in the ptrace backend."""
        self.prev_location = 0
        self.total_edges = 0
        self._decoder.reset()
        self._map_snapshot = classify_counts(bytes(self.edge_map))

    def is_new_coverage(self) -> bool:
        classified = classify_counts(bytes(self.edge_map))
        if classified == self._map_snapshot:
            return False

        self._map_snapshot = classified
        return True

    @property
    def stats(self) -> dict:
        return {
            "pt_blocks": self.total_blocks,
            "pt_bytes": self.total_bytes,
            "pt_packets": self._decoder.packets,
            "pt_dropped_bytes": self._decoder.dropped_bytes,
            "pt_map_entries": self.cumulative_edges,
            "pt_mode": self.mode.value,
        }
