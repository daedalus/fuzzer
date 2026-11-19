"""Structure-aware MPEG Program Stream (.mpg/.vob) mutator.

ISO/IEC 13818-1 §2.5.3. A program stream is a run of start-code units:

  00 00 01 BA  pack header     MPEG-2: 10 bytes + pack_stuffing_length (low
                               3 bits of byte 9); MPEG-1: 8 bytes
  00 00 01 BB  system header   u16 length + body
  00 00 01 xx  PES packet      xx = stream_id (BC..FF), u16 length + body
  00 00 01 B9  program end     no body

  PES body (MPEG-2, stream ids other than BC/BE/BF/F0-F2/F8/FF):
    [length:2] [10 scrambling(2) ...] [PTS_DTS_flags(2) ...]
    [PES_header_data_length] [header data] [payload]

Private stream 1 (BD) opens its payload with a DVD substream id
(AC-3 0x80-0x87, DTS 0x88-0x8F, LPCM 0xA0-0xA7, subpicture 0x20-0x3F):
FFmpeg's `mpeg.c` dispatches on it, a demux surface no other mutator here
reaches. Units are delimited by their length fields, not by scanning, so a
start code inside a payload (e.g. an MPEG-2 sequence header) never splits
a unit; bytes between units are kept as junk units so round trips are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from fuzzer_tool.core.mutations.generic import _swap_pair
from fuzzer_tool.core.rand_pool import RandPool

START_CODE = b"\x00\x00\x01"
START_LEN = 4  # start code prefix + code byte
PROGRAM_END = 0xB9
PACK_CODE = 0xBA
SYSTEM_HEADER = 0xBB
PROGRAM_STREAM_MAP = 0xBC
PRIVATE_1 = 0xBD
FIRST_PS_CODE = PROGRAM_END  # codes below B9 are elementary-stream codes
JUNK = -1  # bytes between units, kept verbatim

LENGTH_LEN = 2
MPEG2_PACK_LEN = 10
MPEG1_PACK_LEN = 8
MPEG2_PACK_MARK = 0x40  # '01' in the top two bits of the first SCR byte
STUFFING_MASK = 0x07
MPEG2_PES_MARK = 0x80  # '10' in the top two bits of the PES flags byte
PES_FIXED_LEN = 3  # flags, flags, PES_header_data_length

# Stream ids whose PES body has no MPEG-2 optional header (§2.4.3.7).
NO_PES_HEADER = frozenset({0xBC, 0xBE, 0xBF, 0xF0, 0xF1, 0xF2, 0xF8, 0xFF})

# PS map, private 1, padding, private 2, audio, video, H.222.1 E, directory.
STREAM_IDS = (0xBC, 0xBD, 0xBE, 0xBF, 0xC0, 0xDF, 0xE0, 0xEF, 0xF8, 0xFF)
# Subpicture, AC-3, DTS, LPCM, MLP, out of range.
SUBSTREAM_IDS = (0x20, 0x80, 0x88, 0xA0, 0xB0, 0xFF, 0x00)
PTS_FLAGS = (0b01, 0b11)  # forbidden; PTS+DTS (may overrun the header)


@dataclass
class PsUnit:
    """One start-code unit: *code* byte (or JUNK) and the bytes after it."""

    code: int
    body: bytes

    def to_bytes(self) -> bytes:
        if self.code == JUNK:
            return self.body
        return START_CODE + bytes([self.code]) + self.body

    @property
    def pes_header_len(self) -> int | None:
        """PES_header_data_length, or None when the body has no MPEG-2 header."""
        b = self.body
        if self.code < PROGRAM_STREAM_MAP or self.code in NO_PES_HEADER:
            return None
        if len(b) < LENGTH_LEN + PES_FIXED_LEN or b[LENGTH_LEN] & 0xC0 != MPEG2_PES_MARK:
            return None
        return b[LENGTH_LEN + 2]


def _is_code(data: bytes, pos: int) -> bool:
    return (
        data[pos : pos + 3] == START_CODE and pos + 3 < len(data) and data[pos + 3] >= FIRST_PS_CODE
    )


def _next_code(data: bytes, start: int) -> int:
    """Offset of the next PS start code at or after *start*, or len(data)."""
    n = len(data)
    i = data.find(START_CODE, start)
    while i != -1 and not _is_code(data, i):
        i = data.find(START_CODE, i + 1)
    return n if i == -1 else i


def _body_len(data: bytes, off: int, code: int) -> int:
    """Declared body length of the unit whose body starts at *off*."""
    n = len(data)
    if code == PROGRAM_END:
        return 0
    if code == PACK_CODE:
        if off >= n or data[off] & 0xC0 != MPEG2_PACK_MARK:
            return MPEG1_PACK_LEN
        stuffing = data[off + 9] & STUFFING_MASK if off + 9 < n else 0
        return MPEG2_PACK_LEN + stuffing
    if off + LENGTH_LEN > n:
        return n - off
    return LENGTH_LEN + int.from_bytes(data[off : off + LENGTH_LEN], "big")


def parse_ps_units(data: bytes) -> list[PsUnit] | None:
    """Split a program stream into units; None unless it opens with a pack header."""
    if data[:START_LEN] != START_CODE + bytes([PACK_CODE]):
        return None

    units: list[PsUnit] = []
    pos, n = 0, len(data)
    while pos < n:
        # Junk up to the next start code (lengths that undershoot, stray bytes).
        if not _is_code(data, pos):
            nxt = _next_code(data, pos + 1)
            units.append(PsUnit(JUNK, data[pos:nxt]))
            pos = nxt
            continue

        code = data[pos + 3]
        body_at = pos + START_LEN
        end = min(n, body_at + _body_len(data, body_at, code))
        units.append(PsUnit(code, data[body_at:end]))
        pos = end

    return units


def serialize_ps_units(units: list[PsUnit]) -> bytes:
    return b"".join(u.to_bytes() for u in units)


class MpegpsMutator:
    """Structure-aware MPEG-PS unit-level mutator."""

    def __init__(self, seed=None):
        # Standalone default pool; a caller's ``rng=`` wins (Hard Rule 16).
        self._rng = RandPool(seed=seed)

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        self._rng = rng or self._rng
        units = parse_ps_units(data)
        if not units:
            return self._generate_random_ps(max_len=max_len, rng=self._rng)

        ops = (
            self._mutate_pes_length,
            self._mutate_stream_id,
            self._mutate_pes_header_len,
            self._mutate_pts_flags,
            self._mutate_pack_header,
            self._mutate_substream,
            self._duplicate_unit,
            self._delete_unit,
            self._reorder_units,
        )
        ops[self._rng.randint(0, len(ops) - 1)](units)
        return serialize_ps_units(units)[:max_len]

    def _mutate_pes_length(self, units: list[PsUnit]) -> None:
        """Set a PES/system-header length to an edge: under- and overshoot
        desync the demuxer's unit walk; 0 means "unbounded" for video PES."""
        cands = [u for u in units if u.code >= SYSTEM_HEADER and len(u.body) >= LENGTH_LEN]
        if not cands:
            return
        unit = self._rng.choice(cands)
        actual = len(unit.body) - LENGTH_LEN
        value = self._rng.choice((0, 1, max(actual - 1, 0), actual + 1, 0xFFFF)) & 0xFFFF
        unit.body = value.to_bytes(LENGTH_LEN, "big") + unit.body[LENGTH_LEN:]

    def _mutate_stream_id(self, units: list[PsUnit]) -> None:
        """Remap a PES stream_id: moves the payload to another parser
        (video <-> audio <-> private <-> padding / PS map)."""
        cands = [u for u in units if u.code >= PROGRAM_STREAM_MAP]
        if not cands:
            return
        unit = self._rng.choice(cands)
        unit.code = self._rng.choice(STREAM_IDS)

    def _mutate_pes_header_len(self, units: list[PsUnit]) -> None:
        """Corrupt PES_header_data_length: the payload start moves past or
        before the real header data."""
        cands = [u for u in units if u.pes_header_len is not None]
        if not cands:
            return
        unit = self._rng.choice(cands)
        b = bytearray(unit.body)
        at = LENGTH_LEN + 2
        b[at] = self._rng.choice((0, 1, 0xFF, (b[at] + 1) & 0xFF))
        unit.body = bytes(b)

    def _mutate_pts_flags(self, units: list[PsUnit]) -> None:
        """Set PTS_DTS_flags to forbidden 01 or to 11 without DTS bytes."""
        cands = [u for u in units if u.pes_header_len is not None]
        if not cands:
            return
        unit = self._rng.choice(cands)
        b = bytearray(unit.body)
        at = LENGTH_LEN + 1
        b[at] = (b[at] & 0x3F) | (self._rng.choice(PTS_FLAGS) << 6)
        unit.body = bytes(b)

    def _mutate_pack_header(self, units: list[PsUnit]) -> None:
        """Pack header: claim 7 stuffing bytes that aren't there, zero the
        mux rate, or flip the MPEG-1/MPEG-2 layout marker."""
        cands = [u for u in units if u.code == PACK_CODE and len(u.body) >= MPEG2_PACK_LEN]
        if not cands:
            return
        unit = self._rng.choice(cands)
        b = bytearray(unit.body)
        sub = self._rng.randint(0, 2)
        if sub == 0:
            b[9] |= STUFFING_MASK
        elif sub == 1:
            b[6:9] = b"\x00\x00\x03"  # program_mux_rate 0, marker bits kept
        else:
            b[0] ^= 0x60  # '01' (MPEG-2) <-> '0010' (MPEG-1)
        unit.body = bytes(b)

    def _mutate_substream(self, units: list[PsUnit]) -> None:
        """Rewrite a private-stream-1 substream id (AC-3/DTS/LPCM/subpicture)."""
        cands = [
            u
            for u in units
            if u.code == PRIVATE_1
            and u.pes_header_len is not None
            and len(u.body) > LENGTH_LEN + PES_FIXED_LEN + u.pes_header_len
        ]
        if not cands:
            return
        unit = self._rng.choice(cands)
        at = LENGTH_LEN + PES_FIXED_LEN + unit.pes_header_len
        b = bytearray(unit.body)
        b[at] = self._rng.choice(SUBSTREAM_IDS)
        unit.body = bytes(b)

    def _duplicate_unit(self, units: list[PsUnit]) -> None:
        """Clone a unit in place (repeated PES / pack headers)."""
        idx = self._rng.randint(0, len(units) - 1)
        units.insert(idx + 1, PsUnit(units[idx].code, units[idx].body))

    def _delete_unit(self, units: list[PsUnit]) -> None:
        """Drop a unit, never the leading pack header (keeps the sniffer hit)."""
        if len(units) > 1:
            units.pop(self._rng.randint(1, len(units) - 1))

    def _reorder_units(self, units: list[PsUnit]) -> None:
        """Swap two units after the leading pack header."""
        if (pair := _swap_pair(len(units) - 1, self._rng)) is not None:
            i, j = pair[0] + 1, pair[1] + 1
            units[i], units[j] = units[j], units[i]

    def _generate_random_ps(self, max_len: int = 65536, rng=None) -> bytes:
        """Pack header + system header + video PES + AC-3 private PES + end."""
        self._rng = rng or self._rng

        # Pack: SCR 0, mux rate 0x189C3 (DVD-typical), no stuffing.
        pack = PsUnit(PACK_CODE, bytes.fromhex("4400040004010189c3f8"))
        # System header: rate bound, bounds, then E0/C0/BD/BF buffer entries.
        system = PsUnit(SYSTEM_HEADER, bytes.fromhex("001280c4e104e1ffe0e0e8c0c020bde03abfe002"))
        pts = bytes.fromhex("2100010001")

        def pes(stream_id: int, payload: bytes) -> PsUnit:
            content = b"\x81\x80\x05" + pts + payload
            return PsUnit(stream_id, len(content).to_bytes(LENGTH_LEN, "big") + content)

        video = pes(0xE0, b"\x00\x00\x01\xb3" + self._rng.randbytes(60))
        audio = pes(PRIVATE_1, b"\x80\x01\x00\x01\x0b\x77" + self._rng.randbytes(40))
        end = PsUnit(PROGRAM_END, b"")
        return serialize_ps_units([pack, system, video, audio, end])[:max_len]
