"""Structure-aware JPEG mutations.

Parses JPEG markers (SOF, DHT, DQT, DRI, SOS, APP, COM) and applies
targeted corruption to header fields, quantization tables, Huffman
tables, and scan parameters. Falls back to random JPEG generation
when input is not valid JPEG.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass

# JPEG marker constants
SOI = 0xD8
EOI = 0xD9
SOF0 = 0xC0
SOF2 = 0xC2
DHT = 0xC4
DQT = 0xDB
DRI = 0xDD
SOS = 0xDA
RST0 = 0xD0
RST1 = 0xD1
RST2 = 0xD2
RST3 = 0xD3
RST4 = 0xD4
RST5 = 0xD5
RST6 = 0xD6
RST7 = 0xD7
APP0 = 0xE0
APP1 = 0xE1
APP15 = 0xEF
COM = 0xFE

# Markers that have segment data (length-prefixed)
LENGTH_MARKERS = {
    SOF0,
    SOF2,
    DHT,
    DQT,
    DRI,
    SOS,
    APP0,
    APP1,
    APP15,
    COM,
    0xC1,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,  # other SOF variants
    0xE2,
    0xE3,
    0xE4,
    0xE5,
    0xE6,
    0xE7,
    0xE8,
    0xE9,
    0xEA,
    0xEB,
    0xEC,
    0xED,
    0xEE,  # other APP
}

# Standalone markers (no data)
STANDALONE_MARKERS = {SOI, EOI, RST0, RST1, RST2, RST3, RST4, RST5, RST6, RST7}


@dataclass
class JpegMarker:
    """A single JPEG marker segment."""

    marker: int  # marker byte (e.g. 0xC0 for SOF0)
    data: bytes  # segment data (without length prefix)
    _scan_data: bytes = b""  # entropy-coded scan data (after SOS, not length-prefixed)

    def serialize(self) -> bytes:
        """Serialize marker to bytes: 0xFF + marker + length + data."""
        if self.marker in STANDALONE_MARKERS:
            return b"\xff" + bytes([self.marker])
        length = len(self.data) + 2  # length includes the 2 length bytes
        result = b"\xff" + bytes([self.marker]) + struct.pack(">H", length) + self.data
        # Append scan data after SOS (not length-prefixed)
        if self.marker == SOS and self._scan_data:
            result += self._scan_data
        return result


def parse_jpeg_markers(data: bytes) -> list[JpegMarker] | None:
    """Parse JPEG data into a list of JpegMarker segments.

    Returns None if data doesn't start with SOI or has no parseable markers.
    Handles the entropy-coded scan data after SOS by collecting bytes
    until the next valid marker (0xFF followed by a recognized marker byte).
    """
    if len(data) < 4 or data[0] != 0xFF or data[1] != SOI:
        return None

    markers = [JpegMarker(marker=SOI, data=b"")]
    pos = 2
    in_scan = False  # True after SOS — scan data runs until next valid marker

    while pos < len(data) - 1:
        if in_scan:
            # In entropy-coded segment: find next 0xFF that's a valid marker
            scan_start = pos
            while pos < len(data) - 1:
                if data[pos] == 0xFF:
                    next_byte = data[pos + 1]
                    if next_byte == 0x00:
                        pos += 2  # skip byte-stuffed 0xFF
                        continue
                    if next_byte == 0xFF:
                        pos += 1  # skip padding, keep scanning
                        continue
                    # Found a real marker — scan data ends BEFORE the 0xFF
                    break
                pos += 1
            scan_data = data[scan_start:pos]
            if markers:
                markers[-1]._scan_data = scan_data
            in_scan = False
            # If we stopped exactly at a standalone marker (e.g. EOI at end of file),
            # handle it directly — the outer loop's padding-skip would drop it
            if pos < len(data) and data[pos] == 0xFF and pos + 1 < len(data):
                next_byte = data[pos + 1]
                if next_byte in STANDALONE_MARKERS:
                    markers.append(JpegMarker(marker=next_byte, data=b""))
                    pos += 2
                    if next_byte == EOI:
                        break
            continue

        # Skip padding 0xFF bytes (allowed by spec between markers)
        while pos < len(data) - 1 and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data) - 1:
            break

        marker_byte = data[pos]
        pos += 1

        if marker_byte in STANDALONE_MARKERS:
            markers.append(JpegMarker(marker=marker_byte, data=b""))
            if marker_byte == EOI:
                break
            continue

        if pos + 1 >= len(data):
            break

        length = struct.unpack(">H", data[pos : pos + 2])[0]
        pos += 2

        if length < 2 or pos + length - 2 > len(data):
            seg_data = data[pos:]
            markers.append(JpegMarker(marker=marker_byte, data=seg_data))
            break

        seg_data = data[pos : pos + length - 2]
        pos += length - 2
        markers.append(JpegMarker(marker=marker_byte, data=seg_data))

        # After SOS, the next data is entropy-coded scan data
        if marker_byte == SOS:
            in_scan = True

    return markers if len(markers) > 1 else None


def serialize_jpeg_markers(markers: list[JpegMarker]) -> bytes:
    """Serialize a list of markers back to JPEG bytes."""
    return b"".join(m.serialize() for m in markers)


def _marker_name(marker: int) -> str:
    """Human-readable marker name."""
    names = {
        SOI: "SOI",
        EOI: "EOI",
        SOF0: "SOF0",
        SOF2: "SOF2",
        DHT: "DHT",
        DQT: "DQT",
        DRI: "DRI",
        SOS: "SOS",
        APP0: "APP0",
        APP1: "APP1",
        COM: "COM",
    }
    return names.get(marker, f"0xFF{marker:02X}")


class JpegMutator:
    """Structure-aware JPEG mutator.

    Dispatches one of 16 mutation operations per call, targeting
    specific JPEG structures for maximum code-path diversity.
    """

    def mutate(self, data: bytes, max_len: int = 4096) -> bytes:
        """Apply one structure-aware JPEG mutation."""
        markers = parse_jpeg_markers(data)
        if markers is None or len(markers) < 2:
            return self._generate_random_jpeg(max_len)

        op = random.randint(0, 15)
        mutators = [
            self._mutate_sof,
            self._mutate_dht,
            self._mutate_dqt,
            self._mutate_dri,
            self._mutate_sos,
            self._mutate_app,
            self._duplicate_marker,
            self._delete_marker,
            self._reorder_markers,
            self._corrupt_scan_data,
            self._add_comment,
            self._truncate_jpeg,
            self._duplicate_sof,
            self._corrupt_marker_length,
            self._swap_quantization_tables,
            self._generate_random_jpeg,
        ]
        result = mutators[op](markers, max_len)
        if isinstance(result, list):
            return serialize_jpeg_markers(result)[:max_len]
        return result[:max_len]

    def _mutate_sof(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Corrupt SOF header fields: precision, height, width, or components."""
        sof = _find_marker(markers, SOF0)
        if sof is None:
            sof = _find_marker(markers, SOF2)
        if sof is None or len(sof.data) < 7:
            return markers

        data = bytearray(sof.data)
        field = random.randint(0, 3)
        if field == 0 and len(data) >= 1:
            # Sample precision (typically 8)
            data[0] = random.choice([8, 12, 16, 0, 255])
        elif field == 1 and len(data) >= 3:
            # Height (2 bytes big-endian)
            h = struct.unpack(">H", data[1:3])[0]
            h = _corrupt_value(h, max_val=65535)
            struct.pack_into(">H", data, 1, h)
        elif field == 2 and len(data) >= 5:
            # Width (2 bytes big-endian)
            w = struct.unpack(">H", data[3:5])[0]
            w = _corrupt_value(w, max_val=65535)
            struct.pack_into(">H", data, 3, w)
        elif field == 3 and len(data) >= 6:
            # Number of components
            data[5] = random.choice([1, 3, 4, 0, 255])

        sof.data = bytes(data)
        return markers

    def _mutate_dht(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Corrupt Huffman table: flip bits in class/type byte or table values."""
        dht = _find_marker(markers, DHT)
        if dht is None or len(dht.data) < 2:
            return markers

        data = bytearray(dht.data)
        if random.random() < 0.3:
            # Corrupt the class/type byte
            data[0] ^= random.randint(1, 0xFF)
        else:
            # Corrupt a random value in the table
            if len(data) > 17:
                idx = random.randint(17, len(data) - 1)
                data[idx] ^= 1 << random.randint(0, 7)

        dht.data = bytes(data)
        return markers

    def _mutate_dqt(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Corrupt quantization table: flip precision bit or table values."""
        dqt = _find_marker(markers, DQT)
        if dqt is None or len(dqt.data) < 2:
            return markers

        data = bytearray(dqt.data)
        if random.random() < 0.3:
            # Corrupt the precision/table ID byte
            data[0] ^= random.randint(1, 0xFF)
        else:
            # Corrupt a random quantization value
            if len(data) > 2:
                idx = random.randint(2, len(data) - 1)
                data[idx] = random.randint(0, 255)

        dqt.data = bytes(data)
        return markers

    def _mutate_dri(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Corrupt or inject DRI (restart interval)."""
        dri = _find_marker(markers, DRI)
        if dri is not None:
            # Corrupt existing DRI
            if len(dri.data) >= 2:
                data = bytearray(dri.data)
                val = struct.unpack(">H", data[0:2])[0]
                val = _corrupt_value(val, max_val=65535)
                struct.pack_into(">H", data, 0, val)
                dri.data = bytes(data)
        else:
            # Inject a DRI marker before SOS
            sos_idx = _find_marker_index(markers, SOS)
            if sos_idx is not None:
                dri_data = struct.pack(">H", random.randint(0, 100))
                markers.insert(sos_idx, JpegMarker(marker=DRI, data=dri_data))
        return markers

    def _mutate_sos(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Corrupt SOS header: spectral selection, successive approximation."""
        sos = _find_marker(markers, SOS)
        if sos is None or len(sos.data) < 4:
            return markers

        data = bytearray(sos.data)
        field = random.randint(0, 2)
        if field == 0 and len(data) >= 1:
            # Number of components
            data[0] = random.choice([1, 3, 0, 255])
        elif field == 1 and len(data) >= 4:
            # Spectral selection start/end
            data[1] = random.randint(0, 63)
            data[2] = random.randint(0, 63)
        elif field == 2 and len(data) >= 4:
            # Successive approximation
            data[3] = random.randint(0, 15)

        sos.data = bytes(data)
        return markers

    def _mutate_app(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Corrupt APP segment data (JFIF, EXIF, etc.)."""
        app_markers = [m for m in markers if APP0 <= m.marker <= APP15]
        if not app_markers:
            return markers

        target = random.choice(app_markers)
        if len(target.data) < 2:
            return markers

        data = bytearray(target.data)
        idx = random.randint(0, len(data) - 1)
        data[idx] ^= 1 << random.randint(0, 7)
        target.data = bytes(data)
        return markers

    def _duplicate_marker(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Clone a random marker and insert it nearby."""
        # Don't duplicate SOI/EOI
        candidates = [i for i, m in enumerate(markers) if m.marker not in (SOI, EOI)]
        if not candidates:
            return markers

        src_idx = random.choice(candidates)
        src = markers[src_idx]
        # Truncate large markers to avoid blowing up size
        clone_data = src.data[: min(len(src.data), 64)]
        clone = JpegMarker(marker=src.marker, data=clone_data)
        insert_pos = min(src_idx + 1, len(markers))
        markers.insert(insert_pos, clone)
        return markers

    def _delete_marker(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Remove a random non-SOI, non-EOI, non-SOF marker."""
        candidates = [i for i, m in enumerate(markers) if m.marker not in (SOI, EOI, SOF0, SOF2)]
        if not candidates:
            return markers
        idx = random.choice(candidates)
        markers.pop(idx)
        return markers

    def _reorder_markers(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Swap two random markers (tests parser ordering tolerance)."""
        candidates = [i for i, m in enumerate(markers) if m.marker not in (SOI,)]
        if len(candidates) < 2:
            return markers
        a, b = random.sample(candidates, 2)
        markers[a], markers[b] = markers[b], markers[a]
        return markers

    def _corrupt_scan_data(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Flip bits in the entropy-coded scan data after SOS."""
        sos_idx = _find_marker_index(markers, SOS)
        if sos_idx is None:
            return markers

        # Find the scan data: everything between SOS and the next marker or EOI
        # The scan data is NOT length-prefixed — it runs until the next 0xFF byte
        # that's followed by a valid marker (not RST). In our parsed representation,
        # we don't have the raw scan data. So we corrupt the SOS data instead.
        sos = markers[sos_idx]
        if len(sos.data) > 0:
            data = bytearray(sos.data)
            # Flip a few bits
            for _ in range(random.randint(1, 4)):
                if data:
                    idx = random.randint(0, len(data) - 1)
                    data[idx] ^= 1 << random.randint(0, 7)
            sos.data = bytes(data)
        return markers

    def _add_comment(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Inject or corrupt a COM (comment) segment."""
        com = _find_marker(markers, COM)
        if com is not None:
            # Corrupt existing comment
            if com.data:
                data = bytearray(com.data)
                idx = random.randint(0, len(data) - 1)
                data[idx] = random.randint(0x20, 0x7E)  # printable ASCII
                com.data = bytes(data)
        else:
            # Inject a new comment before EOI
            comment = bytes(random.randint(0x20, 0x7E) for _ in range(random.randint(4, 32)))
            eoi_idx = _find_marker_index(markers, EOI)
            if eoi_idx is not None:
                markers.insert(eoi_idx, JpegMarker(marker=COM, data=comment))
            else:
                markers.append(JpegMarker(marker=COM, data=comment))
        return markers

    def _truncate_jpeg(self, markers: list[JpegMarker], max_len: int) -> bytes:
        """Remove EOI marker to produce a truncated JPEG."""
        markers = [m for m in markers if m.marker != EOI]
        return serialize_jpeg_markers(markers)

    def _duplicate_sof(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Insert a second SOF marker (tests header validation)."""
        sof = _find_marker(markers, SOF0)
        if sof is None:
            sof = _find_marker(markers, SOF2)
        if sof is None:
            return markers

        clone = JpegMarker(marker=sof.marker, data=sof.data)
        insert_pos = markers.index(sof) + 1
        markers.insert(insert_pos, clone)
        return markers

    def _corrupt_marker_length(self, markers: list[JpegMarker], max_len: int) -> list[JpegMarker]:
        """Write a random length to a marker (tests length validation)."""
        candidates = [m for m in markers if m.marker in LENGTH_MARKERS and len(m.data) > 0]
        if not candidates:
            return markers

        target = random.choice(candidates)
        # Replace data with random length
        new_len = random.randint(0, min(len(target.data) + 10, 256))
        target.data = target.data[:new_len]
        return markers

    def _swap_quantization_tables(
        self, markers: list[JpegMarker], max_len: int
    ) -> list[JpegMarker]:
        """Swap two quantization table entries in DQT."""
        dqt = _find_marker(markers, DQT)
        if dqt is None or len(dqt.data) < 4:
            return markers

        data = bytearray(dqt.data)
        # Find table boundaries: each table is 1 byte header + 64 or 128 values
        # Simple approach: swap two random bytes in the data portion
        if len(data) > 4:
            a = random.randint(2, len(data) - 1)
            b = random.randint(2, len(data) - 1)
            data[a], data[b] = data[b], data[a]
            dqt.data = bytes(data)
        return markers

    def _generate_random_jpeg(self, markers_or_max=None, max_len: int = 4096) -> bytes:
        """Generate a minimal random JPEG from scratch.

        Called from dispatch as _generate_random_jpeg(markers, max_len) or
        standalone as _generate_random_jpeg(max_len=N).
        """
        if isinstance(markers_or_max, int):
            max_len = markers_or_max

        buf = bytearray()
        # SOI
        buf.extend(b"\xff\xd8")

        # APP0 (JFIF)
        app_data = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        buf.extend(b"\xff\xe0")
        buf.extend(struct.pack(">H", len(app_data) + 2))
        buf.extend(app_data)

        # DQT
        precision_id = 0x00  # 8-bit
        qt = bytes(random.randint(1, 255) for _ in range(64))
        dqt_data = bytes([precision_id]) + qt
        buf.extend(b"\xff\xdb")
        buf.extend(struct.pack(">H", len(dqt_data) + 2))
        buf.extend(dqt_data)

        # SOF0
        height = random.randint(1, 256)
        width = random.randint(1, 256)
        num_components = random.choice([1, 3])
        sof_data = bytearray()
        sof_data.append(8)  # precision
        sof_data.extend(struct.pack(">H", height))
        sof_data.extend(struct.pack(">H", width))
        sof_data.append(num_components)
        for c in range(num_components):
            sof_data.append(c + 1)  # component ID
            sof_data.append(0x11)  # sampling 1x1
            sof_data.append(0x00)  # quant table 0
        buf.extend(b"\xff\xc0")
        buf.extend(struct.pack(">H", len(sof_data) + 2))
        buf.extend(sof_data)

        # DHT (AC table for component 1)
        # Build a minimal valid Huffman table: 16 counts + symbols
        counts = bytearray(16)
        symbols = bytearray()
        # Create a table with a few symbols
        num_symbols = random.randint(1, 8)
        for i in range(min(num_symbols, 16)):
            counts[i] = 1
            symbols.append(random.randint(0, 0xFF))
        dht_data = bytes([0x10]) + bytes(counts) + bytes(symbols)  # AC, ID 0
        buf.extend(b"\xff\xc4")
        buf.extend(struct.pack(">H", len(dht_data) + 2))
        buf.extend(dht_data)

        # DRI
        restart_interval = random.randint(0, 10)
        buf.extend(b"\xff\xdd")
        buf.extend(struct.pack(">H", 4))  # length = 4
        buf.extend(struct.pack(">H", restart_interval))

        # SOS
        sos_data = bytearray()
        sos_data.append(num_components)
        for c in range(num_components):
            sos_data.append(c + 1)  # component ID
            sos_data.append(0x00)  # DC table 0, AC table 0
        sos_data.append(0x00)  # spectral selection start
        sos_data.append(0x3F)  # spectral selection end
        sos_data.append(0x00)  # successive approximation
        buf.extend(b"\xff\xda")
        buf.extend(struct.pack(">H", len(sos_data) + 2))
        buf.extend(sos_data)

        # Random scan data (entropy-coded segment)
        # Avoid 0xFF bytes — the parser interprets them as marker starts
        scan_len = random.randint(1, min(256, max_len - len(buf) - 2))
        buf.extend(bytes(random.randint(0, 0xFE) for _ in range(scan_len)))

        # EOI
        buf.extend(b"\xff\xd9")

        return bytes(buf[:max_len])


def _find_marker(markers: list[JpegMarker], marker: int) -> JpegMarker | None:
    """Find first marker of given type."""
    for m in markers:
        if m.marker == marker:
            return m
    return None


def _find_marker_index(markers: list[JpegMarker], marker: int) -> int | None:
    """Find index of first marker of given type."""
    for i, m in enumerate(markers):
        if m.marker == marker:
            return i
    return None


def _corrupt_value(val: int, max_val: int = 0xFFFF) -> int:
    """Apply a random corruption to an integer value."""
    method = random.randint(0, 4)
    if method == 0:  # bit flip
        bit = random.randint(0, min(15, val.bit_length() or 1))
        return val ^ (1 << bit)
    elif method == 1:  # boundary value
        return random.choice([0, 1, max_val, max_val // 2, max_val - 1])
    elif method == 2:  # add/subtract delta
        delta = random.choice([-2, -1, 1, 2, 16, 256])
        return max(0, min(max_val, val + delta))
    elif method == 3:  # random replacement
        return random.randint(0, max_val)
    else:  # clamp to small value
        return random.randint(0, 16)
