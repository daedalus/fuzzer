"""Structure-aware Shorten mutations.

Parses Shorten (.shn) audio format structure (frame headers, CRC, linear prediction).
Targets frame header corruption and linear prediction coefficient manipulation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool


@dataclass
class ShnFrameHeader:
    """Shorten frame header structure."""

    sample_rate: int
    bits_per_sample: int
    channels: int
    frame_samples: int
    frame_crc: int
    lp_order: int


def parse_shn_frame_header(data: bytes) -> ShnFrameHeader | None:
    """Parse Shorten frame header from data."""
    if len(data) < 10:
        return None

    sample_rate = struct.unpack_from("<H", data, 0)[0]
    bits_per_sample = struct.unpack_from("B", data, 2)[0]
    channels = struct.unpack_from("B", data, 3)[0]
    frame_samples = struct.unpack_from("<H", data, 4)[0]
    frame_crc = struct.unpack_from("<H", data, 6)[0]
    lp_order = struct.unpack_from("B", data, 8)[0]

    return ShnFrameHeader(
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        channels=channels,
        frame_samples=frame_samples,
        frame_crc=frame_crc,
        lp_order=lp_order,
    )


class ShnMutator:
    """Structure-aware Shorten mutator.

    Targets frame header corruption and linear prediction coefficient manipulation.
    """

    def __init__(self, seed=None):
        # One pool per mutator, built once. Callers that own a pool pass it
        # as ``rng=`` and it wins for that call; this is the standalone
        # default, never the stdlib module (Hard Rule 16).
        rng = RandPool(seed=seed)
        self._rng = rng

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one Shorten-specific mutation."""
        self._rng = rng or self._rng
        frame_header = parse_shn_frame_header(data)
        if not frame_header:
            return self._generate_random_shn(max_len=max_len, rng=self._rng)

        op = self._rng.randint(0, 4)
        mutators = [
            self._mutate_frame_header,
            self._corrupt_crc,
            self._mutate_lp_coeffs,
            self._shift_samples,
            # The generator replaces the input rather than editing it, so it
            # takes neither the buffer nor the parsed frame header the other entries
            # do. Adapted here rather than given vestigial `_data`/`_frame_header`
            # parameters: that placeholder shape is exactly what f5435af had
            # to unpick across ten generators, where a positional `max_len`
            # landed in the placeholder and the generator silently fell back
            # to its own default.
            lambda _data, _frame_header, max_len: self._generate_random_shn(
                max_len=max_len, rng=self._rng
            ),
        ]
        result = mutators[op](data, frame_header, max_len)
        return result[:max_len]

    def _mutate_frame_header(self, data: bytes, header: ShnFrameHeader, max_len: int) -> bytes:
        """Mutate frame header fields."""
        raw = bytearray(data)
        op = self._rng.randint(0, 5)

        if op == 0 and header.sample_rate < 0xFFFF:
            struct.pack_into("<H", raw, 0, header.sample_rate + self._rng.choice([1, 2, 4, 8]))
        elif op == 1 and header.bits_per_sample < 32:
            struct.pack_into(
                "B", raw, 2, min(32, header.bits_per_sample + self._rng.choice([1, 2, 4, 8]))
            )
        elif op == 2:
            struct.pack_into("B", raw, 3, header.channels + self._rng.choice([0, 1, 2]))
        elif op == 3:
            struct.pack_into(
                "<H",
                raw,
                4,
                header.frame_samples + self._rng.choice([-128, -64, -32, 0, 32, 64, 128]),
            )
        elif op == 4:
            struct.pack_into("<H", raw, 6, header.frame_crc + self._rng.choice([-1, 1]))
        elif op == 5:
            struct.pack_into("B", raw, 8, min(16, header.lp_order + self._rng.choice([0, 1, 2])))

        return bytes(raw[:max_len])

    def _corrupt_crc(self, data: bytes, header: ShnFrameHeader, max_len: int) -> bytes:
        """Corrupt frame CRC to cause decoder rejection."""
        raw = bytearray(data)
        struct.pack_into("<H", raw, 6, 0xFFFF)
        return bytes(raw[:max_len])

    def _mutate_lp_coeffs(self, data: bytes, header: ShnFrameHeader, max_len: int) -> bytes:
        """Mutate linear prediction coefficients after header."""
        raw = bytearray(data)
        coeff_offset = 10
        coeff_size = header.lp_order * 2  # 16-bit signed coefficients

        if coeff_offset + coeff_size <= len(raw):
            for i in range(min(4, header.lp_order)):
                pos = coeff_offset + i * 2
                if pos + 2 <= len(raw):
                    struct.pack_into("<h", raw, pos, self._rng.randint(-32768, 32767))

        return bytes(raw[:max_len])

    def _shift_samples(self, data: bytes, header: ShnFrameHeader, max_len: int) -> bytes:
        """Shift audio sample data to break synchronization."""
        raw = bytearray(data)
        header_size = 10
        coeff_size = header.lp_order * 2
        sample_offset = header_size + coeff_size

        if sample_offset < len(raw):
            shift_amount = self._rng.randint(1, min(4, len(raw) - sample_offset))
            del raw[sample_offset : sample_offset + shift_amount]
            raw.extend(b"\x00" * shift_amount)

        return bytes(raw[:max_len])

    def _generate_random_shn(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate minimal Shorten frame with corrupt header."""
        self._rng = rng or self._rng
        result = bytearray()
        result += struct.pack("<H", 0xFFFF)  # sample_rate (signed overflow)
        result += struct.pack("B", 0xFF)  # bits_per_sample (overflow)
        result += struct.pack("B", 0x02)  # channels (stereo)
        result += struct.pack("<H", 0x1000)  # frame_samples
        result += struct.pack("<H", 0xBEEF)  # frame_crc
        result += struct.pack("B", 0x10)  # lp_order

        return bytes(result[:max_len])


__all__ = ["parse_shn_frame_header", "parse_shorten", "ShnMutator", "ShnFrameHeader"]

# Alias used by the operator handler.
parse_shorten = parse_shn_frame_header
