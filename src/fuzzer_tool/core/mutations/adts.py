"""Structure-aware ADTS (Audio Data Transport Stream) AAC mutator.

ADTS is the raw AAC elementary-stream framing used outside an ISOBMFF/MP4
container (e.g. `.aac` files, some broadcast and streaming pipelines).
Unlike nal.py's start codes, ADTS frames are back-to-back with no
delimiter between them -- the only way to find the next frame is the
12-bit syncword each frame header opens with, so this mutator scans for
that syncword the same way nal.py scans for 0x000001, rather than
trusting the header's own aac_frame_length field (which is one of the
fields under mutation, and would desync the scanner on its own output if
trusted as an authority).

ADTS header (7-byte fixed+variable header; CRC is not modelled -- see
below):

  byte0:    syncword bits 11-4        (0xFF)
  byte1:    syncword bits 3-0 (1111) | ID(1) | layer(2, always 00) | protection_absent(1)
  byte2:    profile(2) | sampling_frequency_index(4) | private_bit(1) | channel_configuration bit2(1)
  byte3:    channel_configuration bits1-0(2) | original_copy(1) | home(1)
            | copyright_id_bit(1) | copyright_id_start(1) | aac_frame_length bits12-11(2)
  byte4:    aac_frame_length bits10-3(8)
  byte5:    aac_frame_length bits2-0(3) | buffer_fullness bits10-6(5)
  byte6:    buffer_fullness bits5-0(6) | number_of_raw_data_blocks_in_frame(2)

If protection_absent is 0 a 2-byte CRC follows; this mutator doesn't
track that split and instead treats everything after the 7-byte header
up to the next syncword as one opaque payload blob (CRC included when
present), matching nal.py's choice not to interpret RBSP contents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

ADTS_HEADER_LEN = 7  # fixed + variable header, CRC (if any) folds into payload

# sampling_frequency_index: 0-12 valid, 13-14 reserved, 15 = explicit freq (not used in ADTS)
SAMPLING_FREQ_VALID = list(range(13))
WEIRD_SAMPLING_FREQ = [13, 14, 15]


@dataclass
class AdtsFrame:
    header: bytes  # 7 bytes
    payload: bytes = b""

    @property
    def protection_absent(self) -> int:
        return self.header[1] & 0x01

    @property
    def profile(self) -> int:
        return (self.header[2] >> 6) & 0x03

    @property
    def sampling_frequency_index(self) -> int:
        return (self.header[2] >> 2) & 0x0F

    @property
    def channel_configuration(self) -> int:
        return ((self.header[2] & 0x01) << 2) | ((self.header[3] >> 6) & 0x03)

    @property
    def frame_length(self) -> int:
        return ((self.header[3] & 0x03) << 11) | (self.header[4] << 3) | ((self.header[5] >> 5) & 0x07)

    def to_bytes(self) -> bytes:
        return self.header + self.payload


def _is_sync(data: bytes, pos: int) -> bool:
    """12-bit syncword (0xFFF) plus the fixed layer field (always 0)."""
    n = len(data)
    return pos + 1 < n and data[pos] == 0xFF and (data[pos + 1] & 0xF6) == 0xF0


def parse_adts_frames(data: bytes) -> list[AdtsFrame] | None:
    """Scan for back-to-back ADTS frames by syncword, not by the declared
    frame_length field. Returns None if no syncword is found or fewer
    than ADTS_HEADER_LEN bytes follow the last one."""
    n = len(data)
    if n < ADTS_HEADER_LEN or not _is_sync(data, 0):
        return None

    frames: list[AdtsFrame] = []
    pos = 0
    while pos + ADTS_HEADER_LEN <= n:
        header = data[pos : pos + ADTS_HEADER_LEN]
        search_start = pos + ADTS_HEADER_LEN
        next_pos = search_start
        while next_pos < n and not _is_sync(data, next_pos):
            next_pos += 1
        payload = data[search_start:next_pos]
        frames.append(AdtsFrame(header=header, payload=payload))
        pos = next_pos

    return frames if frames else None


def serialize_adts_frames(frames: list[AdtsFrame]) -> bytes:
    return b"".join(f.to_bytes() for f in frames)


class AdtsMutator:
    """Structure-aware ADTS AAC frame mutator."""

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        frames = parse_adts_frames(data)
        if not frames:
            return self._generate_random_adts(max_len=max_len, rng=rng)

        op = rng.randint(0, 7)
        mutators = [
            self._mutate_frame_length,
            self._mutate_sampling_frequency_index,
            self._mutate_profile,
            self._mutate_channel_configuration,
            self._mutate_protection_absent,
            self._duplicate_frame,
            self._delete_frame,
            self._reorder_frames,
        ]
        mutators[op](frames, rng)
        return serialize_adts_frames(frames)[:max_len]

    def _mutate_frame_length(self, frames: list[AdtsFrame], rng) -> None:
        """Corrupt the 13-bit aac_frame_length field independent of the
        actual scanned frame size -- the declared-vs-actual mismatch a
        length-trusting decoder would misparse."""
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        bogus = rng.choice([0, 0x1FFF, len(target.to_bytes()) + 1, rng.randint(0, 0x1FFF)])
        hdr[3] = (hdr[3] & 0xFC) | ((bogus >> 11) & 0x03)
        hdr[4] = (bogus >> 3) & 0xFF
        hdr[5] = (hdr[5] & 0x1F) | ((bogus & 0x07) << 5)
        target.header = bytes(hdr)

    def _mutate_sampling_frequency_index(self, frames: list[AdtsFrame], rng) -> None:
        """Set to a reserved index (13/14) or another valid-but-different
        rate -- tests the decoder's sample-rate table lookup bounds."""
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        new_idx = rng.choice(WEIRD_SAMPLING_FREQ + SAMPLING_FREQ_VALID)
        hdr[2] = (hdr[2] & 0xC3) | ((new_idx & 0x0F) << 2)
        target.header = bytes(hdr)

    def _mutate_profile(self, frames: list[AdtsFrame], rng) -> None:
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        hdr[2] = (hdr[2] & 0x3F) | ((rng.randint(0, 3) & 0x03) << 6)
        target.header = bytes(hdr)

    def _mutate_channel_configuration(self, frames: list[AdtsFrame], rng) -> None:
        """channel_configuration=0 means "defined in the AAC payload
        itself" -- forcing that on a stream that never carries a PCE is a
        cheap, decoder-relevant edge case alongside plain out-of-range
        values."""
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        new_cfg = rng.choice([0, 7, rng.randint(0, 7)])
        hdr[2] = (hdr[2] & 0xFE) | ((new_cfg >> 2) & 0x01)
        hdr[3] = (hdr[3] & 0x3F) | ((new_cfg & 0x03) << 6)
        target.header = bytes(hdr)

    def _mutate_protection_absent(self, frames: list[AdtsFrame], rng) -> None:
        """Flip whether a CRC is declared present, without adjusting
        payload framing to match -- tests CRC-presence branch handling
        against payload bytes that don't agree with it."""
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        hdr[1] ^= 0x01
        target.header = bytes(hdr)

    def _duplicate_frame(self, frames: list[AdtsFrame], rng) -> None:
        idx = rng.randint(0, len(frames) - 1)
        orig = frames[idx]
        frames.insert(idx + 1, AdtsFrame(header=orig.header, payload=orig.payload))

    def _delete_frame(self, frames: list[AdtsFrame], rng) -> None:
        if len(frames) > 1:
            frames.pop(rng.randint(0, len(frames) - 1))

    def _reorder_frames(self, frames: list[AdtsFrame], rng) -> None:
        if len(frames) >= 2:
            i, j = rng.sample(range(len(frames)), 2)
            frames[i], frames[j] = frames[j], frames[i]

    def _generate_random_adts(self, max_len: int = 65536, rng=None) -> bytes:
        """Minimal single-frame AAC-LC, 44.1kHz, stereo stream."""
        rng = rng or random
        payload = bytes(rng.randint(0, 255) for _ in range(rng.randint(32, 96)))
        frame_len = ADTS_HEADER_LEN + len(payload)

        profile = 1  # AAC LC
        sampling_freq_index = 4  # 44100 Hz
        channel_config = 2  # stereo

        hdr = bytearray(7)
        hdr[0] = 0xFF
        hdr[1] = 0xF1  # syncword low nibble=1111, ID=0, layer=00, protection_absent=1
        hdr[2] = ((profile & 0x03) << 6) | ((sampling_freq_index & 0x0F) << 2) | ((channel_config >> 2) & 0x01)
        hdr[3] = ((channel_config & 0x03) << 6) | ((frame_len >> 11) & 0x03)
        hdr[4] = (frame_len >> 3) & 0xFF
        hdr[5] = ((frame_len & 0x07) << 5) | 0x1F
        hdr[6] = 0xFC
        frame = AdtsFrame(header=bytes(hdr), payload=payload)
        return frame.to_bytes()[:max_len]
