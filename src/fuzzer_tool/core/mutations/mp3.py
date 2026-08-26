"""Structure-aware MP3 (MPEG-1/2 Layer III, and Layer I/II) frame mutator.

Like adts.py, MP3 frames are back-to-back with no container framing of
their own -- the 11-bit syncword at the start of each 4-byte header is
the only delimiter, and this mutator scans for it the same way (rather
than trusting a frame-length computed from the bitrate/sample-rate
tables, which is itself sensitive to fields under mutation here).

MP3 frame header (4 bytes, MSB first):

  byte0:  syncword bits 10-3           (0xFF)
  byte1:  syncword bits 2-0 (111) | version(2) | layer(2) | protection_bit(1)
  byte2:  bitrate_index(4) | sampling_rate_index(2) | padding_bit(1) | private_bit(1)
  byte3:  channel_mode(2) | mode_extension(2) | copyright(1) | original(1) | emphasis(2)

version: 00=MPEG2.5, 01=reserved, 10=MPEG2, 11=MPEG1
layer:   00=reserved, 01=Layer III, 10=Layer II, 11=Layer I
bitrate_index and sampling_rate_index each have a reserved code
(0xF and 0x3 respectively) that maps to no valid rate in the spec
tables -- exactly the kind of decoder table-lookup edge this mutator
targets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

MP3_HEADER_LEN = 4

RESERVED_BITRATE_INDEX = 0x0F
RESERVED_SAMPLING_RATE_INDEX = 0x03
RESERVED_VERSION = 0x01
RESERVED_LAYER = 0x00


@dataclass
class Mp3Frame:
    header: bytes  # 4 bytes
    payload: bytes = b""

    @property
    def version(self) -> int:
        return (self.header[1] >> 3) & 0x03

    @property
    def layer(self) -> int:
        return (self.header[1] >> 1) & 0x03

    @property
    def bitrate_index(self) -> int:
        return (self.header[2] >> 4) & 0x0F

    @property
    def sampling_rate_index(self) -> int:
        return (self.header[2] >> 2) & 0x03

    def to_bytes(self) -> bytes:
        return self.header + self.payload


def _is_sync(data: bytes, pos: int) -> bool:
    """11-bit syncword plus non-reserved version/layer fields, so this
    doesn't false-trigger on ADTS's coincidentally-similar all-1s prefix
    (ADTS fixes its layer-equivalent field to the value MP3 reserves)."""
    n = len(data)
    if pos + 1 >= n or data[pos] != 0xFF or (data[pos + 1] & 0xE0) != 0xE0:
        return False
    version = (data[pos + 1] >> 3) & 0x03
    layer = (data[pos + 1] >> 1) & 0x03
    return version != RESERVED_VERSION and layer != RESERVED_LAYER


def parse_mp3_frames(data: bytes) -> list[Mp3Frame] | None:
    """Scan for back-to-back MP3 frames by syncword. Returns None if no
    syncword is found or fewer than MP3_HEADER_LEN bytes follow it."""
    n = len(data)
    if n < MP3_HEADER_LEN or not _is_sync(data, 0):
        return None

    frames: list[Mp3Frame] = []
    pos = 0
    while pos + MP3_HEADER_LEN <= n:
        header = data[pos : pos + MP3_HEADER_LEN]
        search_start = pos + MP3_HEADER_LEN
        next_pos = search_start
        while next_pos < n and not _is_sync(data, next_pos):
            next_pos += 1
        payload = data[search_start:next_pos]
        frames.append(Mp3Frame(header=header, payload=payload))
        pos = next_pos

    return frames if frames else None


def serialize_mp3_frames(frames: list[Mp3Frame]) -> bytes:
    return b"".join(f.to_bytes() for f in frames)


class Mp3Mutator:
    """Structure-aware MP3 frame mutator."""

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        rng = rng or random
        frames = parse_mp3_frames(data)
        if not frames:
            return self._generate_random_mp3(max_len=max_len, rng=rng)

        op = rng.randint(0, 7)
        mutators = [
            self._mutate_bitrate_index,
            self._mutate_sampling_rate_index,
            self._mutate_version,
            self._mutate_layer,
            self._mutate_channel_mode,
            self._duplicate_frame,
            self._delete_frame,
            self._reorder_frames,
        ]
        mutators[op](frames, rng)
        return serialize_mp3_frames(frames)[:max_len]

    def _mutate_bitrate_index(self, frames: list[Mp3Frame], rng) -> None:
        """Set to the reserved 0xF index or another value -- the classic
        "free bitrate" / invalid-index decoder edge case."""
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        new_idx = rng.choice([RESERVED_BITRATE_INDEX, 0, rng.randint(0, 0x0F)])
        hdr[2] = (hdr[2] & 0x0F) | ((new_idx & 0x0F) << 4)
        target.header = bytes(hdr)

    def _mutate_sampling_rate_index(self, frames: list[Mp3Frame], rng) -> None:
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        new_idx = rng.choice([RESERVED_SAMPLING_RATE_INDEX, 0, rng.randint(0, 0x03)])
        hdr[2] = (hdr[2] & 0xF3) | ((new_idx & 0x03) << 2)
        target.header = bytes(hdr)

    def _mutate_version(self, frames: list[Mp3Frame], rng) -> None:
        """Flip MPEG version mid-stream (e.g. MPEG1 <-> MPEG2), which
        changes the sampling-rate table the decoder should be indexing
        into without touching the sampling_rate_index value itself."""
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        new_version = rng.choice([0, 2, 3])  # skip the reserved 1
        hdr[1] = (hdr[1] & 0xE7) | ((new_version & 0x03) << 3)
        target.header = bytes(hdr)

    def _mutate_layer(self, frames: list[Mp3Frame], rng) -> None:
        """Relabel Layer I/II/III -- tests layer-dispatch confusion the
        same way nal.py's nal_unit_type mutation does for NAL types."""
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        new_layer = rng.choice([1, 2, 3])  # skip the reserved 0
        hdr[1] = (hdr[1] & 0xF9) | ((new_layer & 0x03) << 1)
        target.header = bytes(hdr)

    def _mutate_channel_mode(self, frames: list[Mp3Frame], rng) -> None:
        target = rng.choice(frames)
        hdr = bytearray(target.header)
        hdr[3] = (hdr[3] & 0x3F) | ((rng.randint(0, 3) & 0x03) << 6)
        target.header = bytes(hdr)

    def _duplicate_frame(self, frames: list[Mp3Frame], rng) -> None:
        idx = rng.randint(0, len(frames) - 1)
        orig = frames[idx]
        frames.insert(idx + 1, Mp3Frame(header=orig.header, payload=orig.payload))

    def _delete_frame(self, frames: list[Mp3Frame], rng) -> None:
        if len(frames) > 1:
            frames.pop(rng.randint(0, len(frames) - 1))

    def _reorder_frames(self, frames: list[Mp3Frame], rng) -> None:
        if len(frames) >= 2:
            i, j = rng.sample(range(len(frames)), 2)
            frames[i], frames[j] = frames[j], frames[i]

    def _generate_random_mp3(self, max_len: int = 65536, rng=None) -> bytes:
        """Minimal single-frame MPEG1 Layer III, 128kbps, 44.1kHz, stereo."""
        rng = rng or random
        payload = bytes(rng.randint(0, 255) for _ in range(rng.randint(64, 128)))

        version = 3  # MPEG1
        layer = 1  # Layer III
        bitrate_index = 9  # 128 kbps for MPEG1 Layer III
        sampling_rate_index = 0  # 44100 Hz
        channel_mode = 1  # joint stereo

        hdr = bytearray(4)
        hdr[0] = 0xFF
        hdr[1] = 0xE0 | (version << 3) | (layer << 1) | 0x01  # protection_bit=1 (no CRC)
        hdr[2] = (bitrate_index << 4) | (sampling_rate_index << 2)
        hdr[3] = channel_mode << 6
        frame = Mp3Frame(header=bytes(hdr), payload=payload)
        return frame.to_bytes()[:max_len]
