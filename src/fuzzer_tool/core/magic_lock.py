"""Magic-byte-preserving format lock mutator.

Detects the format magic prefix at the start of a buffer and preserves it
while mutating the tail. Without this, byte-level corruption of magic bytes
causes avformat_open_input to select the wrong demuxer (or reject the input
entirely) before any decoder logic runs at all.
"""

import random

# (prefix_bytes, description) — longest matching prefix wins.
# Covers the most common FFmpeg auto-probed container formats.
MAGIC_PREFIXES: list[tuple[bytes, str]] = [
    (b"\x00\x00\x00\x1cftyp", "mp4_ftyp_v4_box32"),
    (b"\x00\x00\x00 ftyp", "mp4_ftyp_v4_box36"),
    (b"RIFF", "avi/wav_riff"),
    (b"\x1a\x45\xdf\xa3", "matroska/webm"),
    (b"OggS", "ogg"),
    (b"FLV\x01", "flv"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"BM", "bmp"),
    (b"GIF8", "gif"),
    (b"\x1f\x8b", "gzip"),
    (b"fLaC", "flac"),
    (b"PK\x03\x04", "zip/zip-based"),
    (b"PK\x03\x06", "zip64_central_dir"),
    (b"PK\x03\x08", "zip64_end_of_central_dir"),
    (b"MZ", "pe_dos"),
    (b"\xff\xfb", "mp3_mpeg"),
    (b"\xff\xf3", "mp3_mpeg_alt"),
    (b"\xff\xf2", "mp3_mpeg_alt2"),
    (b"\x42\x4d", "bmp_alt"),
    (b"\x00\x00\x01\x00", "ico"),
    (b"Rar!\x1a\x07", "rar"),
    (b"\x37\x7a\xbc\xaf\x27\x1c", "7z"),
    (b"BZh", "bzip2"),
    (b"\xfd\x37\x7a\x58\x5a\x00", "xz"),
    (b"MAC ", "ape_monkeys_audio"),
    (b"FORM", "aiff"),
    (b"\x30\x26\xb2\x75\x8e\x66\xcf\x11", "wmv_asf"),
    (b"\x00\x00\x00\x14ftyp", "mp4_ftyp_v4_box20"),
    (b"\x00\x00\x00\x18ftyp", "mp4_ftyp_v4_box24"),
    (b"\x00\x00\x00\x0cftyp", "mp4_ftyp_v4_box12"),
    (b"\x00\x00\x00\x10ftyp", "mp4_ftyp_v4_box16"),
    (b"\x00\x00\x00\x0fftyp", "mp4_ftyp_v4_box15"),
    (b"ftyp", "mp4_ftyp_offset4"),
    (b"<\x00\x00\x00", "mp4_broken_ftyp"),
    (b"\x00\x00\x00\x00ftyp", "mp4_ftyp_v4_box0"),
]

# -------- MP4/ISOBMFF ftyp at offsets ---------------------------------------
# The first 4 bytes of an MP4 file are the ftyp box size, so the magic
# "ftyp" is at offset 4-7, not 0.  We handle this separately so the
# generic prefix matcher still sees the size-bytes as part of the locked
# prefix.
_ISOBMFF_INDIRECT_PREFIX_LEN = 8  # first 4 bytes (box size) + "ftyp"


def detect_magic_prefix(data: bytes) -> int:
    """Detect the longest magic prefix at the start of *data*.

    Returns the byte length of the matching prefix (0 for no match).
    """
    if not data:
        return 0

    best = 0
    for prefix, _name in MAGIC_PREFIXES:
        if data.startswith(prefix):
            if len(prefix) > best:
                best = len(prefix)

    # Check ISOBMFF indirect pattern: [u32 size][ftyp...]
    if len(data) >= 8 and data[4:8] == b"ftyp":
        if best < 8:
            best = 8

    return best


def format_lock_havoc(data: bytes, max_len: int, rng: random.Random | None = None) -> bytes | None:
    """Preserve magic prefix and apply aggressive byte mutations to the tail.

    If a recognized magic prefix is found, it is kept intact while the
    remaining bytes (after the prefix) receive 8–16 single-byte havoc
    mutations (the same escalation used during stall recovery).

    Returns mutated bytes (truncated to *max_len*), or *None* when no
    prefix is detected (no-op).
    """
    rng = rng or random
    prefix_len = detect_magic_prefix(data)
    if prefix_len == 0:
        return None

    tail = bytearray(data[prefix_len:])
    if not tail:
        return None

    n = rng.randint(8, 16)
    for _ in range(n):
        idx = rng.randint(0, len(tail) - 1)
        tail[idx] ^= 1 << rng.randint(0, 7)

    result = data[:prefix_len] + bytes(tail)
    return result[:max_len]
