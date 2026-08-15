"""Magic-byte-preserving format lock mutator.

Detects the format magic prefix at the start of a buffer and preserves it
while mutating the tail. Without this, byte-level corruption of magic bytes
causes avformat_open_input to select the wrong demuxer (or reject the input
entirely) before any decoder logic runs at all.

Detection runs in three layers, because not every container announces itself
with a literal byte string at offset 0:

  1. ``MAGIC_PREFIXES``   -- literal magic at offset 0, longest match wins.
  2. ``MAGIC_AT_OFFSET``  -- literal magic anchored at a non-zero offset
                             (ISO-BMFF's ``ftyp`` sits at 4, behind the u32
                             box size).
  3. masked rules         -- formats whose header is a bit pattern rather
                             than a byte string (MPEG-audio / ADTS frame
                             sync, ID3v2 syncsafe sizes). These validate a
                             few header fields before claiming a match, so
                             that arbitrary ``ff ff`` in binary noise does
                             not register as a locked format.
"""

import random

# (prefix_bytes, description) -- literal magic at offset 0, longest match wins.
#
# Deliberately *not* in this table:
#   - Enumerated ISO-BMFF ``<u32 size>ftyp`` variants. The box size is any
#     u32, so enumerating observed values is unbounded and was already dead
#     weight next to the offset-anchored rule below, which catches every
#     size including the ones nobody listed.
#   - A bare ``ftyp`` at offset 0. ``ftyp`` is a box *type*; at offset 0 it
#     is the tail of a size field, never a magic. The entry made any buffer
#     literally beginning "ftyp" report a 4-byte lock on a file that is not
#     valid ISO-BMFF.
#   - ``3c 00 00 00`` ("mp4_broken_ftyp"). 0x3c is '<'; this locked four
#     bytes of anything starting with '<' followed by three NULs and has no
#     relationship to ISO-BMFF.
MAGIC_PREFIXES: list[tuple[bytes, str]] = [
    # -- containers -------------------------------------------------------
    (b"RIFF", "avi/wav_riff"),
    (b"RF64", "wav_rf64"),
    (b"\x1a\x45\xdf\xa3", "matroska/webm"),
    (b"OggS", "ogg"),
    (b"FLV\x01", "flv"),
    (b"\x30\x26\xb2\x75\x8e\x66\xcf\x11", "wmv_asf"),
    (b"FORM", "aiff/iff"),
    (b"\x00\x00\x01\xba", "mpeg_ps"),
    (b"\x00\x00\x01\xb3", "mpeg_video_seq"),
    # -- audio ------------------------------------------------------------
    (b"fLaC", "flac"),
    (b"MAC ", "ape_monkeys_audio"),
    (b"wvpk", "wavpack"),
    (b"MPCK", "musepack_sv8"),
    (b"MP+", "musepack_sv7"),
    (b"TTA1", "true_audio"),
    (b"caff", "caf"),
    (b".snd", "au_sun"),
    (b"#!AMR", "amr"),
    (b"DSD ", "dsf"),
    (b"ADIF", "aac_adif"),
    # -- images -----------------------------------------------------------
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"BM", "bmp"),
    (b"GIF8", "gif"),
    (b"\x00\x00\x01\x00", "ico"),
    (b"II*\x00", "tiff_le"),
    (b"MM\x00*", "tiff_be"),
    # -- compression / archive --------------------------------------------
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"\x04\x22\x4d\x18", "lz4_frame"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"Rar!\x1a\x07", "rar"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    # ZIP signatures are PK followed by a *record type*. The second byte
    # varies per record; PK\x03\x06 and PK\x03\x08 (previously listed as
    # "zip64_central_dir" / "zip64_end_of_central_dir") are not signatures
    # the format defines and matched nothing real.
    (b"PK\x03\x04", "zip_local_file_header"),
    (b"PK\x01\x02", "zip_central_dir_header"),
    (b"PK\x05\x06", "zip_end_of_central_dir"),
    (b"PK\x06\x06", "zip64_end_of_central_dir"),
    (b"PK\x07\x08", "zip_data_descriptor"),
    # -- executables ------------------------------------------------------
    (b"\x7fELF", "elf"),
    (b"MZ", "pe_dos"),
]

# (offset, magic_bytes, locked_prefix_len, description)
#
# ISO-BMFF: bytes 0-3 are the u32 ftyp box size, so the "ftyp" magic lives
# at offset 4. The locked prefix spans both, keeping the size field intact
# alongside the type -- a demuxer that reads a corrupted size walks straight
# off the first box and never reaches any codec.
MAGIC_AT_OFFSET: list[tuple[int, bytes, int, str]] = [
    (4, b"ftyp", 8, "isobmff_mp4/mov/3gp"),
    (4, b"moov", 8, "isobmff_moov_first"),
    (4, b"mdat", 8, "isobmff_mdat_first"),
    (4, b"free", 8, "isobmff_free_first"),
    (4, b"skip", 8, "isobmff_skip_first"),
    (8, b"WAVE", 12, "riff_wave"),
    (8, b"AVI ", 12, "riff_avi"),
    (8, b"AIFF", 12, "iff_aiff"),
    (8, b"AIFC", 12, "iff_aifc"),
]

_ISOBMFF_INDIRECT_PREFIX_LEN = 8  # first 4 bytes (box size) + "ftyp"

# MPEG-1/2/2.5 audio and ADTS AAC both start with an 11-bit frame sync
# (0xFF, then the top three bits of the next byte). That pattern alone
# occurs constantly in binary noise, so a match additionally requires the
# layer, bitrate-index and sampling-rate fields to hold non-reserved values.
_MPEG_SYNC_LOCK_LEN = 2

_ID3V2_HEADER_LEN = 10


def _id3v2_header_len(data: bytes) -> int:
    """Length of a valid ID3v2 header, or 0.

    ID3v2 is 10 bytes: "ID3", version major/revision, flags, then a
    four-byte syncsafe size (each byte < 0x80). Validating the syncsafe
    bytes keeps arbitrary text beginning "ID3" from registering.
    """
    if len(data) < _ID3V2_HEADER_LEN or not data.startswith(b"ID3"):
        return 0
    # Only ID3v2.2/2.3/2.4 exist. Accepting "anything but 0xFF" here (as an
    # earlier revision of this check did) let ordinary prose beginning "ID3 "
    # register as a tagged MP3, because a space is a legal version byte and
    # ASCII always satisfies the syncsafe test below.
    if data[3] not in (2, 3, 4) or data[4] == 0xFF:
        return 0
    if data[5] & 0x0F:  # low flag bits are undefined in every published version
        return 0
    if any(b & 0x80 for b in data[6:10]):  # syncsafe integers have bit7 clear
        return 0
    return _ID3V2_HEADER_LEN


def _mpeg_audio_sync_len(data: bytes) -> int:
    """Length of an MPEG-audio / ADTS frame sync, or 0."""
    if len(data) < 2 or data[0] != 0xFF:
        return 0
    b1 = data[1]
    if (b1 & 0xE0) != 0xE0:  # 11-bit sync not present
        return 0
    version = (b1 >> 3) & 0x03
    layer = (b1 >> 1) & 0x03
    if version == 0x01:  # reserved MPEG version ID
        return 0
    if layer == 0x00:
        # layer==0 is reserved for MPEG audio, but is how ADTS AAC encodes
        # itself. Accept it only for the MPEG-2/MPEG-4 version IDs ADTS uses.
        if version not in (0x02, 0x03):
            return 0
    elif len(data) >= 3:
        if (data[2] >> 4) == 0x0F:  # reserved bitrate index
            return 0
        if ((data[2] >> 2) & 0x03) == 0x03:  # reserved sampling rate
            return 0
    return _MPEG_SYNC_LOCK_LEN


def detect_magic(data: bytes) -> tuple[int, str | None]:
    """Detect the magic prefix at the start of *data*.

    Returns ``(prefix_len, format_name)``; ``(0, None)`` when nothing
    matches. Longest match across all three detection layers wins.
    """
    if not data:
        return 0, None

    best = 0
    name: str | None = None

    for prefix, fmt in MAGIC_PREFIXES:
        if len(prefix) > best and data.startswith(prefix):
            best, name = len(prefix), fmt

    for offset, magic, lock_len, fmt in MAGIC_AT_OFFSET:
        end = offset + len(magic)
        if lock_len > best and len(data) >= end and data[offset:end] == magic:
            best, name = lock_len, fmt

    id3 = _id3v2_header_len(data)
    if id3 > best:
        best, name = id3, "mp3_id3v2"

    sync = _mpeg_audio_sync_len(data)
    if sync > best:
        best, name = sync, "mpeg_audio_sync"

    return best, name


def detect_magic_prefix(data: bytes) -> int:
    """Detect the longest magic prefix at the start of *data*.

    Returns the byte length of the matching prefix (0 for no match).
    """
    return detect_magic(data)[0]


def format_lock_havoc(data: bytes, max_len: int, rng: random.Random | None = None) -> bytes | None:
    """Preserve magic prefix and apply aggressive byte mutations to the tail.

    If a recognized magic prefix is found, it is kept intact while the
    remaining bytes (after the prefix) receive 8-16 single-byte havoc
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
