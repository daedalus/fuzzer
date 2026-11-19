"""ID3v2 tag framing (id3.org v2.4 §3.1).

Most `.mp3` files, and some `.aac`, open with an ID3v2 tag before the
first audio frame. Frame mutators skip it so the sniffer and parser see
the audio, and pass it through untouched.

  "ID3" | version(1) | revision(1) | flags(1) | size(4, syncsafe)
  [body: size bytes] [footer: 10 bytes, when flags & 0x10]
"""

from __future__ import annotations

ID3_MAGIC = b"ID3"
ID3_HEADER_LEN = 10
ID3_FOOTER_FLAG = 0x10
SYNCSAFE_MASK = 0x80  # each size byte carries 7 bits; the top bit is 0


def id3v2_len(data: bytes) -> int:
    """Length of a leading ID3v2 tag, or 0 when absent, malformed or truncated.

    Example: ``b"ID3\\x04\\x00\\x00\\x00\\x00\\x00\\x20" + 32 bytes`` -> 42.
    """
    if len(data) < ID3_HEADER_LEN or data[:3] != ID3_MAGIC:
        return 0

    size_bytes = data[6:ID3_HEADER_LEN]
    if any(b & SYNCSAFE_MASK for b in size_bytes):
        return 0

    size = 0
    for b in size_bytes:
        size = (size << 7) | b

    footer = ID3_HEADER_LEN if data[5] & ID3_FOOTER_FLAG else 0
    total = ID3_HEADER_LEN + size + footer
    return total if total <= len(data) else 0
