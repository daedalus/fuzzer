"""Poly1305 tag repair for authenticated-container targets (``--poly1305-key``).

A target that checks a Poly1305 tag rejects every mutated message before the
parser behind it runs.  Given the one-time key (``r || s``, 32 bytes), the tag
is a pure function of the message, so we mutate the message and recompute the
tag in place.  Without the key the operator cannot work; it is only offered
when the user supplies one.

Scope: raw Poly1305 over the message (RFC 8439 section 2.5).  The ChaCha20-
Poly1305 AEAD construction (AAD padding, length block, per-message key from
ChaCha20) is not modelled.  GHASH is out of scope (needs AES).  Stdlib only
(Hard Rule 51).
"""

from __future__ import annotations

#: Poly1305 prime 2^130 - 5.
_PRIME = (1 << 130) - 5
#: Clamp mask applied to ``r`` (RFC 8439 section 2.5.1).
_R_CLAMP = 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
_BLOCK = 16
TAG_LEN = 16
KEY_LEN = 32
#: Default tag position: the last 16 bytes.
DEFAULT_TAG_AT = -TAG_LEN
_MOD_128 = 1 << 128
#: Message edits: flip bits, insert a byte, delete a byte.
_OP_FLIP, _OP_INSERT, _OP_DELETE = 0, 1, 2
_MAX_FLIPS = 4
_BITS = 8


def poly1305_mac(key: bytes, msg: bytes) -> bytes:
    """Return the 16-byte Poly1305 tag of *msg* under the 32-byte *key*."""
    if len(key) != KEY_LEN:
        raise ValueError(f"Poly1305 key must be {KEY_LEN} bytes, got {len(key)}")
    r = int.from_bytes(key[:16], "little") & _R_CLAMP
    s = int.from_bytes(key[16:], "little")
    acc = 0
    for i in range(0, len(msg), _BLOCK):
        block = msg[i : i + _BLOCK]
        # Each block gets a 0x01 byte appended (the 2^(8*len) bit).
        n = int.from_bytes(block, "little") + (1 << (_BITS * len(block)))
        acc = (acc + n) * r % _PRIME
    return ((acc + s) % _MOD_128).to_bytes(TAG_LEN, "little")


def parse_key(text: str) -> bytes:
    """Parse a 64-hex-digit key; ValueError on anything else."""
    try:
        key = bytes.fromhex(text.strip())
    except ValueError as exc:
        raise ValueError(f"Poly1305 key is not hex: {exc}") from exc
    if len(key) != KEY_LEN:
        raise ValueError(f"Poly1305 key must be {KEY_LEN} bytes ({KEY_LEN * 2} hex digits)")
    return key


def tag_start(total: int, tag_at: int) -> int | None:
    """Offset of the tag in a *total*-byte buffer, or None if it does not fit.

    ``tag_at >= 0`` counts from the start, negative from the end.
    """
    start = tag_at if tag_at >= 0 else total + tag_at
    if start < 0 or start + TAG_LEN > total:
        return None
    return start


def message_of(data: bytes, tag_at: int = DEFAULT_TAG_AT) -> bytes | None:
    """Bytes outside the tag slot, in order; None if the tag does not fit."""
    start = tag_start(len(data), tag_at)
    if start is None:
        return None
    return data[:start] + data[start + TAG_LEN :]


def _assemble(msg: bytes, key: bytes, tag_at: int) -> bytes | None:
    """Place the tag of *msg* at the slot implied by the new total length."""
    start = tag_start(len(msg) + TAG_LEN, tag_at)
    if start is None:
        return None
    return msg[:start] + poly1305_mac(key, msg) + msg[start:]


def retag(data: bytes, key: bytes, tag_at: int = DEFAULT_TAG_AT) -> bytes | None:
    """Recompute the tag of *data* without touching the message."""
    msg = message_of(data, tag_at)
    if msg is None:
        return None
    return _assemble(msg, key, tag_at)


def _edit(msg: bytearray, rng, room: int) -> None:
    """Flip bits, insert a byte, or delete a byte (insert needs *room*)."""
    op = rng.randint(_OP_FLIP, _OP_DELETE)
    if op == _OP_INSERT and room > 0:
        msg.insert(rng.randint(0, len(msg)), rng.randint(0, 255))
        return
    if op == _OP_DELETE and msg:
        del msg[rng.randint(0, len(msg) - 1)]
        return
    if not msg:
        return
    for _ in range(rng.randint(1, min(_MAX_FLIPS, len(msg)))):
        msg[rng.randint(0, len(msg) - 1)] ^= 1 << rng.randint(0, _BITS - 1)


def poly1305_fix(
    data: bytes, key: bytes, rng, max_len: int, tag_at: int = DEFAULT_TAG_AT
) -> bytes | None:
    """Mutate the message part of *data* and rewrite a valid tag.

    Args:
        data: Seed bytes laid out as message plus a 16-byte tag at *tag_at*.
        key: 32-byte one-time key (``r || s``).
        rng: ``RandPool`` (or ScriptedRng in tests).
        max_len: Output length cap (an insert is skipped at the cap).
        tag_at: Tag offset; negative counts from the end.

    Returns:
        New buffer, or None when the tag slot does not fit in *data* or in
        the edited result (nothing to repair).
    """
    msg = message_of(data, tag_at)
    if msg is None:
        return None
    edited = bytearray(msg)
    _edit(edited, rng, max_len - len(data))
    return _assemble(bytes(edited), key, tag_at)
