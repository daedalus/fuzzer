"""Structure-aware base64 frame mutator (header + base64 payload).

Input layout follows targets/libbase64_read.c:

    byte 0       mode (low 2 bits): 0 decode, 1 encode,
                 2 stream decode, 3 stream encode
    byte 1       chunk-size seed (stream modes only)
    rest         payload: base64 text (decode modes), raw bytes (encode modes)

Only decode-mode payloads are base64, so only those are parsed and edited;
encode-mode inputs are left to the byte-level operators.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from fuzzer_tool.core.rand_pool import RandPool

# Standard base64 alphabet + padding.
B64_ALPHA = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_B64_PAD = b"="
_B64_VALID = B64_ALPHA + _B64_PAD
_QUANTUM = 4

# Byte -> alphabet char. 256 is a multiple of 64, so the map stays uniform.
_TO_B64 = bytes(B64_ALPHA[i % len(B64_ALPHA)] for i in range(256))

_GEN_MAX = 64  # generated payload length cap
# Growing reject windows: binary input fails in the first, not after a full scan.
_SNIFF_WINDOWS = (16, 256)
_OP_COUNT = 5  # flip, insert, delete, swap, pad


class B64Mode(IntEnum):
    """Mode selector in byte 0 (low 2 bits)."""

    DECODE = 0
    ENCODE = 1
    STREAM_DECODE = 2
    STREAM_ENCODE = 3


_MODE_MASK = 3
_STREAM_MODES = frozenset({int(B64Mode.STREAM_DECODE), int(B64Mode.STREAM_ENCODE)})
_DECODE_MODES = frozenset({int(B64Mode.DECODE), int(B64Mode.STREAM_DECODE)})

# Header length per mode: mode byte, plus the chunk seed byte in stream modes.
_HEADER_LEN = tuple(2 if m in _STREAM_MODES else 1 for m in range(_MODE_MASK + 1))


def _pad(chars: bytes) -> bytes:
    """Strip padding and re-pad to a whole quantum.

    A 1-char remainder can never decode, so it is dropped:
    "QUJDR" -> "QUJD", "QUJDRA" -> "QUJDRA==", "QQ=" -> "QQ==".
    """
    chars = chars.rstrip(_B64_PAD)
    rem = len(chars) % _QUANTUM
    if rem == 1:
        return chars[:-1]
    return chars + _B64_PAD * ((_QUANTUM - rem) % _QUANTUM)


def parse_base64(data: bytes) -> bytes | None:
    """Return the payload of a decode-mode input if it is base64 text, else None."""
    if not data:
        return None
    mode = data[0] & _MODE_MASK
    if mode not in _DECODE_MODES:
        return None

    h = _HEADER_LEN[mode]
    n = len(data)
    if n <= h:
        return None

    # Windows sliced from data, so a rejected input never copies its payload.
    for w in _SNIFF_WINDOWS:
        if data[h : h + w].translate(None, _B64_VALID):
            return None
        if n <= h + w:
            return data[h:]

    payload = data[h:]
    if payload.translate(None, _B64_VALID):
        return None
    return payload


class Base64Mutator:
    """Structure-aware base64 mutator (header + base64 payload)."""

    def __init__(self, seed=None):
        self._rng = RandPool(seed=seed)

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one base64 mutation; generate a new input if *data* is not decode-mode base64."""
        self._rng = rng or self._rng
        payload = parse_base64(data)
        if payload is None:
            return self._generate_random_base64(max_len=max_len, rng=self._rng)

        r = self._rng
        out = bytearray(payload)
        op = r.randint(0, _OP_COUNT - 1)
        if op == 0:
            i = r.randint(0, len(out) - 1)
            out[i] = r.choice(B64_ALPHA)
        elif op == 1:
            i = r.randint(0, len(out))
            out.insert(i, r.choice(B64_ALPHA))
        elif op == 2:
            if len(out) > 1:
                i = r.randint(0, len(out) - 1)
                del out[i]
        elif op == 3:
            if len(out) > 1:
                i = r.randint(0, len(out) - 1)
                j = r.randint(0, len(out) - 1)
                out[i], out[j] = out[j], out[i]
        else:
            out = bytearray(_pad(bytes(out)))

        header = data[: _HEADER_LEN[data[0] & _MODE_MASK]]
        return (header + bytes(out))[:max_len]

    def _generate_random_base64(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Fresh input for a random mode: padded base64 for decode, raw bytes for encode."""
        self._rng = rng or self._rng
        r = self._rng
        mode = r.randint(0, _MODE_MASK)
        header = bytes([mode])
        if mode in _STREAM_MODES:
            header += r.randbytes(1)

        # One batched draw, mapped onto the alphabet for decode modes.
        body = r.randbytes(r.randint(1, _GEN_MAX))
        if mode in _DECODE_MODES:
            body = _pad(body.translate(_TO_B64))
        return (header + body)[:max_len]


__all__ = ["B64_ALPHA", "B64Mode", "Base64Mutator", "parse_base64"]
