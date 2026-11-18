"""Structure-aware base64 frame mutator (mode byte + base64 payload).

Input layout follows targets/libbase64_read.c: byte 0 is a mode selector
(even -> decode, odd -> encode), the payload starts at offset 1. The
payload is base64-encoded data; the mutator edits it while keeping the
base64 format so the differential codecs keep being exercised.
"""

from __future__ import annotations

from typing import Any

from fuzzer_tool.core.rand_pool import RandPool

# Standard base64 alphabet + padding.
B64_ALPHA = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_B64_SET = frozenset(B64_ALPHA)
_B64_PAD = ord("=")

# libbase64_read.c spends byte 0 on a mode selector; the payload starts at 1.
_MODE_OFFSET = 1
_MIN_LEN = 2  # mode byte + at least one payload char
_MAX_LEN = 4096  # guard


def parse_base64(data: bytes) -> bytes | None:
    """Return the payload (after the mode byte) if it is valid base64."""
    if len(data) < _MIN_LEN:
        return None
    payload = data[_MODE_OFFSET:]
    if not payload:
        return None
    for b in payload:
        if b != _B64_PAD and b not in _B64_SET:
            return None
    return payload


class Base64Mutator:
    """Structure-aware base64 mutator (mode byte + base64 payload)."""

    def __init__(self, seed=None):
        self._rng = RandPool(seed=seed)

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        """Apply one base64 mutation; generate a new base64 payload if *data* is not one."""
        self._rng = rng or self._rng
        payload = parse_base64(data)
        if payload is None:
            return self._generate_random_base64(max_len=max_len, rng=self._rng)

        out = bytearray(payload)
        op = self._rng.randint(0, 4)  # 5 ops: flip, insert, delete, swap, pad
        if op == 0:
            i = self._rng.randint(0, len(out) - 1)
            out[i] = self._rng.choice(B64_ALPHA)
        elif op == 1:
            i = self._rng.randint(0, len(out))
            out.insert(i, self._rng.choice(B64_ALPHA))
        elif op == 2:
            if len(out) > 1:
                i = self._rng.randint(0, len(out) - 1)
                del out[i]
        elif op == 3:
            if len(out) > 1:
                i = self._rng.randint(0, len(out) - 1)
                j = self._rng.randint(0, len(out) - 1)
                out[i], out[j] = out[j], out[i]
        else:
            while out and out[-1] == _B64_PAD:
                out.pop()
            for _ in range(2):
                out.append(_B64_PAD)
        return bytes([data[0]]) + bytes(out)

    def _generate_random_base64(self, max_len: int = 65536, rng: Any = None) -> bytes:
        self._rng = rng or self._rng
        r = self._rng
        n = r.randint(1, 64)
        payload = bytes(r.choice(B64_ALPHA) for _ in range(n))
        if r.randint(0, 3) == 0:
            payload += b"=" * r.randint(0, 2)
        mode = r.randint(0, 3)
        return bytes([mode]) + payload


__all__ = ["B64_ALPHA", "Base64Mutator", "parse_base64"]
