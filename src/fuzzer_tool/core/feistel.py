"""Minimal Feistel network (64-bit blocks).

Ported and cleaned from AIscripts/minimal_feistel_network.py for fuzzer-tool.
Pure-Python, invertible transform useful as:

  - building block for adaptive / regularity mutation operators
  - keyed deterministic byte permutations
  - structured reversible transforms

Round function is SHA-256 based (first 32 bits).
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence


def feistel_round_function(data_bytes: bytes, key_int: int) -> int:
    """Round function: SHA-256(data || key) → first 32 bits as int."""
    key_bytes = key_int.to_bytes(4, byteorder="big", signed=False)
    combined = data_bytes + key_bytes
    hashed = hashlib.sha256(combined).digest()
    return int.from_bytes(hashed[:4], byteorder="big")


def feistel_encrypt(block: bytes, keys: Sequence[int], rounds: int = 4) -> bytes:
    """Encrypt an 8-byte (64-bit) block with a Feistel network.

    Parameters
    ----------
    block:
        Exactly 8 bytes.
    keys:
        Sequence of 32-bit integers used as round keys (cycled if shorter
        than *rounds*).
    rounds:
        Number of Feistel rounds (default 4).
    """
    if len(block) != 8:
        raise ValueError("block must be exactly 8 bytes")
    if not keys:
        raise ValueError("keys must not be empty")

    left = block[:4]
    right = block[4:]

    for i in range(rounds):
        left_int = int.from_bytes(left, "big")
        # F(right, key)
        f_out = feistel_round_function(right, keys[i % len(keys)])
        new_right = left_int ^ f_out
        left = right
        right = new_right.to_bytes(4, "big")

    return left + right


def feistel_decrypt(block: bytes, keys: Sequence[int], rounds: int = 4) -> bytes:
    """Decrypt an 8-byte block previously produced by :func:`feistel_encrypt`."""
    if len(block) != 8:
        raise ValueError("block must be exactly 8 bytes")
    if not keys:
        raise ValueError("keys must not be empty")

    left = block[:4]
    right = block[4:]

    for i in reversed(range(rounds)):
        right_int = int.from_bytes(right, "big")
        f_out = feistel_round_function(left, keys[i % len(keys)])
        new_left = right_int ^ f_out
        right = left
        left = new_left.to_bytes(4, "big")

    return left + right


def feistel_permute(data: bytes, keys: Sequence[int], rounds: int = 4) -> bytes:
    """Apply Feistel to successive 8-byte blocks; trailing bytes are left unchanged.

    Convenience helper for mutation operators that want a reversible
    permutation of a longer buffer.
    """
    if not data:
        return data
    out = bytearray()
    i = 0
    while i + 8 <= len(data):
        out.extend(feistel_encrypt(data[i : i + 8], keys, rounds))
        i += 8
    out.extend(data[i:])
    return bytes(out)


def feistel_unpermute(data: bytes, keys: Sequence[int], rounds: int = 4) -> bytes:
    """Inverse of :func:`feistel_permute`."""
    if not data:
        return data
    out = bytearray()
    i = 0
    while i + 8 <= len(data):
        out.extend(feistel_decrypt(data[i : i + 8], keys, rounds))
        i += 8
    out.extend(data[i:])
    return bytes(out)
