"""Merkle-Damgard length extension (``--hlx-algo``).

A target that checks ``H(key || msg)`` with MD5, SHA-1 or SHA-256 rejects every
mutated message.  The digest is the hash's internal state after the last block,
so from ``(msg, H(key || msg))`` and the key *length* (not the key) we can
append ``padding || suffix`` and compute the digest of the longer message::

    seed   : msg                          | D = H(key || msg)
    output : msg || pad(L) || suffix      | H(key || msg || pad(L) || suffix)
                                            ^ state := D, resume at L + len(pad)

The seed digest must already be valid; every output is again a valid pair, so
extensions chain.  The key length is a guess supplied by the user.  Stdlib
only: hashlib cannot resume from a state, so the compression functions are
implemented here (Hard Rule 51) and checked against hashlib in the tests.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

_BLOCK = 64
_LEN_FIELD = 8
_BITS = 8
_MASK = 0xFFFFFFFF
_PAD_MARK = b"\x80"
#: Longest appended suffix, in bytes.
MAX_SUFFIX = 32
#: Default key length guess (a 128-bit key).
DEFAULT_KEY_LEN = 16
_MD5_ROUNDS = 64
_SHA1_ROUNDS = 80
_SHA256_ROUNDS = 64
_SHA256_FIRST_PRIMES = 64


class _Algo(NamedTuple):
    """Per-hash constants: digest size, word byte order, compression."""

    digest_len: int
    byteorder: str
    state_fmt: str
    compress: object


def _rol(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & _MASK


def _ror(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK


# MD5 sine table: floor(2^32 * |sin(i + 1)|), RFC 1321.
_MD5_K = (
    0xD76AA478,
    0xE8C7B756,
    0x242070DB,
    0xC1BDCEEE,
    0xF57C0FAF,
    0x4787C62A,
    0xA8304613,
    0xFD469501,
    0x698098D8,
    0x8B44F7AF,
    0xFFFF5BB1,
    0x895CD7BE,
    0x6B901122,
    0xFD987193,
    0xA679438E,
    0x49B40821,
    0xF61E2562,
    0xC040B340,
    0x265E5A51,
    0xE9B6C7AA,
    0xD62F105D,
    0x02441453,
    0xD8A1E681,
    0xE7D3FBC8,
    0x21E1CDE6,
    0xC33707D6,
    0xF4D50D87,
    0x455A14ED,
    0xA9E3E905,
    0xFCEFA3F8,
    0x676F02D9,
    0x8D2A4C8A,
    0xFFFA3942,
    0x8771F681,
    0x6D9D6122,
    0xFDE5380C,
    0xA4BEEA44,
    0x4BDECFA9,
    0xF6BB4B60,
    0xBEBFBC70,
    0x289B7EC6,
    0xEAA127FA,
    0xD4EF3085,
    0x04881D05,
    0xD9D4D039,
    0xE6DB99E5,
    0x1FA27CF8,
    0xC4AC5665,
    0xF4292244,
    0x432AFF97,
    0xAB9423A7,
    0xFC93A039,
    0x655B59C3,
    0x8F0CCC92,
    0xFFEFF47D,
    0x85845DD1,
    0x6FA87E4F,
    0xFE2CE6E0,
    0xA3014314,
    0x4E0811A1,
    0xF7537E82,
    0xBD3AF235,
    0x2AD7D2BB,
    0xEB86D391,
)
_MD5_S = (7, 12, 17, 22) * 4 + (5, 9, 14, 20) * 4 + (4, 11, 16, 23) * 4 + (6, 10, 15, 21) * 4


def _md5_block(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    words = struct.unpack("<16I", block)
    a, b, c, d = state
    for i in range(_MD5_ROUNDS):
        if i < 16:
            f, g = (b & c) | (~b & d), i
        elif i < 32:
            f, g = (d & b) | (~d & c), (5 * i + 1) % 16
        elif i < 48:
            f, g = b ^ c ^ d, (3 * i + 5) % 16
        else:
            f, g = c ^ (b | (~d & _MASK)), (7 * i) % 16
        f = (f + a + _MD5_K[i] + words[g]) & _MASK
        a, d, c = d, c, b
        b = (b + _rol(f, _MD5_S[i])) & _MASK
    return tuple((x + y) & _MASK for x, y in zip(state, (a, b, c, d), strict=True))


def _sha1_block(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    w = list(struct.unpack(">16I", block))
    for i in range(16, _SHA1_ROUNDS):
        w.append(_rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1))
    a, b, c, d, e = state
    for i in range(_SHA1_ROUNDS):
        if i < 20:
            f, k = (b & c) | (~b & d), 0x5A827999
        elif i < 40:
            f, k = b ^ c ^ d, 0x6ED9EBA1
        elif i < 60:
            f, k = (b & c) | (b & d) | (c & d), 0x8F1BBCDC
        else:
            f, k = b ^ c ^ d, 0xCA62C1D6
        a, b, c, d, e = (_rol(a, 5) + f + e + k + w[i]) & _MASK, a, _rol(b, 30), c, d
    return tuple((x + y) & _MASK for x, y in zip(state, (a, b, c, d, e), strict=True))


def _icbrt(n: int) -> int:
    """Floor of the integer cube root of *n* (binary search)."""
    lo, hi = 0, 1 << ((n.bit_length() + 2) // 3 + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**3 <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    n = 2
    while len(primes) < count:
        if all(n % p for p in primes if p * p <= n):
            primes.append(n)
        n += 1
    return primes


# SHA-256 K: first 32 fractional bits of cbrt(prime) = icbrt(prime << 96) mod 2^32.
_SHA256_K = tuple(_icbrt(p << 96) & _MASK for p in _first_primes(_SHA256_FIRST_PRIMES))


def _sha256_block(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    w = list(struct.unpack(">16I", block))
    for i in range(16, _SHA256_ROUNDS):
        s0 = _ror(w[i - 15], 7) ^ _ror(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = _ror(w[i - 2], 17) ^ _ror(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & _MASK)
    a, b, c, d, e, f, g, h = state
    for i in range(_SHA256_ROUNDS):
        t1 = (
            h
            + (_ror(e, 6) ^ _ror(e, 11) ^ _ror(e, 25))
            + ((e & f) ^ (~e & g))
            + _SHA256_K[i]
            + w[i]
        ) & _MASK
        t2 = ((_ror(a, 2) ^ _ror(a, 13) ^ _ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & _MASK
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & _MASK, c, b, a, (t1 + t2) & _MASK
    regs = (a, b, c, d, e, f, g, h)
    return tuple((x + y) & _MASK for x, y in zip(state, regs, strict=True))


ALGOS: dict[str, _Algo] = {
    "md5": _Algo(16, "little", "<4I", _md5_block),
    "sha1": _Algo(20, "big", ">5I", _sha1_block),
    "sha256": _Algo(32, "big", ">8I", _sha256_block),
}


def _algo(name: str) -> _Algo:
    try:
        return ALGOS[name]
    except KeyError:
        raise ValueError(f"unknown hash {name!r}; expected one of {sorted(ALGOS)}") from None


def digest_len(name: str) -> int:
    """Digest size in bytes of *name*."""
    return _algo(name).digest_len


def md_pad(name: str, length: int) -> bytes:
    """Padding appended to a *length*-byte message (0x80, zeros, bit length)."""
    order = _algo(name).byteorder
    zeros = (-(length + 1 + _LEN_FIELD)) % _BLOCK
    bits = (length * _BITS).to_bytes(_LEN_FIELD, order)
    return _PAD_MARK + bytes(zeros) + bits


def md_resume(name: str, digest: bytes, prior: int, data: bytes) -> bytes:
    """Digest of ``prior`` already-hashed bytes (block aligned) plus *data*.

    *digest* is the hash state after those ``prior`` bytes, i.e. the digest of
    a message that was padded to exactly ``prior`` bytes.
    """
    algo = _algo(name)
    if len(digest) != algo.digest_len:
        raise ValueError(f"{name} digest must be {algo.digest_len} bytes, got {len(digest)}")
    if prior % _BLOCK:
        raise ValueError(f"prior length {prior} is not a multiple of {_BLOCK}")
    state = struct.unpack(algo.state_fmt, digest)
    tail = data + md_pad(name, prior + len(data))
    for i in range(0, len(tail), _BLOCK):
        state = algo.compress(state, tail[i : i + _BLOCK])
    return struct.pack(algo.state_fmt, *state)


def digest_start(name: str, total: int, digest_at: int) -> int | None:
    """Offset of the digest in a *total*-byte buffer, or None if it does not fit.

    ``digest_at >= 0`` counts from the start, negative from the end.
    """
    size = digest_len(name)
    start = digest_at if digest_at >= 0 else total + digest_at
    if start < 0 or start + size > total:
        return None
    return start


def message_of(name: str, data: bytes, digest_at: int) -> bytes | None:
    """Bytes outside the digest slot, in order; None if the slot does not fit."""
    start = digest_start(name, len(data), digest_at)
    if start is None:
        return None
    return data[:start] + data[start + digest_len(name) :]


def _default_at(name: str) -> int:
    return -digest_len(name)


def hash_len_ext(
    data: bytes,
    name: str,
    key_len: int,
    rng,
    max_len: int,
    digest_at: int | None = None,
) -> bytes | None:
    """Append ``padding || suffix`` to the message and rewrite the digest.

    Args:
        data: Seed laid out as message plus a valid ``H(key || msg)`` digest.
        name: ``md5``, ``sha1`` or ``sha256``.
        key_len: Guessed secret length in bytes.
        rng: ``RandPool`` (or ScriptedRng in tests).
        max_len: Output length cap; the suffix is clamped to fit.
        digest_at: Digest offset; negative counts from the end (default: last
            digest-size bytes).

    Returns:
        New buffer, or None when the digest slot does not fit or the cap
        leaves no room for the padding plus one suffix byte.
    """
    at = _default_at(name) if digest_at is None else digest_at
    start = digest_start(name, len(data), at)
    if start is None:
        return None
    size = digest_len(name)
    digest = data[start : start + size]
    msg = data[:start] + data[start + size :]

    pad = md_pad(name, key_len + len(msg))
    room = max_len - len(data) - len(pad)
    if room < 1:
        return None
    suffix = rng.randbytes(min(rng.randint(1, MAX_SUFFIX), room))
    prior = key_len + len(msg) + len(pad)
    new_digest = md_resume(name, digest, prior, suffix)

    new_msg = msg + pad + suffix
    new_start = digest_start(name, len(new_msg) + size, at)
    if new_start is None:
        return None
    return new_msg[:new_start] + new_digest + new_msg[new_start:]
