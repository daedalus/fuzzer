#!/usr/bin/env python3
"""Birthday search for SHA-1 prefix collisions.

Hashes random 16-byte messages until two distinct ones share the top
``bits`` of their digest. Expected attempts ~ sqrt(pi/2 * 2**bits)
(20 bits: ~1.3k). Memory is O(attempts): one dict entry per message.

Usage:
    python tools/sha1_collision.py [--bits 20] [--max-attempts N] [--seed S]
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass

from fuzzer_tool.core.rand_pool import RandPool

DIGEST_BITS = 160  # SHA-1
MSG_LEN = 16
DEFAULT_BITS = 20
DEFAULT_MAX_ATTEMPTS = 10_000_000
DRAW_BATCH = 4096  # messages per randbytes call: ~20% faster than per-message


@dataclass(frozen=True)
class Collision:
    """Two distinct messages sharing a digest prefix."""

    first: bytes
    second: bytes
    prefix: int
    attempts: int


def sha1_prefix(data: bytes, bits: int) -> int:
    """Top ``bits`` of SHA-1(data), e.g. bits=8 -> first digest byte."""
    nbytes = (bits + 7) >> 3
    head = int.from_bytes(hashlib.sha1(data).digest()[:nbytes], "big")
    return head >> ((nbytes << 3) - bits)


def find_collision(bits: int, max_attempts: int, rng) -> Collision | None:
    """Search up to ``max_attempts`` messages drawn from ``rng.randbytes``.

    Raises:
        ValueError: ``bits`` outside 1..160.
    """
    if not 1 <= bits <= DIGEST_BITS:
        raise ValueError(f"bits must be in 1..{DIGEST_BITS}")

    # Hoisted: prefix = first nbytes of digest, shifted.
    nbytes = (bits + 7) >> 3
    shift = (nbytes << 3) - bits
    sha1 = hashlib.sha1
    seen: dict[int, bytes] = {}

    # Fors down: one randbytes per batch, sliced into messages.
    attempt = 0
    while attempt < max_attempts:
        count = min(DRAW_BATCH, max_attempts - attempt)
        buf = rng.randbytes(count * MSG_LEN)

        for off in range(0, len(buf), MSG_LEN):
            attempt += 1
            msg = buf[off : off + MSG_LEN]
            prefix = int.from_bytes(sha1(msg).digest()[:nbytes], "big") >> shift

            prev = seen.setdefault(prefix, msg)
            if prev is msg or prev == msg:
                continue

            return Collision(prev, msg, prefix, attempt)

    return None


def _report(hit: Collision, bits: int, elapsed: float) -> None:
    """Print the pair and their full digests."""
    h1 = hashlib.sha1(hit.first).hexdigest()
    h2 = hashlib.sha1(hit.second).hexdigest()

    print(f"COLLISION bits={bits} attempts={hit.attempts:,} time={elapsed:.4f}s")
    print(f"msg1 {hit.first.hex()}  sha1 {h1}")
    print(f"msg2 {hit.second.hex()}  sha1 {h2}")
    print(f"prefix {hit.prefix:0{(bits + 3) // 4}x}  full_equal={h1 == h2}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not 1 <= args.bits <= DIGEST_BITS:
        parser.error(f"bits must be between 1 and {DIGEST_BITS}")

    start = time.perf_counter()
    hit = find_collision(args.bits, args.max_attempts, RandPool(seed=args.seed))
    if hit is None:
        print("No collision found within attempt limit.")
        return 1

    _report(hit, args.bits, time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
