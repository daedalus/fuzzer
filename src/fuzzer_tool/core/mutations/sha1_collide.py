"""Truncated SHA-1 second preimage (``--sha1-collide-bits``).

Targets that key caches, dedup sets or hash tables on a truncated SHA-1
treat two inputs with the same top ``bits`` as one. This rewrites an
``WINDOW``-byte window until the mutant's digest prefix equals the seed's::

    seed   : head | window  | tail     SHA-1 -> p.......
    output : head | window' | tail     SHA-1 -> p.......   (top bits equal)

Expected tries 2**bits; the budget is ``BUDGET_FACTOR`` times that
(miss rate e**-4 ~ 2%). The head is hashed once and its state copied per
try; the window sits within ``MAX_TAIL`` bytes of the end so each try
hashes at most ``WINDOW + MAX_TAIL`` bytes.
"""

from __future__ import annotations

import hashlib

DIGEST_BITS = 160  # SHA-1
#: Largest prefix width: 16 bits ~ 262k tries worst case (~0.3 s).
MAX_BITS = 16
#: Bytes rewritten per try.
WINDOW = 8
#: Most bytes hashed after the window per try.
MAX_TAIL = 256
#: Budget in units of the expected 2**bits tries.
BUDGET_FACTOR = 4
#: Candidate windows per randbytes draw (fors down).
DRAW_BATCH = 256


def sha1_prefix(data: bytes, bits: int) -> int:
    """Top ``bits`` of SHA-1(data), e.g. bits=8 -> first digest byte."""
    nbytes = (bits + 7) >> 3
    head = int.from_bytes(hashlib.sha1(data).digest()[:nbytes], "big")
    return head >> ((nbytes << 3) - bits)


def budget(bits: int) -> int:
    """Tries allowed for a ``bits``-wide prefix."""
    return BUDGET_FACTOR << bits


def sha1_collide(data: bytes, bits: int, rng) -> bytes | None:
    """Same-length mutant whose SHA-1 top ``bits`` equal ``data``'s.

    Returns None when ``data`` is shorter than ``WINDOW`` or the budget
    runs out.

    Raises:
        ValueError: ``bits`` outside 1..MAX_BITS.
    """
    if not 1 <= bits <= MAX_BITS:
        raise ValueError(f"bits must be in 1..{MAX_BITS}")

    n = len(data)
    if n < WINDOW:
        return None

    # Window within MAX_TAIL of the end bounds the per-try hash length.
    off = rng.randint(max(0, n - WINDOW - MAX_TAIL), n - WINDOW)
    orig = data[off : off + WINDOW]
    tail = data[off + WINDOW :]
    head = hashlib.sha1(data[:off])

    nbytes = (bits + 7) >> 3
    shift = (nbytes << 3) - bits
    target = sha1_prefix(data, bits)

    left = budget(bits)
    while left > 0:
        count = min(DRAW_BATCH, left)
        left -= count
        buf = rng.randbytes(count * WINDOW)

        for i in range(0, len(buf), WINDOW):
            win = buf[i : i + WINDOW]
            h = head.copy()
            h.update(win)
            h.update(tail)
            if int.from_bytes(h.digest()[:nbytes], "big") >> shift != target:
                continue
            if win == orig:
                continue

            return data[:off] + win + tail

    return None
