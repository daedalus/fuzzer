"""Auto-dictionary: atomic tokens from byteflip trace hashes (AFL auto extras).

A run of bytes where flipping any one yields the *same* changed path, and
that path differs from the neighbours', is one atomic comparison::

    offset   89 P N G 0d 0a 1a 0a | 00 00 00 0d | I H D R | ...
    flip ->  A  A A A A  A  A  A  | B  C  D  E  | F F F F | ...
             '------ token ------'                '-tok-'

``memcmp(p, "IHDR", 4)`` fails the same way whichever byte is wrong. The
byteflip pass already executes every one of these mutants, so the tokens
cost no extra executions.
"""

from array import array

import numpy as np

# AFL's MIN_AUTO_EXTRA / MAX_AUTO_EXTRA: shorter runs are noise, longer ones
# are checksummed or length-gated blobs, not tokens.
MIN_AUTO_TOKEN = 3
MAX_AUTO_TOKEN = 32

# Hash slot value for "trace unchanged" or "never measured".
NO_HASH = 0


def harvest_tokens(data: bytes, hashes: array) -> list[bytes]:
    """Tokens from per-byte flip hashes, in offset order.

    Args:
        data: The seed the byteflips were applied to.
        hashes: ``array("Q")``; ``hashes[i]`` is the path hash observed when
            byte ``i`` was flipped, ``NO_HASH`` if unchanged or unmeasured.

    Returns:
        Each maximal equal-hash run of length MIN..MAX_AUTO_TOKEN whose hash
        is set and whose bytes are not all identical.
    """
    n = min(len(data), len(hashes))
    if n < MIN_AUTO_TOKEN:
        return []

    # Run boundaries: offsets where the hash differs from its predecessor.
    h = np.frombuffer(hashes, np.uint64, count=n)
    starts = np.flatnonzero(np.concatenate(([True], h[1:] != h[:-1])))
    ends = np.append(starts[1:], n)
    sizes = ends - starts
    keep = (h[starts] != NO_HASH) & (sizes >= MIN_AUTO_TOKEN) & (sizes <= MAX_AUTO_TOKEN)

    tokens = []
    for lo, hi in zip(starts[keep].tolist(), ends[keep].tolist(), strict=True):
        tok = data[lo:hi]
        # Uniform fill (zero padding behind one length check) is not a token.
        if tok.count(tok[:1]) == len(tok):
            continue
        tokens.append(tok)
    return tokens


def frozen_tokens(data: bytes, mutable: np.ndarray) -> list[bytes]:
    """Tokens from a LearnAFL path mask (``core/schedulers/pos_learn.py``).

    A byte havoc changed without moving the path is *mutable*; the rest is the
    format the path needs (``MAGIC`` below)::

        data     M A G I C 00 00 H D R X ...
        mutable  . . . . . x  x  . . . . ...
        tokens   '-MAGIC-'       '-HDRX-'

    Args:
        data: The seed the mask was learned on.
        mutable: Bool array; ``mutable[i]`` is set once byte ``i`` was seen
            changed in a path-preserving mutant.

    Returns:
        Each maximal frozen run of length MIN..MAX_AUTO_TOKEN whose bytes are
        not all identical, in offset order.
    """
    n = min(len(data), len(mutable))
    if n < MIN_AUTO_TOKEN:
        return []

    # Run edges: +1 where a frozen run starts, -1 where it ends.
    frozen = np.concatenate(([False], ~mutable[:n], [False])).astype(np.int8)
    edges = np.diff(frozen)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    sizes = ends - starts
    keep = (sizes >= MIN_AUTO_TOKEN) & (sizes <= MAX_AUTO_TOKEN)

    tokens = []
    for lo, hi in zip(starts[keep].tolist(), ends[keep].tolist(), strict=True):
        tok = data[lo:hi]
        if tok.count(tok[:1]) == len(tok):
            continue
        tokens.append(tok)
    return tokens
