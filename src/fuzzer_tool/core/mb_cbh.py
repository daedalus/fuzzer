"""Magic-bytes (MB) and climb-hill (CBH) constraint search.

Port of Angora's ``fuzzer/src/search/mb.rs`` and ``cbh.rs``. Both are
small stochastic searches over the same objective the gradient-descent
port optimizes: make some window of the input equal a comparison operand
observed via cmplog.

The three searches are deliberately different in character, which is why
all three earn a slot rather than collapsing into one:

* ``gradient_descent`` — deterministic ladder (+-1, +-2, +-4, +-8) plus an
  interesting-value escape, anchored on the single best-matching site.
  Converges reliably when the input is already close to the operand.
* ``magic_byte_search`` (MB) — no descent at all. Writes the operand
  verbatim at a candidate site and randomizes elsewhere. This is the
  "just try the constant" move; it solves in one step the common case
  where a magic number simply has to appear, and the randomization keeps
  it from producing the same input every time the operator is picked.
* ``climb_hill`` (CBH) — stochastic: random position, random value,
  accept on improvement. Unlike the gradient ladder it can jump anywhere
  in the byte range in a single step, so it escapes plateaus that the
  bounded ladder walks past, at the cost of being non-deterministic.

MB and CBH are complementary, and the pairing matters. Both CBH and
``gradient_descent`` locate the window to work on via
``_candidate_positions``, which derives sites from *byte-value overlap*
between the input and the operand. When an input shares no bytes with the
operand there is no signal, site selection degrades to random offsets,
and the descent optimizes an arbitrary window. Measured with a 4-byte
operand at a known offset: CBH converges 172/200 seeds when one byte
already matches and 163/200 with two differing, but 0/200 when the input
shares nothing with the operand -- not because the climb fails (it still
solved some site in 135/200 runs) but because it was anchored in the
wrong place. MB has no such dependency: it writes the operand verbatim,
which bootstraps exactly the overlap the other two need.

Both take the candidate-site machinery from ``gradient_descent`` rather
than duplicating it. These are private there but shared deliberately
inside the package -- the site-selection logic is the same problem and
should not drift between the three searches.
"""

from __future__ import annotations

import logging

from fuzzer_tool.core.gradient_descent import _candidate_positions, _window_distance

log = logging.getLogger(__name__)

# CBH is stochastic, so it needs an iteration bound rather than an epoch
# count. Kept small: this runs inline in the mutation path, once per
# fuzz_one() call that selects the operator.
_CBH_MAX_ITERS = 128
_CBH_MAX_STUCK = 64

# How many bytes MB randomizes around the planted magic bytes. Bounded so
# the operator stays a targeted move rather than a havoc pass.
_MB_MAX_RANDOM_BYTES = 8


def _pick_target(cmp_pair: tuple[bytes, bytes]) -> bytes:
    """Return the operand to match: the shorter one, as in gradient_descent.

    The shorter operand is more likely to be the constant side of the
    comparison (a magic number or length field) and is the one worth
    planting.
    """
    op_a, op_b = cmp_pair
    if not op_a and not op_b:
        return b""
    if not op_a:
        return op_b
    if not op_b:
        return op_a
    return op_b if len(op_b) <= len(op_a) else op_a


def magic_byte_search(
    input_buf: bytes,
    cmp_pair: tuple[bytes, bytes],
    rng,
    max_len: int = 0,
) -> bytes:
    """Plant a comparison operand at a candidate site, randomize around it.

    Angora's MB search. Unlike the gradient searches this makes no attempt
    to descend -- it writes the operand verbatim, which is exactly right
    when the branch is a plain equality against a constant.

    Returns the original input unchanged when there is nothing to plant.
    """
    target = _pick_target(cmp_pair)
    if not input_buf or not target:
        return input_buf

    buf = bytearray(input_buf[:max_len] if max_len else input_buf)
    if len(buf) < len(target):
        return input_buf

    candidates = _candidate_positions(bytes(buf), target)
    # Restrict to sites where the operand actually fits.
    candidates = [p for p in candidates if p + len(target) <= len(buf)]
    if not candidates:
        # No overlap-derived site fits; fall back to any valid offset so the
        # operator still does something useful on inputs that share no bytes
        # with the operand yet.
        span = len(buf) - len(target)
        if span < 0:
            return input_buf
        candidates = [rng.randint(0, span)]

    site = rng.choice(candidates)
    buf[site : site + len(target)] = target

    # Randomize a bounded number of bytes outside the planted window, so
    # repeated selections explore rather than returning an identical input.
    n_rand = rng.randint(0, _MB_MAX_RANDOM_BYTES)
    for _ in range(n_rand):
        pos = rng.randint(0, len(buf) - 1)
        if site <= pos < site + len(target):
            continue  # never clobber the bytes we just planted
        buf[pos] = rng.randint(0, 255)

    return bytes(buf)


def climb_hill(
    input_buf: bytes,
    cmp_pair: tuple[bytes, bytes],
    rng,
    max_len: int = 0,
    max_iters: int = _CBH_MAX_ITERS,
) -> bytes:
    """Stochastic hill-climb minimizing distance to a comparison operand.

    Angora's CBH search. Picks a random byte in the scored window and a
    random new value, keeping the change only when it reduces the window
    distance. Complements the deterministic ladder in ``gradient_descent``:
    a single step can move a byte anywhere in 0..255, so it crosses
    plateaus the bounded ladder cannot.

    Returns the original input when no improvement is found.
    """
    target = _pick_target(cmp_pair)
    if not input_buf or not target:
        return input_buf

    buf = bytearray(input_buf[:max_len] if max_len else input_buf)
    if len(buf) < len(target):
        return input_buf

    candidates = _candidate_positions(bytes(buf), target)
    candidates = [p for p in candidates if p + len(target) <= len(buf)]
    if not candidates:
        return input_buf

    # Anchor on the most promising site, matching gradient_descent: the
    # objective is only meaningful relative to a fixed window.
    site = min(candidates, key=lambda p: _window_distance(bytes(buf), p, target))

    best = bytearray(buf)
    best_score = _window_distance(bytes(best), site, target)
    if best_score == 0:
        return bytes(best)

    width = len(target)
    stuck = 0

    for _ in range(max_iters):
        # Flip one random bit rather than randomizing the whole byte.
        # The objective is Hamming distance, so a single-bit flip improves
        # it with probability (differing bits)/8 at a differing position --
        # tractable. Drawing a uniformly random byte value instead only
        # improves with probability ~1/256 per position, which in practice
        # means the stuck counter always fires before convergence: at
        # distance 1 over a 4-byte operand the search would need to hit
        # both the right position and the exact right value.
        # Bit flips also keep CBH genuinely distinct from gradient_descent,
        # whose ladder takes *arithmetic* steps (+-1..+-8).
        pos = site + rng.randint(0, width - 1)
        candidate = bytearray(best)
        candidate[pos] ^= 1 << rng.randint(0, 7)
        score = _window_distance(bytes(candidate), site, target)
        if score < best_score:
            best = candidate
            best_score = score
            stuck = 0
            if best_score == 0:
                break
        else:
            stuck += 1
            if stuck >= _CBH_MAX_STUCK:
                break

    return bytes(best)
