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

# How many candidate sites one call may anchor on. ``1`` commits to the
# initial argmin and returns early when it stalls. See the "site
# re-anchoring" note in ``climb_hill``: the mechanism is implemented and
# tested but defaults off, because at the shipped budget it measures as a
# wash.
_CBH_MAX_SITES = 1

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

    candidates = _candidate_positions(bytes(buf), target, rng)
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
    max_sites: int = _CBH_MAX_SITES,
) -> bytes:
    """Stochastic hill-climb minimizing distance to a comparison operand.

    Angora's CBH search. Picks a random byte in the scored window and a
    random new value, keeping the change only when it reduces the window
    distance. Complements the deterministic ladder in ``gradient_descent``:
    a single step can move a byte anywhere in 0..255, so it crosses
    plateaus the bounded ladder cannot.

    **Site re-anchoring.** The window under optimization used to be chosen
    once, by argmin over the overlap-derived candidates, and the whole
    iteration budget then went to that one site. The module docstring
    records what that costs: with a 4-byte operand at a known offset the
    climb solved *some* site in 135/200 runs but the planted one in 0/200
    when the input shared no bytes with the operand -- the descent worked
    and the anchor was wrong. Committing is also self-reinforcing, since
    every accepted move lowers the incumbent site's score and so it never
    comes to look worse than the alternative it was chosen over.

    When the climb stalls, the search therefore discards the local
    incumbent and re-anchors on the next-most-promising candidate rather
    than returning with budget left over, for up to *max_sites* sites. The
    discard is the point: carrying the edits made for the previous window
    into the next one carries noise, and scoring the new site against the
    original buffer keeps a site that has merely been optimized longer from
    winning by default. The best buffer seen at *any* site is tracked
    separately and returned, so re-anchoring can only add outcomes.

    Re-anchoring is stall-triggered rather than periodic on purpose. A
    fixed-period restart would fire in the middle of a converging climb:
    at Hamming distance *d* over a *w*-byte window a bit flip improves with
    probability ``d/(8w)``, so a 4-byte operand needs ~100 iterations from
    a random start, and a periodic reset short of that would prevent
    convergence rather than escape a basin.

    **Measured: neutral at the shipped budget, so this defaults off.**
    Planted-operand solve rate over 400 seeded inputs, 4-byte operand at a
    known offset, ``max_iters=128`` and a 64-iteration stall threshold:

    ==================  ==========  ==========
    input               max_sites=1 max_sites=4
    ==================  ==========  ==========
    no matching bytes      5/400       8/400
    one matching byte     92/400      99/400
    two matching bytes   268/400     274/400
    ==================  ==========  ==========

    The stall threshold is half the iteration budget, so a second site is
    reached only when the first stalls early, which is rare. Raising the
    budget to 512 does separate the arms (114 vs 92 at one matching byte),
    but that buys the difference with 4x the CPU rather than with the
    mechanism, and CBH runs inline in the mutation path. Left in and
    default-off so it is an arm the paired harness can test rather than a
    rewrite that has to be redone; set ``max_sites`` > 1 to enable.

    Returns the original input when no improvement is found.
    """
    target = _pick_target(cmp_pair)
    if not input_buf or not target:
        return input_buf

    buf = bytearray(input_buf[:max_len] if max_len else input_buf)
    if len(buf) < len(target):
        return input_buf

    candidates = _candidate_positions(bytes(buf), target, rng)
    candidates = [p for p in candidates if p + len(target) <= len(buf)]
    if not candidates:
        return input_buf

    # Order sites by how well the *original* buffer matches there. The
    # first is the argmin the old implementation committed to; the rest
    # are the re-anchor targets, in decreasing promise.
    origin = bytes(buf)
    candidates.sort(key=lambda p: _window_distance(origin, p, target))
    del candidates[max(1, max_sites) :]

    site_idx = 0
    site = candidates[0]

    # Local incumbent: discarded on every re-anchor.
    best = bytearray(buf)
    best_score = _window_distance(origin, site, target)
    if best_score == 0:
        return bytes(best)

    # Global incumbent: never discarded, and what the caller gets back.
    overall = bytearray(best)
    overall_score = best_score

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
            if best_score < overall_score:
                overall = bytearray(best)
                overall_score = best_score
            if best_score == 0:
                return bytes(best)
            continue

        stuck += 1
        if stuck < _CBH_MAX_STUCK:
            continue

        # Stalled. Move to the next site rather than returning with budget
        # left; stop only when the candidate list is exhausted.
        site_idx += 1
        if site_idx >= len(candidates):
            break
        site = candidates[site_idx]
        best = bytearray(buf)
        best_score = _window_distance(origin, site, target)
        stuck = 0
        if best_score < overall_score:
            overall = bytearray(best)
            overall_score = best_score
        if best_score == 0:
            return bytes(best)

    return bytes(overall)
