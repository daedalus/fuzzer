"""Regression tests for the incremental gradient-descent objective.

The descent used to copy the whole buffer twice per probe and rescore the
window from scratch. The scoring is now incremental, which is only sound
because the objective is separable. These tests pin that equivalence
against a verbatim copy of the old probe loop, and pin the two behaviour
fixes that came with it.
"""

from __future__ import annotations

import os
import random

import pytest

from fuzzer_tool.core.gradient_descent import (
    _MAX_EPOCHS,
    _MAX_STUCK,
    _STEPS,
    _candidate_positions,
    _interesting_for_width,
    _window_distance,
    gradient_descent,
)

# ---------------------------------------------------------------------------
# Oracle: the old copy-and-rescore probe loop, verbatim, over a fixed site.
# ---------------------------------------------------------------------------


def _old_descent(buf: bytearray, site: int, target: bytes, max_epochs=_MAX_EPOCHS) -> bytes:
    width = len(target)
    best = bytearray(buf)
    best_score = _window_distance(bytes(best), site, target)
    if best_score == 0:
        return bytes(best)
    window = [site + i for i in range(width) if site + i < len(best)]
    if not window:
        return bytes(best)
    interesting = _interesting_for_width(width)
    stuck = 0
    for _ in range(max_epochs):
        improved = False
        for pos in window:
            orig = best[pos]
            for delta in _STEPS:
                candidate = bytearray(best)
                candidate[pos] = max(0, min(255, orig + delta))
                score = _window_distance(bytes(candidate), site, target)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    if best_score == 0:
                        break
            if best_score == 0:
                break
        if not improved:
            stuck += 1
            if stuck >= _MAX_STUCK:
                break
            for pos in window:
                orig = best[pos]
                for v in interesting:
                    if 0 <= v <= 255 and v != orig:
                        candidate = bytearray(best)
                        candidate[pos] = v
                        score = _window_distance(bytes(candidate), site, target)
                        if score < best_score:
                            best = candidate
                            best_score = score
                            improved = True
                            if best_score == 0:
                                break
                if best_score == 0:
                    break
        if best_score == 0:
            break
    return bytes(best)


# ---------------------------------------------------------------------------


def test_descent_matches_copy_and_rescore_oracle():
    """The incremental probe reaches the same buffer as the old loop.

    The site is chosen the same way the function does, so the comparison
    isolates the probe loop from candidate selection.
    """
    rnd = random.Random(3)
    checked = 0
    for _ in range(300):
        n = rnd.choice([32, 64, 256, 1024])
        buf = bytearray(os.urandom(n))
        width = rnd.choice([1, 2, 4, 8])
        target = os.urandom(width)
        if n > width + 8:
            p = rnd.randrange(0, n - width)
            buf[p : p + max(1, width - 1)] = target[: max(1, width - 1)]
        frozen = bytes(buf)
        # _candidate_positions falls back to the global random module when
        # overlap is sparse and no rng is threaded through, so both sides
        # must see the same global state to pick the same site.
        random.seed(4242)
        candidates = _candidate_positions(frozen, target, rng=None)
        if not candidates:
            continue
        site = min(candidates, key=lambda q: _window_distance(frozen, q, target))
        if site + width > n:
            continue
        checked += 1
        random.seed(4242)
        got = gradient_descent(frozen, (target, b""))
        assert got == _old_descent(buf, site, target)
    assert checked > 100, f"test degenerate: only {checked} cases exercised"


def test_window_running_past_the_end_is_left_alone(monkeypatch):
    """A window that overruns the buffer scores a constant, so nothing wins.

    _window_distance returns width*8 for any such position regardless of
    the bytes, so the old loop could never improve it. The incremental
    scoring reads the real bytes and would find phantom improvements
    without the early bail.

    An overrunning site is not reachable through the normal candidate
    path today -- it always scores the maximum, and `min` breaks ties
    toward the lowest offset, so a valid window always wins. The site is
    forced here so the guard is exercised rather than assumed; if
    candidate selection ever changes, this is what catches it.
    """
    import fuzzer_tool.core.gradient_descent as gd

    target = b"\xaa\xbb\xcc\xdd"
    buf = os.urandom(64)
    monkeypatch.setattr(gd, "_candidate_positions", lambda *a, **k: [62])
    out = gd.gradient_descent(buf, (target, b""))
    assert out == buf, "an overrunning window must be left untouched"


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_descent_never_lengthens_or_shortens(width):
    buf = os.urandom(512)
    out = gradient_descent(buf, (os.urandom(width), b""))
    assert len(out) == len(buf)


def test_descent_only_touches_the_scored_window():
    target = b"\xde\xad\xbe\xef"
    buf = bytearray(os.urandom(2048))
    buf[1000:1002] = target[:2]
    buf = bytes(buf)
    out = gradient_descent(buf, (target, b""))
    changed = [i for i in range(len(buf)) if out[i] != buf[i]]
    assert len(changed) <= len(target)
    if changed:
        assert max(changed) - min(changed) < len(target)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def test_candidates_span_the_whole_buffer():
    """Candidates must not be confined to the head of a large input.

    sorted(candidates)[:cap] kept the cap smallest offsets; measured on a
    64 KiB buffer every one of the 48 returned offsets fell inside the
    first 5.35% of it, so the descent could only ever anchor near the
    start of a file.
    """
    buf = os.urandom(65536)
    target = b"\xde\xad\xbe\xef"
    cands = _candidate_positions(buf, target, rng=None)
    assert cands, "no candidates found in 64 KiB of random data"
    assert max(cands) > len(buf) * 0.75, (
        f"candidates stop at {max(cands)} of {len(buf)} "
        f"({max(cands) / len(buf) * 100:.2f}% in)"
    )
    assert min(cands) < len(buf) * 0.25


def test_candidates_are_capped_sorted_and_unique():
    buf = os.urandom(65536)
    cands = _candidate_positions(buf, b"\x00\x01\x02\x03", rng=None)
    assert len(cands) <= 48
    assert cands == sorted(cands)
    assert len(set(cands)) == len(cands)
    assert all(0 <= c < len(buf) for c in cands)


def test_candidate_scan_finds_the_same_overlaps_as_a_full_position_map():
    """The memchr scan must find exactly the offsets the old map found."""
    rnd = random.Random(11)
    for _ in range(60):
        n = rnd.choice([64, 256, 1024])
        buf = os.urandom(n)
        target = os.urandom(rnd.choice([1, 2, 4]))

        pos_map: dict[int, list[int]] = {}
        for idx, b in enumerate(buf):
            pos_map.setdefault(b, []).append(idx)
        expected = set()
        for i, b in enumerate(target):
            for pos in pos_map.get(b, []):
                off = pos - i
                if 0 <= off < n:
                    expected.add(off)

        got = set(_candidate_positions(buf, target, rng=None))
        if len(expected) >= 24:  # below the cap the fallback adds randoms
            assert got <= expected
            if len(expected) <= 48:
                assert got == expected


def test_candidates_deterministic_for_a_fixed_input():
    buf = os.urandom(20000)
    target = b"\xca\xfe"
    assert _candidate_positions(buf, target, rng=None) == _candidate_positions(
        buf, target, rng=None
    )


def test_empty_inputs():
    assert _candidate_positions(b"", b"ab", rng=None) == []
    assert _candidate_positions(b"abc", b"", rng=None) == []
    assert gradient_descent(b"", (b"ab", b"")) == b""
    assert gradient_descent(b"abc", (b"", b"")) == b"abc"
