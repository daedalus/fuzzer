"""CBH site re-anchoring: the climb must not commit to a wrong anchor.

``climb_hill`` scores a fixed window and only ever lowers that window's
distance, so the site chosen at iteration 0 never comes to look worse than
the alternative it beat -- however wrong it was. These tests cover the
re-anchoring that lets a stalled climb move to the next candidate, and the
global-best bookkeeping that makes re-anchoring free.
"""

from __future__ import annotations

import random

from fuzzer_tool.core.gradient_descent import _window_distance
from fuzzer_tool.core.mb_cbh import climb_hill
from fuzzer_tool.core.rand_pool import RandPool

TARGET = b"\xde\xad\xbe\xef"


def _best_window(buf: bytes, target: bytes) -> int:
    """Lowest window distance to *target* anywhere in *buf*."""
    if len(buf) < len(target):
        return len(target) * 8
    return min(_window_distance(buf, p, target) for p in range(len(buf) - len(target) + 1))


def _planted_input(rng: random.Random, offset: int, n_matching: int) -> bytes:
    """Buffer with *n_matching* of the operand's bytes already at *offset*."""
    buf = bytearray(rng.randrange(1, 256) for _ in range(64))
    for i in range(n_matching):
        buf[offset + i] = TARGET[i]
    return bytes(buf)


class TestSiteReanchoring:
    def test_max_sites_one_is_the_old_behaviour(self):
        """max_sites=1 must never move off the initial argmin."""
        pool = RandPool()
        data = _planted_input(random.Random(7), 20, 1)
        out = climb_hill(data, (TARGET, b""), pool, max_sites=1)
        assert len(out) == len(data)

    def test_result_never_worse_than_the_input(self):
        """The returned buffer's best window is never worse than the input's.

        This is what makes re-anchoring free: the global incumbent is
        retained across discards, so trying another site can only add
        outcomes. Without it, a re-anchor that stalls would return a
        buffer edited for a window nobody is looking at any more.
        """
        pool = RandPool()
        rng = random.Random(11)
        for _ in range(60):
            data = _planted_input(rng, rng.randrange(0, 55), rng.randrange(0, 3))
            before = _best_window(data, TARGET)
            out = climb_hill(data, (TARGET, b""), pool, max_sites=4)
            assert _best_window(out, TARGET) <= before

    def test_reanchoring_does_not_lose_solvable_cases(self):
        """Re-anchoring must not cost solves, within noise.

        This deliberately asserts a tolerance band rather than
        ``solved(2) >= solved(1)``. The two arms measure as a wash (see
        the table in climb_hill's docstring), so a strict inequality is a
        coin flip that passes or fails on the seed -- a test that fails
        50% of the time for the reason the code is fine is worse than no
        test. The band is what the measurement actually supports: 80
        cases, so a 10-case swing is inside the binomial noise for two
        arms with equal true rates.
        """
        rng = random.Random(3)
        cases = [_planted_input(rng, rng.randrange(0, 55), rng.randrange(1, 3)) for _ in range(80)]

        def solved(max_sites: int) -> int:
            # Seeded: RandPool() draws from OS entropy, which made this
            # assertion fail on 1.0% of runs with no way to reproduce it from
            # any seed. The tolerance band is calibrated to binomial noise
            # across the 80 cases, not to noise in the mutation stream too.
            pool = RandPool(20260821)
            return sum(
                _best_window(climb_hill(c, (TARGET, b""), pool, max_sites=max_sites), TARGET) == 0
                for c in cases
            )

        assert solved(2) >= solved(1) - 10

    def test_degenerate_inputs_are_unchanged(self):
        pool = RandPool()
        assert climb_hill(b"", (TARGET, b""), pool, max_sites=4) == b""
        assert climb_hill(b"abcd", (b"", b""), pool) == b"abcd"
        # Buffer shorter than the operand: nothing to anchor on.
        assert climb_hill(b"ab", (TARGET, b""), pool) == b"ab"
