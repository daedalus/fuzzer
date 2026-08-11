"""The gradient ladder must be able to reach every bit.

``_window_distance`` is a bitwise Hamming distance; ``_STEPS`` are
arithmetic perturbations. They agree only where the step is a
non-carrying power of two, so a ladder stopping at 8 cannot express the
single-bit correction for bits 4-7 at any byte value -- and the descent
then stalls on roughly half of all operand bytes.
"""

from __future__ import annotations

import random

from fuzzer_tool.core.gradient_descent import _STEPS, gradient_descent
from fuzzer_tool.core.rand_pool import RandPool

OPERAND = b"\xde\xad\xbe\xef"


class TestLadderCoverage:
    def test_every_single_bit_correction_is_reachable(self):
        """For each bit and each byte value, some step reaches the flip."""
        for orig in range(256):
            for bit in range(8):
                want = orig ^ (1 << bit)
                assert any(max(0, min(255, orig + d)) == want for d in _STEPS), (
                    f"no step reaches bit {bit} from {orig:#04x}"
                )

    def test_steps_are_signed_powers_of_two(self):
        assert sorted(abs(s) for s in _STEPS) == sorted([1, 2, 4, 8, 16, 32, 64, 128] * 2)

    def test_small_steps_come_first(self):
        """Ordering matters: the loop takes the first improving step it
        finds, so a large step must not pre-empt a small one."""
        magnitudes = [abs(s) for s in _STEPS]
        assert magnitudes == sorted(magnitudes)


class TestDescentSolves:
    def _planted(self, rng, n_matching):
        buf = bytearray(rng.randrange(256) for _ in range(64))
        off = rng.randrange(0, 60)
        for i in range(64):
            while buf[i] in OPERAND:
                buf[i] = rng.randrange(256)
        for i in range(n_matching):
            buf[off + i] = OPERAND[i]
        return bytes(buf), off

    def test_solves_when_two_bytes_already_match(self):
        """Regression floor. Measured at 85.8% with the full ladder and
        1.2% without, so 50% separates the two by a wide margin without
        being a seed-sensitive assertion."""
        rng = random.Random(99)
        pool = RandPool()
        hits = 0
        trials = 200
        for _ in range(trials):
            buf, off = self._planted(rng, 2)
            out = gradient_descent(buf, (OPERAND, OPERAND), rng=pool)
            hits += out[off : off + 4] == OPERAND
        assert hits / trials > 0.5

    def test_plants_the_operand_somewhere_almost_always(self):
        rng = random.Random(100)
        pool = RandPool()
        hits = 0
        trials = 200
        for _ in range(trials):
            buf, _ = self._planted(rng, 1)
            hits += OPERAND in gradient_descent(buf, (OPERAND, OPERAND), rng=pool)
        assert hits / trials > 0.8


class TestPickTarget:
    def test_empty_operand_does_not_win_by_being_shortest(self):
        """A bare min-by-length picks the zero-length side and no-ops."""
        from fuzzer_tool.core.gradient_descent import pick_target

        assert pick_target((OPERAND, b"")) == OPERAND
        assert pick_target((b"", OPERAND)) == OPERAND
        assert pick_target((b"", b"")) == b""
        assert pick_target((b"longer", b"short")) == b"short"

    def test_mb_cbh_shares_the_definition(self):
        from fuzzer_tool.core.gradient_descent import pick_target
        from fuzzer_tool.core.mb_cbh import _pick_target

        assert _pick_target is pick_target
