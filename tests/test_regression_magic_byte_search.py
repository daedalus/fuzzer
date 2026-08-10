"""Regression tests for the Angora MB/CBH search port (`core/mb_cbh.py`)."""

from __future__ import annotations

import random

from fuzzer_tool.core.mb_cbh import _pick_target, climb_hill, magic_byte_search
from fuzzer_tool.core.rand_pool import RandPool


class _Rng:
    """Deterministic stand-in for the fuzzer's rand pool."""

    def __init__(self, seed: int = 1234):
        self._r = random.Random(seed)

    def randint(self, a, b):
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)

    def randrange_list(self, n, count):
        # Mirrors RandPool.randrange_list, used by the sparse-overlap
        # fallback in _candidate_positions.
        return [self._r.randrange(n) for _ in range(count)]


class TestPickTarget:
    def test_prefers_shorter_operand(self):
        assert _pick_target((b"aaaaaa", b"bb")) == b"bb"
        assert _pick_target((b"cc", b"dddddd")) == b"cc"

    def test_handles_one_empty_operand(self):
        assert _pick_target((b"", b"abc")) == b"abc"
        assert _pick_target((b"abc", b"")) == b"abc"

    def test_both_empty(self):
        assert _pick_target((b"", b"")) == b""


class TestMagicByteSearch:
    def test_plants_operand_somewhere_in_output(self):
        target = b"\xde\xad\xbe\xef"
        buf = b"\x00" * 256
        out = magic_byte_search(buf, (target, target), _Rng(), max_len=4096)
        assert target in out

    def test_noop_on_empty_input(self):
        assert magic_byte_search(b"", (b"ab", b"ab"), _Rng()) == b""

    def test_noop_on_empty_pair(self):
        assert magic_byte_search(b"hello", (b"", b""), _Rng()) == b"hello"

    def test_input_shorter_than_operand_returns_unchanged(self):
        assert magic_byte_search(b"ab", (b"abcdefgh", b"abcdefgh"), _Rng()) == b"ab"

    def test_respects_max_len(self):
        out = magic_byte_search(b"x" * 500, (b"ab", b"ab"), _Rng(), max_len=64)
        assert len(out) <= 64

    def test_plants_even_with_no_byte_overlap(self):
        """Fallback path: input shares no bytes with the operand, so the
        overlap-derived candidate list is empty and a random offset is
        used instead. The operand must still get planted."""
        target = b"\xaa\xbb"
        buf = b"\x01" * 128
        out = magic_byte_search(buf, (target, target), _Rng(), max_len=4096)
        assert target in out

    def test_randomization_does_not_clobber_planted_bytes(self):
        """Across many seeds the planted operand must survive the
        surrounding randomization every time."""
        target = b"\x11\x22\x33\x44"
        buf = b"\x00" * 128
        for seed in range(40):
            out = magic_byte_search(buf, (target, target), _Rng(seed), max_len=4096)
            assert target in out, f"operand lost at seed {seed}"

    def test_length_preserved(self):
        buf = b"z" * 300
        out = magic_byte_search(buf, (b"\x07\x08", b"\x07\x08"), _Rng(), max_len=4096)
        assert len(out) == len(buf)


class TestClimbHill:
    def test_solves_operand_deep_in_buffer(self):
        """The whole point: the operand sits far from offset 0 and one
        byte away from matching. CBH must close that gap."""
        target = b"\xde\xad\xbe\xef"
        buf = bytearray(b"A" * 1024)
        buf[600:604] = b"\xde\xad\xbe\xee"
        out = climb_hill(bytes(buf), (target, target), _Rng(), max_len=4096)
        assert out[600:604] == target

    def test_noop_on_empty_input(self):
        assert climb_hill(b"", (b"ab", b"ab"), _Rng()) == b""

    def test_noop_on_empty_pair(self):
        assert climb_hill(b"hello", (b"", b""), _Rng()) == b"hello"

    def test_exact_match_returns_immediately(self):
        buf = b"prefix\xaa\xbbsuffix"
        out = climb_hill(buf, (b"\xaa\xbb", b"\xaa\xbb"), _Rng(), max_len=4096)
        assert out == buf

    def test_never_worsens_the_objective(self):
        """Hill climbing accepts only improvements, so the returned input
        can never score worse than the input it started from."""
        from fuzzer_tool.core.gradient_descent import _candidate_positions, _window_distance

        target = b"\x10\x20\x30"
        buf = bytearray(b"\x00" * 200)
        buf[80:83] = b"\x10\x25\x35"
        inp = bytes(buf)

        cands = [p for p in _candidate_positions(inp, target) if p + 3 <= len(inp)]
        site = min(cands, key=lambda p: _window_distance(inp, p, target))
        before = _window_distance(inp, site, target)

        out = climb_hill(inp, (target, target), _Rng(), max_len=4096)
        after = _window_distance(out, site, target)
        assert after <= before

    def test_length_preserved(self):
        buf = b"q" * 400
        out = climb_hill(buf, (b"\x01\x02", b"\x01\x02"), _Rng(), max_len=4096)
        assert len(out) == len(buf)

    def test_respects_max_len(self):
        out = climb_hill(b"x" * 500, (b"ab", b"ab"), _Rng(), max_len=64)
        assert len(out) <= 64


class TestRegistration:
    def test_both_operators_registered(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        names = REGISTRY.names()
        assert "magic_byte_search" in names
        assert "climb_hill" in names

    def test_categorized_as_adaptive(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        assert REGISTRY.category_of("magic_byte_search") == "adaptive"
        assert REGISTRY.category_of("climb_hill") == "adaptive"

    def test_gated_on_cmplog(self):
        """Both need cmplog pairs; with no cmplog they must not be offered."""
        from fuzzer_tool.core.operator_registry import REGISTRY

        class _NoCmplog:
            _cmplog = None

        avail = REGISTRY.available(_NoCmplog(), b"data")
        assert "magic_byte_search" not in avail
        assert "climb_hill" not in avail


class TestSeededReproducibility:
    """Candidate-site selection must draw from the fuzzer's seeded pool.

    `_candidate_positions` used to fall back to `random.sample()` on the
    `random` module's *global* state when input/operand byte overlap was
    sparse. RandPool refills from numpy's global RNG and `-s` seeds numpy
    (fuzzer.py: `np.random.seed(seed)`), so the fallback was outside the
    seeded path entirely: pinning the campaign seed did not pin these
    operators. Measured before the fix: with the fuzzer rng pinned and
    only the Python global rng perturbed, six trials gave five distinct
    outputs. Affected gradient_descent, and magic_byte_search/climb_hill
    once they began sharing the same helper.
    """

    # No byte overlap between input and operand, so the sparse-overlap
    # fallback is guaranteed to fire.
    _BUF = b"\x01" * 128
    _TARGET = b"\xaa\xbb"

    def _under_pinned_seed(self, fn):
        outs = set()
        for _trial in range(6):
            outs.add(fn(RandPool(seed=42)))
        return outs

    def test_magic_byte_search_is_reproducible(self):
        outs = self._under_pinned_seed(
            lambda r: magic_byte_search(self._BUF, (self._TARGET, self._TARGET), r, max_len=4096)
        )
        assert len(outs) == 1

    def test_climb_hill_is_reproducible(self):
        outs = self._under_pinned_seed(
            lambda r: climb_hill(self._BUF, (self._TARGET, self._TARGET), r, max_len=4096)
        )
        assert len(outs) == 1

    def test_gradient_descent_is_reproducible(self):
        from fuzzer_tool.core.gradient_descent import gradient_descent

        outs = self._under_pinned_seed(
            lambda r: gradient_descent(self._BUF, (self._TARGET, self._TARGET), max_len=4096, rng=r)
        )
        assert len(outs) == 1

    def test_candidate_positions_uses_supplied_rng(self):
        """Directly: two different seeded pools must disagree, proving the
        fallback reads the supplied rng rather than ignoring it."""
        import numpy as np

        from fuzzer_tool.core.gradient_descent import _candidate_positions
        from fuzzer_tool.core.rand_pool import RandPool

        np.random.seed(1)
        a = _candidate_positions(self._BUF, self._TARGET, RandPool())
        np.random.seed(2)
        b = _candidate_positions(self._BUF, self._TARGET, RandPool())
        assert a != b
