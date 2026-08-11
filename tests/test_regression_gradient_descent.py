"""Regression tests for ``fuzzer_tool.core.gradient_descent``."""

from __future__ import annotations

from fuzzer_tool.core.gradient_descent import gradient_descent


class TestGradientDescent:
    """Port and behavior checks for the Angora GdSearch port."""

    def test_returns_bytes(self):
        out = gradient_descent(b"hello", (b"h", b"q"))
        assert isinstance(out, bytes)

    def test_noop_when_empty_buf(self):
        out = gradient_descent(b"", (b"a", b"b"))
        assert out == b""

    def test_noop_when_empty_pair(self):
        out = gradient_descent(b"hello", (b"", b""))
        assert out == b"hello"

    def test_improves_toward_target(self):
        # Target byte 'A' (0x41) is 6 bits away from 'h' (0x68).
        # Gradient descent should move toward it.
        out = gradient_descent(b"hello world", (b"h", b"A"))
        assert len(out) == len(b"hello world")
        # At least one byte position should differ from input.
        assert out != b"hello world"

    def test_exact_match_stops_early(self):
        # When input already matches one operand, distance is 0.
        out = gradient_descent(b"exact", (b"exact", b"nomatch"))
        assert out == b"exact"

    def test_length_respects_max_len(self):
        out = gradient_descent(b"short", (b"x" * 20, b"y" * 20), max_len=3)
        assert len(out) <= 3

    def test_interesting_value_escape(self):
        # Local-minimum escape: start at a position with no single-step path.
        # An interesting value should break the plateau.
        buf = bytes([0x7F] * 8)
        target = bytes([0x00] * 8)
        out = gradient_descent(buf, (b"\x00" * 8, target))
        # At least one byte should have changed after the interesting-value pass.
        assert out != buf

    def test_deterministic_with_same_input(self):
        buf = b"determinism check"
        pair = (b"d", b"D")
        a = gradient_descent(buf, pair)
        b = gradient_descent(buf, pair)
        assert a == b

    def test_prefers_shorter_operand(self):
        # When operands differ in length, the shorter is the default target.
        buf = b"aaaa"
        out = gradient_descent(buf, (b"aaaaaa", b"bb"))
        assert len(out) == len(buf)

    def test_candidates_fallback_random(self):
        # Input with no overlapping byte values to target should still
        # produce a mutated output via the random fallback path.
        buf = bytes([0xFF] * 16)
        target = bytes([0x00] * 8)
        out = gradient_descent(buf, (b"\x00" * 8, target))
        # Output should differ because random candidates are injected.
        assert out != buf


class TestOperandAtNonZeroOffset:
    """The original objective was blind to everything past the operand width.

    ``_distance(buf, target)`` compared the whole buffer against a short
    operand: Hamming over ``buf[:len(target)]`` plus a constant length
    delta. Mutating any byte at or beyond ``len(target)`` therefore left
    the score unchanged, so no perturbation there was ever accepted --
    the descent could only ever modify the first few bytes of the input,
    even though ``_candidate_positions`` correctly identified sites deep
    in the buffer. Every original test happened to place the operand at
    offset 0, which masked this completely.
    """

    def test_solves_operand_deep_in_buffer(self):
        target = b"\xde\xad\xbe\xef"
        buf = bytearray(b"A" * 1024)
        buf[600:604] = b"\xde\xad\xbe\xee"  # one byte short of the target
        out = gradient_descent(bytes(buf), (target, target), max_len=4096)
        assert out[600:604] == target

    def test_modifies_bytes_beyond_operand_width(self):
        """Directly pins the blind spot: the byte that must change sits
        far past len(target), where the old objective had zero gradient."""
        target = b"\x11\x22\x33\x44"
        buf = bytearray(b"\x00" * 512)
        buf[300:304] = b"\x11\x22\x33\x40"
        inp = bytes(buf)
        out = gradient_descent(inp, (target, target), max_len=4096)
        changed = [i for i in range(len(inp)) if inp[i] != out[i]]
        assert changed, "descent made no change at all"
        assert all(i >= len(target) for i in changed)

    def test_picks_best_matching_site_among_several(self):
        """With several partial matches, the descent should anchor on the
        closest one rather than an arbitrary or first-found position."""
        target = b"\xaa\xbb\xcc\xdd"
        buf = bytearray(b"\x00" * 600)
        buf[100:104] = b"\xaa\x00\x00\x00"  # 1/4 match
        buf[400:404] = b"\xaa\xbb\xcc\xd5"  # near-exact
        out = gradient_descent(bytes(buf), (target, target), max_len=4096)
        assert out[400:404] == target

    def test_length_is_preserved(self):
        target = b"\x77\x88"
        buf = bytes(b"z" * 300)
        out = gradient_descent(buf, (target, target), max_len=4096)
        assert len(out) == len(buf)
