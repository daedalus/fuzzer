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
