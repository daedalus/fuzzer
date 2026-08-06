"""Regression tests: vectorized partial_parse equivalence.

`partial_parse` used to branch on every byte, appending non-delimiter
bytes one at a time (~2.7M list.append calls per 8k executions in a
profile). Only 8 byte values can affect the tree structure — the
delimiter opens and their closes — so the rewrite locates those
positions in one vectorized pass and copies the literal runs between
them as whole slices.

That is a pure performance change: the parse must produce byte-identical
output. This module pins that with a reference implementation of the
original per-byte algorithm plus randomized differential testing, because
the failure mode is subtle — a slicing off-by-one would corrupt mutated
inputs rather than raise, and `tree_mutator` has already had one real
round-trip bug (an end_sequence-style phantom byte) go unnoticed.

Also pinned: the numpy path and the scalar fallback must agree, since
short inputs and numpy-less environments take the latter.
"""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core import tree_mutator as tm
from fuzzer_tool.core.tree_mutator import (
    _BYTE_BYTES,
    _CLOSE_TABLE,
    _DELIM_CLOSE,
    _Node,
    partial_parse,
)


def reference_parse(data: bytes) -> _Node:
    """The original per-byte implementation, kept as the oracle."""
    root = _Node()
    stack = [root]
    i = 0
    buf: list[bytes] = []

    def flush():
        if buf:
            chunk = b"".join(buf)
            if stack:
                stack[-1].children.append(chunk)
            buf.clear()

    n = len(data)
    while i < n:
        byte = data[i]
        close = _DELIM_CLOSE[byte]
        if close != 0xFF:
            if byte == close:
                if stack and stack[-1].open == byte:
                    flush()
                    if len(stack) > 1:
                        stack[-1].closed = True
                        stack.pop()
                else:
                    flush()
                    node = _Node(byte)
                    stack[-1].children.append(node)
                    stack.append(node)
            else:
                flush()
                node = _Node(byte)
                stack[-1].children.append(node)
                stack.append(node)
        elif stack:
            top = stack[-1]
            if top.open is not None and byte == _CLOSE_TABLE[top.open]:
                flush()
                if len(stack) > 1:
                    top.closed = True
                    stack.pop()
            else:
                buf.append(_BYTE_BYTES[byte])
        else:
            buf.append(_BYTE_BYTES[byte])
        i += 1

    flush()
    return root


ALPHABET = b"abcXY({[\")]}'\x00\xff"


class TestEquivalence:
    def test_matches_reference_on_random_inputs(self):
        rng = random.Random(7)
        for _ in range(3000):
            n = rng.randint(0, 300)
            data = bytes(rng.choice(ALPHABET) for _ in range(n))
            assert partial_parse(data).flatten() == reference_parse(data).flatten(), (
                f"diverged on {data[:60]!r}"
            )

    def test_matches_reference_on_binary_data(self):
        """Binary payloads are the common case and have few delimiters —
        the path where bulk slicing does the most work."""
        rng = random.Random(11)
        for size in (64, 256, 1024, 4096):
            data = bytes(rng.randint(0, 255) for _ in range(size))
            assert partial_parse(data).flatten() == reference_parse(data).flatten()

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"a",
            b"abc",
            b"(",
            b")",
            b"()",
            b"(abc)",
            b"[a{b}c]",
            b'"quoted"',
            b"'single'",
            b"((((",
            b"))))",
            b"(unclosed",
            b"unopened)",
            b"a" * 500,
            bytes(range(256)),
            b"(" * 200,
            b'{"k":[1,2]}',
        ],
    )
    def test_matches_reference_on_edge_cases(self, data):
        assert partial_parse(data).flatten() == reference_parse(data).flatten()


class TestRoundTrip:
    """Inputs with no structural bytes must survive byte-identical."""

    def test_plain_bytes_round_trip_exactly(self):
        rng = random.Random(3)
        plain = bytes(b for b in range(256) if _DELIM_CLOSE[b] == 0xFF and b not in b"\")]}'")
        for size in (1, 63, 64, 65, 512, 4096):
            data = bytes(rng.choice(plain) for _ in range(size))
            assert partial_parse(data).flatten() == data

    def test_balanced_delimiters_round_trip(self):
        for data in (b"(a)", b"[b]", b"{c}", b"(a(b)c)", b'"q"'):
            assert partial_parse(data).flatten() == data


class TestVectorAndScalarPathsAgree:
    """Short inputs and numpy-less environments use the scalar fallback."""

    def test_paths_agree_across_the_length_threshold(self):
        rng = random.Random(5)
        threshold = tm._VECTOR_MIN_LEN
        for size in (threshold - 2, threshold - 1, threshold, threshold + 1, threshold * 3):
            data = bytes(rng.choice(ALPHABET) for _ in range(size))
            assert partial_parse(data).flatten() == reference_parse(data).flatten()

    def test_scalar_fallback_matches_numpy_path(self, monkeypatch):
        rng = random.Random(13)
        for _ in range(300):
            data = bytes(rng.choice(ALPHABET) for _ in range(rng.randint(64, 400)))
            vector_positions = tm._interesting_positions(data)
            monkeypatch.setattr(tm, "_np", None)
            scalar_positions = tm._interesting_positions(data)
            monkeypatch.undo()
            assert vector_positions == scalar_positions, f"diverged on {data[:40]!r}"

    def test_positions_are_sorted_and_correct(self):
        rng = random.Random(17)
        data = bytes(rng.choice(ALPHABET) for _ in range(500))
        positions = tm._interesting_positions(data)
        assert positions == sorted(positions)
        expected = [i for i, b in enumerate(data) if tm._INTERESTING_LUT[b]]
        assert positions == expected
