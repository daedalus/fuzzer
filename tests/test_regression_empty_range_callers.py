"""Callers that asked RandPool for empty ranges (exposed 2026-11-12).

``randint(a, b < a)`` used to return ``a``; it now raises like CPython.
``structured._lz_edit`` and ``GzipMutator._generate_random_gzip`` relied on
the old silent clamp: the first aborted a fuzz campaign via
``_op_lz_dict_mutate``.
"""

import zlib

import pytest

from fuzzer_tool.core.mutations.gzip import GzipMutator
from fuzzer_tool.core.mutations.structured import _lz_edit
from fuzzer_tool.core.rand_pool import RandPool

MIN_MATCH = 3
MAX_MATCH = 64


class _BoundsRng:
    """Answers each ``randint`` with the scripted end ("lo"/"hi"; lo when the
    script runs out), records ``(a, b)``, and raises on an empty range."""

    def __init__(self, picks=()):
        self.picks = iter(picks)
        self.bounds: list[tuple[int, int]] = []

    def randint(self, a: int, b: int) -> int:
        if b < a:
            raise ValueError(f"empty range for randint({a}, {b})")
        self.bounds.append((a, b))
        return b if next(self.picks, "lo") == "hi" else a


# ── _lz_edit ─────────────────────────────────────────────────────────────


def test_regression_lz_edit_literal_at_pos_zero():
    """pos 0 has no history: distance clamps to 1 (was randint(1, 0))."""
    tokens = [("lit", 65)]
    _lz_edit(tokens, bytearray(b"A" * 8), MIN_MATCH, MAX_MATCH, _BoundsRng())
    assert tokens == [("ref", 1, MIN_MATCH)]


def test_regression_lz_edit_short_tail():
    """Adversarial: fewer than min_match bytes left (the randint(3, 1) crash)."""
    block = bytearray(b"AB")
    tokens = [("lit", 65), ("lit", 66)]
    # n_mut=1, pos=1 (one byte left), dist=1, run_len
    _lz_edit(tokens, block, MIN_MATCH, MAX_MATCH, _BoundsRng(["lo", "hi"]))
    assert tokens == [("lit", 65), ("ref", 1, MIN_MATCH)]


def test_regression_lz_edit_normal_bounds_unchanged():
    """Falsification: in-range literal keeps its original bounds."""
    block = bytearray(range(100))
    tokens = [("lit", b) for b in block[:10]]
    rng = _BoundsRng(["lo", "hi"])
    _lz_edit(tokens, block, MIN_MATCH, MAX_MATCH, rng)

    # n_mut, pos, dist, run_len for the one edit (pos = last token).
    pos = len(tokens) - 1
    assert rng.bounds == [(1, 4), (0, pos), (1, pos), (MIN_MATCH, MAX_MATCH)]


def test_regression_lz_dict_mutate_short_buffers():
    """Wiring: the operator entry point survives every short length."""
    from fuzzer_tool.core.mutations.structured import lz_dict_mutate

    for n in range(8, 64):
        out = lz_dict_mutate(bytes(n), RandPool(seed=n))
        assert len(out) == n


# ── GzipMutator._generate_random_gzip ────────────────────────────────────


@pytest.mark.parametrize("max_len", [0, 1, 10, 20, 21])
def test_regression_gzip_tiny_max_len(max_len):
    """max_len below the gzip overhead still yields bytes within max_len."""
    out = GzipMutator().mutate(b"not gzip", max_len=max_len, rng=RandPool(seed=1))
    assert isinstance(out, bytes)
    assert len(out) <= max_len


def test_regression_gzip_tiny_payload_is_one_byte():
    """Adversarial: the payload floor is 1 byte, decodable before truncation."""
    out = GzipMutator()._generate_random_gzip(max_len=5, rng=RandPool(seed=2))
    assert len(zlib.decompress(out, 31)) == 1


def test_regression_gzip_normal_max_len_unchanged():
    """Falsification: roomy max_len still draws payloads up to 128 bytes."""
    rng = _BoundsRng()
    GzipMutator()._generate_random_gzip(max_len=4096, rng=rng)
    assert rng.bounds[0] == (1, 128)


# ── ZlibMutator._generate_random_zlib (same pattern, ~10-byte overhead) ──


@pytest.mark.parametrize("max_len", [0, 1, 10, 11])
def test_regression_zlib_tiny_max_len(max_len):
    from fuzzer_tool.core.mutations.zlib import ZlibMutator

    out = ZlibMutator().mutate(b"not zlib", max_len=max_len, rng=RandPool(seed=1))
    assert isinstance(out, bytes)
    assert len(out) <= max_len


def test_regression_zlib_normal_max_len_unchanged():
    """Falsification: roomy max_len still draws payloads up to 128 bytes."""
    from fuzzer_tool.core.mutations.zlib import ZlibMutator

    rng = _BoundsRng()
    ZlibMutator()._generate_random_zlib(max_len=4096, rng=rng)
    assert rng.bounds[0] == (1, 128)
