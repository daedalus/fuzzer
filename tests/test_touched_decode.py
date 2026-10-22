"""_touched_slots bit decode: both paths against an independent full unpack.

The decoder strips the lowest set bit per word while every word has at most
_LOWBIT_MAX_POP bits, and unpacks whole words otherwise. Words straddling that
cap, full words, bit 0/63 and padding past num_entries are the edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import _LOWBIT_MAX_POP
from tests.test_shm_touched_scan import _cov

ALL_ONES = np.uint64(0xFFFFFFFFFFFFFFFF)


def _oracle(words: np.ndarray, n: int) -> np.ndarray:
    """Every set bit below *n*, by unpacking the whole bitmap."""
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")
    return np.flatnonzero(bits[:n])


def _word(bits: list[int]) -> np.uint64:
    return np.uint64(sum(1 << b for b in bits))


def _decode(size: int, fill) -> tuple[np.ndarray, np.ndarray]:
    """(decoded, oracle) after *fill* writes the bitmap of a fresh segment."""
    cov = _cov(size)
    try:
        words = cov._touched_view()
        fill(words)
        return cov._touched_slots().copy(), _oracle(words.copy(), size)
    finally:
        cov.cleanup()


class TestDecode:
    @pytest.mark.parametrize("pop", [1, _LOWBIT_MAX_POP, _LOWBIT_MAX_POP + 1, 64])
    def test_popcount_at_and_past_the_cap(self, pop):
        bits = list(range(0, 64, 64 // pop))[:pop] if pop < 64 else list(range(64))
        bits[-1] = 63  # top bit always set

        def fill(w):
            w[0] = _word(bits)
            w[5] = _word([0])
            w[-1] = _word([63])

        got, want = _decode(4096, fill)
        assert got.tolist() == want.tolist()

    def test_ascending_across_mixed_words(self):
        def fill(w):
            w[1] = _word([63, 0, 31])
            w[2] = ALL_ONES
            w[3] = _word([7])

        got, want = _decode(4096, fill)
        assert np.all(np.diff(got) > 0)
        assert got.tolist() == want.tolist()

    @pytest.mark.parametrize("dense", [False, True])
    def test_padding_dropped_on_both_paths(self, dense):
        size = 1000  # last word covers 960..1023

        def fill(w):
            w[-1] = ALL_ONES if dense else _word([1, 50, 63])

        got, want = _decode(size, fill)
        assert got.tolist() == want.tolist()
        assert got.max() < size

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("density", [0.001, 0.05, 0.5])
    def test_random_bitmaps(self, seed, density):
        rng = np.random.default_rng(seed)
        size = 65536

        def fill(w):
            slots = np.flatnonzero(rng.random(size) < density)
            np.bitwise_or.at(w, slots >> 6, np.uint64(1) << (slots & 63).astype(np.uint64))

        got, want = _decode(size, fill)
        assert got.tolist() == want.tolist()

    def test_empty(self):
        got, want = _decode(4096, lambda w: None)
        assert got.size == 0 == want.size


class TestOracle:
    def test_control_agrees_with_itself(self):
        w = np.array([_word([0, 63]), ALL_ONES], dtype=np.uint64)
        assert _oracle(w, 128).tolist() == _oracle(w.copy(), 128).tolist()

    def test_falsification_one_flipped_bit_is_seen(self):
        def fill(w):
            w[0] = _word([3, 40])

        got, want = _decode(4096, fill)
        flipped = np.zeros(4096 // 64, dtype=np.uint64)
        flipped[0] = _word([3])
        assert got.tolist() != _oracle(flipped, 4096).tolist()
        assert got.tolist() == want.tolist()
