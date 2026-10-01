"""P1-1 regression: deterministic mutation stream must not allocate per mutant.

The old implementation did ``bytearray(data)`` + ``bytes(mutant)`` for every
mutant (O(n²) bytes copied).  The optimized version holds one persistent
scratch buffer, mutates in place, yields ``bytes(scratch)``, and restores.

This test proves equivalence mutant-by-mutant against a reference
implementation that copies per mutant (``tests/support/det_reference.py``), over a sweep of
seed lengths and cap values including caps that land mid-pass.
"""

from itertools import zip_longest

import pytest

from fuzzer_tool.services.operators import _deterministic_mutation_stream
from tests.support.det_reference import reference_stream as _reference_deterministic_mutation_stream


class TestDeterministicMutationEquivalence:
    """Mutant-by-mutant equivalence against the reference implementation."""

    @pytest.mark.parametrize("length", [1, 7, 16, 100, 256, 1000, 4096])
    @pytest.mark.parametrize("cap", [1, 7, 15, 50, 100, 65536, 200_000])
    def test_equivalence_sweep(self, length, cap):
        """Every seed length × cap combination must match the reference."""
        data = bytes(range(256)) * (length // 256 + 1)
        data = data[:length]

        # Streamed pairwise: a 4 KiB seed under a 200k cap would hold two
        # ~800 MB lists.
        pairs = zip_longest(
            _reference_deterministic_mutation_stream(data, cap),
            _deterministic_mutation_stream(data, cap),
        )
        for i, (expected, actual) in enumerate(pairs):
            assert actual == expected, f"length={length}, cap={cap}: first diff at mutant {i}"

    def test_empty_seed(self):
        assert list(_deterministic_mutation_stream(b"", 1000)) == []
        assert _deterministic_mutation_stream.last_truncated == 0

    def test_no_cap(self):
        """When the schedule fits under the cap, every pass runs to completion."""
        data = b"\x00\x01\x02\x03"
        expected = list(_reference_deterministic_mutation_stream(data, 1_000_000))
        actual = list(_deterministic_mutation_stream(data, 1_000_000))
        assert actual == expected
        assert _deterministic_mutation_stream.last_truncated == 0

    def test_mid_pass_truncation_bitflip(self):
        """Cap lands inside the bitflip pass: only some bits of some bytes flip."""
        data = b"\x00\x00\x00"
        # 3 bytes * 8 bits = 24 mutants; cap at 10 should stop mid-bitflip.
        expected = list(_reference_deterministic_mutation_stream(data, 10))
        actual = list(_deterministic_mutation_stream(data, 10))
        assert actual == expected
        assert len(actual) == 10

    def test_mid_pass_truncation_arithmetic(self):
        """Cap lands inside the arithmetic pass: +delta consumed, -delta skipped."""
        # Use a seed where bitflip+byteflip consume less than cap, arithmetic exceeds.
        data = b"A" * 10  # bit=80, byte=10, arith=320
        cap = 100  # consumes all bitflip (80) + 20 byteflip
        expected = list(_reference_deterministic_mutation_stream(data, cap))
        actual = list(_deterministic_mutation_stream(data, cap))
        assert actual == expected

    def test_mid_pass_truncation_after_byteflip(self):
        """Cap lands inside arithmetic: verify the scratch is restored correctly."""
        data = b"\x42" * 5  # bit=40, byte=5, arith=80
        cap = 50  # all bitflip + all byteflip + 5 arithmetic
        expected = list(_reference_deterministic_mutation_stream(data, cap))
        actual = list(_deterministic_mutation_stream(data, cap))
        assert actual == expected

    def test_each_mutant_differs_within_one_window(self):
        """Structural invariant: every mutant changes one window of <= 4 bytes."""
        data = b"The quick brown fox jumps over the lazy dog."
        for mutant in _deterministic_mutation_stream(data, 1000):
            diffs = [i for i, (a, b) in enumerate(zip(mutant, data, strict=True)) if a != b]
            assert diffs and diffs[-1] - diffs[0] < 4
