"""P1-1 regression: deterministic mutation stream must not allocate per mutant.

The old implementation did ``bytearray(data)`` + ``bytes(mutant)`` for every
mutant (2·33·n² bytes copied).  The optimized version holds one persistent
scratch buffer, mutates in place, yields ``bytes(scratch)``, and restores.

This test proves equivalence mutant-by-mutant against a reference
implementation that uses the old copy-per-mutant policy, over a sweep of
seed lengths and cap values including caps that land mid-pass.
"""

import pytest

from fuzzer_tool.core.mutations import ARITHMETIC_DELTAS, INTERESTING_UNSIGNED_8
from fuzzer_tool.services.operators import _deterministic_mutation_stream


def _reference_deterministic_mutation_stream(data: bytes, max_mutations: int):
    """Reference: the old copy-per-mutant implementation (P1-1 pre-fix)."""
    length = len(data)
    if length == 0:
        return

    n_arith_deltas = len(ARITHMETIC_DELTAS)
    n_interesting = len(INTERESTING_UNSIGNED_8)
    cost_bit = length * 8
    cost_byte = length
    cost_arith = length * n_arith_deltas * 2
    cost_interesting = length * n_interesting
    full_cost = cost_bit + cost_byte + cost_arith + cost_interesting

    if full_cost <= max_mutations:
        quotas = [cost_bit, cost_byte, cost_arith, cost_interesting]
    else:
        costs = [cost_bit, cost_byte, cost_arith, cost_interesting]
        quotas = [int(max_mutations * c / full_cost) for c in costs]
        shortfall = max_mutations - sum(quotas)
        order = sorted(range(4), key=lambda i: costs[i] - quotas[i], reverse=True)
        for i in order:
            if shortfall <= 0:
                break
            add = min(shortfall, costs[i] - quotas[i])
            if add > 0:
                quotas[i] += add
                shortfall -= add

    q_bit, q_byte, q_arith, q_interesting = quotas

    # bitflip
    pass_n = 0
    for byte_idx in range(length):
        if pass_n >= q_bit:
            break
        orig = data[byte_idx]
        for bit in range(8):
            if pass_n >= q_bit:
                break
            mutant = bytearray(data)
            mutant[byte_idx] = orig ^ (1 << bit)
            yield bytes(mutant)
            pass_n += 1

    # byteflip
    pass_n = 0
    for byte_idx in range(length):
        if pass_n >= q_byte:
            break
        mutant = bytearray(data)
        mutant[byte_idx] ^= 0xFF
        yield bytes(mutant)
        pass_n += 1

    # arithmetic
    pass_n = 0
    for byte_idx in range(length):
        if pass_n >= q_arith:
            break
        orig = data[byte_idx]
        for delta in ARITHMETIC_DELTAS:
            if pass_n >= q_arith:
                break
            mutant = bytearray(data)
            mutant[byte_idx] = (orig + delta) & 0xFF
            yield bytes(mutant)
            pass_n += 1
            if pass_n >= q_arith:
                break
            mutant = bytearray(data)
            mutant[byte_idx] = (orig - delta) & 0xFF
            yield bytes(mutant)
            pass_n += 1

    # interesting
    pass_n = 0
    for byte_idx in range(length):
        if pass_n >= q_interesting:
            break
        for val in INTERESTING_UNSIGNED_8:
            if pass_n >= q_interesting:
                break
            mutant = bytearray(data)
            mutant[byte_idx] = val & 0xFF
            yield bytes(mutant)
            pass_n += 1


class TestDeterministicMutationEquivalence:
    """Mutant-by-mutant equivalence against the reference implementation."""

    @pytest.mark.parametrize("length", [1, 7, 16, 100, 256, 1000, 4096])
    @pytest.mark.parametrize("cap", [1, 7, 15, 50, 100, 65536, 200_000])
    def test_equivalence_sweep(self, length, cap):
        """Every seed length × cap combination must match the reference."""
        data = bytes(range(256)) * (length // 256 + 1)
        data = data[:length]

        expected = list(_reference_deterministic_mutation_stream(data, cap))
        actual = list(_deterministic_mutation_stream(data, cap))

        assert actual == expected, (
            f"Mismatch at length={length}, cap={cap}. "
            f"Expected {len(expected)} mutants, got {len(actual)}. "
            f"First diff at index {next(i for i, (a, e) in enumerate(zip(actual, expected)) if a != e)}"
        )

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

    def test_each_mutant_differs_by_exactly_one_position(self):
        """Structural invariant: every mutant is one edit away from the seed."""
        data = b"The quick brown fox jumps over the lazy dog."
        for mutant in _deterministic_mutation_stream(data, 1000):
            diffs = sum(1 for a, b in zip(mutant, data) if a != b)
            # For bitflip, only one bit differs within one byte
            # For byteflip/arithmetic/interesting, one byte differs
            assert diffs <= 1, "Mutant differs at more than one position"
