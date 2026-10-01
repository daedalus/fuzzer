"""Deterministic stage: arith deltas 1..ARITH_MAX and the 16/32-bit passes.

The stage used power-of-two arith deltas -- half of them repeat a bitflip
and they skip the small offsets length fields need -- and had no 16/32-bit
arith or interesting-value pass. Expected mutants come from
``tests/support/det_reference.py``, which copies per mutant and dedups with
whole-mutant sets.
"""

import pytest

from fuzzer_tool.core.mutations import ARITH_MAX, INTERESTING_UNSIGNED_8
from fuzzer_tool.services.operators import (
    _DET_EFF_INERT,
    _DET_EFF_LIVE,
    DeterministicEffectorMap,
    _det_cost_per_byte,
    _deterministic_mutation_stream,
)
from tests.support.det_reference import PER_BYTE, reference_stream


def _stream(data: bytes, cap: int = 10**9) -> list[bytes]:
    return list(_deterministic_mutation_stream(data, cap))


# Edge cases: empty-ish, shorter than a 16/32-bit window, carry-heavy
# 0x00/0xFF runs, palindromes (LE == BE), and a varied seed.
SEEDS = [
    b"\x00",
    b"\x01\xff",
    b"\xff\xff\xff",
    b"\x00" * 6,
    b"\xff" * 4 + b"\x00" * 4,
    b"\x7f\x80\x80\x7f",
    bytes(range(0, 256, 9)),
    b"The quick brown fox",
]


class TestReferenceEquivalence:
    @pytest.mark.parametrize("data", SEEDS)
    def test_full_schedule(self, data):
        assert _stream(data) == list(reference_stream(data))
        assert _deterministic_mutation_stream.last_truncated == 0

    @pytest.mark.parametrize("cap", [1, 9, 50, 333, 1000])
    @pytest.mark.parametrize("data", SEEDS[3:])
    def test_capped_schedule(self, data, cap):
        assert _stream(data, cap) == list(reference_stream(data, cap))

    def test_per_byte_cost(self):
        assert _det_cost_per_byte() == PER_BYTE


class TestArith8:
    def test_falsification_odd_deltas_present(self):
        # Old deltas (powers of two) could never add 3 or 35.
        out = set(_stream(b"\x01"))
        assert b"\x04" in out  # +3
        assert bytes([0x01 + ARITH_MAX]) in out

    def test_bitflip_repeats_dropped(self):
        # 0x00 + 1, 0x00 + 2 are single-bit flips: bitflip pass owns them.
        out = _stream(b"\x00")
        assert out.count(b"\x01") == 1
        assert out.count(b"\x02") == 1

    def test_two_bit_xor_kept(self):
        # 0x01 + 1 = 0x02 is a 2-bit XOR; this stage has no 2/1 walk.
        assert b"\x02" in _stream(b"\x01")


class TestWidePasses:
    def test_arith16_big_endian_carry(self):
        assert b"\x01\x00" in _stream(b"\x00\xff")  # BE 0x00FF + 1

    def test_arith32_carry_over_low_half(self):
        assert b"\x00\x00\x01\x00" in _stream(b"\xff\xff\x00\x00")  # LE 0xFFFF + 1

    def test_interest16_both_endians(self):
        out = set(_stream(b"\x41\x41"))
        assert b"\xe8\x03" in out and b"\x03\xe8" in out  # 1000

    def test_interest32_present(self):
        assert b"\xff\xff\xff\x7f" in set(_stream(b"AAAA"))  # INT32_MAX LE

    @pytest.mark.parametrize("data", SEEDS)
    def test_adversarial_no_duplicates_no_seed_copies(self, data):
        out = _stream(data)
        assert len(out) == len(set(out))
        assert data not in out

    def test_mutants_stay_inside_one_window(self):
        data = bytes(range(0, 256, 7))
        for m in _stream(data):
            diffs = [i for i, (a, b) in enumerate(zip(data, m, strict=True)) if a != b]
            assert diffs and diffs[-1] - diffs[0] < 4

    def test_interest8_drops_arith_repeats(self):
        # 0x03 - 3 = 0x00 is arith8's; interest8 must not resend it.
        assert _stream(b"\x03").count(b"\x00") == 1
        assert set(INTERESTING_UNSIGNED_8) <= {m[0] for m in _stream(b"\x03")} | {3}


class TestEffectorGating:
    def _drive(self, data: bytes, live: set[int]) -> list[bytes]:
        effector = DeterministicEffectorMap(len(data))
        out = []
        for m in _deterministic_mutation_stream(data, 10**9, effector=effector):
            if effector.pending >= 0:
                p = effector.pending
                effector.eff[p] = _DET_EFF_LIVE if p in live else _DET_EFF_INERT
                effector.pending = -1
            out.append(m)
        return out

    def test_wide_sites_gated_by_window(self):
        data = bytes(range(0x41, 0x41 + 24))
        live = {5, 17}
        assert self._drive(data, live) == list(reference_stream(data, live=live))

    def test_window_with_one_live_byte_runs(self):
        # Site 2's 32-bit window covers live byte 5: a pure-inert window would not.
        data = b"\xff" * 12
        out = self._drive(data, {5})
        touched = {
            i
            for m in out[9 * len(data) :]
            for i, (a, b) in enumerate(zip(data, m, strict=True))
            if a != b
        }
        assert touched and all(2 <= i <= 8 for i in touched)
