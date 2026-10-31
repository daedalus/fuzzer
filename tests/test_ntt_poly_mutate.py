"""Tests for the NTT polynomial mutator and its registry wiring."""

from __future__ import annotations

from fuzzer_tool.core.mutations.ntt_poly import (
    _MOD_BE,
    _MOD_LE,
    _ROOT_BE,
    ntt_poly_mutate,
    sniff_ntt_modulus,
)
from fuzzer_tool.core.ntt import DEFAULT_MOD
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng


class TestSniffer:
    def test_sniffs_be_modulus(self):
        data = b"\x00\x01" + _MOD_BE + b"\x02"
        assert sniff_ntt_modulus(data) is True

    def test_sniffs_le_modulus(self):
        data = b"\x00\x01" + _MOD_LE + b"\x02"
        assert sniff_ntt_modulus(data) is True

    def test_falsification_no_modulus(self):
        data = b"\x00" * 64
        assert sniff_ntt_modulus(data) is False


class TestMutate:
    def test_injects_modulus_when_absent(self):
        rng = RandPool(seed=1)
        data = b"\x00" * 32
        out = ntt_poly_mutate(data, rng)
        assert sniff_ntt_modulus(out) is True

    def test_empty_input_returns_modulus(self):
        rng = RandPool(seed=2)
        out = ntt_poly_mutate(b"", rng)
        assert sniff_ntt_modulus(out) is True
        assert DEFAULT_MOD.to_bytes(4, "big") in out or DEFAULT_MOD.to_bytes(4, "little") in out

    def test_transform_preserves_length_when_modulus_present(self):
        rng = RandPool(seed=3)
        # 64 bytes = 16 little-endian 32-bit coeffs (power-of-two length).
        data = _MOD_BE + b"\x00" * 60
        out = ntt_poly_mutate(data, rng)
        assert len(out) == len(data)
        assert sniff_ntt_modulus(out) is True

    def test_adversarial_short_buffer(self):
        rng = RandPool(seed=4)
        data = b"\x01\x02"
        out = ntt_poly_mutate(data, rng)
        assert isinstance(out, (bytes, bytearray))
        assert len(out) >= 2


class TestInjectPreservesLength:
    """Inject used `_region(max_len=8)` as a cap; it rounds to multiples of 8.

    A 4-7 byte draw rounded to 0, `_region` returned (0, 0), and the
    short-buffer fallback appended the modulus to a 512-byte buffer.
    """

    def test_regression_ntt_inject_grew_long_buffer(self):
        # Draws: slot length 5, offset 101 (aligned to 100), BE modulus.
        rng = ScriptedRng(randints=[5, 101, 0])
        out = ntt_poly_mutate(bytes(512), rng)

        assert len(out) == 512
        assert out[100:104] == _MOD_BE
        assert out[:100] + out[104:] == bytes(508)

    def test_falsification_wide_slot_still_carries_root(self):
        # Slot draw 200 caps at 8: modulus, root coin 0, BE root.
        rng = ScriptedRng(randints=[200, 64, 0, 0, 0])
        out = ntt_poly_mutate(bytes(512), rng)

        assert len(out) == 512
        assert out[64:72] == _MOD_BE + _ROOT_BE

    def test_adversarial_minimum_buffer(self):
        # 5 bytes, slot 5 at offset 0: modulus fills 4, no growth.
        rng = ScriptedRng(randints=[5, 0, 0])
        out = ntt_poly_mutate(bytes(5), rng)

        assert len(out) == 5
        assert out[:4] == _MOD_BE


class TestRegistryWiring:
    def test_registered_in_regularity_band(self):
        assert "ntt_poly_mutate" in REGISTRY.names()
        assert REGISTRY.category_of("ntt_poly_mutate") == "regularity"
