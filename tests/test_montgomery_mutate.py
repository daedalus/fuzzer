"""Falsification + adversarial tests for montgomery_mutate (math-port P1)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations.montgomery import (
    SECP256K1_FIELD_P,
    SECP256K1_ORDER_N,
    _barrett_mu,
    _montgomery_r,
    _montgomery_r2,
    _redc_n0_prime,
    montgomery_mutate,
    sniff_secp_modulus,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool


class TestSniffer:
    def test_sniffs_field_prime(self):
        data = b"\x00\x01" + SECP256K1_FIELD_P + b"\x02"
        assert sniff_secp_modulus(data) is True

    def test_sniffs_curve_order(self):
        data = b"\x00\x01" + SECP256K1_ORDER_N + b"\x02"
        assert sniff_secp_modulus(data) is True

    def test_falsification_no_modulus(self):
        """Falsification: seed without the field prime / order → refuse."""
        data = b"\x00" * 64
        assert sniff_secp_modulus(data) is False


class TestDerivedConstants:
    def test_n0_prime_is_limb_width(self):
        n = int.from_bytes(SECP256K1_FIELD_P, "big")
        n0 = _redc_n0_prime(n)
        assert 0 <= n0 < (1 << 64)
        # n * n0' ≡ -1 (mod 2^64)
        assert ((n * n0) + 1) & ((1 << 64) - 1) == 0

    def test_r_and_r2_in_range(self):
        n = int.from_bytes(SECP256K1_FIELD_P, "big")
        r = _montgomery_r(n)
        r2 = _montgomery_r2(n)
        assert 0 < r < n
        assert 0 < r2 < n
        assert (r * r) % n == r2

    def test_barrett_mu_positive(self):
        n = int.from_bytes(SECP256K1_FIELD_P, "big")
        mu = _barrett_mu(n)
        assert mu > 0


class TestMutate:
    def test_injects_prime_when_absent(self):
        rng = RandPool(seed=1)
        data = b"\x00" * 80
        out = montgomery_mutate(data, rng)
        assert SECP256K1_FIELD_P in out or len(out) > len(data)

    def test_preserves_length_when_modulus_present(self):
        rng = RandPool(seed=2)
        data = b"\x00\x01" + SECP256K1_FIELD_P + b"\x00" * 32
        out = montgomery_mutate(data, rng)
        assert len(out) == len(data)

    def test_adversarial_near_prime(self):
        """Adversarial: prime mutated to a near-prime → no crash, graceful."""
        rng = RandPool(seed=3)
        # Flip one bit of the field prime so the literal no longer matches,
        # then still call the mutator (should inject or no-op safely).
        near = bytearray(SECP256K1_FIELD_P)
        near[-1] ^= 0x01
        data = b"\x00\x01" + bytes(near) + b"\x00" * 32
        out = montgomery_mutate(data, rng)
        assert isinstance(out, (bytes, bytearray))
        assert len(out) >= 2


class TestRegistryWiring:
    def test_registered_in_format_band(self):
        assert "montgomery_mutate" in REGISTRY.names()
        assert REGISTRY.category_of("montgomery_mutate") == "format"
