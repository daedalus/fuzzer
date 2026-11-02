"""montgomery_word_mutate: word-size (32/64-bit) REDC / Barrett constants."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core import numtheory as nt
from fuzzer_tool.core.mutations.montgomery import (
    WORD_MODULI,
    find_word_modulus,
    montgomery_word_mutate,
    sniff_word_modulus,
    word_constants,
)
from fuzzer_tool.core.operator_registry import _FORMAT_SNIFFERS, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool


@pytest.mark.parametrize(("n", "w"), WORD_MODULI)
def test_table_entries_are_odd_primes_that_fit(n, w):
    assert n % 2 == 1
    assert 0 < n < (1 << (8 * w))
    assert nt.is_prime(n)


def test_table_has_no_duplicate_literals():
    lits = {n.to_bytes(w, o) for n, w in WORD_MODULI for o in ("little", "big")}
    assert len(lits) == 2 * len(WORD_MODULI)


@pytest.mark.parametrize(("n", "w"), WORD_MODULI)
def test_constants_satisfy_their_definitions(n, w):
    bits = 8 * w
    big_r = 1 << bits
    r, r2, n_neg_inv, n_inv, barrett, nm1, np1, n2m1 = word_constants(n, bits)
    assert r == big_r % n
    assert r2 == (big_r * big_r) % n
    assert (n * n_neg_inv + 1) % big_r == 0  # n' = -n^-1 mod R
    assert (n * n_inv - 1) % big_r == 0  # n^-1 mod R
    assert barrett == (big_r * big_r // n) % big_r
    assert (nm1, np1, n2m1) == ((n - 1) % big_r, (n + 1) % big_r, (2 * n - 1) % big_r)


@pytest.mark.parametrize(("n", "w"), WORD_MODULI)
def test_n_prime_drives_a_real_redc(n, w):
    """n' and R^2 mod n must make REDC compute a*b*R^-1 mod n."""
    bits = 8 * w
    big_r = 1 << bits
    _, r2, n_prime, _, _, _, _, _ = word_constants(n, bits)
    rng = random.Random(n)

    def redc(t: int) -> int:
        m = ((t % big_r) * n_prime) % big_r
        u = (t + m * n) // big_r
        return u - n if u >= n else u

    for _ in range(50):
        a, b = rng.randrange(n), rng.randrange(n)
        am, bm = redc(a * r2), redc(b * r2)  # into Montgomery form
        assert redc(redc(am * bm)) == (a * b) % n  # product, then out of the form


class TestSniffer:
    @pytest.mark.parametrize("order", ["little", "big"])
    @pytest.mark.parametrize(("n", "w"), WORD_MODULI)
    def test_finds_each_literal(self, n, w, order):
        data = bytes(5) + n.to_bytes(w, order) + bytes(7)
        assert find_word_modulus(data) == (n, w, order, 5)
        assert sniff_word_modulus(data)

    def test_falsification_no_modulus(self):
        assert not sniff_word_modulus(bytes(64))
        assert not sniff_word_modulus(bytes(range(64)))

    def test_near_miss_one_bit_flipped(self):
        lit = bytearray((998244353).to_bytes(4, "little"))
        lit[0] ^= 1
        assert not sniff_word_modulus(bytes(8) + bytes(lit) + bytes(8))

    def test_registered_as_format_sniffer(self):
        assert "montgomery_word_mutate" in _FORMAT_SNIFFERS
        assert "montgomery_word_mutate" in REGISTRY.names()


class TestMutate:
    @pytest.mark.parametrize(("n", "w"), WORD_MODULI)
    def test_preserves_length_and_alignment(self, n, w):
        # Literal at offset 3: every written slot must be 3 mod w.
        data = bytes(3) + n.to_bytes(w, "little") + bytes(40)
        for seed in range(40):
            out = montgomery_word_mutate(data, RandPool(seed=seed))
            assert len(out) == len(data)
            changed = [i for i in range(len(data)) if out[i] != data[i]]
            if changed:
                start = 3 + ((min(changed) - 3) // w) * w  # slot grid anchored at the literal
                assert all(start <= i < start + w for i in changed)

    def test_writes_a_derived_constant_in_the_literals_byte_order(self):
        n = 998244353
        for order in ("little", "big"):
            data = bytes(4) + n.to_bytes(4, order) + bytes(24)
            wanted = {c.to_bytes(4, order) for c in word_constants(n, 32)}
            seen = set()
            for seed in range(200):
                out = montgomery_word_mutate(data, RandPool(seed=seed))
                for off in range(0, len(out) - 3, 4):
                    word = out[off : off + 4]
                    if word != data[off : off + 4]:
                        seen.add(word)
                        assert word in wanted
            assert len(seen) >= 4  # not stuck on one constant

    def test_no_modulus_is_a_no_op(self):
        data = bytes(range(64))
        assert montgomery_word_mutate(data, RandPool(seed=1)) == data

    def test_minimal_buffer_with_only_the_literal(self):
        lit = (2147483647).to_bytes(4, "big")
        out = montgomery_word_mutate(lit, RandPool(seed=3))
        assert len(out) == 4
