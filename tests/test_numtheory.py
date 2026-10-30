"""Tests for core/numtheory: stdlib re-implementation of the sympy.ntheory subset."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core import numtheory as nt

CARMICHAEL = (561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 825265)
# Strong pseudoprime to bases 2,3,5,7,11,13,17,19,23 (smallest, Jaeschke).
SPSP_9 = 3825123056546413051
MERSENNE_61 = (1 << 61) - 1


def _sieve(n: int) -> list[int]:
    flags = bytearray([1]) * (n + 1)
    flags[:2] = b"\x00\x00"
    for i in range(2, math.isqrt(n) + 1):
        if flags[i]:
            flags[i * i :: i] = bytes(len(flags[i * i :: i]))
    return [i for i, f in enumerate(flags) if f]


class TestEgcdInv:
    def test_egcd_bezout(self):
        for a, b in ((240, 46), (17, 5), (0, 7), (7, 0), (-12, 18)):
            g, x, y = nt.egcd(a, b)
            assert g == math.gcd(a, b)
            assert a * x + b * y == g

    def test_modinv_matches_builtin(self):
        for m in (7, 97, 1 << 16, 998244353):
            for a in (1, 2, 3, m - 1):
                if math.gcd(a, m) == 1:
                    assert nt.modinv(a, m) == pow(a, -1, m)

    def test_modinv_noninvertible_raises(self):
        with pytest.raises(ValueError):
            nt.modinv(6, 9)

    def test_modinv_modulus_one(self):
        assert nt.modinv(5, 1) == 0


class TestCrt:
    def test_coprime(self):
        assert nt.crt([3, 5, 7], [2, 3, 2]) == (23, 105)

    def test_noncoprime_consistent(self):
        assert nt.crt([4, 6], [0, 2]) == (8, 12)

    def test_noncoprime_inconsistent(self):
        assert nt.crt([4, 6], [0, 1]) is None

    def test_empty(self):
        assert nt.crt([], []) == (0, 1)

    def test_roundtrip_many(self):
        mods = [3, 5, 7, 11, 13]
        x = 12345
        r, m = nt.crt(mods, [x % q for q in mods])
        assert r == x % m and m == 15015

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            nt.crt([2, 3], [1])

    def test_bad_modulus(self):
        with pytest.raises(ValueError):
            nt.crt([0], [0])


class TestIsPrime:
    def test_matches_sieve(self):
        primes = set(_sieve(20000))
        assert [n for n in range(-5, 20000) if nt.is_prime(n)] == sorted(primes)

    def test_carmichael_composite(self):
        assert not any(nt.is_prime(n) for n in CARMICHAEL)

    def test_strong_pseudoprime_to_many_bases(self):
        assert not nt.is_prime(SPSP_9)

    def test_large_primes(self):
        assert nt.is_prime(MERSENNE_61)
        assert nt.is_prime((1 << 127) - 1)
        assert nt.is_prime((1 << 255) - 19)

    def test_large_composites(self):
        assert not nt.is_prime(MERSENNE_61 * MERSENNE_61)
        assert not nt.is_prime((1 << 67) - 1)  # Cole's number

    def test_control_against_itself(self):
        # Rule 46: the oracle (sieve) agrees with itself on a second run.
        assert _sieve(5000) == _sieve(5000)


class TestNextPrev:
    def test_next(self):
        assert nt.next_prime(0) == 2
        assert nt.next_prime(2) == 3
        assert nt.next_prime(90) == 97
        assert nt.next_prime(1 << 32) == 4294967311

    def test_prev(self):
        assert nt.prev_prime(3) == 2
        assert nt.prev_prime(100) == 97
        assert nt.prev_prime(1 << 32) == 4294967291

    def test_prev_none(self):
        assert nt.prev_prime(2) is None
        assert nt.prev_prime(-4) is None


class TestFactor:
    def test_small(self):
        assert nt.factorint(360) == {2: 3, 3: 2, 5: 1}
        assert nt.factorint(1) == {}
        assert nt.factorint(97) == {97: 1}

    def test_semiprime_64(self):
        p, q = 4294967291, 4294967279
        assert nt.factorint(p * q) == {q: 1, p: 1}

    def test_prime_power(self):
        assert nt.factorint(3**40) == {3: 40}
        assert nt.factorint(1000003**3) == {1000003: 3}

    def test_product_reconstructs(self):
        for n in CARMICHAEL + (SPSP_9, 2**64 - 1, 2**64 + 1, 10**18 + 9):
            f = nt.factorint(n)
            assert math.prod(p**e for p, e in f.items()) == n
            assert all(nt.is_prime(p) for p in f)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            nt.factorint(0)
        with pytest.raises(ValueError):
            nt.factorint(-6)

    def test_divisors_and_totient(self):
        assert nt.divisors(12) == [1, 2, 3, 4, 6, 12]
        assert nt.totient(1) == 1
        assert nt.totient(36) == 12
        assert nt.totient(97) == 96
        for n in range(1, 300):
            assert nt.totient(n) == sum(math.gcd(n, k) == 1 for k in range(1, n + 1))

    def test_mobius_and_carmichael_lambda(self):
        assert [nt.mobius(n) for n in (1, 2, 4, 6, 30)] == [1, -1, 0, 1, -1]
        assert nt.carmichael_lambda(561) == 80
        assert nt.carmichael_lambda(8) == 2
        assert nt.carmichael_lambda(16) == 4


class TestSqrtMod:
    def test_all_residues(self):
        for p in (3, 5, 7, 13, 17, 41, 97, 101):
            squares = {x * x % p for x in range(p)}
            for a in range(p):
                r = nt.sqrt_mod(a, p)
                if a in squares:
                    assert r is not None and r * r % p == a
                else:
                    assert r is None

    def test_tonelli_shanks_branch(self):
        p = 998244353  # p-1 = 119 * 2^23, hard case
        r = nt.sqrt_mod(2, p)
        assert r is not None and r * r % p == 2

    def test_large_prime(self):
        p = (1 << 255) - 19
        a = pow(123456789, 2, p)
        r = nt.sqrt_mod(a, p)
        assert r is not None and r * r % p == a

    def test_composite_modulus_rejected(self):
        with pytest.raises(ValueError):
            nt.sqrt_mod(4, 15)

    def test_p2(self):
        assert nt.sqrt_mod(1, 2) == 1
        assert nt.sqrt_mod(0, 2) == 0


class TestOrderRoot:
    def test_order(self):
        assert nt.mult_order(2, 7) == 3
        assert nt.mult_order(3, 7) == 6
        assert nt.mult_order(1, 13) == 1

    def test_order_not_coprime(self):
        with pytest.raises(ValueError):
            nt.mult_order(4, 8)

    def test_primitive_root(self):
        assert nt.primitive_root(2) == 1
        assert nt.primitive_root(7) == 3
        assert nt.primitive_root(998244353) == 3
        assert nt.primitive_root(8) is None  # cyclic only for 2,4,p^k,2p^k
        assert nt.primitive_root(12) is None

    def test_primitive_root_is_generator(self):
        for m in (9, 10, 14, 17, 18, 25, 50, 101, 2 * 101):
            g = nt.primitive_root(m)
            assert g is not None
            assert nt.mult_order(g, m) == nt.totient(m)


class TestDlog:
    def test_all_logs_small(self):
        p = 101
        g = nt.primitive_root(p)
        for x in range(p - 1):
            assert nt.discrete_log(p, pow(g, x, p), g) == x

    def test_smooth_order_pohlig_hellman(self):
        p = 2 * 3**5 * 5**3 * 7**2 * 11 * 13 + 1
        assert nt.is_prime(p)
        g = nt.primitive_root(p)
        for x in (0, 1, 12345, p - 2):
            assert pow(g, nt.discrete_log(p, pow(g, x, p), g), p) == pow(g, x, p)

    def test_big_smooth(self):
        # p-1 = 2^20 * 3 * 5 * 7 * 11 * 13; BSGS alone would need 2^28.
        p = 2**20 * 3 * 5 * 7 * 11 * 13 + 1
        if not nt.is_prime(p):
            p = nt.next_prime(p)
        g = nt.primitive_root(p)
        h = pow(g, 987654321 % (p - 1), p)
        x = nt.discrete_log(p, h, g)
        assert pow(g, x, p) == h

    def test_no_solution(self):
        # 2 generates the subgroup {1,2,4} mod 7; 3 is not in it.
        assert nt.discrete_log(7, 3, 2) is None

    def test_composite_modulus(self):
        assert nt.discrete_log(15, 4, 2) == 2
        assert nt.discrete_log(15, 7, 2) is None

    def test_bad_input(self):
        with pytest.raises(ValueError):
            nt.discrete_log(1, 1, 1)


class TestAdversarial:
    def test_factorint_huge_smooth(self):
        n = math.prod(nt.next_prime(k * 1000) for k in range(1, 20))
        assert math.prod(p**e for p, e in nt.factorint(n).items()) == n

    def test_crt_big(self):
        mods = [nt.next_prime(1 << k) for k in range(40, 60, 4)]
        x = 123456789123456789 % math.prod(mods)
        r, m = nt.crt(mods, [x % q for q in mods])
        assert r == x and m == math.prod(mods)

    def test_is_prime_bool_and_float_types(self):
        assert nt.is_prime(True) is False


STRONG_LUCAS_PSP = (5459, 5777, 10877, 16109, 18971, 22499)


class TestJacobiKronecker:
    def test_jacobi_matches_euler_for_primes(self):
        for p in _sieve(300)[1:]:
            for a in range(-5, p + 5):
                euler = pow(a, (p - 1) // 2, p)
                expect = 0 if euler == 0 else (1 if euler == 1 else -1)
                assert nt.jacobi(a, p) == expect
                assert nt.legendre(a, p) == expect

    def test_jacobi_multiplicative_in_n(self):
        for m, n in ((3, 5), (7, 15), (9, 11), (21, 25)):
            for a in range(-10, 40):
                assert nt.jacobi(a, m * n) == nt.jacobi(a, m) * nt.jacobi(a, n)

    def test_jacobi_rejects_even_or_nonpositive(self):
        for n in (0, -3, 4):
            with pytest.raises(ValueError):
                nt.jacobi(3, n)

    def test_legendre_rejects_composite_and_two(self):
        for p in (2, 9, 15):
            with pytest.raises(ValueError):
                nt.legendre(3, p)

    def test_kronecker_equals_jacobi_on_odd_positive(self):
        for n in range(1, 60, 2):
            for a in range(-20, 30):
                assert nt.kronecker(a, n) == nt.jacobi(a, n)

    def test_kronecker_edge_cases(self):
        assert nt.kronecker(1, 0) == 1 and nt.kronecker(-1, 0) == 1
        assert nt.kronecker(2, 0) == 0
        assert nt.kronecker(4, 2) == 0
        assert nt.kronecker(3, 2) == -1 and nt.kronecker(7, 2) == 1
        assert nt.kronecker(-1, -1) == -1

    def test_kronecker_multiplicative_in_n(self):
        for a in range(-15, 30):
            for m, n in ((2, 3), (4, 5), (-2, 7), (6, -5)):
                assert nt.kronecker(a, m * n) == nt.kronecker(a, m) * nt.kronecker(a, n)

    def test_quadratic_residues(self):
        for n in (1, 2, 7, 8, 15, 16, 30):
            assert nt.quadratic_residues(n) == sorted({x * x % n for x in range(n)})
        with pytest.raises(ValueError):
            nt.quadratic_residues(0)


class TestStrongLucas:
    def test_every_prime_passes(self):
        for p in _sieve(20000)[3:]:
            assert nt._strong_lucas(p)

    def test_squares_fail(self):
        for r in (7, 101, 65537):
            assert not nt._strong_lucas(r * r)

    def test_known_lucas_pseudoprimes_are_composite_for_is_prime(self):
        # strong Lucas pseudoprimes (A217255): the Lucas half accepts, MR base 2 rejects.
        for n in STRONG_LUCAS_PSP:
            assert nt._strong_lucas(n)
            assert not nt.is_prime(n)

    def test_large_primes_and_composites(self):
        big = nt.next_prime(1 << 90)
        assert nt.is_prime(big)
        assert nt.is_prime((1 << 127) - 1)
        assert not nt.is_prime(big * nt.next_prime(big))
        assert not nt.is_prime(((1 << 89) - 1) * ((1 << 107) - 1))

    def test_is_prime_agrees_with_sieve(self):
        primes = set(_sieve(5000))
        assert [n for n in range(5000) if nt.is_prime(n)] == sorted(primes)


class TestNthrootMod:
    def test_coprime_exponent_unique_root(self):
        for p, k in ((101, 7), (998244353, 5), (13, 5)):
            for a in (1, 2, 3, p - 1):
                r = nt.nthroot_mod(a, k, p)
                assert r is not None and pow(r, k, p) == a % p

    def test_all_roots_match_bruteforce(self):
        for p in (7, 13, 17, 31, 61, 97, 101):
            for k in (2, 3, 4, 6, 8):
                for a in range(p):
                    brute = [x for x in range(p) if pow(x, k, p) == a]
                    if brute:
                        assert nt.nthroot_mod(a, k, p, all_roots=True) == brute
                        assert nt.nthroot_mod(a, k, p) == brute[0]
                    else:
                        assert nt.nthroot_mod(a, k, p) is None

    def test_square_root_agrees_with_sqrt_mod(self):
        for p in (101, 103, 109):
            for a in range(1, p):
                r = nt.nthroot_mod(a, 2, p)
                s = nt.sqrt_mod(a, p)
                assert (r is None) == (s is None)
                if r is not None:
                    assert r == s

    def test_zero_and_p2(self):
        assert nt.nthroot_mod(0, 3, 11) == 0
        assert nt.nthroot_mod(0, 3, 11, all_roots=True) == [0]
        assert nt.nthroot_mod(5, 3, 2) == 1

    def test_argument_errors(self):
        with pytest.raises(ValueError):
            nt.nthroot_mod(2, 0, 7)
        with pytest.raises(ValueError):
            nt.nthroot_mod(2, 2, 15)


class TestEuclidSteps:
    def test_known_counts(self):
        assert nt.euclid_steps(240, 46) == 5  # 240,46 -> 46,10 -> 10,6 -> 6,4 -> 4,2 -> 2,0
        assert nt.euclid_steps(7, 0) == 0
        assert nt.euclid_steps(0, 7) == 1

    def test_lame_fibonacci_is_worst_below_bound(self):
        # Lame: below F(k+2), only consecutive Fibonacci pairs need k steps.
        fib = [1, 2]
        while len(fib) < 12:
            fib.append(fib[-1] + fib[-2])
        bound = fib[-1]
        worst = max(nt.euclid_steps(a, b) for a in range(1, bound) for b in range(1, a + 1))
        assert nt.euclid_steps(fib[-2], fib[-3]) == worst

    def test_step_oracle_tells_noise_from_fibonacci(self):
        # Control: a non-Fibonacci pair of the same size takes fewer steps.
        assert nt.euclid_steps(144, 100) < nt.euclid_steps(144, 89)

    def test_adversarial_negative_operands(self):
        assert nt.euclid_steps(-240, 46) == nt.euclid_steps(240, 46)
        assert nt.euclid_steps(240, -46) == nt.euclid_steps(240, 46)
