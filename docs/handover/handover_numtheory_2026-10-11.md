# Handover: numtheory + prime_inject (2026-10-11)

Origin: Copilot job "explore what sympy has to offer ... implement it ourselves" stopped after reading (no code). Rule: Hard Rule 51.

## Done
- `core/numtheory.py`: egcd, modinv, crt, is_prime, next/prev_prime, factorint, divisors, totient, mobius, carmichael_lambda, sqrt_mod, mult_order, primitive_root, discrete_log. 45 tests.
- `prime_inject` operator (regularity), registered in `operator_registry.py`, `_op_prime_inject` in `operators.py`. 15 tests.

## Done (follow-up, same day)
- `jacobi`, `legendre`, `kronecker`, `quadratic_residues`, `nthroot_mod` (prime modulus, one root or all), strong Lucas folded into `is_prime` above 3.3e24 (BPSW). 18 new tests in `tests/test_numtheory.py`; differential-checked against sympy (not a test dependency).

## Not done
- No A/B of `prime_inject` on a real target (needs one with a primality or modular check).
- Not used by `ntt.py` (fixed modulus), `lcg_recovery.py` (lattice only), `fcsr_recovery.py` (own 2-D Gauss reduction), `int_checksum_solver.py`.
- sympy areas not ported: polynomials/Groebner, Diophantine, matrices over Z, simplify, solvers, continued fractions beyond existing, `nthroot_mod` for composite moduli, ECM/quadratic sieve, index calculus.
- `architecture.png` untouched (no new subsystem).

## Known issues
- `test_regression_scheduler_operator_reach.py::...[BayesUCBScheduler]` exceeds 15 s on clean master too.
