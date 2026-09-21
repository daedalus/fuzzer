# Handover: port candidates from `../math/` catalog (2026-09-21)

## Ask

Survey `/home/dclavijo/my_code/math/` (55 algorithm files, mostly stdlib or
gmpy2; all GPLv3) and plan which algorithms slot into the fuzzer. Prior art:
`docs/handover/handover_algorithm_catalogue_survey_2026-09-06.md`,
`handover_oeis_port_candidates_2026-09-12.md`,
`handover_ffmpeg_port_candidates_2026-09-16.md` (same catalogue-review pattern).

## Filter applied

The fuzzer already reimplements every general-purpose statistical primitive
scipy-free (Hard Rule 51): `core/chi_squared.py` (Lentz CF + Lanczos),
`core/rand_pool.py` (PCG64), `core/running_stats.py`, `core/gaussian.py`,
Tim Ratliff-style beta via numpy in the schedulers, numpy FFT in
`core/periodicity.py`. Nothing that duplicates those ships. The catalogue's
pure-number-theory items (Legendre, Wilson's theorem, Pisano, sieves) have no
consumer in-tree — skipped unless a listed integration provides one.

## Ranked integrations

### P1. Montgomery REDC / Barrett literal injection → secp256k1 target

`../math/redc.py` (Montgomery REDC class), `../math/barret.py`. The fuzzer
already vendors secp256k1 (`tools/vendor_secp256k1.sh`, `targets/secp256k1_*`).
secp256k1 uses its own libsecp Montgomery/Barrett-style field arithmetic
(`secp256k1_fe_mul_inner` is a fixed-window Montgomery mul). Today the target
gets flat-byte mutations; coverage of the gmpn/field hot loops is shallow.

Plan:
- New structure-aware mutator `core/mutations/montgomery.py` registered in
  `core/operator_registry.py` (category band + `_AVAILABLE` predicate gated on
  seed being secp256k1-classified), deriving `N`, `n0'` and Barrett `µ/2^k`
  tables from the 32-byte modulus literal found in the seed and splicing them
  into field-width-aligned spots.
- Sniffer predicate keyed on the secp256k1 curve order / field prime appearing
  as a big-endian constant — reuse the pattern from `lz4`/`asf` mutators.
- Value: forces the target into REDC encode/decode branches it never reaches
  under random bytes.

Wiring: register in `_CATEGORIES` + `_AVAILABLE` only; dispatch, ops list, arms
derive from `REGISTRY` (Hard Rule 12). Tests per Hard Rule 23: one falsification
(seed without the field prime → mutator refuses, falls back) + one adversarial
(prime mutated to a near-prime → no crash, graceful).

### P2. Faster inverse-incomplete-beta → `op_bayes_ucb.beta_quantile`

`core/schedulers/op_bayes_ucb.py:239` `beta_quantile` − docstring admits
~27 µs/quantile of CF + bisection, and `select_op` bisects only the surviving
shortlist because of it (lines 118–129). The catalogue has accrual beta-quantile
ideas; implement ourselves, scipy-free (Hard Rule 51).

Plan:
- Replace the outer bisection seeding with the existing Cornish-Fisher + `norm_ppf`
  approximation (already in the module, line ~144/284) then Newton-polish against
  the continued-fraction `beta_cdf` instead of grid bisection. Target ≤2 iterations.
- Keep `BISECT_TOL` fallback path; add a fast-path flag so an A/B in
  `tools/bench_sweep.sh` can measure it.

Wiring: none outside the module — `beta_quantile` is the single seam. Verify no
speed regression (Hard Rule 41) and determinism via scripted RNG (Hard Rule 39).
Tests: quantile equality against the bisection path on a grid of (a,b,p); control
asserts the reference path matches *itself* (Hard Rule 45).

### P3. GF(2) polynomial mul/gcd speedup → `core/gf2_common.py`

`poly_mul`/`poly_gcd`/`poly_powmod` (`gf2_common.py:57/87/94`) are pure-Python
shift loops. Consumers: `berlekamp_massey.py` CRC-poly recovery, Rabin
irreducibility, `prng_state_recovery.py`. Catalogue's `binary_polinomial_factoring.sage`
and `GaussGF2.py` are reference materials only — GPLv3 + Sage, not vendorable.

Plan:
- Packed-int carryless multiply: `poly_mul` via nibble lookup (4-bit × limb
  tables) matching existing `(int, int) -> int` signatures, no API change.
- Only if measured hot (profile first — Hard Rule 41): table-assisted `poly_gcd`
  mirroring `gf2_common`'s structure.

Wiring: none — same signatures. Regression tests: `poly_mul(a,b) == poly_mul_ref`
over random-integer poly pairs plus the existing `_prime_factors`/Rabin callers.

### P4. KS p-value unification (refactor)

`edge_tracker.py:317` `_kolmogorov_pvalue` (20-term asymptotic) vs
`randomness.py:732` `_ks_exact_cdf` (Marsaglia exact) — documented duplication.
Unify on the exact routine; split a shared `core/ks_pvalue.py`. Pure refactor,
zero behaviour change; the existing two test files guard both callers.

### P5. χ² p-value acceleration (conditional)

`chi_squared_pvalue` (Lentz CF, 30–100 iters) is per-analyzer-per-tick under
`multiple_testing.py`/FDR. Only if a profile shows it hot skimming.

### P6. Nearest-coprime closed form → `exhaustive_pool._coprime_stride`

Golden-ratio coprime stride walks `math.gcd` outward line scan
(`exhaustive_pool.py:145`). Closed-form nearest-coprime to ⌊space/φ⌋ would cut
O(space) worst case. Only matters for prime-space pools; marginal — park unless
a benchmark target shows it.

## Rejected

- `dft.py`/`fft.py` — nothing beats numpy `rfft` already in `periodicity.py`.
- Legendre symbol, Wilson, Pisano, sieves, Euclidean varieties, `gmpy2`-bound
  factoring (`fermat_factor`, `euler_integer_factorization`) — no consumer.
- `anomaly_detection.py`, `mean.py`, `birthdayparadox.py` — `running_stats`/
  `chi_squared`/`rand_pool` already cover these better.

## License note

`../math/` is GPLv3. A port is a reimplementation of an *idea*, never a verbatim
copy of the file, and must be scipy-free + stdlib/numpy only. Everything above
is written from scratch against the existing module conventions, not vendored.

## Sequencing

1. P2 (self-contained, measurable, single seam) — do first, cheapest win.
2. P3 (contained, pure speedup, no wiring).
3. P1 (new mutator + registry wiring + secp256k1 build).
4. P4 (refactor), P5/P6 only on profile evidence.

Each step: TODO first (Hard Rule 10), regression tests (Hard Rule 23), suite for
affected files, `ruff format/check`, docs per convention, commit+push.
