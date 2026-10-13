# Handover — pattern-table gaps (follow-up to `handover_mutator_gaps_2026-10-11.md`)

**Date:** 2026-10-12
**Base:** `6c86d7c5` (`Add --affine-mod: solve (a*x+b) % m == t compares`)
**Status:** done and tested; coverage effect on a real target **not measured**.

## What happened to O1-O7

| ID | Outcome |
|---|---|
| O1 `leb128_overlong` | Shipped upstream as `varint_reencode` (`0ce384af`: LEB128, VLQ, QUIC, EBML). Its EBML mask bug was fixed there too. |
| O2 `float_special` | **Mostly a false gap.** `float_squeeze` already carried sNaN, -0.0, denormals, max finite (`structured._FLOAT_PATTERNS`). Residue added to that table. |
| O3 `case_map_grow` | Real. Added as 11 `_FUNNY_UNICODE` entries, not an operator. |
| O4 `unicode_ctrl` | **Mostly a false gap.** `_FUNNY_UNICODE` already had RLO/LRO, zero-width, BOMs. Residue (NEL, U+2028/2029, LRM/RLM, isolates, ALM) added. |
| O5 `int_str_boundary` | Real: `_random_num` drew only random 30-bit values and squares. Fifth strategy `_DECIMAL_EDGES`. |
| O6 `hash_len_ext` | Not started. Needs a digest-field oracle. |
| O7 sequence-level mutation | Not started. Needs a message-boundary model. |

The 2026-10-11 chat summary was wrong about O2 (`bitcast_float` does emit NaN/inf) and
overstated O4. Tables over operators: every new operator is a scheduler arm that competes
for slots (see the `arith_mid` TODO), and the existing dispatch already draws from these
tables.

## Changes

- `core/mutations/structured.py`: `_FLOAT_PATTERNS` +7 (f64/f32 pairs): smallest normal,
  smallest negative denormal, most negative finite, NaN with all payload bits set,
  `2^53-1`/`2^53`/`2^53+2` (f32: `2^24` equivalents).
- `core/mutations/generic.py`: `_FUNNY_UNICODE` +21 (11 length-changing case mappings,
  10 line/bidi controls); `_DECIMAL_EDGES` (`2^(w-1)-1`, `2^(w-1)`, `2^w-1`, `2^w` for
  w in 8/16/32/64, plus `2^53`); `_random_num` draws it as strategy 4 of 0-4.
- No registry change: no operator was added or renamed.

## Separate fix: six tests were red at `6c86d7c5`

`tests/test_exhaustive_pool.py` failed 6/109 before any of this work.

1. `ExhaustivePool.sample` rejected `range`/`bytearray` (`RandPool.sample` accepts both);
   `arith_mid` calls `rng.sample(range(n), k)`.
2. `MinimalFuzzer` (`tests/support/operator_env.py`) lacked `poly1305_key`,
   `poly1305_tag_at`, `rs_nsym`, `rs_block`, `rs_fcr`, `rs_poly`.
3. `_op_poly1305_fix` / `_op_rs_fix` raised on an unset option (`rs_fix` with
   `nsym=None`: `TypeError` in `block_spans`). They now `_op_declined`.

Regression tests: `tests/test_regression_exhaustive_pool_gated_ops.py` (5 fail without the
fixes, pass with them).

## Verification

- `tests/test_pattern_gaps.py`: 95 tests, oracles are Python's own float, `unicodedata`
  and `str.upper()/lower()` semantics, not the tables. 70 failed before the change.
- 52 test files touching the changed tables, handlers, `operator_env` or the exhaustive
  pool: 2052 passed, 4 skipped.
- `ruff format` clean. `ruff check src tests`: 59 errors at `6c86d7c5` and after
  (none added); one is an import-order warning in `services/operators.py` from `0ce384af`,
  left alone.
- Not run: full suite (Hard Rule 50), fuzzgoat calibration (Hard Rule 52, no clang here).
- Speed: the only hot-path change is one extra branch in `_random_num`; not timed.

## Open

See the TODO entry "Pattern-table gaps unmeasured": paired fuzzgoat A/B; whether the
boundary strategy (1/5 of `ascii_num_replace` draws) starves the random-value ones;
`bitcast_float` still writes only 0, +-inf, one NaN, x*1e3; O6 and O7.
