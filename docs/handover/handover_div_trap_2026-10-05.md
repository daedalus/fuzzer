# Handover: `div_trap` and a RandPool fix in the Hacker's Delight operators

Date: 2026-10-05. Source: Hacker's Delight (Warren, 2002) ch. 9-10, read against
`fuzzer-tool` at `7dd7860e`.

## Gap audit (book vs repo)

| Item | Verdict |
|---|---|
| `(INT_MIN, -1)` in adjacent fields | Real gap -- **shipped** as `div_trap` |
| True bit-matrix transpose (HD 7-3) | Real gap, unbuilt: `bit_transpose_8..64` swaps random bit pairs, it is not a transpose |
| Sub-byte sign extension, masked-merge crossover | Real gaps, unbuilt (see the bithacks handover) |
| Negabinary (ch. 12), Hilbert/Morton (ch. 14), pext/pdep (7-5) | Low value, skipped |
| popcount, clz/ctz, log2, SWAR haszero, Gray, shim slot math | Already covered |

## `div_trap`

`MIN / -1` and `MIN % -1` overflow the signed range: x86 `idiv` raises SIGFPE,
Rust and Swift panic. Both values are already in the interesting-value tables,
but one field at a time; the trap needs a dividend and a divisor in
neighbouring fields.

Draw order: `choice(widths that fit twice)`, `randint(offset)`,
`choice(endian)`, `choice(order)`. Overwrites exactly `2 * width` bytes,
length-preserving; inputs shorter than 2 bytes return untouched with no draws.
Widths 1/2 are included for languages that do not promote small ints.

Falsifiable invariant (`test_pair_decodes_to_the_trap`): over 40 fixed seeds the
output contains `(MIN, -1)` or `(-1, MIN)` at some width/endianness/offset with
everything outside the window unchanged. Sensitivity control: `(MIN + 1, -1)`
fits the signed range, so the check separates the trap from an ordinary divide.

Wiring (Hard Rule 12): operator in `core/mutations/structured.py`,
`_CATEGORIES["regularity"]`, `_op_div_trap` via `_regularity`,
`REGULARITY_OPS` in the registry test, DEEP_DIVE bullet, CHANGELOG.

Not done: a gap between the two fields (a parser may read them several bytes
apart), and no run against a real target.

## Bug found in `7dd7860e`: the six HD operators crash on `RandPool`

`hackers_delight._pick_window` drew `rng.randrange(0, n)`. `RandPool.randrange`
takes one argument, so every `rightmost_*` and `same_popcount_*` handler raised
`TypeError` in a real run. The module's own tests used `random.Random`, so they
stayed green; it surfaced only when the registry test's operator list was
corrected (it lacked `same_popcount_next/prev`). Fix: `rng.randint(0, n - w)`.
Regression: `test_operators_run_on_randpool` (6 operators x 8 seeds).

Two ruff findings in `hackers_delight.py` and its test (UP035, I001) were
already there and are left alone.
