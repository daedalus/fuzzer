# Covering-array PNG IHDR field generator

2026-09-21, base HEAD f1327b6f.

## What this is

The P3 "covering-array header-field generator" candidate from
`docs/handover/handover_generators_2026-09-20.md`: the one genuinely
novel gap identified there, not a promotion of an existing algorithm.

`core/covering_array.py` -- generic, stdlib-only, pure-Python AETG-style
greedy t-way covering array construction. `core/mutations/covering_array_mutate.py`
-- `covering_array_ihdr` (category `format`), a self-registering
`MutatorBase` that sweeps a pairwise (t=2) covering array over PNG
IHDR's 7 fields (width, height, bit_depth, color_type, compression,
filter, interlace).

## Why pairwise, and why IHDR

`png.py`'s existing `_mutate_ihdr` draws one field per call, independently.
Across many calls every individual boundary value gets tried, but two
fields are never deliberately set together -- a decoder bug gated on a
specific pair (e.g. `color_type=3` indexed, which per the PNG spec
restricts `bit_depth` to 1/2/4/8, paired with `bit_depth=16`, valid alone)
is reached only by two independent draws landing right in the same call.
A pairwise covering array guarantees every field-pair/value-pair
combination appears in some row.

Field domains (`_FIELDS` in `covering_array_mutate.py`) reuse boundary
values `png.py`'s own mutator already draws from (`bit_depth`'s
`[0,1,2,4,8,16,255]`, `interlace`'s `[0,1,42,255]`) rather than inventing
an unrelated set, plus the PNG-spec-valid values for fields it doesn't
enumerate (`color_type`, `compression`, `filter`).

Measured on the domains actually shipped: 487 required pairwise tuples,
covered in 57-63 rows across a 200-seed sweep (vs. 44,100 for the
exhaustive cross product).

## Design choices worth recording

- **Round-robin, not random redraw.** The array is built once (lazily,
  on first `mutate()` call, using the fuzzer's own rng for
  reproducibility under `--seed`) and then applied row-by-row in order,
  wrapping. Redrawing a random row per call would still be a valid
  mutation but would silently drop the coverage guarantee this operator
  exists for -- verified by `TestRoundRobinCoverage` in
  `tests/test_covering_array_mutate.py`, which calls `mutate()` 700
  times and checks `covering_array.verify_coverage()` against the
  observed rows.
- **`core/covering_array.py` knows nothing about PNG.** It's parameters
  and value domains in, rows out. A second format (or a second PNG
  chunk, e.g. a future gAMA/pHYs covering array) reuses it directly --
  the only new code needed is a `_FIELDS`-shaped table and an
  operator body shaped like `PngCoveringArrayMutator`.
- **Gated purely by format sniff, no new CLI flag.** Matches
  `der_len_mutate`/`tree_generate`'s precedent: availability is a
  statement about the input, not a mode switch. The doc's own design
  principle (mutation beats generation early, generation catches up
  late, so every generator must be able to decay toward zero share)
  is handled by the existing Elo/bandit scheduler automatically, same
  as every other operator -- nothing bespoke needed here.

## A real bug, found by testing

The first version of the greedy loop broke out the first time a round
of `candidate_pool` (50) random candidate rows failed to improve
coverage at all. That happens by chance once `needed` is down to its
last few tuples -- e.g. one specific `(color_type=255, compression=255)`
pair among the 487, each independently-drawn row only a few percent
likely to hit it, and P(all 50 miss) is non-trivial. Caught by a
bounded-row-count property test (`test_png_ihdr_shaped_domains_fully_covered`,
seed 1234) failing intermittently; root-caused by direct
`missing_tuples()` inspection, fixed with a bounded retry loop (up to
200 rounds) instead of a single-round give-up. Verified clean across a
200-seed sweep, and a 25-seed slice of that is now a permanent
regression test (`test_full_coverage_across_many_seeds_png_ihdr_domains`).

## Verification

- 42 new tests (17 in `test_covering_array.py` for the generic
  algorithm, 25 in `test_covering_array_mutate.py` for the PNG
  operator), all passing.
- `ruff check` / `ruff format --check`: clean on both new modules.
- `mypy src/` (whole-package, not per-file -- per-file checking of
  brand-new modules against this codebase's existing untyped
  `MutatorBase.mutate(rng, **ctx)` base signature produces noise that
  doesn't reproduce in the full-package run): 673 errors / 102 files
  after this change, identical to the pre-existing baseline with these
  two files absent -- both new modules contribute zero errors. Not
  added to the mypy exemption list (`test_regression_mypy_ratchet.py`
  still passes).
- Directed sweep (386 tests: `test_covering_array{,_mutate}.py`,
  `test_operator_smoke.py`, `test_wfc_chunks.py`,
  `test_regression_mutator_interface.py`, `test_png_mutations_unit.py`,
  `test_regression_mypy_ratchet.py`, `test_regression_operator_registry.py`,
  `test_regression_scheduler_fallback_precedence.py`,
  `test_regression_scheduler_operator_reach.py`, `test_new_operators.py`,
  `test_regression_no_op_mutations.py`): 385 passed, 1 pre-existing
  failure (`rasc_chunk_mutate`/`tiff_chunk_mutate` unreachable by that
  test's sweep fixture) confirmed identical against a clean stash of
  the same HEAD -- unrelated to this change.
- `test_regression_mutator_interface.py`'s
  `TestGlobalRegistryUnaffected.test_builtin_registry_mutators_are_known`
  is an explicit allow-list of every registered `MutatorBase`; updated
  to include `covering_array_ihdr` as the sixth entry (was failing
  before the update, as expected -- the list is deliberately exhaustive).

## Not done / open

- PNG only. A second format (candidates: ISO-BMFF `ftyp`/`tkhd` fields,
  RIFF/WEBP `VP8X` flags) follows the same pattern but wasn't built --
  no format-specific field table exists yet for anything but PNG IHDR.
- No campaign-level measurement of whether this operator's arm actually
  earns selection share under the Elo/bandit scheduler (the G0 bench
  arms added yesterday don't cover it either -- it's a new arm, not one
  of the four G0 was built for). Per the doc's own literature caveat,
  this is expected to matter more at longer campaign horizons than at
  5 minutes; no campaign has been run to check.
