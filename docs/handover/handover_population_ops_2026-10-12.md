# Handover — population operators (line_blend, field_minmax, nm_reflect, opposition)

**Date:** 2026-10-12
**Base:** `f374ea84` (`Add handover: operator gaps vs AFL++/libFuzzer/honggfuzz/Radamsa (O1-O7)`)
**Status:** four operators implemented, tested, documented. No coverage A/B on a real
target. Everything marked **[verified]** was run or read in the tree at the base
commit; **[unverified]** is a hypothesis.

## What was added

Four structural-band operators in `core/mutations/generic.py`, next to the
extrapolation family (`arith_mid`, `diff_evo`, `field_median`). All use the same
field grid, none needs `--lineage`.

| Operator | Inputs | Per-field formula (1/2/4-byte, BE or LE) | Param (drawn) |
|---|---|---|---|
| `line_blend` | base `a`, one seed `b` | `a + ((b - a) * num >> 3)`, saturating | `num` in `_LINE_BLEND_NUMS` = {-4..-1, 1..7, 9..12} (t = num/8, so t in [-1/2, 3/2], never 0 or 1) |
| `field_minmax` | base `a`, one seed `b` | `min(a, b)` or `max(a, b)` | 0 = min, 1 = max |
| `nm_reflect` | base `a`, seeds `b`, `c` | `cen + alpha*(cen - a)`, `cen = (b+c)/2`; computed as `(2cen + (s*(2cen - 2a) >> 1)) >> 1`, saturating | `s` in `_NM_SCALES` = (1, 2, 4), alpha = s/2 |
| `opposition` | base `a`, seeds `b`, `c` | `lo + hi - a`, `lo`/`hi` = min/max of the three; never leaves `[lo, hi]` | none |

Why each one exists:

- `line_blend`: `arith_mid` only reaches k/4 for k = 1..3, i.e. inside the pair. t < 0 and t > 1 step *past*
  either parent, so a boundary beyond both seeds is reachable. It was named `blx_blend` in the chat
  analysis; renamed because it draws one t per call (extended line recombination, Muhlenbein), not an
  independent uniform sample per field (BLX-alpha). One t keeps the RNG scripted and exact in tests.
- `field_minmax`: pushes each field to one end of the range the two seeds span; the child mixes both parents.
- `nm_reflect`: multi-seed version of `lineage_step` that needs no parent chain. alpha = 1 is the
  parallelogram rule `b + c - a`. `a` is "the worst vertex" only by construction (the base is the one that
  moves); no fitness is ranked **[verified]** in the code.
- `opposition`: opposition-based learning with the range taken from three seeds, not the whole corpus.

Shared behaviour (all four):

- Length is `len(a)`; short donors are padded with `a`'s own tail (`_padded`).
- Declines (returns `a`) when inputs are empty or all agree, and a declined call draws **no** randomness
  (tested with a `NoDraw` rng that raises on any draw).
- Grid = `_field_grid(n, differs, rng, choices)`: draws shape, then the parameter (when `choices`), then the
  phase (width > 1 only). Order matters for scripted tests: `Scripted(shape, param, phase)`.
- Saturation is int64 + `np.clip`, same as `diff_evo`. `opposition` and `field_minmax` cannot leave the
  parents' range, so they do not clip.

## Wiring (every site touched)

| File | Change |
|---|---|
| `src/fuzzer_tool/core/mutations/generic.py` | `_LINE_BLEND_NUMS`, `_LINE_BLEND_SHIFT`, `line_blend`, `field_minmax`, `_NM_SCALES`, `nm_reflect`, `opposition`, inserted before `DICT_MUTATIONS`. Re-exported by `core/mutations/__init__.py` through `import *` (no `__all__`). |
| `src/fuzzer_tool/services/operators.py` | import the four; `OperatorEngine._pair_child(fn, buf, data)` (one `_distinct_donors(data, 1)` donor, declines when the child equals the base or the donor); `_op_line_blend`, `_op_field_minmax` via `_pair_child`; `_op_nm_reflect`, `_op_opposition` with `_distinct_donors(data, 2)`, declining when the child equals the base. |
| `src/fuzzer_tool/core/operator_registry.py` | four names added to the `structural` set. **No** `_AVAILABLE` entry (no lineage needed). |
| `tests/test_population_ops.py` | 124 tests (new file). |
| `tools/arith_mid_experiment.py` | arms `line_blend`, `field_minmax`; three-seed arms (`diff_evo`, `field_median`, `nm_reflect`, `opposition`) in `ARMS3`, corpus `[LOW, MID, HIGH]`. |
| `CHANGELOG.md`, `docs/DEEP_DIVE.md`, `docs/TODO.md` | entries. |

Donor choice differs from `arith_mid`/`geo_mid`/`power_mid`: those use `_recombine`
(`_unrelated_donor`, which filters ancestors and so needs `--lineage`). The new pair ops use
`_distinct_donors`, like `diff_evo`: ancestry is not filtered, because a population operator does not
need an unrelated donor. If an A/B shows parent/child pairs waste execs, switch `_pair_child` to
`_unrelated_donor`; that would also require a `_use_lineage` gate in `_AVAILABLE`.

## Pitfalls found while building (read before touching the grid)

1. **Phase decides whether a wide field is seen.** The grid starts at the first differing byte minus a
   drawn phase. `be(0)` vs `be(1000)` differ only in bytes 2-3; at phase 0 a 4-byte field starting at byte 2
   does not fit (`_arith_count` returns 0), so the grid silently falls back to bytes. Tests that mean a
   4-byte integer must pass `phase=2`. Two of my first tests were wrong for exactly this reason.
2. **`field_minmax` over ordered parents returns a parent only on an aligned grid.** With `a < b` in every
   integer, min is `a` and max is `b` when the grid is aligned (phase 3 on `be(1, 2)` vs `be(9, 7)`).
   A misaligned or byte grid covers only part of the span and mixes bytes of both parents, which is a real
   child. So the engine does not always decline there; `test_engine_field_minmax_never_returns_a_parent`
   checks the invariant that matters (no returned child equals a parent).
3. **Engine tests need crossing fields** for `field_minmax` (`a=be(1,70)`, `b=be(9,7)`), otherwise the
   aligned draws are declined and few children appear.
4. **Floor shifts.** `line_blend`/`nm_reflect` use `>>` on signed int64, so negative steps floor
   (gap -3, t = 1/8 gives -1, not 0). Tested explicitly; a truncating division would leave small
   negative steps stuck.

## Verification (at the base commit plus this patch)

- `tests/test_population_ops.py`: 124 passed.
- `tests/test_regression_no_op_mutations.py` (the sweep over every registered operator),
  `test_extrapolation_ops.py`, `test_arith_mid.py`, `test_geo_mid.py`, `test_power_mid.py`: 262 passed.
- Registry-aware tests (`test_regression_registry_plan`, `test_regression_scheduler_operator_reach`,
  `test_ltl_wiring`, `test_regression_parity_ops`): 159 passed.
- `ruff format` clean on the changed files. `ruff check src` reports 15 errors; the same 15 exist at the base
  commit (newer ruff than the repo was written for), none in the new code. `lizard`: CCN 4/5/7/7 for the
  four operators, 7 for `_pair_child`.
- Cost per call (numpy 2.4.4, one core): `field_minmax` 7-11 us, `opposition` 12-23 us, `line_blend` 14-23 us,
  `nm_reflect` 19-31 us, for 16 B / 256 B / 4 KiB inputs. `arith_mid` for reference: 10-18 us.

## Synthetic gates (`targets/arith_gates.c`), 10000 execs, 20 seeds

| Arm | G3 (BE16 window) | G1 / G2 (32-bit windows) |
|---|---|---|
| `arith_mid`, `geo_mid`, `power_mid`, `line_blend` | 20/20 | 1/0 (arith_mid), 0/0 (others) |
| `bit_cross` | 13/20 | 1/0 |
| `diff_evo` (three-seed corpus) | 13/20 | 0/0 |
| `nm_reflect` (three-seed corpus) | 12/20 | 0/0 |
| `byte_rand` | 6/20 | 0/1 |
| `field_minmax`, `opposition`, `field_median`, `splice`, `crossover`, `identity` | 0/20 | 0/0 |

Reading it honestly:

- **No evidence the new operators help.** `line_blend` ties `arith_mid` on the only gate the pair ops can
  solve. Nothing separates them on the 32-bit windows.
- `field_minmax`, `opposition` and `field_median` pick values already present in a parent, so a window gate
  cannot reward them. That is expected, not a defect; they target per-field *combinations*, which this
  target does not test.
- Three-seed arms start from `[LOW, MID, HIGH]`; they are not comparable one-to-one with the two-seed
  arms.
- **The baseline in `docs/TODO.md` (`power_mid` entry: `arith_mid` 12/20, G1/G2 gates 10/9) does not
  reproduce.** The original, unmodified harness at this base gives `arith_mid` 0/20 with G1 1/20, G2 0/20,
  G3 20/20. The target has not changed since the arith_mid commit and `RandPool` has not changed since
  2026-10-08. I did not find the cause (candidates **[unverified]**: a different execs/seed invocation, a
  different oracle build). Recheck before using any synthetic number to rank operators.

## Not done

- No fuzzgoat (clang) A/B; the only real-target evidence is none (Hard Rule 52 wants one).
- `position_marginal` (UMDA per aligned offset) from the chat analysis was **not** built: it needs a
  corpus-wide per-offset histogram, which is a larger change than a three-seed operator.
- `varint_reencode`, `sign_extend`, `bcd_pack`, `line_ending_swap`, `timestamp_specials`, `nul_inject`
  from the chat analysis are still open. `O1`-`O7` in `handover_mutator_gaps_2026-10-11.md` overlap with
  the first of these (overlong LEB128 is O1); do those first.
- No gate that rewards a value *past* both seeds, which is what `line_blend` t > 1 and `nm_reflect`
  alpha = 2 are for. Writing one in `targets/arith_gates.c` is the cheapest way to test the claim.

## Next steps, in order

1. Add the "past both seeds" gate and a per-field-combination gate to `targets/arith_gates.c`; rerun
   `PYTHONPATH=src python tools/arith_mid_experiment.py --execs 10000 --seeds 20`.
2. Resolve the baseline discrepancy above.
3. Paired fuzzgoat A/B with and without the four operators.
4. Drop whichever of `field_minmax`, `opposition`, `power_mid`, `geo_mid` shows no yield; each arm costs a
   scheduler slot.

## Running it

```
pip install pytest pytest-timeout numpy ppmd z3-solver ruff lizard --break-system-packages
PYTHONPATH=src python -m pytest tests/test_population_ops.py tests/test_regression_no_op_mutations.py -q
PYTHONPATH=src python tools/arith_mid_experiment.py --execs 10000 --seeds 20   # needs clang
```

`git commit` in a fresh sandbox needs `git config user.name` / `user.email`.
