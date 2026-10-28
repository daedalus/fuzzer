# Bit provenance abstract domain: relevance to the fuzzer

**Date:** 2026-10-28
**Scope:** analysis of <https://pypy.org/posts/2026/10/toy-bit-provenance.html> (CF Bolz-Tereick, "The Shift of Theseus", 2026-10-27) against `core/gf2_common.py`, `core/rq_encodings.py`, `core/colorization.py`, `core/weizz_tags.py`, `core/live_bit_mask.py`, `core/cond_stmt.py`, `adapters/track_parser.py`, `core/mutations/int_encodings.py`.
**Status:** analysis only. **F2 and F3 below were wrong; see Correction.**

## Correction (2026-10-28, same day)

F2/F3 claimed the fuzzer cannot solve shifted, masked, rotated or byte-swapped compares. It can: `--derived-cmp` runs `core/derived_fit.fit_gf2` / `solve_gf2` (via `services/affine_mod.py::_solve_gf2`), which is the bit-Jacobian probe proposed here (unit-vector flips give the columns, 3 confirms, XOR elimination). The first analysis grepped `rq_encodings.py` and `core/` for encoder names and missed `derived_fit`.

Checked by running `fit_gf2`/`solve_gf2` on a 32-bit field:

| Operand model | fit | solve |
|---|---|---|
| `(x>>4)&0x3f`, `rotl(x,7)`, bswap, `~x&0xff`, `x^(x>>1)` | ok | exact |
| `(x&0xf)\|0x80` vs target 0 | ok | None (correct) |
| `x+0x0f0f0f0f`, `x*x` | None (fail closed) | n/a |

A from-scratch implementation (`core/bit_jacobian.py`, 35 tests) matched these results case for case and was discarded as a duplicate. Only extras: a care mask for partial compares and MOVE/MIX/DEAD labels, which do not justify a parallel module. No production code changed, nothing prototyped. The author's gist (code for the post) was not reachable from the analysis sandbox, so the post's code was read, not run.

## 1. What the post describes

A forward abstract domain over straight-line SSA traces. For each value it stores 64 entries, one per bit, each either:

- constant `0` or `1`, or
- `SourceBit(op, index, inverted)`: this bit equals (or is the negation of) bit `index` of an earlier value.

Transfer functions:

- Constant shifts only permute entries and fill with constants. A non-constant shift amount yields fresh bits (`from_source(result)`).
- `~` flips `inverted` and constant bits.
- `& | ^` work per bit: constant cases (`a&0`, `a|1`, `a^1`), same-source cases (`a&a=a`, `a^a=0`, `a&~a=0`, `a|~a=1`). Anything else becomes a fresh `SourceBit(result, i)`.

Optimizer uses of the result, in the order the post adds them:

1. Whole value is constant: fold.
2. All bits equal the unconstrained source: replace by the source.
3. Provenance already seen in the block (hash map from provenance to op): reuse that op.
4. Provenance equals a shift of some source (`recognize()`, validated through the same transfer functions): emit that single shift.

Reported results (PyPy JIT traces, integer ops only):

| Workload | Unoptimized | Provenance only | RPython JIT | JIT then provenance |
|---|---|---|---|---|
| Pydrofoil | 1,484,985 | 822,864 (44.59%) | 769,758 (48.16%) | 656,555 (55.79%) |
| PyPy | 1,442,782 | 1,374,803 (4.71%) | 1,089,196 (24.51%) | 1,089,196 (24.51%) |

The author's own conclusion: it does not beat the tuned optimizer, and adds nothing on the PyPy workload after the JIT passes. Its value is that one small domain finds rewrites (double byte swap, `reverse . (<<7) . reverse` = `>>7`, 63 ops down to 3) without a rule per combination. Related work cited: LLVM `BitTracker`/`HexagonBitSimplify`, `collectBitParts`, GCC `store-merging` byte-swap recognizer.

## 2. Findings against the fuzzer

### F1. The domain itself is already subsumed by `gf2_common`

`gf2_common.py` has `apply_bitmask_map`, `invert_bitmask_map`, `verified_apply_inverse`, `compose_bitmask_maps`, `compose_linear_runs`: a map is a list of per-output-bit masks over input bits (linear over GF(2)).

Bit provenance is the sub-case where each mask has one bit set, plus a constant and an inversion flag. The fuzzer's maps are strictly more expressive: XOR of several input bits stays exact, where the post's domain degrades to fresh bits. Recommendation: do not port the domain. If an affine constant term is ever needed for composed maps, that is an extension of the existing mask representation, not a new module. (Not checked: whether `compose_bitmask_maps` currently carries a constant term.)

### F2. Retracted

Redqueen encoders (`rq_encodings.py`) have no shift/mask/rotate encoder, but `--derived-cmp` covers these compares; see Correction.

### F3. Retracted

The black-box bit-Jacobian already exists as `derived_fit.fit_gf2`.

### F4. Cancelling operator stacks (low expected value)

Post examples (byte swap twice, reverse-shift-reverse) correspond to havoc stacks in which later operators undo earlier ones (`bit_matrix_transpose`, `bit_compress`, `sign_extend`, byte swaps in `core/mutations/int_encodings.py`). Composing a stack on a word and skipping stacks whose result equals the original or an already-tried value would save executions. Exact cancellation under random stacking is rare, so the expected saving is small and unmeasured. Not recommended before F2/F3.

### F5. Related idea in `LiveBitMaskEstimator`, speculative

`core/live_bit_mask.py` estimates the live-bit set by OR-accumulating diffs. "Bit i is the same as bit j" is the relation provenance tracks and the OR mask cannot express (a prior session found the OR mask saturating toward N, see the QEA Grover note). Whether an equality-class estimator would converge better is untested. Listed for completeness, no evidence either way.

## 3. Remaining items (none implemented)

| # | Item | Test |
|---|---|---|
| 1 | Measure how often `--derived-cmp` GF(2) fits fire on real targets (`_outcomes["gf2"]`), to judge the default-off setting | fuzzgoat, ffmpeg_read |
| 2 | Cancelling-stack filter (F4) | only if duplicate stack results exceed a few percent |
| 3 | Equality-class live-bit estimator (F5) | speculative, untested |

## 4. Caveats

- The post's performance numbers concern a compiler's integer optimizer; none of them transfer to fuzzing throughput or coverage.
- Nothing in this document was run against the fuzzer. The missing-encoder claim was wrong (see Correction); the other claims come from reading code at commit `c6e0cc7`.
- The post itself says provenance adds nothing after the JIT's integer passes on the PyPy workload, so the technique's marginal value is workload-dependent even in its home domain.
