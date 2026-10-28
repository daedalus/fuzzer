# Bit provenance abstract domain: relevance to the fuzzer

**Date:** 2026-10-28
**Scope:** analysis of <https://pypy.org/posts/2026/10/toy-bit-provenance.html> (CF Bolz-Tereick, "The Shift of Theseus", 2026-10-27) against `core/gf2_common.py`, `core/rq_encodings.py`, `core/colorization.py`, `core/weizz_tags.py`, `core/live_bit_mask.py`, `core/cond_stmt.py`, `adapters/track_parser.py`, `core/mutations/int_encodings.py`.
**Status:** analysis only. No production code changed, nothing prototyped. The author's gist (code for the post) was not reachable from the analysis sandbox, so the post's code was read, not run.

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

### F2. Gap: no shift/mask/rotate/byte-field encoder in the Redqueen-style I2S set

`core/rq_encodings.py` has encoders for plain/zext/sext, ASCII, C-string, hex, base64, UTF-16, case, LEB128, CRC (fixed and model), FNV-1a, byte hashes, bijective mixers, line codes and XOR-key. A grep over `core/` and `services/` for `jacobian`, `bitfield`, `rotl`, `bswap`, `masked_cmp` found nothing in the encoder path (`hashing.py:_rotl` is an internal hash helper). So a comparison like `((x >> k) & m) == c`, a rotated operand, or a packed bit-field is only reachable by luck of the operand bytes appearing verbatim in the input.

Provenance suggests the general form of the fix: when the operand is a signed bit-permutation (plus constants) of an input window, the exact preimage of the compared bits is determined bit by bit, and masked-off bits are free. This replaces a family of per-pattern encoders by one mechanism.

Limit: this needs the operand-producing chain or an empirical equivalent (F3). cmplog records operand values, not the instruction chain.

### F3. Black-box bit-level provenance is feasible without new instrumentation

Probe: for an input field of n bits, flip each bit, run, read the cmplog operand for the site, and record which operand bits changed. This is a bit-level Jacobian over GF(2). A column with exactly one set entry is a pure move (provenance of the post's kind); a column with several is a linear mix (still usable via the `gf2_common` machinery).

This is the bit-granular analogue of what `colorization.py` and `weizz_tags.py` (differential flips, per-byte dependency bitvectors) already do per byte.

Cost: about one execution per field bit (a few hundred for a 4-8 byte field), amortized per site and cached per seed lineage.
Validity: holds only if the site is linear or a move in the probed neighborhood. Needs a verification execution on the predicted preimage (same discipline as `verified_apply_inverse`).

### F4. Cancelling operator stacks (low expected value)

Post examples (byte swap twice, reverse-shift-reverse) correspond to havoc stacks in which later operators undo earlier ones (`bit_matrix_transpose`, `bit_compress`, `sign_extend`, byte swaps in `core/mutations/int_encodings.py`). Composing a stack on a word and skipping stacks whose result equals the original or an already-tried value would save executions. Exact cancellation under random stacking is rare, so the expected saving is small and unmeasured. Not recommended before F2/F3.

### F5. Related idea in `LiveBitMaskEstimator`, speculative

`core/live_bit_mask.py` estimates the live-bit set by OR-accumulating diffs. "Bit i is the same as bit j" is the relation provenance tracks and the OR mask cannot express (a prior session found the OR mask saturating toward N, see the QEA Grover note). Whether an equality-class estimator would converge better is untested. Listed for completeness, no evidence either way.

## 3. Proposed next steps (none implemented)

| # | Item | Where | Test |
|---|---|---|---|
| 1 | Bit-Jacobian probe per cmplog site, classify columns as move / mix / dead | new module under `core/`, reusing `gf2_common` and `weizz_tags` flip machinery | synthetic target with `((x >> k) & m) == c`, rotated and byte-swapped variants |
| 2 | Encoder that builds I2S replacements from a verified move/mix map | `core/rq_encodings.py` (new `Encoder` subclass, gated flag) | paired A/B vs current cmplog on the synthetic target, then fuzzgoat |
| 3 | Measure how often real cmplog operands are pure moves vs mixed | tool script over recorded cmplog on fuzzgoat, ffmpeg_read | decides whether item 2 is worth keeping |
| 4 | Cancelling-stack filter | `services/operators.py` | only if a measurement shows duplicate stack results above a few percent |

Suggested order: 3 first (cheapest, decides the rest), then 1, 2.

## 4. Caveats

- The post's performance numbers concern a compiler's integer optimizer; none of them transfer to fuzzing throughput or coverage.
- Nothing in this document was run against the fuzzer. Claims about missing encoders come from reading `rq_encodings.py` class names and grep, at commit `c6e0cc7`.
- The post itself says provenance adds nothing after the JIT's integer passes on the PyPy workload, so the technique's marginal value is workload-dependent even in its home domain.
