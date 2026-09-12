# Handover — OEIS-to-fuzzer port candidates

**Date:** 2026-09-12
**Base (fuzzer):** `bb27903` (HEAD at time of analysis)
**Companion:** `daedalus/myoeis` at `91998f7` — a README of ~127 published
sequences, no code. Nothing to port from that repo directly; the value is in
the *techniques* the sequence descriptions name.

This document is a survey, not a design. Nothing here has an oracle, a
falsifier, or a measured baseline yet — that is the first step for whoever
picks an item up, not something done in advance for them.

---

## Disposition (2026-09-12, follow-up)

All four were re-gated with actual evidence before touching code, per this
document's own standard. Two flipped, two didn't move:

- **C1 (Eytzinger layout) — still open.** Checked `cfg_cache.py`/`icfg.py`:
  no repeated binary search over a static sorted array in a hot path, only
  one-time sorts at construction. The gating question in C1 below is still
  unanswered. Not implemented.
- **C2 (quasiperiodicity) — implemented.** `core/quasiperiodicity.py` +
  `tests/test_quasiperiodicity.py`, wired into `core/analyzer_registry.py`
  behind `--corpus-quasiperiodicity` (opt-in, unmeasured, same policy as
  `--tang`/`--continuum`), consumed in `seed_picker.py` and
  `corpus_manager.py` alongside the existing PPMD bonus. The "decide between
  this and CDC" question C2 posed was resolved in favor of this one, on the
  grounds that it answers the specific question corpus minimization needs
  (per-seed coverability) rather than the cross-seed question CDC answers;
  CDC remains a separate, still-open port.
- **C3 (NAF) — implemented, gate correction.** The original "no crypto
  scalar-mult target" claim below was wrong — `targets/secp256k1_read.c`
  exercises ECDSA/ECDH/Schnorr, all of which do scalar multiplication.
  `core/mutations/naf_scalar.py` + `tests/test_naf_scalar.py`, registered as
  the `naf_scalar_mutate` operator.
- **C4 (Batcher sort) — rejected, not deferred.** Found after the fact:
  `handover_algorithm_catalogue_survey_2026-09-06.md` already surveyed every
  sort call site in both repos (47 modules, ~6,614 `sorted()`/`.sort()`
  calls) and concluded none are hot enough for a hand-rolled sorting network
  to win. C4's own gating question ("is any sort site a measured
  bottleneck?") was already answered "no" by that survey before this
  document was written. Left below unmodified for the record, but treat it
  as closed, not merely unstarted.

---

## 0. What was already found to be covered

Before listing gaps, the negative result: most of the obvious overlap between
number-theoretic/combinatorial sequence techniques and fuzzing primitives is
**already implemented**, which narrows this from "port these" to "these four
remain."

| Technique (OEIS sequence(s)) | Fuzzer location |
|---|---|
| Berlekamp–Massey / GF(2) linear complexity (A397621) | `core/berlekamp_massey.py`, `core/gf2_common.py` |
| De Bruijn sequence construction (A398874, FKM-style) | `core/mutations/structured.py::de_bruijn_bytes/bits`, `core/debruijn_cache.py` |
| Configurable CRC | `core/crc32.py` |
| Golomb/Rice coding (A380294) | `core/mutations/structured.py::golomb` |
| Elias coding family | referenced in `core/mutations/generic.py`, `core/operator_registry.py` |
| Floyd cycle detection (Pollard-rho-adjacent; A361913, A379863, A373879) | `core/cycle_detect.py`, consumed by `core/schedulers/monte_carlo.py` |

Checked and confirmed **absent**, decimal check-digit schemes (Damm/Verhoeff,
A375584/A374967) — rejected as a port target: these detect single-digit and
transposition errors in human-entered decimal strings, a different error
model from anything a binary format parser needs to reject. Not listed below.

---

## Candidates

### C1. Eytzinger array layout for hot-path lookup structures

**Sequences:** A369802 (inversion count), A375825 (triangle), A370006 (SJT
rank), A378488 (n-queens table) — all use the Eytzinger (BFS-order) layout of
an array as their underlying object.

**What it is:** a cache-oblivious layout for a sorted array used for binary
search — children of index `i` sit at `2i+1`/`2i+2` (heap order) instead of
sequential order, so a binary search walks contiguous cache lines instead of
striding across the whole array.

**Where it could apply:** `core/cfg_cache.py` and `core/icfg.py` both do
repeated lookups against decoded structures that are read far more often
than written (per the cfg_cache docstring, the whole point is amortizing
decode cost across runs). If either does binary search or repeated
point-lookups over a static, sorted array — **unverified, check before
starting** — re-laying it out Eytzinger-style is a candidate. `cpu_cache.py`
already establishes that this codebase measures cache residency rather than
assuming it (see its own header: touched working-set size predicted access
cost, allocation size did not), so this fits the file's own standard of
evidence.

**Gating question, unanswered:** does anything in the hot path actually do
repeated binary search over a static array large enough for cache layout to
matter? If the answer is "no, everything hot is a hash table," this is not
worth doing — Eytzinger only helps binary search, not hashing. **Answer this
before writing any code.**

### C2. Quasiperiodicity / string-cover detection for corpus dedup

**Sequence:** A366160 (numbers whose binary expansion is not quasiperiodic).

**What it is:** a string `w` is quasiperiodic if it is entirely covered by
(possibly overlapping) occurrences of some shorter string `c` (its "cover").
Distinct from ordinary periodicity (which requires the repeats to tile
exactly, non-overlapping) and computable in O(n) via a KMP-failure-function
walk.

**Why it's a candidate:** the fuzzer already has two structure/novelty
detectors on the same axis and this is a third, different one —
`core/corpus_compression.py` (PPMD compression ratio: "does this seed fit the
corpus's byte-level context model") and `core/periodicity.py` (FFT
autocorrelation: "does this buffer look like N repeats of an L-byte record").
Neither answers "is this seed covered by repeated occurrences of a shorter
substring" — a question closer to what corpus minimization actually wants
(redundant internal structure, not just redundancy against the rest of the
corpus).

**Explicit link to a previously-recorded gap:** a prior session identified
that `core/bloom.py`'s fuzzy dedup (`hamming_distance`-based, same-length
only per its own docstring) is blind to near-duplicates with
insertions/deletions, and proposed porting `TheAlgorithms/Python`'s rolling-hash
(Rabin–Karp) as a basis for content-defined chunking — not yet implemented.
Quasiperiodicity/cover detection is an **alternative** approach to the same
underlying problem (detecting repeated internal structure despite shifts),
not an addition to it. **Whoever picks this up should decide between the two
approaches, not build both** — the gating question is which one actually
answers what corpus minimization needs, which is not yet stated precisely
enough to answer on paper.

### C3. Non-adjacent form (NAF) — low priority

**Sequence:** A379015 (reversed NAF representation of n).

**What it is:** a signed-digit binary representation with no two adjacent
nonzero digits, guaranteeing the minimal Hamming weight among representations
using digits {-1, 0, 1}. Used in cryptographic scalar multiplication to
reduce the number of point-addition steps.

**Why it's low priority:** grep confirms no existing NAF, no scalar
multiplication, and nothing crypto-scalar-shaped in the tree (`feistel.py`
exists but is a block-cipher-structure mutator, a different thing). This is
only worth doing if a target under fuzz does elliptic-curve or other
scalar-multiplication crypto, which is not established. **Do not start this
without first confirming a target that would exercise it** — otherwise it is
a primitive with zero call sites, the exact P2 anti-pattern documented
elsewhere in this directory (unwired code that nobody reads).

### C4. Batcher odd-even merge sort as a branchless/vectorizable sort

**Sequence:** A375649 (comparator count of Batcher's odd-even merge sort).

**What it is:** a sorting network — a fixed, data-independent sequence of
compare-exchange operations, which makes it branchless and trivially
SIMD/vectorizable, at the cost of `O(n log^2 n)` comparisons versus an
optimal sort's `O(n log n)`.

**Where it could apply:** `tools/vectorize_operators.py` exists and its name
suggests exactly this kind of work is already a stated goal; `services/
seed_picker.py`'s CDF-based weighted sampling (already noted elsewhere as
using cached CDF + bisect rather than naive reconstruction) is a candidate
consumer if its sort step is ever profiled as a bottleneck — **unverified,
not currently known to be hot.**

**Gating question, unanswered:** is any sort in the hot path actually a
measured bottleneck? A sorting network is only a win when branch
misprediction or vectorization width dominates a comparison sort's cost, and
that has not been measured anywhere in this codebase for any sort site.
**Profile before proposing this as a change**, not after.

---

## How to read priority here

None of C1–C4 has the evidence to be tiered P0–P4 the way the rest of this
directory does — each still needs its gating question answered on paper, and
in C1's and C4's case, a profiling run establishing that the hot path they'd
touch is actually hot. Treat this document as a triage input to *produce*
that evidence, not as a backlog ready to implement against.

---

## Second pass (2026-09-12, later) — audit against live source

Re-read this document against the tree at `e258d8b` instead of re-running the
survey. The six "already covered" rows and the C1–C4 dispositions hold. What
the first pass missed is below.

### One of its neighbours was rejected on a premise that is false

`handover_algorithm_catalogue_survey_2026-09-06.md` §10 rejects Gray code on
structural grounds and closes with an escape clause: *"Gray code would only pay
if we added an exhaustive enumeration of all `2^k` values of a `k`-bit field,
which nothing does today."* `core/exhaustive_pool.py` is that enumeration, and
it has been since P1-5. The rejection's reasoning about the deterministic stage
is correct and unchanged — each mutant there is one edit from the base seed, not
from the previous mutant. The escape clause is what fires.

The applicable object is not the reflected binary code, though; it is the
ordering of a mixed-radix counter, which is where **A382221** (products of
primitive roots), **A371531** (multiplicative order) and **A374720** (RC4-like
KSA permutation rank) come in.

### Measured: a budget-truncated walk is a prefix of one draw, not a sample

`max_runs` does not shorten the walk evenly. The odometer carries left, so the
leading positions never move:

| space | budget | distinct values reached per draw position |
|---|---|---|
| 4 × `randrange(256)` = 4.29e9 | 20,000 | 1, 1, 79, 256 |
| 4 × `randrange(256)` = 4.29e9 | 1,000,000 (default) | 1, 16, 256, 256 |
| `delta_encode`, bounds [7,7,2,2,33,2,33] | 4,000 (harness) | 1, 1, 2, 2, 33, 2, 33 |
| `gray_code`, bounds [7,6,3,3,8,3,8] | 4,000 (harness) | 2, 7, 3, 3, 8, 3, 8 |

`exhausted` correctly reports that the space was not covered. It does not report
that the omission is all at one end.

Consequence, which is the reason this was worth doing at all: **27 of 208
operators** return `over_budget` at the harness budget, and
`tests/test_exhaustive_pool.py` skips those for every invariant it states —
including `test_no_operator_exceeds_max_len_on_any_reachable_path`, the test
that caught `utf8_widen` and `regex_bomb`. They were not unproven, they were
unexamined.

### Implemented

`ExhaustivePool(order=ORDER_SPREAD)`: visit `(k * stride) mod space` with
`stride` coprime to `space` and nearest to `space/phi`. Coprimality gives the
full period, the phi placement gives the low discrepancy; both are required and
both are asserted. For `space = 2**32` the nearest coprime is 2654435769, which
is Knuth's multiplicative-hash constant. The same 20,000-run budget on the
4.29e9 space now reaches all 256 values at all four positions.

Three properties were kept deliberately, and each cost something:

- **It only engages when the space exceeds the budget.** Reordering a walk that
  can finish would trade the `exhausted` guarantee for nothing. Spread order
  therefore never reports `exhausted`, and `strided_runs` is what distinguishes
  "sampled" from "walked".
- **The threshold is re-tested on every advance, not decided after run 0.** The
  first run of a path-dependent operator sees its narrowest tree —
  `randrange(first + 1)` with `first == 0` is a bound of 1, which is not a
  choice and is not recorded. `_shape` only grows, so the handover to the stride
  happens at most once and never reverses.
- **`NondeterministicDrawError` cannot fire in spread order**, because nothing
  is replayed value-for-value; a mismatched bound is clamped instead and counted
  on `shape_divergences`. That error is how outside entropy shows up, and the
  sweep asserting no operator has any still runs in lexicographic order.

### What the newly-checked operators showed

Nothing. No `max_len` violation and no non-bytes return across the 27, at caps
8, 4, 2 and 1. Recorded because the negative result is the result.

One reclassification: `avif_chunk_mutate`, `golomb` and `pgs_chunk_mutate`
report `too_deep` under spread order where the odometer called them
`over_budget`. The odometer had never reached their deep paths, so "over budget"
was hiding "exceeds `max_depth=16`". Those three remain unchecked, and raising
the harness depth is the follow-up.

### New candidates, gated

- **A375585 / A375959 — compression-amplification record oracle.** The sequence
  is a record-setting search over zlib output length. `targets/zlib_read.c`
  discards `total_out`; `_max_counts` records per-edge maxima and nothing
  records output size. `core/validity.py`'s `--reject-code` is the precedent for
  a harness-chosen scalar channel. **Gate: build that channel first**, or this
  is an oracle with nothing to read.
- **A374625 / A374849 / A383976 / A375156 — line-code transducer mutators.**
  A374625 is Manchester encode (1→10, 0→01), A374849 its decode. The
  structured-regularity family's entry price is naming the statistical test the
  fill defeats (§10, cellular automata, rejected for not paying it); this pays
  it — `core/randomness.py::runs_test` exists and Manchester caps every run at
  two bits, an extremal rather than random value. **Gate: no target decodes a
  line code.** Same shape as C3's original gate, and note C3's gate turned out
  to be wrong.
- **A380790 — Erdős–Turán Golomb ruler as a non-adaptive probe design** for
  `core/colorization.py`, which binary-searches ranges adaptively. Weak on
  merit: non-adaptive group testing spends more execs to save rounds, and execs
  are the currency. **Only worth revisiting if round count becomes the
  constraint** (parallel workers).

### Closed, with the reason

- **A360760 (CRC-16-IBM polynomial).** `core/crc32.py`'s model is
  `(poly, width, init, final_xor, reflect_in, reflect_out)` — width is a
  parameter — and `checksum_learner`'s GF(2)-affine family recovers CRC-32/16/8
  generically via Berlekamp–Massey. A hardcoded CRC-16 table adds nothing.
- **A379604 (bucket-sort max occupancy), A384543 (distinct `i XOR j` values).**
  Both are collision models for a hashed map. `aeedaf5` made the SHM front
  region one field per address and `adapters/shm.py` states no hash collisions;
  `edge_tracker.birthday_collision_risk` covers the rest. Moot, not deferred.
- **A381457 (bitonic sorter).** Same class as C4 and closed by the same
  47-module sort survey.
- **C1 (Eytzinger), second data point.** Every `bisect` site in the tree is a
  CDF draw over a small array (`seed_picker.py:114`, `operators.py:4447,4459`)
  or an `insort` into a sliding window (`execution_time.py`). Still no repeated
  binary search over a large static array, so still open and still unstarted for
  the same reason.
