# Handover: porting alias-solving, support-recovery, and timing-analysis algorithms into the fuzzer

Status: proposal, nothing merged yet.
Sources:
- `xoreaxeaxeax/skitter-creek-bath-salts` — `analysis/unspaghettify.py`,
  `analysis/gather_aliases.py`, `userspace/alias_map.h`.
- `LaurieWired/tailslayer` — `discovery/trefi_probe.c`,
  `discovery/benchmark/benchmark.cpp`, `discovery/benchmark/stats.cpp`.
Target repo: `daedalus/fuzzer`.

**Revision note (round 2):** reviewed the C implementation of
skitter-creek-bath-salts (`userspace/*.c`, `kernel/spaghettify.c`) and
confirmed there is no assembly in that repo (no `.S`/`.asm` files exist —
checked, not assumed). That review mostly reconfirmed the same GF(2) math
already covered in item 2 below, but surfaced two concrete refinements
folded into items 1 and 2: a stricter per-query verified-inverse pattern,
and a fixed-point/witness discrimination check for model verification.

**Revision note (round 3):** reviewed `LaurieWired/tailslayer`, a DRAM
tail-latency-reduction library — no exploit content, purely a performance
library plus its discovery/measurement tooling. Nothing there overlaps
with the skitter-creek items (different problem domain: timing statistics,
not GF(2) address algebra). Three new items (5, 6, 7) below, all
statistics/algorithms utilities with no hardware dependency once ported —
confirmed against the actual fuzzer code first: `services/runner.py`
currently applies one fixed `f.timeout` to every exec (no calibrated
per-target baseline), and `periodicity.py` is FFT-only (no cheap
prior-guided harmonic check) — so items 5 and 6 fill real, verified gaps
rather than assumed ones.

## Scope

Two independent algorithmic cores are in scope, from two different files:

1. **Address-alias solving math** (`unspaghettify.py`): recovering an
   unknown GF(2)-linear (XOR-of-selected-bits) map from observed
   input/output pairs, incrementally, with Z3, plus its Gaussian-elimination
   pseudo-inverse and map-composition helpers.
2. **Online support recovery / explore-exploit convergence**
   (`gather_aliases.py`, `MaskState`): an OR-accumulator that estimates
   *which bits matter at all* from noisy XOR-difference samples, and a
   convergence-triggered switch from broad random sampling to narrow
   exhaustive search once that estimate stabilizes.

Everything AMD/DRAM/register-specific in that repo stays there — it's not
applicable and isn't being touched, in either case.

The reason (1) ports cleanly: `unspaghettify.py`'s problem is "recover
unknown map f: {0,1}^n -> {0,1}^n where each output bit is some XOR of a
subset of input bits, given (input, output) pairs, incrementally, with early
UNSAT detection." That is the *checksum-field* and *mutation-composition*
problem the fuzzer already has open work on, minus the specific application.

The reason (2) ports cleanly: `MaskState.observe_hit` never touches
addresses, registers, or hardware semantics at all — it's pure information
theory over an opaque `n`-bit latent (live/dead per position), estimated
from a stream of XOR-difference samples. The fuzzer's exact analog is a
coverage-bitmap diff per mutant, and "which input bits actually move
coverage" is precisely the effector-map problem `schedules.py` and
`format_learner.py` currently solve more expensively (static sweep, or the
heavier `mi.py` / `transfer_entropy.py` estimators).

## What NOT to port

From skitter-creek-bath-salts:
- Register offsets, MCT/DCT encoding, DIMM geometry parsing (`gather_aliases*.py`)
- Anything requiring physical hardware access or kernel module interaction
- The live terminal visualization / GIF rendering machinery in
  `unspaghettify.py` (ANSI rendering, timers, quit-listener threads) — cute
  for a demo repo, dead weight in a headless fuzzer core module
- `gather_aliases.py`'s `dram_alias_argv_tail`, `flips_for`,
  `resolve_at_state`, `RunLog`, DIMM-size parsing, and the `--pa`/`--mask`
  CLI plumbing generally — only `MaskState`'s accumulator/convergence logic
  is in scope, not the harness around it

From tailslayer:
- The `HedgedReader` template itself and the DRAM channel-scrambling
  address math (`get_next_logical_index_address`, `DEFAULT_CHANNEL_OFFSET`
  etc.) — this is a memory-hardware performance trick with no fuzzing
  analog; nothing in the fuzzer replicates data across hardware channels.
  Worth naming explicitly since it's the library's actual headline feature
  and the most tempting thing to reach for, but it doesn't fit.
- `rdtsc`/`clflush`/core-pinning primitives (`hw_utils.hpp`,
  `detail::rdtsc_lfence` etc.) — x86-specific timing primitives; the
  fuzzer's own timing already goes through whatever
  perf-counter/instrumentation layer `f._perf_counters` provides, and nothing
  here should introduce a second, platform-specific timing source.
- `app_config.cpp/hpp`, `main.cpp`'s CLI/CSV-output plumbing — harness
  code, not algorithm.

## Seven ports, ranked by value

### 1. `IncrementalAliasSolver` → new `core/xor_map_solver.py`, wired into `checksum_learner.py`

**Source:** `analysis/unspaghettify.py:117-208`

**What it does there:** one Z3 `Solver` per output bit. Fixed constraints
(exactly-one-nonempty xor_mask, unexplored-bit identity/exclusion rules) are
added once at construction. Each `add_alias()` call appends only the new
pair's constraints to all `BITS` solvers. Z3 retains learned clauses across
`check()` calls on the same `Solver` object, so solve cost amortizes across
the run instead of restarting from scratch per batch.

**Why it's relevant:** `checksum_learner.py`'s GF(2) path
(`recover_polynomial_gcd` / `recover_lfsr` in `berlekamp_massey.py`) assumes
the target is a CRC-style affine polynomial over a sequential byte stream.
That's the wrong model for a field that's some fixed-but-unknown XOR-of-bits
combination not expressible as a single LFSR tap polynomial — e.g. a packed
bitmask/flags field, or a checksum that XORs non-adjacent byte ranges. Right
now `checksum_learner.py` has no fallback for that shape; the field gets
permanently rejected (per the module's own docstring, this is exactly the
kind of failure that blocks everything downstream of validation).

**Port plan:**

- New module `core/xor_map_solver.py`, mirroring `smt_solver.py`
  conventions: lazy `import z3` behind `_z3_available()`, module-level
  `_SOLVER_TIMEOUT_MS` constant, no hard dependency (already an optional
  extra — `smt = ["z3-solver>=4.13"]` in `pyproject.toml`, nothing new to
  add).
- Class `IncrementalXorMapSolver(n_bits: int)`:
  - `add_pair(input_bits: int, output_bits: int) -> None` — append
    constraints incrementally (rename `add_alias` → `add_pair`, drop the
    `unexplored_bits` identity-forcing constructor argument; the fuzzer
    case doesn't have a "known-unexplored" bit set the way the DRAM case
    does — every bit is fair game unless/until contradicted by evidence).
  - `solve() -> tuple[list[list[int]] | None, bool]` — returns
    `(solution, is_sat)`, same shape as `unspaghettify.check()`. `solution[j]`
    = list of input bit indices that XOR to produce output bit `j`.
  - Drop the `get_stats()` / `_cached_stats` live-display plumbing — no
    UI in this codebase; if progress visibility is wanted later, that's a
    `stats_reporter.py` concern, not this module's.
- `checksum_learner.py` integration:
  - New model family alongside the existing GF(2)-affine and integer-linear
    ones: `XorBitmaskModel`. Gate it *after* both current paths fail to
    verify (same "try affine first, more general model only on failure"
    ordering the module already uses for GF(2) vs int-linear).
  - Feed it from the same three pair sources already collected
    (`_cmplog.pairs`, format-aware extraction, seed_meta) — no new
    plumbing needed to *get* pairs, only to route them into the new solver
    when the existing recovery paths return unverified.
  - Cap `n_bits` to the field width actually observed (8/16/32/64), same
    as existing bounds — do not attempt this on multi-KB buffers; this is
    a small-field (checksum/flags-width) technique, not a whole-input one.

**Cost control (required before merge, not optional):**

`checksum_learner.py` already has a documented incident (`CHECKSUM_PAIRS_MAX`,
`RECOVERY_RETRY_BATCH`) from an unbounded-cost recovery loop blocking
`fuzz_one()` for 30+ seconds. This port must not repeat that:

- Per-bit-solver timeout (mirror `smt_solver.py`'s `_SOLVER_TIMEOUT_MS = 50`,
  not `unspaghettify.py`'s 100ms — that repo's timeout was tuned for a
  human watching a live terminal, not a fuzzing hot path).
- Reuse `CHECKSUM_PAIRS_MAX` / `RECOVERY_RETRY_BATCH` as the gating inputs
  to this path too — do not let it re-run on every new pair.
- A `test_regression_*_cost_bound.py` test (existing pattern — see
  `test_regression_checksum_cost_bound.py`) asserting wall-clock stays
  bounded as pair count grows to `CHECKSUM_PAIRS_MAX`.

**Validation plan (falsification-first, do this before wiring in):**

1. Synthetic ground truth: generate a known random XOR-bitmask map over
   8/16/32-bit fields, generate N random (input, output) pairs under it.
2. Assert `IncrementalXorMapSolver` recovers the *exact* map, not just *a*
   satisfying map — GF(2) linear systems can be underdetermined with few
   samples, so the test must include an under-determined case and assert
   the solver reports it honestly (multiple bits still ambiguous) rather
   than confidently returning a wrong-but-satisfying assignment.
3. Adversarial case: pairs generated under a model that is *not*
   expressible as XOR-of-bits (e.g. a real CRC with a real polynomial with
   carry-dependent behavior wouldn't apply here, but a mod-N integer
   checksum would) — assert UNSAT / no-solution, not a bogus fit. This is
   the same fail-closed bar `int_checksum_solver.py` documents for its own
   GCD approach ("a genuinely wrong modulus was never produced").
4. Only after 1-3 pass does this get called from `checksum_learner.py`.

**Witness selection for post-recovery verification (added on C-source
review — source: `userspace/alias_map.h`'s `calibrate()` /
`pa_calibratable()`):**

Before a candidate `XorBitmaskModel` is trusted and cached to
`f.checksum_poly`, it must be checked against held-out pairs the way
`int_checksum_solver.verify_model` already does for the integer family.
The C tool's `pa_calibratable()` explicitly rejects any candidate
verification address that's a **fixed point** of the recovered map
(`adj_alias == adj_safe`), because a fixed-point witness would make
calibration report success even for a wrong map — the check is vacuous at
that specific input regardless of correctness.

The same failure shape is possible here: a verification pair `(input,
output)` where the *candidate model's specific bit selection* happens not
to matter for that input (e.g. the differing bits between two observed
inputs don't touch any bit position the candidate model claims is live)
will "verify" the candidate without actually discriminating it from a
wrong one. `int_checksum_solver.verify_model` already guards a related but
not identical failure mode — it requires matched pairs to carry at least
two *distinct* checksum values, so a degenerate constant-output model
can't pass by matching a run of identical observations. That's global
output diversity, not per-witness discriminating power against the
specific candidate. The `XorBitmaskModel` verification step should add the
narrower check `calibrate()` does: for each candidate verification pair,
confirm the candidate model actually predicts *different* outputs for the
two inputs being compared (i.e., the pair isn't a fixed point *of this
candidate*), not just that the overall matched set has output diversity.
Reject candidate-pair combinations where it doesn't, the same way
`pa_calibratable()` refuses to even attempt calibration at a fixed-point
address rather than reporting a misleading pass.

File: `tests/test_xor_map_solver.py` (new), following the structure of
`tests/test_int_checksum_solver.py`.

### 2. `invert_xor_map` (GF(2) Gaussian elimination) → `core/gf2_linalg.py`, used by `root_cause.py`

**Source:** `analysis/unspaghettify.py:263-297` (`invert_xor_map`,
`inverse_transform`, `forward_transform`); refined against
`userspace/alias_map.h:140-190` (`apply_xor_map`, `compute_inverse`), the
C twin of the same algorithm, on second-pass review.

**What it does there:** given a solved XOR map (list of bitmasks, one per
output bit), computes a pseudo-inverse via Gaussian elimination over GF(2)
so addresses can be mapped scrambled→physical as well as physical→scrambled.

**Why it's relevant:** `root_cause.py` does Levenshtein-align +
Zeller's-ddmin over edit scripts to find the minimal causal diff between a
baseline and a crashing input. That's byte-level, transform-agnostic. If a
target applies a linear (XOR-based) transform to input bytes before use
(bit-packed protocol fields, simple obfuscation/scrambling layers seen in
some parsers), the *minimal diff in the transformed domain* and the
*minimal diff in the original input domain* aren't the same thing — ddmin
finds the former; a human wants the latter. Having a generic GF(2) map
inverter available means: if `checksum_learner.py` (or a future format
model) has already recovered a linear map for some field, `root_cause.py`
could optionally map a minimized transformed-domain diff back through the
inverse to report which *original* input bits are causal, not just which
transformed bits.

**Port plan:**

- New module `core/gf2_linalg.py` — generic, no z3 dependency (this part
  of the source repo is pure bitmask arithmetic, not Z3-based):
  - `invert_bitmask_map(masks: list[int], n_bits: int) -> list[int] | None`
    — Gaussian elimination pseudo-inverse; returns `None` if singular
    (non-invertible — must be handled, not asserted away; a solved XOR
    map from partial evidence is not guaranteed invertible).
  - `verified_apply_inverse(masks_fwd: list[int], masks_inv: list[int],
    value: int) -> int | None` — **added on C-source review.** The
    Python-only plan above only catches *total* singularity at
    construction time (`invert_bitmask_map` returns `None` once, globally).
    The C implementation is stricter and catches more: `target_to_alias`
    never trusts the pseudo-inverse output directly — it forward-applies
    the candidate result and rejects it *per query* if the round-trip
    doesn't reproduce the input:
    ```c
    adj_alias = apply_xor_map(map->inverse, map->bits, adj_target);
    if (apply_xor_map(map->forward, map->bits, adj_alias) != adj_target)
        return -1;
    ```
    This catches rank-deficient rows that were silently zeroed during
    construction (per `compute_inverse`'s own comment: "pa[col]
    unrecoverable; row left as identity, zeroed below") but only actually
    break round-trip correctness for *specific* input values whose
    recovery depended on exactly those dropped rows — a failure mode a
    one-time global rank check at construction can't see, because the
    matrix isn't fully singular, just partially so. `verified_apply_inverse`
    ports this per-query recheck: compute the candidate inverse
    application, forward-apply it, compare, return `None` on mismatch
    instead of a silently-wrong value. This is the function `root_cause.py`
    should actually call (see item below) — not the raw
    `invert_bitmask_map` output applied blind.
  - `compose_bitmask_maps(inner: list[int], outer: list[int]) -> list[int]`
    (port of `compose_xor_maps`) — apply for reason (3) below.
  - `apply_bitmask_map(masks: list[int], value: int) -> int` (port of
    `forward_transform`/`inverse_transform`, unified into one function
    since forward vs inverse is just "which map you pass in").
- This module has no dependents changed in this PR — it's a leaf utility.
  Land it and its tests standalone first. Wiring it into `root_cause.py`
  is a **separate, later PR**, gated on item 1 actually producing linear
  maps worth inverting in practice. Don't wire speculative plumbing into
  `root_cause.py` before there's a real producer of `masks` to feed it.

**Validation plan:**

1. Round-trip property test: for random invertible bitmask maps,
   `apply_bitmask_map(invert(...), apply_bitmask_map(masks, x)) == x` for
   all `x` in a sampled range (and exhaustively for `n_bits <= 16`).
2. Singular-map test: construct a known-non-invertible map (e.g. two output
   bits both mapping to the same single input bit), assert `None`, not a
   silently wrong pseudo-inverse.
3. **Partial-rank test (added on C-source review):** construct a map that
   is *not* globally singular — `invert_bitmask_map` succeeds and returns a
   non-`None` result — but where a specific subset of input values depend
   on the dropped/unrecoverable rows from construction. Assert
   `verified_apply_inverse` returns `None` for exactly that subset while
   `invert_bitmask_map`'s raw output would silently round-trip incorrectly
   for the same inputs if applied without the forward-recheck. This is the
   test that would have caught the gap between "matrix construction didn't
   error" and "this specific value is actually recoverable" — the
   distinction the plain singular-map test in (2) doesn't exercise.

File: `tests/test_gf2_linalg.py` (new).

### 3. `compose_xor_maps` applied to `lineage.py` mutation chains — exploratory, not committed

**Source:** `analysis/unspaghettify.py:286-296`

**Why it's *not* a clean port (flagging honestly rather than overselling):**
`lineage.py`'s mutation chain is a sequence of heterogeneous operators —
havoc byte flips, splices, dictionary insertions, structural mutations —
most of which are **not** linear-in-GF(2) (insertions/deletions change
length; splices aren't bitwise-linear; only pure bit/byte-flip operators
are XOR-linear). `compose_bitmask_maps` only composes cleanly when every
step in the chain is a fixed-width XOR-linear map. That's a small, specific
subset of `lineage.py`'s operator set (`operator_categories.py` — check the
"bitflip" family specifically).

**Recommendation:** do not build a general lineage-composition feature
around this. If there's a concrete need later (e.g. compressing a long run
of consecutive pure-bitflip mutations in a chain into one composed map for
faster replay), it's a narrow, well-scoped follow-up: filter the chain to
maximal runs of XOR-linear-only operators, compose only within those runs,
leave everything else as-is. Not doing this now — listed here so it isn't
silently reinvented later without the context of why it's hard.

### 4. `MaskState` OR-accumulator convergence → new `core/live_bit_mask.py`, feeding `schedules.py` / `format_learner.py`

**Source:** `analysis/gather_aliases.py:363-400` (`MaskState`,
`observe_hit`), constants `MASK_SWITCH_AFTER = 200` / `RANDOM_SAMPLES =
10000` at lines 37/51.

**Information-theoretic reading of what it does there:** model each of the
`n` candidate bit positions as an unknown binary latent — *live* (actually
participates in the observed alias relation) or *dead* (never does).
Before any samples, that's `n` bits of entropy over the support set. Each
sample's `target ^ alias` is a noisy query that reveals a subset of the
live bits — whichever ones happened to differ on that draw — and
`accumulated_mask |= (target ^ alias)` is a monotone, one-sided estimator
of the true live-bit set: it can only grow, never shrink, and converges to
exactly the true support as samples accumulate, because a truly-live bit
has vanishing probability of *never once* appearing in the diff over many
draws (coupon-collector tail). `MASK_SWITCH_AFTER` (200 hits) is a fixed
stopping rule betting that the mask's growth rate — which decays
geometrically, since already-confirmed bits contribute zero new
information — has dropped low enough to stop paying for broad low-yield
sampling (`RANDOM_SAMPLES = 10000` per probe in wide mode) and switch to
exhaustive search restricted to just the converged mask
(`samples_for` returns `0`). That's an explore→exploit transition
triggered by a convergence detector on a sufficient statistic, not a fixed
iteration budget — the interesting part, not the OR itself.

**Why it's relevant, and how it differs from item 1:** item 1
(`xor_map_solver.py`) answers *"what exactly is the map"* — precise,
exact, Z3-backed, expensive per bit. This answers a cheaper, different,
prior question: *"which input bits/bytes matter at all, to anything"* —
O(1) work per sample, no solver. That's the effector-map / byte-sensitivity
problem `schedules.py`'s energy allocator and `format_learner.py` need an
answer to before spending mutation budget, and it's currently solved either
by a one-shot static sweep (toggle-and-check, not adaptive to new evidence
as the run progresses) or by the much heavier distributional estimators in
`mi.py` / `transfer_entropy.py`. This is a legitimate cheap first-pass
filter to run ahead of those, not a replacement for either existing
approach.

**Port plan:**

- New module `core/live_bit_mask.py`, no external dependencies (pure
  bitmask arithmetic, same footing as `core/gf2_linalg.py`):
  - `LiveBitMaskEstimator(n_bits: int, switch_after: int = 200)` — port of
    `MaskState`, renamed for domain: `observe(baseline: int, mutant: int)`
    replaces `observe_hit(target, alias, run_log)`; drop the `run_log`
    event-emission entirely (that's `gather_aliases.py`'s CLI-progress
    concern, not this module's — if progress needs surfacing later, do it
    through whatever stats-reporting convention `schedules.py` already
    uses, not by porting `RunLog`).
  - `.mask` (accumulated live-bit mask so far), `.samples_seen`,
    `.is_converged` (mirrors the `switched_to_dynamic` flag — true once
    `switch_after` consecutive `observe()` calls passed without the mask
    growing; note this is a *stronger* condition than the source's
    "total hits >= threshold" and is deliberately stricter, see validation
    note below).
  - No mode-switching CLI/sampling logic ported (`samples_for`,
    `choose_initial_mask`, `pick_pa`) — those govern *how DRAM addresses
    get chosen to probe*, a concern with no fuzzer analog; the fuzzer
    already has its own input-selection machinery (`schedules.py`, `ga.py`)
    and this module only needs to consume `(baseline, mutant)` coverage
    pairs that machinery already produces, not decide what to try next.
- Fuzzer-domain mapping: `target`/`alias` (two addresses known to alias)
  → `baseline_coverage_bitmap`/`mutant_coverage_bitmap` (coverage bitmaps
  from two runs of the same seed differing in one byte/bit). The XOR
  reveals which *coverage bits* moved; accumulated over many mutants of the
  same seed, the converged mask tells you which coverage-edge positions are
  reachable-and-sensitive at all for that seed — feed that back per-byte
  (which input byte was mutated to produce each sample) to get a per-byte
  liveness map, i.e. an online effector map.
- Integration points, both consumers only, no shared state between them:
  - `schedules.py`: gate mutation-site selection so bytes with zero
    observed liveness after convergence get down-weighted, same spirit as
    existing power-schedule weighting, one more signal into that existing
    allocator — not a new allocator.
  - `format_learner.py`: a converged all-dead byte run within a field is
    itself a signal (probably padding, alignment, or an unparsed/ignored
    region) — cheap corroborating evidence for format inference, to be
    combined with, not replace, whatever grammar/structural signals it
    already uses.

**Validation plan (falsification-first):**

1. Synthetic ground truth: fixed byte-vector length, a known subset of
   byte positions wired to affect a synthetic "coverage" function, the
   rest wired to no-ops. Generate random mutants, observe coverage-bitmap
   diffs, assert the converged mask's *byte-level* projection exactly
   equals the known-live set.
2. Convergence-threshold sensitivity: sweep `switch_after`, plot
   false-negative rate (truly-live bytes wrongly excluded because they
   rarely trigger — e.g. a byte gating a rare branch) vs. samples spent.
   Report this curve plainly rather than picking one `switch_after` value
   by feel; a rare-but-real live byte is exactly the failure mode a
   convergence detector can silently produce, and it's the reason this
   module's `is_converged` uses a stricter consecutive-no-growth condition
   than the source's simple cumulative-count threshold — that stricter
   test still needs to be checked against this sweep, not assumed correct.
3. Explicit non-goal check: confirm the estimator does *not* claim
   liveness precision it doesn't have — `.mask` says "moved coverage at
   least once," never "moved coverage on every mutation," and no code path
   should conflate the two (this is the same fail-closed discipline as
   `int_checksum_solver.py`'s modulus recovery: absence of evidence is
   reported as unresolved, not as a negative claim).

File: `tests/test_live_bit_mask.py` (new).

### 5. Calibrated robust-threshold spike detector → new `core/exec_time_anomaly.py`, informing `runner.py`

**Source:** `LaurieWired/tailslayer`, `discovery/trefi_probe.c:134-224`
(`calibrate_tsc_ghz`, calibration loop, main probe loop).

**What it does there:** before probing for DRAM refresh stalls, it runs a
calibration phase — thousands of baseline `timed_probe()` samples with no
manipulation — sorts them, and computes percentiles (median, p90, p99,
p999, p9999). The live-probe anomaly threshold is then set as `thresh_mult
* median` (default 2x), **not** mean ± N·stddev. Using the median as the
center is deliberate and load-bearing: latency distributions are heavily
right-skewed (occasional huge stalls), so a mean-based threshold gets
dragged upward by the very spikes it's trying to detect, while the median
stays robust regardless of how bad the tail is. Every live sample exceeding
that threshold gets recorded as a timestamped `spike` for later analysis;
everything else is discarded immediately (no need to keep full-resolution
non-anomalous data).

**Why it's relevant:** `runner.py` applies one fixed `f.timeout` to every
execution, uniformly across the whole corpus (confirmed at
`services/runner.py:382`, with the documented one-second
`_INITIAL_STOP_TIMEOUT` ceiling as a separate, unrelated hang-closing
mechanism). A single fixed ceiling has two failure modes a calibrated
threshold doesn't:

- **False negatives:** an input that runs slow-but-under-`f.timeout` never
  gets flagged as unusual, even if it's 50x slower than every other input
  in the corpus for that seed family — exactly the kind of signal that
  points at an algorithmic-complexity bug (quadratic blowup, hash-flooding,
  pathological regex backtracking) that a fixed timeout set conservatively
  high enough to avoid false hangs will never surface at all.
- **False positives / lost throughput:** a timeout set conservatively low
  to catch those cases would instead false-positive on inputs that are
  legitimately slow for that target (large valid inputs, expensive-but-not-
  buggy code paths), burning campaign time on retries or wrongly discarding
  productive seeds.

A per-target (or per-seed-family) calibrated baseline sidesteps both: it
answers "is this execution unusual *for this target*," not "did it exceed
an arbitrary global number."

**Port plan:**

- New module `core/exec_time_anomaly.py`, no C/asm/hardware dependency —
  pure statistics over an execution-time stream the fuzzer already
  measures per-exec (`f._perf_counters` per the code touched in
  `runner.py` around line 247 already threads perf data through; this
  consumes existing measurement, adds no new instrumentation):
  - `ExecTimeCalibrator(min_samples: int = 200)`: `observe(exec_time: float)`
    accumulates baseline samples; `.threshold(mult: float = 2.0) ->
    float | None` returns `mult * median` once `min_samples` reached, else
    `None` (fail-closed — no premature threshold on an under-sampled
    baseline, mirroring `trefi_probe.c`'s own `n_spikes < 10` "insufficient
    data" bail-out).
  - `is_anomalous(exec_time: float, threshold: float) -> bool` — trivial,
    but named and tested so call sites read as intent, not a bare `>`.
  - No port of the CSV/spike-buffer/timestamp-array machinery verbatim;
    the fuzzer already has crash/interesting-input recording paths
    (`crash_metadata.py` per the module list) — an anomalous-but-
    non-crashing slow input should be routed through whatever "interesting,
    save it" path already exists for coverage-increasing inputs, tagged
    with the anomaly, not written to a separate bespoke CSV format.
- Integration is additive only, never replacing the hard safety timeout:
  `f.timeout` stays exactly as-is as the hang-closing ceiling
  (`runner.py`'s existing logic, including the documented fork-safety
  reasoning around `_INITIAL_STOP_TIMEOUT`, is untouched). This module
  only adds a *softer*, statistically-grounded signal for "unusually slow
  but still completed" execs, surfaced as a new interestingness signal
  alongside coverage — not a new rejection/timeout path.

**Validation plan (falsification-first):**

1. Synthetic baseline + injected-anomaly test: generate a baseline
   distribution with known right-skew (e.g. log-normal), inject a small
   fraction of samples at 10-50x the median, assert the calibrated
   threshold flags the injected anomalies and not the baseline noise
   across a range of `thresh_mult` values — report the false-positive /
   false-negative tradeoff curve explicitly rather than asserting one
   `thresh_mult` is simply "correct."
2. Median-vs-mean regression test: construct a baseline where mean-based
   thresholding would demonstrably misfire (heavy-tailed distribution
   where a `mean + 2*stddev` threshold either flags most of the tail as
   "baseline" or flags none of it) and assert the median-based approach
   doesn't share that failure — this is the specific property being
   ported, so it needs its own explicit test, not just an implicit
   assumption.
3. Insufficient-data guard test: assert `.threshold()` returns `None`,
   not a garbage value, below `min_samples`.

File: `tests/test_exec_time_anomaly.py` (new).

### 6. Harmonic-binned periodicity classifier → complement to `periodicity.py`

**Source:** `discovery/trefi_probe.c:236-329` (interval computation, harmonic
binning, fine histogram peak-finding, verdict thresholds).

**What it does there:** given a candidate period `T` from domain knowledge
(the DRAM tREFI spec value, in this case), it doesn't reach for FFT.
Instead: compute inter-spike intervals, bin each interval by whether it
falls within ±15% of `1T`, `2T`, or `3T`, and report what fraction of all
intervals land on some harmonic of `T` at all. A verdict (`PERIODIC` /
`WEAK SIGNAL` / `NO SIGNAL`) falls out of simple fixed thresholds on that
fraction (>30% / >15% / else). Separately, a fine-grained 200-bin histogram
restricted to the region right around `1T` locates the exact peak, giving
a precise period estimate and a deviation-from-expected percentage.

**Why it's relevant, and how it differs from what's already in
`periodicity.py`:** `SpectralPeriodicity`/`fisher_g_pvalue` in
`periodicity.py` is a general-purpose, no-prior-knowledge FFT approach —
it has to search the whole frequency spectrum because it doesn't assume
anything about what period to look for. Harmonic binning is the opposite
regime: cheap, interpretable, *and requires a prior* — an expected period
`T` from domain knowledge. The fuzzer has exactly this kind of prior
available in several places: a target's declared/observed heartbeat
interval, a suspected polling-loop period inferred from source or docs, or
even a period `berlekamp_massey.py`/`chi_squared.py` already flagged as a
candidate on a previous, cheaper pass. When such a prior exists, harmonic
binning is strictly cheaper than a full FFT (`O(n)` single pass over
intervals vs `O(n log n)`) and gives a directly interpretable "% of events
at this exact harmonic" statistic instead of a spectral-density plot that
still needs peak-picking and a significance test.

**Port plan:**

- New function(s) in `periodicity.py` itself (not a new module — this is
  a second, cheaper *mode* of the same "is this signal periodic" question
  the module already owns, not a separate concern):
  - `harmonic_fraction(intervals: list[float], expected_period: float,
    n_harmonics: int = 3, tolerance: float = 0.15) -> dict[int, float]` —
    returns `{harmonic: fraction}` plus a `total` key, port of the binning
    loop.
  - `locate_peak_period(intervals: list[float], expected_period: float,
    bins: int = 200, search_width: float = 0.5) -> tuple[float, float]` —
    returns `(peak_period, deviation_fraction)`, port of the fine
    histogram.
  - `classify_periodicity(harmonic_total_fraction: float) -> Literal
    ["periodic", "weak", "none"]` — port of the verdict thresholds
    (0.30 / 0.15), as named constants at module level, not magic numbers
    inline (match the existing convention of named threshold constants
    seen throughout `core/`, e.g. `CHECKSUM_PAIRS_MAX`).
- Call-site guidance in the docstring: this is only valid when a real
  prior `expected_period` exists; document plainly that this is *not* a
  replacement for `SpectralPeriodicity` when no prior is available — the
  two are for different situations, not competing implementations of the
  same thing.

**Validation plan:**

1. Synthetic periodic signal at a known period with known noise level,
   assert `classify_periodicity` returns `"periodic"` when noise is low
   and correctly degrades to `"weak"`/`"none"` as noise increases —
   sweep noise level and report the transition, don't just check one
   fixed noise setting.
2. Non-periodic (pure random interval) control: assert classification is
   `"none"`, guarding against a threshold picked so loose it calls noise
   periodic.
3. Cross-check against `SpectralPeriodicity` on the same synthetic signal:
   the two methods should agree on the qualitative periodic/non-periodic
   call even though they use unrelated math — a disagreement on synthetic
   ground truth is a bug in one of them, worth a test that catches it.

File: extend `tests/test_periodicity.py` (existing) rather than a new file.

### 7. K-way sliding-window temporal join → new `core/temporal_join.py`

**Source:** `discovery/benchmark/benchmark.cpp:206-257`
(`Benchmark::pair_samples_n`).

**What it does there:** aligns `N` independently-timestamped sample
streams (one per memory channel/replica, each with its own clock drift and
sampling jitter) into matched tuples. It's a generalized K-way merge: track
one read pointer per stream, find the stream with the earliest current
timestamp, and either (a) accept a match if all `N` pointers' timestamps
fall within a small tolerance window (`MAX_PAIR_GAP`), advancing every
pointer together, or (b) if the spread is too wide, advance only the
laggard pointer and retry. This is the multi-way generalization of the
classic two-pointer merge-join, adapted for approximate (tolerance-window)
rather than exact key matching.

**Why it's relevant:** several places in the fuzzer produce independently
timestamped event streams that currently have no shared alignment utility:
distributed/parallel worker telemetry (corpus-sync events, crash
timestamps across workers with independent clocks), or — most directly —
correlating the anomalous-execution timestamps item 5 would produce
against the mutation-event stream that produced each input, to attribute
*which specific byte/operator* was responsible for a timing anomaly rather
than just knowing "some exec around this time was slow." Right now that
attribution would require exact-index correlation (assuming perfectly
synchronous single-threaded execution); a tolerance-window K-way join
makes the same correlation possible when the streams involved are async
or independently clocked, which is exactly the situation once anything
runs on multiple worker threads/processes.

**Port plan:**

- New module `core/temporal_join.py`, no dependencies:
  - `join_streams(streams: list[list[tuple[float, T]]], max_gap: float) ->
    list[tuple[T, ...]]` — generic over payload type `T` via a `TypeVar`;
    port of `pair_samples_n`'s pointer-advance logic, generalized from the
    source's `latency`-specific payload (`min_latency` selection) to
    "return whichever payload the caller cares about"; the *value*
    selection policy (source took `min` because replicated data made any
    replica's value equally valid) is caller-specific, so make it a
    parameter: `value_fn: Callable[[list[T]], T] = lambda vals: vals[0]`,
    defaulting to first-stream-wins rather than assuming `min` is always
    right for the fuzzer's use cases.
  - Keep the algorithm's core property explicit in the docstring: this is
    a **greedy, single-pass, non-optimal** join (it never backtracks a
    laggard-advance decision), which is the right tradeoff for the
    100k+-sample real-time use case it was built for, but should be
    documented as a known limitation, not silently assumed lossless.
- No integration into item 5 in this PR — land as a standalone, tested
  leaf utility first, same "don't wire speculative plumbing ahead of a
  real producer" discipline as item 2's `gf2_linalg.py`. The item-5 ×
  item-7 combination (join anomaly timestamps to mutation events) is a
  concrete follow-up once item 5 actually exists and produces timestamped
  output worth joining against.

**Validation plan:**

1. Exact-alignment test: `N` streams with identical timestamps, assert
   every event pairs (100% pairing rate, matching the source's own
   diagnostic `n_paired/n_samples` reporting convention).
2. Drift test: streams with a small constant clock offset within
   `max_gap`, assert pairing still succeeds; streams with offset exceeding
   `max_gap`, assert the laggard-advance logic correctly drops orphaned
   events from the faster stream rather than mispairing them.
3. Adversarial ordering test: a stream with a burst of closely-spaced
   events immediately after a gap, verifying the greedy advance doesn't
   pair the wrong burst member (a real risk in any greedy nearest-match
   join) — construct a case by hand where a greedy choice is provably
   suboptimal versus a hypothetical backtracking join, and assert the
   module's docstring claim about this limitation is actually true of the
   implementation, not just asserted in prose.

File: `tests/test_temporal_join.py` (new).

## Sequencing

1. `core/gf2_linalg.py` + tests (item 2) — no dependencies, smallest, do
   first, unblocks nothing but itself.
2. `core/live_bit_mask.py` + tests (item 4) — no dependencies, no solver,
   cheap; do alongside item 2 since both are leaf utilities with no
   integration risk yet.
3. `core/xor_map_solver.py` + tests (item 1, solver only, not wired in) —
   depends on nothing but z3 (already optional dep).
4. Cost-bound test for the solver in isolation (synthetic benchmark, not
   yet inside `checksum_learner.py`'s hot path).
5. Wire item 1 into `checksum_learner.py` as a third model family, gated
   behind the existing two failing first. Add/extend
   `test_regression_checksum_cost_bound.py` to cover the new path.
6. Run item 4's convergence-threshold sensitivity sweep (validation step 2
   above) on real corpus seeds, not just synthetic ones, before wiring its
   output into `schedules.py`'s weighting — a real coverage bitmap may have
   noisier/rarer-triggering bits than the synthetic test covers.
7. Wire item 4 into `schedules.py` (byte down-weighting) and
   `format_learner.py` (padding/dead-region signal) as two separate,
   independently revertible changes.
8. Item 3 (lineage composition) — explicitly deferred, not scheduled.
9. `core/exec_time_anomaly.py` + tests (item 5, calibrator only, not wired
   into `runner.py` yet) — no dependencies beyond existing perf-counter
   plumbing already in `runner.py`.
10. `harmonic_fraction`/`locate_peak_period`/`classify_periodicity` added
    to `periodicity.py` + tests (item 6) — no dependencies, independent of
    everything else in this doc.
11. `core/temporal_join.py` + tests (item 7) — no dependencies; can land
    any time, including before item 5, since it doesn't need item 5's
    output to be tested standalone.
12. Wire item 5 into `runner.py` as an additive interestingness signal
    (never replacing the hard `f.timeout` safety ceiling) — gated on
    item 9's synthetic false-positive/negative sweep giving an acceptable
    tradeoff at some `thresh_mult`.
13. Item 5 × item 7 combination (join anomaly timestamps to mutation
    events for byte-level attribution) — explicitly deferred until item 5
    has real campaign output to join against, same discipline as item 3.

Each step lands as its own PR. No step depends on a later step being
merged first, so this can stop after any step with a coherent, tested
state — falsification-first, same as the rest of the repo's convention:
if the synthetic validation in step 3 turns up that recovery is unreliable
below some sample-count threshold, or step 6's real-corpus sweep shows
item 4's false-negative rate doesn't converge to something acceptable
within a reasonable sample budget, or step 9's synthetic sweep shows item
5's threshold can't separate injected anomalies from baseline noise at any
reasonable `thresh_mult`, any of these is a valid place to stop and report
the null result rather than proceeding regardless.

## Open questions for Gabriel

- Is there a real target in the corpus (ffmpeg/png/etc. fixtures, or
  something in `docs/FINDINGS`) with a known non-CRC, non-Adler/Fletcher
  bitmask-style checksum/flags field to use as a *real* (not just
  synthetic) validation case for item 1? Absent one, ship with synthetic
  validation only and say so plainly in the module docstring.
- `_z3_available()` gating means item 1 silently no-ops on installs without
  the `smt` extra — confirm that's the desired behavior (matches
  `smt_solver.py`'s existing precedent) rather than a hard dependency bump.
- For item 4: does `schedules.py` already have a per-byte weighting hook
  clean enough to attach a new signal to, or does adding one require
  touching the allocator's existing structure? Worth a quick look before
  step 7 rather than discovering it mid-PR.
- For item 4: is there recorded coverage-bitmap-per-mutant data anywhere
  already (from a past campaign, logged for other analysis) that could
  serve as the "real corpus seeds" input to the step 6 sensitivity sweep,
  or does that need a fresh instrumented run?
- For item 5: is there a known past campaign with a real, confirmed
  algorithmic-complexity bug (a target that was slow-but-under-timeout
  before it was found some other way) to use as a real-world validation
  case, the same ask as the item-1 open question above? Absent one, item
  5 ships with synthetic-only validation and the doc should say so.
- For item 5: should the calibration baseline be scoped per-target,
  per-seed-family, or global-per-campaign? The source tool calibrates
  once per hardware config because that's stable for the run; the
  fuzzer's execution-time distribution likely shifts across seed families
  within one campaign (a PNG parser and a font parser have different
  "normal"), so a single global baseline risks the same
  mean-vs-median-style misfire the module is designed to avoid at a
  coarser level. Worth deciding before step 12, not during it.
- For item 6: which existing modules (or targets) actually have a usable
  `expected_period` prior available today? If none currently do, item 6
  is correct to land as a tested-but-unused utility and wait for a real
  caller, same as items 2 and 7.
