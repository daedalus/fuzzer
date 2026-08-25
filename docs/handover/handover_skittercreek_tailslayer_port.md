# Handover: skittercreek/tailslayer ports — remaining work

**Cleaned round 13** (against `6e04ed8`, 1011 commits). Completed work has been
removed from this document; what follows is only what is still open, plus the
constraints and landmines that open work depends on. See "What was removed"
at the end for where the deleted material lives.

Sources, for the two items still unported:
- `xoreaxeaxeax/skitter-creek-bath-salts` — `analysis/unspaghettify.py`,
  `analysis/gather_aliases.py`, `userspace/alias_map.h`.
- `LaurieWired/tailslayer` — `discovery/trefi_probe.c`,
  `discovery/benchmark/benchmark.cpp`, `discovery/benchmark/stats.cpp`.

**Status (round 14):** items 1, 2, 4, 5, 6 and 7 are implemented, wired, and
tested. Open items A, C, D, E and F from the round-13 cleanup are now done;
G (the intermittent `shmat()` failure) and B (item 4's validation gate) are
the only ones left, plus items 3 and 13, which stay deferred by design.

Round 14 also found and fixed a **critical pre-existing bug**: the edge table
saturated after a few hundred executions and coverage guidance silently
stopped, on the default execution path. See C2. It was found by trying to
quantify the probe-window tradeoff in C, not by looking for it — the
load-factor figure that item made its case from turned out to be the
symptom.

---

## Open work

### A. Item 1 — cost-bound regression test — **DONE**

`tests/test_regression_checksum_cost_bound.py::TestXorPathCostBounded`.
Two assertions: an absolute ceiling at the pair cap, and cap-invariance
(cost at `CHECKSUM_PAIRS_MAX` is the same whether the learner has seen that
many pairs or eight times that many) — the latter being the shape of the
original incident, which was unbounded growth rather than one slow call.
Measured 7 ms at the cap; verified to catch an injected 1.2 s regression.

### B. Item 4 — false-negative dead-region case — **UNBLOCKED, one run left**

`LiveBitMaskEstimator` is implemented and wired into both consumers, with
`_LIVENESS_SWITCH_AFTER = 200` and `_LIVENESS_DEAD_WEIGHT = 0.1`. Step 6 was
stuck for four rounds not on the estimator but on the target matrix: zlib,
png (twice) and jpeg all produced zero genuinely-dead regions, so the
false-negative rate had nothing to be measured against. Two of those misses
are structural rather than accidental — compressed data has no padding, and
any CRC-covered format rules out coverage-dead bytes outright, since
mutating any byte moves the CRC-check edge regardless of semantic relevance.

**That blocker is gone.** `tools/gen_synthetic_target.py` produces a target
with a byte region that is read but provably never reaches a branch, and no
checksum anywhere. Measured on the deterministic variant: live prefix 60/60
mutations move coverage, dead region **0/60**, identical-input reruns 0/20
differ. Ground truth by construction, asserted in
`tests/test_synthetic_target.py` so it cannot silently rot.

**Remaining work is one sweep**, not a decision any more: run
`tools/sweep_liveness_thresholds.py` against the synthetic target and
calibrate `_LIVENESS_DEAD_WEIGHT` and `_LIVENESS_SWITCH_AFTER` against a
known-dead region. Until that runs, both stay conservative guesses and the
comments at `services/operators.py` and `core/live_bit_mask.py` say so.

One caveat to carry into that sweep: use `--unstable 0`. On the unstable
variant, dead-region mutations appear to move coverage 40/60 of the time
purely from ASLR-gated edges — nondeterminism destroys the liveness signal
outright, which is worth knowing before running this against any real target
that has not been stability-calibrated first.

See `docs/sweeps/synthetic_target_ground_truth_2026-08-19.md`.

### C. §2 — bounded probe window — **DONE, and it uncovered a worse bug**

`__AFL_PROBE_MAX` (default 64) bounds both lookup and insertion in
`__afl_map_edge`; exhaustion routes through `__afl_note_drop()`, observable
via `read_dropped_edges()`. `ShmCoverage.PROBE_MAX` mirrors it, with a test
asserting the two constants cannot drift — `record_edge()` is a Python
mirror of the same loop, and an unbounded mirror would place edges the shim
could never find.

64 rather than 16. Now **measured** against the real shim rather than
simulated (201,279 distinct random edge ids into a 262,144-entry map —
`ffmpeg_read`'s shape, load 0.77): window 8 loses 4.39% of edges, 16 loses
1.55%, 32 loses 0.38%, 64 loses 0.04%. Close to the simulation that
motivated the choice, so 64 stands. A dropped edge is *permanently*
invisible, not delayed. Also removed the O(map_size) `all_live` second scan.

**The cost side did not survive measurement.** ns/edge is flat from window 8
to 1024 — the hot path at this load is dominated by cache misses over a 2 MB
table, not probe count, and lookups terminate at the matching entry
regardless of the bound. The bound is retained because it converts an
unbounded worst case into a constant, but §2's premise that probe cost was
worth attacking is not supported by measurement, and there is no speed to be
gained by lowering the window.

**The load-factor premise was wrong, and finding out why turned up a
critical bug.** See below.

### C2. Edge table saturation / coverage blackout — **FOUND AND FIXED**

`__afl_map_edge`'s probe loop had a branch for "stale entry with matching
edge_id — reclaim". The comment said reclaim; the code fell through and kept
probing. So every execution inserted a *fresh duplicate* of every edge that
fired, in a new slot, and never freed the stale copy. Occupancy grew by
(edges per execution) per execution, regardless of how few distinct edges
the target had.

Measured against the real shim (C harness compiled with `-include
afl_shim.c`, driven against a live `ShmCoverage`), 8192-entry map, 43
distinct edges — `png_read`'s shape:

| execs | slots occupied | edges visible to the fuzzer |
|---|---:|---:|
| 1 | 43 | 43 |
| 100 | 4,343 | 43 |
| 190 | ~8,000 | 43 |
| **200** | **8,192 (saturated)** | **0** |
| 300 | 8,192 | 43 — but 256-execution-old data aliasing |

Once saturated, no slot can be claimed, so the current execution writes
nothing and `get_edge_ids()` — which filters by generation — returns empty.
Coverage guidance silently stops mid-campaign. It did not even register as a
drop: `__afl_note_drop()` was gated on every slot being *live*, and a
saturated table is full of *stale* entries, so `read_dropped_edges()` stayed
at 0 throughout. Past 256 executions the 8-bit generation wraps, so a
saturated table reports a 256-exec-old execution's edges as current.

Affects every mode that resets between runs — in-process (non-`direct_lite`),
forkserver, and fork+exec — i.e. everything except `direct_lite`. **The
forkserver is the default path.** Introduced by the generation-tagging work
(`1eb7979`) that this document previously recorded as a clean win, which is
worth noting: the benchmark that justified it measured reset cost in
isolation and would not have caught this.

Fixed by reclaiming a stale same-edge entry in place. Occupancy now converges
to the union of distinct edges ever seen, bounded by the target's guard
count. Regression test: `tests/test_regression_shm_stale_reclaim.py` (6
cases, drives the real C shim; 5 of 6 fail against the pre-fix shim).

This also explains why the load-factor reasoning in the original §2 was
wrong: load was never ~13% during a campaign, it was heading to 100% within
a couple hundred executions. Bolting a 16-slot window onto a saturating
table would have dropped nearly everything.

### C3. Generation-tag aliasing (ghost edges) — **FOUND AND FIXED**

Found while auditing C2's fix rather than during it, and it is a separate
bug: reclaiming stale entries in place does not help, because the tag space
is too small rather than the reclaim logic being wrong.

Generation tags are 8 bits, so they repeat every 256 resets. An entry keeps
the tag of the last execution in which its edge fired, so an edge that fires
once and then goes quiet is read as live again exactly 256 executions later
— credited to an execution that never reached that code. Measured: fire edge
A once, then run N executions that never fire it; A reappears in the live
set at N = 256 and 512, and is absent at 255, 257 and 511. That on/off
signature is what distinguishes aliasing from a leak.

Fixed by wiping the table when the counter wraps to 0, bounding staleness to
one 256-execution cycle. Cost is one memset per 256 resets — ~86.9 µs
amortised to ~0.34 µs per execution, against the 2.2 µs the generation
scheme saves on every other execution, so the win that motivated generation
tagging is kept and only the aliasing is paid for.

**The process failure is worth more than the bug.** C2's original test suite
declared the wrap safe, and it was wrong: every test fired the same edges on
every execution, so entries were always refreshed with the current tag and
the aliasing case was never constructed. The test passed for a reason
unrelated to the property it claimed to check. `tests/
test_regression_shm_stale_reclaim.py` now has both halves, with the
steady-state one explicitly labelled as the easy half that is misleading
alone.

### D. Unstable-edge calibration — **DONE, opt-in**

`Fuzzer._calibrate_seed_stability(data, n_runs)` re-runs an accepted seed
and masks edges that do not reproduce, via the new
`ShmCoverage.mask_edges()`. Called from `CorpusManager.save_to_corpus()`
after the seed is committed, so calibration never rejects a seed — it only
ever masks edges. CLI: `--calibrate-stability N`, default 0 (off).

Two design points worth keeping:

- **Path-hash divergence is a screen, not the verdict.** The hash is order-
  and multiplicity-sensitive, so it moves whenever the same edges fire in a
  different order or a different number of times. Masking on hash divergence
  alone would suppress edges that are perfectly deterministic about *which*
  code runs. Identical hashes across all runs short-circuits to "stable";
  divergence falls through to the per-edge set-diff, which decides.
- **Masking covers hit-count buckets, not just edge ids.** Novelty has two
  sources — an unseen edge id and an unseen (edge, count-bucket) pair — and
  suppressing only the first leaves an unstable edge free to re-register
  every time its trip count crosses a bucket boundary, which for a
  nondeterministic edge is exactly what keeps happening. All eight buckets
  are marked.

**Use N=8, not 3.** Measured against the synthetic target's ASLR-gated
variant, where ground truth is 3 unstable edges: N=3 recovers the full set
only 35% of the time and misses it entirely 6% of the time; N=8 recovers it
87% of the time. Found the hard way — a live 3-run calibration returned zero
unstable edges on an input that demonstrably varies. The failure is silent
and asymmetric: a missed unstable edge keeps absorbing energy for the rest
of the campaign. The handover's original "run each new seed 3×" suggestion
was folklore.

**Opt-in, not default**, because it costs `n_runs` extra executions per
accepted seed and no A/B against a real target has been run. That is the
standing rule in this tree, not timidity — see "no coverage delta has ever
been measured" below. Turning it on by default is a reasonable follow-up
*after* someone measures it.

### E. `--cmplog` defaults off — **DONE**

Now `argparse.BooleanOptionalAction` with `default=None`: `None` = auto
(enable when `_detect_cmplog()` succeeds), `--cmplog` = force on,
`--no-cmplog` = force off. Resolved in the constructor, before
`CmplogCollector` is built — the pre-existing detection call at the
direct_lite decision runs only when `self._cmplog` is already non-None, so
it could refine *how* cmplog runs but never *whether* it runs. Detection is
skipped entirely when the flag is explicit, keeping two `nm` subprocesses
off the startup path for users who already decided.

### F. Path hash as a second coverage dimension — **CLOSED as superseded**

The instability detector in (D) is the better use of the same primitive, and
it now exists. A separate Honggfuzz-style unique-path counter is not
proposed; reopen only if someone actually wants path-count deduplication as
a distinct signal.

### G. Intermittent `shmat()` failure — open thread, no fix

Root cause still unknown, and the only genuinely open thread from the
`docs/handover/` investigations (the other two are fixed — see "What was
removed"). Twice in roughly fifty runs, the first `ShmCoverage` constructed in
a process read back an empty edge table after a child that exited 0. Header
`edge_count` was not captured at the time, so it is unknown whether the child
failed to attach or the parent raced the read.

The stale-view hypothesis (`ShmCoverage.cleanup()` leaving `from_address`
views bound to a detached mapping) is fixed but **not established as the
cause**: 0 failures in ~40 runs since, against a pre-fix rate of roughly 1 in
25 mixed-file runs, which is not yet conclusive.

Nothing to do proactively. If the drop-counter tests in
`tests/test_ctx_and_map_size.py` go intermittently red in CI, this is the
thread to pull, and the thing to capture is the SHM header
(`read_edge_count()`, `read_diag()`) alongside the child's exit status.

Note the segment is created and read successfully by the parent in the same
test, and `ipcs -m` shows no leak, so segment exhaustion (limit 4096) is not
the explanation.

### H. Deferred by design — items 3 and 13

Both correctly deferred rather than dropped. Kept here so they are not
silently reinvented without the context of why they were skipped.

**Item 3 — `compose_bitmask_maps` over `lineage.py` mutation chains.**
`compose_bitmask_maps` exists in `core/gf2_common.py` and has zero callers.
The blocker is real: `lineage.py`'s mutation chain is heterogeneous — havoc
byte flips, splices, dictionary insertions, structural mutations — and most
are **not** linear in GF(2). Insertions and deletions change length; splices
aren't bitwise-linear; only pure bit/byte-flip operators are XOR-linear.
Composition is only valid when every step in the chain is a fixed-width
XOR-linear map, which is a small subset of the operator set (see the
"bitflip" family in `operator_categories.py`). Do not build a general
lineage-composition feature around this. If a concrete need appears — e.g.
compressing a long run of consecutive pure-bitflip mutations into one composed
map for faster replay — the scoped version is: filter the chain to maximal
runs of XOR-linear-only operators, compose only within those runs, leave
everything else alone.

**Item 13 — item 5 × item 7, byte-level attribution of timing anomalies.**
Joining `ExecTimeCalibrator`'s anomaly timestamps to the mutation-event
stream would attribute a slow execution to a specific byte/operator rather
than to "some exec around this time." Both halves exist —
`core/temporal_join.py` and `core/exec_time_anomaly.py` — and
`report.py`'s `_temporal_correlation` already uses `join_streams`, but for
coverage/discovery snapshot streams, not this. Deferred until item 5 has real
campaign output to join against; same discipline as item 3.

---

## Constraints that still bind

Carried forward from removed sections because open work depends on them.

**No coverage delta has ever been measured.** Not for any item in this
document, open or closed — including the C2 fix, whose effect on edges/second
is unknown even though its effect on correctness is not. What C2 establishes
is that any pre-fix coverage measurement over more than a few hundred
executions was measuring a table that had already gone dark, so historical
edges/second figures on the reset paths should be treated as unreliable
rather than as a baseline to compare against. The probe-cost table in (C) is simulated; the
reset-cost figures are measured but on one machine. No edges/second number
anywhere here comes from an A/B against a real target. `tools/bench_paired.py`
against a fixed seed and a fixed exec budget before trusting any of it.

**Target matrix (census 2026-08-16, `--clang-scov`), plus one synthetic.**
`tools/gen_synthetic_target.py` generates a target with a controllable guard
count (`--blocks`), a provably coverage-dead byte region, no checksum, and
optional ASLR-gated unstable blocks. It is generated rather than committed
(20,000 blocks is 1.2 MB of C) and built by
`tests/test_synthetic_target.py`, which also asserts its ground-truth
properties still hold. Use it for anything needing a known answer; use the
real targets below for anything needing realism.

All 9 non-trivial real binaries carry a `__sancov_guards` section, so sizing
is exact and nothing falls back to branch-density estimation:

| target | guards | map |
|--------|-------:|----:|
| ffmpeg_read | 201,279 | 262,144 |
| png_read | 7,381 | 8,192 |
| gzip_read | 843 | 8,192 |
| zlib_read | 843 | 8,192 |
| tracecmp_target_tcg | 354 | 8,192 |
| png_read_nosan | 297 | 8,192 |
| zlib_read_nosan | 243 | 8,192 |
| cmplog_exercise_tcg | 155 | 8,192 |
| proto_target | 99 | 8,192 |

`ffmpeg_read` is the only *real* target that exercises the map cap; the
synthetic target above reaches any load on demand, which is how (C) was
finally measured rather than simulated. Everything else sizes to the
floor at ~13% load. A CTX build would multiply distinct ids by call-graph
fan-in and is the next stress test after that; every target above is `ctx=0`.
Re-run with `parse_sancov_guard_count()` before doing this work.

**`grep_read` is not an instrumented target.** It builds, but has no
`__sancov_guards` section, and `targets/grep_read.c:86` `execlp`s the *system*
`grep` binary — so the work being fuzzed happens in an uninstrumented process
and the harness reports no edges from it regardless of compilation. It
contributes nothing to a coverage A/B. Worth knowing before anyone picks it as
a "real target" to benchmark against.

**`resize()` must not copy table entries.** Load-bearing invariant introduced
by the generation-tagged reset. `resize()` copies only the 24-byte front
header; per-execution table entries are deliberately not copied because their
positions are modulus-dependent. This is safe because the slow path filters by
generation and the new table starts empty. Do not "fix" this.

**Interpreting reset cost.** The reset is irrelevant at the default map size
and bites only in one combination: a target fast enough for the clear to
dominate *and* edgy enough to need a map above 262,144. Heavy targets need the
big map but have the long exec to absorb it. Relevant if anyone revisits map
sizing.

---

## Open questions for Gabriel

- Is there a real target in the corpus (ffmpeg/png fixtures, or something in
  `docs/FINDINGS`) with a known non-CRC, non-Adler/Fletcher bitmask-style
  checksum/flags field, to use as a *real* validation case for item 1? Absent
  one, it ships with synthetic validation only and says so in the module
  docstring.
- ~~For item 4 / (B): does a target exist in this tree with neither a
  whole-file nor a per-chunk checksum?~~ **Answered: no real one, so one was
  built.** See (B) and `tools/gen_synthetic_target.py`. The follow-on
  question is whether calibrating `_LIVENESS_DEAD_WEIGHT` against a
  synthetic dead region is worth acting on, given no real format in the
  matrix has such a region — the calibration is sound, but its
  generalisation to real targets is an assumption worth stating out loud.

---

## What was removed in the round-13 cleanup

Deleted because complete; recoverable from git history and the artifacts named
here.

- **Items 2, 5, 6, 7 port plans and validation plans** — all implemented,
  wired, and tested. `core/gf2_common.py` (bitmask-vector layer, `899b59c`,
  wired into `root_cause.py` in `def6d97`); `core/exec_time_anomaly.py`
  (`1b94d4b`, wired `2b01039`); `periodicity.py` harmonic functions
  (`29e3515`); `core/temporal_join.py` (`a5040c4`).
- **Item 1's port plan** — implemented, though not as planned: `43a119e`
  replaced the per-output-bit z3 design with Gauss-Jordan elimination over F2
  (150,939 ms → 0.5 ms) and added a full-rank determinacy gate. That commit
  message is the authoritative description; only test (A) remains.
- **Item 4's implementation and wiring detail** (rounds 6–7) — done. Only the
  step 6 validation gate survives, as (B).
- **Revision notes, rounds 2–12** — a changelog, now in `git log`.
- **Generation-tagged reset narrative and memset benchmark tables** — done in
  `1eb7979`. Only the two surviving invariants were kept above.
- **"Dead classes — wire or delete"** — now fully obsolete.
  `CoverageHomogeneityDetector` is instantiated at `fuzzer.py:1531`,
  `track_parser` is imported by `core/cmplog.py:32`, and `SanitizerReport` is
  used in `services/differential.py`. Nothing on that list is dead any more.
- **Two of the three `docs/handover/` loose threads** — the z3 finalization
  segfault is fixed in `core/z3_lifecycle.py` (`a537614`) and the SHM/
  forkserver hang in `a267ff8` (`__AFL_FORKSRV=1` opt-in). The dated writeups
  remain in `docs/handover/` as history. Only the `shmat()` thread is open, as
  (G).
- **The havoc short-circuit** — fixed; regression test
  `tests/test_regression_havoc_short_circuit.py`.
- **Resolved open questions** — item 5's calibration scoping and validation
  target, item 6's `expected_period` prior, item 1's `_z3_available()` gating
  (moot: no z3 dependency), and item 4's `schedules.py` weighting-hook
  question (answered by the round-7 wiring).
- **"What NOT to port"** — the source repos are done being mined; the only
  unported items are 3 and 13, whose own constraints are recorded above.
