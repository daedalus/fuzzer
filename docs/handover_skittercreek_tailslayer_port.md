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

**Status:** items 1, 2, 4, 5, 6 and 7 are implemented, wired, and tested.
Items 3 and 13 are deferred by design, not dropped — reasons preserved below.
Item 4 is functionally complete but its validation gate (step 6) has not
closed after four real campaigns, and the honest close-out for that is a
decision, not more work. Item 1 has one outstanding regression test.

---

## Open work

### A. Item 1 — cost-bound regression test inside `checksum_learner.py`

The only outstanding piece of item 1. The XOR model family is wired
(`XorBitmaskModel`, `ensure_xor_model`, `_verify_xor`), gated behind the
GF(2) and integer paths failing to verify and sharing their attempt counter
via `_maybe_recover`/`RECOVERY_RETRY_BATCH`. What never landed is the test:
`tests/test_regression_checksum_cost_bound.py` has no coverage of the XOR
path — its only `xor` mention is a comment about `final_xor` in CRCs.

`tests/test_xor_map_solver.py` asserts `elapsed < 1.0` for 32-bit recovery,
which bounds the solver standalone. That is not the same assertion: it says
nothing about cost inside `fuzz_one()`'s hot path, which is where this
module's documented 30+ second incident happened.

Lower urgency than when first written. Recovery is now Gauss-Jordan
elimination over F2, not a SAT search — polynomial, terminating, bounded by
`_MAX_PAIRS = 128` and `_MAX_FIELD_BITS = 32`, and measured at 0.5 ms on a
32×32 system with 64 pairs. So this is a regression guard against a future
change reintroducing search, not a fix for a live hazard. Write it as a
wall-clock assertion as pair count grows to `CHECKSUM_PAIRS_MAX`, matching
the existing pattern in that file.

### B. Item 4 — the false-negative dead-region case (Sequencing step 6)

`LiveBitMaskEstimator` is implemented and wired into both consumers
(`OperatorEngine.record_coverage_diff` region down-weighting,
`FormatLearner.record_liveness` padding corroboration), with
`_LIVENESS_SWITCH_AFTER = 200` and `_LIVENESS_DEAD_WEIGHT = 0.1`. Both
consumers remain gated on a real-corpus sensitivity sweep that has now run
four times and still has not exercised the failure mode it exists to catch.

| round | target | samples | result |
|---|---|---|---|
| 9 | `zlib_read` | 3,192 | threshold-stable, no dead region — compressed data has no padding |
| 10 | `png_read` | 16,063 | threshold-stable, no dead region — every chunk is CRC-guarded |
| 11 | `jpeg_read` | — | threshold-stable, no dead region — all samples attributed to region 0; 3.0% diff yield |

Threshold stability now holds on three real targets: the same final live
mask at every `switch_after` ∈ {50, 100, 200, 400, 800}. Zero false-dead
verdicts across all four campaigns. But also **zero true-dead regions**, so
the false-negative rate — the actual thing step 6 exists to measure — is
still unmeasured, and `_LIVENESS_DEAD_WEIGHT = 0.1` remains a conservative
guess rather than a calibrated value.

**The three failures are not the same failure.** Each target missed for a
different structural reason, and two of them are categorical rather than
sampling accidents: any CRC-covered format structurally rules out
coverage-dead bytes, because mutating *any* byte flips the CRC-check edge
regardless of semantic relevance. JPEG's miss was different again —
attribution, not structure: despite seeds up to 66 KB with injected
`APPn`/`COM` payloads, every usable sample landed in region 0, so the
per-region behaviour was never exercised at all.

**So this is a decision, not a task.** Target selection should start from
"which formats in the matrix have neither a whole-file nor a per-chunk
checksum" rather than from another round of corpus engineering. Round 11's
own alternative — force mutations into `APPn` trailer tails rather than
uniform random offsets — would fix JPEG's attribution problem specifically.
If neither yields a target with genuinely unchecked bytes, the honest
close-out is to record the false-negative rate as **unmeasurable against the
current target matrix**, keep the conservative weight on that basis, and
close step 6 rather than spending a fifth round on it.

Raw data and per-round writeups: `docs/sweeps/`.

### C. §2 — bounded probe window for `__afl_map_edge`

`adapters/afl_shim.c` linear-probes an open-addressing table on **every edge
execution**, not just every unique edge. A loop body that runs 10k times pays
the probe cost 10k times. The probe loop still runs to at most `map_size`
iterations (`afl_shim.c:560`). Simulated average probes per insertion,
uniform random ids:

| load | avg probes |
|------|-----------|
| 0.49 | 1.5 |
| 0.79 | 2.8 |
| 0.95 | 12.1 |
| 1.00 | 57.2 |
| saturated | `map_size` |

Two ways out, neither taken:

- **Bound the probe window** to 8–16 slots, then give up. Converts a
  `map_size`-iteration worst case into a constant, trading a drop rate for a
  bounded cost. Cheap to evaluate rather than speculative: the shim counts
  dropped edges into the SHM header, so the trade is directly observable via
  `ShmCoverage.read_dropped_edges()`.
- **Stop open-addressing on the hot path.** AFL's `mem[(prev^cur) & mask]++`
  is one load, one add, one store. You keep exact edge IDs today at the cost
  of a hash probe on every branch; you could get both by keeping the classic
  bitmap in the hot path and recovering identity lazily (the guard→address
  map is already parsed in `elf.py`).

The per-exec reset half of §2 is **done** (generation tagging, `1eb7979`,
86.9 µs → 2.2 µs on a 262,144-entry map). Only the probe bound is open.

This was previously blocked on having a target that exceeds the 8,192 floor.
That blocker is gone — see the census below — but note the load factor at the
small end is ~13%, averaging ~1.07 probes, so bounding the window buys
nothing for those binaries and measures noise against them.

### D. Unstable-edge calibration (suggested, not implemented)

The highest-leverage item in this group. There is no AFL-style per-seed
calibration: run a new seed N times, mask edges that don't reproduce.
`_run_calibration` (`fuzzer.py:3828`) is a bootstrap warm-up loop, not this.
Without it, nondeterministic edges — ASLR-, time-, or
uninitialized-memory-dependent — read as an endless supply of new coverage
and permanently absorb energy.

Nearly free from machinery already present: run each new seed 3× and compare
`read_path_hash()` (`shm.py:328`, already called at `fuzzer.py:3354` and
`:4013`). Divergence means unstable; fall back to per-edge set-diff to find
which ones.

### E. `--cmplog` defaults off

Still `action="store_true"` at `cli/commands.py:1874`. Given the i2s/Redqueen
work already in the tree and `_detect_cmplog()` (`fuzzer.py:377`), which
reliably identifies instrumented targets, it should default on whenever
detection succeeds. Magic-value and checksum branches are where the edge
count plateaus on real formats. `--no-forkserver` in the same parser is the
opt-out pattern to copy.

### F. Path hash as a second coverage dimension

The shim maintains an order- and multiplicity-sensitive rolling path hash and
`read_path_hash()` exposes it, but it is used only as a cheap change detector.
Honggfuzz-style unique-path counting is one option; the instability detector
in (D) is the better use of the same primitive.

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
document, open or closed. The probe-cost table in (C) is simulated; the
reset-cost figures are measured but on one machine. No edges/second number
anywhere here comes from an A/B against a real target. `tools/bench_paired.py`
against a fixed seed and a fixed exec budget before trusting any of it.

**Target matrix (census 2026-08-16, `--clang-scov`).** All 9 non-trivial
binaries carry a `__sancov_guards` section, so sizing is exact and nothing
falls back to branch-density estimation:

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

`ffmpeg_read` is the only target that exercises the map cap, and therefore the
only one against which (C) can be validated. Everything else sizes to the
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
- For item 4 / (B) above: does a target exist in this tree with neither a
  whole-file nor a per-chunk checksum? This is now the deciding question for
  whether step 6 can ever close, and it is cheaper to answer than to run
  another sweep.

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
