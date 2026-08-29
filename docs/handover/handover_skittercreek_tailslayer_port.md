# Handover: skittercreek/tailslayer ports — remaining work

**Cleaned round 15.** Completed work is removed from this document as it
closes; what follows is only what is still open, plus the constraints and
landmines that open work depends on. See "What was removed" at the end for
where the deleted material lives.

Sources, for the two items still unported:
- `xoreaxeaxeax/skitter-creek-bath-salts` — `analysis/unspaghettify.py`,
  `analysis/gather_aliases.py`, `userspace/alias_map.h`.
- `LaurieWired/tailslayer` — `discovery/trefi_probe.c`,
  `discovery/benchmark/benchmark.cpp`, `discovery/benchmark/stats.cpp`.

**Status (round 16):** items 1, 2, 4, 5, 6 and 7 are implemented, wired and
tested. Round 16 closed B (item 4's validation gate — calibrated against the
synthetic known-dead region). The only genuinely open thread left is G, the
intermittent `shmat()` failure. Items 3 and 13 stay deferred by design (H).
The closed rounds' writeups are pruned; the ledger at the end says where each
one's artifacts live.

---

## Open work

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
is unknown even though its effect on correctness is not. What the round-14 edge-table
saturation fix establishes is that any pre-fix coverage measurement over more
than a few hundred executions was measuring a table that had already gone
dark, so historical edges/second figures on the reset paths should be treated
as unreliable rather than as a baseline to compare against. The reset-cost
figures are measured, but on one machine. No edges/second number
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
synthetic target above reaches any load on demand, which is how the probe
window was finally measured rather than simulated. Everything else sizes to the
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
  built.** See `tools/gen_synthetic_target.py`.
- ~~The follow-on: is calibrating the liveness thresholds against a synthetic
  dead region worth acting on, given no real format has one?~~ **Answered by
  the round-16 sweep** (`docs/sweeps/synthetic_liveness_calibration_2026-08-29.md`).
  Yes for the dead side: the estimator gives the known-dead region a correct
  DEAD verdict at every `switch_after` and never misclassifies the live
  region. No for lowering the default: `switch_after` is unconstrained from
  below by correctness *on this target*, because the case a high floor
  guards against — a real cold-but-live region that emits a long no-growth
  run before its first edge — is exactly what a synthetic target cannot
  exhibit. So the dead side is now measured and the cold-live floor is a
  stated assumption about real targets, not a guess. The generalisation
  caveat you asked to state out loud is stated, at both constants.

---

## What was removed

Deleted because complete; recoverable from git history and the artifacts named
here.

### Round 13

- **Items 2, 5, 6, 7 port plans and validation plans** — all implemented,
  wired, and tested. `core/gf2_common.py` (bitmask-vector layer, `899b59c`,
  wired into `root_cause.py` in `def6d97`); `core/exec_time_anomaly.py`
  (`1b94d4b`, wired `2b01039`); `periodicity.py` harmonic functions
  (`29e3515`); `core/temporal_join.py` (`a5040c4`).
- **Item 1's port plan** — implemented, though not as planned: `43a119e`
  replaced the per-output-bit z3 design with Gauss-Jordan elimination over F2
  (150,939 ms → 0.5 ms) and added a full-rank determinacy gate. That commit
  message is the authoritative description. Its cost-bound test (A) has since
  landed too — see Round 15 below.
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
- **The "Edge coverage analysis" appendix** — merged into this document as an
  appendix in `852e274` (2026-08-18), then dropped by the round-13 cut
  (`97c1d52`) the following day without an entry here. Recorded now, because
  four source comments went on citing `docs/edge-coverage-analysis.md` for
  another ten days after it stopped existing. Nothing in it is lost work: its
  §2 reset cost is fixed in `1eb7979`, its probe-window bound shipped as
  `__AFL_PROBE_MAX` (and the shim now carries better, *measured* drop rates
  than the appendix's simulated table), unstable-edge calibration is (D),
  cmplog-defaults-off is (E), path-hash-as-second-dimension is (F), the havoc
  short-circuit is fixed, its "dead classes" list is obsolete, and its
  §6 deterministic stage is wired and covered by
  `tests/test_regression_havoc_short_circuit.py` and
  `tests/test_deterministic_stage.py`. Its one genuinely open item, the
  intermittent `shmat()` failure, is (G) above and carries every detail the
  appendix had.

### Round 15

- **(A) item 1's cost-bound regression test** — done;
  `tests/test_regression_checksum_cost_bound.py::TestXorPathCostBounded`.
- **(C) bounded probe window**, and the two bugs it uncovered while being
  measured: **(C2) edge-table saturation / coverage blackout** and **(C3)
  generation-tag aliasing (ghost edges)**. Both were critical pre-existing
  bugs on the default execution path, both fixed. The two invariants they left
  behind are carried into "Constraints that still bind"; the narrative is in
  git history.
- **(D) unstable-edge calibration** — done, opt-in.
- **(E) `--cmplog` defaults off** — done.
- **(F) path hash as a second coverage dimension** — closed as superseded, not
  implemented. Do not re-propose it without reading why in git history first.

### Round 16

- **(B) item 4's step-6 validation gate** — done. The blocker turned out not
  to be the run but the tool: `tools/sweep_liveness_thresholds.py` could not
  execute a target at all (its `--target` arg was dead, no `--unstable`
  flag, both paths drove `FormatLearner` over fabricated transitions). Fixed
  by adding a `--synthetic-target` mode that builds
  `gen_synthetic_target.py`'s known-dead target and drives the real
  `LiveBitMaskEstimator` against it. Result: the dead region earns a DEAD
  verdict at every `switch_after` in {50,100,200,400,800} (in exactly
  `switch_after` samples) and the live region never does — false-negative and
  false-positive rate both 0. `_LIVENESS_SWITCH_AFTER = 200` and
  `_LIVENESS_DEAD_WEIGHT = 0.1` are retained, but their comments now cite the
  measurement instead of calling themselves guesses; the one thing the
  synthetic target cannot calibrate (a real cold-but-live region's no-growth
  floor) is stated at both constants. Full numbers and the `--unstable 0`
  requirement (reproduced: 66/120 dead-region mutations move coverage on
  `--unstable 4`) in `docs/sweeps/synthetic_liveness_calibration_2026-08-29.md`.
  Regression coverage in `tests/test_sweep_liveness_thresholds.py`.
