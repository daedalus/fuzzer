# daedalus/fuzzer — edge coverage analysis

Static analysis of `src/fuzzer_tool` (120k LOC, 351 py files) against the question:
what gets more edges per wall-clock second?

**Scope:** pending items only. Completed work is documented in commit messages
and the historical record in `docs/handover/`.

**Caveat up front:** this document was originally written without clang in the
container, so no sancov target was ever built or fuzzed — everything was read from
source, plus simulation and microbenchmarks run in isolation. **No coverage delta has
been measured, for the open items or the closed ones.** The probe-cost table in §2 is
simulated; the memset table is measured but on one machine. A/B with
`tools/bench_paired.py` before trusting any of it.

`apt-get install clang` now succeeds, so the instrumented matrix can be built and the
guard census re-run on demand; §2 records two such runs, 2026-08-14 and 2026-08-16.
What still has not happened is a *fuzzing* run: no edges/second figure in this document
comes from an A/B against a real target.

Ordered by expected impact.

---

## §2 — `__afl_map_edge` is O(map_size) per edge hit

**Status: PARTLY OPEN.** Sizing and saturation detection are fixed. The probe cost
on the hot path and the per-exec reset cost are not.

`adapters/afl_shim.c` linear-probes an open-addressing table on **every edge
execution**, not just every unique edge. A loop body that runs 10k times pays the probe
cost 10k times. Simulated average probes per insertion (uniform random ids):

| load | avg probes |
|------|-----------|
| 0.49 | 1.5 |
| 0.79 | 2.8 |
| 0.95 | 12.1 |
| 1.00 | 57.2 |
| saturated | `map_size` |

Two ways out, neither taken yet:

- **Bound the probe window** to 8–16 slots, then give up. Converts a `map_size`-iteration
  worst case into a constant, trading a drop rate for a bounded cost. This is now cheap
  to evaluate rather than speculative: the shim counts dropped edges into the SHM header
  (offset 4, bits 8–31), so the trade is directly observable —
  `ShmCoverage.read_dropped_edges()`.
- **Stop open-addressing on the hot path.** AFL's `mem[(prev^cur) & mask]++` is one load,
  one add, one store. You keep exact edge IDs today at the cost of a hash probe on every
  branch; you could get both by keeping the classic bitmap in the hot path and recovering
  identity lazily (the guard→address map is already parsed in `elf.py`).

**Prerequisite for either: make the per-execution reset O(1).**
`ShmCoverage.reset_edge_map()` memsets the whole table before every execution, so table
size is a direct per-exec tax, which is why the map cap was set to 262144:

| entries | table | memset per exec |
|---------|-------|-----------------|
| 8,192 | 0.1 MiB | 3.8 µs |
| 131,072 | 1.0 MiB | 84.6 µs |
| 262,144 | 2.0 MiB | 86.9 µs |
| 1,048,576 | 8.0 MiB | 352.8 µs |
| 4,194,304 | 32.0 MiB | 4740.0 µs |

At a 100 µs execution the 1 MiB clear already rivals the run. Generation-tagged entries
fix this: bump a byte in the header per exec, treat any entry whose tag is stale as
empty, memset for real only on the 256-cycle wrap. Reset stops being a function of map
size, and the cap can follow instrumentation size instead of clear bandwidth. It is a
hot-path change to `__afl_map_edge` plus every numpy reader on the Python side, so it
wants its own patch and its own benchmark.

Note the two costs pull in opposite directions — a bigger map makes probes cheaper and
clears more expensive — and neither has been measured against a real target. The net
sign on edges/second is therefore unknown, and could be negative on a target that was
not actually saturating.

**Measured 2026-08-14, after the forkserver landed.** The memset table above reproduces
on this machine (2.6 µs @ 8K, 82.2 µs @ 262K, 350.6 µs @ 1M, 1499.5 µs @ 4M). What the
forkserver changes is what those numbers *mean*, because the exec they are a fraction of
got several times cheaper:

| map | share of a 305 µs exec (light target) | share of an 8000 µs exec (ASAN target) |
|-----|---------------------------------------|----------------------------------------|
| 8,192 | 0.9% | 0.0% |
| 262,144 | 27.0% | 1.0% |
| 1,048,576 | 115.0% | 4.4% |

So the reset is irrelevant at the default map size and only bites in one specific
combination: a target fast enough for the clear to dominate *and* edgy enough to need a
map above 262144. A heavy target needs the big map but also has the long exec to absorb
it. That is narrower than "at a 100 µs execution the 1 MiB clear already rivals the run"
suggests, and it should inform how much risk this change is worth.

**A cheaper fix was tried and does not work.** Resetting only the occupied slots from
the Python side (no shim change, no layout change) is **8x slower than the memset**, at
every occupancy from 8 to 10,000 active entries: finding the occupied slots costs a
`np.flatnonzero` over the whole table, which is O(map_size) exactly like the memset but
against a hand-tuned `memset`. Generation tagging really is the only route to O(1).

**One hazard for whoever implements it.** The generation counter needs somewhere to
live, and the front header is full (24 bytes, all four fields in use; `diag`'s 32 bits
are split between ctx width and the drop counter). Growing `SHM_HEADER_SIZE` means the
edge table moves, and `SHM_METADATA_SIZE` in `adapters/shm.py` must move with it — a
target built against the *old* shim then writes its table at the old offset while the
fuzzer reads at the new one. That failure is silent and looks like a coverage
regression, not a version mismatch. Either carry a version marker in `diag` and refuse
to attach on a mismatch, or find the bits inside the existing 24 (the top 8 bits of
`count` are the obvious candidate: counts are clamped to 255 for bucketing long before
they approach 2^24).

**Measured 2026-08-14 on real instrumented targets — and the premise does not hold
here.** `tools/build_targets.sh --clang-scov` produces a genuine trace-pc-guard matrix.
Exact block counts, and the map each target asks for:

| target | guards | map | edges hit | dropped |
|--------|-------:|----:|----------:|--------:|
| gzip_read | 527 | 8,192 | 13 | 0 |
| tracecmp_target_tcg | 349 | 8,192 | 11 | 0 |
| png_read | 311 | 8,192 | 43 | 0 |
| zlib_read | 251 | 8,192 | 21 | 0 |
| proto_target | 94 | 8,192 | 4 | 0 |
| test_target | 91 | 8,192 | 4 | 0 |

All 16 instrumented targets size to the **floor**, `MAP_SIZE_DEFAULT` = 8192. The cap
that the O(1) reset exists to lift is 262144 — roughly 500x above anything in the tree.
Dropped edges are zero everywhere and the load factor is about 13%, so the probe window
averages ~1.07 probes and bounding it would buy nothing either. **Both halves of this
section fix costs that no target in this matrix pays.** At 8192 entries the reset is
4.1 µs.

Two caveats keep this section open rather than closed:

- The six library-backed targets (ffmpeg, fgrep, secp256k1, lz4, jpeg, unrar) did not
  build — `vendor/ffmpeg` is an unbuilt source tree here — and those are precisely the
  ones whose guard counts could reach the cap. The census covers the small end only.
- Guard counts are per-binary, not per-`edge_id`. A CTX build multiplies distinct ids by
  call-graph fan-in; every target above is `ctx=0`.

Re-run the census before doing any of this work. `parse_sancov_guard_count()` makes it
a one-liner per target.

**Census re-run 2026-08-16 — same answer, and the caveats above are unchanged.** Full
`tools/build_targets.sh --clang-scov`, then `parse_sancov_guard_count()` +
`estimate_map_size_detail()` over everything in `targets/`. 16 binaries carry a
`__sancov_guards` section; every one of them sizes to the floor:

| target | guards | map | ctx | source |
|--------|-------:|----:|----:|--------|
| gzip_read | 532 | 8,192 | 0 | sancov_guards |
| gzip_read_nosan | 500 | 8,192 | 0 | sancov_guards |
| tracecmp_target_tcg | 354 | 8,192 | 0 | sancov_guards |
| png_read | 316 | 8,192 | 0 | sancov_guards |
| png_read_nosan | 297 | 8,192 | 0 | sancov_guards |
| zlib_read | 256 | 8,192 | 0 | sancov_guards |
| zlib_read_nosan | 243 | 8,192 | 0 | sancov_guards |
| cmplog_exercise_tcg | 155 | 8,192 | 0 | sancov_guards |
| proto_target | 99 | 8,192 | 0 | sancov_guards |
| test_target | 96 | 8,192 | 0 | sancov_guards |
| asan_target | 90 | 8,192 | 0 | sancov_guards |

(`.so` and `_nosan` variants elided where they duplicate a row.) Every size is `exact`
— `source == "sancov_guards"` for all 16, so `parse_sancov_guard_count()` is doing its
job and nothing falls back to branch-density estimation any more. Max guard count in
the tree is **532**, against a cap of 262,144. `reset_edge_map()` at the default map
re-measures at **2.5 µs/exec** here (2.6 µs on 08-14).

Two things worth adding to the caveat list rather than the finding:

- The six library-backed targets are still absent — `ffmpeg_read`, `fgrep_read`,
  `jpeg_read`, `lz4_read`, `secp256k1_read` and `unrar_read` all fail to build with no
  `vendor/` tree present. The census still covers the small end only, and those are
  still the only candidates for reaching the cap.
- `grep_read` **builds but is not an instrumented target at all.** It has no
  `__sancov_guards` section, and `targets/grep_read.c:86` `execlp`s the *system* `grep`
  binary — so the work being fuzzed happens in an uninstrumented process and the
  harness reports no edges from it regardless of how it was compiled. It contributes
  nothing to this census and would contribute nothing to a coverage A/B either. Worth
  knowing before anyone picks it as a "real target" to benchmark against.

So §2 stays open on the same terms: the costs it proposes to fix are ones no target in
this tree currently pays, and that has now been confirmed twice, two days apart, on a
freshly built matrix.

---

## Suggested but not implemented

**Unstable-edge calibration.** There is no AFL-style per-seed calibration (run a new
seed N times, mask edges that don't reproduce). `_run_calibration` is a bootstrap
warm-up loop, not this. Without it, nondeterministic edges — ASLR-dependent, time-
dependent, uninitialized-memory-dependent — read as an endless supply of new coverage
and permanently absorb energy. You get this nearly free from machinery you already
have: run each new seed 3× and compare `read_path_hash()`. Divergence means unstable;
fall back to per-edge set-diff to find which ones.

**Path hash as a second coverage dimension.** The shim maintains an order- and
multiplicity-sensitive rolling path hash and `read_path_hash()` exposes it, but it is
used only as a cheap change detector in the fast path. Honggfuzz-style unique-path
counting is one option; the instability detector above is the better use.

**cmplog defaults off.** Given the i2s/Redqueen work already in the tree and
`_detect_cmplog()` (`fuzzer.py:377`) which reliably identifies instrumented targets,
`--cmplog` should default on whenever detection succeeds. Magic-value and checksum
branches are where the edge count plateaus on real formats.

**The havoc short-circuit.** `mutate()` at `services/operators.py:2827`:

```python
if op == "havoc":
    ...
    return result
```

Drawing `havoc` discards the remaining `n_mutations - 1` iterations. Since
`n_mutations` is scaled by `_last_perf_score`, this means **the entire seed energy
multiplier is a no-op whenever havoc is drawn early** — and havoc is the most-drawn
operator. Either make havoc terminal by design and scale its internal stack depth by
`perf_score` instead, or don't return early.

---

## Dead classes — wire or delete

`MonteCarloScheduler` and `EpsilonGreedyScheduler` are now instantiated
(`fuzzer.py:1327`, `fuzzer.py:1376`) and `SanitizerReport` is now built via `.parse()`
on ASAN/UBSAN replay (`fuzzer.py:3772`, `fuzzer.py:3787`) — all three drop off this
list. Still never instantiated anywhere in `src/`:
`CoverageHomogeneityDetector` (`core/critical_slowing.py` — a stall predictor,
directly on topic), plus `adapters/track_parser.py` (not imported anywhere in `src/`,
only in tests).

---

## Open loose threads

**Intermittent `shmat()` failure.** Root cause still unknown. The stale-view hypothesis
(`ShmCoverage.cleanup()` leaving `from_address` views bound to a detached mapping) is
fixed but not established as the cause; zero hits in full suite since. Reproduction
attempts: 0 failures in ~40 runs, but the pre-fix rate was roughly 1 in 25 mixed-file
runs, so that is not yet conclusive. Still open: *why* `shmat()` intermittently fails.
The next occurrence will say, which is more than any of the four sightings could. Note
the segment is created and read successfully by the parent in the same test, and
`ipcs -m` shows no leak, so segment exhaustion (limit 4096) is not the obvious
explanation.

Twice in roughly fifty runs, the first `ShmCoverage` constructed in a process read back
an empty edge table after a child that exited 0 — header `edge_count` not captured at
the time, so I cannot say whether the child failed to attach or the parent raced the
read. Never reproduced across 20 consecutive full-file test runs afterwards, and not
observed on the pre-commit-3 tree either, though the sample there is smaller.

Recorded rather than dropped: if the drop-counter tests in `tests/test_ctx_and_map_size.py`
go intermittently red in CI, this is the thread to pull, and the thing to capture is the
SHM header (`read_edge_count()`, `read_diag()`) alongside the child's exit status.

**Not related to the intermittent full-suite `Segmentation fault`.** That one was chased
on the assumption it was SHM, and it is not: it is `Z3_del_context` running during
interpreter finalization, after every test has already passed. Diagnosis, the four
falsified hypotheses, and the reproduction harness are in
`docs/handover/suite_segfault_z3_finalization_2026-08-16.md`. Keep the two apart — a
crash at shutdown and an empty edge table mid-run have no evidence linking them.

---

## Suggested order

1. **Build a target that actually needs the map.** Two censuses two days apart put every
   instrumented binary in this tree at 532 guards or fewer, sizing to the 8,192 floor —
   ~500x under the cap that items 2 and 3 exist to lift. Until `vendor/` builds, or a
   CTX target multiplies ids by call-graph fan-in, those two items optimise a cost
   nothing pays and cannot be validated even if implemented. This is the blocker.
2. **§2 — generation-tagged reset.** The largest open *code* item. Unblocks map sizes
   above 262144 by making the per-exec clear O(1). Do this before concluding anything
   about §2's probe cost, and note it turns the table-copy behaviour in `resize()`
   from cosmetic to load-bearing. `resize()` also has a live consumer now
   (`ForkserverRunner.update_shm_after_resize()`), which respawns the loader so its
   children inherit the new segment id.
3. **§2 — bounded probe window.** Cheap to evaluate now that drops are counted — but
   drops are zero everywhere in the current matrix and the load factor is ~13%, so
   there is nothing to observe until item 1 lands.

Items 2 and 3 are independently testable with `tools/bench_paired.py` against a fixed
seed and a fixed exec budget — but only once item 1 supplies a target whose map exceeds
the floor. Against the current matrix both measure noise.
