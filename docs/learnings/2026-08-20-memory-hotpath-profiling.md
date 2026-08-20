# Memory Hotpath Profiling: Method Errors and Measured Results

**Date**: 2026-08-20
**Status**: Four perf patches landed; profiling methodology corrected
**Target**: `ffmpeg_read` (nosan), `--elo all`, memray

## Executive Summary

Profiled the allocation hotpath under `--elo all` and landed four
optimizations. The optimizations are the smaller half of the value here. The
larger half is that **four separate predictions were overturned by
measurement**, and two profiling runs produced numbers that looked
authoritative and were meaningless. This document records the method errors
in more detail than the wins, because the wins are in the commit messages and
the errors are not.

The single most important finding: **profiling against a target whose
dependencies are uninstrumented ranks the hotpath wrong.** Not
approximately-wrong — inverted.

## The Coverage-Density Trap

The first profile ran against `ffmpeg_read_nosan.so` linked to *system*
libav. System libav carries no `trace-pc-guard` instrumentation, so coverage
came only from the harness file: **90 edges**. The profile looked clean and
produced a confident ranking.

Rebuilt against a vendored, coverage-instrumented ffmpeg: **4,248 edges**, a
47x increase. The hotpath ranking inverted:

| site | harness-only rank | instrumented rank |
|---|---|---|
| `contextual.py` np.stack | #1 by count (5.5M allocs) | not in top 8 |
| `shm.py` `get_edge_ids` | #3 | #2 (11.5 GB) |
| `seed_picker.py:465` | absent | #1 (13.4 GB) |
| `shapley.py:95` | absent | #1 by count (1.27M) |

Two of the top sites were completely invisible in the first profile. The
reason is structural: scheduler cost scales with *operator count*, which is
fixed, while coverage-collection and corpus-analysis cost scale with *edge
count*, which the uninstrumented build had suppressed by 47x.

**Rule**: before profiling, check `Edges discovered` against the target's
plausible size. Three-digit edge counts on a real codec mean the dependencies
are uninstrumented and any ranking derived from that run is suspect. See also
`docs/learnings/2026-08-07-uninstrumented-system-libs-coverage.md`.

## Fuzzer-Level A/B Comparison Does Not Work

Three profiled runs at a fixed 1,025 execs with `-s 42`:

| run | edges | allocations | GB churned | peak MB |
|---|---|---|---|---|
| baseline | 4,514 | 2.72M | 33.4 | 340.5 |
| after shm+shapley | 4,308 | 1.55M | 37.1 | — |
| after regression fix | 5,000 | 3.41M | 70.9 | 407.3 |

The third run allocates 2x the second while containing strictly more
optimization. `-s 42` does not make the run deterministic: exec count is
fixed but the *trajectory* is not, and allocation volume tracks
edges-discovered and corpus size, not execs. A run that finds more coverage
does more work.

This invalidated an entire round of before/after numbers that were already
written up. **Do not A/B optimizations at the fuzzer level.** Extract the
function and benchmark it in isolation with fixed inputs. Every trustworthy
number in this session came from an isolated harness.

If end-to-end validation is needed, the metric has to be normalized per unit
of work discovered (allocations per edge, or per corpus entry), and even then
it needs repeats — not a single paired run.

## Churn Is Not Pressure

`memray stats` reports `total_bytes_allocated`, which is **churn**:
everything allocated over the run, including memory freed microseconds later.
It is not RSS and not peak.

`seed_picker.py:465` was the largest site in the instrumented profile at
13.4 GB churned. Peak RSS for the whole process was ~370 MB. The 13.4 GB is
transient set objects that the allocator absorbs and reuses. Replacing them
with an allocation-free counting loop measured **2x slower** (1359 vs 708
us/call) for churn reduction that would not move peak RSS at all.

For a fuzzer whose objective is execs/sec, the metrics that matter are peak
RSS and CPU. Ranking sites by churn points at the wrong ones. Churn is a
useful *signal* — it locates code doing per-exec allocation — but it is not
itself the thing to minimize.

## tracemalloc Corrupts Timing

Timing a function with `tracemalloc` running inflates allocation-heavy code
severalfold:

```
get_edge_ids, with tracemalloc active:   3612.2 us/call
get_edge_ids, timing only:               1047.1 us/call
```

3.4x error, and it scales with allocation count, so it distorts exactly the
comparison a memory optimization cares about. Measure time and allocations in
separate passes.

## Four Predictions, Four Refutations

Each of these was a change that was mechanically obvious, implemented, and
measured worse than what it replaced.

**1. `np.take(out=)` into a reused buffer.** Intended to eliminate the
allocation from a fancy-index gather. `np.take`'s wrapper allocates
regardless of `out=`:

```
fancy index      9.92 us/call   170 KB
np.take(out=)   19.38 us/call   171 KB
```

2x slower, identical allocation. Reverted. Note that ufunc `out=` (e.g.
`np.multiply(..., out=)`) *does* honor the buffer — the failure is specific
to `np.take`'s Python-level dispatch.

**2. Allocation-free set-intersection counting.** `sum(len(a & b) ...)`
allocates an intersection set per pair. Counting membership over the smaller
set instead: 1359 vs 708 us. C-level set intersection beats a Python loop
even when the loop allocates nothing.

**3. Per-seed index sampling.** Replacing an O(n^2) list rebuild with
`rng.sample(range(n), 65)` per seed measured *slower* than the O(n^2) version
below n~800 — `rng.sample` overhead per seed dominates at realistic corpus
sizes. The variant that actually wins draws the pool **once per pass**
(23x at n=2000), which is a different algorithm, not a micro-optimization.

**4. `.so` direct_lite as a 5-10x throughput lever.** Estimated from
comparing 70-105 eps (direct_lite, system ffmpeg, 90 edges) against 6-13 eps
(subprocess, instrumented, 4,248 edges). Those runs differed in *two*
variables. Measured head-to-head with coverage held constant:

```
ffmpeg_read_vendor_nosan.so  direct_lite  12 eps  83s
ffmpeg_read_nosan (exe)      subprocess   11 eps  92s
```

**1.1x.** The throughput collapse was caused by instrumentation density, not
process-spawn overhead. The `.so` is still worth having — it is the format
that enables direct_lite, in-target cmplog, and persistent mode — but not for
speed.

The common thread: all four were ranked by *mechanism* ("this allocates, so
removing the allocation is faster") rather than by measured cost. Mechanism
reasoning identified the right sites and the wrong fixes.

## Build Notes

**`--disable-x86asm` does not disable inline asm.** Linking the vendored
static archives into a `.so` failed with:

```
ld: libavcodec.a(cavsdsp.o): relocation R_X86_64_PC32 against `ff_pw_5'
    can not be used when making a shared object; recompile with -fPIC
```

`--enable-pic` and `--disable-x86asm` both applied (`CONFIG_PIC 1`,
`HAVE_X86ASM 0`), but `--disable-x86asm` only disables external nasm/yasm
files. Inline asm in the C sources still emits non-PIC relocations: 1,353 in
`cavsdsp.o`, 1,041 in `vc1dsp.o`, 247 in `h264chroma.o`. **`--disable-inline-asm`
is required for the `.so` target.** The executable target links the identical
archives without complaint, which is why this only surfaces on the `.so` path.

Other environment findings, lower confidence, may be container-specific:

- `configure` failure is not checked before `make` runs. A failed configure
  leaves `SRC_PATH` empty and `make` reports
  `No rule to make target '/tests/Makefile'`, which points nowhere near the
  real cause (in this case a missing `libclang_rt.ubsan_standalone-x86_64.a`).
  An exit-status check would have saved substantial time.
- `python3 -m fuzzer_tool.cli.commands` imports and exits 0 silently — no
  `__main__` guard. This silently produced an empty profile that looked like
  a successful run. Use the `fuzzer-tool` console script.
- The build comment in `targets/ffmpeg_read.c` references the removed
  `cmplog_shim.c` and omits `-D__AFL_CMPLOG=1`; copy-pasting it yields
  `undefined symbol: __sanitizer_cov_trace_const_cmp1`.
- `tools/profile_hotpath.py:18` hardcodes an absolute `os.chdir`;
  `build_targets.sh` has the same pattern for `TAILSLAYER`.
- The vendored tree resolved to `/vendor/ffmpeg`, outside the repo, while
  `.gitignore:67` ignores `vendor/`. A build artifact landing outside both
  the repo and the ignore rule is worth confirming as intentional.

## What Landed

| commit | measured |
|---|---|
| `perf(contextual)` | 8.61M -> 2.36M allocs; `select_op` 1.97x |
| `perf(shm)` | 1.59x, 26% less peak |
| `perf(shapley)` | 1.27M allocations eliminated |
| `perf(seed_picker)` | 2.8x at n=200, 23.1x at n=2000 |

`perf(seed_picker)` carries a **semantic change**: lineage-diversity peers are
now shared across seeds within a pass rather than drawn per seed. Each seed
still gets 64 peers and the pool redraws each pass, but within a pass the
estimates are correlated. If lineage diversity is load-bearing for
scheduling, this wants an A/B on edge discovery over a long run — which, per
the section above, means many repeats, not one paired comparison.

`perf(shm)` also documents that the remaining cost is not the numpy work:
mask+take is 225 us of 659, `set(tolist())` is 383 us. That 383 us is not
recoverable by changing the numpy — it is inherent to materializing 4-5k
Python ints per exec.

## Open

- **Bitmaps for edge sets.** Dense-index interning validated: 45,258 edges
  across 40 growth steps with no index shift, mask algebra matches set
  algebra across a growth boundary. Measured 21x on `len(a & b)`, 42x on
  `a - b`, and 125.7 MB -> 3.1 MB for a 500-seed corpus. `len()` regresses
  28x, so cardinality wants caching. Per-exec build cost is a **wash**
  (181 vs 179 us) — the entire case is downstream algebra and stored size.
  Given this session's record, treat those microbenchmark numbers as an upper
  bound on end-to-end effect. Wants a flag and a differential harness
  asserting both paths agree, not a big-bang rewrite.
- `edge_tracker.py:440` (`compute_signature`, 1.5 GB) and `shm.py:273`
  (`get_edge_counts`, 910 MB) unexamined; plausibly the same
  column-vs-struct pattern as `perf(shm)`.
- `test_shim_disambiguates_shared_function_by_caller` failed once in a full
  run, then passed 3x in isolation and on a repeat full run. Likely flaky
  rather than related to these changes, but it involves compilation and
  subprocesses and may be genuinely timing-sensitive.
