# daedalus/fuzzer — edge coverage analysis

Static analysis of `src/fuzzer_tool` (120k LOC, 351 py files) against the question:
what gets more edges per wall-clock second?

**Scope:** open items only. Fixed findings have been removed — see the summary table
below for what they were and which commit closed each.

**Caveat up front:** this document was originally written without clang in the
container, so no sancov target was ever built or fuzzed — everything was read from
source, plus simulation and microbenchmarks run in isolation. **No coverage delta has
been measured, for the open items or the closed ones.** The probe-cost table in §2 is
simulated; the memset table is measured but on one machine. A/B with
`tools/bench_paired.py` before trusting any of it.

§1 is the exception and the cautionary tale: it is the one item that was measured, and
measuring it showed the prescribed fix was worth 0.99x. Throughput numbers there are
real, but come from `test_target` and a synthetic ASAN target in a sandbox, not from a
vendored target.

Ordered by expected impact.

---

## Already fixed — not covered below

These are closed and their sections have been removed from this document. The full
reasoning lives in the commit messages, which is where it stays useful.

| commit | what it closed |
|--------|----------------|
| `fix(shm): disable ASLR so context-sensitive edge_ids survive across execs` | Context-sensitive `edge_id`s derive from runtime return addresses, but `_seen_edge_ids` outlives the target process. Under ASLR every exec of a CTX build reported a fresh edge set. |
| `fix(build): build vendored libs with frame pointers for the CTX walk` | Vendored zlib/libpng/libjpeg/ffmpeg/lz4/secp256k1 were built without `-fno-omit-frame-pointer` while being linked into CTX-enabled `_nosan` targets, so the context walk silently read the wrong caller. |
| `fix(coverage): size the map for edges, load factor and context width` | Map sized from block count with no edge/block ratio, no load-factor headroom, and no knowledge of context width; plus the shim now counts dropped edges, so saturation is detectable instead of self-masking. |
| `fix(coverage): keep the SHM resize, drop the state it needlessly destroyed` | Resize wiped every accumulated statistic on a premise true for AFL's bitmap but false here, so the next execution re-reported all known edges as new — defeating the stall detector that had triggered it. Also stopped copying the old table into the new segment at the old modulus. |
| `fix(coverage): prevent valid edge_id 0 from being treated as an empty slot` | `edge_id == 0` was indistinguishable from an empty slot in the open-addressing table, so valid edges that XORed to zero were silently dropped and the slot reclaimed by later collisions. Forced to 1 with `edge_id |= 1`. |
| `fix(scheduler): wire favored into FAST/COE and add cull_queue/top_rated` | `favored` was threaded through `SeedScorer` but never computed; FAST/COE ran permanently in unfavored mode and `top_rated`/`cull_queue` minimal-set-cover did not exist. Added `_cull_queue()` to `Fuzzer`, periodic favored recomputation, and passed `favored` at the score call site. |
| `fix(coverage): make hit counts part of the novelty decision on the SHM path` | `_check_new_coverage` decided interestingness by set membership only, so loop-count-guarded branches were invisible. Added disjoint bucket-bit ladders (`core/count_class.py`) keyed by edge_id in a dense virgin map on the hot SHM path; counts clamped at 255, fast path unchanged. |
| `fix(shm): stop the no-change fast path from reporting a phantom edge loss` | The fast path returned `(False, set())` on every unchanged exec; callers caching that return value read it as "zero edges fired," corrupting the next diff into reporting the whole edge set as new. Added `ShmCoverage._last_ids`, returned by the fast path instead of an empty set. |
| `feat(shim): add a real AFL-style forkserver` + `fix(forkserver): drive the shim forkserver, drop the bitmap round-trip` + `feat(fuzzer): enable the forkserver on the default execution path` | §1. See that section — the fix this document prescribed (delete the bitmap-file round-trip) was necessary but worth 0.99x on its own, because `fuzz_loader.c` did fork+**exec** per input and still paid the full ELF load + linker + libc + ASAN init every execution. The server had to move into the target. 2.77x end to end through the CLI. |
| `feat(fuzzer): add SkipDet deterministic stage` + `fix(fuzzer): guard deterministic stage coverage path and meta access` + follow-up merge | First landed as `Fuzzer._run_deterministic_stage()`: a standalone blocking loop, always on, that dispatched `_op_bit_flip` once per byte (one random bit, not a true 1/1 walk) and never called `save_to_corpus` — any interesting mutant it found was discarded. Merged into `OperatorEngine.maybe_deterministic_mutation()`: a real bitflip-1/1 + byte-flip-8/8 + arithmetic + interesting-value generator, drained from inside `mutate()` so discoveries flow through the same `fuzz_one()` → `save_to_corpus` path as every other mutation. Also fixed `trace_mini` indexing (`edge_id // 8`, out-of-bounds-dropped almost every real edge_id) to fold by `edge_id % map_size` instead. Opt-out via `--no-deterministic`. |

Two open items **depend** on the sizing commit (the third entry above) and are carried
forward in §2: bounding the probe window (now measurable, because drops are counted)
and making the per-exec reset O(1) (the precondition for maps larger than 262144).

Tier 2 (hit-count bucketing, favored wiring) and Tier 3's two open items (deterministic
stages, empty-edge-set fast path) are now all complete, and §1 is closed. §2's
generation-tagged reset is the largest remaining item.

---

## Tier 1 — throughput (edges/sec is the metric, not edges/exec)

### 1. The forkserver is commented out on the default execution path

**Status: FIXED**, but *not* by the fix this section originally prescribed. Read the
correction below before reusing any of the reasoning here.

The original diagnosis was that the child inherits `__AFL_SHM_ID` and `afl_shim.c`'s
constructor already attaches, so the bitmap never needed to travel: delete the
bitmap-file path in `fuzz_loader.c` and the `ctypes.memmove` in `runner.py`, and the
forkserver works. That much is true and was necessary — but it bought **nothing**.
Measured against `posix_spawn` with only that deletion applied:

| target | spawn | after the prescribed fix | speedup |
|--------|-------|--------------------------|---------|
| `test_target` | 612 exec/s | 648 exec/s | 1.06x |
| ASAN + heavy static init | 93.8 exec/s | 92.6 exec/s | **0.99x** |

**`fuzz_loader.c` was never a forkserver.** `run_executable()` did fork **+ exec**
per input, so every execution still paid the full ELF load, dynamic linker, libc init
and ASAN init — the exact cost this section set out to remove. The name was the only
thing forkserver-like about it. The 2–10x figure quoted above assumes AFL's design,
where the *target* hosts the server and children fork from a process already sitting
past its own initialisation; nothing in this tree did that. `afl_shim.c` had no
forkserver at all — no `__AFL_INIT`, no FORKSRV handshake. (The `__AFL_INIT()` /
`__AFL_LOOP()` calls in `png_read.c` and `grep_read.c` come from real AFL++ builds,
not from this shim.)

The actual fix moves the server into the target: `__afl_start_forkserver()` in the
shim constructor, speaking AFL's protocol on AFL's fd numbers (198/199), with
`fuzz_loader.c` driving it and falling back to fork+exec for targets built against an
older shim.

| target | spawn | true forkserver | speedup |
|--------|-------|-----------------|---------|
| `test_target` | 623 exec/s | 3281 exec/s | **5.27x** |
| ASAN + heavy static init | 90.6 exec/s | 125 exec/s | **1.38x** |
| full CLI, end to end | 484 eps | 1341 eps | **2.77x** |

The ASAN target gains least because forking a process carrying ASAN's shadow mapping
is itself expensive. Worth knowing before extrapolating to the vendored targets —
this is the one number here most likely to mislead.

**Semantics change, deliberately.** Coverage recorded during pre-fork initialisation
is no longer re-recorded per execution. Verified that the forkserver edge set is a
strict *subset* of the spawn edge set for every input, and that the dropped
difference is identical across all inputs — constant init edges, which carry no
signal. Input discrimination is unchanged. This matches AFL, but it does mean edge
sets are not comparable across `--no-forkserver`.

Four defects were fixed on the way, each of which would have bitten on the first run:
dropped child stderr (ASAN exits 1, so `SanitizerReport.parse(stderr)` is the *only*
crash signal there is); a zero-length `RUN` that never replied and hung the caller
until its join timeout; an oversized `RUN` that desynced the pipe; and timeouts
reported as `-SIGKILL`, which is in `SIGNAL_CRASH_CODES` and would have filed every
slow input as a crash. The forkserver parent also had to stop recording its own
control flow — see the learnings note.

### 2. `__afl_map_edge` is O(map_size) per edge hit

**Status: PARTLY OPEN.** Sizing and saturation detection are fixed (commit 3). The
probe cost on the hot path is not.

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
size is a direct per-exec tax, which is why commit 3 could only raise the cap to 262144:

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
sign of commit 3 on edges/second is therefore unknown, and could be negative on a target
that was not actually saturating.

### 4. No hit-count bucketing on the SHM path

**Status: FIXED.** `_check_new_coverage` (`adapters/shm.py`) now maintains a dense
virgin map keyed by `edge_id` with disjoint bucket-bit ladders from
`core/count_class.py`. Counts are clamped at 255; the fast path is unchanged.
Loop-count-guarded branches are visible to the scheduler without affecting the
set-membership "new edge" statistic.

### 5. `favored` is threaded through the whole scheduler and never computed

**Status: FIXED.** `_cull_queue()` now computes the favored set from `EdgeTracker.seed_edges`
and `seed_meta`, using `exec_us * input_size` as seed cost and greedy rare-edge-priority
cover to select favorites. It runs periodically in the main fuzz loop, and the score call
site passes `favored=(seed_key in self._favored)` into `SeedScorer.score()`.

### 6. `core/skipdet.py` is entirely dead, because there are no deterministic stages

**Status: FIXED.** `OperatorEngine.maybe_deterministic_mutation()` (`operators.py`)
queues a per-seed bitflip 1/1 + byte flip 8/8 + arithmetic + interesting-value schedule
(`_deterministic_mutation_stream`), gated by `SkipDetector.should_det_fuzz` via a
synthesized positional bitmap (`core.skipdet.trace_mini_from_edges`, folding
`edge_id % map_size` since this fuzzer's edge_ids are sparse hashes with no positional
bitmap to hand `should_det_fuzz` directly). Drains one mutation per `mutate()` call
rather than a parallel execution loop, so a deterministic-stage discovery goes through
`fuzz_one()`'s normal `save_to_corpus` path like any other mutation.

A first version of this landed directly in `Fuzzer._run_deterministic_stage()`: always
on, no opt-out, dispatching `_op_bit_flip` once per byte instead of a true 1/1 walk
(that handler picks one random bit, so it covered ⅛ of what a real bitflip pass
covers), indexing its trace bitmap by `edge_id // 8` and dropping anything past the
first `map_size` bytes (nearly every real edge_id, since they're hashes, not small
positional integers) — and running its own blocking exec loop that updated
edge-tracking bookkeeping but never called `save_to_corpus`, so any mutant it found
interesting was immediately discarded. All four fixed in the merge above. On by
default, matching the shipped behavior; `--no-deterministic` opts out, since a full
pass still costs `8*len(seed)` execs per favored seed and hasn't been benchmarked
against a real target.

### 8. The fast path returns an empty edge set that callers treat as real

**Status: FIXED.** `ShmCoverage` now tracks `_last_ids`, the edge set from the last
slow-path scan, and the fast path returns that instead of a fresh `set()` when nothing
changed. Callers diffing consecutive returns (`Fuzzer._prev_edge_set`,
`_current_edges_cache`) now see "same as before" instead of a phantom total edge loss.

---

## Tier 3 — algorithms worth adding

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
`_detect_cmplog()` (`fuzzer.py:278`) which reliably identifies instrumented targets,
`--cmplog` should default on whenever detection succeeds. Magic-value and checksum
branches are where the edge count plateaus on real formats.

**The havoc short-circuit.** `mutate()` at `services/operators.py:2932`:

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

**Dead classes — wire or delete.** `MonteCarloScheduler` and `EpsilonGreedyScheduler`
are now instantiated (`fuzzer.py:1136`, `fuzzer.py:1185`) and `SanitizerReport` is now
built via `.parse()` on ASAN/UBSAN replay (`fuzzer.py:3399`, `fuzzer.py:3414`) — all
three drop off this list since the last pass. Still never instantiated anywhere in
`src/`: `CoverageHomogeneityDetector` (`core/critical_slowing.py` — a stall predictor,
directly on topic), `KalmanFilter`, `CondStmt`, `MutatorBase`, `ConstraintSet` (wfc),
plus `adapters/track_parser.py` (referenced only in a docstring, never imported).

---

## Suggested order

1. ~~**§1 — forkserver.**~~ Done — see §1, and note that what closed it is not what
   this document predicted would close it.
2. **§2 — generation-tagged reset.** Now the largest open item. Unblocks map sizes
   above 262144 by making the per-exec clear O(1). Do this before concluding anything
   about §2's probe cost, and note it turns the table-copy behaviour in `resize()`
   from cosmetic to load-bearing. `resize()` also has a live consumer now
   (`ForkserverRunner.update_shm_after_resize()`), which respawns the loader so its
   children inherit the new segment id.
3. **§2 — bounded probe window.** Cheap to evaluate now that drops are counted.

**The already-fixed commits still need an A/B before they become defaults.** Nothing
here has been measured against a real target. In particular the sizing commit enlarges
maps, which costs more per-execution memset (§2), so its net effect on edges/second
could be negative on a target that was not saturating. This now includes §6's
deterministic stage: it's on by default (`--no-deterministic` to opt out) and hasn't
been benchmarked against a real target either.

Items 1, 2 and 3 are independently testable with `tools/bench_paired.py` against
a fixed seed and a fixed exec budget.

## Loose thread

Twice in roughly fifty runs, the first `ShmCoverage` constructed in a process read back
an empty edge table after a child that exited 0 — header `edge_count` not captured at
the time, so I cannot say whether the child failed to attach or the parent raced the
read. Never reproduced across 20 consecutive full-file test runs afterwards, and not
observed on the pre-commit-3 tree either, though the sample there is smaller.

Recorded rather than dropped: if the drop-counter tests in `tests/test_ctx_and_map_size.py`
go intermittently red in CI, this is the thread to pull, and the thing to capture is the
SHM header (`read_edge_count()`, `read_diag()`) alongside the child's exit status.
