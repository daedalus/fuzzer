# daedalus/fuzzer — edge coverage analysis

Static analysis of `src/fuzzer_tool` (116k LOC, 341 py files) against the question:
what gets more edges per wall-clock second?

**Scope:** open items only. Fixed findings have been removed — see the summary table
below for what they were and which commit closed each.

**Caveat up front:** no clang in this container, so no sancov target was ever built or
fuzzed. Everything here is read from source, plus simulation and microbenchmarks run in
isolation. **No coverage delta has been measured, for the open items or the closed
ones.** The probe-cost table in §2 is simulated; the memset table is measured but on
one machine. A/B with `tools/bench_paired.py` before trusting any of it.

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

Two open items **depend** on the sizing commit (the third entry above) and are carried
forward in §2: bounding the probe window (now measurable, because drops are counted)
and making the per-exec reset O(1) (the precondition for maps larger than 262144).

Tier 2 (hit-count bucketing, favored wiring) is now complete. Tier 3 (deterministic
stages, empty-edge-set fast path) is untouched. §1 remains the largest single item.

---

## Tier 1 — throughput (edges/sec is the metric, not edges/exec)

### 1. The forkserver is commented out on the default execution path

**Status: OPEN.** Still the largest single item in this document.

`services/fuzzer.py:1777-1785`:

```python
# Forkserver: use C fuzz_loader for default execution path when available.
# Currently disabled: fuzz_loader reads bitmap from file while target
# writes to SHM — these are disconnected.
# if not self._inprocess_runner and not self._persistent_runner and not self.ptrace_cov:
#     from fuzzer_tool.adapters.forkserver import ForkserverRunner
```

So unless the user passes `--inprocess` or `--persistent`, **every single execution
is a `posix_spawn` + full ELF load + dynamic linker + libc init + constructor run**
(`services/runner.py:210-250`, `run_target_fast`). This is the classic 2–10× (often
more, on targets with heavy static init) throughput gap versus AFL.

The infrastructure exists: `adapters/forkserver.py`, `adapters/fuzz_loader.c`,
`ForkserverRunner.run_one`, and the consumer at `runner.py:205-208`.

The stated blocker is solvable and the fix removes code rather than adding it: the
child inherits `__AFL_SHM_ID` through the environment, and `afl_shim.c`'s
constructor already attaches to it on its own. So the forkserver child does not need
to hand a bitmap back at all — drop the bitmap-file path in `fuzz_loader.c`, and drop
the `ctypes.memmove(shm._ptr, bitmap, ...)` at `runner.py:207`. The parent reads SHM
directly, exactly as it does in `direct_lite` mode.

This is the highest-leverage change in the tree.

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

**Status: OPEN.**

`SkipDetector`, `build_skip_eff_map`, `should_skip_deterministic`, and the block-flip
inference stage — all unreferenced outside the file itself. The only surviving trace
is `_op_skipdet_probe`, a single one-shot havoc operator.

It is dead for a structural reason. `OperatorEngine.mutate()`
(`services/operators.py:2608`) is one-shot: draw `n_mutations` operators from a bandit,
apply each once at a bandit-chosen position. All the AFL deterministic operators exist
(`_op_bit_flip`, `_op_byte_flip`, `_op_arithmetic`, `_op_interesting_8/16/32`,
`_op_auto_extras`) but **nothing ever walks them systematically across a seed**.

That is the biggest missing algorithm, not just missing wiring. AFL's bitflip 1/1 pass
is where the effector map comes from, and the effector map is what makes the arith and
interesting-value passes cheap enough to be worth running. Without stages you get
neither. A per-seed deterministic pass, gated by `SkipDetector.should_skip_deterministic`
so it only runs on favored, not-yet-determinized seeds, plugs both holes at once — and
it composes with §5, since SkipDet's gate is defined in terms of `seed_favored`.

### 8. The fast path returns an empty edge set that callers treat as real

**Status: OPEN.**

`_check_new_coverage` early-returns `(False, set())` when `edge_count` and `path_hash`
are unchanged. `fuzzer.py:2626` assigns that to `self._current_edges_cache`, and the
format learner's `elif` branch (`fuzzer.py:2692`) then sets
`self._prev_edge_set = set()`. The next `new_edges = current - prev` diff therefore
reports the entire edge set as newly discovered.

Fix: return the previous edge set, or use `None` for "not scanned" and have callers
distinguish it from "scanned, empty."

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
`_detect_cmplog()` (`fuzzer.py:274`) which reliably identifies instrumented targets,
`--cmplog` should default on whenever detection succeeds. Magic-value and checksum
branches are where the edge count plateaus on real formats.

**The havoc short-circuit.** `mutate()` at `services/operators.py:2712`:

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

**Dead classes — wire or delete.** Never instantiated anywhere in `src/`:
`CoverageHomogeneityDetector` (`core/critical_slowing.py` — a stall predictor, directly
on topic), `KalmanFilter`, `CondStmt`, `MutatorBase`, `ConstraintSet` (wfc),
`SanitizerReport` (imported at `fuzzer.py:35`, never constructed), plus
`adapters/track_parser.py`, `schedulers/epsilon_greedy.py`, `schedulers/monte_carlo.py`.

---

## Suggested order

1. **§1 — forkserver.** Largest throughput win, largest diff.
2. **§2 — generation-tagged reset.** Unblocks map sizes above 262144 by making the
   per-exec clear O(1). Do this before concluding anything about §2's probe cost, and
   note it turns the table-copy behaviour in `resize()` from cosmetic to load-bearing.
3. **§2 — bounded probe window.** Cheap to evaluate now that drops are counted.
4. **§6 — deterministic stages + SkipDet.** Real design work.
5. **§8 — the empty-edge-set fast path.** Small, affects only the format learner.

**The already-fixed commits still need an A/B before they become defaults.** Nothing
here has been measured against a real target. In particular the sizing commit enlarges
maps, which costs more per-execution memset (§2), so its net effect on edges/second
could be negative on a target that was not saturating.

Items 1, 2, 3, 4 and 5 are independently testable with `tools/bench_paired.py` against
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
