# Handover — Classical Job-Scheduling Algorithms in the Fuzzer

**Date:** 2026-09-02
**Base:** `3480de6` (`docs(skill): crash-analysis now points at .json sidecars`)
**Status: PLAN ONLY. NOTHING IMPLEMENTED.** Every measurement below was taken
on this tree; every proposal below is unwritten code.

Sources surveyed:

- Lawler's algorithm — `1|prec|f_max`, exact, O(n²)
- Earliest deadline first (EDF) — optimal on a preemptive uniprocessor
- Dynamic priority scheduling / least slack time (LST)
- Multifit — `P||C_max` via first-fit-decreasing + binary search on capacity
- Modified due-date (MDD) heuristic — Baker & Bertrand 1982, SMTWTP
- Optimal job scheduling — the Graham/Lawler/Lenstra/Rinnooy Kan `α|β|γ` notation

---

## 0. Rule 1 — where this goes, and where it must not go

New primitives live in **`src/fuzzer_tool/core/job_scheduling.py`**. They are
pure functions over `(id, processing_time, due_date, precedence)` tuples with no
`Fuzzer` reference and no I/O.

They do **not** go in `core/schedulers/`. That package's own docstring is
*"Operator-selection schedulers (bandit algorithms)"*, every member implements
`select_op` / `record` / `bandit_stats`, and Hard Rule 40 requires each one
armed through `_register_arms` to declare `supports_priors`. None of the six
algorithms here is a bandit: they are deterministic sequencers over jobs with a
known processing time and a due date. There is no arm to reward, nothing to
record, and no prior to accept. Registering them there would force a fake
bandit interface onto a sorting routine — exactly the "parallel way of doing
something the repo already does" that Hard Rule 1 forbids.

The consumers are in `services/`, one per landing zone below.

---

## 1. The framing: what the fuzzer already has and what it lacks

In `α|β|γ` terms, the fuzzer contains at least four distinct scheduling
problems that today share zero machinery:

| Landing zone | Problem | Current policy |
|---|---|---|
| A. Maintenance tick | `1\|prec,r_j,d_j\|L_max` | fixed program order, one shared gate |
| B. Deterministic stage | `1\|chains,d̄\|ΣU_j` | strict prefix, truncated by a cap |
| C. Seed selection | `1\|r_j\|` (no deadline at all) | weighted sampling, no latency bound |
| D. Parallel workers | `P\|\|C_max` | content-hash partition, load-oblivious |

**The enabler already exists.** `core/cost_ledger.py` gives a measured,
persisted per-seed `p_j`: `total_time / cost_samples`, with the corpus mean
substituted for seeds that have no samples (deliberately distinguishing
*unmeasured* from *measured as free*). Getting `p_j` is normally the expensive
half of applying any of these algorithms, and it is already done and already
survives `--resume` (`b393145`, keyspace fixed in `3b2c6c0`).

**What is missing is `d_j`.** The `seed_meta` schema is `fuzz_count`,
`coverage_edges`, `added_at`, `momentum`, `total_time`, `cost_samples`,
`input_size`, `valid`, `timed_out`, `avg_distance`, `input_entropy`,
`redqueen_offsets`, `redqueen_matches`. There is no `last_picked`. The `age`
term in `seed_picker.py:1091/1329/1392` is `now - added_at` — age since
*admission*, not since *last visit*. Nothing in the tree bounds how long a seed
can go unselected. The staleness term at `seed_picker.py:606-608` is
`fuzz_count / (coverage + 1)` against a `50.0 * T` threshold, which is a
productivity ratio, not a waiting time.

That single missing field is what gates landing zone C, and it is the only
schema change any of this needs.

---

## 2. Landing zone A — the maintenance tick is an unscheduled job set

### What exists

`services/fuzzer.py:6034-6151` is one monolithic block behind one gate:

```
effective_interval = self._stats_effective_interval()     # :6034  (~10s of work)
if self.exec_count - self._last_stats_exec >= effective_interval:
    self._cull_queue()                                    # :6038
    shannon_entropy_global() + _record_entropy_sample()
    self._allan.update(delta)
    homogeneity: get_edge_counts() + per-column chi-squared
    record_coverage_snapshot()
    print_stats() / _append_coverage_log() / _record_discovery_snapshot()
    self._regime.observe(...)  + regime-driven strategy adjustment
    stall detection -> _maybe_trigger_stall_recovery()
    self._check_memory_and_prune()                        # :6134
    self._check_corpus_size_and_prune()                   # :6138
    gc.collect()  (i % 500)                               # :6143
    self._dump_stats(); self._save_state()                # :6146-6147
self._run_crash_replays()      (i % 500)                  # :6149
self._run_sanitizer_replays()  (i % 500)                  # :6151
```

Fourteen jobs, one due date, one order, chosen by where each line happens to
sit in the file.

### Measured cost spread

Synthetic corpus, same container, `EdgeTracker` populated directly (see §10 for
the script):

| job | 200 seeds / 300 edges | 1000 / 800 | 2000 / 1500 |
|---|---|---|---|
| `_cull_queue` | 15.5 ms | 177.1 ms | 557.7 ms |
| `shannon_entropy_global` | 0.20 ms | 0.92 ms | 1.56 ms |
| `record_coverage_snapshot` | ~0.00 ms | ~0.00 ms | ~0.00 ms |

At 2000 seeds one job of the fourteen costs ~5.5% of a 10-second interval and
rides in the same batch as jobs costing three orders of magnitude less.
`_cull_queue` is `O(Σ|edges|)` twice over — once to build `top_rated`, once for
the greedy cover — so it grows with the corpus while the tick spacing does not.

### The evidence that this is already felt

Three jobs inside the block have grown their own private period or budget gate,
independently, each solving one instance of the problem the queue would solve
once:

- `_run_crash_replays(budget_ms=200)` / `_run_sanitizer_replays(budget_ms=200)`
  (`fuzzer.py:4559`, `:4562`) — an execution budget, plus an `i % 500` gate
- `_check_memory_and_prune` — an internal `self.exec_count -
  self._last_memory_prune_exec < 1000` early return
- `gc.collect()` — an `i % 500` gate

Three mechanisms, three shapes, no shared vocabulary.

### Real precedence constraints

This is why **Lawler** belongs here and not elsewhere. The block has genuine
precedence relations, currently enforced only by line order:

- `_cull_queue` → `SeedScorer.score(favored=...)`. `_cull_queue` writes
  `self._favored` (`:3215`); `fuzzer.py:5998` reads it into the power schedule,
  where `_fast_factor` / `_coe_factor` / `coe_skip` branch on it
  (`schedules.py:259/264/489/509`). A stale favored set silently mis-scores
  every seed picked before the next tick.
- `_check_memory_and_prune` / `_check_corpus_size_and_prune` → `_save_state`.
  Both can call `_auto_minimize_corpus()` and change `self.corpus`; saving
  first would persist a corpus about to be evicted.
- `_regime.observe` → the regime-driven strategy adjustment → stall detection.
  The adjustment reads `self._stall_recovery_active`, which stall detection
  then re-tests.

Lawler is exactly `1|prec|f_max`: schedule back to front, repeatedly taking the
job with no unscheduled successors and the *latest* due date, placing it last.
Exact and O(n²) at n = 14, which is free.

### Proposal

`services/maintenance.py` holding a `MaintenanceQueue`:

- Each job registers `(name, callable, period, predecessors)`.
- `p_j` is learned by EWMA of observed wall time — no static table to drift.
- `d_j = last_completion + period`.
- **EDF** picks the next job on each tick.
- **MDD** (`max(t + p_j, d_j)`) takes over under overload. This is not
  decoration: EDF's documented failure mode is that when the system is
  overloaded, *which* jobs miss is unpredictable. A fuzzer under a slow target
  is routinely overloaded relative to a 10-second tick. MDD accounts for the
  partial sequence already built, which is precisely the difference from plain
  EDD.
- **Lawler** computes the static order once from the precedence DAG; EDF/MDD
  choose among the jobs whose predecessors are satisfied.
- A per-tick budget, so one expensive job defers rather than blocking.

The three ad-hoc gates above get deleted and re-expressed as periods —
Hard Rule 1's "fix it in one place for everything".

---

## 3. Landing zone B — the deterministic stage silently drops two of its four passes

### What exists

`services/operators.py:240`, `_deterministic_mutation_stream`, walks in strict
prefix order: bitflip 1/1 → byteflip 8/8 → 8-bit arithmetic → interesting
values. Cost per input byte: 8 + 1 + 16 + 8 = 33 mutants. The whole stream is
truncated by `MAX_DET_MUTATIONS = MAX_QUICK_EFF_EXECS = 65536`
(`core/skipdet.py:29`).

### Measured

Running the generator itself and bucketing each yield by stage boundary:

| `len(seed)` | yielded | bitflip | byteflip | arith | interesting |
|---|---|---|---|---|---|
| 64 | 2112 | 100% | 100% | 100% | 100% |
| 512 | 16896 | 100% | 100% | 100% | 100% |
| 1986 | 65536 | 100% | 100% | 100% | 100% |
| 2621 | 65536 | 100% | 100% | 100% | 0.1% |
| 7281 | 65536 | 100% | 100% | 0.0% | 0.0% |
| 8192 | 65536 | 100% | 0.0% | 0.0% | 0.0% |
| 16384 | 65536 | 50% | 0.0% | 0.0% | 0.0% |

The breakpoints follow directly from the cost model: interesting-value
substitution is complete only while `33·len ≤ 65536` (len ≤ 1986), arithmetic
only while `25·len ≤ 65536` (len ≤ 2621), byteflip only while `9·len ≤ 65536`
(len ≤ 7281). **At and above 8192 bytes the deterministic stage is bitflip
1/1 and nothing else.**

The docstring says the cap "simply truncates the schedule when it's exceeded,
same as AFL++'s own time-boxing". The truncation is not uniform: it is a
prefix, so the cap does not shorten four passes, it deletes two of them
outright. Nothing reports this; there is no counter, no warning, and no test
covering a seed above the breakpoint.

### Is it live?

Yes, though not on every configuration. `parallel.py`'s `max_len` default is
4096, which sits between the arithmetic and byteflip breakpoints — so even
there, interesting-value substitution is already gone. The Boltzmann A/B ran
`direct_lite` at `-m 65536`, and vendored corpora (ffmpeg, sqlite) routinely
carry seeds well past 8192 bytes.

### Proposal

This is `1|chains,d̄|ΣU_j`: a fixed budget, jobs with processing times, maximize
the number completed on time. Replace the strict prefix with a quota split —
each pass gets a share of `MAX_DET_MUTATIONS` proportional to its information
density per exec, and the round-robin interleaves them. Four partial passes
beat two complete ones and two absent ones, because the passes probe
structurally different things (single-bit sensitivity vs. integer boundary
conditions); running only the first says nothing about the fourth.

Note the honest limit: the *right* quota split is an empirical question this
handover does not answer. The defensible first version is equal-yield-per-pass
plus a stat counter reporting the truncation, so the question becomes
measurable. Shipping the counter alone would already be an improvement over
silent deletion.

---

## 4. Landing zone C — a starvation bound for seed selection

### What exists

`SeedPicker.pick_seed` dispatches across thirteen strategies (Elo-arbitrated,
`seed_picker.py:140-208`). Every one of them is a *scoring* policy: it computes
a weight and samples. None of them bounds revisit latency. A seed whose weight
is persistently small is picked with small probability forever; there is no
mechanism that eventually forces it.

`SeedScorer` (`core/schedules.py`) assigns **energy** — how many mutations a
seed gets when chosen. Nothing assigns a **period** — when it must next be
chosen. Those are different quantities and the tree only has the first.

### Proposal — a thin layer, not a replacement

1. Add `last_picked` to `seed_meta`. Cheap now that `3b2c6c0` keys the metadata
   by a 16-char content hash instead of `seed.hex()` (the old key was the whole
   seed content, and `save_state` dropped every entry ≥ 128 bytes).
2. Derive a per-seed period from the energy multiplier `SeedScorer` already
   computes: high energy → short period. No new signal is invented; the
   existing schedule is reinterpreted as a rate.
3. Compute **least slack time** — `slack = (last_picked + period) - now`, the
   LST rule from the dynamic-priority reference. When any seed's slack goes
   negative, force it as the next pick; otherwise the existing picker decides
   untouched.

That converts an unbounded worst case into a bounded one without changing the
weight mixture at all. It is deliberately an override, not a new arm: adding a
fourteenth Elo strategy would make the bound probabilistic again, which defeats
the point.

### The honest risk

This is the same axis the Boltzmann cost-energy A/B landed on, and that
returned a **bounded null**: effect under ~5 edges on png/jpeg, under ~10 on
grep, with the direction inconsistent across targets. The measured intra-cell
noise floor was sd ≈ 4.6 edges (png), 4.7 (jpeg), 12.3 (grep). Any A/B here
inherits that: **replicates, not seeds**, resolve it, and 60 of 120 cells in
that matrix (zlib/lz4/gzip) were bit-for-bit deterministic and could not
produce a discordant pair in either direction.

Before spending cells, run `tools/cost_dispersion.py` per target — the §1
identity from the Boltzmann handover (under uniform per-exec cost, the two arms
are the *same computation*) is a property of the target, not the harness.
Measured on grep: CV 0.922, p90/p10 7.18×, max/min 32.5× over 810 seeds.

---

## 5. Landing zone D — Multifit, and the finding that blocks it

### The intended application

`P||C_max`: partition the initial corpus across `-j N` workers by measured cost
so no worker draws a systematically more expensive share. Today assignment is
`core/parallel_fractal_partition.py::assign_worker` — a fractal jittered
Voronoi root cell of `sha256(seed)`. That is stable, coordination-free, and
completely load-oblivious: it is a hash, and the cost ledger it could consult
is right there.

### The blocking finding: there is no initial corpus distribution at all

`run_parallel` (`services/parallel.py:330`) creates `corpus_dir/.wN` per worker
(`:81-82`) and starts each `Fuzzer` against that empty directory. The only
inbound path is `_sync_corpus_in` (`:223`), which iterates `parent_dir.iterdir()`
and skips everything not matching `.w*`. Neither `run_parallel` nor
`cli/commands.py:347` copies, splits, or imports the user's existing corpus.

Verified on this tree — a pre-existing corpus of 6 seeds (5 loose at the top
level plus one under `seeds/`), two worker dirs, `_sync_corpus_in` for worker 0:

```
imported by worker 0 from a 6-seed pre-existing corpus: 0
```

Workers start empty and can only exchange seeds they discover themselves. On a
fresh `-j N` run the entire supplied corpus is unreachable.

`tests/test_parallel.py` covers sibling-to-sibling sync only (`.w0` → importer),
so nothing in the suite observes this. Compare the failure mode
`_sync_corpus_in`'s own docstring records: the same function once listed
non-recursively, transferred zero seeds, and imported `state.pkl.gz` as a
garbage seed into every sibling. Same shape, one layer up.

### Consequence for the plan

Multifit cannot be ported yet: its input does not exist. The commit that
introduces an initial distribution is a prerequisite *and* is the bug fix. Ship
round-robin first — correct, monotone, and enough to close the hole — then
Multifit behind a flag on top of it.

### The Multifit caveat to carry into that commit

Multifit is **not monotone**: decreasing one input can increase the returned
makespan (the reference's own example: with n=3, changing a 17 to a 16 forces
FFD from 3 bins to 4). Per-seed costs here are EWMA estimates that drift every
tick, so a partition recomputed on live costs can thrash between assignments
for no reason connected to load. Recompute on a long period with hysteresis, or
only at campaign start. The current fractal partition's best property — a seed
maps to the same worker in every run regardless of discovery order — is exactly
what a cost-driven repartition gives up, and that trade should be stated in the
flag's help text.

---

## 6. What does not apply — considered and rejected

Kept as its own section on purpose: "absent from the doc" and "considered and
rejected" are different states and only one should be re-proposable.

- **MDD / EDD for seed calibration.** `_calibrate_seed_baselines`
  (`fuzzer.py:5364`) runs every corpus seed once, in `list(self.corpus)` order,
  with no budget and no deadline. It is a real scheduling problem — but the
  objective is mean flow time, and the optimal rule for `1||ΣC_j` is **SPT**
  (shortest processing time first), not a due-date rule. There are no due dates
  to modify. Worth doing; not one of the six.
- **The `α|β|γ` notation as a module.** It classifies, it does not compute.
  Its value here is a docstring convention: each consumer states its triple, so
  nobody later applies an `L_max` algorithm to a `C_max` problem. That
  discipline is the entire contribution — no code.
- **Multifit for the continuous fuzz loop.** Workers do not consume a fixed job
  list; they run until stopped. Makespan is undefined. Multifit applies only to
  bounded batches: initial distribution, crash-replay sets, minimization
  passes.
- **EDF as the seed picker.** EDF is optimal for *meeting deadlines*. Fuzzing
  has no real deadlines — a seed that is never re-fuzzed costs discovery, not a
  missed guarantee. Replacing the weighted picker with EDF would discard every
  coverage signal in `_compute_weights` in exchange for a property nobody
  asked for. Use LST as an override only (§4).
- **Lawler for landing zones B, C, D.** No precedence constraints exist in any
  of them. Lawler with an empty precedence relation degenerates to EDD, so
  using it there would be a more expensive way to sort.

---

## 7. Commit sequence

Ordered so each commit is independently defensible and the two with measured
evidence land first.

1. **`core/job_scheduling.py`** — pure primitives: `edf_order`, `mdd_order`,
   `wmdd_order`, `lawler_order`, `ffd_pack`, `multifit`. No fuzzer imports.
   Falsification and adversarial tests per Hard Rule 23; `lizard --CCN=15` per
   Rule 45. Nothing wired yet.
2. **Deterministic-stage quota** (§3) — the measured finding. Ships with the
   truncation counter and a test at each breakpoint length (1986 / 2621 / 7281
   / 8192). Self-contained; no A/B needed.
3. **Initial corpus distribution to workers** (§5) — the bug. Round-robin,
   plus a regression test asserting a non-empty import from a pre-existing
   corpus. No Multifit yet.
4. **`services/maintenance.py`** (§2) — `MaintenanceQueue` with Lawler-ordered
   precedence and EDF/MDD dispatch, replacing the monolithic block and
   absorbing the three ad-hoc gates.
5. **Multifit partition** behind `--partition=cost` (§5), on top of commit 3,
   with hysteresis and the monotonicity caveat documented.
6. **`last_picked` + LST override** (§4) behind a flag, with a replicated A/B
   using `tools/bench_replicated.py` and `--lock-single-thread`.

Commits 2 and 3 are bug fixes with evidence and do not need an A/B. Commit 6
does, and is the most expensive thing here.

---

## 8. Falsifiers

Stated up front so a null result is a result and not an ambiguity.

- **§2 (tick).** Falsified if, with the queue installed, the observed p95
  lateness per job is not lower than the current block's, or if total tick
  wall time rises. Instrument first: log per-job wall time in the existing
  block for one campaign before changing anything. If the spread turns out to
  be flat on real targets, the queue buys nothing and should not ship.
- **§3 (deterministic stage).** Falsified if a quota split finds strictly fewer
  edges than bitflip-only on seeds above 8192 bytes — i.e. if bitflip really is
  worth the whole budget. Testable directly against a vendored ffmpeg or sqlite
  corpus.
- **§4 (LST).** Falsified by the same bounded-null shape as the Boltzmann A/B.
  Pre-register the effect size the matrix can resolve *before* running it; a
  7/2 interim split over ten cells is not a weak version of the result, it is a
  different quantity (that lesson is in
  `docs/learnings/2026-08-30-boltzmann-ab-result.md`).
- **§5 (Multifit).** Falsified if per-worker exec counts under a cost partition
  are no more even than under the fractal hash partition. Measure the
  coefficient of variation of per-worker eps; if the hash is already even
  because seed cost does not correlate with content, Multifit is dead and the
  round-robin fix (commit 3) is the whole win.

---

## 9. Open questions

- What is the actual per-job cost distribution in the tick on a real target?
  The §2 numbers are synthetic. `_cull_queue` dominates by construction, but
  the χ² homogeneity block calls `shm_cov.get_edge_counts()` and loops the full
  hit-count map every tick, and that was not measured here.
- Does seed cost correlate with the fractal root cell? If it does not, the hash
  partition is already balanced in expectation and §5's Multifit half is moot
  (the round-robin half is not).
- Should the deterministic quota be static or adaptive? An adaptive quota is
  another bandit and belongs in `core/schedulers/` if so — which would be the
  one place any of this touches that package.
- Interaction with `SkipDetector.should_det_fuzz`: the quota changes what
  "deterministically explored" means for a large seed, and `trace_mini_from_edges`
  feeds that decision. Not analysed.

---

## 10. Reproducing the measurements

Container: single core; install deps with
`pip install --break-system-packages xxhash numpy`.

**Deterministic-stage truncation (§3):**

```python
import sys; sys.path.insert(0, "src")
from collections import Counter
from fuzzer_tool.services.operators import _deterministic_mutation_stream

def stage_of(i, L):
    if i < 8 * L: return "bitflip"
    if i < 9 * L: return "byteflip"
    if i < 25 * L: return "arith"
    return "interesting"

for L in (64, 512, 1986, 2621, 7281, 8192, 16384):
    c, n = Counter(), 0
    for i, _ in enumerate(_deterministic_mutation_stream(bytes(L))):
        c[stage_of(i, L)] += 1; n += 1
    full = {"bitflip": 8*L, "byteflip": L, "arith": 16*L, "interesting": 8*L}
    print(L, n, {k: round(100.0 * c.get(k, 0) / full[k], 1) for k in full})
```

**Empty parallel workers (§5):**

```python
import sys, tempfile; sys.path.insert(0, "src")
from pathlib import Path
from fuzzer_tool.services.parallel import _sync_corpus_in
from fuzzer_tool.adapters.filesystem import hash_data

class FakeFuzzer:
    def __init__(self): self.seen_hashes = set(); self.got = []
    def save_to_corpus(self, d): self.got.append(d)

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    for i in range(5):
        d = b"seedcontent%d" % i
        (root / f"id_{hash_data(d)}").write_bytes(d)
    (root / "seeds").mkdir()
    (root / "seeds" / f"id_{hash_data(b'nested')}").write_bytes(b"nested")
    for w in range(2):
        (root / f".w{w}" / "seeds").mkdir(parents=True)
    f = FakeFuzzer()
    _sync_corpus_in(root, f, self_dir=root / ".w0")
    print("imported:", len(f.got))       # -> 0
```

**Tick job costs (§2):** populate an `EdgeTracker` with N synthetic seeds
(`seed_edges`, `_edge_owner_count`) plus a matching `seed_meta` carrying
`total_time` / `cost_samples` / `input_size`, then call
`Fuzzer._cull_queue(stub)` on a stub exposing `_edge_tracker`, `seed_meta` and
`mean_exec_time()`. Timings above used 200/1000/2000 seeds at 300/800/1500
edges each.
