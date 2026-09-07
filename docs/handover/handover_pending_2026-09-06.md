# Handover — open work, prioritised

**Date:** 2026-09-06
**Base:** `d0ba9ad` (`fix(tests): stop the FormatFuzzer registry leak and refresh three stale inventories`)
**Replaces:** the open half of twenty-two documents under `docs/handover/`.
**Companion:** `handover_done_2026-09-06.md` — completed work, the removal
ledger (§12 there) and the lessons (§13 there). Nothing appears in both.

Every item below was **re-checked against live source at `d0ba9ad`**, not
against its source document's status line. Where an item's anchors had drifted
they were re-resolved; where a cited commit no longer exists that is said. Four
items that the old documents listed as open are **not here** because they have
since landed — GARCH, Navier–Stokes, `_swap_tuple` implementation, and the
FFmpeg build findings F1/F2/N1/N2/N5/N6 (see the done document, §5 and §8).

---

## 0. How this is ordered

Priority is by **what the item is**, not by how interesting it is:

| Tier | Meaning |
|---|---|
| **P0** | A defect in shipped code that a running campaign can hit. Ordered within the tier by whether it corrupts a *decision* or only a *display*. |
| **P1** | A measured win with an equivalence oracle already in hand. Not a bug, but the number is real and the correctness argument is settled. |
| **P2** | Code that shipped and is **not wired**. Dead on arrival. Each is cheap to either connect or retire, and leaving it in the middle is the worst of the three. |
| **P3** | Design work with genuine content, each gated on a **stated question that must be answered on paper before code moves**. |
| **P4** | Genuine, blocking nothing. Pick up when the tiers above are clear. |
| **E** | Evaluation runs. No code change; machine time and a decision. |

Two rules carried from the done document that apply to everything here:

- **A `[ ]` is not evidence.** Re-verify against live source before starting.
- **"Absent from this document" and "considered and rejected" are different
  states.** Only the first should be re-proposed. §11 says where the rejections
  live.

---

## P0 — defects in shipped code

### P0-1. The deterministic stage truncates as a *prefix*, silently deleting two whole passes

`services/operators.py:272 _deterministic_mutation_stream`, cap
`MAX_DET_MUTATIONS = MAX_QUICK_EFF_EXECS = 65536` (`core/skipdet.py`).

The schedule costs **33 mutants per byte** (8 bitflip + 1 byteflip + 16
arithmetic + 8 interesting — counted by running the generator), so the cap is
reached partway through and the remaining passes never run at all. Measured by
input length:

| len | bitflip | byteflip | arith | interesting |
|---|---|---|---|---|
| 1986 | 100% | 100% | 100% | 100% |
| 2621 | 100% | 100% | 100% | **0.1%** |
| 7281 | 100% | 100% | **0%** | 0% |
| 8192+ | 100% | **0%** | 0% | 0% |
| 16384 | **50%** | 0% | 0% | 0% |

The docstring says it "truncates the schedule, same as AFL++'s own
time-boxing". It does not shorten four passes — it **deletes two entirely**.
No counter, no warning, no test above the breakpoint. This is live: the default
`max_len` in `parallel.py` is 4096, already past the interesting-value
breakpoint.

**Fix shape:** a per-pass quota rather than a flat prefix cap, plus a truncation
counter surfaced in stats. Bug with evidence — **no A/B needed**.

### P0-2. `run_parallel` never distributes the pre-existing corpus

`services/parallel.py:81-82` creates `corpus_dir/.wN` **empty**
(`worker_corpus.mkdir(parents=True, exist_ok=True)`), and the only inbound path
is `_sync_corpus_in`, which scans *sibling* `.w*` directories. Neither
`run_parallel` nor `cli/commands.py` copies anything in.

Verified with a 6-seed corpus (5 loose + 1 under `seeds/`): **0 imported**.
`tests/test_parallel.py` only covers sibling-to-sibling sync. This is the same
failure mode that `_sync_corpus_in`'s own docstring records (non-recursive
listing → 0 seeds, `state.pkl.gz` imported as garbage) one level up.

Also **blocks** the Multifit port in P3-3: its input does not exist.

**Fix shape:** round-robin the discovered corpus across workers at startup, plus
a regression test. Bug — **no A/B needed**.

### P0-3. `_record_cmp_progress` is per-callback, not per PC site

`services/fuzzer.py:3461`, reached from `:3937`.

The last open item from the cmplog counter thread; the other four closed. The
measured case: `memcmp` reports `(4,3)` folded across callbacks where the
per-site truth is `(3,3)` and `(1,0)`. Folding by callback merges progress at
one comparison site with stagnation at another, so the growth threshold fires
on the wrong evidence. The per-site counters already exist (`.sites`); the
consumer does not read them.

### P0-4. The colorization cache is keyed on a bare `id()`

`services/operators.py:943` (`pairs_id = id(cmplog_pairs)`), and the same shape
at `:1886` and `:1902`.

CPython reuses an object's id after it is freed, so a rebuilt pairs list can
land on a freed address and read a colorization mask computed for a **different
operand set**. The neighbouring memo at `:614` does the right thing —
`id(cmplog_pairs) + len(cmplog_pairs)` — which is evidence the hazard was
already understood at one site and not generalised. (Same shape as the
`id(seed_buf)` finding that killed the SJT state design; see the done document
§10.)

**Fix shape:** mix in `len()` at minimum, or key on a monotonically bumped pool
version. Cheap.

### P0-5. `shapley._prune_edges` discards the numerically *low* half of the edge-id space

`core/shapley.py:74`. The docstring says "Drop oldest half of tracked edges";
the body does `edges = sorted(self._all_edges); drop = edges[:len(edges)//2]` —
it sorts **numerically** and drops the smallest ids. `_all_edges` is a `set`;
there is no insertion order to recover. Edge ids are `prev_loc ^ cur_loc`, a
hash: **the id says nothing about age.**

Measured: two operators of **identical** productivity (4 edges per execution,
40,000 executions) differing only in which half of the 65,536-entry map their
edges land in → tracked 7080, low-half 1125 vs high-half 5955, Shapley values
**0.841 vs 0.159 — 5.3× credit distortion**. And it compounds: low-id edges are
evicted, rediscovered, re-added and evicted again on every prune.

**Scope, stated honestly and belonging in the commit message:**
`shapley_values()` is read only at `services/stats.py:362` and `:953`, both
display, behind `--shapley`. It feeds **no** scheduling decision. This is a
wrong number in a report, not a misdirected campaign — which is why it sits at
the bottom of P0 rather than the top.

**Two valid fixes.** Minimum: evict by `_edge_total` ascending (the right
quantity is already tracked; the eviction ignoring it *is* the defect).
Principled: **Misra–Gries** over `SHAPLEY_EDGES_MAX = 10,000` counters, which
*guarantees* that any item with frequency > n/(k+1) survives, because it
decrements **all** counters rather than deleting a subset.
**The falsifying test is already specified**: two operators, equal
productivity, disjoint id ranges; assert the Shapley values land within noise.
It fails hard against today's code.

---

## P1 — measured wins, oracle in hand

### P1-1. The deterministic stream copies the seed twice per mutant

`services/operators.py:270`. `bytearray(data)` + `bytes(mutant)` for **every**
mutant, at 33 mutants per byte → `2·33·n²` bytes copied. Measured with no cap:

| n | mutants | time | per mutant |
|---|---|---|---|
| 256 | 8,448 | 2.0 ms | 0.24 µs |
| 1,024 | 33,792 | 11.4 ms | — |
| 4,096 | 135,168 | 56.5 ms | — |
| 16,384 | 540,672 | **618.2 ms** | 1.14 µs |

16× the size costs 54× the time. A prototype holding **one persistent scratch
bytearray** (edit a byte, `yield bytes(scratch)`, restore) gives 0.49 µs/mutant
at n=16384 — **2.3×** — with the same contract. Constant factor, not asymptotic:
`bytes(scratch)` is still O(n) per mutant and the contract requires it (dropping
it would mean handing out a memoryview that changes under the consumer).

**This is the third instance of one policy and should be written as policy:**
*any operator that evaluates many candidate edits against a buffer should mutate
in place and undo, not copy.* First was `gradient_descent` (0.79 → 0.088 ms at
8 KiB, shipped); second was the dancing-links framing (P4-4). This is the
largest remaining site.

**Mandatory caveat.** The prototype produced a *different* mutant count (25/byte
vs 33) because the arithmetic pass was reconstructed rather than copied.
Equivalence must be **proved mutant by mutant** (`list(old(d)) == list(new(d))`)
over a sweep of seeds **and of caps**, including caps that land mid-pass —
`MAX_DET_MUTATIONS` truncation interacts with the buffer restore. Do not ship
this on a spot check.

Note P0-1 touches the same function. Land P0-1 first (it changes *what* the
schedule emits); then P1-1's oracle is written against the corrected schedule.

### P1-2. Aho–Corasick for the cmplog operand scan

`core/colorizer.py:161 colorize_from_cmplog` walks each operand separately with
`input_data.find(token, pos)` → O(P·n) with P = 2× the pair count, and
`CMPLOG_PAIRS_MAX = 5000` means **up to 10,000 tokens**.
`core/weizz_tags.py:580 build_tag_map_from_cmplog` has the identical shape over
the same pool (step 1 of its own docstring) — **one automaton serves both**.

Where the cost actually is, decomposed at n=64 KiB, P=512 pairs: the `find()`
calls alone 32.02 ms; find + the Python marking loop 31.78 ms; `color_mask()`
genexpr 2.05 ms; the same mask via numpy 0.13 ms. **The scans are the entire
cost** (only 854 match spans), so the obvious reflex — vectorise the inner loop
— buys nothing. `color_mask` is genuinely 16×, but on a 2 ms term.

Measured, pure-Python AC (trie goto + BFS failure links + suffix-merged
outputs), span sets verified identical:

| n | P | find | AC search | ratio |
|---|---|---|---|---|
| 4,096 | 64 | 0.24 ms | 0.39 ms | **0.61×** |
| 4,096 | 512 | 1.90 | 0.73 | 2.61× |
| 16,384 | 512 | 6.36 | 2.40 | 2.65× |
| 65,536 | 512 | 30.96 | 8.51 | 3.64× |
| 16,384 | 2,048 | 32.65 | 3.42 | 9.55× |
| 65,536 | 2,048 | 123.25 | 12.26 | **10.05×** |

Build cost 0.62–24.5 ms depending on pattern count.

**The caveat that decides the design:** at small P the automaton **loses**, for
a structural reason and not a tunable one — AC search is a Python loop over
bytes (~133 ns/byte) while `find` is a C scan. The crossover is near
**P ≈ 128–256**. So the dispatch is the same shape the diff work settled on:
neither a flag nor a threshold on `n`, but a decision on **the quantity that
drives the cost** — here the token count, because AC is O(n) independent of P.
The build amortises for free by caching the automaton per pair-pool version.

Two side findings in the same code: **duplicate tokens are re-scanned** (at
P=512 the 1,024 tokens hold 1,022 distinct values and the baseline scanned all
1,024 — 854 spans vs the deduplicated automaton's 852); and the `id()` cache key
is P0-4.

**Open question that must be answered before porting:**
`build_tag_map_from_cmplog` processes pairs in a **deliberate order**
(`_pair_key`: shorter/more specific operands claim bytes first) and an automaton
returns everything in one pass without that order. The port must **collect**
matches and then apply them in the existing order, verifying tag equivalence
byte for byte — not assuming it.

### P1-3. The `seen_hashes` clear silently re-admits every previously seen seed, once

`adapters/filesystem.py:553, 626, 686, 724` — `if len(seen_hashes) >
SEEN_HASHES_MAX: seen_hashes.clear()` with `SEEN_HASHES_MAX = 200_000`.

These four are **not** the same as the other cache-clear sites in the tree: they
are dedup sets that interact with the bloom filter. After the clear the bloom
still answers "seen" for every prior hash while `seen_hashes` says no, so the
`elif` fails, control falls to the `else`, and **every previously seen seed is
re-admitted exactly once more**. Bounded re-admission, not an unbounded leak,
and rare at 200,000 — but it is a *correctness* effect of a memory policy, and
it belongs in its own commit, separate from any cache refactor.

**Do not batch this with P3-6** (the four ad-hoc eviction policies). That is a
behaviour-changing refactor of five sites; this is a bug.

---

## P2 — shipped but unwired

Each of these is code that exists, has tests, and is read by nobody in
production. All four are cheap. **The point of this tier is that leaving them
half-connected is worse than either finishing or retiring them** — a module that
lands, goes green and is never read is exactly the failure mode that the
Navier–Stokes plan's falsifier list omitted, and it is what happened to P2-1.

### P2-1. `core/target_difficulty.py` has no production consumer

224 lines — `estimate_isoperimetric_profile`, `estimate_percolation_threshold`,
`estimate_growth_curve` — imported **only** by
`tests/test_target_difficulty.py`. Its own document states it is "called at
fuzzer startup… drives time budget, initial corpus size, and operator
preselection". That wiring was never done.

Design note that changes the target: per Diskin–Easo–Radhakrishnan–Sudakov–
Tassion (arXiv:2603.03257), supercritical sharpness holds for **every** infinite
transitive graph, stated purely in terms of the isoperimetric function `Φ(n)`
and with no assumption on degree distribution. So this should aim at estimating
**`Φ`**, not at a `p_c` formula. That also makes it the input Module 5 (P4-8)
needs.

**Decide explicitly:** wire it, or mark it as a diagnostic with no scheduling
role and say so in its docstring.

### P2-2. `_swap_tuple` has zero production call sites

`core/mutations/generic.py:78`, with the even-`m` rotation and the odd-`m`
parity-trap fallback, plus `tests/test_swap_tuple.py`. Nothing calls it.

The tree therefore carries **one unwired permutation primitive**, and any new
permutation generator would be the second before the first is connected. That
sequencing decision should be made deliberately.

If it is wired, the constraints are already measured (done document §3): never
`m=3` alone (3-cycles are even, reaching only `A_n` — a regression, not an
extension); `m` must be drawn **relative to `n`**, because `n` ranges from ~4
(top-level ISO-BMFF boxes, WebP chunks) to thousands (MPEG-TS packets, NAL
units, x86 instructions); keep `m=2` as a separate bandit arm so the scheduler
still sees the scale signal; and mind `ExhaustivePool`, which enumerates
`n!/(n−k)!` exactly — at n=16, k=5 that is 524,160 runs against
`DEFAULT_MAX_RUNS = 1,000,000`, and at n=20 with m=5 it is 1,860,480, which
**exceeds the cap, truncates the walk, and leaves `exhausted` False with
`budget_exhausted` True — silently degrading the falsification guarantee for a
whole class of operator tests.**

Untested hypothesis to measure, not assume: in formats with offset tables
(ISO-BMFF `stco`/`stsz`, the ZIP central directory, SQLite page pointers) a
permutation of `m` breaks `m` pointers instead of 2, so early parser rejection
may be *more* likely and deep paths *less*. Measure it with the fuzzer's own
signals.

### P2-3. `invasion_select`'s `frontier_edges` is never passed

`services/seed_picker.py:126` takes `frontier_edges: set[int] | None = None`;
production calls `invasion_select(op_stats, flux_map=flux_map)` at
`services/operators.py:3647`. So the "nothing to invade on this frontier"
short-circuit is **dead outside the tests**. The work is at the call site, not
in the signature.

### P2-4. `_op_secretary` feeds a display and nothing else

`services/fuzzer.py:2058` declares it, `:4309` populates it via
`SecretaryStopping.observe(a/(a+b))`, and the only reader is
`services/stats.py:943` — display. Nothing stops, skips or reweights on it.

The companion finding is that `core/secretary.py` **is not a stopping rule** as
written: its rank term cannot bind, and the surviving clause is a clock rather
than a productivity test. Either give it a consumer or retire it. If it is
replaced, the natural successor is the retirement-value formulation in P3-2.

### P2-5. `CoverageRegimeDetector._classify` ignores 3 of its 5 arguments

`core/coverage_regime.py:128`. `discovery_rate`, `allan_delta` and `exec_count`
are passed by `observe()` and never read (verified by AST). `allan_delta` is
documented as unused; `discovery_rate` is not. **Any new signal added as a sixth
argument lands in the same hole** — which is what would have happened to the
GARCH σ̂² had it been placed where its document proposed. Minor in the same
file: `observe` is annotated `-> None` and returns `self._regime`.

---

## P3 — design work, each gated on a stated question

Nothing in this tier should start before its question is answered **on paper**.

### P3-1. Lexicase selection against the scalar fitness in `core/ga.py`

Source: Schulte, Ruchti, Noonan, Ciarletta, Loginov, *"Evolving Byte-Equivalent
Decompilation from Big Code"*, §III-E.

Fitness becomes a **vector of independent test cases** instead of a scalar sum,
and selection filters the population by best performance over a **random order**
of those tests until one survives. The stated effect: a candidate covering a
region nobody else covers survives with high probability even if it fails almost
everything else.

`core/ga.py` does the opposite today: `FitnessFunction` computes the scalar
`w_novelty·novelty + w_diversity·diversity` and `select_parent` runs a rank
tournament over it.

**Why this fits rather than merely rhymes:** we already have the per-test-case
fitness vector the method needs, and it is the right object — **a seed's edge
set**. Each edge is a test the seed passes or fails. A seed owning a rare edge
*is* the "covers a region nobody else covers" case, and under the current scalar
it is dominated by seeds with broad but redundant coverage. We already have
**two hand-built approximations** of what lexicase would give structurally: the
rarity bonus in `seed_picker` (`RARE_EDGE_OWNERS`/`RARE_EDGE_GAIN`) and MinHash
LSH speciation in `ga.py:132`. Both say "do not lose the seed that is unique in
some dimension". Lexicase gets it from the selection **rule** rather than a bonus
with a calibrated constant.

**The cost, stated plainly:** naive lexicase is O(pop_size · n_tests) per
selection and n_tests here is the distinct edge count — **8,189 on our ffmpeg
target**. Infeasible per pick.

**The gating question, to answer before any code:** what is the right test set —
all edges (slow), a random sample, or only the rare ones? **If the third turns
out equivalent to the rarity bonus that already exists, lexicase buys nothing
and must be rejected.** The A/B has to be able to return "no difference".

### P3-2. Gittins index, and the retirement value that would replace `core/secretary.py`

`core/gittins.py` does not exist. The index table is cheap and correct to
compute; the measured open question is **whether it picks differently** from
what we already do — that measurement exists in the source analysis and should
be re-read before starting.

Two consumers were scoped: the seed picker, and a stopping rule replacing
`core/secretary.py` (see P2-4). The theoretical caveat must be stated in the
docstring rather than glossed: the Gittins index is exactly optimal for the
**discounted** FABP, and our problem is not that problem.

### P3-3. Classical job scheduling — `core/job_scheduling.py`

Does not exist. Six algorithms surveyed: Lawler (`1|prec|f_max`, exact, O(n²)),
EDF, LST, Multifit, MDD, and the `α|β|γ` taxonomy.

**Placement is settled and is not cosmetic.** These go in
`core/job_scheduling.py` as **pure functions** over `(id, processing_time,
due_date, precedence)` with no `Fuzzer` reference and no I/O. They do **not** go
in `core/schedulers/`, whose docstring is "Operator-selection schedulers (bandit
algorithms)", whose members all implement `select_op`/`record`/`bandit_stats`,
and where Hard Rule 40 requires `supports_priors` on anything armed through
`_register_arms`. None of the six is a bandit: they are deterministic sequencers.
There is no arm to reward.

**The enabler already exists** — `core/cost_ledger.py` supplies a measured,
persisted `p_j` (with a corpus mean for seeds lacking samples). **What is missing
is `d_j`:** `seed_meta` has no `last_picked`; `age` in `seed_picker` is
`now − added_at` (age since *admission*, not since last visit); and the staleness
term is `fuzz_count/(coverage+1)` against `50.0·T`, which is productivity, not
waiting time. **Nothing bounds revisit latency.**

Measured maintenance-tick costs that motivate the queue
(`services/fuzzer.py`, fourteen jobs behind **one** gate in fixed program
order), synthetic corpus: `_cull_queue` 15.5 ms (200 seeds) / 177.1 ms (1000) /
557.7 ms (2000); `shannon_entropy_global` 0.20/0.92/1.56 ms;
`record_coverage_snapshot` ~0. At 2000 seeds one job eats ~5.5% of the 10 s
interval.

**Evidence the problem is already felt:** three tick jobs have their own
independent ad-hoc gate — `_run_crash_replays`/`_run_sanitizer_replays` with
`budget_ms=200` plus `i%500`, `_check_memory_and_prune` with an internal
1000-exec early return, and `gc.collect` with `i%500`. Three mechanisms, three
shapes, no shared vocabulary.

**Real precedences exist in the tick** and today are enforced only by line
order, which is what justifies Lawler (free at n=14): `_cull_queue` writes
`self._favored`, which the power schedule reads and on which `_fast_factor` /
`_coe_factor` / `coe_skip` branch; `_check_memory_and_prune` and
`_check_corpus_size_and_prune` can call `_auto_minimize_corpus` and change
`self.corpus` before `_save_state`; and `_regime.observe` → strategy adjustment
→ stall detection.

**Sequence (6 commits):** (1) `core/job_scheduling.py` primitives
(`edf_order`/`mdd_order`/`wmdd_order`/`lawler_order`/`ffd_pack`/`multifit`), no
wiring; (2) **P0-1**; (3) **P0-2**; (4) `services/maintenance.py` with a
Lawler + EDF/MDD `MaintenanceQueue` absorbing the three ad-hoc gates; (5)
Multifit behind `--partition=cost` with hysteresis; (6) `last_picked` + an LST
override behind a flag, with a replicated A/B.

**Multifit is not monotone** — lowering an input can raise the makespan (the
worked example: n=3, changing a 17 to a 16 takes FFD from 3 bins to 4). Per-seed
costs are EWMAs that drift every tick, so a partition recomputed on live costs
can oscillate independently of load → hysteresis, or campaign start only. And
the best property of the current fractal partitioning — a seed lands in the same
worker on every run regardless of discovery order — is exactly what a cost-based
partition gives up.

Item (6) inherits the A/B requirement from the Boltzmann result: bounded null,
noise floor sd ≈ 4.6 edges (png), 4.7 (jpeg), 12.3 (grep), and **replicates are
what resolve it, not seeds**. Run `tools/cost_dispersion.py` per target before
spending cells — the collapse identity is a property of the target, not the
harness.

### P3-4. Koopman search-effort allocation as a power schedule

`grep -i koopman src/` returns nothing. A validated closed form exists in the
source analysis, along with **a falsification condition that must be read before
spending cells** and the missing input it needs. Treat the falsifier as the
first deliverable.

### P3-5. Growing Tree / Growing Forest as the parameterisation of our schedulers

Not a port — a way to *describe* the space of schedulers we already have. Growing
Tree is one loop with one policy knob: how you pick the next cell from the
frontier list. Always newest ⇒ recursive backtracker (DFS); always random ⇒
approximately Prim; always oldest ⇒ minimum "river" factor. The reference demo
exposes it as a literal string: `random:50, newest:30, oldest:75, middle:100`.
The quoted claim: *Growing Forest can duplicate Recursive Backtracking,
Kruskal's, Prim's and Growing Tree by parameter choice alone.*

The mapping: our corpus is a frontier and the seed picker is a policy over it. We
have **nine** operator schedulers plus several seed-selection arms, each its own
class, with **no shared parameterisation saying how any two differ**. This would
let `bench_paired.py` sweep a continuum instead of A/B-ing named
implementations.

**Most actionable piece: Houston's algorithm** — run Aldous–Broder until a
minimum number of cells is visited, then switch to Wilson's: cheap and biased
early when almost everything is new, expensive and uniform late when the
frontier is sparse, at the cost of the uniformity guarantee. **That is exactly
the shape of our saturation gate** (the ≥99% gate collapsing subsumption /
diversity / Wasserstein / proximity to neutral). Houston's is the same trade in
the other direction and gives a principled name to something we do ad hoc.

**Vocabulary worth adopting regardless:** the reference characterisation table
scores each algorithm on Bias Free? / Uniform? / Memory / Time / Dead End % /
Solution %, and separates two properties we conflate — **bias free** (treats all
directions alike) vs **uniform** (generates each outcome with equal
probability), noting only bias-free algorithms can be uniform, and that "no"
(reaches everything, not equiprobably) and "never" (some outcomes unreachable)
are **different** failures. Applied here: a scheduler that can **never** select
certain sequences is a distinct and worse defect than one that selects them
rarely — precisely the class of bug found in the scheduler→mutator reach audit,
where Hierarchical silently dropped runtime-registered operators. We had no
vocabulary for that.

**Gating question:** does the Growing Tree parameterisation genuinely subsume two
of our schedulers, or only appear to? **Answer on paper before moving code.**

*Colour note for whoever writes this up: the reference implementation is called
**Daedalus**. Coincidence with the org name; it will confuse readers.*

### P3-6. The four ad-hoc eviction policies

Four distinct hand-rolled answers to one question: `length_mi._prune_lengths`
(top 50% by count), `length_mi:45` (top half of edges by count),
`shapley._prune_edges` (**P0-5**), `mi._evict_least_observed` (single worst
observed), and `fuzzer.py:3629` (dictionary, keep `dyn_cap//2`). Three of five
are frequency-aware, which is the right instinct; none has the Misra–Gries
guarantee, and the halving ones share a flaw — an evicted item restarts at zero,
so a genuinely frequent item that dips below the median for a while is evicted,
resets, and is evicted again.

**Do not batch this with P0-5.** That is a bug with a written falsifier; this is
a five-site refactor changing behaviour nothing currently measures, and it needs
its own before/after.

---

## P4 — genuine, blocking nothing

1. **Perlin noise as a sibling to `fractal_voronoi`.** Same idea with a
   *continuous* field: sample `noise(i/scale)` per byte and modulate mutation
   **intensity** (probability of touching byte `i`, or delta magnitude), with
   correlation length as the knob and fBm octaves for multi-scale bursts. Fits
   the existing plumbing (`MutatorBase` / `MutationContext` /
   `REGISTRY.register_mutator`), is deterministic via its permutation table so
   `--seed` is free, and is **much** cheaper than Voronoi (4 dot products + 3
   lerps + 3 lookups per sample, no search — against `_nearest_site` sweeping
   5×5 per byte, which was 36% of a campaign before the plan cache). Gap
   verified: of 163 operators, `fractal`/`voronoi` match only
   `fractal_voronoi`, and `noise`/`wave` match nothing.
   **Warning for the implementer: the widely-linked reference snippet is
   wrong** — it prints `b00 = p[i+by0]; b10 = p[j+by1]; b01 = p[i+by0]; b11 =
   p[j+by1]`, so `b00==b01` and `b10==b11`, duplicating two corners and
   degenerating the y interpolation. Perlin's original is `b00=p[i+by0],
   b10=p[j+by0], b01=p[i+by1], b11=p[j+by1]`. The prose is fine; only the
   snippet is transposed.
2. **DEFLATE *structure* mutation.** We have two positions and the gap is
   between them: `mutations/zlib.py`/`gzip.py` corrupt the **compressed** bytes
   in place (their own docstring notes this breaks the stream so the target
   aborts in decompression and the parser behind it never runs), while
   `mutations/recompress.py` inflates, mutates the **plaintext**, re-deflates and
   fixes the trailer. **Nothing touches DEFLATE structure** — not the block-type
   field, not the dynamic Huffman header (HLIT/HDIST/HCLEN), not the code-length
   order permutation, not the distance/length back-reference pairs. That is where
   the decompressor's error paths live, and you only reach them by producing a
   stream well-formed enough to be *decoded* and then wrong. `recompress.py`
   already has the plumbing (bounded inflate with `max_length`, magic sniff
   before any work, size cap, memoised cache). **Do not scope zstd in** — it is
   not vendored.
3. **Burrows–Wheeler as a mutation *domain*, not as compression.** A
   **reversible** permutation grouping bytes by following context; the
   "transform → mutate → invert" plumbing already exists in `recompress.py`. A
   one-byte edit in BWT space is a context-correlated edit in the original,
   scattered across every position sharing that context. **None of the 163 live
   operators produces that** — ours are positional or value-based, never
   context-grouped, including the four template-driven `ff_*`. Verified absent:
   `grep -i burrows/bwt` → zero.
4. **The dancing-links *mechanism*, as policy.** Reject the literal reading:
   corpus minimisation is **set cover**, not exact cover (seeds overlap and must),
   `services/minimize.py:149` correctly uses greedy set cover, and
   `core/percolation.py::bootstrap_minimize_corpus` already extends it with
   iterative k-rigid-core reduction. DLX over 65,536 edges × thousands of seeds
   would not terminate and would solve a problem we do not have. **Take the
   mechanism instead:** dancing links' real contribution is O(1) backtracking —
   unlink a node, restore by relinking, rather than copying state. That is
   P1-1's policy generalised. Where to check whether the pattern repeats: the
   depth-4 AST-scan sites — `build_tag_map_from_cmplog`,
   `_extract_parser_tokens`, `gradient_cmp`, `_compute_bb_values`,
   `colorize_from_cmplog`.
5. **Large Neighbourhood Search refinements** (destroy + repair; ~4 pages, §3.5
   of the LEGO thesis — **do not read past it**, the rest is quadratic
   programming for static structural integrity). Three refinements, ranked:
   (a) **critical destroy** — destroy the part of the solution that is failing,
   not a random part; (b) **escalating neighbourhood** — destroy the critical
   element plus its k-ring, raising k after ~10 failed repairs; (c) **switch to
   a random objective when the objective is blind**.
   **(c) is the one worth thinking about:** it is the honest version of what
   `--reseed-on-stall` gestures at. The thesis argues a stall is **evidence that
   the objective is not measuring what is stuck**, so randomising the
   *objective* beats randomising the input. We randomise the input. These are
   different claims and can be tested against each other on the same stall
   condition. (b) maps to havoc stack depth: `n_mutations` scales with
   `perf_score` but **not** with consecutive failure on *that* seed. The thesis
   also criticises pairing directed destroy with random repair as
   "guess-and-check" — which applies to us verbatim wherever cmplog identifies a
   comparison wall and havoc then mutates without using it.
6. **Frequent-itemset mining over `edge_cooccurrence`** (today display-only).
   Pairwise is the k=2 slice of "which *sets* of edges always fire together".
   Two uses: collapse always-co-occurring edges to one element **before** the
   greedy set cover, and withhold full `RARE_EDGE_GAIN` from an edge that is
   rare only because its whole itemset is. **Design only, nothing measured.**
   Two questions gate it: feasibility (FP-growth is O(transactions × items) and
   our transactions are sets of up to 8,189 edges), and — **if the maximal
   itemsets turn out to be "the basic blocks inside a function", this reproduces
   the ICFG we already build in `core/icfg.py`, more expensively and less
   exactly. The A/B must be able to return "this is just the ICFG".**
7. **Smith–Waterman for homologous crossover.** Previously blocked by the
   Levenshtein OOM, now unblocked. The new design question: **global alignment
   may be the wrong tool.** Two seeds share *regions* (header, chunk, table)
   inside unrelated content; SW finds the best **local** alignments and ignores
   the rest, which is precisely the relation crossover needs, while Levenshtein
   forces end-to-end correspondence and aligns noise with noise to get it.
   **Careful: SW as written in every textbook is the same O(n·m) table the diff
   work exists to avoid** — port the scoring rule and traceback onto the
   affordability dispatch we already have, not the table.
   Companion from the same paper, unblocked with it: **diff-targeted mutation** —
   draw havoc targets from the currently non-matching regions with probability
   `TargetChance` rather than uniformly. We have the machinery
   (`colorization.py`, cmplog); nothing gates havoc site selection on a per-seed
   failure map.
8. **Percolation Modules 5 and 6.** Module 5 (first-passage percolation for time
   budgeting) is absent and wants P2-1's `Φ` estimate as its input; Module 6
   (universality → strategy transfer) is absent. Neither blocks anything.
9. **FormatFuzzer phases 2–3** — decision-seed mode, then hardening. Phase 1 is
   live. Resolve the eager-registration question from the done document §4 while
   here.
10. **Fractal Voronoi approaches B and C** — fractal coverage-space seed
    prioritisation, and fractal corpus partitioning for parallel fuzzing (note
    the interaction with P3-3 item 5: partition stability is the property a
    cost-based split gives up).
11. **skittercreek item G — intermittent `shmat()` failure.** Root cause
    unknown. Twice in ~50 runs the first `ShmCoverage` in a process read back an
    empty edge table after a child exited 0; the header `edge_count` was not
    captured, so it is unknown whether the child failed to attach or the parent
    raced the read. The stale-view hypothesis (`cleanup()` leaving
    `from_address` views bound to a detached mapping) is **fixed but not
    established as the cause**: 0 failures in ~40 runs since, against a pre-fix
    rate near 1 in 25, which is not conclusive. **Nothing to do proactively.** If
    the drop-counter tests in `tests/test_ctx_and_map_size.py` go intermittently
    red, pull this thread and capture the SHM header (`read_edge_count()`,
    `read_diag()`) alongside the child's exit status. Segment exhaustion is
    ruled out — the parent creates and reads the segment in the same test and
    `ipcs -m` shows no leak.
12. **skittercreek item H — item 13, deferred by design.** Byte-level
    attribution of timing anomalies (item 5 × item 7): joining
    `ExecTimeCalibrator`'s anomaly timestamps to the mutation-event stream would
    attribute a slow execution to a specific byte/operator instead of "some exec
    around this time". Both halves exist (`core/temporal_join.py`,
    `core/exec_time_anomaly.py`) and `report.py::_temporal_correlation` already
    uses `join_streams` — but for coverage/discovery snapshot streams, not this.
    **Kept here so it is not silently reinvented without the context of why it
    was skipped.**
13. **Combinatorics gaps §10b–10h**, unchanged and each named with a file:line
    in the source survey: 10b the `ExhaustivePool` `allow_bulk` gate is
    over-conservative; 10c the `rng.random() < p` coin-flip idiom defeats
    enumeration (~20 sites, follow-up to the tigerbeetle P1-5 work); 10d the
    pairwise Markov chain is first-order only; 10e the grammar's full derivation
    space is unreachable; 10g `byte_shuffle` is registered but only the byte
    version exists; 10h `core/markov.py` state transfer across runs is
    unverified. §10a, §10a.1 and §10f are done; §10i is a **verified no-op** —
    the literal fix proposed there changes nothing for any blend weight below
    1.0, and the one real (narrower) edge case found instead is recorded in
    place. Do not re-propose 10i.
14. **SJT as a brute-force optimality oracle** for P3-3's
    `core/job_scheduling.py`. The one surviving use from that whole thread. For
    `1||Σf_j` an adjacent interchange at `(i,i+1)` changes only `C_i`, so the
    objective delta is O(1) rather than O(n), and the SJT walk **is** the
    exchange-argument chain proving SPT/EDD optimal, so a counterexample emerges
    as a named interchange rather than a bare number. **State the crossover, do
    not claim it unconditionally:** measured at n=8 over all 8! sequences, the
    incremental version **loses** at 0.93× with cheap inline arithmetic (because
    `itertools.permutations` runs in C and the delta runs in Python), wins 2.06×
    once the per-job term costs a function call, and 3.88× at ~80 flops/job.
15. **Remaining FFmpeg build items**, in the revised order from the source
    diagnostic: **F3** — port `build_ffmpeg_ready.sh` fully to the new roots;
    **N4** — a failed FFmpeg rebuild silently falls back to stale archives;
    **N3/F5** — `--minimal` does not reach the libraries that actually get
    linked; **F8** — the hardcoded developer paths at `tools/build_targets.sh:52`
    (`mkdir -p /home/dclavijo/tmp`), `:53` (`export
    TMPDIR=/home/dclavijo/tmp`) and `:210` (`TAILSLAYER` default); **F9** — the
    `ERR` trap's wording. F8 is trivial and the only one that misbehaves on
    every machine that is not the author's.

---

## E — evaluation runs, no code change

These need machine time and a decision, not a patch. Each already has its
harness.

- **E1. Weizz paired bench.** The **only** unticked item on the Weizz acceptance
  checklist. `tools/bench_paired.py` against a live target.
- **E2. FormatFuzzer paired run.** Acceptance asks for either a coverage gain or
  a validity gain, with a **throughput regression ≤ 15%** (or automatic Elo
  de-prioritisation).
- **E3. Alpha-beta MCTS A/B** against the plain `MCTSSeedScheduler` on
  `targets/png_read` and `targets/ffmpeg_read`.
- **E4. Fractal Voronoi A/B** — the operator shipped without one.
- **E6. Tang low-rank seed arm A/B.** `--tang` shipped opt-in on 2026-09-07 with
  a measured *negative* on predictive validity but **no interventional run**:
  the score adds nothing over `A.sum(1)` (mean partial Spearman +0.006 over ten
  campaigns, Wilcoxon p=1.0). Full findings, and the seven methodology gaps that
  qualify them, in `docs/handover/handover_tang_recommendation_2026-09-07.md`.
  The higher-value run is item 2 of that document's §10 — re-measure the
  low-rank structure on ffmpeg, where the assumption plausibly holds — because
  a negative A/B on png/zlib would not distinguish "the method fails" from
  "these targets are too small".
- **E5. GARCH and `--continuum`** — both shipped opt-in and unmeasured. For
  GARCH specifically, the ACF study needs **snapshots dumped to disk, not live
  state**: `record_discovery_snapshot` caps history at 500 and trims to 250, one
  snapshot per tick ≈ 10 s of work, while an MLE GARCH(1,1) wants ~500–1000
  observations.

**Before spending cells on any of these, read the Boltzmann result** (done
document §2). The design that produced it — paired in time, arm selected by
source tree via `PYTHONPATH`, `--lock-single-thread`, replicates over seeds — is
the one to reuse, and its power table is the one to size against. In particular:
60 of 120 cells per arm on the standard set (zlib + lz4 + gzip) **cannot**
produce a discordant pair in either direction, because those targets are
bit-for-bit deterministic and saturated. Do not buy guaranteed ties.

**Sampling-axis caveat for anything time-series.** The stats tick is
`_stats_effective_interval()` = 10 × mean eps, i.e. constant in **wall clock**
and variable in **executions**. GARCH(1,1) assumes an equally spaced grid. Allan
looks at edges per tick (uniform in time); CSD looks at edges per kexec on a
time grid, so throughput fluctuation leaks into its variance. Say which axis a
new signal lives on.

---

## 11. Where the rejections live

Nothing here should be re-derived from a source that has already been assessed.

- **`docs/port-backlog.md`** — the single list of what to port, merging four
  surveys, with a **"Rejected"** section that is the most valuable and most
  easily lost part of any survey. It also carries the source-quality accounting:
  one source is the only one with serious statistics and its conclusions stand;
  another reports single numbers per cell with no variance and an invisible 10×
  throughput tax across all its charts — **take nothing from its numbers**.
  Section **J** (the FFmpeg harness work) keeps its 7-commit sequence intact and
  is the only fully specified block ready to start.
- **`docs/TODO.md`** — 11 open items outside the handovers, including the
  successor to Persistence Mechanics §1c (should cumulative execution cost enter
  the eviction ordering — now testable, and inheriting the Boltzmann A/B
  requirement because it changes which seeds exist), the leak-severity/edge-count
  probe, GEP yield at `-O2`, the redundant `reset_bitmap()` on the runner path,
  the inert `--reject-code` validity channel, SGFuzz transitions without a state
  tree, cross-worker collaborative scheduling, four checksum follow-ons, and the
  last retry-until-hit test (`test_fuse_old`).
- **`docs/bugreport_2026-08-21_merged.md`** — ~40 open findings. **Numbering was
  deliberately not compacted**: the gaps are the closed items, and renumbering
  would silently re-point every by-number reference in commits, learnings and
  code comments. **Locate by symbol, not by line** — the anchors have drifted.
  The 40 open ones have **not** been re-verified; that is an audit, not a
  cleanup.
- **Standing rejections worth naming here so they do not return as fresh ideas:**
  the whole shortest-path family (our distance channel is an *unweighted reverse
  BFS*, already O(m+n) — `grep -i dijkstra src/` correctly returns zero); the
  whole MST family for crash clustering (single-linkage at threshold `t` really
  is the MST minus heavy edges, but our similarity graph is **complete**, so
  linear in edges is still quadratic in n — A2 was solved with exact bounds
  instead); DMMSY-SSSP (unweighted graphs, small graphs, and the headline speedup
  is an AVX-512 + V-Cache result wearing an algorithm's name); Gray code for the
  deterministic stage (each mutant is one edit from the **seed**, not from the
  previous mutant, so walking a Gray code changes *what* the schedule tests — the
  in-place win is P1-1 and needs no Gray code); `linear_congruential_generator`
  (**actively harmful** — `--seed` reproducibility runs through `RandPool` and its
  refill accounting); the single-pattern search family (ours is `bytes.find`,
  memchr in C; Aho–Corasick wins only by being **multi**-pattern);
  `cellular_automata` as a structured-fill operator (the thirteen regularity
  operators each invert a **named** statistical test — the price of entry is
  naming the test you defeat); `highest_response_ratio_next` and
  `multi_level_feedback_queue` (optimise a **latency** objective nobody asked
  for, at the cost of the coverage signal in `_compute_weights`; MLFQ's only new
  idea, demotion on budget consumption, is already approximated by the staleness
  term); and EDF as a seed picker (same reason — LST goes in as an **override**,
  not as a new arm, because a fourteenth Elo arm would make the bound
  probabilistic).
