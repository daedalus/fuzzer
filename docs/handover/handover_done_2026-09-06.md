# Handover — completed work, consolidated

**Date:** 2026-09-06
**Base:** `d0ba9ad` (`fix(tests): stop the FormatFuzzer registry leak and refresh three stale inventories`)
**Replaces:** twenty-two documents under `docs/handover/`. See §12 for the
removal ledger — every one is recoverable from git and §12 says which section
absorbed it.

**Companion:** `handover_pending_2026-09-06.md` holds everything still open,
prioritised. Nothing appears in both. If an item is in neither, it was
rejected — see §11 of the pending document for where the rejection registries
live.

---

## 0. How this was built, and how to read it

Every status line below was checked **against live source at `d0ba9ad`**, not
against the source document's own header. That is not pedantry: four of the
twenty-two documents carried a header that had been wrong for weeks in one
direction or the other, and three cited commits that no longer resolve. The
corrections are in §11.

An entry here means: the code is in the tree, it has tests, and a named
artifact can be pointed at. An entry does **not** mean the feature was
measured to help — where a paired bench is still owed, the entry says so and
the bench itself is filed as an open evaluation item in the pending document.

Conventions used throughout:

- `file.py:NNN` anchors were re-resolved at `d0ba9ad`. Line numbers drift;
  **locate by symbol, not by line**, the same caveat that heads
  `docs/bugreport_2026-08-21_merged.md`.
- "Shipped, unmeasured" and "shipped, measured" are different states and are
  labelled differently.
- "Absent from this document" and "considered and rejected" are different
  states. Only the first should ever be re-proposed.

---

## 1. Coverage, instrumentation and the shim

**Edge coverage and map sizing.** ASLR instability fix for CTX-sensitive
builds; `-fno-omit-frame-pointer` across vendored libraries; map sizing from
ELF symbol scanning for context width; hit-count bucketing on the SHM path as
a separate `uint8` dense array; the `edge_id |= 1` fix; generation-tagged SHM
reset. Probe-window bound shipped as `__AFL_PROBE_MAX=64`, with measured drop
rates in `afl_shim.c` against `ffmpeg_read` at load 0.77 (window 16 → 1.60%,
window 64 → 0.04%).

**n-gram coverage.** Complete, no gaps: compat branch at `k=2`, ring + FNV-1a
for `k>2`, `_Static_assert(4096)`, the `__afl_ngram_k_N` symbol,
`detect_ngram_k` / `ngram_inflation_factor` / `MapSizeEstimate.ngram_k` in
`elf.py`, the ptrace twin, ~16 `_ng2`/`_ng3` targets in `build_targets.sh`.
The resume guard lives as `current_coverage_contract()` /
`check_coverage_contract()` in `corpus_manager.py` and covers `ngram_k` and
`node_channel` together.

**K-Scheduler / Katz centrality.** Complete: `icfg.py`, `horizon.py`,
`katz.py`, `katz_channel.py`, the node-bitmap channel (`node_idx` packed into
the distance-table entry, eager write in the same probe), the `katz` arm in
`seed_picker` and `SeedScorer.SCHEDULES`, all three W3 transforms
(connectivity-preserving visited-node deletion, Tarjan DAG, seed adjunction)
and non-uniform β, with persisted state. 72 tests across 8 files.
**Calibration numbers were moved into `DEEP_DIVE.md` before the plan document
was deleted** — 1.91% median vs 4.21% mean, all three ablations above the
headline, Kendall tau 0.01–0.09 and negative on two targets, and the point
that `CTX_SENSITIVE` saturates `V` first. Without those, someone promotes the
arm to default or removes a `horizon.py` step.

**Forkserver.** Real forkserver in the `afl_shim.c` constructor — 5.27×
speedup on light targets — with the `READY <mode>` protocol suffix for
observability. `cmplog_shim.c` merged into `afl_shim.c` behind build-mode
gates. `libc_shm.py` as the shared binding module for a correct `shmat`
restype.

**cmplog comparison counters.** Per-callback counters, forkserver reset,
dense reward, wall detection, visibility checking, per-PC-site counters — all
four opportunities from that thread closed. One follow-up remains open (P0-3
in the pending document).

**Region liveness (skittercreek item B).** `LiveBitMaskEstimator` calibrated
against a synthetic target with ground truth. Measured with `--unstable 0`,
400 blocks, fanout 32, 900 samples per region, deterministic across 8 reruns:
dead region `[32,96)` moves coverage 0/900; live prefix `[0,32)` 900/900 with
its first edge on sample 1. `DEAD` verdict in the dead region at every
`switch_after` in {50,100,200,400,800}, never in the live region. **Zero false
negatives, zero false positives.** Neither `_LIVENESS_SWITCH_AFTER=200` nor
`_LIVENESS_DEAD_WEIGHT=0.1` changed — what justifies keeping 200 is the case
the synthetic target *cannot* exhibit (a cold-but-live region with a long
leading-zero run), so the dead side is now measured and the cold-live floor is
a *stated* assumption rather than an unexamined one. Tooling:
`tools/sweep_liveness_thresholds.py --synthetic-target`, writeup in
`docs/sweeps/synthetic_liveness_calibration_2026-08-29.md`.

**ASLR discipline for calibration harnesses.** `services/fuzzer.py` calls
`disable_aslr()` at startup and `personality(ADDR_NO_RANDOMIZE)` is inherited
across fork *and* execve, so every target the fuzzer runs has ASLR off. A bare
`subprocess.run` harness therefore characterises a target the fuzzer never
executes. Measured on the same target and seed: ASLR on → 1–3 unstable edges
in 30 runs, 56–66/120 dead-region mutations "move" coverage, verdict LIVE
(wrong); ASLR off → 0 unstable edges, 0/120, verdict DEAD (correct). The sweep
now sets the personality **in the forked child via `preexec_fn`**, because
`personality()` is process-global and irreversible — calling it in the parent
leaked ASLR-off into the pytest process and silently *skipped* two tests in
another file depending on collection order.

---

## 2. Schedulers, seed selection and energy

**Edge-distribution signal repair** (four defects, all measured before and
after):

1. `_cull_queue` ordered edges with `rare_edge_count(e)`, which expects a
   seed key and therefore returned 0 for every edge — AFL's set-cover rarity
   ordering was dead. Now orders by `edge_owner_count(e)` ascending with edge
   id as tiebreak, via a new public accessor on `EdgeTracker` that separates
   the per-edge key space from the per-seed one.
2. `_weight_edge_penalties` used `_global_edge_hits` (bucketed execution
   volume) as a rarity proxy instead of `_edge_owner_count`; `rare_count` and
   `gap_score` were identical so the bonus applied twice, multiplicatively and
   uncapped; and `mean_hits > 3` rewarded *hot* edges. Now rarity is over
   `_edge_owner_count`, the log2 bonus applies once, and the mean-hits term
   became a crowding penalty. Named constants `RARE_EDGE_OWNERS`,
   `RARE_EDGE_GAIN`, `CROWDED_EDGE_OWNERS`. Measured: a seed owning 4 loop
   edges 35.7 → 2.16; 3 edges shared by 8 seeds 1.50 → 1.00; 20 rare edges
   77.0 → 3.20; 3 edges shared by 30 seeds 3.70 → 0.50.
3. `good_turing_estimate` built its frequency spectrum from
   `_global_edge_hits` — the wrong sampling model, and the damping/caps were
   patches for it. Rewritten as **incidence-based Chao2** with a
   bias-corrected branch under `Q2<10`, Chao 1987 variance, log-transformed
   95% interval and Chao & Jost sample coverage. New keys
   `m`/`chao2`/`ci_low`/`ci_high`/`sample_coverage`/`discovery_probability`;
   old keys left intact for `crash_eta`.
4. Edge ids are `prev_loc ^ cur_loc` — a hash with no metric structure — so
   Wasserstein over the edge index and `compute_coverage_proximity` measured
   nothing about the program. Measured: proximity returned exactly 1.0 for all
   30 test seeds and the Wasserstein weight spanned only [1.10, 1.34]. The
   ground metric moved to the `log2(1 + hit count)` axis, and proximity was
   redefined as the fraction of edges discovered in the last quarter of the
   coverage clock (using `_edge_first_seen`, which had no consumers). After:
   proximity spans 0.0–0.966; the Wasserstein weight separates loop-bearing
   seeds (mean 1.417) from flat ones (mean 0.676).
   **Recorded so it is not re-proposed:** trace-pc-guard guards *are*
   sequential integers, but they are folded with `prev_loc` before reaching
   Python, so **no** execution path — not `--clang-scov`, not ptrace — has an
   ordered edge axis. "Gate it behind scov" was wrong.

**`_edge_owner_count` invariants.** Three defects, all falsified before fixing:
`from_dict` rebuilt the attribute as a bare dict, so the defaultdict invariant
held only until the first state restore; bare subscripting of a defaultdict
*inserts*, so a documented read-only accessor was growing the map on every
call; and `_maybe_prune` evicted seeds from `seed_edges` and eight companion
maps but never adjusted `_edge_owner_count`, so counts only rose and kept
crediting edges to seeds that no longer exist (measured: 12 owners against 5
real). The last one was split into its own commit because it changes energy
allocation. Tests: `TestEdgeOwnerCountInvariants` (6 cases, all six verified
failing pre-fix).
Microbenchmark kept because it argues against the obvious follow-up: bare dict
+ `.get(e,0)` 117.4 µs, defaultdict + `m[e]` 111.3 µs, bare dict + `m[e]`
102.4 µs. The defaultdict buys ~5%; bare dict with guaranteed key presence at
the write sites buys ~13% *and* has no insert-on-read semantics.

**Seed-picker hot path.** Profiled against ffmpeg (1500 execs,
`ffmpeg_read_nosan.so` in-process): `_pick_seed` was 51% of runtime, SHM scans
5.8%, actual ffmpeg execution 3% — 91 ms per pick against 0.8 ms per
execution. Two fixes: the overlap sum at `seed_picker.py` (26% of total on its
own, 802,763 calls, 12.69 s) was fused into the owner-count loop with a
`Counter` built once per pass; and `shm.py`'s `_active_columns` /
`_active_edge_ids` (each masking a 262,144-entry table, ~1,800 live = 0.7%
occupancy, once each per execution) were rewritten as a sparse-first `_scan`
with `flatnonzero` and a memo on `(edge_count, path_hash, generation,
num_entries)`. Measured at fixed seed, 1500 execs: **230.2 s → 70.6 s
(3.26×)**; weights pass 71.4 → 11.6 ms; SHM scan 1234 → 707 µs.
Equivalence was proved *at function level* (full `_compute_weights` vectors
against a verbatim copy of the old implementation, frozen clock, 8 random
corpora, 0 discrepancies) — an end-to-end A/B does **not** reproduce byte for
byte, because `age = now - added_at` means a faster run reaches a different
clock state.

**Saturation gate.** The ≥99% gate that collapses subsumption / diversity /
Wasserstein / proximity to neutral multipliers used to latch: it was
invalidated only on discovering a new edge, which is exactly what the gate
makes less likely, and Chao2 returns 1.0 for *any* plateau (measured: 60 seeds
over a closed 500-edge universe → 1.0; a single starved seed → 1.0). Now
refreshed every `SATURATION_REFRESH_EXECS=2000`, with a stall override at
`SATURATION_STALL_EXECS=20000`, and `_cached_weights` flushed on every flip.

**Scheduler→mutator reach.** Two gaps closed: `Hierarchical` silently dropped
every operator registered at runtime via `REGISTRY.register_mutator()` because
`_op_to_cat` was seeded from the import-time `OPERATOR_CATEGORIES` snapshot
(measured 0 pulls out of 60,000), while GP-UCB gave it an all-zero feature
vector, which under the RBF means ~0.61 similarity to *every* category rather
than none; and `cmaes` was missing from the `available` list in the Elo ballot
in `services/operators.py::select_op`, so its dispatch branch was dead code
while CMA-ES registered arms and received `record()` without ever selecting.
Fixed with `category_of()` / `refresh()` / `UNCATEGORIZED` in
`core/operator_categories.py`, a memoised `_cat_for()` in Hierarchical,
`_ensure_category()` with padding in GP-UCB, and `cmaes` in the ballot.

**Bandit convergence harness** (`tests/support/bandit_env.py`) — found and
fixed bugs in MOpt, GP-UCB, Hierarchical and CMA-ES; cost-aware reward wiring
across all schedulers; LinUCB contextual bandit completed.

**Minimax / adversarial search.** All five phases are in the tree, verified by
inspection: `AlphaBetaMCTSSeedScheduler` in `core/schedulers/mcts.py` with
`_pick_mcts_seed()` integration; `risk_matrix` in `core/elo.py:229`;
`select_op_minimax` in `core/schedulers/monte_carlo.py:1776`; minimax framing
in `core/cond_stmt.py` and `core/smt_solver.py`; `core/rate_distortion.py` +
`corpus_manager.py` for robust corpus admission.
**Correction:** the source document claimed all five landed in `b49441b`, and
that hash does not resolve in the current history. The code is present; the
commit reference was stale. The A/B against the plain `MCTSSeedScheduler` is
still owed (pending document, §E).
Note also that the *research* companion opened with "the file where all the
percolation primitives live is `core/minimax.py`" — a copy-paste from the
percolation document. `core/minimax.py` never existed and does not need to;
the work landed in the five files above.

**Boltzmann cost energy, A/B — RUN AND CLOSED.** The debt from `ab07835`
(`_pick_boltzmann_seed` reading `effective_fuzz_count` instead of
`fuzz_count`, never benchmarked) is paid. Result under the single-process
lock, 10 seeds × 3 replicates, `direct_lite`, `-m 65536`, 10k execs,
in-process, arms paired in time:

| target | W/L/T | median | McNemar p |
|---|---|---|---|
| `png_read.so` | 3/6/1 | −2.0 edges | 0.508 |
| `jpeg_read.so` | 2/5/3 | −1.5 edges | 0.453 |
| pooled | 5/11/4 | — | 0.210 |
| `grep_read.so` (20 seeds) | 12/7/1 | +3.0 edges | 0.359 |

**This is a bounded null, not a bare null.** With the measured within-cell
dispersion (sd ≈ 4.6 edges png, 4.7 jpeg, 12.3 grep) the design had ~100%
power against a 10-edge effect, ~77% against 5 and 15% against 2. So the
change moves edge discovery by less than ~5 edges in either direction. To
resolve less than that you need **more replicates, not more seeds** — the
noise is within-cell. Direction leaned consistently *against* the change on
png and jpeg and slightly for it on grep; nothing supports a sign.
Full writeup: `docs/learnings/2026-08-30-boltzmann-ab-result.md`.

---

## 3. Mutation operators

Registry now carries **163 operators** (`REGISTRY.names()`). `_CATEGORIES`
carries 156, and the difference is exactly the seven self-registering
`MutatorBase` classes — `ff_png`, `ff_zip`, `ff_isobmff`, `ff_jpeg`,
`fractal_voronoi`, `weizz_chunk_mutate`, `weizz_field_mutate`. `_CATEGORIES`
is a strict subset of the registry (the reverse difference is empty). That is
the documented design, not drift. **163 is the number to cite.**

**Shipped this cycle:** `bit_rotate`, `bit_shift`, `span_invert`, `bit_repack`
(sniffer-gated); 13 structured regularity operators, each the inverse of a
*named* statistical test from the dieharder battery in `randomness.py`
(`fibonacci_pairs`, `monotone_fill`, `de_bruijn_fill`, `rank_deficient`,
`spectral_peak`, …, plus `cycle_lock`); `span_reverse` and `span_relocate`
(the TSP 2-opt / Or-opt neighbourhood — verified absent beforehand: of 157
operators, zero reversed a span or relocated one without changing length);
`_op_region_shuffle`; AVIF and SQLite format mutators completed and registered;
ogg/flv/asf/riff completed earlier in the same shape.

**Adaptive havoc sub-operator weighting** with inverse-CDF sampling;
`--reseed-on-stall`; coverage-guided mode on by default (`--no-coverage` to
opt out); a bloom filter for duplicate-execution dedup; PerfFuzz-style
per-edge max hit count (`_max_counts`); the timing-contamination fix (moving
`t_start` below mutation cost).

**`_swap_pair(domain, rng, *, start=0)`** in `core/mutations/generic.py`
deduplicates the C(n,2) swap idiom across **16** format mutators (adts, arm,
asf, avif, isobmff, mp3, mpegts, nal, pgs, protobuf, riff, sqlite, webm, webp,
x86, zip). 20 regression tests in `tests/test_swap_pair.py`. The idiom was not
textually identical across all call sites — three distinct shapes existed,
which is what fixed the helper's signature.

**`_swap_tuple(domain, rng, m, *, start=0)`** is implemented, with the even-m
rotation and odd-m parity-trap fallback. It has **zero production call sites**
— see P2-2 in the pending document.
The parity trap is the finding worth keeping: derangements of 3 are exactly
the 3-cycles, which are *even* permutations, so a pure 3-cycle operator reaches
only `A_n` — BFS on the Cayley graph confirms 360/720 (n=6), 2520/5040 (n=7),
20160/40320 (n=8). **k=1 alone is a regression, not an extension**; m=4 does
generate all of `S_n`. Hit rate on a specific pair does *not* collapse:
`P = [m(m−1)/(n(n−1))]·[!(m−2)/!m]`, plateauing near 0.5 of the m=2 rate and
independent of n. The real payoff is Cayley **diameter** (m=2 → n−1; m=3 →
3,3,4; m=4 → 2,3,3; m=5 → 2,2,2), which under coverage-gated corpus admission
is the number of *accepted* entries needed, not executions.

**Generator `max_len` was positional-argument shadowed in ten generators**,
not three as the source note claimed. All ten carried a vestigial first
parameter (`_chunks`/`_doc`/`_boxes`/`_units`/`_fields`/`_nodes`/`_segments`/
`_elements`/`_insns`/`_words`) copied from bmp/gzip/jpeg/zlib, where the slot
*is* a real overload the body decodes. Nobody passed or read it in the ten, so
every `mutate()` call landed the cap in the placeholder and the generator used
its own default. Measured with `max_len=64`: webp 92 bytes, zip 116, isobmff
137, gif 131, webm 104, pgs 79, nal 67. It reached real output —
`WebpMutator().mutate(b"xx", max_len=32)` returned 92 bytes — because on the
unparseable-input branch `mutate()` returns the generator result directly.
Fixed by converting 11 call sites to keyword and adding bmp's `isinstance(...,
int)` coercion to all ten signatures. `TestGeneratorMaxLenIsHonoured` (24
cases, 9 failing pre-fix).

**`RandPool.sample` Floyd's algorithm** for `k>=3`, audited clean: uniform over
k-subsets (χ²=20.9, df=19), no duplicates or out-of-range up to n=65536 and
k=32. Measured 8.6× at n=256 k=3, 9.1× at n=65536 k=3, ~1.5–1.8× at k=32 — a
*constant-factor* win from dropping the numpy call, not asymptotic; timings
are flat in n, so `Generator.choice` never materialised the population.
**The `k==1` and `k==2` fast paths were deliberately preserved**: a dozen
mutators call `rng.sample(range(len(x)), 2)` and changing them would move
byte-for-byte output of every seeded run.

**`_op_region_shuffle` and the fabricated-offset fix.** `f._last_mutation_offset
= byte_idx` was assigned blanketly from `select_position()` *before* the
operator ran, so operators that ignore `byte_idx` entirely (`chunk_shuffle`,
`byte_shuffle`, `block_shuffle_variable`, `token_shuffle`) fed the liveness
estimator a fabricated offset. Now `None if op in _DELOCALISED_OPS else
byte_idx`. Paired with a new region-confined shuffle that reports a true
offset. Ground-truth target `targets/order_sensitivity.c` plus
`tests/test_region_order_attribution.py`; sweep in
`docs/sweeps/region_order_attribution_2026-09-04.md`.

**Earlier operator work:** `crc_learn` no-op fix (the XOR-only model was never
consumed), `xor_map_solver` endianness bug, the dieharder battery port
(`randomness.py`), integer-modulus checksum recovery (Adler-32,
Fletcher-16/32), the Angora-port operators (`magic_byte_search`, `climb_hill`,
`gradient_descent` window bug), the LZ4 vendored target.

**`core/debruijn_cache.py`** — the per-process de Bruijn construction cache
(§10f of the combinatorics survey).

---

## 4. Structure-aware generation

**Weizz tag map (P1–P5).** `core/weizz_tags.py` + `tests/test_weizz_tags.py`
(10/10). **No new comparison tracer** — tags consume the existing
`CmplogCollector.pairs` (plus optional `_pair_pc`) and optional colorization
taint regions. API: `TagCollectorConfig`, `build_tag_map_from_cmplog`,
`collect_structure_map`, `attach_tags_to_meta`, `load_tags_from_meta`.
Operators live: `weizz_chunk_delete`, `weizz_chunk_dup`, `weizz_chunk_swap`,
`weizz_field_havoc`, `weizz_tag_repair`, `weizz_field_mutate`,
`weizz_chunk_mutate`. P4 (tag-restricted surgical solve) is
`_weizz_restricted_find` in `_op_condstmt_solve`
(`services/operators.py:2059`). **The only unticked item on the acceptance
checklist is the paired bench** — an evaluation run, not a code change; filed
in the pending document under §E.

**FormatFuzzer, phase 1.** `core/mutations/formatfuzzer.py`, four
self-registering `MutatorBase` wrappers (`ff_png`, `ff_zip`, `ff_isobmff`,
`ff_jpeg`), the `formatfuzzer_enabled` gate on `MutationContext`, and CLI flags
`--formatfuzzer` / `--ff-bin-dir` / `--ff-templates`
(`cli/commands.py:3009`). Smoke tests present. Phases 2 (decision-seed mode)
and 3 (hardening + the paired 24 h run) are open.
**Design question raised and deliberately left open:** the four `ff_*`
mutators register **unconditionally at import** from the bottom of
`operator_registry.py`, even though the feature is gated behind
`--formatfuzzer`. Both import orders were checked and the snapshot matches the
registry today, so the hazard documented in `operator_categories.py`'s
docstring (snapshot taken before a `register_mutator` → Hierarchical drops the
arm, measured 0 pulls of 60,000) is **not** firing — but eager registration of
a gated feature is exactly the shape that produces it.

**Fractal jittered Voronoi.** `core/mutations/fractal_voronoi.py` shipped and
registered (Approach A — spatial meta-mutation operator: map the buffer to a 2D
grid, partition into Voronoi cells, assign each cell a sub-operator by its
root's hash, blend at boundaries). `core/parallel_fractal_partition.py` also
exists. Geometry caches were later moved onto the instance (`6b3b7c3`) after
`_nearest_site` sweeping 5×5 per byte measured at 36% of a campaign.
Approaches B (fractal coverage-space seed prioritisation) and C are open;
the A/B campaign is owed.

---

## 5. Analysis, diagnostics and modelling

**Percolation — Modules 1, 2 and 4 are live and consumed.**
Module 1 bootstrap-percolation corpus minimisation and
`bootstrap_minimize_corpus` (extending the greedy set cover at
`services/minimize.py:149` with iterative k-rigid-core reduction); Module 2
coverage phase-transition detection, wired into the main loop; Module 4
`invasion_select` (`services/seed_picker.py:126`), which *is* on the Elo ballot
and in `_OPERATOR_STRATEGY_NAMES` — the `cmaes` omission was not repeated —
with `tests/test_invasion_select.py` and `tests/test_invasion_elo_integration.py`.

**Module 3 is a special case and is filed as open, not done.**
`core/target_difficulty.py` exists (224 lines,
`estimate_isoperimetric_profile` / `estimate_percolation_threshold` /
`estimate_growth_curve`) and has tests, but **no production consumer** — only
`tests/test_target_difficulty.py` imports it. Its own document claims it is
"called at fuzzer startup… drives time budget, initial corpus size, and
operator preselection". That wiring was never done. See P2-1.

Literature note kept because it changes the design: Diskin, Easo,
Radhakrishnan, Sudakov & Tassion, *"Supercritical sharpness of percolation"*
(arXiv:2603.03257) proves supercritical sharpness for **every** infinite
transitive graph, stated purely in terms of the isoperimetric function
`Φ(n)`, with no assumption on degree distribution. That answers the
heavy-tailed-degree falsifier directly and means Module 3 should target
estimating `Φ` rather than anything else.

**GARCH(1,1) — implemented and wired, opt-in.** `core/garch.py` (440 lines,
`OnlineGarch11`), consumed by `core/coverage_regime.py` via a `_garch_spike()`
reason string, persisted through `save`/`load`, behind `--garch`
(`services/fuzzer.py:1866,1885`).
**The implementation took the audit's decisive correction.** The proposed test
would have been invalid as written: `discovery_rate()` is a **sliding window of
5 snapshots**, so consecutive samples share 4 of 5 and it acts as an MA(4)
smoother. Measured — a *constant-rate* Poisson process (zero clustering by
construction) pushed through the live function gives squared-residual ACF at
lags 1–6 of +0.549 +0.225 +0.047 +0.012 +0.008 +0.005 and Ljung–Box(10) =
1418.9 against χ²₀.₉₅(10)=18.3, a rejection by 77×; the unwindowed control on
the same process gives +0.026 and LB 15.3, correctly failing to reject. An MLE
GARCH(1,1) on the windowed pure-noise series returns α = 0.54.
The shipped code instead feeds the **non-overlapping** `delta = current_edges −
self._last_allan_edge_count` series at `services/fuzzer.py:6267`, one sample
per tick — the same series Allan already consumes. That is the right fix and
it needed no new plumbing.
Secondary discriminator, still useful: the window artifact cuts off at lag 4 by
construction, while real GARCH persistence decays geometrically past it.
Precedent worth citing that the original document did not: `cmplog.py:372`
already runs a fast/slow EWMA pair per callback for the wall detector — the
same O(1) recursive update, applied to the level rather than the squared
residual. And `critical_slowing.py` already learned the inverse lesson, keeping
`_raw_history` separate from `_history` precisely because KF smoothing inflates
lag-1 autocorrelation.

**Navier–Stokes continuum diagnostics — implemented and wired, opt-in.**
`core/navier_stokes.py` (253 lines, `ContinuumField`), behind `--continuum`,
feeding `invasion_select(op_stats, flux_map=flux_map)` at
`services/operators.py:3647` and `coverage_regime.py`.
**Scoped as steady diagnostics with no time-stepping — which is what the audit
recommended, for a reason worth preserving.** Tao's 2014 construction (finite-
time blowup for an *averaged* 3D Navier–Stokes) builds a `B̃` that preserves the
cancellation law `⟨B̃(u,u),u⟩=0` — hence the energy identity — and essentially
every function-space upper bound the true `B` satisfies, and still admits
solutions that blow up in finite time. Consequences here: (a) qualitative
behaviour is **not** inherited through averaging, so "start with a surrogate
flux and refine to real finite volumes later" is not a free continuity
hypothesis — what is measured on the surrogate says nothing about the refined
model or the reverse; (b) the graph mapping preserves *neither* invariant —
there is no Leray projection on the horizon graph, because the sparse linear
solve that would impose `∇·u=0` was explicitly declined, so there is no
cancellation law at all. Tao *keeps* the identity and still blows up. Expect
state divergence under explicit time-stepping and do **not** read it as a
discretisation bug; and note that once you clamp the "physics" it does nothing a
bounded heuristic would not. **The gradient of a scalar field is safe; the
advective step is what inherits the blowup risk.** That is why the shipped
version is steady-only.
What *does* transfer, and is testable without any fluid model: Tao's blowup
mechanism is a von Neumann machine that replicates at finer scale after a
**delay**, and the delay is load-bearing — Katz–Pavlovic fails for the opposite
reason, energy reaching high modes prematurely and being dissipated before the
singularity. Translated: energy leaked to the next frontier too early is
dissipated before it pays; saturate a scale, then dump. That is a claim about
power schedules and stall recovery, and the crude version already exists as
`ops._havoc_energy_scale *= 1.5` in SUBCRITICAL.
Cost caveat inherited from the analogue: `KatzChannel.ensure_scores` needed
`_RECOMPUTE_MIN_INTERVAL = 50` **plus** a self-amortising clock gate
(`_MAX_RECOMPUTE_OVERHEAD = 0.05`, `_COST_GATE_FLOOR = 0.005`) because
`build_horizon_graph` is a per-U BFS through V and costs *more* early in a
campaign. An exec-interval gate alone is not enough.
Feasibility caveat: the graph is conditional. `KatzChannel.build` returns
`None` unless the target has trace-pc, `TargetDistance.load()` works, target
functions resolve, the ICFG is non-empty and the probe_key→node table is
non-empty — and it is disabled entirely in directed mode. So any field needs
defined behaviour with no graph; the graph-free inputs (rarity, inverse
coverage density) are the honest first target.

**QEA as a Hilbert-space object.** Analysis stands: each bit is `(α, β)` with
only `α` stored and no phase term, so every bit sits on the real non-negative
quarter of the unit circle — a mean-field product state, not a point in a
genuine `2ⁿ`-dimensional register, and therefore structurally unable to
represent cross-bit correlation. Two things landed on top: `rotation_gate` was
fixed to perform a true angular rotation rather than a linear walk (`2edbce1`,
independently); and **intra-byte coupling** shipped (`coupling` tensor on
`QEAIndividual`, `collapse_correlated()` / `update_couplings()`, serialised in
`to_dict`/`from_dict`), plus opt-in algorithmic cooling (Δθ decay).
Stated plainly and kept: the coupling reaches **within a byte only**, and it
was implemented on explicit request, not because a benchmark showed a gap.

**Persistence Mechanics (external paper, evaluated).** One item ported; four
were already in the tree under other names and are recorded so they are not
re-proposed; nothing ported from the demo scripts. The gating measurement was
run — the ledger is **not** clustered — so the item survived, and §1a/§1b are
implemented on top of a new `core/cost_ledger.py`
(`docs/learnings/2026-08-29-per-seed-cost-ledger.md`,
`tests/test_cost_ledger_consumers.py`). §1c is **deliberately not written**:
its host function no longer runs. The successor question — whether cumulative
execution cost should enter the eviction ordering — was promoted to its own
open item and lives in `docs/TODO.md`, not here.

---

## 6. Similarity, diff and crash clustering

**A1 — the Levenshtein DP was an OOM, not a slowdown.**
`core/similarity.py::_levenshtein_align_numpy` allocated `dp = np.empty((n+1,
m+1), int32)` for traceback. Measured: n=512 → 1.1 ms / 1.1 MB; n=2048 → 74.5
ms / 16.8 MB; n=8192 → 1093.6 ms / 268.6 MB; n=16384 → 3941.0 ms / 1074 MB. A
64 KiB crash against a 64 KiB seed needs **17 GB**. `adapters/filesystem.py`
already had a 512-byte cap with the comment naming the O(n·m) cost;
`core/root_cause.py` and `core/crash_metadata.py` had none — and they are the
two that see input-controlled sizes. The cost was understood at one call site
and never generalised.

Shipped design, after two corrections:

- Myers O(ND) alone was **not** the answer. With dissimilar inputs Myers loses
  badly (measured 25–100× *slower* than the numpy DP; a random 4 KiB pair
  reached D=7224 and took 7.4 s against 97 ms). The correct cost model is
  **~O(D² + N)**, not O(N·D) — verified by 253 ms at n=65536 with D=652 (long
  snakes, cheap) against 7.4 s at n=4096 with D=7224 (no snakes, 52M diagonal
  steps).
- So the dispatch is **by affordability, not by flag and not by a bound scaled
  with N**: if the DP table fits the byte budget → DP (behaviour identical to
  before for every input that was already survivable); else Myers bounded at
  D=2048 (a constant, because D fixes the cost, not N); else
  `_coarse_block_diff` — never an allocation. Budget default raised 64 → 128
  MiB, because at 64 MiB a dissimilar 4 KiB pair asked for 67.1 MB, just over,
  and fell to Myers at 4.7× the cost.
- `--diff-myers` was **removed**; `--diff-myers-max-d` and
  `--diff-myers-max-bytes` remain as knobs. `configure_diff_myers(enabled=…)`
  became `configure_diff_limits(max_d, max_bytes)`.

Final numbers, median of 5 interleaved in one process. Dissimilar (must not
regress): n=512 2.8 ms vs DP 2.7; n=4096 79.0 vs 87.3. Similar (the case that
died): n=8192 4.0 ms vs 344.6 = **86.5×**; n=16384 7.6 vs 1267.4 = **167.3×**;
n=65536 63.7 ms against a 17.2 GB table that simply dies. All round-trip.
`_coarse_block_diff` was verified over 500 random pairs, 0 round-trip failures,
**before** being made reachable by default — which answers the open question of
whether a coarse fallback can honour the replay contract. It can.

A `_myers_backtrack` bug was found and fixed in the process: it returned edit
scripts that do not reconstruct the target. The tell was that the **op count
went down** as edit distance went up (~35 ops lost per extra edit). Cause: it
unwound snakes with a greedy `while x>0 and y>0 and a[x-1]==b[y-1]` instead of
bounding them by the previous endpoint, so at level d it ate match runs
belonging to lower levels. Real impact: `root_cause` replays these scripts
**positionally** in delta debugging, so a truncated script rebuilds a
non-crashing candidate and fails as "minimisation did not reproduce", not as a
diff bug. Fixed by bounding the snake at `(prev_x, prev_y)`, one edit per
level, plus the `d==0` tail and the pure-prefix insert/delete runs the old
version never emitted. Myers minimises INDEL distance while the DP minimises
Levenshtein, so `_coalesce_indel_pairs` folds adjacent delete+insert into a
`replace` — **honestly: the fold is not always possible** (substitution costs 2
for Myers, 1 for the DP), so a Myers script can carry 1–2 ops *more* than
optimal, measured in 12 of 400 cases. Never fewer, which would be an invalid
script; the test asserts that direction.

**A2 — `cluster_crashes` was quadratic, and the first fix was wrong.**
Measured with 5 well-separated synthetic families: n=100 3.7 s; n=200 16.1 s;
n=400 59.3 s (79,800 similarities), reachable from `services/report.py`.
Cluster **quality** was fine — 5/5 families recovered at every size; there was
no chaining defect to report (an early "clusters=1" reading was an artifact of
a generator drawing 6 frames from a pool of 8, not of the algorithm).

The MinHash-LSH sparsifier that shipped first lost ~80% of the pairs it should
have joined: against the dense implementation as oracle, 83 of 360
configurations differed and LSH **always** under-clustered, often returning n
clusters. Direct recall probe: 20 signatures with 5 above-threshold pairs by
ground truth — dense found 5, LSH found 1, **recall ≈ 20%**.
**The cause is a metric mismatch, not tuning.** MinHash LSH approximates
**Jaccard** over token sets; the clustering threshold is on **Levenshtein**
similarity. A pair can sit at Levenshtein 0.8 with a Jaccard far below the
banding threshold and never become a candidate. No banding parameter fixes
that, because the two metrics do not order pairs the same way. Second,
independent defect found on removal: the banding fed the builtin `hash()`,
which is per-process randomised, so the candidate set was not reproducible
across runs either.

Shipped instead: **exact bounds, no flag, nothing to opt into.** Both metrics
have the form `1 − dist/max_len`, so a pair passes only if `dist ≤ (1 −
threshold)·max_len`, giving two sound lower bounds on `dist` — (a) `dist ≥
|len_a − len_b|`, so the shorter side must be at least `threshold·len_long`,
enforced by a window over length order with a **monotone** pointer; and (b)
`dist ≥ max_len − |bag_a ∩ bag_b|`, so the multiset intersection must reach
`threshold·max_len` — plus a single-linkage skip (`if find(i)==find(j):
continue`). These prune only pairs that provably cannot pass, so the clusters
are **identical** to all-pairs. Verified over **1080 configurations against the
dense oracle (n up to 25, thresholds 0.5/0.7/0.85, frame_lists
none/all/partial/empty) with zero discrepancies**, plus a clean self-control.
Cost, with the signature-string metric and identical clusters throughout:
n=100 4.34 → 0.094 s (46×); n=200 17.30 → 0.203 (85×); n=400 69.09 → 0.516
(**134×**).
**Measurement caveat worth keeping:** with `frame_lists` supplied the metric is
token-level over 6 tokens and is very cheap (n=400 only 1.24 s dense, 11×) —
the expensive case, and the one to cite, is the signature-string path.
`normalize_frame` was hoisted out of the pair loop as an *exact* transformation
(`crash_signature_similarity(a,b)` is by definition
`levenshtein_similarity(normalize_frame(a).encode(),
normalize_frame(b).encode())`, and `frame_sequence_similarity` already
truncates to `frames[:8]`), verified over 3000 cases with 0 discrepancies.
CLI: `--crash-cluster-lsh` / `--crash-cluster-lsh-threshold` →
`--crash-cluster-threshold`.
`tests/test_regression_crash_cluster_exact.py` (16 tests) carries **its own
copy of the dense oracle** rather than importing it, so touching production
cannot move the oracle too. Falsified twice: 10 of 16 fail against the LSH
implementation forced on, and 6 of 16 against a deliberately unsound
multiset bound (+1).

**Union-find in the same site** kept the union-by-rank that arrived with the
LSH commit; path halving was already there.

---

## 7. Crash handling, corpus and state

**`save_crash` did the expensive work before the cheap rejection.**
`CorpusManager.save_crash` built the entire sidecar — `find_nearest_corpus`
over the whole corpus, target hash, GDB replay — and only then called
`filesystem.save_crash`, which discards the crash if the hash or signature was
already seen. Measured: `find_nearest_corpus` 250 ms with 1000 seeds of 4 KiB
and **2.2 s** with 2000 of 16 KiB; `CrashTracer()` forks two `which` calls
(2.6 ms) even with no gdb present. The docstring's "~1 s on the rare input that
crashes" stops being true exactly when the fuzzer finds its first bug. Fixed
with a pure `classify_crash()` in `filesystem.py` returning a `CrashVerdict`,
enrichment only when novel, and the verdict passed back to `save_crash` so the
report is parsed once instead of twice.

**Crashing seeds grew without bound.** Every crashing input was written to
`seeds/crashing/` marked irreplaceable. Now capped at
`CRASHING_SEEDS_PER_SIG=64` per signature, keyed by `matched_signature or
signature` — because ASAN signatures are grouped by 0.8 similarity, and
without that key every fuzzy match read count 0 and stayed exempt (measured:
two signatures differing only in the inner frame score 0.818 and merge).

**cmplog artifacts were never cleaned up.** `CmplogCollector.stop()` was called
from nowhere in the package (only from `tests/test_cmplog.py`), and
`_cleanup_stale_cmplog_files()` likewise — two mechanisms, neither wired. Each
run mints a fresh uuid, so `~/.cache/fuzzer_cmplog` grew forever (within a run
files are truncated in place, which is why a single campaign never showed it).
The sweep also missed `.sites`, the largest of the three. `stop()` now runs in
`run()`'s finally; the sweep runs in `start()` **with a minimum age**
(`_CMPLOG_STALE_AGE_S = 24 h`) — which is probably why it was never wired:
parallel workers are separate processes over the same directory and names carry
a uuid, not a pid, so an unconditional startup sweep deletes another worker's
live log.

**`ChecksumLearner` threw away its evidence on every resume.** `to_dict` wrote
`pair_count`, which `from_dict` never read — the only `to_dict`/`from_dict`
asymmetry in the tree, verified mechanically across all 11 classes. `_pairs`
was not serialised and recovery needs `min_pairs=64`, so a run that stopped and
resumed often never reached the threshold. Now persists `_pairs` +
`_total_pairs_seen` + `_pairs_attempted_at` — **both counters or neither**,
because restoring the total without the attempt marker would make the next
`add_pairs()` look like a full `RECOVERY_RETRY_BATCH` and immediately re-run
GCD/BM. Budget is in **bytes** (`CHECKSUM_STATE_BYTES_MAX = 256 KiB`), not in
count, because the data side of a pair is the entire checksummed region (an
IDAT can be megabytes) and `CHECKSUM_PAIRS_MAX=128` bounds the count, not the
size.

**`save_state` discarded `seed_meta` for every seed over 128 bytes.**
The guard was `key = seed.hex(); if len(key) >= 256: continue`, and the key was
the hex of the **entire seed content**, not a hash. Measured on a live png
campaign: the corpus held seeds up to 2858 bytes while the longest persisted
key was 234 chars = 117 bytes. The load side read exactly what save wrote, so
for any seed over 128 bytes the whole entry was lost through `--resume` —
`fuzz_count`, `coverage_edges`, `added_at`, `momentum`, `lineage_depth`,
redqueen offsets/matches **and the cost ledger**. Fixed upstream by keying on a
16-char content hash with a legacy fallback, which also closed a second defect:
`parent_key` values *inside* each entry were already 16-char hashes, so
`services/tmin.py`'s lineage walk could never resolve for any seed over 8
bytes. The two key spaces now agree.

**Earlier fixes in this family:** generation tag wrap in
`ShmCoverage.reset_edge_map()`, ptrace-mode timeout misclassification,
`read_bitmap()` wrong SHM offset, cmplog environment leak, `run_target_fast`
blocking waitpid, `parse_dict_line` keeping AFL dictionary quotes, grammar
loader backslash handling, undefined grammar rule silent substitution, global
`np.random` state not seeded by `--seed`, differential stderr mismatch logic,
crash count inflated by non-binary sidecars, RSS measured as monotonic peak
instead of current, `import_corpus` format-selection tautology, the Lanczos
log-gamma constant, and the repeated flat-listing defect across
report/minimize/root_cause/parallel worker sync (finding #67) — the corpus walk
is now consolidated into `adapters/filesystem.discover_seed_files()`.

---

## 8. Build, targets and vendoring

**Path layout.** `FUZZ_VENDOR_ROOT` (default `~/fuzzing/vendoring/`, legacy
`vendor/` via `--in-tree-vendor`) and `FUZZ_BUILD_ROOT` (default
`~/fuzzing/builds/`, legacy `targets/` via `--in-tree-targets`), documented
consistently in `AGENTS.md`. `--in-tree-targets` now points the build root at
the same place, closing the half-and-half tree where binaries sat in
`./targets` while archives and `.sancov_stamp` stayed under `$HOME`.
`TARGETS_SRC` is assigned unconditionally (`build_targets.sh:110`), separating
the tracked `.c` sources from the build output — which is what made the
documented no-flag invocation work again.

**Library derivation.** `ffmpeg_extralibs()` reads
`EXTRALIBS-{avformat,avcodec,avutil,swresample}` from `ffbuild/config.mak`
instead of hardcoding `-llzma -lbz2`. The hardcode was a link requirement for a
feature that may be off: `CONFIG_LZMA` did not even appear in `config.mak` on
that tree, yet `-llzma` had to resolve, so any box without `liblzma-dev` lost
all five `ffmpeg_read` variants with only `WARN: failed: ffmpeg_read` and no
cause (helpers send stderr to `/dev/null`). Same shape for `-ljpeg` and
`jpeg_read`.
**The essential subtlety:** `EXTRALIBS` is a **superset** — it records what
configure detected, not what the archives reference. This tree emits `-lX11`
via `EXTRALIBS-avutil` while `libavutil.a` has **no** X11 symbol (`nm -u`
empty), so passing it through verbatim would only move the bug to
`libx11-dev`. Every `-l` is therefore probed with a trivial link and dropped if
it does not resolve. The real derived list here is `-lm -latomic -lbz2 -lz
-pthread` — no lzma, and with `-latomic`, which the hand-written list lacked.
`_opt_libs()` in `build_ffmpeg_ready.sh` does the same.

**Build observability.** `configure`/`make`/every compile append to
`$BUILD_LOG`, and `warn_failed` prints the first real error line — the jpeg
failure now reports itself as `targets/jpeg_read.c:15:10: fatal error:
'jpeglib.h' file not found`, which `/dev/null` used to swallow. And the script
now **exits non-zero** when `BUILD_FAILURES > 0`, with
`FUZZ_BUILD_TOLERATE_FAILURES=1` to opt out for local runs — a build that fails
should say so in `$?`, not only in the log.

**Staging.** The `rm -rf "$STAGE_DIR"` is gone; the stage persists with its
objects, gated by two stamps (config flags + compiler; vendored revision +
`patches/`). The source stamp hashes **contents**, not filenames.

**Assembly.** `--disable-x86asm` is no longer hardcoded: `vendor_ffmpeg.sh:50`
probes for `nasm`/`yasm` and only disables when absent, with a warning. This
matters more than it looks — the SIMD paths are a large, heavily optimised
slice of `libavcodec` and are where much of the historical memory-safety
history lives; the C fallbacks are a different code path with different bounds
behaviour. Hardcoding it shrank coverage and bug surface on *every* machine,
quietly.

**`_pick_cc()` fallback corruption.** It sent its warning through `warn()`,
which writes to stdout, and `DEFAULT_CC="$(_pick_cc)"` captured all of it —
leaving `DEFAULT_CC` as text+gcc (not a command) on any machine without clang.
Because helpers discard stderr, the only symptom was "objects failed" and
missing targets.

**Targets.** SQLite vendored (`tools/vendor_sqlite.sh`, amalgamation 3.53.4,
years walked newest-to-oldest because the year is not derivable from the
version, `--url=` for mirrors) + `targets/sqlite_read.c`, `.so` only.
**Key decision: no mode byte**, because the `sqlite_chunk_mutate` sniffer is
`len(d)>=100 and d[:16]==MAGIC` — a prefix would shift the magic to offset 1
and the whole corpus would be mutated as flat bytes with no visible symptom.
DB path: deserialize + `PRAGMA integrity_check(4)` + `sqlite_master` + scans
reading every column (without reading a column the record decoder never runs);
sandbox `:memory:`, DEFENSIVE, `TRUSTED_SCHEMA=0`, authorizer, progress
handler, hard heap limit.

**grep ported in-process.** `targets/grep_read.c` used to `fork`/`exec`
`/usr/bin/grep` on every execution — two silent consequences: **zero coverage
of grep** (the code under test ran in an uninstrumented process image, so every
edge that target ever reported came from the wrapper), and one process spawn
per execution, exactly the cost used to disqualify the `locked` set,
reintroduced inside a `direct_lite` member. Measured by the cost ledger: 1.27
ms of matcher time per execution against a 13–26 ms cell — over 90% spawn
overhead. `tools/vendor_grep.sh` already existed and worked, and nothing
consumed it.
Now links grep's own matchers (gnulib `lib/dfa.c`, Commentz-Walter
`src/kwset.c`, PCRE2 when present) and drives them in-process. Measured after:
**753 edges against ~195**, and they are DFA basic blocks rather than harness
blocks; throughput roughly doubled (76 vs 38 eps); 20,000 executions across all
9 modes under ASAN+LeakSanitizer with malformed patterns, no leaks or errors.
Design notes worth keeping: `dfaerror()` is `_Noreturn` and the DFA engine calls
it for *most* fuzzer-generated patterns, so the harness uses setjmp/longjmp with
a guard on the jump buffer (jumping into a dead frame is UB that would surface
as an irreproducible crash, so aborting there is the honest choice); the input
format is unchanged so corpora and dictionaries still parse, but the pattern is
now passed as (pointer, length) rather than a C string — the wrapper had to
NUL-terminate it for argv, which truncated every pattern at its first zero byte
and made that region of the input space unreachable; and the modes now name
**engine configurations**, not CLI flags, because the old `MODE_INVERT` /
`MODE_WORD` claimed to exercise `-v`/`-w`, which live in grep's output layer,
not the matchers.
`tests/test_grep_target.py` (9 cases). The first four are **source-level
reversion guards** (`execlp`, `fork`, `/usr/bin/grep`) precisely because both
wrapper failure modes are silent: a suite that cannot detect the target
re-spawning would report a healthy campaign over fake coverage. Falsified both
ways.
Build constraints found: `-include config.h` **must** precede `-include $SHIM`
or gnulib's replacement headers `#error`; `libgreputils.a` must be built with
`-fPIC` or the `.so` link dies on `relocation R_X86_64_PC32 against symbol
stderr@@GLIBC_2.2.5`; and PCRE2 is detected by test-linking rather than
trusting `HAVE_LIBPCRE` from `config.h` — that macro records what configure
saw, not what links here.

**FFmpeg vendoring works on a clean container.** `tools/vendor_ffmpeg.sh
--nosan --minimal` then `tools/build_targets.sh`; all five variants build
(`ffmpeg_read`, `ffmpeg_read_nosan`, `ffmpeg_read.so`, `ffmpeg_read_asan.so`,
`ffmpeg_read_ubsan.so`; the ASAN pass copies `vendor/ffmpeg` →
`vendor/ffmpeg_asan` and rebuilds itself). ~4 min on 1 core for vendoring,
142 MB, 70 trace-cmp sites in libavformat and 77 in libavcodec.
Prerequisites that belong next to `clang`: `nasm`, `libjpeg-dev`, and
`apt-get update` **first** (without it `apt-get install clang` 404s on
`libc6-i386`/`libxml2-dev`).

---

## 9. Benchmarking and measurement harness

**`tools/bench_lock.py` + `--lock-single-thread`.** Two levels: across
processes, an **exclusive** `flock` that does not queue and does not
warn-and-continue but exits `rc=3` printing the owner's command line — failing
loudly is the point, since a run that merely waited would have been launched by
someone who believed they were alone. Within the process,
`OMP_NUM_THREADS=1` + BLAS equivalents + CPU pinning. It is `flock` and
deliberately **not** a pidfile: if the owner dies to a VM restart the kernel
releases the lock and a stale pid would sit there inert — not hypothetical, the
VM restarted twice during the run it was written for.
`tests/test_bench_lock.py` (6 tests), falsified by swapping `LOCK_EX` for
`LOCK_SH`.

**`tools/bench_replicated.py` + `analyse_replicated.py`.** Runs both arms
**paired in time** (one after the other inside each replicate, rather than a
whole arm then the other, which confounds machine drift with arm) and selects
the arm by **source tree via `PYTHONPATH`** instead of editing the file between
runs — the arm cannot be mislabelled because the tree it ran from *is* what
defines it.

**`tools/noise_probe.py`.** Measured floors, same arm, same seed:
`png_read.so` 45–63 and 39–66 edges (CV 0.118 / 0.189); `jpeg_read.so`
201–206 and 196–209 (CV 0.010 / 0.026); `grep_read.so` 792–803 (CV 0.007);
`zlib_read.so`, `lz4_read.so`, `gzip_read.so` **bit-for-bit deterministic and
saturated**. Null discordance over same-arm replicate pairs: **37/80 = 46.2%**
— nearly half the pairs McNemar would score as a win or loss are pure noise.
Two readings, both load-bearing: 60 of 120 cells per arm (zlib+lz4+gzip)
**cannot** produce a discordant pair in either direction, so that budget buys a
guaranteed tie; and the noise does **not** invalidate McNemar — arm assignment
is independent of it, so discordants still split 50/50 under the null and type
I error is controlled — what it destroys is **power**.

**`tools/cost_dispersion.py`.** Reads the persisted cost ledger after a
campaign and reports p90/p10, max/min and CV of per-seed mean execution cost,
with a verdict threshold. This exists because the identity that disqualifies
uniform-cost targets (under constant per-execution cost, `effective_fuzz_count`
= `total_time/mean_exec` reduces **exactly** to `fuzz_count`, collapsing both
arms into the same computation) is a property of the **target**, not the
harness — so check it per target before spending cells. Measured on grep: CV
0.922, p90/p10 7.18×, max/min 32.5× over 810 seeds with cost samples.

**`--by-target`** in `tools/bench_paired.py`.

---

## 10. Investigations closed

**Intermittent full-suite `Segmentation fault` — FIXED.** `core/z3_lifecycle.py`
(`guard_z3_shutdown`, wired into `core/smt_solver.py` and
`core/structural_constraints.py`) closed it in `a537614`. The header said "fix
not yet written" for two weeks after it was fixed.
Kept for the investigation, which is still worth something: the crash was **not**
where the output stopped. Output ended mid-file at ~90%
(`tests/test_structural_constraints.py`) with no summary and no traceback; the
fault was at interpreter finalisation. Reproduction rate 1 crash in 8 full-suite
runs. This is also the motivating case for the component-set bisect in
`docs/port-backlog.md` (F5): a `SIGSEGV` with `ip 0` and a silent faulthandler
cost a session on BLAS-thread and shutdown-buffering theories before the cause
turned out to be a vendored-zlib link mismatch, which an operator-set bisect
would have ruled out in minutes.

**`test_shim_updates_edge_count_after_target_call` hang — FIXED** in `a267ff8`
and its follow-up.
**The lesson is the reason this is kept, and it outranks the hang.** The first
fix landed only half of Option B: the shim gate went in and **no loader ever set
`__AFL_FORKSRV`**, so the forkserver silently fell back to `run_executable()`
for every RUN — and all eight assertions in
`tests/test_regression_forkserver_shm.py` passed either way, because the two
execution modes are externally indistinguishable over the protocol. A suite that
cannot detect that a feature was effectively reverted is the finding.
The hang itself never reproduced locally (~0.4 s, passing single, ×5, whole
file, and under `pytest -n auto`), so it was environmental, state-dependent, or
already fixed.

**SJT adjacent transpositions — CLOSED, NOT ADOPTED.** Ten findings, an A/B with
ground truth, and a negative result for SJT specifically. Recorded in full so
it is not re-proposed:

- The planned state design cannot work. `id(seed_buf)` is address reuse
  (5 sequential bytearrays → 1 distinct id); the "weak-key LRU" alternative does
  not exist at all (`weakref.ref` on `bytes`/`bytearray` raises `TypeError`;
  only `memoryview` supports it). The real failure mode is **not** restart
  collapse but **silent cross-contamination**: simulated over 20,000 execs the
  random-slice branch hits 51.3%, and 5,269 of those 10,269 hits inherited a key
  created for a *differently sized* buffer.
- The genuine upside is real but small and conditional: within one unbroken walk
  SJT never repeats — n=8, budget 10,000 → 10,001 distinct vs 7,153 for a random
  adjacent-transposition walk. But that baseline **does not exist in the tree**.
  Against the real baselines the honest framing is: a single adjacent swap from
  the parent saturates at 7 distinct outputs at n=8 (ceiling n−1), `_swap_pair`
  saturates at 28 (ceiling C(8,2)). So with its state broken SJT degrades to a
  **4× regression against `_swap_pair`**.
- The falsification claim does not hold: SJT is deterministic and draws nothing,
  so `ExhaustivePool` has no tree to enumerate. Under the house seam
  (`tests/support/operator_env.py::make_minimal_fuzzer(pool=…)`) it yields
  exactly one run and is unfalsifiable.
- Band mismatch, confirmed exactly: all 16 `_swap_pair` call sites are inside
  `*_chunk_mutate` operators and all 36 of those are in the **format** band; the
  plan landed SJT in **structural**. The locality argument (preserve length
  fields, checksums, relative offsets) only has teeth in the format band —
  precisely where the plan did not put it.
- **The A/B, 5 arms × 1200 observations × 5 seeds, against a purpose-built
  ground-truth target** (`targets/order_sensitivity.c`, 16384 B = 32 records,
  4 profiled regions; ASLR off via `preexec_fn`; **fresh SHM per execution** —
  reusing one accumulates edges and makes every comparison report "moved
  coverage"):

  | arm | M1 | wrong verdicts |
  |---|---|---|
  | A `chunk_shuffle`, fabricated offset (status quo) | 0.4% | 2/4 — loses both dead regions, all 5 seeds |
  | B random adjacent swap, true offset | 100% | 1/4 |
  | C **SJT as specified by the handover** | 100% | 1/4 — identical to B on all 5 seeds |
  | D SJT with randomised start | 100% | 0/4 on 3 seeds, 1/4 on 2 — unstable |
  | E free permutation **confined to a region**, no SJT | 100% | **0/4 on all 5 seeds** |

  The entire gain comes from **confining the reorder to a region and reporting
  its true offset**, not from SJT. That is a one-line change to
  `_op_chunk_shuffle`; the new generator adds nothing.
- **Why, and this is the mathematical point:** deciding whether a region is
  order-live needs **independent samples** of `S_n`, not a **connected walk**.
  SJT's Gray-code property is exactly what makes it a poor sampler here — it
  trades independence for locality, and locality is not what the estimator
  needs. Arm E draws independently per observation and hits the r3 trigger at
  ~12% per draw; C and D explore a small neighbourhood of one start.
- **Correction to a hypothesis previously recorded as rejected:** positional
  bias *was* mismeasured. Counting "swap positions touched" shows all positions
  touched immediately. What matters is the **content** of each slot: at n=8 slot
  0's content first changes at step n−1=7, but the specific value that triggers
  r3 does not reach slot 0 until step **5376 of 40320**, and 0 times in the
  first 300. SJT's exhaustiveness is real but **asymptotic**; within any bounded
  observation budget it is a **local** sampler. That is the mechanical reason C
  ties B.
- One conditional use survives and is filed as open: a brute-force optimality
  oracle for a future `core/job_scheduling.py`. For `1||Σf_j` an adjacent
  interchange at `(i,i+1)` changes only `C_i`, so the objective delta is O(1)
  instead of O(n), and the SJT walk **is** the exchange-argument chain that
  proves SPT/EDD optimal, so a counterexample comes out as a named interchange.
  Measured at n=8 over all 8! sequences: with cheap inline arithmetic the
  incremental version **loses** at 0.93× (because `itertools.permutations` runs
  in C and the delta runs in Python); it wins 2.06× once the per-job term costs
  a function call, and 3.88× at ~80 flops/job. State the crossover; do not claim
  it unconditionally.

**`docs/edge-coverage-analysis.md` — resolved, nothing to re-merge.** It was
merged as an appendix to the skittercreek handover in `852e274` (18 Aug) and the
round-13 prune (`97c1d52`, 19 Aug, 1466 → 312 lines) deleted it the next day
**without recording it in that document's own "What was removed" ledger**. Four
code comments cited the dead path for ten more days. Recovered from `97c1d52^`
and audited section by section: everything is closed or already carried
forward.

---

## 11. Corrections found while consolidating

1. **Three cited commits do not resolve** in the current history: `b49441b`
   (claimed to carry all five minimax phases), `71f2e02` and `bbb2645` (claimed
   to fix the staging `rm -rf` and the discarded build output). In all three
   cases the **code is present and was verified by inspection**; the hashes are
   stale, presumably from a rebase. Do not chase them.
2. **The percolation document contradicted itself**: its header said Modules
   1–4 IMPLEMENTED while its own §1 still read "Module 2 is live; modules 1, 3,
   4, 5, 6 remain as proposed designs", and its Module 3 section still carried
   an "Implementation plan". Resolved above: 1, 2 and 4 are live and consumed;
   3 exists but is unwired; 5 and 6 are absent.
3. **The GARCH document's Kalman inventory line was false.** It claimed a
   Kalman filter denoises the discovery rate upstream of CSD/Allan. The live
   Kalman is `_eps_kf` in `stats.py:630` and filters **EPS (executions/second)**,
   not the discovery rate; its output feeds the avg-eps path. CSD receives
   `discovery_rate()` **raw**. `CriticalSlowingDown` does take a `denoiser`
   parameter, but production constructs it with `denoiser=None`.
4. **The Navier–Stokes document inherited "Modules 1–4 live" as a premise** from
   the contradictory percolation header, and its falsifier list did not include
   the failure mode it is most exposed to — a graph field module that lands,
   passes tests, and is read by nobody. That is precisely what happened to
   Module 3.
5. **`CoverageRegimeDetector._classify` already ignores 3 of its 5 arguments** —
   `discovery_rate`, `allan_delta` and `exec_count` are passed by `observe()`
   and never read (verified by AST). `allan_delta` is documented as unused;
   `discovery_rate` is not. Any sixth argument added there lands in the same
   hole. Minor: `observe` is annotated `-> None` but returns `self._regime`.
6. **The SJT document's `_swap_pair` count drifted** from 15 to 16 (mp3 joined),
   and its table omitted `_swap_tuple`, which now exists but is unwired.
7. **F2 is closed by documentation, not by code**: the build-root default
   (`~/fuzzing/builds`) that disagreed with three documents is now what
   `AGENTS.md` states in three places. The disagreement was the defect.
8. **Line-number drift is systemic.** The anchors in
   `docs/bugreport_2026-08-21_merged.md` — `filesystem.py:608`, `kalman.py:370`,
   `edge_tracker.py:1336`, `elo.py:685`, `shapley.py:174`, `count_class.py:41-49`
   — point at unrelated code today. Locate by symbol. Worked example, and drift
   in the *other* direction: finding 83 is now documented in
   `core/count_class.py` as deliberate behaviour with
   `test_count_class_exhaustive.py` pinning both ladders separately, so what the
   report still lists as a defect was decided as intentional without the entry
   being touched.

---

## 12. What was removed — ledger

Every file below was deleted by the last commit of the series that added this
document. Full text is one `git show` away; the section here says what absorbed
it and, where applicable, what is still open.

| Removed | Absorbed by | Still open? |
|---|---|---|
| `P1_weizz_tags_README.md` | §4 | paired bench → pending §E |
| `fractal-voronoi-integration.md` | §4 | approaches B/C, A/B → pending P4 |
| `handover_algorithm_catalogue_survey_2026-09-06.md` | §3, §11 | K1/K2/K3/K4/K5/K6/K7/K8 → pending P0/P1/P4 |
| `handover_bandit_stopping_search_2026-09-02.md` | — | entirely open → pending P2/P3 |
| `handover_boltzmann_ab_2026-08-30.md` | §2, §8, §9 | closed |
| `handover_combinatorics_permutations_2026-09-02.md` | §3 | §10b–10h → pending P3/P4 |
| `handover_ffmpeg_build_paths_2026-09-03.md` | §8 | F3/F5/F8/F9, N3/N4 → pending P2/P4 |
| `handover_formatfuzzer_integration_2026-09-06.md` | §4 | phases 2–3 → pending P4/E |
| `handover_garch_volatility_modelling_2026-09-05.md` | §5, §11 | closed (implemented) |
| `handover_job_scheduling_2026-09-02.md` | — | two measured bugs → pending P0; plan → P3 |
| `handover_minimax_alphabeta_adversarial_search_2026-09-01.md` | §2 | superseded by its implementation note |
| `handover_minimax_implementation_2026-09-01.md` | §2, §11 | A/B → pending §E |
| `handover_navier_stokes_coverage_flow_2026-09-05.md` | §5 | time-stepping deliberately not done |
| `handover_percolation_theory_2026-08-31.md` | §5, §11 | Module 3 wiring → P2; Modules 5/6 → P4 |
| `handover_persistence_mechanics_2026-08-29.md` | §5 | §1c successor lives in `docs/TODO.md` |
| `handover_qea_hilbert_space_analysis_2026-08-31.md` | §5 | closed |
| `handover_seventeen_source_survey_2026-09-06.md` | §6, §3 | B2/B3/C1/C3/C4/C5 → pending P3/P4 |
| `handover_tang_recommendation_2026-09-07.md` | §14 | E6 A/B still open → pending §E |
| `handover_sjt_adjacent_transpositions_2026-09-04.md` | §10 | closed, not adopted; oracle use → P4 |
| `handover_skittercreek_tailslayer_port.md` | §1, §3 | item G, item H → pending P4 |
| `handover_weizz_structure_aware_port_2026-08-31.md` | §4 | paired bench → pending §E |
| `suite_segfault_z3_finalization_2026-08-16.md` | §10 | closed |
| `test_shm_hang_2026-08-14.md` | §10 | closed |

**The ledger exists because of a specific failure.** A document that is pruned
repeatedly needs a per-round "what was removed" record, because a closed item
still needs a pointer to where its artifacts live or the next person re-proposes
it — and because without one, entire documents vanish silently, which is exactly
how `docs/edge-coverage-analysis.md` was lost for ten days (§10).

---

## 13. Lessons worth keeping

These cost real sessions. They are not about any one feature.

- **A `[ ]` is not evidence of anything.** Check every closed item against live
  source before deleting it, and every open item against live source before
  working it. Four headers here were wrong in one direction or the other.
- **Falsify before writing the docstring that says what the test catches.** Two
  guards written for `region_shuffle` turned out to be **unfalsifiable** — 11
  tests passed with the clamp removed *and* with the identity early-return
  nulled, because neither had an observable difference. The fix was to correct
  the *assertions*, not to invent contorted tests.
- **If the bug is data-dependent, a sweep does not replace one known-failing
  input.** The first Myers regression test **passed against the broken code**: a
  parameterised sweep over n ∈ {80,128,512,2048} × k ∈ 1..20 passed entirely,
  because whether the greedy unwind overshot depended on where match runs fell.
  It only became a test once a falsifying pair was found (by sweeping 3000
  seeds) and **pinned as literals**.
- **A suite that cannot detect a reverted feature is the finding.** Twice: the
  forkserver (§10) and the grep wrapper (§8).
- **When a cost estimate is the reason *not* to do something, measure it** —
  especially across a port that changed the execution path. grep was deferred at
  "~250 s/cell, ~4 h" written up persuasively enough to look considered; a cell
  is ~40 s and the full run took ~80 min. The 250 s predated the in-process
  port.
- **Low relative noise is not low noise.** grep's CV of 0.007 on a base of ~800
  is ~12 edges of absolute dispersion against png's ~4.6, so it needed **twice**
  the seeds for comparable power. What competes with effect size is **absolute**
  dispersion.
- **An interim split over half a matrix is a different quantity, not a weaker
  version of the result.** grep's first 10 seeds gave 7W/2L, median +4.5,
  p=0.180 — *in the direction the mechanism predicts*. Seeds 10–19 gave exactly
  5W/5L.
- **Mutate in place and undo; do not copy.** Three instances now:
  `gradient_descent` (0.79 → 0.088 ms at 8 KiB, shipped), the dancing-links
  framing, and the deterministic stream (still open, P1-1). Write it as policy:
  **any operator that evaluates many candidate edits against a buffer should
  mutate in place and undo.**
- **One container, one benchmark.** A second campaign appeared in the same
  single-core container mid-run; cells went 22 s → 50 s. That is not slowness,
  it is **contamination** — machine load feeds `mean_exec`, the quantity under
  test. Hence `--lock-single-thread`.
- **Never read a completion stamp.** The lock correctly rejected a duplicate
  launch with `rc=3`, but the rejected process still ran the shell's
  `; echo DONE > stamp`, so a completion stamp appeared with 17 cells left. Read
  the checkpointed JSON or the owner's pid. The log does not work either: two
  processes redirecting to one path interleave and the file reads as binary.
- **`git stash push <pathspec>` on an untracked file fails the whole push
  silently**, so a "baseline" can run *with* the fix in place. The tell was two
  identical numbers where the patched run should have been higher. For a real
  baseline: `git checkout --` the tracked files and **move** new files out of
  the tree — and commit before measuring, because `git checkout --` on an
  uncommitted fix deletes it.
- **Clear `tools/__pycache__` when falsifying anything under `tools/`** — it is
  not part of the installed package, so stale bytecode survives a source
  restore.
- **`OperatorEngine.ctx` is a property that rebuilds the `MutationContext` on a
  cold cache**, so `eng.ctx.seed_meta = {...}` writes to a throwaway object and
  the test passes for the wrong reason. Prime `eng._ctx_cache = eng.ctx`.
- **Renumbering a findings list re-points every reference by number** in
  commits, learnings and code comments. The gaps in
  `docs/bugreport_2026-08-21_merged.md` *are* the closed items; leave them.
- **`CHANGELOG.md` is append-only.** It cites a path that has been dead since
  August, three times, and that is correct — those entries were true when
  written.


---

## 14. Tang low-rank seed scheduler (2026-09-07)

#### 0. One-line summary

The port works and the low-rank reconstruction is genuinely good; as a *scheduler
signal* it is a laundered row sum — controlling for `A.sum(1)` it adds nothing
(mean partial Spearman **+0.006** across ten campaigns, Wilcoxon **p = 1.0**).
It ships behind `--tang`, off, because a measured negative is worth more in-tree
than out, and because the paper's subroutines are independently useful.

**Do not enable it as a default without running `bench_paired.py` first.** No
interventional evidence exists. §7 lists exactly what was never tested.

---

#### 1. What landed

| File | What |
|---|---|
| `src/fuzzer_tool/core/schedulers/tang.py` | `TangRecommendationScheduler`, plus faithful ports of Prop 4.2 (inner-product estimation), Prop 4.3 (rejection sampling from `Vw`) and `modfkv_sample_complexity` (Alg. 2's `q`) |
| `src/fuzzer_tool/services/fuzzer.py` | `"tang"` in `_SEED_STRATEGY_NAMES`; constructor kwargs; refit hook on the exec path |
| `src/fuzzer_tool/services/seed_picker.py` | `_pick_tang_seed`, the Elo arm, availability gate |
| `src/fuzzer_tool/cli/commands.py` | `--tang`, `--tang-rank`, `--tang-refit-interval` |
| `tests/test_tang_scheduler.py` | 31 tests |
| `docs/DEEP_DIVE.md` | feature entry (Hard Rule 11) |

### Mapping

```
users    -> corpus seeds        (EdgeTracker.seed_edges keys)
products -> coverage edges
A[i][j]  -> hit count of edge j under seed i  (EdgeTracker.seed_hit_counts)
```

Algorithm 3 samples an entry from row *i* of `D = A Vhat Vhat^T`. Read as a
recommender it answers "which edge should this seed reach next"; read as a
scheduler it yields a per-seed energy, which is what the arm consumes.

### What is deliberately *not* a faithful port

`refit()` runs a dense truncated SVD, not ModFKV. That is the **stronger**
estimator — no row/column subsampling noise — so every negative result below is
a negative on Tang's best case, not on a degraded stand-in. §2 is why ModFKV
itself cannot run here.

---

#### 2. ModFKV cannot run at our dimensions

`q = Theta(K^4 / eps_bar^2)`, `K = ||A||_F^2 / sigma^2`, `eps_bar = eta*eps^2`.
`modfkv_sample_complexity()` keeps this checkable rather than folklore.

| corpus | shape | K at sigma_3 | q | rows available |
|---|---|---|---|---|
| png | 116 x 884 | 6.75 | 828,570 | **116** |
| png (binary form) | 116 x 884 | 65.3 | 7.3e9 | 116 |
| zlib | 49 x 229 | 55.3 | 3.7e9 | **49** |

Even at the most forgiving parameters anywhere (`K=4, eps=0.5, eta=0.2`),
`q ~ 1e5`. The subsample is larger than the input. The bound is independent of
`m, n` — which is the paper's entire point — but our `m, n` are *already*
10^2–10^4, so the constant is what governs.

---

#### 3. The low-rank assumption: it depends on the cell value, and the two Tang assumptions conflict

`rho_k = ||A - A_k||_F / ||A||_F`. Tang wants `rho << 1` at constant `k`.

| matrix | form | rho@1 | rho@3 | rho@10 | rho@20 |
|---|---|---|---|---|---|
| png 116x884 | hit counts | 0.748 | 0.308 | 0.123 | 0.050 |
| png 116x884 | binary | 0.402 | 0.330 | 0.232 | 0.173 |
| zlib 49x229 | hit counts | 0.371 | 0.145 | **0.051** | 0.025 |
| zlib 49x229 | binary | 0.646 | 0.485 | 0.299 | 0.172 |

Hit counts fit much better. **But they destroy the other assumption.** The
`(gamma, zeta)`-typicality of §5.1 (row norms within a factor of the mean):

| matrix | typical fraction at gamma=1.0, binary | ... hit counts |
|---|---|---|
| png 116x884 | 91.4% | **17.2%** |
| zlib 49x229 | 91.8% | **6.1%** |

The value form that makes the matrix low-rank is the form that makes row norms
heavy-tailed. **Tang's two structural assumptions cannot be satisfied
simultaneously on this data.** This is the most transferable finding here and
applies to any future collaborative-filtering proposal over coverage.

---

#### 4. The sampling direction is backwards for a fuzzer

Sampling proportional to `|D_ij|^2` samples by magnitude, and magnitude is
popularity. `E[owner_count of the drawn edge]` vs a uniform draw:

| corpus | rank 2 | rank 5 | rank 10 | uniform | singleton edges |
|---|---|---|---|---|---|
| png 116x884 | 63.0 | 67.5 | 70.3 | 46.96 | 11.9% |
| zlib 49x229 | 24.6 | 25.3 | 27.3 | 12.79 | 13.1% |

1.3–1.9x more crowded than uniform, on corpora where 10–19% of edges are
singletons. That fights `RARE_EDGE_OWNERS` and the crowding penalty in
`seed_picker` head-on — the exact signal the edge-distribution work installed.

`P(the draw is an edge the seed does not already cover)` is 36%/40% at rank 2
but 12.7%/5.0% at rank 10: as a "what next" oracle it is mostly self-recall.

### Inverting it does not help — and is not even a separate measurement

`frontier_mass = 1 - covered_mass` **exactly** (rows of `P` sum to 1; measured
`max|covered + frontier - 1| = 6.7e-16`). So the inverted score is the same
number sign-flipped, and it is anti-predictive in 10/10 campaigns. `mode="frontier"`
exists to keep this falsifiable, not as a candidate. `test_frontier_mode_is_the_complement_of_tang_mode`
asserts the identity so the docstring cannot drift from the code.

---

#### 5. The decisive measurement

Prospective design: `edge_first_seen` gives a discovery clock. Split edges at
the 60% quantile, build `A` from early edges only, restrict to seeds present
early, label each seed with how many *late* edges it ends up owning.

Ten independent campaigns (png and zlib x `-s 5/17/29/41`, plus two earlier
runs), 19–87 seeds each after filtering. Spearman vs the label:

| score | range | consistent |
|---|---|---|
| degree | +0.27 … +0.77 | 10/10 positive |
| total hits | +0.16 … +0.80 | 10/10 positive |
| Tang l2 covered mass | +0.04 … +0.79 | 10/10 positive |
| frontier mass | -0.02 … -0.79 | 10/10 negative (see §4: same fact) |
| singletons owned | mixed | 6+/3- |

**Partial correlation of the Tang score controlling for total hit volume:**
mean **+0.006**, positive in 5/10, Wilcoxon signed-rank vs 0 **p = 1.0**. A
pure-noise control through the same path gives +0.01.

Every bit of the Tang score's predictive power is `A.sum(1)`.

---

#### 6. What was *wrong* in the first pass, and why

Two claims made mid-analysis were later overturned. Both are recorded because
the failure modes recur.

### 6.1 A silent binarisation destroyed rounds 1–2

After the `to_dict` round-trip, `seed_edges` holds edge ids as `int` while
`seed_hit_counts` and `edge_owner_count` hold them as `str`. An analysis harness
reading the raw dict gets `hc.get(e, 1)` missing every time and a **silently
binary matrix**, with no error.

`EdgeTracker.from_dict` (edge_tracker.py:2765-2816) *does* re-cast with `int(e)`
on all six maps, so **this is not a fuzzer bug** — but anything reading the
payload directly must do the same. `test_string_keyed_hit_counts_do_not_silently_binarise`
pins it.

**The tell:** hit-count, binary and log1p forms returned identical spectra to
three decimals. When two different transforms of the data give the same number,
the problem is upstream of both — same signature as the truncated `random_list`
lesson.

### 6.2 "Low-rank only buys 7% over a column sum" was an artifact

The round-1 holdout masked entries MCAR. But missingness here is fuzz-driven:
`rho(fuzz_count, per-row observed density)` is +0.40 to +0.62 on zlib
(p down to 4e-10), +0.343 on png — so Tang's uniform-subsample assumption (♣)
is badly violated. Redone with correct hit counts *and* MNAR masking (biased
toward low-count entries, which is how coverage is actually missed):

| corpus | mask | rank 10 | popularity |
|---|---|---|---|
| png 116x884 | MNAR | 0.387 | 0.247 |
| png 116x884 | MCAR | 0.391 | 0.347 |
| zlib 49x229 | MNAR | **0.523** | 0.083 |
| zlib 49x229 | MCAR | 0.442 | 0.151 |

**Tang's reconstruction fidelity is real** — 1.6x to 6.3x over popularity, not
7%. The scheduler verdict is unchanged because §5 was already run on hit counts.

---

#### 7. Methodology gaps — read this before trusting §5

1. **The label is downstream of the policy under test.** `seed_edges` grows on
   every execution, so "owns late edges" partly measures `fuzz_count`, which the
   *current* scheduler decides. Measured `rho(fuzz_count, label)` up to +0.60.
   Controlling for it the conclusion survives (Tang +0.368 with 9/10 positive,
   degree +0.407 with 10/10), but no IPS estimator and no propensity were used.
   This is the same plumbing gap `handover_nn_over_metrics` identified.
2. **Two clocks, never aligned.** The split uses `edge_first_seen` (a counter)
   while the seed filter uses `added_at` (wall time). Verified no contamination
   (0 kept seeds have their first edge on the late side), but the
   `added_at <= p75` filter is a near no-op; what actually filters is "owns an
   early edge", which is weaker than "existed early".
3. **No interventional evidence at all.** Everything is observational on a
   corpus the current scheduler produced. The project standard is
   `bench_paired.py` with a pre-registered threshold. It was not run.
4. **Two targets, both small and single-format** (png, zlib). ffmpeg — 8189
   edges, genuinely multimodal — is where the low-rank assumption would have its
   best shot, and it would not build in that container (vendored headers).
5. **Statistical power.** n runs 19–87 after filtering; no weighting, no
   intervals. At n=19 a rho of +0.30 has 95% CI [-0.18, +0.66]; it excludes zero
   only around n=55. The eight replication runs share seed files, target and
   mutator set — only `-s` differs — so effective independence is well under 10.
6. **Rank fixed at 10, no cross-validation.** ModFKV thresholds by `sigma`, not
   by `k`, so even the idealised estimator is not the paper's.
7. **Cost timings used dense random matrices**, not real sparse sets, and
   incremental rank-1 update was never considered — the matrix grows one row per
   accepted seed, so a streaming update would weaken the cost objection.

---

#### 8. Cost

Randomized SVD (k=10, oversampling 8) and the matrix build from Python sets:

| shape | build | SVD | project + sample one seed |
|---|---|---|---|
| 116 x 884 | 6.0 ms | 1.18 ms | 0.019 ms |
| 500 x 2000 | 40.8 ms | 5.20 ms | 0.026 ms |
| 2000 x 8189 | 658 ms | 102 ms | 0.082 ms |
| 8000 x 8189 | 2407 ms | 481 ms | 0.083 ms |

The build dominates. Same shape as the NN-over-metrics finding: the cost is
feature extraction, not the model. This is why `maybe_refit` is on the exec path
behind an interval gate and never on the pick path.

**Lemma 3.1's BST is also a loss.** It looked like the one portable ingredient —
a real answer to `_cdf_pick` rebuilding `itertools.accumulate` every 200 execs.
In CPython a Fenwick point update costs 1.09 us at N=8000 while the full cumsum
rebuild amortized over 200 picks costs 0.88 us, and its draw is 5–6x slower than
`bisect`. `accumulate` and `bisect` run in C; the tree runs in Python. Do not
re-propose it without a C extension.

---

#### 9. Rejected, with reasons (do not re-propose without new evidence)

| Idea | Why it died |
|---|---|
| Inverted frontier score (`mass on uncovered / owner_count`) | Algebraically `1 - covered_mass`; anti-predictive 10/10 |
| Residual after removing the rank-1 popularity component | +0.45 zlib, -0.19 png — sign flip, noise |
| Transpose (for a rare late edge, which seed reaches it) | Degenerate: `\|\|D_i\|\|` gives *exactly* degree's precision (0.067/0.067, 0.516/0.516) |
| Operator x edge matrix | Cannot be built from state — `corpus['op_edges']` is a scalar credit float per operator (119 entries), not a matrix. Would need new instrumentation |
| Degree-normalized label (to test the degree confound) | Degree still wins (+0.338, +0.498); frontier/owners flips sign across datasets |
| Lemma 3.1 Fenwick/BST sampler | §8 — loses to `accumulate`+`bisect` in CPython |

---

#### 10. If someone picks this up

Ordered by what would actually change the answer:

1. **Run `bench_paired.py`** with `--tang` vs baseline, pre-registered threshold.
   Everything above is observational; this is the only test that settles it.
2. **Build ffmpeg and re-measure §3 and §5 there.** 8189 edges and real
   modality is the one condition under which the low-rank assumption plausibly
   holds. If `rho@10 < 0.05` *and* typicality survives, §3's conflict is
   target-specific rather than structural, and this reopens.
3. **Log the training set.** Gaps 1 and 3 both reduce to "there is no
   off-policy log". The sequence in `handover_nn_over_metrics` §5 (append-mode
   `dump_stats`, chosen operator in the ablation log, raw M3 inputs, an IPS
   estimator with a falsification test that `final_w` *is* the propensity) is
   worth doing on its own and would make this measurable properly.
4. **Use `recommend()` for edges, not seeds.** The edge answer is the paper's
   actual contribution and was never the thing that failed — a directed-fuzzing
   or frontier-targeting consumer is the honest use, and none exists yet.
