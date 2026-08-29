# Port backlog

The single list of what is still worth porting into `daedalus/fuzzer`, merged
from four surveys that each kept their own ranking:

| origin | dates | what it covered |
|---|---|---|
| six external sources (nowarp, fitzgen, bernsteinbear ×3, Binvariants, PowerFuzz) | 2026-08-23 | technique-level items, `R*` |
| "A Tale Of Four Fuzzers" (matklad / TigerBeetle) | 2026-08-21 | self-testing methodology, `P*` |
| internet survey (AFL++/academic/grammar/substrate) | 2026-08-25 | 29 tiered candidates |
| GitHub fuzzer survey + FFmpeg harness survey | 2026-08-28 | engine `I.*`, harness `II.*` |

Merging them was the point: four documents proposing four different routes to
structure-aware generation, and three proposing overlapping ways to decide which
bytes to mutate, is not visible while each lives in its own file. Items are
grouped by **the problem they solve**, ranked within the group, and every entry
keeps its origin tag so the original reasoning is findable in git history.

**Before starting anything here, grep for it.** Every one of these surveys
opened with a shortlist and found most of it already implemented — the
2026-08-28 pass began with five engine "gaps" and four were in the tree. The
effort estimates are guesses made from the sources; only the status column of
each original was ever grepped against live code, and that was weeks ago.

**Everything here should land behind an A/B flag** in the `--no-adaptive-havoc`
shape and be measured through `tools/bench_paired.py`. No item below has a
measured coverage delta.

---

## A — Structure-aware generation

Four proposals for one gap. Ranked by plumbing already in place, which is not
the order any single source gave.

**A1. Grimoire-style generalization** (`I.4`, LibAFL `GeneralizationStage`) —
blank spans of an input, re-execute, keep the spans whose removal does not
change coverage. Yields structure and recombinable tokens with **no grammar
supplied**, which is exactly where the ~155 hand-written format mutators have
nothing to offer. Lands as a new stage beside deterministic/havoc and reuses the
colorization executor loop almost verbatim — same "mutate, compare path
checksum" shape. **Effort M, and cheapest of this group because the plumbing
exists.** Start here.

**A2. Gramatron — grammar automatons instead of parse trees** (`I.3`,
HexHive/Gramatron ISSTA'21, also vendored in AFL++). Restructures the CFG into
an FSA so an input is a *walk*; splice becomes a state-matched cut of two walks.
Claims unbiased sampling and more aggressive mutation than parse-tree operators.
Lands in `core/grammar.py` next to `TreeMutator`/`SubtreePopulation`; the
`.gram` loader is the input side already. Caveat: exactness holds only for
non-self-embedding grammars. **Effort M.**

**A3. Tree-sitter operator category** (`R6`). `grammar.py` and
`tree_mutator.py` already cover the space; what tree-sitter adds is the
*ecosystem* — maintained grammars for real languages instead of hand-written
S-expressions. The 13 strategies (`ts-del`, `ts-bank`, `ts-add`, `ts-swap`,
`ts-shrink`, `ts-lit`, `ts-dup`, `ts-ins`, `ts-range`, `ts-chaos`, `ts-kdel`,
`ts-kins`, `ts-stutter`) ship with hardcoded weights upstream; our schedulers
would learn the split per target — seed from the source's weights, let the
bandit move them. The genuine architectural addition is the **bank**: a
`TSSymbol`-indexed pool of subtrees harvested across the corpus, refreshed as
the queue grows, keyed by symbol *name* not numeric `TSSymbol` so it survives
`--resume`. Caveats from the source: grammar quality dominates (grammars
emitting `ERROR` nodes degrade everything downstream, and `ts-chaos` patches
that rather than fixing it), and `ts-ins`/`ts-add` without a good initial corpus
accumulates oversized entries — wire to the minimizer. **Effort ~1 week
including the bank, and only worth it if a real text-format target is in scope.**

**A4. FormatFuzzer decision seeds** (`#9`, USENIX Sec '21, `uds-se/FormatFuzzer`).
Compiles community 010 Editor binary templates (170+ formats incl. MP4/PNG/AVI/ZIP)
into parser+generator pairs; the byte fuzzer mutates choice bits while output
stays valid. Would generalize the hand-written mutator family. **Effort M.**

---

## B — Deciding which bytes to mutate

**B1. Identifier canonicalization** (`R3`) — ~50 lines, and the only cheap item
in this group. Splice-style mutations shuffle identifiers, so results fail on
trivial cross-reference errors long before reaching anything interesting. Emit a
*second, canonicalized* corpus where every identifier is renamed to a uniform
deterministic pattern (`v0`, `v1`, …) and fuzz that; splices then cross-reference
correctly by construction. A `tools/` preprocessing step producing a sibling
corpus dir — keep both, never overwrite. Whether a binary analogue exists
(normalizing internal offsets/IDs before splicing) needs its own analysis;
`core/field_constraints.py` is where that would start. **Do not assume it
transfers. Effort ~half a day for the text case.**

**B2. FairFuzz-style rare-branch masks** (`I.5`) — per rare edge, compute which
byte positions can be mutated while still hitting it, and restrict mutation to
the complement. The "which branch is rare" half is done: `_edge_owner_count`
rarity was corrected in the edge-distribution work (`0afc439`). Only the mask
half is missing. **Effort L–M, lower confidence than the rest — measure against
the existing rarity bonus first, the two may overlap.**

**B3. Deviating-block probing taint** (`#15`, WindRanger ICSE '22) — static +
dynamic identification of "deviation basic blocks" en route to targets; effector-map
probing maps stubborn branches to controlling offsets. cmplog solves comparisons
globally; this localizes them. **Effort M.**

**B4. Learned mutation-field gradients** (`#16`, IDFuzz USENIX Sec '25) — small
network trained on historically productive mutations near targets; gradients
locate critical input fields. Reported −91.9% ineffective mutations, 2.48×
faster CVE repro. **Effort M.**

**B5. DataFlowTrace taint + focus_function** (`#20`, libFuzzer/OSS-Fuzz DFSan) —
byte→function taint traces concentrate mutations on relevant bytes and energy on
a focus function. Exact mapping, versus colorization's trial-and-error. Needs a
second dfsan-instrumented build plus a Python trace collector. **Effort M–H.**

**B6. Input Processing Tree auto-repair** (`#21`, NestFuzz CCS '23) — DFSan-taint-derived
inter-field and nesting dependencies; cascading mutation auto-repairs
length/offset/container fields when mutating nested structures. Automates what
`frameshift.py` plus the hand-written mutators do by hand. **Effort M–H.**

---

## C — Feedback channels beyond edges

**C1. Invariant-violation feedback** (`R2`, FuturesLab/Binvariants FSE'26) —
the highest-novelty item in this whole backlog, and the one that fits a real
gap: we are rich on the mutation side and conventional on the feedback side
(edges, hit-count buckets, per-edge max, cmplog). Binvariants learns
register-level likely invariants offline and uses **violations** as signal
alongside coverage. It needs ASLR disabled for consistent basic-block addresses
between learning and fuzzing — already done for context-sensitive edge coverage,
so the prerequisite is in place.

The repo's own unimplemented enhancements are the interesting part: *adaptive
learning during fuzzing*, applying invariant updates to a **copy** and
committing only on `FSRV_RUN_OK` so crashing runs don't corrupt the set; and, as
invariants stabilize and violations get rarer, a `perf_score`/`top_rated`
reweighting toward early-stage test cases — which is exactly what our schedulers
are for, arguably a better home for adaptive RLI than AFL++. Cross-basic-block
invariants are also unimplemented upstream; it is single-block only.

**Scope it down before building anything.** Full register-level tracing is
weeks. Start with invariants over the **cmplog operand values already captured**
in `core/cmplog.py` — learn ranges per comparison site, flag violations. Reuses
existing plumbing (`core/checksum_learner.py` already mines `f._cmplog.pairs`
for a different purpose, so the access pattern exists) and answers whether the
signal is worth the full build. **Open risk, and the first thing to measure:**
the proxy may saturate immediately — if nearly every input violates some learned
range, the signal is noise. **Effort ~1 week for the proxy.**

**C2. Campaign-level novelty rate** (`R5`) — the fraction of generated inputs
exercising at least one previously-unseen edge. We have novelty *signals* but
not this *metric*. It converges much faster than total edge count, so it would
make `bench_paired.py` cells shorter and lower-variance. Compute it offline from
recorded runs, not in the hot loop. **Cheap, and it improves the measurement
apparatus every other item here depends on.**

**C3. LBR branch sampling** (`#19`, perf `PERF_SAMPLE_BRANCH_STACK`, Haswell+) —
last-32-branches statistical samples at near-zero cost; a secondary signal for
uninstrumented dependency code. One more event type in `perf_event.py`. **Effort L.**

**C4. Intel PT coverage** (`#27`, honggfuzz `--linux_perf_ipt_block` + libipt;
PTrix AsiaCCS '19) — module-free PT via perf AUX mmap buffers on kernel ≥4.2.
Indirect-block-only decoding is the proven stability baseline (~10–15%
overhead). `perf_shim.c` extension plus a libipt decode thread. **Effort M–H.**

---

## D — Seed scheduling and energy

**D1. Reachable-uncovered weighting** (`#13`, PrescientFuzz arXiv 2024) —
per-seed BFS-counted reachable-uncovered blocks from its trace, inverse-rarity
and depth weighted. Nearly free on the CFG/DWARF pipeline we already have; lands
as a seed-scoring feature atop `cfg.py`/`cfg_cache.py`. **Effort L.**

**D2. EcoFuzz energy allocation** (`#12`, USENIX Sec '20) — adversarial-MAB
estimating per-seed new-path reward probability against observed average cost;
reported −32% execs at equal coverage. A scheduler arm plus a `SeedScorer`
schedule. **Effort L.**

**D3. Reaching-probability directed mode** (`#11`, SelectFuzz IEEE S&P '23) —
block fitness as averaged successor reaching probability instead of graph
distance, instrumenting only target-relevant blocks (<2% of reachable BBs) so
irrelevant coverage never pollutes feedback. `distance.py` math plus one LLVM
pass. **Effort L–M.**

**D4. Measure, don't model, in `parallel.py`** (TigerBeetle) — the one genuine
*algorithmic* port from that source; everything else it offers is testing
methodology (see section F). `services/parallel.py` syncs corpora between
workers on a **fixed `sync_interval: int = 30`** via `_sync_corpus_in`, scanning
sibling directories on a per-sibling cursor. That is a static, modelled topology
on a static, modelled cadence, and it assumes the cost/benefit of syncing is
uniform across siblings and constant over time. It is not — late in a campaign
most synced seeds are redundant.

The PCC-style alternative maps cleanly: track, per sibling, the fraction of
imported seeds that produced new coverage; periodically run an experiment with a
different interval or a different subset of siblings; keep the change if
measured edges-per-import-second improves. The measurement machinery exists
(`core/seed_quality.py`, `core/elo.py`, `services/stats.py`) and, ironically,
ten bandits sit in `core/schedulers/` solving exactly this problem while pointed
only at operator selection. The scheduler-convergence harness is what would tell
you whether the adaptive version actually beats the fixed 30s — build the fuzzer
that can judge the algorithm, then build the algorithm.

**D5. Read the per-seed cost ledger we already keep** (Persistence Mechanics,
2026-08-29) — `meta["total_time"]` accumulates per-seed target time in
`Fuzzer.fuzz_one`, and all three of its readers (`Fuzzer._cull_queue`,
`Fuzzer.run`, `CorpusManager.auto_minimize_corpus`) divide it by `fuzz_count`
to recover a mean `exec_us`. The accumulated quantity is never read as such.
Three candidate consumers, ranked: stale-seed detection in
`StatsReporter._print_summary_seeds` (currently `fuzz_count >= 50`, a count where
cost is the question, and display-only so the change is nearly free); the energy
term in `SeedPicker._pick_boltzmann_seed` (`E = log(fuzz_count + 1)`); and the
age fallback in `EdgeTracker._maybe_prune`, which has no cost term at all.

**All three collapse to nothing if per-seed exec cost is tightly clustered on our
targets — measure that first**, and if it is, delete this entry and record the
measurement. Full analysis, including what does *not* port from that source, in
`docs/handover/handover_persistence_mechanics_2026-08-29.md`. **Effort S–M.**

---

## E — Corpus, reduction, crash triage

**E1. Crash message normalization and grouping** (`R5`) — `core/trace.py`
produces backtraces; what is missing is normalizing the message (strip
identifiers, source locations, numbers), grouping across workers, and caching
groups so previously-reported bugs don't resurface. The source's Solidity
campaign went from 157 AFL-unique crashes to 16 distinct locations. **Warning
worth repeating: backtrace parsing and grouping is where valid bugs get silently
dropped — test against known-distinct crashes.**

**E2. Grammar-aware reduction** (`#10`, Perses ICSE '18; ProbDD/WDD ICSE '25) —
reduce along grammar/token trees (~2% of ddmin output size); ddmin-to-fixpoint
alone shrinks ~68% further. Lands in `tmin.py`. **Effort L–M.**

**E3. OptiMin — corpus minimization as MaxSAT** (`I.2`, HexHive ISSTA'21) — edge
coverage as hard constraints, seed exclusion as soft constraints, solved with
EvalMaxSAT; markedly smaller corpora than greedy `afl-cmin`. Lands in
`services/minimize.py`, whose coverage mode is currently greedy set-cover over
edge maps. z3 is already a CI extra, and the paper's weighted variant (minimize
file size while maximizing edge hit counts) maps onto the rarity/crowding
weights from the edge-distribution work. **Effort L–M. Caveat straight from the
paper, do not oversell it: minimized-corpus *size* differences did not translate
to statistically significant bug or coverage differences — the win is iteration
rate.**

**E4. Root-cause crash clustering** (`#26`, Igor CCS '21) — minimize PoC traces
via coverage-reduction fuzzing, cluster by CFG similarity (Weisfeiler-Lehman
kernel). Kept 39 real bugs distinct where stack hashes inflate counts 10–100×.
Goes beyond the stack-hash dedup already shipped; reuses edge maps plus E2.
**Effort M–H.**

---

## F — Testing our own machinery

From the TigerBeetle source, whose framing is not about fuzzing *targets* but
about **fuzzing your own subsystems**: minimal data interfaces → drive them
through a seeded generator → assert an always-true invariant.

**F1. Minimal mutation interface** (`P1-4`) — the clearest structural finding in
the repo. `services/operators.py` (3,511 lines) references `self.f` — the
`Fuzzer` instance, itself 5,239 lines — **423 times**, but touches only **29
distinct attributes**, and the distribution is extreme:

| attribute | refs |
|---|---:|
| `self.f.max_len` | 133 |
| `self.f._rand_pool` | 116 |
| `self.f.corpus` | 17 |
| `self.f._cmplog` | 15 |
| `self.f._crash_mi` | 9 |
| everything else (24 attrs) | ≤ 5 each |

**249 of 423 references — 59% — are an `int` and a PRNG.** The essential
interface of the mutation layer is roughly `(max_len: int, rng: RandPool)` plus
read-only views of the corpus and cmplog tables; the rest is accidental
dependency, mostly lazily-constructed format-mutator singletons that could be
owned by the operators themselves.

The cost is already being paid: `tests/test_operator_smoke.py` has to build a
full `Fuzzer` — temp corpus dir, temp crashes dir, a compiled `targets/test_target`
binary — to call `_op_bit_flip` on a bytearray, which is why that test can only
afford one buffer instead of thousands. Fix the interface and that test becomes
a real fuzzer for free.

The prerequisite is done: `MutationContext` already replaces the `Fuzzer` in
`core/mutator_interface.py`'s `mutate()` and `is_available()`, closing the
`**ctx` leak while that interface still had no implementors. **What is not
started is the `operators.py` extraction itself** — threading `ctx` instead of
`self.f`, mechanical for most of it since 249 refs are a rename of
`self.f.max_len` → `ctx.max_len`.

Related and cheap while in there: `_havoc_table`, `_havoc_trials`,
`_elite_pool_corpus_len`, `_redqueen_sorted_version`, `_region_cache` and
`_invariants_corpus_len` are all derived caches of corpus state whose staleness
would be silent. `self._invariants` in `operators.py` plus
`tests/test_regression_hotpath_invariants.py` is the right pattern already —
each of those deserves a line in it.

**F2. Distribution assertions** (`P0-2`) — matklad's `route_decode` test looked
fine and generated **zero** valid codes in 100 million attempts; the fix was
counters plus `assert(counts.valid > 50)`. Nothing in our suite asserts anything
about the distribution of its own generated inputs. The clearest case is
`tests/test_operator_smoke.py`, which exercises every operator against a
*single* 32-byte buffer, `b"\x00\x01...\x07" * 4` — operators gated on structure
(`_op_der_*`, `nal`, `isobmff`, `protobuf`, `zip`, the 13 regularity operators)
decline on that buffer and the test still passes. It measures "the dispatch
table is wired up", not "the operators work".

The pattern is established in `tests/test_scheduler_convergence.py::TestHarnessCoverage`
and not yet applied elsewhere. Wherever a test generates inputs and branches on
validity, count both sides and assert both are non-negligible: the
`operator_registry` availability predicates and `MutatorBase.is_available` (a
permanently-declining operator should be a test failure, not a silent pass),
cmplog pair matching, and `core/structural_constraints.py` /
`core/field_constraints.py` (both satisfiable and violating instances). The
quick-and-dirty version is worth adopting as a review habit: drop an
`assert False` into a branch you believe is reached and confirm the suite goes red.

**F3. Random-method-order harnesses** (`P2-7`) — our subsystem tests call
methods in the order the `Fuzzer` happens to call them, which restricts them to
executions the real campaign produces. Instantiate one object, call every public
method in random order obeying only documented preconditions, and check one
weak but always-true invariant:

- `core/edge_tracker.py` — every tracked edge index stays within the map size;
  owner counts never go negative. (An owner-count bug on the SHM path has
  already been fixed once.)
- `core/bloom.py` — no false negatives, ever, under any interleaving of
  `add`/`query`/resize.
- `services/corpus_manager.py` — corpus entry count matches on-disk file count;
  every entry's edge set is a subset of the global map.
- `core/state_store.py` — `get`/`set`/`save`/`load`/`cleanup_legacy` in random
  order; `__len__` always equals the number of sections surviving a save/load cycle.

**F4. `--performance` mode** (`P2-8`) — we are closer to this than to anything
else in that source. `tools/gen_synthetic_target.py` is the right primitive and
its docstring already states the principle better than the source does: *ground
truth is known by construction rather than inferred from the target's behaviour,
which is what makes a false-negative rate measurable at all.* `--blocks`,
`--unstable N` and the provably-dead byte region are three known-by-construction
knobs; `tools/bench_havoc_subop.py`, `docs/sweeps/` and
`tests/test_bench_paired_stats.py` are the beginnings of the harness. Formalise
a fixed-profile campaign mode — pinned seed, pinned synthetic target, pinned
fault profile, N execs — reporting a counter table (execs, unique edges, corpus
size, operator fire counts, cmplog hits), then A/B features against it.
**Inherit the caveat verbatim: this is hard to turn into a test that fails
*only* on bugs, so run it as an experiment harness, not a CI gate.**

**F5. Component-set bisect** (`R1`, `tekknolagi/omegastar`, under 200 lines and
already extracted non-JIT-specific) — bisect the **set of enabled components**,
not the input. Needs a consistent reproducer, stable identifiers, the ability to
set the active list, and the ability to observe which were used; `REGISTRY` has
all four. The motivating case is already paid for:
`docs/handover/suite_segfault_z3_finalization_2026-08-16.md`, where a `SIGSEGV`
(`ip 0`, silent faulthandler) cost a session to hypotheses about BLAS threads
and shutdown-timing buffering before the cause turned out to be a vendored-zlib
link mismatch. An operator-set bisect answers "is any operator implicated at
all?" — no — in minutes. Applications: fuzzer-internal crash or hang → bisect
the enabled operator set; minimization hang → bisect the seed corpus; scheduler
misbehaviour → bisect the enabled scheduler set. Take the delta-debugging
detail: when a set reproduces but neither half does, hold one half fixed and
bisect the other, which is how operator *interactions* surface.

**Gating question, and it is F6:** bisect results are meaningless against a
non-deterministic reproducer. **Effort ~1–2 days once that is answered.**

**F6. Seed-discipline residue** (`P0-1`) — the pytest plugin landed and all 14
bare `Random()` calls are migrated and guarded by
`tests/test_seed_discipline.py`. The ~250 hardcoded seed literals were
deliberately not migrated. Whether deterministic replay under a fixed seed holds
today is an open question, and it gates F5.

**F7. Havoc sub-op availability** (`R5`) — `REGISTRY.available()` already does
enumerate-then-choose at the top level, but havoc's 11 inline branches in
`_apply_single_mutation` still select-then-guard, which is why guard failures
count as trials and why the 15% uniform floor exists to stop short-input-hostile
branches from starving. An availability predicate at that level would let the
floor shrink. Narrow, but it is the one place that argument still applies.

**F8. Repair as a stage** (`R5`) — checksum and field repair exist per-format;
making it a stage that runs once per mutated buffer, after the full mutation
round, would make all 155 operators coherent for free. **Must run after the
round, not per sub-mutation, or havoc pays it 2–16 times.** Open design
question to settle deliberately rather than by accident: repair rewrites bytes
the operator wrote, which muddies credit assignment — do repair-modified bytes
count against the operator?

**F9. `cycle_lock` regularity operator** (`R4`) — the 100-prisoners puzzle is
governed by the longest-cycle distribution in a random permutation. An operator
building a permutation as a single n-cycle (worst case for any bounded
pointer-chase) or as all fixed points stresses traversal-depth paths in targets
that follow index chains: hash probe sequences, linked-list-in-array, jump
tables. Sits beside `perm_lock` in `core/mutations/structured.py`,
length-preserving, same construction discipline as the other regularity
operators. Read the `_FORMAT_BOOTSTRAP_RATE` caveat in `docs/TODO.md` before
reading anything into its measured yield. **Effort ~2 hours.**

---

## G — Execution substrate

**G1. SAND — decouple sanitization from the fuzzing loop** (`I.1`,
`wtdcode/sand-aflpp`, ICSE'25, merged upstream as AFL++ PR #2288) — fuzz a plain
build; forward only inputs with a *unique execution pattern* to separately-built
ASan/MSan binaries. The sanitizer builds carry sanitizer instrumentation and a
forkserver but **no coverage instrumentation**, so they never touch the bitmap.
Authors state the approach is fuzzer-agnostic and easy to port. Lands in
`build_targets.sh` (a `--san` sibling of the existing `.so` targets),
`services/runner.py` (second executor), and the interesting-input gate in
`services/fuzzer.py`. Fits what we have: the bloom filter for execution dedup
already computes something close to the "execution pattern" predicate, and the
multi-build target scripts exist. **Effort M. Highest expected value of the
absent set.**

**G2. userfaultfd write-protect snapshots** (`#18`, uffd-fuzz 2022) —
pure-userland dirty-page tracking in wp-mode: pristine bytes memcpy'd back per
iteration, restore <2 µs, ~1.8× median over persistent-mode fork reset.
x86-64-only. Extends `afl_shim.c` and `inprocess.py`; must intercept
mmap/mprotect. **Effort M.**

**G3. Static rewriting of uninstrumented binaries** (`#17`,
E9Patch/E9Tool/E9AFL, `GJDuck/e9afl`) — instruction-punning trampoline injection
into stripped x86-64 ELFs at near-native speed, so vendored or shipped binaries
are fuzzable without a rebuild. An offline e9tool step reusing the shim runtime;
the grep-class targets are the motivating case. **Effort M.**

**G4. Generator fault-injection** (`#28`, Fuzztruction USENIX Sec '23) —
fault-inject compile-time-instrumented *generator* programs so outputs are
almost-valid, bypassing parsing/CRC/encryption checks wholesale. A different
route than the checksum-learning stack, not a replacement for it. Needs
per-target generator/consumer pairing. **Effort H.**

**G5. Full-VM snapshot reset** (`#29`, Nyx / AFL++ Nyx mode) — KVM full-state
snapshots thousands per second, running instrumented userland targets on a
vanilla kernel ≥5.11; `nyx_packer.py` is Python. **Only if slow-init or stateful
targets come to dominate. Effort H.**

---

## H — Solver

**H1. Concolic query piping** (`#14`, symcc/SymQEMU, bundled with AFL++) —
branch constraints collected during execution piped into the solver loop; goes
beyond redqueen encodings on nested comparisons. Feeds `smt_solver.py` /
`z3_lifecycle.py`; the concolic loop is already sketched at
`core/path_constraints.py:20`. **Effort M.**

**H2. JIT constraint evaluation** (`#25`, JIGSAW IEEE S&P '22) — branch
constraints JIT-compiled to native functions with Angora-style gradient descent
at ~600K–12M evals/sec; the evidence suggests demoting Z3 to a fallback. **The
cheap first step is a pure-Python numeric-gradient evaluator**, which answers
whether the direction is worth the JIT. **Effort M–H.**

---

## I — LLM-assisted

Local models only; all three sources ran without API dependence, and cost is not
the objection.

**I1. LLM-written generator programs** (`#23`, SeedMind / SeedSmith) — the LLM
writes *generator programs* refined by execution and coverage feedback
(<$0.5/harness), and the result works with any downstream fuzzer. Complements
the hand-written mutators for formats we have no mutator for. **Effort L–M,
cheapest of this group.**

**I2. Plateau-triggered generation** (`#22`, ChatAFL NDSS '24) — a local LLM
extracts message grammars, enriches the corpus with missing message types, and
generates targeted inputs when coverage saturates (+47% states vs AFLNet). An
orchestrator hook plus a local LLM server. **Effort M.**

**I3. Fuzzer-space evolution** (`#24`, ELFuzz USENIX Sec '25) — LLM-driven
evolution of generation-based fuzzers embedding grammar and semantic
constraints; ran fully locally on CodeLlama-13B and found 5 cvc5 0-days. An
alternative maintenance path for the mutator set rather than an addition to it.
**Effort M.**

---

## J — Harness work: the FFmpeg targets

This is a different axis from everything above — it improves the *targets*, not
the fuzzing loop, and the 2026-08-28 survey found it is where the cheapest
coverage is. FFmpeg ships six OSS-Fuzz harnesses (dec/dem/enc/bsf/sws/swr, 1,620
lines under `FFmpeg/tools/`); our `targets/ffmpeg_read.c` (749 lines) covers the
union of dem+dec and nothing else.

Status grepped against `a8ccf8c`, FFmpeg at `d411d9e`. Everything below is
**absent** unless noted.

| FFmpeg mechanism | Status here |
|---|---|
| Trailing parameter block | Absent. Attempted once as the "footer header" of `285d0fa`, since reverted — see J.1 for why that attempt was wrong. |
| `FUZZ_TAG` packet framing | Absent. One input = one `avformat_open_input`. No delimiter-aware operator in `core/mutations/generic.py` either. |
| Rotating pattern registers (`keyframes`, `flushpattern`) | Absent. No flush, no discard/keyframe flags, no reset schedule. |
| Allocation budget via `get_buffer2` | Absent. 0 hits for `max_pixels`/`max_samples`. |
| Deterministic iteration bound | Partial and wall-clock: `g_watchdog_budget_ms = 900` plus `total_packets > 500`. Neither bounds work *inside* one packet. |
| Seekable fuzzed I/O + declared filesize | Absent. `avio_alloc_context` gets `NULL` for both write and seek, so **all seek-dependent demuxer code is unreachable**. |
| Filename/extension synthesis for probing | Absent. `avformat_open_input(&fmt_ctx, NULL, NULL, NULL)`. |
| Decoder knobs from input | Absent. 0 hits for `err_recognition`, `lowres`, `idct_algo`, `skip_frame`, `flags2`, `workaround_bugs`, `strict_std`. |
| `extradata` injection | Absent. Our three `extradata` hits are `fuzz_touch` reads of demuxer output, not injection. |
| Parser stage (`av_parser_parse2`) | Absent. |
| Drain at end (`send_packet(ctx, NULL)`) | Absent. Delayed frames and the flush path are never exercised. |
| Contract assertion (`av_assert0(ret != AVERROR_BUG)`) | Absent here **and in every other target**. We have no oracle beyond crash/timeout/sanitizer. |
| bsf / sws / swr / enc coverage | Absent. Note `libswresample.a` is *already linked* into every ffmpeg target and never called — `fuzz_swr` costs no build change. |

### Commit sequence

Each step is independently testable and independently revertable. Do not
collapse them.

1. **`targets/fuzz_params.h`** — a saturating tail-block reader, plus
   `tests/test_fuzz_params.py` compiling it with gcc and asserting the three
   invariants of J.1 directly: short input → block skipped; truncated block →
   zeros, not an overrun; every random 1024-byte block → in-range values. This
   is the file `285d0fa` was missing, and its absence is why that commit did not
   compile.
2. **Work budgets** — `get_buffer2` override, `max_pixels`/`max_samples`,
   interrupt callback, accumulators. Watchdog demoted to a backstop, with the
   fired-bound recorded. Measurable on its own: run the existing ffmpeg corpus
   before and after and compare the timeout count.
3. **Tag framing + rotating registers** — packet loop rewrite,
   `dictionaries/ffmpeg.dict`. Coverage delta is the acceptance criterion, and
   with SGFuzz state transitions in the tree the state-transition count is the
   more sensitive of the two signals.
4. **Decoder knob surface** — the J.4 field list, wired to the block from step 1.
5. **Seekable I/O + filename synthesis** — `io_seek`, lying filesize, extension table.
6. **`fuzz_swr`** as a second entry point in the same `.so` via
   `--inprocess-func`; `fuzz_bsf` after it. Do **not** copy FFmpeg's
   one-binary-per-codec model.
7. **Contract assertions** across all targets, as its own commit — a new oracle,
   not an ffmpeg change.

**Steps 1–4 carry the value; 5–7 are optional for a first pass.**

**Why the parameter block goes at the tail, not the head:** the same reasoning
as the sqlite target's absent mode byte. A prefix displaces the format magic by
one byte, every sniffer stops matching, and the whole corpus is silently mutated
as flat bytes with no visible symptom. Read a fixed window off the **end**
instead, and skip it entirely when the input is shorter than the window.

The remaining per-step design detail lives in the 2026-08-28 survey, deleted by
this merge. Recover it with
`git log --diff-filter=D --name-only -- 'docs/handover/handover_port_candidate*'`
and `git show <commit>^:<path>`.

---

## Rejected — do not re-propose without reading why

Merged from four "not worth porting" lists. Each was considered and declined;
that is a different state from "absent".

**Persistence Mechanics: the dissipation density, the exponential filter, the
contention model, the six pillars.** Evaluated 2026-08-29; only the cumulative
ledger survived, as D5. `D = xi * K * f` is `perf_score` relettered —
`core/schedules.py` is already multiplicative and `cost = exec_us * input_size`
already exists in two places; adding a flux term is a category error, since a
fuzzer chooses its transition rate rather than observing it. The exponential
survival filter is the shipped `--boltzmann` arm. Appendix A's `rho / (1 - rho)`
contention model, applied to the bounded linear probe in `adapters/afl_shim.c`,
is strictly weaker than the drop rates already measured there against
`ffmpeg_read` at load 0.77, and linear probing wants Knuth's closed form rather
than a memoryless queue. The six pillars map onto subsystems without producing a
decision. Nothing ports from the accompanying demo scripts, which are matplotlib
illustration and do not faithfully implement the paper's own equation 8.

**Do not disable cmplog.** One source disables cmplog/redqueen, reasoning that
input-to-state machinery is built for byte-level mutation and only adds overhead
once a grammar mutator produces valid tokens. True for text-source targets;
false for png/lz4/zlib, where input-to-state is what gets past magic bytes and
checksums. If anything it argues for a **per-target** default — justify that
through `bench_paired.py`, not through the post.

**MetaMut as a code generator.** 884 LLM-generated mutators presuppose a typed
AST library per language, a pipeline per language, and (the source's own figure)
~7% needing manual fixes. Our operators are format-agnostic. What transfers is
the 15-verb catalog — `swap remove add duplicate negate modify inline wrap
unwrap reorder lift sink split merge toggle` — cross-producted against
structures our formats actually have (chunks, length fields, offsets, tables,
checksums), as a gap-finding exercise against the existing 155. An afternoon
with a spreadsheet, not a pipeline.

**`multifuzz` orchestration.** Exists to unify AFL++/honggfuzz/libFuzzer config;
we are our own fuzzer. The salvageable direction is the opposite one — syncing
with an AFL++ instance over a shared corpus dir (`-F`) for diversity.

**Row polymorphism.** An HM type-inference tutorial. The only thread: `ts-bank`
(A3) would match donors by `TSSymbol` equality, the crudest possible
compatibility relation, where row unification's four-case analysis is what a
real structural check looks like. Relevant only if A3 ever grows a typed donor
filter.

**PowerFuzz's TCFG / ordered edge trace.** The TCFG exists to *infer* a CFG when
the binary is inaccessible. `core/cfg.py` builds real CFGs from a real decoder.
An inferred structure with 97.66% node precision and **77.61% recall** is
strictly worse than what we already compute, and the ordered-edge-trace
instrumentation it needs (a ring buffer of edge IDs in `afl_shim.c`, on the hot
path) is real cost for a structure available statically.

**`tsgen` corpus generation.** The source's own central result is that mutation
beat generation, including at cold start where every arm began from an empty
corpus — a direct argument against a generator as a priority. Its remaining job
would be feeding A3's bank constructs to splice, which only matters if A3
happens first.

**Exhaustive enumeration over `RandPool`'s continuous methods.** Not possible;
scoped out explicitly rather than fudged.

**`std.testing.random_seed` verbatim, and ARR itself.** Language-specific and
consensus-specific respectively. The pytest plugin (F6) and D4 are the real ports.

**`av_force_cpu_flags(0)`.** In FFmpeg's own builds this reaches the scalar
reference implementations, which is real coverage. In ours `vendor_ffmpeg.sh`
already passes `--disable-x86asm`, so most of the SIMD is not compiled in and
the knob is close to a no-op. Revisit only if the vendored build ever enables asm.

**One binary per codec.** Use `--inprocess-func` (J step 6) instead.

**`error()` → `exit(1)` on allocation failure.** FFmpeg exits the process when a
harness allocation fails; under our in-process runner that takes the campaign
down rather than the test case — the same reasoning already documented for the
watchdog in `ffmpeg_read.c`. Return early instead.

**CoreSight/ETM** — requires ARM64 SoCs plus a u-dma-buf module; inapplicable to
an x86-64 host. **PEBS sampling** — dominated by C3 for coverage purposes.
**Snappy adaptive snapshots, GPTrace (ICSE '26)** — immature or heavyweight,
watch-list only. **DARWIN, MobFuzz, ParmeSan, DAFL/DeepGo/Lyso/Prospector,
HyLLfuzz** — overlap the existing GA/MOpt/hierarchical-bandit/directed-distance
stack; DeepGo's RL path-transition model is worth revisiting only if directed
mode plateaus.

---

## How much to trust the sources

Source quality varies sharply and should govern how much weight each prior gets.

- **fitzgen's structure-aware experiment** is the only source with proper
  statistics — 20 trials, Mann-Whitney U, ratio ± CI, p-values. Treat its
  conclusions as real. Its headline (mutation > generation, 36–49% at 5 min,
  1–2% at 24 h) is the basis for rejecting `tsgen`.
- **nowarp's compiler-testing post** is candid that it has no per-mutator
  ablation and reports no edges-per-hour. Its 100-ICE result is real but
  attributes credit to nothing. Weak priors.
- **PowerFuzz** reports single numbers per cell, no trials, no variance, no
  p-values; its "+22.7%" is unfalsifiable from the paper. Coverage is plotted
  against *number of inputs* while 10 power traces are captured and averaged per
  input, so a 10× throughput tax is invisible in every plot. And the reported
  coverage comes from ground-truth instrumentation, not from the 77.61%-recall
  signal the fuzzer actually saw. **Take nothing on its numbers.**
- **Binvariants** is peer-reviewed (FSE'26) with a real trophy case (hdf5,
  Bento4, StormLib, catdoc, gpmf-parser, camlpdf, audiofile, nconvert) — but
  only the repo README was read, not the paper.
- **Crash-count ground truth (Igor):** coverage-profile dedup inflates bug
  counts by 2–3 orders of magnitude and stack hashes by 1–2, which is the
  argument for E4.
- **Hidden-edge undercount:** AFL++ 4.35a's pcguard rewrite found ~5–8% of edges
  missed by vanilla LLVM sancov ("hidden" decisions). Our shim sits on the same
  callbacks — worth a probe against the vendored ffmpeg build.
- **AFL++ 4.20c changed the forkserver protocol** (new targets incompatible with
  old afl-fuzz). Matters only if we interop with stock AFL++ binaries.
- **Directed fuzzing is the field's active frontier** — ≥4 papers at USENIX Sec
  '25 alone (IDFuzz, Lyso, WDFuzz, ELFuzz).

---

## Open questions gating work here

- Does deterministic replay under a fixed seed hold today? Gates F5, and lowers
  variance for every `bench_paired.py` cell.
- Does C1's cmplog-operand invariant proxy produce a usable violation rate, or
  saturate immediately?
- Does B1's canonicalization have a meaningful binary-format analogue?
- Under F8, do repair-modified bytes count against the operator that was credited?
- Is there a text-format target actually in scope? A3 and B1 are both worthless
  without one.
- Should colorization be opt-out rather than opt-in? It is the precondition that
  makes the transformation solver's operand→offset mapping unambiguous, and the
  per-site PC keys make the candidate filter sharper than AFL++'s callback-level
  one. Measure the exec cost against the `--colorize` budget on a real target
  first. This is the single most actionable item the 2026-08-28 survey produced.
- Entropic and our Chao2 rewrite (`good_turing_estimate`) come from the same
  STADS framework and should be reasoned about together rather than as two
  unrelated estimators. Nobody has.
- **Still owed, and the excuse is gone:** the adaptive-havoc edges-per-hour
  comparison against `--no-adaptive-havoc`. `tools/bench_paired.py` exists. It
  should be the first thing pointed at any of this, before A3 adds an operator
  category that makes scheduler attribution harder.

---

## Verification recipe

`git am` the series onto a fresh clone at the base commit, build, run the full
suite, and compare against a **base run in the same container** — not against
the 7-failure figure from the author's machine. Without clang the base is 13
failures / 5665 passing; with clang, 33 / 5728. The distance, ICFG, scov and
tracecmp families skip rather than fail when clang is missing, which is the
whole difference.

Reproducing the ffmpeg target from scratch: `apt-get update` first (without it
`apt-get install clang` 404s on libc6-i386/libxml2-dev), then
`tools/vendor_ffmpeg.sh --nosan --minimal`, then
`tools/build_ffmpeg_ready.sh --minimal --no-vendor`. `--minimal` restricts the
build to mov/matroska/wav/aiff/flac/mp3/ogg demuxers and seven decoders — enough
for J steps 1–5, **not enough for a coverage comparison that means anything**.

Watch disk during the suite: run it in chunks with a private `TMPDIR` and clean
between chunks (~5 MB of scratch that way).

`tools/profile_hotpath.py` cannot drive the ffmpeg target — it has
`os.chdir("/home/dclavijo/my_code/fuzzer")` hardcoded and does not emit
`--inprocess-direct`/`--inprocess-func`. Any before/after measurement for J
steps 2 and 3 needs this fixed or needs another instrument.
