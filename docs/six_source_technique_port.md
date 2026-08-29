# Porting six external fuzzing sources into `daedalus/fuzzer`

Repo analysed: `github.com/daedalus/fuzzer` @ `2f848aa` (full clone, 2026-08-23).
Scale: 155 `_op_*` operators, 10 schedulers, ~90 modules under `core/`.

Sources reviewed:

| Tag | Source |
|---|---|
| **A** | [nowarp — Compiler Testing Part 1](https://nowarp.io/blog/compiler-testing-part-1/) (2026-04-24) |
| **B** | [fitzgen — A Structure-Aware Fuzzing Experiment](https://fitzgen.com/2026/06/01/structure-aware-fuzzing-experiment.html) (2026-06-01) |
| **C** | [bernsteinbear — Cinder JIT bisect](https://bernsteinbear.com/blog/cinder-jit-bisect/) (2022-10-18) |
| **D** | [FuturesLab/Binvariants](https://github.com/FuturesLab/Binvariants) (FSE'26) |
| **E** | [PowerFuzz](https://arxiv.org/html/2606.24692v1) (arXiv 2606.24692) |
| **F** | [bernsteinbear — Row polymorphism](https://bernsteinbear.com/blog/row-poly/) (2024-10-22) |
| **G** | [bernsteinbear — 100 prisoners](https://bernsteinbear.com/blog/understanding-the-100-prisoners-problem/) (2019-03-11) |

## Read this first

This document was drafted from the sources alone and then audited against live
source. **The audit killed most of it.** Eight of the fifteen items drafted were
already implemented, three of them by machinery strictly better than what the
source proposed. That is the same failure mode the TODO/bugreport audits keep
finding, except here the stale document was the one being written, before it was
ever committed.

The already-implemented items are pruned to a one-line list under Status. The
prose that follows covers only the items that survived.

## Status

The drafted items the audit found already covered by the tree are pruned from
this file — operator validation harness (`tools/measure_operators.py`),
statistical A/B harness (`tools/bench_paired.py`, McNemar exact on discordant
pairs, which is the correct test for a paired design and better than the
source's unpaired Mann-Whitney), candidate enumeration before selection
(`REGISTRY.available` / `OperatorEngine.build_ops`), crash-site triage
(`core/trace.py`), differential trace alignment (`core/root_cause.py`,
byte-level rather than trace-level but the same question), grammar-aware
mutation (`core/grammar.py`, `core/tree_mutator.py`), post-mutation repair
(`core/checksum_learner.py`, `core/int_checksum.py`, `core/crc32.py`,
`core/field_constraints.py`, `core/structural_constraints.py`), depth-prioritized
energy (`core/cfg.py` + `core/distance.py`, real static distance beating
PowerFuzz's inferred-depth proxy), and novelty signal (PPMD seed novelty,
honggfuzz power factors, per-edge max hit count). Listed by name only so nobody
re-surveys them; the per-item justification is in git history.

Four of them left a narrow residue that is NOT covered — see R5. Of what
remains, two items are demoted rather than queued (NG5, NG6).

| item | source | state |
|---|---|---|
| Component-set bisect | C | **not started.** Genuinely absent — the only `bisect` in `services/fuzzer.py` is the stdlib CDF lookup in the havoc sub-op table. |
| Invariant-violation feedback | D | **not started.** The nearest thing is the operator invariant audit in `docs/TODO.md`, which is about our operators, not the target's registers. |
| Ordered edge trace / path-prefix tree | E | **not started**, and mostly not wanted. See NG5. |
| Identifier canonicalization | A | **not started.** Cheap. |
| `cycle_lock` operator | G | **not started.** Cheap. |
| `tsgen` corpus seeding | A | **not started**, and demoted. See NG6. |
| Row polymorphism | F | nothing actionable. NG4. |

## What actually remains

Ranked by value-per-effort. Everything here should land behind an A/B flag in
the `--no-adaptive-havoc` shape and be measured through `bench_paired.py`.

### R1 — Component-set bisect (source C)

The only fully-absent item with an immediate, already-paid-for motivating case.

Bisect the **set of enabled components**, not the input. Requirements are a
consistent reproducer, stable identifiers, the ability to set the active list,
and the ability to observe which were used. We have all four for `REGISTRY`.

The motivating case is `docs/handover/suite_segfault_z3_finalization_2026-08-16.md`.
That `SIGSEGV` (`ip 0`, silent faulthandler) cost a session to hypotheses about
BLAS threads and shutdown-timing buffering before the cause turned out to be a
vendored-zlib link mismatch. An operator-set bisect answers "is any operator
implicated at all?" — answer: no — in minutes.

Applications: fuzzer-internal crash or hang → bisect the enabled operator set;
minimization hang → bisect the seed corpus; scheduler misbehaviour → bisect the
enabled scheduler set.

Two details worth taking. When a set reproduces but neither half does, hold one
half fixed and bisect the other — that is delta debugging, and it is how
operator *interactions* surface. And the implementation is already extracted
non-JIT-specific and copy-pastable at `tekknolagi/omegastar`, under 200 lines.

**Gating question:** deterministic replay under a fixed seed. Bisect results are
meaningless against a non-deterministic reproducer. The seed-discipline work
(P0-1 in `docs/tigerbeetle_four_fuzzers_port.md`) is a prerequisite, and it is
only partly landed — ~250 hardcoded seed literals were deliberately not
migrated.

**Effort:** ~1–2 days after the gating question is answered.

### R2 — Invariant-violation feedback (source D)

The highest-novelty item, and the one that fits an actual gap: we are rich on
the mutation side and conventional on the feedback side (edges, hit-count
buckets, per-edge max, cmplog).

Binvariants learns register-level likely invariants offline, then uses
**violations** as fuzzing signal alongside coverage. It requires ASLR disabled
for consistent basic-block addresses between learning and fuzzing — we already
did that work for context-sensitive edge coverage, so the prerequisite is in
place.

The repo's own unimplemented enhancements are the interesting part:

- *Adaptive learning during fuzzing*, applying invariant updates to a **copy**
  and committing only on `FSRV_RUN_OK`, so crashing or timing-out runs don't
  corrupt the invariant set.
- As invariants stabilize, violations get rarer, biasing seed selection toward
  early-stage test cases — requiring a `perf_score`/`top_rated` adjustment.
  That reweighting problem is what our schedulers are for. This is arguably a
  better home for adaptive RLI than AFL++.
- Cross-basic-block invariants; currently single-block only.

**Scoping.** Full register-level tracing is weeks. Start with invariants over
the **cmplog operand values already captured** in `core/cmplog.py`: learn ranges
per comparison site, flag violations. Reuses existing plumbing and answers
whether the signal is worth the full build. Note `core/checksum_learner.py`
already mines `f._cmplog.pairs` for a different purpose, so the access pattern
exists.

**Open risk:** the proxy may saturate immediately — if nearly every input
violates some learned range, the signal is noise. That is the first thing to
measure, before any integration.

**Effort:** ~1 week for the cmplog-operand proxy.

### R3 — Identifier canonicalization (source A)

~50 lines. Splice-style mutations shuffle identifiers, so results fail on
trivial cross-reference errors long before reaching anything interesting. Emit a
*second, canonicalized* corpus where every identifier is renamed to a uniform
deterministic pattern (`v0`, `v1`, …) and fuzz that; splices then cross-reference
correctly by construction.

Generalized: **canonicalize cross-references so splices don't produce
trivially-invalid inputs.** A `tools/` preprocessing step producing a sibling
corpus dir. Keep both, never overwrite.

Whether a binary analogue exists (normalizing internal offsets/IDs before
splicing) needs its own analysis — `core/field_constraints.py` is where that
would start. Do not assume it transfers.

**Effort:** ~half a day for the text case.

### R4 — `cycle_lock` regularity operator (source G)

The 100 prisoners puzzle is governed by the longest-cycle distribution in a
random permutation. An operator building a permutation as a single n-cycle
(worst case for any bounded pointer-chase) or as all fixed points stresses
traversal-depth paths in targets following index chains: hash probe sequences,
linked-list-in-array, jump tables.

Sits beside `perm_lock` in `core/mutations/structured.py`, length-preserving,
same construction discipline as the other regularity operators. Note the
`_FORMAT_BOOTSTRAP_RATE` caveat in `docs/TODO.md` before reading anything into
its measured yield.

**Effort:** ~2 hours.

### R5 — Residues of the already-done items

Small, individually cheap, listed together because none justifies its own
section.

- **Crash message normalization and grouping** (A). `core/trace.py` produces
  backtraces; what is missing is normalizing the message (strip identifiers,
  source locations, numbers), grouping across workers, and caching groups so
  previously-reported bugs don't resurface. The source's Solidity campaign went
  from 157 AFL-unique crashes to 16 distinct locations. Warning from the source
  worth repeating: backtrace parsing and grouping is where valid bugs get
  silently dropped — test against known-distinct crashes.
- **Campaign-level novelty rate** (E). Fraction of generated inputs exercising
  at least one previously-unseen edge. We have novelty *signals* but not this
  *metric*. It converges much faster than total edge count, so it would make
  `bench_paired.py` cells shorter and lower-variance. Compute it offline from
  recorded runs, not in the hot loop.
- **Havoc sub-op availability** (B). `REGISTRY.available()` already does
  enumerate-then-choose at the top level, but havoc's 11 inline branches in
  `_apply_single_mutation` still select-then-guard, which is why guard failures
  count as trials and why the 15% uniform floor exists to stop
  short-input-hostile branches from starving. An availability predicate at that
  level would let the floor shrink. Narrow, but it is the one place the source's
  argument still applies.
- **Repair as a stage** (B). Checksum and field repair exist per-format; making
  it a stage that runs once per mutated buffer, after the full mutation round,
  would let all 155 operators be coherent for free. Must run after the round,
  not per sub-mutation, or havoc pays it 2–16 times. Open design question:
  repair rewrites bytes the operator wrote, which muddies credit assignment —
  decide deliberately whether repair-modified bytes count against the operator.

### R6 — Tree-sitter operator category (source A)

Lowest priority of the surviving items, because `grammar.py` and
`tree_mutator.py` already cover the space. What tree-sitter adds is the
*ecosystem*: free maintained grammars for real languages, rather than
hand-written S-expressions or delimiter heuristics.

The 13 named strategies (`ts-del`, `ts-bank`, `ts-add`, `ts-swap`, `ts-shrink`,
`ts-lit`, `ts-dup`, `ts-ins`, `ts-range`, `ts-chaos`, `ts-kdel`, `ts-kins`,
`ts-stutter`) ship with hardcoded weights upstream; our schedulers would learn
the split per target instead. Seed from the source's weights, let the bandit
move them.

The genuine architectural addition is the **bank**: a `TSSymbol`-indexed pool of
subtrees harvested across the corpus, refreshed as the queue grows. Decisions
needed on refresh cadence, memory cap, and persistence across `--resume` (keyed
by symbol *name*, not numeric `TSSymbol`, same reasoning as the havoc sub-op
counts).

Caveats from the source: grammar quality dominates — grammars emitting `ERROR`
nodes on valid input degrade everything downstream, and `ts-chaos` patches that
rather than fixing it. And enabling `ts-ins`/`ts-add` without a good initial
corpus accumulates oversized entries; wire to the existing minimizer.

**Effort:** ~1 week including the bank. Only worth it if a real text-format
target is in scope.

## Non-goals

**NG1 — Do not disable cmplog.** Source A disables cmplog/redqueen, reasoning
that input-to-state machinery is built for byte-level mutation and only adds
overhead once a grammar mutator produces valid tokens. That holds for
text-source targets. It does not hold for png/lz4/zlib, where input-to-state is
what gets past magic bytes and checksums. If anything the post argues for a
**per-target** default; justify that through `bench_paired.py`, not through the
post.

**NG2 — MetaMut as a code generator.** 884 LLM-generated mutators presuppose a
typed AST library per language, a pipeline per language, and (the source's own
figure) ~7% needing manual fixes. Our operators are format-agnostic. What
transfers is the 15-verb catalog — `swap remove add duplicate negate modify
inline wrap unwrap reorder lift sink split merge toggle` — cross-producted
against structures our formats actually have (chunks, length fields, offsets,
tables, checksums), as a gap-finding exercise against the existing 155. An
afternoon with a spreadsheet, not a pipeline.

**NG3 — `multifuzz` orchestration.** Exists to unify AFL++/honggfuzz/libFuzzer
config; we are our own fuzzer. The salvageable direction is the opposite one —
syncing with an AFL++ instance over a shared corpus dir (`-F`) for diversity.

**NG4 — Row polymorphism.** Source F is an HM type-inference tutorial. The only
thread: `ts-bank` would match donors by `TSSymbol` equality, the crudest
possible compatibility relation, where row unification's four-case analysis is
what a real structural check looks like. Relevant only if R6 ever grows a typed
donor filter.

**NG5 — PowerFuzz's TCFG.** The TCFG exists to *infer* a CFG when the binary is
inaccessible. We have `core/cfg.py` building real CFGs from a real decoder. An
inferred structure with 97.66% node precision and **77.61% recall** is strictly
worse than the one we already compute. The ordered-edge-trace instrumentation it
would need (a ring buffer of edge IDs in `afl_shim.c`, on the hot path) is real
cost for a structure we can get statically. Dropped.

**NG6 — `tsgen` corpus generation.** Source B's central result is that mutation
beat generation, including at cold start where all arms began from an empty
corpus. That is a direct argument against a generator as a priority. `tsgen`'s
remaining job would be feeding R6's bank constructs to splice — which only
matters if R6 happens first.

## Honest accounting

- Nothing in this document has been implemented or benchmarked.
- **Still owed, unchanged:** the adaptive-havoc edges-per-hour comparison
  against `--no-adaptive-havoc`. `tools/bench_paired.py` now exists, so the
  excuse is gone. It should be the first thing pointed at it, before R6 adds an
  operator category that makes scheduler attribution harder.
- Source quality varies sharply and should govern how much weight each prior
  gets:
  - **B (fitzgen)** is the only one with proper statistics — 20 trials,
    Mann-Whitney U, ratio ± CI, p-values. Treat its conclusions as real. Its
    headline (mutation > generation, 36–49% at 5 min, 1–2% at 24 h) is the basis
    for NG6.
  - **A (nowarp)** is candid that it has no per-mutator ablation and reports no
    edges-per-hour. Its 100-ICE result is real but attributes credit to nothing.
    Weak priors.
  - **E (PowerFuzz)** reports single numbers per cell, no trials, no variance,
    no p-values; the "+22.7%" is unfalsifiable from the paper. Coverage is
    plotted against *number of inputs* while 10 power traces are captured and
    averaged per input, so a 10x throughput tax is invisible in every plot. And
    the reported coverage comes from ground-truth instrumentation, not from the
    77.61%-recall signal the fuzzer actually saw. Take nothing on its numbers.
  - **D (Binvariants)** is peer-reviewed (FSE'26) with a real trophy case
    (hdf5, Bento4, StormLib, catdoc, gpmf-parser, camlpdf, audiofile, nconvert),
    but only the repo README was read, not the paper.
- The eight items marked already-done were not verified by running anything.
  They were verified by reading module docstrings and call sites. If one of them
  turns out to be aspirational, this table is wrong in the optimistic direction.

## Open questions

- Does deterministic replay under a fixed seed hold today? Gates R1, and lowers
  variance for every `bench_paired.py` cell.
- Does the cmplog-operand invariant proxy (R2) produce a usable violation rate,
  or saturate immediately?
- Does R3's canonicalization have a meaningful binary-format analogue?
- Under R5's repair stage, do repair-modified bytes count against the operator
  that was credited?
- Is there a text-format target actually in scope? R6 and R3 are both worthless
  without one.
