# Handover: GitHub fuzzer survey — port candidates (2026-08-28)

**Base commit:** `300d649` ("Count comparisons per call site, not only per callback")
**Trigger:** survey of other fuzzers on GitHub for features worth porting.
**Predecessor:** `docs/web_research_port_candidates_2026-08.md` (2026-08-25). That
doc's Tier tables are still the master list; this one records a *second* pass with
a different source set (AFL++ `docs/features.md`, LibAFL component list, libFuzzer,
HexHive artifacts, `wtdcode/sand-aflpp`) and, more importantly, the audit of the
shortlist against the live tree.

## Headline finding: most of the obvious shortlist is already implemented

The first-pass shortlist from the sources was: Redqueen colorization, CmpLog
transformation solving, MendelFuzz deterministic-stage pruning, Entropic power
schedule, autotokens. **Four of the five already exist in the tree.** Anyone
picking this up should grep before writing anything — the survey sources describe
these as AFL++/libFuzzer novelties, and they read as gaps until you look.

| Candidate | Source | Status in tree |
|---|---|---|
| Redqueen colorization | AFL++ `src/afl-fuzz-redqueen.c`, RUB-SysSec/redqueen | **Present** — `core/colorization.py` (binary-search over ranges, path-checksum preserved, taint regions returned); driven from `services/fuzzer.py::_colorize_seed` (~4404), cached per seed. **Off by default** (`--colorize`, `colorize_max_execs=512`). |
| CmpLog transformation solving | AFL++ `-l 3` / `AFL_CMPLOG_TRANSFORM` | **Present** — `core/rq_encodings.py` (plain substitution, zero/sign extension, ASCII-number, C-string termination, split 64-bit words) driven by `services/operators.py::_op_redqueen_xform` (~486), with single-byte XOR/arith/boundary/hex/toupper/tolower fallback. |
| MendelFuzz / SkipDet | HexHive/MendelFuzz-Artifact; AFL++ default `skipdet` | **Present** — `core/skipdet.py` (`SkipDetector`, `trace_mini_from_edges`, inference stage with block flips), wired at `services/fuzzer.py:1422`. Note the adaptation already documented in that module: our `edge_id` is a `ctx ^ prev_loc ^ cur_loc` hash, not a map index, so the positional bitmap SkipDet wants has to be synthesised. |
| Entropic power schedule | libFuzzer `-entropic` (FSE'20, STADS) | **Present** — landed 2026-08-24, `core/schedules.py::SeedScorer._entropic_factor`, `--schedule entropic`. Worth knowing: Entropic and our Chao2 rewrite (`good_turing_estimate`) come from the same STADS framework, so they should be reasoned about together, not as two unrelated estimators. |
| autotokens | AFL++ LTO auto-dictionary | **Present** — landed 2026-08-24, `import_corpus.py`, `import --autotokens`. |

Only two of those are default-on. **Colorization being opt-out-by-default is the
single most actionable item in this doc**: it is the precondition that makes the
transformation solver's operand→offset mapping unambiguous, and we now have
per-site PC keys (`300d649`) that make the candidate filter sharper than AFL++'s
callback-level one. Before flipping the default, measure the exec cost against
the `--colorize` budget on a real target (ffmpeg recipe in
`docs/handover/`-adjacent notes and `docs/TODO.md`).

## Genuinely absent — ranked

### 1. SAND — decouple sanitization from the fuzzing loop
`wtdcode/sand-aflpp` (ICSE'25, merged upstream as AFL++ PR #2288). Fuzz a plain
build; forward only inputs with a *unique execution pattern* to separately-built
ASan/MSan binaries. The sanitizer builds carry sanitizer instrumentation and a
forkserver but **no coverage instrumentation**, so they never touch the bitmap.
Authors state the approach is fuzzer-agnostic and easy to port.
- Lands in: `build_targets.sh` (a `--san` sibling of the existing `.so` targets),
  `services/runner.py` (second executor), the interesting-input gate in
  `services/fuzzer.py`.
- Fits: we already have the bloom filter for execution dedup and multi-build
  target scripts; the "execution pattern" predicate is close to what the bloom
  path already computes.
- Effort: M. Highest expected value of the absent set.

### 2. OptiMin — corpus minimization as MaxSAT
`HexHive/fuzzing-seed-selection` (ISSTA'21). Edge coverage as hard constraints,
seed exclusion as soft constraints; solved with EvalMaxSAT. Produces markedly
smaller corpora than greedy `afl-cmin`.
- Lands in: `services/minimize.py`, whose coverage mode is currently greedy
  set-cover over edge maps.
- Fits: z3 is already a CI extra; the paper's weighted variant (minimize file
  size while maximizing edge hit counts) maps onto the rarity/crowding weights
  from the edge-distribution work.
- Effort: L–M. Caveat from the paper: minimized-corpus *size* differences did not
  translate to statistically significant bug/coverage differences — the win is
  iteration rate, not coverage. Do not oversell it.

### 3. Gramatron — grammar automatons instead of parse trees
`HexHive/Gramatron` (ISSTA'21); also vendored as `custom_mutators/gramatron` in
AFL++. Restructures the CFG into an FSA so an input is a *walk*; splice becomes a
state-matched cut of two walks. Claims unbiased sampling from the input state
space and much more aggressive mutation than parse-tree operators.
- Lands in: `core/grammar.py` alongside `TreeMutator`/`SubtreePopulation`; the
  `.gram` loader is the input side already.
- Caveat: exactness holds only for non-self-embedding grammars; self-embedding
  rules give a subset of the language.
- Effort: M.

### 4. Grimoire-style generalization
LibAFL `GeneralizationStage`. Blank spans of an input, re-execute, and keep the
spans whose removal does not change coverage — yields structure (and
recombinable tokens) with **no grammar supplied**. Complements the ~155
hand-written format mutators exactly where we have no format mutator for a target.
- Lands in: a new stage next to the deterministic/havoc schedule; reuses the
  colorization executor loop almost verbatim (same "mutate, compare path
  checksum" shape).
- Effort: M. Cheapest of the structure-aware options because the plumbing exists.

### 5. FairFuzz-style rare-branch masks
Compute, per rare edge, which byte positions can be mutated while still hitting
it, and restrict mutation to the complement.
- Fits: `_edge_owner_count` rarity is now correct (edge-distribution work,
  `0afc439`), so the "which branch is rare" half is done; only the mask half is
  missing.
- Effort: L–M. Lower confidence than the others — measure against the existing
  rarity bonus before committing, the two may overlap.

Not carried forward: N-gram coverage, trace-div/trace-gep, Nyx, Intel PT, LLM
paths — all already tabled in `docs/web_research_port_candidates_2026-08.md`
with the same or better analysis.

## Open threads inherited by whoever picks this up

Verified against `300d649`, all still open:

1. **`_record_cmp_progress` is still callback-granular** (`services/fuzzer.py:3070`,
   called at `:3543`). There are only 27 callback buckets, so the reward it feeds
   is 27-dimensional. The per-site PC table from `300d649` is what makes it dense.
   The measurement that justifies it: a target with one always-satisfied and one
   never-satisfied `memcmp` reads as `memcmp (4, 3)` per callback — 75% satisfied,
   no wall visible — but `(3,3)` and `(1,0)` per site. The bucket inverts the
   reading, it does not merely blur it.
2. **Weight-cache invalidation** (`services/seed_picker.py:1110`):
   `cache_key = (corpus_version, f.exec_count // 50)` with
   `corpus_version = len(f.corpus)`, so every corpus admission forces a full
   recompute (measured: 271 recomputes in 1500 execs, one per 5.5 execs, against
   the intended one per 50). Not fixable without a semantic decision, because
   `weights` is positionally bound to `f.corpus`: either pad new seeds with a
   neutral weight until the next 50-bucket, or splice per-seed weights onto the
   cached list (needs the pass aggregates and the Pareto front).
3. **Positional-argument defect in three generators** — still present:
   `webp.py:132` calls `_generate_random_webp(max_len, rng=...)` against
   `def _generate_random_webp(self, _chunks=None, max_len=4096, rng=None)`;
   same shape at `zip.py:260` / `zip.py:405` (`_doc`) and `isobmff.py:306` /
   `isobmff.py:422` (`_boxes`). `max_len` lands in the first parameter and the
   real `max_len` keeps its default. Silent — no exception, just an ignored bound.

## Method note

Sources are web-only; effort estimates are guesses until a per-item audit against
live code happens. What is *not* a guess is the status column in the first table
— that was grepped against `300d649`. Repeat that step before starting any item
here: this pass began with five "gaps" and four of them were already in the tree.
