# Handover — generator slots: audit of wfc / mcts / alphabeta / bootstrap, and what to add

**Date:** 2026-09-20
**Base:** `8216cedb` (`docs(deep-dive): document utf8_seq_mutate`)
**Status:** analysis (commit `075ddedf`) plus three fixes applied on top — see
"Resolved" below. Every number was measured in-container and every reproduction
is inlined in the appendix so it can be re-run from a clean checkout
(`pip install -e .`, numpy required).

## Resolved

| Item | Commit | Before → after (same probe) |
|---|---|---|
| **P1-1** bootstrap batch removal | `cc4e9b1a` | fixture loses edge 3 → 0 lost; 8/400 random post-greedy corpora lose coverage → **0/400** (seeds removed 149 → 140: the 9 kept are exactly the ones needed) |
| **P0-1** alphabeta arm | `bc9852ab` | 1 distinct seed, 0 non-root picks in 300 rounds → **146 distinct, 217 non-root**; 5.3 / 17.2 / 41.1 ms per select → **0.1 ms** at 605 / 5,465 / 27,305 nodes |
| **P2-3** cold-start seeds | `c5414f04` | see the addendum under P2-3 |
| **P2-2** Boltzmann / cycle-lemma wiring | `9fb172f2`, `a9098a76` | `generate(boltzmann=True)` wired into `Grammar.mutate`'s replacement paths; `cycle_lemma_dyck_bytes` wired as the `tree_generate` operator. (Landed between this handover and the P2-1 pass below; not recorded here until now.) |
| **P2-1** learned-adjacency WFC (isobmff/webp/riff/gif) | `fd7c5cf8` | `AdjacencyTable.from_corpus` had no caller, 0 formats beyond png/jpeg/bmp had a chunk-order table → `core/wfc_chunks.py` + `wfc_reorder_learned` operator, one learned table per format via `on_new_coverage`; re-parses 100% of calls, ≥95% of strict-mode calls change the input, 0.7–3 ms/call |
| **G0** bench arms for the generation group | `4164def0` | `bench_paired.py` had 0 arms for wfc/mcts/alphabeta/bootstrap → `wfc`, `elo-lineage` (baseline), `elo-mcts`, `elo-alphabeta`, `bootstrap`; `ARM_BASELINES` records each pairing and `tests/test_bench_paired_arms.py` holds every arm to "baseline + added flags" against the real parser. **No campaign was run**: these are registered arms, not results. |
| **P2-1 rollout** ogg/flv/nal/asf/mpegts/zip | this series | 4 of 10 formats covered → 10 of 11 (webm is the one left). Three defects in the shared reorder core, each pinned by a test that fails on the old code: `violate` mode indexed one cell past the grid when nothing was pinned last (`mutate` swallowed the `IndexError`, so a fraction of violate calls silently declined) and overwrote the pinned last cell when something was; chunks a collapse under-placed were appended *after* the pinned-last chunk; strict mode's "never emits an unobserved adjacency" was only checked on a re-parsed output, which re-segments positional formats (GIF: 2/60 calls emitted `extension→trailer`). Now: pins reserved by identity, skipped kinds placed at a table-legal slot, an illegal collapse retried (8 attempts) → **0 illegal adjacencies over 500 seeds × 10 formats** |
| scheduler adaptability test | this series | `…test_every_exported_operator_scheduler_is_adaptable` excluded 2 of the 5 `seed_*` exports by name and failed on Tang, SeedCanary and KruskalCount; now excludes by `seed_*`/`op_*` module, and pins ≥25 tested and that no `op_*` export is skipped |
| **P2-1 rollout** webm | this series | 10 of 11 formats covered → **11 of 11**. `parse_webm` always returns exactly the top two elements (EBML header, Segment), so the free-order sequence WFC can act on is one level down: the Segment's own children (SeekHead/Info/Tracks/Cluster\*/Cues/...). Added `WEBM_FORMAT`/`_try_webm` on that basis (element ID as tile, nothing pinned). Found a real, unrelated bug while writing the fixture: `core/mutations/webm.py::_parse_element`'s size-vint mask was only correct for a 1-byte size vint (`7 + 8*(length-1)` vs. the correct `7*length`), so any element ≥127 bytes — every real Segment/Cluster — failed to parse; fixed, and `tests/test_webm.py` is this module's first dedicated test file (it had none). `tests/test_wfc_chunks.py`'s generic per-format sweep also needed a self-sufficient stub serializer for webm's unbound global (mirroring `FLV_FORMAT`'s stub header) — its first version used an 8-byte-`0xff` "unknown size" that doesn't actually round-trip through this module's `_read_vint` (documented as a separate known gap in `test_webm.py`, not fixed, since real unknown-size elements elsewhere already use the 1-byte form that does). |

Still open: **P3 and P4** (design work gated on stated questions; nothing in
them was started).
G0 gaps: no arm for `--grammar-boltzmann` (needs a target that consumes a
grammar; none in `eval_set.py`) or `tree_generate` (no flag, always on when the
seed has brackets, so it is a source-edit arm like `boltzmann-cost`), and no
`ffmpeg_read` target set for the isobmff/riff/gif side of `wfc`.

Caveats on the rollout: tables are in process memory and rebuilt from
admissions, not persisted across restarts (the handover proposed
`state_store`); the strict-mode guarantee is empirical, not absolute — a chunk
with no table-legal slot in any of 8 collapses is kept rather than dropped;
and `on_new_coverage` learns even when `--wfc` is off (the NAL parser is a
per-byte Python scan, so admitting a large NAL-sniffed seed costs real time).

Tests run were the affected modules only. One failure remains, present on the
untouched base (confirmed via `git stash`):
`test_regression_no_op_mutations::…test_every_selectable_operator_is_reachable`
— now flagging `ffconcat_chunk_mutate`, `magicyuv_chunk_mutate`,
`rasc_chunk_mutate`, `shorten_chunk_mutate`, `tiff_chunk_mutate` never offered
by the sweep (a flaky subset each run; `rasc_chunk_mutate`/`tiff_chunk_mutate`
were already in it before this series). `test_regression_mutator_interface::…test_builtin_registry_mutators_are_known`
(P2-1 registered `wfc_reorder_learned` without listing it) was fixed earlier
in this series and stays fixed. Suite for the webm addition specifically:
`test_wfc_chunks.py`, `test_webm.py` (new — this module's first dedicated
test file), `test_regression_operator_registry.py`,
`test_regression_no_op_mutations.py`, `test_regression_format_op_gating.py`,
`test_regression_scheduler_operator_reach.py`,
`test_regression_mutator_interface.py`, `test_operator_smoke.py`,
`test_ffmpeg_port_mutators.py`, `test_new_format_mutators.py`, `test_wfc.py`
— 374 passed, the one pre-existing failure above. No full-suite run was made.

**Tiering** follows `handover_pending_2026-09-06.md` §0 — P0 defect a running
campaign can hit, P1 measured win with the correctness argument settled, P2
shipped-and-unwired, P3 design work gated on a stated question, P4 genuine but
blocking nothing, plus a Rejected section.

## Trigger

"We have wfc, mcts, alpha-beta, bootstrap generators — what other generators
can we add?" Before adding anything, the four existing ones were audited,
because two of them turned out not to do what their names and docstrings say.

---

## 0. What the four "generators" actually are

The features banner (`services/fuzzer.py:7770-7779`) files `wfc`, `mcts`,
`alphabeta`, `corpus-boost` and `bootstrap` under "Generation". Only one of
them synthesizes bytes.

| Name | Flag | What it does | Real slot |
|---|---|---|---|
| wfc | `--wfc` | Reorders chunks in PNG/JPEG/BMP (hard-coded adjacency tables); generates BMP pixel rows | **byte/structure synthesizer** |
| mcts | `--mcts` | UCT descent over `LineageTree`, returns a corpus seed | seed picker (Elo arm `mcts`) |
| alphabeta | `--alphabeta` | "alpha-beta over the lineage tree", returns a corpus seed | seed picker (Elo arm `alphabeta`) |
| bootstrap | `--bootstrap`, `--bootstrap-k` | `bootstrap_minimize_corpus`: k-rigid-core corpus reduction, run only inside `corpus_manager.py:1437` | corpus reduction |

Other things that do produce candidate inputs, so nothing below duplicates them:
`markov-gen` (`MarkovChain`/`MarkovEnsemble`, multi-order already exists),
`Grammar.generate` / `generate_boltzmann` / `TreeMutator`, `versifier_generate`,
35 `_generate_random_<fmt>` functions in `core/mutations/`, and
`seed_kruskal_count.generate`.

## 1. The prior every new generator has to beat

`docs/port-backlog.md` ("How much to trust the sources") accepts fitzgen's
structure-aware experiment as the one properly-powered source: **mutation beats
generation by 36–49% at 5 minutes and 1–2% at 24 hours**, and rejects `tsgen`
on that basis. Design rules that follow:

1. A generator is an **arbitrated arm** (Elo / bandit) that can decay to a
   near-zero share, never a fixed pipeline stage.
2. It costs on the order of a millisecond per output (see the WFC cost-law
   learning, `docs/learnings/2026-08-10-wfc-pixel-hang-cost-law.md`: bound work
   where the cost is incurred *and* at the caller — ≤64 tiles, bounded cells).
3. Its case is **cold start and plateaus**, not steady state.
4. It ships behind a measured A/B. `tools/bench_paired.py` has **no** wfc, mcts,
   alphabeta or bootstrap arm today (see G0).

---

## Summary table

| Tier | Item | Where | One line |
|---|---|---|---|
| **P0-1** | alphabeta arm only ever returns a root | `core/schedulers/seed_mcts.py:350` | 1 distinct seed in 300 rounds, 0 non-root picks; 5–41 ms/select |
| **P1-1** | bootstrap removes seeds in batches | `core/percolation.py:60-88` | Loses coverage in its own test fixture (edge 3) |
| **P2-1** | `AdjacencyTable.from_corpus` has no production caller | `core/wfc.py:137` | WFC reorder exists for 3 formats; 11 more have parse/serialize pairs |
| **P2-2** | `generate(boltzmann=True)` and `cycle_lemma_dyck_bytes` have no callers | `core/grammar.py:259`, `core/tree_mutator.py:582` | Shipped, tested, unreachable |
| **P2-3** | Cold-start seed synthesis is mostly unreachable | `services/seed_picker.py:926` | `bmp`/`zlib` branches dead; 4 of 35 generators reachable |
| **G0** | Bench arms for the generation group | `tools/bench_paired.py` | Prerequisite for every A/B below |
| **P3-1** | 2-D record-grid WFC | new | Only place WFC beats a bigram chain |
| **P3-2** | Covering-array (t-way) header-field generator | new | No covering-array code in the tree |
| **P3-3** | PUCT prior / RAVE over lineage ops | `seed_mcts.py` | Gated on G0 |
| **P3-4** | NRPA over operator sequences | new | Gated on an offline order-effect test |
| **P3-5** | Regime-gated generation share | `services/fuzzer.py:8375` | SUBCRITICAL only bumps havoc energy today |
| **P3-6** | Grimoire-style generalization → `Grammar` | backlog A1 | Gives the Boltzmann sampler a grammar source |
| **P4-1** | SMT solution sampling under field constraints | new | z3 is an optional dependency |
| **Rejected** | Coreness as a seed picker | — | Spearman 1.000 vs unique-edge count |

---

## P0-1. The alphabeta arm cannot select anything but a root

**Where.** `AlphaBetaMCTSSeedScheduler.select`, `core/schedulers/seed_mcts.py:350`.

**What it claims.** "Fuzz the maximizer, target response the minimizer"
(`seed_picker.py:408`). It is offered as a distinct alternative to `mcts`.

**What it computes.** `select()` loops `for root in roots`, evaluates
`_alpha_beta(root, …)`, and assigns `best_key = root`. The returned key is
always a member of `tree.roots()`. Descendants are only ever *evaluated*, never
*returned*. There is no exploration term (`self.exploration` is stored and
printed, never read), unvisited nodes evaluate to a neutral 0.5, and
`_last_path = [best_key]` so backpropagation credits only the root. Once one
root's mean exceeds 0.5 the arm is a deterministic argmax over roots.

The minimizer layer has no model of the target. It takes the min over children's
*running means of the same reward*, so one sterile child zeroes the subtree's
value — the search prefers subtrees where every child is productive, which is
close to none of them.

**Measured** (§ appendix A1; random forest, 3 roots/3-way branching/depth 4,
363 nodes, 300 select+update rounds, all seeds eligible):

| scheduler | distinct seeds picked | non-root picks | time |
|---|---|---|---|
| `MCTSSeedScheduler` | 170 | 228 / 300 | 0.00 s |
| `AlphaBetaMCTSSeedScheduler` | **1** | **0 / 300** | 1.00 s |

Per-select cost of the alphabeta arm grows with the tree: **5.3 / 17.2 / 41.1 ms**
at 605 / 5,465 / 27,305 nodes (64-step iterative deepening over the whole
forest, no transposition table).

**Consequence.** Elo arbitration hides this by demoting the arm, but it still
burns its share of matches and 5–41 ms per pick. The E3 A/B in
`handover_FINDINGS.md` ("alpha-beta MCTS vs plain MCTS", status MISSING) would
have measured a constant.

**Fix — do not just make it return descendants.** The min layer is the second
defect. Replace the policy with **Thompson-sampling tree descent** (a non-UCB
tree policy, consistent with the non-UCB direction of
`handover_non_ucb_schedulers_2026-09-13.md`):

- At each node the options are {stop here, if eligible} ∪ {each child}. Sample
  θ from a Beta (or Gaussian on the squashed reward) posterior per option —
  children use their *subtree-aggregated* statistics, "stop" uses the node's own.
  Descend into the argmax; stop when "stop" wins. Mirror the structure of
  `MCTSSeedScheduler.select` (`_self_uct` / `_roots`).
- Backprop credits the whole path, as `MCTSSeedScheduler.update` does.
- `from_dict` must tolerate the legacy `alphabeta` state-store blob
  (`visits`/`values`); treat it as a warm prior or discard it.
- Keep `--alphabeta` as the flag (avoid breaking `parallel.py`/CLI), or rename
  and alias. Update the docstring: it is no longer minimax.

**Tests.** A non-root pick occurs; >1 distinct seed over 300 rounds on a
bushy tree; per-select cost bounded at 27k nodes; an unproductive subtree's
share decays; legacy state loads.

**Effort S–M. Gate:** G0 arm for `mcts` vs `alphabeta`.

## P1-1. `bootstrap_minimize_corpus` removes mutually-redundant seeds together

**Where.** `core/percolation.py:60-88`; caller `services/corpus_manager.py:1437`.

**What it claims.** The caller's comment: "capture transitive redundancy that
single-pass greedy set-cover leaves behind". Redundancy removal should preserve
coverage.

**What it does.** Each round collects every seed with `< k` unique edges into
`to_remove` and deletes them all at once. Two seeds that each cover an edge only
the other also covers both have zero unique edges, so both go, and the edge is
lost. The module's own fixture pins this:
`test_transitive_redundancy_removal` (A={1,2}, B={2,3}, C={3,4}, D={4,5})
expects B *and* C removed — edge 3 was covered only by them.

**Measured** (appendix A4):

- the fixture: **edges lost = [3]**;
- 400 random corpora, each first reduced by greedy set-cover (what
  `corpus_manager` hands the pass): coverage lost in **8/400 (2%)**, **11 of
  68,091 edges (0.02%)**; 149 of 8,670 seeds removed.

Severity is low because greedy cover pre-filters most mutual redundancy, the
flag is opt-in, and there is **no coverage-restore pass after the bootstrap
step** (the "Recovered N seeds" logic runs before it).

**Fix.** For `k == 1`, remove one seed per round — the zero-unique seed with the
fewest edges, ties by index — and recompute, so the result is coverage-preserving
by construction. Keep batch removal for `k >= 2` (the "k-rigid core" is lossy by
definition) and say so in the docstring. Fixture update: A, C, D kept, B removed.
Add a property test: `k=1` never reduces the union of covered edges, on random
corpora.

**Effort S.**

## P2-1. WFC learns adjacency nowhere, and covers three formats

**Where.** `core/wfc.py:137` (`AdjacencyTable.from_corpus`); callers of
`WaveGrid`: `mutations/png.py:272`, `jpeg.py:431`, `bmp.py:325` only, all with
hard-coded tables (`ConstraintSet.png_chunks`, `jpeg_markers`). Repo-wide,
`from_corpus` is called from `tests/test_wfc.py` only.

**What is available.** These formats already have a parse/serialize pair and
therefore a chunk-kind sequence WFC could reorder: riff (`parse_riff_chunks`),
webp, isobmff (`parse_boxes`), gif (`parse_gif`), ogg, flv, asf, mpegts, webm,
nal, zip.

**Measured** (appendix A2, A3). Synthetic formats, learned-bigram 1-D WFC with
HDR/END pinned, 300 runs each:

| tile types | seqs learned from | cells | solved | novel (not an observed sequence) | ms/run |
|---|---|---|---|---|---|
| 6 | 3 | 12 | 300/300 | 300/300 | 1.2 |
| 12 | 4 | 20 | 300/300 | 300/300 | 3.5 |
| 20 | 6 | 30 | 300/300 | 300/300 | 6.4 |

On the real PNG table (6 types, 7 cells, IHDR…IEND pinned) there are exactly 20
valid sequences; WFC emits all 20, total-variation distance **0.168** from
uniform, 0.57 ms/run.

**Two honest limits.** (a) "Novel" here means a composition of observed
bigrams; whether real targets accept it is **unmeasured**. (b) With a nearest-
neighbour `AdjacencyTable`, 1-D WFC is a bigram Markov chain with pinned ends. It
cannot express cardinality ("exactly one IHDR") or non-local ("IDAT run
contiguous") constraints. The `wfc.py` docstring's "non-causal global
consistency" is true in 1-D only through endpoint pinning.

**Proposal.**

- `core/wfc_chunks.py`: `ChunkFormat(name, parse, serialize, kind, pin_first,
  pin_last)` and `wfc_reorder_chunks(fmt, chunks, table, rng, mode, max_len)`.
  Map result cells back to chunk payloads by kind, cycling when a kind is
  over-subscribed — mirror the result handling in `png.py::_wfc_reorder`.
- A per-format table store, updated on corpus admission from the admitted seed's
  kind sequence, persisted through `state_store`, capped (≤64 tiles, bounded
  cells, per the cost-law learning). Merge tables from all corpus seeds of that
  format.
- Two modes drawn per call: **strict** (observed bigrams only) and **violate**
  (copy the table, `add_forward(a, b)` for one *unobserved* pair, pin cells so the
  pair is actually used). The second is the fuzzing-relevant one: an ordering
  exactly one step outside anything the corpus has shown.
- One operator `wfc_reorder_learned`, category `"format"`, gated by the existing
  format sniffers (`operator_registry.py:556-591`, `_format_available`).
- Rollout order: isobmff (box order), riff/webp, gif, then ogg/flv/nal. zip last
  (central-directory consistency).

**Wiring checklist** (footprint of `6018c475`, `utf8_seq_mutate`): new module,
`core/operator_registry.py` (1 line), `services/operators.py` handler, tests.
Update `tests/test_regression_operator_registry.py`,
`test_regression_no_op_mutations.py`, `test_regression_format_op_gating.py`,
`test_regression_scheduler_operator_reach.py`.

**Acceptance.** Output parses with the format's own parser; differs from the
parent ≥ 95% of calls; ≤ 10 ms; strict mode never emits an unobserved adjacency;
violate mode emits exactly one. **Effort M for the helper, S per format.**

## P2-2. Two generators nothing calls

- `Grammar.generate(..., boltzmann=False)` (`core/grammar.py:259`): the default is
  False and no caller passes True. `Grammar.mutate` (`grammar.py:681-728`) and
  `TreeMutator` (`:1251`) call flat `generate`. The Boltzmann sampler removes the
  Dyck-path size bias measured in `handover_trees.md` §5; it is not on any path
  a campaign reaches.
- `tree_mutator.cycle_lemma_dyck_bytes` (`:582`, exported in `__all__`): no
  caller.

**Proposal.** In `Grammar.mutate`'s replacement paths, call
`generate(boltzmann=True, target_size=<span length>)` behind a flag (default
off until G0). For `cycle_lemma_dyck_bytes`, either wire it as a nesting-depth
operator that fires only where the seed has bracket delimiters, or delete it —
an unreachable function is a maintenance cost with no benefit. **Effort S.**

## P2-3. Cold-start seed synthesis is mostly unreachable

**Where.** `services/seed_picker.py:926` (`_format_aware_seed`), fed by
`core/target_profiler.py:715-800`.

- The profiler can emit **21 distinct** `format_signature` values (17 magic-byte
  names plus `json`, `archive`, `webp`, `protobuf` from symbol/string heuristics).
- `_format_aware_seed` has branches for `png, jpeg, gif, webp, webm, zip,
  protobuf, bmp, zlib, gzip`. **`bmp` and `zlib` are dead** — `format_signature`
  is assigned only in `target_profiler.py` and never to either value.
- Of the 35 `_generate_random_<fmt>` functions, 4 are called from cold start
  (webp, webm, zip, protobuf). `riff` is sniffed by magic and has a generator but
  no branch. Everything else — including `elf`, `pdf`, `pe`, `html`, `xml`,
  `json/text` — falls to **4–64 random bytes** (`seed_picker.py:991`).
- `rodata_strings`/`magic_bytes` reach the *dictionary* (`fuzzer.py:1573-1586`)
  and never a synthesized seed.

**Proposal.** Delete or make reachable the `bmp` and `zlib` branches; add `riff`
(and any format whose generator exists and whose signature the profiler can emit)
via a dict dispatch instead of an if-chain; for unknown formats, build the
fallback seed from `magic_bytes[0]` + a few dictionary tokens instead of pure
random bytes. Given §1, treat the fallback change as P4 until measured with
`tools/novelty_rate.py`. **Effort S.**

**Addendum (found while fixing this).** All six *constant* cold-start seeds were
rejected by real decoders, measured with Pillow / `zlib` / `gzip`: png (IHDR
declared 10 bytes, not 13, and no IDAT), jpeg (no DQT/SOF/DHT/SOS), gif (declared
a 256-entry colour table then ended), bmp (DIB header a field short), zlib (a
`78 9c` header prepended to a stream that already had one), gzip (zlib-wrapped
stream where raw deflate is required). The gzip branch *is* reachable (magic
`1f8b`), so a gzip target started from an undecodable seed. Fixed in
`core/minimal_seeds.py` (1x1 PNG/JPEG/GIF/BMP, valid zlib/gzip), each checked
against the format's parser and Pillow; `_format_aware_seed` is now a dispatch
over those plus the shipped generators, adds `riff`, and clamps to `max_len`.
`bmp`/`zlib` remain unreachable from the profiler and are documented as such;
adding a zlib symbol detector was rejected because libz symbols appear in most
binaries and would mislabel PNG/ELF-linked targets.

---

## G0. Prerequisite: bench arms for the generation group

`tools/bench_paired.py` has arms for garch/continuum but **none** for `wfc`,
`mcts`, `alphabeta`, `bootstrap` (checked by grep). Add them, plus one for each
new arm below, before claiming any win. Reuse the design in
`handover_FINDINGS.md`'s Boltzmann result: paired in time, arm chosen by source
tree via `PYTHONPATH`, `--lock-single-thread`, replicates over seeds. Size
against its power table — on bit-for-bit-deterministic saturated targets (zlib,
lz4, gzip) half the cells cannot produce a discordant pair; use `png_read` and
`ffmpeg_read` as E3 specifies.

## P3. Design work, each gated on a stated question

**P3-1. 2-D record-grid WFC.** Cells = fixed-width records at the stride
`core/periodicity.py` already detects (`seed_meta["record_stride"]`); tiles =
record classes; vertical adjacency = consecutive records. This is the one place
WFC's constraint propagation does more than a bigram chain. *Gate:* the fraction
of corpus seeds per target with a detected stride. If it is near zero on the
targets in `targets/`, drop it. `png.py` already carries a
`TODO(periodicity)` marking exactly this gap.

**P3-2. Covering-array (t-way) header-field generator.** No covering-array,
orthogonal-array, Sobol/Halton or Latin-hypercube code exists in `src/`. Header
fields with small value sets (PNG IHDR has 7: width, height, bit depth, colour
type, compression, filter, interlace) are currently explored by independent
mutation. A pairwise covering array over 7 fields × 5 values needs ≥ 25 rows
versus 5⁷ = 78,125 exhaustive and guarantees every value pair co-occurs.
Greedy IPO/AETG in `core/covering_array.py`; value sets from the existing
boundary tables (`png.py` `TIME_BOUNDARY_TUPLES` style, `structural_constraints`
`GOALS`); serialize through the format's serializer and fix dependent fields with
`field_constraints.repair`. Invalid combinations are the *point* (parser validity
checks). *Gate:* rows ≤ 3× the lower bound, and an A/B on a header-heavy target.

**P3-3. PUCT prior / RAVE over lineage ops.** PUCT: `Q + c·P·√N/(1+n)` with `P`
from normalized `subtree_weight`. RAVE/AMAF: share statistics across nodes by
operator using `LineageNode.child_ops`. Both are small edits to
`MCTSSeedScheduler`. *Gate:* G0 (mcts vs alphabeta-replacement vs each variant).

**P3-4. NRPA over operator sequences.** Nested rollout policy adaptation would
learn a policy over havoc stacks. `LineageNode.child_ops` already stores the op
list of every inbound edge, and `MonteCarloScheduler.transition_counts` tracks
pairs. *Gate (offline, no new runs):* on an existing campaign's lineage, test
whether consecutive-op pairs predict `node_weight > 0` beyond a permutation null.
If order carries no signal, NRPA has nothing to learn — stop.

**P3-5. Regime-gated generation share.** `CoverageRegime` is already computed
(`fuzzer.py:8362-8393`). On SUBCRITICAL the response is `havoc_energy_scale ×1.5`
(cap 5.0) plus stall recovery; no generator arm is boosted. Add a
`_generation_share` multiplier consumed by the markov seed arm and any new
generator operator. *Gate:* read the rest of `_maybe_trigger_stall_recovery`
(`fuzzer.py:6818`, not fully read here) to confirm it doesn't already route to
generation, then log hit-rate by regime to see whether generators pay in
SUBCRITICAL at all (§1 predicts they pay only there).

**P3-6. Grimoire-style generalization → `Grammar`** (`port-backlog.md` A1, "start
here"; not implemented — no `generalization` code in `src/`). It supplies the
missing grammar *source* for the Boltzmann sampler (P2-2): blank spans, re-run,
keep spans whose removal doesn't change coverage, emit alternation/repeat rules
into a `Grammar`. Gramatron (A2) is the alternative sampler. Cross-reference only;
no new proposal.

## P4

**P4-1. SMT solution sampling.** `field_constraints` and `structural_constraints`
*repair* a mutant to satisfy coupled fields; nothing *samples distinct solutions*
from a constraint set. z3 is an optional dependency (`smt` extra), so this is
unavailable by default. Blocking-clause enumeration is enough at these sizes.

## Rejected

**Coreness (round-of-removal) as a seed-picker arm.** Measured on 12 random
corpora × 40 seeds (appendix A5): Spearman **1.000** (min 0.999) against the
plain unique-edge count and 0.766 against `Σ 1/count(e)`. Coreness ≥ unique count,
and rank-identical to it unless the corpus has cascade chains — the chain fixture
in P1-1 is the case where they differ, and it is rare after greedy cover.

**Also not proposed:** a PPM/CTW-style variable-order generator
(`MarkovEnsemble` is already multi-order); LLM generator programs
(`port-backlog.md` I1/I2 — tracked there).

---

## What was not verified

- **No real-target A/B.** Nothing here measures edges or crashes on a live
  target. Every WFC number is synthetic; "novel" ≠ "accepted by the parser".
- The coreness result is on random synthetic corpora, which under-represent
  cascade structure.
- `_maybe_trigger_stall_recovery` was read only through its docstring (P3-5).
- The E3 status ("MISSING") is taken from `handover_FINDINGS.md`; I confirmed only
  the absence of arms in `bench_paired.py`.

## Appendix — reproductions

All scripts assume `cd <repo> && pip install -e . && python <script>`. Runtime is
seconds each.

### A1. Alpha-beta vs MCTS selection probe (P0-1)

```python
import random
from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.core.schedulers.seed_mcts import MCTSSeedScheduler, AlphaBetaMCTSSeedScheduler

def build(n_roots=3, fanout=3, depth=4, seed=1):
    r = random.Random(seed); t = LineageTree(); keys=[]; c=0
    def mk(): 
        nonlocal c; c+=1; return f"{c:016x}"
    frontier=[]
    for _ in range(n_roots):
        k=mk(); t.insert(None,k,["x"],[0],r.randint(0,5)); keys.append(k); frontier.append((k,0))
    while frontier:
        p,d=frontier.pop()
        if d>=depth: continue
        for _ in range(fanout):
            k=mk(); t.insert(p,k,["x"],[0],r.randint(0,5)); keys.append(k); frontier.append((k,d+1))
    return t, keys

t, keys = build()
elig=set(keys)
roots=set(t.roots())
print("nodes",len(keys),"roots",len(roots))
for name,S in [("mcts",MCTSSeedScheduler),("alphabeta",AlphaBetaMCTSSeedScheduler)]:
    s=S(rng=None) if False else S()
    picked=[]
    r=random.Random(7)
    import time; t0=time.time()
    for i in range(300):
        k=s.select(t,elig)
        picked.append(k)
        s.update(r.random()*4)
    dt=time.time()-t0
    nonroot=sum(1 for k in picked if k not in roots)
    print(name,"distinct picked",len(set(picked)),"non-root picks",nonroot,"/300",f"{dt:.2f}s")

print("--- scaling")
import time
for fan,dep in [(3,4),(3,6),(4,6)]:
    t,keys=build(n_roots=5,fanout=fan,depth=dep)
    elig=set(keys)
    s=AlphaBetaMCTSSeedScheduler()
    t0=time.time()
    for _ in range(5):
        s.select(t,elig); s.update(1.0)
    print(len(keys),"nodes:",f"{(time.time()-t0)/5*1000:.1f} ms/select")
```

### A2. 1-D WFC on the PNG table vs exact enumeration (P2-1)

```python
import itertools, collections, math, time
from fuzzer_tool.core.wfc import ConstraintSet, Tile, WaveGrid

adj = ConstraintSet.png_chunks()
types = [b"IHDR", b"PLTE", b"tEXt", b"gAMA", b"IDAT", b"IEND"]
L = 7
tiles = [Tile(name=t) for t in types]
idx = {t:i for i,t in enumerate(types)}

# exact enumeration of all valid length-L sequences pinned IHDR..IEND under the same table
valid=[]
def rec(seq):
    if len(seq)==L:
        if seq[-1]==b"IEND": valid.append(tuple(seq))
        return
    for t in types:
        if adj.compatible(seq[-1], t, "right") and adj.compatible(t, seq[-1], "left"):
            rec(seq+[t])
rec([b"IHDR"])
print("valid sequences (exact DP/enumeration):", len(valid))

def run(n):
    cnt=collections.Counter(); fail=0
    t0=time.time()
    for s in range(n):
        w = WaveGrid(tiles, adj, width=L, height=1)
        for j in range(len(tiles)):
            w.superpositions[0][j] = (j==idx[b"IHDR"])
            w.superpositions[-1][j] = (j==idx[b"IEND"])
        r = w.run(seed=s, max_restarts=3, ac3_budget=2000)
        row = r[0] if r else None
        if not row or any(c is None for c in row) or tuple(row) not in set(valid):
            fail+=1; continue
        cnt[tuple(row)]+=1
    return cnt, fail, time.time()-t0

N=3000
cnt, fail, dt = run(N)
tot=sum(cnt.values())
p=[cnt.get(v,0)/tot for v in valid]
u=1/len(valid)
tv=0.5*sum(abs(x-u) for x in p)
H=-sum(x*math.log2(x) for x in p if x>0)
print(f"WFC: N={N} fail/invalid={fail} coverage={sum(1 for x in p if x>0)}/{len(valid)} TV-from-uniform={tv:.3f} entropy={H:.2f}/{math.log2(len(valid)):.2f} bits  {dt/N*1000:.2f} ms/run")
top=cnt.most_common(3); print("top-3 mass:", [round(c/tot,3) for _,c in top])
```

### A3. Learned-bigram WFC on synthetic formats (P2-1)

```python
import random, time
from fuzzer_tool.core.wfc import AdjacencyTable, Tile, WaveGrid

def trial(n_types, n_seqs, L, seed):
    r=random.Random(seed)
    types=[f"T{i}".encode() for i in range(n_types)]
    # synthetic "format": HDR, then a random-walk over a sparse transition graph, then END
    hdr, end = b"HDR", b"END"
    allt=[hdr]+types+[end]
    succ={t:r.sample(types, k=min(3,len(types))) for t in types}
    succ[hdr]=r.sample(types,k=3)
    def gen():
        s=[hdr]
        for _ in range(L-2):
            s.append(r.choice(succ[s[-1]]))
        s.append(end); return s
    seqs=[gen() for _ in range(n_seqs)]
    table=AdjacencyTable.from_corpus(allt, seqs)
    # END adjacency: allow last type -> END as observed via from_corpus already
    tiles=[Tile(name=t) for t in allt]
    idx={t:i for i,t in enumerate(allt)}
    observed=set(tuple(s) for s in seqs)
    ok=novel=fail=0; t0=time.time(); N=300
    for s in range(N):
        w=WaveGrid(tiles,table,width=L,height=1)
        for j in range(len(tiles)):
            w.superpositions[0][j]=(j==idx[hdr]); w.superpositions[-1][j]=(j==idx[end])
        res=w.run(seed=s,max_restarts=3,ac3_budget=2000)
        row=res[0] if res else None
        if not row or any(c is None for c in row): fail+=1; continue
        ok+=1
        if tuple(row) not in observed: novel+=1
    return ok,novel,fail,(time.time()-t0)/N*1000

for n_types,n_seqs,L in [(6,3,12),(6,10,12),(12,4,20),(12,20,20),(20,6,30)]:
    ok,novel,fail,ms=trial(n_types,n_seqs,L,1)
    print(f"types={n_types:2d} seqs={n_seqs:2d} L={L:2d}: solved {ok}/300 novel-of-solved {novel}/{ok} ({100*novel/max(ok,1):.0f}%) fail {fail} {ms:.1f} ms/run")
```

### A4. Coverage loss from batch removal in bootstrap_minimize_corpus (P1-1)

```python
import random, sys
sys.path.insert(0,"tests")
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.percolation import bootstrap_minimize_corpus
import xxhash
def sk(b): return xxhash.xxh64(b).hexdigest()[:16]

def build(spec):
    corpus=[f"s{n}".encode() for n in spec]
    et=EdgeTracker(max_tracked_seeds=10000)
    for n,edges in spec.items():
        k=sk(f"s{n}".encode()); et.seed_edges[k]=set(edges); et.seed_hit_counts[k]={}
    return corpus,et,{f"s{n}".encode():set(e) for n,e in spec.items()}

def greedy_cover(spec):
    uncovered=set().union(*spec.values()); chosen=[]
    while uncovered:
        best=max(spec,key=lambda n:len(spec[n]&uncovered))
        if not spec[best]&uncovered: break
        chosen.append(best); uncovered-=spec[best]
    return {n:spec[n] for n in chosen}

# 1) the repo's own test fixture
spec={"A":{1,2},"B":{2,3},"C":{3,4},"D":{4,5}}
c,et,m=build(spec); kept,rem=bootstrap_minimize_corpus(c,et,k=1)
before=set().union(*m.values()); after=set().union(*(m[s] for s in kept)) if kept else set()
print("test_transitive_redundancy fixture: edges lost =",sorted(before-after))

# 2) random corpora, AFTER greedy set cover (what corpus_manager hands the bootstrap pass)
trials=400; hit=0; lost_tot=0; edge_tot=0; seeds_removed=0; seeds_tot=0
for t in range(trials):
    r=random.Random(t); U=r.randint(60,300); n=r.randint(20,120)
    spec={i:set(r.sample(range(U),r.randint(3,25))) for i in range(n)}
    cov=greedy_cover(spec)
    c,et,m=build(cov)
    kept,rem=bootstrap_minimize_corpus(c,et,k=1)
    before=set().union(*m.values()); after=set().union(*(m[s] for s in kept)) if kept else set()
    lost=before-after
    seeds_tot+=len(c); seeds_removed+=len(rem)
    if lost: hit+=1
    lost_tot+=len(lost); edge_tot+=len(before)
print(f"post-greedy-cover corpora: coverage lost in {hit}/{trials} ({100*hit/trials:.0f}%) ; edges lost {lost_tot}/{edge_tot} ({100*lost_tot/edge_tot:.2f}%) ; seeds removed {seeds_removed}/{seeds_tot}")
```

### A5. Coreness vs unique-edge count / rare-edge sum (Rejected)

```python
import random, numpy as np
from collections import Counter

def peel(seed_edges, k):
    alive=set(seed_edges)
    while True:
        cnt=Counter(e for s in alive for e in seed_edges[s])
        drop=[s for s in alive if sum(1 for e in seed_edges[s] if cnt[e]==1)<k]
        if not drop: return alive
        # remove one batch simultaneously (as bootstrap_minimize does per round)
        alive-=set(drop)

def coreness(seed_edges, kmax=12):
    c={s:0 for s in seed_edges}
    for k in range(1,kmax+1):
        alive=peel(seed_edges,k)
        if not alive: break
        for s in alive: c[s]=k
    return c

def rank(a):
    a=np.asarray(a,float); order=a.argsort(kind="mergesort"); r=np.empty(len(a)); 
    # average ranks for ties
    i=0; s=a[order]
    while i<len(a):
        j=i
        while j+1<len(a) and s[j+1]==s[i]: j+=1
        r[order[i:j+1]]=(i+j)/2; i=j+1
    return r
def spearman(x,y):
    rx,ry=rank(x),rank(y)
    if rx.std()==0 or ry.std()==0: return float("nan")
    return float(np.corrcoef(rx,ry)[0,1])

rows=[]
for trial in range(12):
    r=random.Random(trial); N=40; universe=400
    seed_edges={}
    shared=r.sample(range(universe),80)
    for i in range(N):
        kind=r.choice(["redundant","random","private","hub"])
        if kind=="redundant": e=set(r.sample(shared,r.randint(10,30)))
        elif kind=="random": e=set(r.sample(range(universe),r.randint(5,40)))
        elif kind=="private": e=set(r.sample(range(universe),r.randint(3,10)))|{1000+trial*100+i,2000+trial*100+i}
        else: e=set(r.sample(shared[:20],r.randint(8,20)))|set(r.sample(range(universe),4))
        seed_edges[f"s{i}"]=e
    cnt=Counter(e for s in seed_edges.values() for e in s)
    keys=list(seed_edges)
    uniq=[sum(1 for e in seed_edges[s] if cnt[e]==1) for s in keys]
    rare=[sum(1/cnt[e] for e in seed_edges[s]) for s in keys]
    size=[len(seed_edges[s]) for s in keys]
    c=coreness(seed_edges); cc=[c[s] for s in keys]
    rows.append((spearman(cc,uniq),spearman(cc,rare),spearman(cc,size),len(set(cc))))
a=np.array(rows,float)
print("coreness vs   unique-edge-count  rare-edge-sum  |S|   distinct-levels")
print("mean Spearman:  %.3f            %.3f         %.3f   %.1f"%tuple(np.nanmean(a,axis=0)))
print("min  Spearman:  %.3f            %.3f         %.3f"%tuple(np.nanmin(a[:,:3],axis=0)))
```
