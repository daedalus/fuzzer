# Handover — seventeen-source survey: mazes, diff, SSSP, noise, LNS, sampling, compression

Original 2026-09-06. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_seventeen_source_survey_2026-09-06.md`.
Verified against HEAD `4c021daa`. Rejections (D1 KKT MST, D2 DMMSY-SSSP, R1–R3
catalogues) stay rejected; see the original.

---

## B2 — Lexicase selection in `core/ga.py`

Source: Schulte et al., *Evolving Byte-Equivalent Decompilation from Big Code*,
§III-E. Fitness = vector of independent tests; selection filters the population
over a random test order until one survives. Keeps seeds unique in one dimension.

- Today: `FitnessFunction` is scalar (`w_novelty·novelty + w_diversity·diversity`);
  `select_parent` is a rank tournament. No `lexicase` in `src/`.
- Test vector = seed edge set. Existing approximations: rare-edge bonus
  (`seed_picker` `RARE_EDGE_OWNERS`/`RARE_EDGE_GAIN`), `ga.py::Speciation` LSH.
- Cost: O(pop · n_tests); n_tests = 8,189 edges on ffmpeg. Infeasible naive.
- **Open question (answer on paper first):** test set = all / random sample /
  rare edges only. If rare-only equals the rarity bonus, reject.
- Plan: `--ga-lexicase`, `--ga-lexicase-tests {rare,sample,all}` (default `rare`).
- Acceptance: paired A/B vs rarity bonus that can return "no difference".

Same paper, two companions (also `handover_pending_2026-09-06.md` P4-7):

- **Homologous crossover.** Align parents, map crossover point through the
  alignment. Unblocked (A1 shipped: affordability dispatch in
  `core/similarity.py`). Open: local (Smith–Waterman) vs global alignment; do not
  port the O(n·m) table — reuse the dispatch.
- **Diff-targeted mutation.** Draw havoc sites from a per-seed failure map with
  probability `TargetChance`, else uniform. Machinery exists
  (`core/colorizer.py`, cmplog); nothing gates havoc site selection on it.

## B3 — Growing Tree / Houston as scheduler parameterisation

Sources: jamisbuck.org/mazes, astrolog.org labyrinth page. Gating question
answered in `handover_maze_algorithms_2026-09-24.md` §1: **partially** —
`round_robin`/`weighted` map to oldest/random; `newest` (DFS) arm missing;
signal-driven arms not subsumed. Remaining work tracked there:

- `SeedNewestScheduler` (not in `services/seed_picker.py`) + A/B.
- Growing Tree mixture as a `tools/bench_paired.py` sweep axis.
- Houston ↔ saturation gate: vocabulary only, no consumer.
- Adopt astrolog vocabulary: "never reachable" vs "rarely selected" are distinct
  scheduler defects (cf. Hierarchical reach bug).

## C2 — `span_reverse` / `span_relocate` evaluation

Shipped opt-in (`--op-span-reverse`, `--op-span-relocate`;
`core/operator_registry.py` `_AVAILABLE`). Survey's validation rule unmet:
no paired A/B recorded. Acceptance: `bench_paired.py` run on png/jpeg/grep
showing win or explicit "no difference"; then decide default-on.

## A2 follow-up — cluster hierarchy in reports

`cluster_crashes` (`core/crash_metadata.py`) is exact bound-pruned single linkage
at a fixed `threshold=0.7`. The MST encodes every threshold; `services/report.py`
could show clusters at several granularities from one computation. Not built.
Optional.

## C3 — In-place mutate + undo (dancing-links mechanism)

Reject DLX for minimisation (set cover, not exact cover; `services/minimize.py`
greedy + `core/percolation.py::bootstrap_minimize_corpus` are correct). Take the
mechanism: evaluate candidate edits in place and undo, don't copy per candidate
(precedent: `core/gradient_descent.py` probe fix, 0.79 → 0.088 ms at 8 KiB).

- Audit sites: `core/weizz_tags.py::build_tag_map_from_cmplog`,
  `core/target_profiler.py::_extract_parser_tokens`,
  `core/gradient_cmp.py::gradient_cmp`,
  `core/analyzers/analyzer_distance.py::_compute_bb_values`,
  `core/colorizer.py::colorize_from_cmplog`.
- Acceptance: bit-identical output vs copy path; measured allocation drop.
  Where no copy-per-candidate exists, record and close.

## C4 — LNS refinements on stall (LEGO thesis §3.5 only)

None implemented (`critical_destroy`, `random_objective` absent).

- (a) Critical destroy: target the failing region (cmplog wall / colourisation
  map) instead of a random span.
- (b) Escalating neighbourhood: widen havoc radius on consecutive failure *on
  this seed*. Today `n_mutations` scales with `perf_score` and stall recovery
  (`services/operators.py`, honggfuzz-style 8–16), not per-seed failure count.
- (c) Randomise the objective on stall vs `--reseed-on-stall` (randomises input).
- **Open question:** A/B (c) against reseed on the same stall detector.
