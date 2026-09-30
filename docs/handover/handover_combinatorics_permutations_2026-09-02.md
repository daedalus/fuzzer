# Handover — Combinatorics and Permutations Across `fuzzer-tool`

**Date:** 2026-09-02. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_combinatorics_permutations_2026-09-02.md`.
**Verified against:** `4c021daa`.

Litmus test for any item (original §0): a combinatorics addition must serve one
of (1) reaching worst-case program paths, (2) allocating budget to productive
operators, or (3) falsifying operator reachability via `ExhaustivePool`.

---

## 1. `_swap_tuple` (m>2) — wired at byte level only; yield unmeasured (orig. §10a.1)

**State:** `_swap_tuple(domain, rng, m, *, start=0)` (`core/mutations/generic.py`)
has one caller: `OperatorEngine._op_swap_bytes` (`services/operators.py`), which
takes m∈{4,5} with p=0.15. The per-format element swaps (`core/mutations/<format>.py`,
32 `_swap_pair` sites) are still m=2 only.

**Open:**
- No A/B of m>2 vs m=2 edge yield. The only positive signal (4/4 new-edge on
  ffmpeg isobmff) was m=2.
- Validity-cliff hypothesis (m permuted elements break m offsets, so parsers
  reject earlier) was *not* supported on `libsqlite3` / `libavformat`
  top-level box order. Untested: ZIP central directory, and a harness that
  dereferences `stco`/`stsz` entries deep enough to hit a stale offset.
- Whether to extend m>2 to structural element swaps.

**Constraints if extended:** draw m relative to n (n ranges ~4 to thousands);
keep m=2 as its own arm; use `m = 2+k`, not `2k` (odd m=3,5 minimise Cayley
diameter); `ExhaustivePool` enumerates `n!/(n−m)!` — n=20, m=5 exceeds
`DEFAULT_MAX_RUNS` and leaves `exhausted=False`.

**Acceptance:** fuzzgoat + one offset-table target, paired runs, edges/exec for
m>2-enabled vs m=2-only, control-vs-self per Hard Rule 46.

## 2. `ExhaustivePool` bulk gate is all-or-nothing (orig. §10b) — CLOSED 2026-09-26

Per-call *and* per-run path budget implemented (`max_bulk_paths_per_call`,
default 65,536); `allow_bulk=True` still bypasses both for larger cases.
Full writeup: `docs/handover/handover_coin_flip_bulk_budget_2026-09-26.md`
§1.

## 3. Coin-flip idiom blocks enumeration (orig. §10c) — mostly CLOSED 2026-09-26

All 80 fixed-literal-probability sites (of 93 total) rewritten to bounded
draws; operator table's `"continuous"` count dropped 34 → 2 (both genuine
continuous draws, not coin flips). 13 runtime-probability sites (a
scheduler's `epsilon`, a GA's `crossover_rate`, etc.) left open — no
decision made yet on discretizing a runtime float without reopening the
bulk-budget problem item 2 just closed for explicit bulk calls. Full
writeup, including the operator-by-operator before/after and the fallout
in tests that monkeypatched `.random()` directly:
`docs/handover/handover_coin_flip_bulk_budget_2026-09-26.md` §2.

Hard Rule 41 (no speed regression) not separately profiled this pass —
`randint(0, N-1) < K` and `random() < p` are both single calls into
`RandPool`'s own bounded-draw fast path (`randint` doesn't delegate to
`random`), so no regression is expected, but this wasn't benchmarked.

## 4. Operator Markov chain is first-order only (orig. §10d)

**Where:** `core/schedulers/op_monte_carlo.py` — `transition_counts[prev][next]`,
`transition_total`, `select_op` pairwise blend.

**Proposal:** sparse second-order table `(prev2, prev) -> next`, observed
triples only (dense is ~150³ ≈ 3.4M counters; Hard Rule 54).

**Open:** does second order add signal? Needs an A/B with `--pairwise-blend > 0`
before any wiring; persisted via existing state machinery if adopted.

## 5. Grammar derivation space: no skeleton set (orig. §10e)

**Where:** `core/grammar.py` `Grammar.generate` / `generate_boltzmann`.
Boltzmann sampling (size-uniform) has since landed; a precomputed set of
minimum representative derivations (one per derivation shape) for bootstrap
seeding has not.

**Open:** needs a finite canonical-form projection, which not every grammar
has. Decide whether bootstrap coverage justifies it given Boltzmann sampling.

---

## 6. Gap analysis of remaining combinatorics candidates (2026-09-29, read-only)

**Verified against:** `37caa730`. Code and handover reading only; nothing was
implemented or measured. Ranked against the litmus test at the top.

**Already in tree (do not re-propose):** t=2 covering array
(`core/covering_array.py`, used only by `covering_array_ihdr`), de Bruijn
(`structured.py`, `debruijn_cache.py`), Feistel, `_swap_tuple`,
`ExhaustivePool`, Kruskal count, greedy set cover (`minimize.py`,
`corpus_manager.py`), weighted MDS local search, Moser-Tardos
(`wfc_chunks`, opt-in `resample=mt|hybrid`), combinatorial bandits
(`op_cucb`, `op_c2ucb`, `op_topk`).

| # | Candidate | Litmus | Status |
|---|-----------|--------|--------|
| 1 | Failure-inducing combination search over covering-array rows (FIC-style): after a crashing row, run follow-up rows to isolate the minimal field pair/triple | 1, feeds `root_cause` | **Implemented 2026-09-29** (`core/failure_inducing.py`; PNG IHDR adapter in `covering_array_mutate.py`; `root_cause --isolate-png-ihdr`). Unit-tested with mocked oracles and validated against `targets/proto_target.c` (real ASAN crashes, see section 7); usefulness on a real campaign not measured. See section 7. |
| 2 | Covering-array extensions: (a) second formats (ISO-BMFF `tkhd`/`ftyp`, RIFF `VP8X` flags, gzip/zip flags); (b) t=3 for small k; (c) forbidden-combination constraints | 1 | **Implemented 2026-09-29/30** (see section 8): (a) gzip, WebP `VP8X`, ISO-BMFF `ftyp`, ZIP local header; (b) tested; (c) done. Not done: ISO-BMFF `tkhd`. Measure whether the IHDR arm earns any selection share first (open in `handover_covering_array_ihdr_2026-09-21.md`). For (c), Moser-Tardos-style repair only pays once constraints exist (`handover_moser_tardos_2026-09-28.md`). |
| 3 | Orthogonal / fractional-factorial designs (12-16 runs) instead of grid sweeps for open hyperparameters (PLL `kp`/`ki`, lock thresholds, `explore_floor`) | benchmark tooling, not core | **Implemented 2026-09-29** as `tools/lib/factorial_design.py`; no sweep has been converted to use it yet (section 8). |
| 4 | Rank/unrank (Lehmer code, combinadics) for m-tuple swaps: deterministic non-repeating enumeration and uniform sampling without replacement | 3 | **Library implemented 2026-09-29** (`core/combinadic.py`); not wired to `_swap_tuple` (m>2 yield still unmeasured, section 1). |
| 5 | Group testing (d-disjunct matrices) for which-bytes-matter inference, non-adaptive and parallel, vs ddmin in `tmin` / colorizer | 1 | **Implemented and benchmarked 2026-09-29**; not wired. Uses ~1.7-2.3x the oracle calls of binary splitting for 1-2 rounds instead of 9-13 (section 8). |

**Do not pursue:** Moser-Tardos beyond the shipped option (lost to restart
below ~0.88 table density); Ramsey, Sperner, Burnside, Prufer, Lyndon,
Steiner systems (fail the litmus test); combinatorial bandits (already
present).

## 7. Failure-inducing combination isolation (item 1, implemented 2026-09-29)

- **What:** `failure_inducing.isolate(fail_row, value_sets, fails)` finds an
  inclusion-minimal set of parameters (with values) of a failing row that
  still triggers the failure. It finds a passing companion that differs on
  every movable parameter, then does chunked greedy removal over
  `hybrid(S)` (failing values on `S`, companion's elsewhere) to a 1-minimal
  fixed point: about `k` probes for `k` parameters, memoized, budget-capped
  (`max_probes`, default 500). `FailureIsolator` holds the state (AGENTS
  rule 55); `isolate()` is the thin wrapper.
- **Outcomes:** `isolated`, `unconditional` (no passing companion in
  `companion_tries`: parameters do not explain the failure),
  `truncated` (budget hit; schema still fails but may be non-minimal),
  `not_failing` (row did not fail on replay).
- **Assumption, not enforced:** failure is monotone in `S` and the
  companion does not trigger a second failure (no masking). Set
  `verify_samples` to probe random rows containing the schema; `verified`
  False plus a `counterexample` row means the schema is not sufficient.
- **Wiring:** `isolate_png_ihdr_failure(data, fails)` rebuilds the PNG with
  alternative IHDR values from the `covering_array_ihdr` domains (CRCs
  recomputed by `serialize_png_chunks`). `root_cause --isolate-png-ihdr`
  (service kwarg `isolate_png_ihdr`) runs it after the byte-level ddmin,
  reusing the same same-signature crash oracle, adds `field_schema` to the
  result and an "IHDR fields responsible" line to the report.
- **Not done / caveats:** only PNG IHDR has an adapter; a second format is
  one `_FIELDS`-shaped table plus a serializer. It is a post-hoc analysis
  tool, not a fuzz-time operator, so the litmus test is "feeds
  root_cause", not selection share. The PNG IHDR adapter has no crashing
  PNG target in tree (`png_read` fails via libpng error exit 1, which
  `root_cause` does not count as a crash); no campaign or A/B was run.
- **Validated against a real crashing target (2026-09-29):**
  `targets/proto_target.c` built with `gcc -O0 -fsanitize=address` (its
  own `main`, stdin). Its four crash families have ground truth by
  construction; `root_cause --isolate-fields` returned exactly the needed
  fields in 16-23 probes each, with irrelevant fields excluded and
  `verified` True:
  `OPENVRLE`+0xDEAD (null deref) -> magic, ver, cmd, a, b, word;
  `OPENVWX` (heap overflow) -> magic, ver, cmd, a only (b, word, tail
  dropped); `OPENVSUM`+0xCAFEBABE (stack overflow) -> magic, ver, cmd, a,
  b, cs; `CLOSED` (abort) -> magic, ver, cmd. A non-crashing input
  reports "Crash not reproduced"; a spec longer than the input reports
  "too short". Locked in by `tests/test_isolate_fields_proto_target.py`
  (skips without gcc/ASAN). This checks correctness on monotone
  single-schema crashes; masking and multi-schema crashes are covered
  only by the mock-oracle unit tests.
- **Tests:** `tests/test_failure_inducing.py`, additions to
  `tests/test_covering_array_mutate.py` and `tests/test_root_cause.py`
  (98 pass together with `test_covering_array.py`). The rest of the suite
  was not run.

## 8. Items implemented 2026-09-29 (second pass)

- **Rank/unrank** (`core/combinadic.py`, `tests/test_combinadic.py`): lexicographic `unrank_perm/rank_perm/unrank_comb/rank_comb`, `sample_indices` (Floyd over big ints, 32-bit limbs plus 64 bias bits, because `RandPool.randrange` is one 32-bit draw and cannot reach a 3e16-sized space), `sample_perms/sample_combs`. Nothing calls it yet.
- **Covering array** (`core/covering_array.py`): `forbidden=[{param: value}]` on `generate/verify_coverage/missing_tuples/required_tuple_count`. Tuples containing a forbidden assignment are not required; single-parameter bans shrink the domain; rows are repaired by Moser-Tardos resampling (100-step budget, then the candidate is dropped). A tuple whose every completion is forbidden by multi-parameter constraints stays reported as missing rather than looping. t=3 is covered by tests (`TestStrengthThree`); row counts were not tuned.
- **`covering_array_gzip`** (`core/mutations/covering_array_gzip.py`): CM, FLG, MTIME, XFL, OS; 5 fields, gated on the gzip magic. Same open question as the IHDR arm: selection share never measured. ISO-BMFF, RIFF `VP8X` and ZIP followed on 2026-09-30 (below).
- **`tools/lib/factorial_design.py`**: Hadamard designs (Sylvester, Paley I, doubling; orders 4-64 that are constructible, 28 is skipped), `fold_over`, `main_effects`, `screen`. Main effects only; no interactions. No existing sweep script was rewritten to use it.
- **`core/group_testing.py`**, `tools/bench_group_testing.py`. OR oracle. Result of `python tools/bench_group_testing.py --trials 100` (mean oracle calls / sequential rounds, group testing vs level-order splitting):

  | n | d | GT tests | GT rounds | split tests | split rounds |
  |---|---|---|---|---|---|
  | 256 | 1 | 39.0 | 1.00 | 17.0 | 9 |
  | 256 | 4 | 98.1 | 1.05 | 50.3 | 9 |
  | 256 | 16 | 330.1 | 1.12 | 137.9 | 9 |
  | 4096 | 1 | 59.0 | 1.00 | 25.0 | 13 |
  | 4096 | 4 | 146.0 | 1.02 | 81.9 | 13 |
  | 4096 | 16 | 495.1 | 1.07 | 260.6 | 13 |

  It only pays when executions parallelise more than ~2x and latency is the cost. The baseline is binary splitting under an OR oracle, **not** the repo's `ddmin_edits`, whose oracle is conjunctive ("this subset still crashes"); the two are not directly comparable and no comparison against `tmin`/colorizer was made. A soundness bug was found and fixed while testing: an item in no pool survives COMP with no evidence, so every non-certain candidate is now tested individually.
- **Second-order operator chain** (section 4): `core/op_chain2.py` (sparse, 4096-context cap, evicts least-observed) and `MonteCarloScheduler(second_order_blend=)` / `--second-order-blend`. Off by default and bookkeeping-free when off (an always-on version measured ~10% slower per `record()`). Synthetic A/B (order-2 environment, 5 seeds, `pairwise_blend=0.5` baseline, control-vs-self checked): success rate 0.45 (range 0.09-0.67) baseline vs 0.57 (0.50-0.67) with second order; memoryless environment 1.0 vs 1.0. **No real-target A/B was run**, so the handover's condition for adoption is unmet. Not persisted across runs (the first-order matrix is not either, outside `save_transitions`).

- **Covering-array formats, second pass (2026-09-30)** (`core/mutations/covering_array_container.py`, `tests/test_covering_array_container.py`): `covering_array_webp` (RIFF+`WEBP`+`VP8X`: riff_size, chunk_size, flags, reserved, width-1, height-1), `covering_array_isobmff` (`ftyp` size, major brand, minor version) and `covering_array_zip` (first local file header: version, flags, method, comp/file size, name/extra length). One shared base class, one subclass per format, same round-robin scheme as gzip. Rows vs full grid: 49 vs 6480, 30 vs 90, 50 vs 16200. Offsets are pinned in tests against real `zipfile`-built ZIPs and hand-built WebP/MP4. **Not done:** ISO-BMFF `tkhd` (variable depth, needs a box walker), ZIP central directory/EOCD, CRC field. **Not measured:** selection share of any of the five covering-array arms on a real target, and whether any of them finds a bug the byte-level operators miss.

**Still open:** section 1 (m>2 swap A/B on fuzzgoat plus an offset-table target); section 3's 13 runtime-probability sites; section 5 (grammar skeleton set); the remaining covering-array formats.
