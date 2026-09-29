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
| 1 | Failure-inducing combination search over covering-array rows (FIC-style): after a crashing row, run follow-up rows to isolate the minimal field pair/triple | 1, feeds `root_cause` | Not started. Nothing equivalent exists; stdlib-only. |
| 2 | Covering-array extensions: (a) second formats (ISO-BMFF `tkhd`/`ftyp`, RIFF `VP8X` flags, gzip/zip flags); (b) t=3 for small k; (c) forbidden-combination constraints | 1 | Not started. Measure whether the IHDR arm earns any selection share first (open in `handover_covering_array_ihdr_2026-09-21.md`). For (c), Moser-Tardos-style repair only pays once constraints exist (`handover_moser_tardos_2026-09-28.md`). |
| 3 | Orthogonal / fractional-factorial designs (12-16 runs) instead of grid sweeps for open hyperparameters (PLL `kp`/`ki`, lock thresholds, `explore_floor`) | benchmark tooling, not core | Not started. |
| 4 | Rank/unrank (Lehmer code, combinadics) for m-tuple swaps: deterministic non-repeating enumeration and uniform sampling without replacement | 3 | Not started. Depends on item 1 above (m>2 yield unmeasured). |
| 5 | Group testing (d-disjunct matrices) for which-bytes-matter inference, non-adaptive and parallel, vs ddmin in `tmin` / colorizer | 1 | Speculative. Benchmark against ddmin first. |

**Do not pursue:** Moser-Tardos beyond the shipped option (lost to restart
below ~0.88 table density); Ramsey, Sperner, Burnside, Prufer, Lyndon,
Steiner systems (fail the litmus test); combinatorial bandits (already
present).

