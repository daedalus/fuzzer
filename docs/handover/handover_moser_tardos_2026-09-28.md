# Handover: Moser-Tardos in the fuzzer

> **Status (2026-09-28): analysis only. Nothing implemented.** Repo HEAD at time of analysis: `bf4ada86`.

## Terminology (read first)

**"LLL" in this document means the Lovász Local Lemma, NOT Lenstra–Lenstra–Lovász lattice reduction.**

The repo already has the other LLL (`core/lattice.py::lll_reduce`, used by `core/lcg_recovery.py`, `analyzers/analyzer_prng_state_learner.py`, `edge_diagnostic` `lll_rows`). It is unrelated to this work. A grep for `LLL` in this repo returns only lattice hits; Moser-Tardos and the Lovász Local Lemma appear nowhere. To avoid confusion, this document writes **"Lovász Local Lemma"** in full or **"local lemma"**, never bare "LLL".

## Background

- **Moser-Tardos**: constructive Lovász Local Lemma. Given independent random variables and "bad events" each depending on a subset of them: sample all variables; while some bad event holds, resample only the variables of that event. Terminates fast in expectation if the local lemma condition holds.
- **Local lemma (symmetric) condition**: `p * e * (d + 1) <= 1`, where `p` = max probability of a bad event and `d` = max number of other events any event shares variables with.
- **Permutation variant** (Harris-Srinivasan): the resampling analogue when the state space is permutations, not independent variables. Resampling is done by swaps.

## Verdict

| Module | Fit | Reason |
|---|---|---|
| `core/wfc_chunks.py::wfc_reorder_chunks` | **Good (only real candidate)** | Illegal adjacencies are local bad events; today's fix is full restart. |
| `core/covering_array.py` | Marginal | AETG greedy already works; Moser-Tardos only useful with forbidden value combos. |
| `core/field_constraints.py::repair` | None | Deterministic functions of data, not random variables; cycles solved exactly (GF(2) CRC, z3). |
| `core/structural_constraints.py` | None | Arithmetic goals (offset+size wrap, TLV nesting) are solved for, not sampled. |

## Candidate: `wfc_reorder_chunks`

Current behaviour (`wfc_chunks.py`):
- Runs up to `_COLLAPSE_ATTEMPTS = 8` complete 1-D WFC collapses (`_collapse_cells` -> `_map_cells`).
- Scores each with `_illegal_adjacencies` (count of consecutive chunk pairs the learned `AdjacencyTable` never observed).
- Keeps the best (fewest illegal); stops at 0.
- Fallback: `_shuffle_fallback`.

Mapping to Moser-Tardos:
- **Bad event** `A_i`: adjacency `(order[i], order[i+1])` not observed in the table.
- **Dependency graph**: path graph, event `i` shares a chunk with `i-1` and `i+1`, so `d <= 2`.
- **Condition**: `p * e * 3 <= 1`, so `p <= ~0.123`. `p` is roughly the fraction of kind pairs absent from the table, so the table needs to be about 88% dense. Tables learned from small corpora across up to `MAX_TILES = 64` kinds will usually be far sparser. **The guarantee will mostly not apply.** Use it as a heuristic (local repair instead of global restart), not as a bound.

Two caveats that change the algorithm:
1. **Not independent variables.** The output must be a permutation of the input chunks (multiset preserved). Plain cell resampling breaks this. Use the **permutation variant**: on a violated adjacency, resample by swapping the offending chunk(s) with random other positions, then re-check the neighbouring events.
2. **`mode="violate"`.** `_add_one_unobserved_pair` plus pinning forces exactly one deliberately-illegal (w.r.t. the original table) adjacency. That event must be **excluded** from the resample set, and pinned cells (`pin_first`/`pin_last`, forced pair) must not be swapped.

## Proposed prototype (not started)

1. Keep one initial `_collapse_cells` + `_map_cells` (or a random permutation) as the starting sample.
2. Loop with a budget: find violated adjacencies under `work_table`; pick one (random or leftmost); swap-resample its chunks among unpinned positions; recompute affected adjacencies only (O(1) per swap, vs a full collapse today).
3. Stop at 0 violations or budget; fall back to the existing best-of-N result / `_shuffle_fallback`.
4. Gate behind a flag or parameter; default stays current behaviour until measured.

Start reading at `_collapse_cells`, `_map_cells`, `_place_leftovers`, `_legal_slot` (`_PLACEMENT_PROBES`), which I did not read in full. `_place_leftovers`/`_legal_slot` already do a bounded local search for a legal slot and overlap conceptually with the resampling step.

## Benchmark plan (the deciding measurement)

- Metrics: illegal-adjacency rate after reorder; wall time per call; distribution of table density per format.
- Formats: the 11 in `wfc_chunks.py` (riff, webp, isobmff, gif, ogg, flv, asf, mpegts, webm, nal, zip), 500 seeds each, matching the existing zero-illegal fixture claim.
- Sweep table density to find where the local lemma condition (`p <= ~0.123`) actually holds vs where local repair still wins empirically.
- Compare against the current restart loop with equal attempt budget.
- Follows the repo A/B posture: off by default, excluded from `--hail-mary` until measured.

## Marginal candidate: `covering_array.py`

Only worth touching if rows gain forbidden value combinations (e.g. PNG IHDR `color_type=3` with invalid `bit_depth`). Then a Moser-Tardos-style repair (resample only the offending parameters) beats rejecting whole candidate rows. No change needed for the current unconstrained case.

## Do not pursue

- Applying it to `field_constraints.repair` or `structural_constraints`: those are exact, deterministic, and cheaper than any resampling scheme.

## Open questions

- Actual per-format table density from real corpora (unmeasured).
- Whether swap-based repair beats best-of-8 restart on time at equal illegal-adjacency rate.
