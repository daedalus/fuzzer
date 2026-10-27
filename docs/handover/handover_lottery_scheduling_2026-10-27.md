# Handover — Lottery scheduling: what to port beyond `Stride`

Date: 2026-10-27
fuzzer HEAD: `b5789e61`
References: Waldspurger & Weihl, *Lottery Scheduling* (OSDI 1994); Waldspurger & Weihl, *Stride Scheduling* (MIT/LCS/TM-528, 1995)
Status: analysis only. Nothing implemented, nothing measured.

## 1. Summary

The repo already implements the deterministic half of the lottery family: `Stride` in `core/fair_queue.py`, wired as `seed_stride` (`core/schedulers/seed_stride.py`) and `op_stride` (`core/schedulers/op_stride.py`). Randomizing the draw itself adds almost nothing. What the 1994 paper has and the repo lacks is the machinery around the draw.

| # | Port | Target | Value | Cost | Needs benchmark |
|---|------|--------|-------|------|-----------------|
| 1 | Compensation tickets (cost-aware stride) | `Stride` in `core/fair_queue.py` | Medium-high | Low | Yes (wall-time metric) |
| 2 | Ticket currencies (hierarchical share) | new `CurrencyStride`, seed picker | Medium | Medium | Yes |
| 3 | Inverse lottery (randomized eviction) | corpus cap, `seeds/pruned/`, caches | Medium | Low | Yes |
| 4 | Partial-sum tree draw (O(log n) updates) | seed picker, if randomized draw is wanted | Low-medium | Medium | Yes |
| 5 | Ticket transfer | lineage credit | Low | Medium | Yes |

Order: 1 first (smallest, one file). 2 and 3 after the paired benchmark exists. 4 and 5 only if the earlier ones show a signal.

Explicitly not worth doing: a randomized lottery arm over operators (section 3.1).

## 2. What exists today

- `Stride` (`core/fair_queue.py`): each flow holds tickets (its weight) and a pass. A pick serves the lowest pass and advances it by `1 / tickets`. A heap makes a pick O(log n). A joining flow starts at the lowest live pass. `_rebuild` runs whenever the flow list differs from the last one (`flows != self._seen`), and costs O(n).
- Module table classes `Stride` as "counts", not "cost": it takes no cost signal. `DeficitRR`, `WeightedFairQueue` and `EEVDF` do.
- `SeedStrideScheduler`: tickets come from the caller per pick (`weight_fn`; `SeedPicker` gives AFL-favored seeds 2x). Flat weight equals `seed_round_robin`, which is the falsification condition.
- `OpStrideScheduler`: tickets are each operator's Beta(1, 1) posterior mean. Experimental, Elo-only, absent from `_FALLBACK_PRECEDENCE`. Its docstring already frames it as "a Thompson share without the sampling variance".
- Related, same session: `minimize --edge-components` (`2783bd47`, connected components of the seed Gram matrix) and `minimize --count-aware {bucket,thermo}` (`a0ad6b7b`). Both are inputs for port 2.
- No compensation, currency, transfer or inverse-lottery code exists (grep for `compensat|transfer|currenc|inflat` finds nothing relevant in `fair_queue.py` or `core/schedulers/`).

## 3. Findings

### 3.1 Randomizing the draw adds little

- Operators: a lottery with posterior-mean tickets is probability matching. It sits between `op_stride` (deterministic) and Thompson (`op_monte_carlo`). It adds no behaviour the two ends do not already cover.
- Seeds with flat tickets: uniform random, already the baseline.
- Trade-off from the paper: lottery allocation error grows like the square root of the number of draws (binomial variance); stride error is bounded by one quantum. Randomness buys statelessness, not accuracy.

Conclusion: do not add a lottery arm over operators. Any randomized draw is justified only by port 4's data-structure benefit.

### 3.2 Port 1 — Compensation tickets

Gap: `Stride.pick` advances the pass by a fixed `1/tickets` however long the pick took. A slow seed and a fast seed with equal tickets get equal picks, so the slow one takes far more wall time. On targets with a wide exec-cost spread (ffmpeg inputs, the 7 eps case) that is a pick-fair, not time-fair, share.

Paper: a client that uses only a fraction `f` of its quantum gets its tickets inflated by `1/f` (lottery), or its pass advanced by `f / tickets` (stride).

Design sketch:
- Keep `pick(flows, weight)` unchanged in behaviour (provisional advance by `1/tickets`).
- Add `settle(flow, used)`: adds `(used - 1) / tickets` to that flow's pass, where `used` is cost divided by a running nominal quantum (EMA of observed costs). Heap entries are tuples, so use lazy deletion: a `dict` from flow to the live entry, a version counter, stale entries skipped on pop.
- Cost passes through `neutral()`, so garbage cost is 1 (never free, never infinite), matching the module contract.
- Callers: the seed picker and the fuzz round report exec time (or execs consumed) for the pick.

Falsification: constant cost must reproduce the current `Stride` sequence exactly. Add that as a test in `tests/test_fair_queue.py`.

Check before implementing (not verified in this analysis): `SeedPicker`'s weight function may already include a speed or perf-score factor (AFL-style). If so, compensation double-counts speed and the ticket weight must be stripped of it first.

### 3.3 Port 2 — Ticket currencies

Idea: tickets are denominated per group. A group's backing is set independently of how many members it has, so adding or inflating members inside one group never dilutes the others.

Why it fits: the 2026-10-26 Gram analysis (`handover_coverage_gram_matrix_2026-10-26.md`, sections 8 and 9) found 56% (fuzzgoat) and 60% (png_read) of seeds removable by subsumption with union coverage preserved. In a flat ticket system those near-duplicates compete with distinct seeds for share. A currency per group lets them stay in the corpus without taking share.

Design sketch:
- `CurrencyStride`: level 1 is a `Stride` over groups with weight = group backing; level 2 is a `Stride` per group over members with member weights. A single group reduces exactly to flat `Stride` (falsification).
- Group key candidates: connected component from `--edge-components`; cell from `core/fractal_partition.py`; favored versus non-favored.
- Group backing candidates, to be compared: equal; proportional to `sqrt(size)`; proportional to the group's recent new-edge rate.
- The same structure extends upward (target, then seed group, then seed) and matches the existing `tgt_*` and `seed_*` axes.

Open, not verified: whether the component computation can run online or only inside `minimize`. If offline only, groups would be refreshed on a cadence rather than per pick.

### 3.4 Port 3 — Inverse lottery

Paper: for space-shared resources, pick the victim with probability proportional to `1 / tickets`, so low-value holders lose the resource more often but never deterministically.

Candidate uses:
- Evicting seeds when the corpus hits a memory cap (the seed-slide memory cap site).
- Choosing which seeds go to `seeds/pruned/`.
- Cache eviction where an LRU is currently used.

Why it helps: set-cover minimization is deterministic and keeps one covering set. A randomized cull keeps the survivors diverse and cannot systematically evict one lineage.

Design sketch: a small pure function `inverse_lottery(keys, tickets, rng, floor)` returning a victim. Tickets are floored so a zero-ticket key cannot get infinite probability, and capped so one extreme value does not dominate. Eviction call sites still have to be located in the code.

### 3.5 Port 4 — Partial-sum tree draw

Problem: `Stride._rebuild` is O(n) on every membership change. The corpus changes constantly, so the pass state is repeatedly rebuilt.

Idea: a Fenwick (or segment) tree over weights. `set(key, w)` is O(log n); `sample(u)` descends the tree in O(log n); removal sets the weight to zero and reuses the slot; compact when slots exceed twice the live count.

Constraints:
- The draw must use the fuzzer's seeded RNG so runs and `--resume` stay reproducible (verify which RNG the picker uses before wiring).
- It trades stride's bounded error for binomial variance, so it applies to seeds, not to the small operator set.

### 3.6 Port 5 — Ticket transfer

Idea: a seed whose children keep finding coverage lends tickets back to its parent. `--lineage` already covers most of this credit assignment, so value is low. Only worth revisiting if lineage shows a measurable gap.

### 3.7 Alternate reading: lottery ticket hypothesis

If "the lottery" meant the lottery ticket hypothesis (Frankle & Carbin), the analogue is iterative pruning of the operator set with rewinding to early statistics. `successive_elim` and `topk` already approximate pruning without rewinding. Not developed here.

## 4. Benchmark and falsification plan

- Prerequisite from earlier handovers: no scheduler arm has an A/B measurement yet. Arms need to be added to `tools/bench_paired.py`; `--pos-arena-arms` only subsets the position arena.
- Port 1 must be measured on edges at fixed wall time, not only at fixed execs; at fixed execs it cannot show an effect.
- Targets: fuzzgoat, png_read (both already built for the Gram work), and ffmpeg (widest exec-cost spread).
- Each port needs its degenerate case as a unit test: constant cost (port 1), single group (port 2), equal tickets (port 3), static membership (port 4).
- Report paired differences with the existing multiple-testing utilities (`multiple_testing.py`), not single-run edge counts.

## 5. Open decisions

1. Start with port 1? It touches `core/fair_queue.py` and `tests/test_fair_queue.py` only.
2. Port 2 group key: `--edge-components` versus `fractal_partition` cells, and which backing rule.
3. Port 3: which eviction sites to cover first.
4. Whether to build the paired-benchmark arm subset flag before any of these.
