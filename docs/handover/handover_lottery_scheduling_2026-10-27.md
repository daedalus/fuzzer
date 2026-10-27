# Handover — Lottery scheduling: what to port beyond `Stride`

Date: 2026-10-27
fuzzer HEAD: `b5789e61`
References: Waldspurger & Weihl, *Lottery Scheduling* (OSDI 1994); Waldspurger & Weihl, *Stride Scheduling* (MIT/LCS/TM-528, 1995)
Status: all five ports implemented behind off-by-default flags and unit-tested; none benchmarked. Ports 1 and 3 are in sections 6; ports 2, 4 and 5 are one seed arm, section 7.

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

Resolved at implementation (section 6): `SeedPicker`'s weight is favored-only (`FAVORED_WEIGHT` or 1.0), so there is no speed factor to double-count, and a per-key `cost` signal (ledger exec time over the corpus mean) already exists. The `settle` design above was dropped: cost is charged at pick time from that prediction, which needs no lazy-deletion heap.

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

Design sketch: a small pure function returning victims. Tickets are floored so a zero-ticket key cannot get infinite probability. No cap is needed (see section 6: the key is linear in the ticket). The eviction site is `CorpusManager._select_budget`, count-budget path.

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

## 6. Implementation status (2026-10-27, on top of `ba6cf7eb`)

Implemented, off by default, unit-tested, not benchmarked:

| Port | Flag | Where | Notes |
|------|------|-------|-------|
| 1 Compensation tickets | `--seed-stride-cost` (with `--seed-stride-scheduler`) | `Stride.pick(flows, weight, cost=None)` in `core/fair_queue.py`; `SeedStrideScheduler(cost_aware=...)`; `"stride": ("weight", "cost")` in `_OS_SEED_SIGNALS` | stride = `neutral(cost) / neutral(weight)`. Flat cost reproduces plain stride exactly (tested). Garbage cost counts as 1. Cost is the cost-ledger prediction, charged at pick time, so the `settle` step of section 3.2 was not needed. |
| 3 Inverse lottery | `--inverse-lottery-select` | new `core/inverse_lottery.py`; `CorpusManager._inverse_lottery_select_optional`, called from `_select_budget` after the mds/minimax branches | k victims without replacement as the k smallest `Exp(1) * score` (Efraimidis-Spirakis), O(n log n), one pass. Equal scores give a uniform cull (tested). Count-budget path only; `--max-corpus-bytes` knapsack and `--mds-select` / `--minimax-select` take precedence. |

Also added: bench arms `seed-stride` (vs `baseline`), `seed-stride-cost` (vs `seed-stride`), `inverse-lottery-select` (vs `minimize-2k`) in `tools/lib/bench_paired.py`; a row in `docs/AB_PENDING.md`; a CHANGELOG entry. Neither flag is in `_HAIL_MARY_FLAGS` (a test pins that until there is a measurement).

Tests: `tests/test_fair_queue.py` (Stride cost), `tests/test_inverse_lottery.py`, `tests/test_lottery_flags.py`, and two cases in `tests/test_os_net_scheduler_wiring.py` (cost flag on and off).

Deviations from the sketch in sections 3.2 and 3.4: no `settle`; no ticket cap.

Pitfall found while writing the corpus test: when several seeds hold identical edge sets, `_greedy_cover` breaks the tie by `id(seed)`, so which seed is mandatory differs from run to run even with a fixed RNG. A test of any selection branch needs one seed that alone holds an edge, or its control is not deterministic.

Not implemented: ports 2 (currencies), 4 (partial-sum tree), 5 (transfer). Open decisions 2-4 of section 5 still stand; decision 1 (start with port 1) is done.

Next, in order: run `seed-stride-cost` vs `seed-stride` on ffmpeg at fixed wall time; run `inverse-lottery-select` vs `minimize-2k` on fuzzgoat and png_read; then decide on port 2.

## 7. Ports 2, 4 and 5: the `lottery` seed arm (2026-10-27, on top of `59381eac`)

The three remaining ports are one arm, because in the 1994 paper they are one mechanism: a lottery over tickets (port 4), tickets denominated in currencies (port 2), and tickets lent between related clients (port 5).

| Port | Flag | Where |
|------|------|-------|
| 4 Partial-sum tree | `--seed-lottery-scheduler` | `core/fenwick_sampler.py` (`FenwickSampler`: O(log n) set/remove/draw, slot reuse, compaction, drift-safe descent); `core/schedulers/seed_lottery.py` (`SeedLotteryScheduler`, Elo arm `lottery`) |
| 2 Currencies | `--seed-lottery-currency {off,equal,sqrt}` | currency = connected component of the seed Gram matrix (`core/edge_components.components`), computed online in `SeedPicker._component_labels` |
| 5 Transfer | `--seed-lottery-transfer F` | `SeedLotteryScheduler.record`; parent from the existing `flow` signal (lineage `parent_key`), so it needs `--lineage` |

Behaviour:
- Tickets = weight / cost (compensation always on in this arm). Flat weight and cost is a uniform random pick (tested).
- Between refreshes a seed keeps the tickets it was admitted with; every 256 picks all seeds are re-read (weight, cost, group, parent) and transfer credit halves. A joiner is drawable on the next pick without waiting for a refresh.
- Currencies: a draw picks a component by backing (`equal` = 1, `sqrt` = sqrt(size)), then a seed inside it. One group, or `off`, is the flat pool (tested). `off` never calls the group signal, so it costs nothing.
- Components are recomputed when the corpus size drifts more than 10% or after 5000 execs; a seed admitted since gets its own `FRESH_GROUP` currency until then.
- Transfer: a successful seed adds `F x reward` to its lineage parent's credit; tickets = base x (1 + credit), credit capped at 4, halves every refresh, dropped when the parent leaves the corpus.

Bench arms: `seed-lottery` (vs `seed-stride-cost`), `seed-lottery-currency` (vs `seed-lottery`), `seed-lottery-lineage` (vs `seed-lottery`), `seed-lottery-transfer` (vs `seed-lottery-lineage`, because `--lineage` alone is a change and must not be credited to the transfer).

### Measured (one sandbox microbenchmark, not a campaign)

Scheduling overhead per pick, `Stride` (rebuilds on a membership change) against the lottery arm, one run each:

| corpus | no churn | one new seed per 10 picks | one new seed per pick |
|-------:|---------|---------------------------|-----------------------|
| 500 | 1.0 / 5.6 us | 15 / 12 us | 593 / 161 us |
| 2,500 | 3.0 / 26.5 us | 93 / 48 us | 1,382 / 355 us |
| 10,000 | 9.2 / 97.8 us | 446 / 193 us | 5,310 / 1,115 us |

(stride / lottery.) The tree wins 2-5x under churn and loses 5-10x on a static corpus, because the arm re-reads every seed once per 256 picks (about 25 ms at 10,000 seeds). In absolute terms both are small next to a pick's cost: one pick is followed by a whole mutation round of target executions. So the Fenwick tree is not a throughput argument; whether the lottery is a better scheduler is a coverage question only the paired runs can answer. This lowers the value rating of port 4 given in section 1.

### Open risks, not resolved

1. Giant component: `edge_components` strips edges held by more than 10% of seeds, but a real corpus may still be one component holding most seeds, which makes `--seed-lottery-currency` a no-op (one currency). Not checked on a real corpus here; the Gram handover's `minimize -c --edge-components` report prints the component sizes. Look at that before running the currency arm.
2. Transfer credit memory is short by design (halves every 256 picks); the cap (4) and the half-life are not tuned.
3. The lottery arm is compensated unconditionally, so `seed-lottery` vs `seed-stride-cost` isolates the random draw, not compensation. `seed-lottery` vs plain `seed-stride` would mix the two.
4. Currency and transfer arms are untested on a real target; the unit tests check the mechanics (shares, caps, decay, drops), not that they find more coverage.

Not done: nothing from the five ports remains to implement. Remaining work is measurement (section 4 plan plus the arms above).
