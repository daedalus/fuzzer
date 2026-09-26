# Handover — Index Policies, Optimal Stopping and Search Allocation

Original 2026-09-02 (base `dc30854`). Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_bandit_stopping_search_2026-09-02.md`
(measurements, Gittins/Koopman prototypes, rejected alternatives in §10).
Verified against `4c021daa`. Nothing below is implemented. Tracked as P3-2 / P3-4
in `handover_pending_2026-09-06.md`.

---

## 1. `core/secretary.py` is not a stopping rule — and it is live

Defect unchanged since the original measurement:

- `_rank_of_best()` is bounded by `1/(1-decay) = 20` (`decay = 0.95`);
  `threshold = int(n/e)` reaches 183 at `window_size = 500`. For `n ≥ 58` the
  rank clause always passes → rule is "no all-time record in last `n/e` obs".
  `_record_count`, `_best_idx` are written/persisted, never read by a decision.
- `_best_value` is all-time, never recomputed on window slide.
- Seed stream (`services/fuzzer.py`, `discovery_rate = len(new) / fuzz_count`,
  cumulative `fuzz_count`) carries a `1/t` envelope: a seed finding a new edge on
  50% of execs is stopped at obs 20 (`min_observations`) and never recovers.

Consumers (the class docstring's "display only / nothing calls `should_stop`"
claim, added in `0fd5f25d` for P2-4, is **false**):

- `services/seed_picker.py::_weight_secretary_and_cached` — `w *= 0.01` on stop.
  Net effect of `--secretary`: 100× preference for seeds with < 20 observations.
- `services/corpus_manager.py` — `_corpus_secretary.should_stop()` →
  `f._defer_minimize()`; `discovery_rate()` decays globally, so it stops permanently.
- `_op_secretary`: per-op per-iteration `observe()`, read only as a count in
  `services/stats.py`. Pure overhead.

Action: remove `--secretary` (module, three instances, `w *= 0.01`, defer trigger,
CLI flags `--secretary*`, stats/report lines), or replace with §2's retirement
value. Record the falsified hypothesis in `docs/learnings/`. Regression tests
(exact, Rule 39): strictly increasing 500-point stream → `rank == 20.0`,
`threshold == 183`; `len(new)/fuzz_count` stream at p=0.5 → `should_stop()` true
at obs 20. At minimum, fix the false docstring.

## 2. Gittins index — `core/gittins.py` (absent)

```
build_table(n_max=300, gamma=0.99, grid=2048) -> np.ndarray
index(alpha, beta, *, gamma) -> float            # bilinear interp; mu fallback outside table
retirement_value(alpha, beta, *, gamma) -> float
```

Backward induction over `(α, β)` with `λ` as vector axis:
`W(a,b) = max(λ/(1-γ), μ + γ[μ·W(a+1,b) + (1-μ)·W(a,b+1)])`. Prototype matched
Gittins–Jones `ν(1,1) = 0.7029` (γ=0.9), 0.8694 (γ=0.99); N=300 builds in ~1 s,
350 KiB. Monotonicity/`ν ≥ μ` fail only past `α+β > N` → fall back to `μ`.

Consumers:

1. **Seed arm `gittins`**: `BayesianSeedQuality.gittins_select()` beside
   `select_seed()`; ballot + `_SEED_STRATEGY_NAMES` + persistence. A/B vs
   `bayesian` (`seed_picker.py::_pick_bayesian_seed`) — same posteriors, policy-only
   difference.
2. **Retirement value replacing §1**: stop when index < best alternative. Monotone
   in evidence, comparative, reversible, no tuning knobs.

Operator ballot: optional selectable arm only (97.5% argmax agreement with
posterior mean at hot, 155 arms). Docstring must state the theorem does not
strictly apply (restless arms, policy-dependent arrivals, count not Bernoulli).

Gate before A/B (§9 of original): dump `BayesianSeedQuality` `(α, β)` after a
100k-exec campaign per target. Median `α+β > ~200` → hot → Gittins ≈ greedy; stop.

Open questions:

1. Cold vs hot seed posteriors in a real campaign (the gate above).
2. `MonteCarloScheduler.record()` adds `weight` on success, `1` on failure →
   weighted pseudo-counts, not a Beta posterior. Quantify distortion on table lookup.
3. Discount `γ`: `BayesianSeedQuality` default `decay = 1.0` is undiscounted
   (Gittins undefined). Pick a decay or derive `γ` from campaign length; document why.
4. `EdgeTracker._maybe_prune` eviction ≠ low-index retirement; do not conflate
   when measuring the stopping rule.

## 3. Cost-ledger second moment

Add `total_time_sq` to `core/cost_ledger.py`, written in lockstep with
`total_time` / `cost_samples`, persisted. Gives per-seed Var/CV (within-seed,
unlike `tools/cost_dispersion.py`'s across-seed). Prerequisite for any stochastic
scheduling index (preemptive `1|r_j,pmtn|E[ΣC_j]`). Instrumentation only.

## 4. Explore-then-commit null arm (absent)

`core/schedulers/op_etc.py`: `m` uniform rounds per op, then permanent commit.
`supports_priors = False` (uniform exploration cannot use a prior; Rule 40).
Plus a `--baseline-etc` arm in `tools/lib/bench_replicated.py`. Purpose: a floor —
how much coverage is due to scheduling vs operator quality. No default change.

## 5. Koopman allocation — `--schedule koopman` (absent)

`t_i = max(0, (1/λ_i) ln(p_i λ_i / θ))`, `θ` by bisection so `Σt_i = T`.
`λ_i` = new edges per **second** (`meta["coverage_edges"] / meta["total_time"]`),
`T` = one maintenance interval. Add to `core/schedules.py::SeedScorer.SCHEDULES`.

- Missing input `p_i`: per-seed Chao2 from `seed_edges` + `_edge_owner_count`
  (`core/edge_tracker.py`); corpus-level `good_turing_estimate()` is display-only
  apart from `SeedPicker._saturation_gate`.
- Gate first: histogram `λ_i·T` on one real campaign per target. Bulk > ~5 →
  uniform (no-op); bulk < ~0.01 → greedy (no-op). Proceed only in the middle.

## 6. D-UCB cold-regime check

At coef 2.0, D-UCB agreed with Gittins 0.0% cold at 155 arms (picks least-sampled
arm). Live default is `exploration = 0.25` (`core/schedulers/op_ducb.py`,
`UCB_WIDTH_COEFF = 2.0`), tuned on 12 arms. Re-measure cold-start behaviour at
155 arms with the live default; if still degenerate, tune — do not replace.

## A/B design (applies to §2, §5)

`tools/lib/bench_replicated.py`, `--lock-single-thread`, time-paired arms. Noise
floor sd ≈ 4.6 edges png, 4.7 jpeg, 12.3 grep; replicates, not seeds, buy power.
