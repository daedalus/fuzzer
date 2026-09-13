# P3-3 job scheduling, steps 4–5: what `--job-scheduler` actually buys

2026-09-12. Companion to `handover_pending_2026-09-06.md` §P3-3. Steps 1–5
of the six-commit sequence are shipped, gated behind `--job-scheduler`
(default off, excluded from `--hail-mary` — see that flag's justification
comment in `cli/commands.py` next to `_HAIL_MARY_FLAGS`). This doc records
the empirical answer to the obvious next question: does turning it on
help, and by how much?

Both simulations below use the actual shipped code
(`services/maintenance.py`, `core/parallel_cost_partition.py`,
`core/parallel_fractal_partition.py`), not a reimplementation of their
logic, so the numbers are reproducible against the current source rather
than a paraphrase of it.

## 1. MaintenanceQueue consolidation (step 4): call-count, not throughput

The three ad-hoc gates it absorbs (`_run_crash_replays`/
`_run_sanitizer_replays` on `i % 500`, `_check_memory_and_prune`'s internal
1000-exec throttle, `gc.collect` on `i % 500`) had two different shapes.
Memory pruning was already checked once per stats tick (gated internally);
crash/sanitizer replays and `gc.collect` were checked on raw loop-iteration
count, decoupled from `_stats_effective_interval()`'s ~10×-mean-EPS
spacing. Consolidating into one queue, ticked inside the stats-tick block,
changes the cadence of the second group only — and the direction of that
change depends on target speed, not just its magnitude.

Simulated over 10M execs, `mutations_per_input=8` (default):

| EPS | legacy `gc` calls | `--job-scheduler` `gc` calls | ratio |
|---|---|---|---|
| 10 | 2,500 | 20,000 | 8x **more** |
| 200 | 2,500 | 5,000 | 2x more |
| 400 | 2,500 | 2,500 | crossover |
| 800 | 2,500 | 1,250 | 2x fewer |
| 10,000 | 2,500 | 100 | 25x fewer |
| 100,000 | 2,500 | 10 | 250x fewer |

Memory-prune call counts were identical in both modes at every EPS tested
(as expected: its cadence didn't change, only where the throttle lives).

The crossover is exact, not approximate: `eps = 50 × mutations_per_input`
(400 at the default). Below it, `--job-scheduler` fires `gc`/replay checks
*more* often than the legacy path, because tying them to a ~10-second
stats cadence is a tighter bound than the old fixed 4,000-exec
(`500 × mutations_per_input`) iteration count for a slow target. Above it,
the reduction is substantial and grows with speed — 250x fewer calls at
100k execs/sec.

**Reading this correctly:** this is a call-overhead measurement, not a
fuzzing-throughput measurement. No campaign was run; nothing here claims
more edges found or higher effective EPS. What it establishes is that
`--job-scheduler`'s consolidation is a net reduction in wasted maintenance
calls only above the crossover EPS, and a net increase below it — the
opposite of what a flag named "job scheduler" might suggest by default.
This is consistent with why it's gated rather than replacing the legacy
path: the right choice depends on the target's speed, which the flag
doesn't (and shouldn't try to) infer automatically.

## 2. Cost-based partitioning (step 5): load balance at campaign start

Simulated initial-corpus assignment for 2,000 seeds across hash-mod,
fractal Voronoi (`core/parallel_fractal_partition.py`), and Multifit-packed
cost (`core/parallel_cost_partition.py`, byte size as the cost proxy — see
the caveat below), on two size distributions: skewed (90% seeds at
10–200B, 10% at 2–50KB — representative of format/crash-min corpora) and
uniform (10–5000B).

| corpus | workers | hash-mod | fractal Voronoi | Multifit (cost) |
|---|---|---|---|---|
| skewed | 4 | 1.14x ideal | 2.28x ideal | 1.00x |
| skewed | 8 | 1.33x | 2.58x | 1.00x |
| skewed | 16 | 1.86x | **5.83x** | 1.00x |
| uniform | 16 | 1.31x | **5.79x** | 1.00x |

"Ideal" is total corpus bytes ÷ worker count; the ratio is the most-loaded
worker's share over that ideal, i.e. how much longer the slowest worker's
initial-seed-replay phase takes relative to a perfectly balanced split.

Fractal Voronoi partitioning is not a regression here — it was never
designed to balance load, only to give a seed a content-stable worker
assignment independent of discovery order. It has zero awareness of seed
cost, so its imbalance is a property of whatever the hash happens to do,
and gets worse as worker count grows (more partitions for the hash to
misallocate across). Multifit is the right tool for this specific
objective (minimize makespan across identical machines) and achieves
near-perfect balance in every configuration tested, which is expected
rather than surprising — bin-packing to balance load is exactly the
problem it solves.

**What this does and does not extend to:**

- Scope is the one-shot initial distribution
  (`_distribute_initial_corpus`) only. Corpus growth during the campaign
  still syncs via fractal Voronoi (if `--job-scheduler`'s
  `fractal_partition` companion is also set) or full sharing (if not) —
  see `_sync_corpus_in`'s `accept_for_worker` gate, which this work did
  not touch. The load-balance gain is confined to the campaign's warm-up
  window, not sustained across its lifetime.
- The cost proxy is **byte size**, not measured target execution time,
  because no execution history exists before any seed has run. This is
  stated in `parallel_cost_partition.py`'s module docstring as a real gap,
  not glossed over. The simulated gain is contingent on size correlating
  with actual per-seed cost for the target in question — plausible for
  many targets (larger inputs often mean more parsing/processing) but not
  guaranteed, and actively wrong for targets whose cost is dominated by
  something else (e.g. early-exit parsers, or cost driven by input
  *structure* rather than *length*).
- No live recompute is wired in. `maybe_repartition`'s hysteresis (max
  per-id drift over a threshold, default 25%, triggers a repack;
  otherwise returns the previous assignment unchanged) exists and is
  tested, but nothing calls it against `core/cost_ledger.py`'s per-tick
  measured costs during a run. That remains open — see
  `handover_pending_2026-09-06.md`'s P3-3 status note.

## Net assessment

Both pieces are real, measured improvements over specific baselines, in
specific conditions, not unconditional wins:

- Housekeeping consolidation reduces call overhead above ~400 execs/sec at
  default settings (250x at 100k eps) and increases it below that (up to
  8x at 10 eps). Absolute overhead at low EPS is small regardless, so the
  practical harm of the "wrong side" of the crossover is limited, but it
  is a real, sign-flipping effect, not just a magnitude difference.
- Cost partitioning removes a real straggler effect at campaign start
  (up to 5.8x worse than ideal for fractal partitioning at 16 workers),
  bounded to the warm-up phase and to how well byte size proxies actual
  cost for the target.

Recommendation, unchanged from why the flag is gated rather than
defaulted-on: treat `--job-scheduler` as something to A/B per target
(`tools/cost_dispersion.py`'s per-target noise-floor measurement is the
right instrument, per the Boltzmann precedent already established for
this codebase) rather than a strict upgrade to enable everywhere.

## Reproducing these numbers

Both simulations are ~30-line scripts against the shipped modules, not
committed as tests (they measure a *scenario*, not a *contract* — nothing
here is a regression to catch). To reproduce:

- Housekeeping cadence: drive
  `services.maintenance.MaintenanceQueue.tick()` with the same
  `_stats_effective_interval()` formula (`max(1, int(10 * eps))`) at a
  fixed EPS and count job firings against the legacy `i % 500` /
  `iters = execs // mutations_per_input` arithmetic.
- Load balance: generate a synthetic corpus (`{seed_id: size_bytes}`),
  partition it with `core.parallel_fractal_partition.assign_worker`, a
  plain `sha256(id) % m` hash, and
  `core.parallel_cost_partition.compute_partition`, and compare
  `max(per_worker_load) / (total / m)`.
