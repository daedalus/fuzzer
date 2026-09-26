# Handover — Classical Job-Scheduling Algorithms in the Fuzzer

Original 2026-09-02 (base `3480de6`). Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_job_scheduling_2026-09-02.md`.
Verified against `4c021daa`. Shipped: `core/job_scheduling.py` primitives,
deterministic-stage quota + rotation, `MaintenanceQueue` (`--job-scheduler`),
`last_picked` + `--lst-revisit`. Measurements of shipped parts:
`handover_job_scheduling_p3_3_2026-09-12.md`.

---

## 1. Maintenance tick — queue covers 3 of ~14 jobs

`services/maintenance.py::MaintenanceQueue` absorbs only the three ad-hoc gates
(crash/sanitizer replays, memory prune, `gc.collect`) and orders due jobs with
`lawler_order`. Not done from the original proposal:

- Rest of the stats-tick block in `services/fuzzer.py` (`_cull_queue`, entropy,
  χ² homogeneity, snapshots, regime, stall detection, corpus-size prune,
  `_dump_stats`/`_save_state`) still runs in fixed line order.
- Real precedences not expressed as `predecessors`: `_cull_queue` → power schedule
  (`self._favored`); prune → `_save_state`; `_regime.observe` → strategy
  adjustment → stall detection.
- No learned `p_j` (EWMA wall time), no EDF/MDD dispatch, no per-tick budget.

Gate first (falsifier): log per-job wall time in the existing block for one real
campaign. Synthetic: `_cull_queue` 15.5/177/558 ms at 200/1000/2000 seeds vs
~1 ms for the rest; the χ² block (`get_edge_counts()` per tick) was not measured.
Flat spread → do not extend the queue. Accept: lower p95 per-job lateness, no
rise in total tick wall time.

## 2. A/B runs owed

- `--lst-revisit` — tracked in `docs/TODO.md` (sweep `D` vs corpus size × mean
  round cost; below that product it degenerates to oldest-first round robin).
- `--job-scheduler` — call-overhead sign flips at `eps = 50 × mutations_per_input`
  (p3_3 doc §1); no campaign run. A/B per target.

Both: `tools/cost_dispersion.py` per target first; `tools/lib/bench_replicated.py`
`--lock-single-thread`; noise floor sd ≈ 4.6 (png) / 4.7 (jpeg) / 12.3 (grep);
pre-register resolvable effect size.

## 3. Deterministic stage

- **Truncation counter not surfaced.** `services/operators.py` sets
  `_deterministic_mutation_stream.last_truncated`; nothing reads it (no stats
  line). Wire into `services/stats.py`.
- **Falsifier unrun:** quota split vs bitflip-only on seeds > 8192 B (vendored
  ffmpeg / sqlite corpus). Fewer edges with quotas → bitflip deserves the budget.
- **Open:** static vs adaptive quota (`_split_det_quota` is cost-proportional; an
  adaptive one is a bandit → `core/schedulers/`).
- **Open:** `SkipDetector.should_det_fuzz` / `seed_passed_det` interaction. A
  seed marked det-passed after one truncated run is not re-consulted, so the
  `fuzz_count` start-offset rotation may never tile a long seed. Verify.

## 4. Seed calibration order (SPT)

`services/fuzzer.py::_calibrate_seed_baselines` runs `list(self.corpus)` order,
no budget. Objective is mean flow time (`1||ΣC_j`) → SPT (shortest expected time
first, `input_size` proxy before any timing exists). Small, independent change.
