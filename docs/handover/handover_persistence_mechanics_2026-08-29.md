# Handover: Persistence Mechanics — what ports, what does not

**Original:** 2026-08-29 (evaluation of K. L. Meyer, *Persistence Mechanics*).
Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_persistence_mechanics_2026-08-29.md`.
**Verified against:** `4c021daa`. Rejected ports (§2 of the original) live in
`docs/port-backlog.md` "Rejected".

---

## 1c. Cumulative cost in `_maybe_prune` ordering

**What.** `EdgeTracker._maybe_prune` (`core/edge_tracker.py`) orders evictions
by unique-coverage loss, ties by insertion order. 99.9% of candidates tie at
loss 0, so the tiebreak is the eviction policy
(`docs/learnings/2026-08-30-prune-ceiling-and-eviction.md`).

**Open question.** Should cumulative cost per retained edge
(`core/cost_ledger.py::effective_fuzz_count`) break those ties? Write any
consumer against `effective_fuzz_count`, not raw `total_time`: it reduces to
the count form on flat-cost targets by construction.

**Acceptance.** Paired A/B on the `direct_lite` sets, per-target breakdown.
Same item as `handover_boltzmann_ab_2026-08-30.md` §1 and `docs/TODO.md`.

## 1a follow-up. `STALE_SEED_EXEC_EQUIVALENTS` is uncalibrated

`services/stats.py::STALE_SEED_EXEC_EQUIVALENTS = 50`, used in
`StatsReporter._print_summary_seeds`, was carried over from the count-based
threshold and never calibrated. Open: pick a value from measured
exec-equivalents-to-first-edge distributions, or document it as arbitrary.
