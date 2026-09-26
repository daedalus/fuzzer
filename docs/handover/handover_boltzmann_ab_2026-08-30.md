# Handover — the Boltzmann energy A/B

**Original:** 2026-08-30. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_boltzmann_ab_2026-08-30.md`.
**Verified against:** `4c021daa`. The A/B itself ran and closed
(`docs/learnings/2026-08-30-boltzmann-ab-result.md`); one follow-on remains.

---

## 1. Should cumulative execution cost enter the eviction ordering?

**What.** `EdgeTracker._maybe_prune` (`core/edge_tracker.py`) evicts
cheapest-first by unique-coverage loss (`_unique_loss`), ties broken by
insertion order. No cost term.

**Why it matters.** Instrumented on png_read/gzip_read, 99.9% of eviction
candidates have loss 0, so the insertion-order tiebreak decides nearly every
prune (`docs/learnings/2026-08-30-prune-ceiling-and-eviction.md`). Accumulated
cost (`core/cost_ledger.py::effective_fuzz_count`, `seed_exec_time`) is the
signal that varies most across those ties. Also tracked in `docs/TODO.md`.

**Open question.** Replace or precede the insertion-order tiebreak with cost per
retained edge? Down-weighting expensive seeds may drop deep paths.

**How to measure.** Changes which seeds exist, so it needs a paired A/B. Use
`tools/lib/bench_paired.py` on the `direct_lite` / `direct_lite_signal` sets
(`tools/lib/eval_set.py`), never `locked`: process-spawn cost flattens per-seed
cost dispersion (png_read subprocess p90/p10 1.06x vs direct_lite 4.32x), so an
arm that depends on cost variance reads as a guaranteed null there. Reuse the
Boltzmann protocol: paired in time, `-m 65536`, per-target breakdowns, skip
saturated zlib/lz4/gzip cells.

**Acceptance.** Per-target W/L and median Δ edges on png/jpeg/grep with a power
statement; adopt only on a non-negative result.
