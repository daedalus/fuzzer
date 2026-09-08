# Handover: statistical analysis gaps — COMPLETED

**Date:** 2026-09-07
**Base:** `b80c6d7`
**Status:** IMPLEMENTED — all five areas shipped in commit `47cb5b1`
**Replaces:** `docs/handover/handover_statistical_gaps_2026-09-07.md` (deleted; full text in git)

---

## 0. One-line summary

All five queued improvements are implemented and tested. Wilcoxon, MCAR/MNAR,
and Spearman are wired into live analysis paths; KS is surfaced but measured
redundant with Wasserstein and therefore not wired into the diversity weight;
KL-UCB is promoted to full peer schedulers (KL_DUCB/KL_SWUCB), with the
`kl_ucb` flag removed from DUCB/SWUCB (Gaussian width is now the only path
on those two).

---

## 1. Wilcoxon signed-rank — `tools/bench_paired.py`, `compare()`

`_wilcoxon_signed_rank(deltas)` added next to `_mcnemar_exact` at
`tools/bench_paired.py:266`. Exact permutation distribution for n <= 20
(all 2^n sign assignments, two-sided), normal approximation with continuity
correction above, zero deltas dropped before ranking, average ranks for ties.
`compare()` returns `wilcoxon_stat` / `wilcoxon_p`; `cmd_analyse` prints
`Wilcox` per target (per-target only, never pooled). Tests in
`tests/test_bench_paired_stats.py::TestWilcoxonSignedRank`.

## 2. MCAR / MNAR diagnostic — `tools/bench_paired.py`, `compare()`

`dropped` counter replaced by `dropped_base` / `dropped_test`; pooled total kept
as `dropped_no_coverage` for backward compatibility. `cmd_analyse` prints the
unequal-drop warning when the two arms lose different numbers of cells.
Test `TestPairing.test_drop_counts_are_per_arm`.

## 3. Spearman on (regime, Re) — `coverage_regime.py`, `continuum_correlation()`

Returns `{"rho", "p_value", "bucket_means"}`. `_spearman_rank_correlation`
hand-rolled (average-rank ties, Student-t p-value via Lentz continued-fraction
incomplete beta). `_REGIME_RANK` maps the plain enum to SUBCRITICAL < CRITICAL <
SUPERCRITICAL. Tests in `tests/test_navier_stokes_wiring.py`.

## 4. KS surfaced — `edge_tracker.py`

`ks_vs_aggregate(hc)` and shared `_aggregate_norms(hc)` added; `_cdf_walk` already
computed KS on every call. Measurement script `tools/measure_ks_signal.py`:
KS correlates 0.9993 with Wasserstein on the synthetic corpus, so the
diversity-weight wiring stays gated. Test `test_ks_distance_separates_loop_heavy_seeds`
in `tests/test_edge_ground_metric.py`.

## 5. KL-UCB — `core/schedulers/_kl_ucb.py`, `ducb.py`, `swucb.py`

Shared Bernoulli upper bound (`kl_upper_bound`). Promoted to two full peer
schedulers (`KL_DUCBScheduler`, `KL_SWUCBScheduler`); the `kl_ucb` flag was
removed from `DUCBScheduler` and `SWUCBScheduler`, whose `_width()` is now the
Gaussian form only. `tools/measure_klucb_signal.py` runs all four side by side:
SW-UCB KL 0.997 vs Gaussian 0.985, DUCB KL 0.777 vs Gaussian 0.968 — not a
one-size win, so the KL variants stay opt-in via `--kl-ducb` / `--kl-swucb`.
FPL (`core/schedulers/fpl.py`) has since been implemented (commit `ed09d59`).

---

## 6. What is still open

- FPL scheduler (`core/schedulers/fpl.py`) — implemented in commit `ed09d59`,
  wired into the service layer (`fuzzer.py`) and CLI (`--fpl`, `--fpl-epsilon`).
  Convergence verified on StationaryBernoulli (tail share 0.999, regret
  slope 0.012).
- KS wiring into `compute_hitcount_diversity_weight` — gated until a real
  corpus shows KS separating a signal Wasserstein misses.
- `kl_ucb=True` per-target enablement — gated until the target's reward
  distribution is measured.
