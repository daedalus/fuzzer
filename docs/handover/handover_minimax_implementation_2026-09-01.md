# Implementation Note — Minimax Estimator and Algorithm for Fuzzer Enhancement

Original 2026-09-01. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_minimax_implementation_2026-09-01.md`
(research companion: `git show 1c689e8a^:docs/handover/handover_minimax_alphabeta_adversarial_search_2026-09-01.md`).
Verified against `4c021daa`.

---

## 1. E3 — A/B the `alphabeta` seed arm (unrun)

The arm is no longer minimax: `26b367ab` replaced alpha-beta descent (1 distinct
seed / 300 picks, root-only) with Thompson descent over the lineage tree. Class
name, `--alphabeta` flag and state key kept
(`core/schedulers/seed_mcts.py::AlphaBetaMCTSSeedScheduler`,
`services/seed_picker.py::_pick_alphabeta_seed`).

- Harness exists: `tools/lib/bench_paired.py` arms `elo-alphabeta` vs `elo-mcts`
  (`analyse --baseline elo-mcts`).
- Targets: `png_read`, `ffmpeg_read`. Replicated, `--lock-single-thread`.
- Accept: effect resolved above noise floor (sd ≈ 4.6 edges png) or a bounded null.

## 2. Phases 2–5 — shipped, unreachable, untested

Code exists; nothing in the fuzz loop calls it; no falsification/adversarial tests
(Hard Rule 23). Decide per item: wire behind a flag + test + measure, or delete
(Hard Rule 7).

| Phase | Symbol | Gap |
|---|---|---|
| 2 risk matrix | `core/analyzers/analyzer_elo.py::EloTracker.select_minimax_scheduler`, `record_match(target=)` | Live fuzzer builds `BayesianEloTracker` (`core/analyzer_registry.py`); `EloTracker(use_minimax=True)` never constructed. No `--use-minimax` CLI flag. Offline half works: `bench_paired.py --risk-matrix`. |
| 3 comparison wall | `core/smt_solver.py::Z3Solver.solve_comparison_wall` / `_alpha_beta_wall` | No caller. Ordering fixed in `de9b09b4` (tested: `tests/test_regression_alpha_beta_wall_disjunctive.py`). |
| 4 operator sequencing | `core/schedulers/op_monte_carlo.py::MonteCarloScheduler.select_op_minimax` | No caller, no test. |
| 5 robust corpus | `services/corpus_manager.py::CorpusManager.minimax_robust_admission` → `core/rate_distortion.py::minimax_robust_corpus_admission` / `minimax_robust_pruning` | No caller, no test. |

Validation owed if wired:

- **Phase 2:** heterogeneous set (png, jpeg, ffmpeg, sqlite). Prediction: lower
  variance of edge-discovery rate vs Elo mix, 5–15% lower mean. Falsified if
  indistinguishable from Elo (no scheduler catastrophically bad on any target).
- **Phase 3:** beats Z3-only on ≥3 sequential walls (PNG signature → IHDR → IDAT
  CRC → filter). Slower on single comparisons is expected, not a failure.
- **Phase 4:** vs Thompson on comparison-wall targets; edge rate and
  time-to-first-crash. Beam/depth cost must not regress EPS (Hard Rule 41).
- **Phase 5:** vs greedy set-cover on resilience (remove one seed, measure
  coverage drop).

## 3. Open research questions

Only meaningful if the matching phase is kept.

1. Transposition table for the operator game tree (Phase 4): same sequence via
   different paths; cache minimax values.
2. Alpha-beta at the mutation level (search mutations within one seed; target
   response as minimizer) — the direct route to walls (Phase 3).
3. Spectral gap (`MonteCarloScheduler.spectral_gap()`) vs minimax value: small
   gap = stuck operator cycle, where lookahead should help most. Testable as a
   gate for Phase 4.
