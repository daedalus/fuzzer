# Implementation Note — Minimax Estimator and Algorithm for Fuzzer Enhancement

Original 2026-09-01. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_minimax_implementation_2026-09-01.md`
(research companion: `git show 1c689e8a^:docs/handover/handover_minimax_alphabeta_adversarial_search_2026-09-01.md`).
Verified against `4c021daa`.

---

## 1. E3 — A/B the `alphabeta` seed arm (png: null; ffmpeg: unrun)

The arm is no longer minimax: `26b367ab` replaced alpha-beta descent (1 distinct
seed / 300 picks, root-only) with Thompson descent over the lineage tree. Class
name, `--alphabeta` flag and state key kept
(`core/schedulers/seed_mcts.py::AlphaBetaMCTSSeedScheduler`,
`services/seed_picker.py::_pick_alphabeta_seed`).

**png result (2026-09-27, `container_signal`, `png_read_noasan.so`, 20 seeds x
10k execs):** `elo-alphabeta` vs `elo-mcts` 12W/8L/0T, McNemar p=0.50,
Wilcoxon p=0.67, median Δ +8 edges. Means 130.6 (sd 27.0) vs 131.0 (sd 20.8).
Noise: runs of the same `elo-mcts` cell span a median 10 edges (9 cells:
0-39), so Δ is inside the noise floor. Null, not bounded tightly: a ~10pt
win-rate effect needs ~100 cells. Run unlocked, 3 parallel shards (edges at a
fixed exec budget; eps not comparable). Details:
`docs/learnings/2026-09-27-alphabeta-vs-mcts-png.md`.

Open: `ffmpeg_read` (needs `tools/vendor_ffmpeg.sh`, and a new target set in `tools/lib/eval_set.py`: none contains ffmpeg). Keep `--alphabeta` only
if ffmpeg shows an effect; png gives no reason to prefer it over `--mcts`.

## 2. Phases 2–5 — wired, tested, unmeasured

Wired in PR #20 (2026-09-27) behind `--risk-matrix` (P2), `--wall-order` (P3),
`--op-minimax` (P4), `--minimax-select` / `minimize --minimax-robust` (P5); the
three fuzz flags are on under `--hail-mary`. Bench arms: `elo-op-minimax`,
`wall-order`, `minimax-select`. The "Gap" column below is the pre-wiring state.

| Phase | Symbol | Gap |
|---|---|---|
| 2 risk matrix | `core/analyzers/analyzer_elo.py::EloTracker.select_minimax_scheduler`, `record_match(target=)` | Live fuzzer builds `BayesianEloTracker` (`core/analyzer_registry.py`); `EloTracker(use_minimax=True)` never constructed. No `--use-minimax` CLI flag. Offline half works: `bench_paired.py --risk-matrix`. |
| 3 comparison wall | `core/smt_solver.py::Z3Solver.solve_comparison_wall` / `_alpha_beta_wall` | No caller. Ordering fixed in `de9b09b4` (tested: `tests/test_regression_alpha_beta_wall_disjunctive.py`). |
| 4 operator sequencing | `core/schedulers/op_monte_carlo.py::MonteCarloScheduler.select_op_minimax` | No caller, no test. |
| 5 robust corpus | `services/corpus_manager.py::CorpusManager.minimax_robust_admission` → `core/rate_distortion.py::minimax_robust_corpus_admission` / `minimax_robust_pruning` | No caller, no test. |

Validation owed:

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
