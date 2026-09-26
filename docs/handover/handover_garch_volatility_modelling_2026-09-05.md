# Handover — GARCH for discovery-rate volatility modelling

**Original:** 2026-09-05. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_garch_volatility_modelling_2026-09-05.md`.
**Verified against:** `4c021daa`. `OnlineGarch11`
(`core/analyzers/analyzer_garch.py`) is live behind `--garch` (also in
`--hail-mary`), fed per-tick edge deltas in `services/fuzzer.py`, consumed by
`CoverageRegimeDetector._garch_spike` (`core/analyzers/analyzer_coverage_regime.py`).
Built before its own gating measurements (§5 of the original); those are open.

---

## 1. Empirical ACF of squared discovery residuals

**What.** Measure autocorrelation of squared per-tick edge deltas on long
campaigns (png, ffmpeg). Weak clustering ⇒ GARCH adds nothing over CSD +
structure function; drop it.

**Constraints.**
- Use the **non-overlapping** per-tick delta, not
  `stats_reporter.discovery_rate()`: its 5-snapshot window is MA(4) and
  manufactures ARCH on pure noise (α≈0.54). Only lags > `OVERLAP_ARTIFACT_LAGS`
  (4) count; `arch_effect` encodes this.
- Dump the series to disk. Live history is capped (`record_discovery_snapshot`
  in `services/stats_reporter.py`: 500, trimmed to 250); one tick ≈ 10 s, MLE
  GARCH(1,1) wants 500–1000 points.
- Sampling axis: the tick (`Fuzzer._stats_effective_interval`) is fixed in wall
  clock, variable in executions. State which axis the result is on.

**Acceptance.** Ljung-Box on lags 5–10 per target, recorded in `docs/learnings/`.

## 2. A/B the `garch` arm

`tools/lib/bench_paired.py` registers `garch` (`--garch` vs `baseline`); never
run. Use the Boltzmann protocol (`docs/learnings/2026-08-30-boltzmann-ab-result.md`).
On a null or loss, drop `garch` from `_HAIL_MARY_FLAGS` (`cli/commands.py`).

## 3. Context-feature placement — undecided

σ̂²ₜ₊₁ reaches only the regime detector. Not a feature in any contextual /
LinUCB / Elo path. Decide only after §1–2 show signal; register any new arm or
context dimension through `_OPERATOR_STRATEGY_NAMES` / the services layer.
