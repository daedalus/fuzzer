# Handover — GARCH for discovery-rate volatility modelling

**Date:** 2026-09-05
**Base:** `6b3b7c3` ("refactor(fractal_voronoi): hold the geometry caches on the instance")
**Status:** Analysis only. No code landed. GARCH is absent from the tree.
Positions a parametric conditional-variance model against the existing
discovery-rate signal stack (CSD + Allan + Kalman + CoverageRegime).

---

## 0. Scope

GARCH(1,1) (and low-order variants) models the *conditional* variance of a
time series:

\[
\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2
\]

with \(\omega>0\), \(\alpha,\beta\ge0\), \(\alpha+\beta<1\).

Applied to the edge-discovery rate (or per-operator reward / new-edge series)
it would supply a forward-looking volatility forecast. This document records
what already exists, what GARCH would add, placement options, and the
decision criteria for whether it earns a seat.

---

## 1. Inventory — existing variance / regime surface

| Module | Role | Lines |
|---|---|---|
| `core/running_stats.py` | Online mean/var/skew/kurtosis (Welford/Pébay, windowed) | 271 |
| `core/critical_slowing.py` | Rising variance + lag-1 autocorr + skew as bifurcation precursor | 332 |
| `core/allan_variance.py` | Overlapping Allan deviation + dispersion index for stall/fatigue | 360 |
| `core/kalman.py` | 1-D / 2-D KF; denoised discovery rate upstream of CSD/Allan | — |
| `core/coverage_regime.py` | Combines CSD + homogeneity + stall into percolation phase labels | — |
| `core/elo.py` | Uses `RunningMoments` for kurtosis-scaled UCB trust gate | 1012 |

`CoverageRegimeDetector` already consumes the CSD state and produces
actionable regime labels for the main loop. Kalman supplies a causal
smoothed rate with uncertainty. No module maintains an autoregressive
model of the *conditional variance process itself*.

Operator and seed scheduling already consume variance estimates
(Thompson posteriors, Elo UCB, GP-UCB, etc.) but treat them as static
moments, not as a dynamical system.

---

## 2. What GARCH would provide

1. **Volatility clustering**  
   High-discovery periods tend to follow high-discovery periods. CSD reacts
   to *rising* variance; GARCH forecasts the *next* conditional variance
   \(\hat\sigma_{t+1}^2\).

2. **Forward-looking regime signal**  
   Multi-step forecasts could modulate:
   - energy / power schedules (`core/schedules.py`),
   - Elo K-factor or match weighting,
   - bandit exploration bonuses (contextual / hierarchical / GP-UCB),
   - decision to intensify vs. abandon a corpus region.

3. **Feature for contextual bandits**  
   Conditional variance or standardised residual as an extra context
   dimension for LinUCB / hierarchical Thompson.

4. **Noise-type separation**  
   Distinguishes pure heteroskedasticity / clustering from:
   - CSD’s bifurcation precursor (rising var + autocorr),
   - Allan’s stall / fatigue regimes (adev slope and level).

5. **Secondary**  
   Residual heteroskedasticity modelling for more accurate posterior
   widths on operator rewards.

---

## 3. Placement options

| Placement | Consumer | Notes |
|---|---|---|
| New `core/garch.py` (or `volatility.py`) | `CoverageRegimeDetector` or stats reporter | Cleanest; keep O(1) online update (EWMA-GARCH or recursive MLE on sliding window) |
| Inside `CriticalSlowingDown` | existing CSD callers | Risk of conflating two distinct signals; CSD is intentionally non-parametric |
| Feature vector for `contextual` / Elo | `services/operators.py`, `elo.py` | Low surface area; does not require a full regime label |
| Power-schedule multiplier | `SeedScorer` | Continuous allocation, same shape as existing energy schedules |

Preferred first step: a standalone online GARCH (or simpler EWMA variance)
module that emits \(\hat\sigma_{t+1}^2\) and a clustering flag, then wire it
as an optional input to `CoverageRegimeDetector` and as a context feature.
Do not embed strategy logic inside the detector (same rule as the current
regime combiner).

---

## 4. Cost and fit

- **Stylistic fit:** High. The tree already invests in information-dense
  statistical feedback (CSD, Allan, Kalman, percolation, mutual information,
  transfer entropy, etc.). A parametric conditional-variance model is
  consistent.
- **Hot-path cost:** Must stay O(1) per observation. Full MLE refits on a
  long window are unacceptable; recursive / EWMA-style updates or infrequent
  refits on a bounded buffer are required.
- **Hyperparameters:** \(\omega,\alpha,\beta\) (or equivalent decay) plus
  stationarity checks. Another set of knobs; must be justified by measured
  lift on discovery rate or crash yield.
- **Not required** for any current functionality. Pure incremental upgrade.

---

## 5. Decision criteria (before any code)

1. Measure the empirical autocorrelation of squared discovery-rate residuals
   on several long campaigns (png, ffmpeg, etc.). If clustering is weak,
   GARCH adds little over the existing CSD + Allan stack.
2. Prototype an online EWMA variance (simplest GARCH special case) and A/B
   it as a context feature or energy multiplier against the current
   `CoverageRegime` + power schedules.
3. Only graduate to a full GARCH(1,1) recursive update if the EWMA already
   shows measurable lift and residual autocorrelation remains.

---

## 6. Open items

- [ ] Empirical ACF / PACF of squared discovery-rate residuals on production
  campaigns (data already collectable from existing stats dumps).
- [ ] Decide whether the signal belongs in the regime detector, the
  contextual bandit feature vector, or both.
- [ ] If landed: register any new scheduler arm or context dimension through
  the existing `_OPERATOR_STRATEGY_NAMES` / services layer conventions;
  never hard-code op lists.
- [ ] Keep the same falsification + adversarial test discipline required by
  AGENTS.md for any new module.

---

## 7. Related handovers / docs

- `docs/handover/handover_percolation_theory_2026-08-31.md` — regime /
  phase classification surface.
- `docs/handover/handover_bandit_stopping_search_2026-09-02.md` — inventory
  of existing bandits and stopping rules.
- `core/critical_slowing.py`, `core/allan_variance.py`, `core/kalman.py`,
  `core/coverage_regime.py` — the concrete modules this would sit beside.
