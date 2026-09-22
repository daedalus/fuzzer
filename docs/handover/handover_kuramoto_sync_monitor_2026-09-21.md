# KuramotoSyncMonitor: wiring `OpKuramotoScheduler.diagnostics()` into live telemetry

## Background

`docs/action_plan_compositional_stability.md` (consolidating an integration
analysis against Murat Demirbas's "In Search of a Compositional Theory of
Self-Stabilization") named its P0 item "wire `core/kuramoto.py`'s
`critical_coupling`/`spectral_radius` into live telemetry," on the premise
that no such wiring existed yet.

That premise was half right. `core/kuramoto.py` itself is still a
standalone diagnostic (per its own docstring and
`handover_kuramoto_oscillators_2026-09-19.md`), but
`core/schedulers/op_kuramoto.py`'s `OpKuramotoScheduler.diagnostics()`
already computes `spectral_radius`/`critical_coupling` live, against its
own real operator-transition graph. What was actually still missing —
confirmed by `grep -rn "\.diagnostics(" src/ tests/ docs/` returning zero
hits outside `op_kuramoto.py`'s own module and tests — is that nothing
*calls* `diagnostics()` on a schedule, logs it, or watches its order
parameter r(t) for a trend. This patch is that missing piece, not a
reimplementation of what `diagnostics()` already does.

## What this adds

`core/analyzers/analyzer_kuramoto_sync.py`: `KuramotoSyncMonitor`, a small
wrapper around the existing `CriticalSlowingDown` detector (unmodified —
no new variance/autocorrelation math), applied to the order parameter r(t)
instead of the discovery-rate series it normally watches. `observe(diag)`
ingests one `OpKuramotoScheduler.diagnostics()` snapshot; `status()`
returns `(detected, reason)` with the same contract
`CriticalSlowingDown.is_approaching_transition()` already has.

Wiring, all read-only:

- `core/analyzer_registry.py`: registered as `kuramoto_sync`, soft-requires
  `f._op_kuramoto` (available only when `--op-kuramoto` is on — no new CLI
  flag of its own).
- `services/stats.py`: `_print_stats_kuramoto_sync_str` samples
  `diagnostics()` once per stats tick and appends
  `| sync: r=.. Kc=..` (plus a `[SYNC: ...]` bracket on detection) to the
  live status line, mirroring `_print_stats_dr_str`'s CSD bracket.
- `services/report.py`: `_distribution_diagnostics` gets a "Kuramoto order
  param (r)" block (mean/stddev/skew/kurt + latest Kc/rho), mirroring the
  existing "Discovery rate" block sourced from `f._csd`.

Neither call site feeds back into `select_op` — `diagnostics()`'s own
"never consumed by `select_op` itself" discipline is preserved exactly;
this only adds a second reader.

State is deliberately **not** persisted across resume, matching `csd`
(check: no `state_store.set("csd", ...)` call exists anywhere either) —
both recalibrate within `min_observations` ticks, so the added complexity
of resume-restore isn't worth it for either.

## Verification

`tests/test_analyzer_kuramoto_sync.py` (new, 12 tests): construction
defaults, `observe()` field extraction and graceful degradation on a
partial/missing-key dict, `status()`'s `n_arms < 2` gate, a detection test
mirroring `test_critical_slowing.py`'s own rising-series case, reset, and a
save/load roundtrip.

`test_regression_analyzer_registry.py`: `kuramoto_sync` added to
`_FLAG_GATED`/`_ALL_NAMES`, plus explicit on/off gating tests.

`_print_stats_kuramoto_sync_str` initially broke four existing
`test_stats_reporter.py`/`test_regression_stats_eps_stabilization.py`
tests that construct `f = MagicMock()`: an unconfigured Mock answers
`f._op_kuramoto` with another Mock (truthy, not `None`), so the `is None`
guard alone didn't stop `diagnostics()` from being called on a Mock and
returning a Mock where a dict was expected. Fixed with the same
type-check-not-None-check discipline `_print_stats_garch_str` already
documents for exactly this failure mode.

Full affected suite (`test_analyzer_kuramoto_sync.py`,
`test_regression_analyzer_registry.py`, `test_stats_reporter.py`,
`test_regression_stats_eps_stabilization.py`, `test_report.py`,
`test_op_kuramoto.py`, `test_op_katz.py`, `test_kuramoto.py`,
`test_regression_enabled_features_op_kuramoto.py`): 248/248 pass.

## Suggested next steps

- Let this accumulate real telemetry across a `--op-kuramoto` campaign,
  then check the cross-correlation `docs/action_plan_compositional_stability.md`
  actually asked for: does rising variance in per-operator reward rates
  (`analyzer_critical_slowing.py`'s own series) precede a rise in r(t)
  here, the way critical-slowing-down theory predicts one precursor should
  track the other?
- P1 of the same action plan (badness-indexed exploration floor) is a
  separate, unrelated change — see
  `handover_badness_indexed_floor_2026-09-21.md`.
