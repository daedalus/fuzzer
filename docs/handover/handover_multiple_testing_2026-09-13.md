# Multiple-testing correction across concurrent dispersion/Ljung-Box tests

Date: 2026-09-13. Base: `85f07ad` (post op_katz/op_tang/Strahler pull).

## What prompted this

A statistical-analysis audit of the tree found that three independent
formal hypothesis tests run every stats tick on the *same* per-tick
edge-discovery-count series (`delta` in `services/fuzzer.py`'s stats loop):

- `structure_function.DispersionIndex.dispersion_pvalue` -- Poisson
  dispersion test, `(n-1)*D ~ chi-squared(n-1)`.
- `discovery_uniformity.dispersion_pvalue` -- the *same* statistic,
  independently windowed (different `min_obs`/window defaults).
- `garch.ljung_box` -- Ljung-Box portmanteau test for ARCH effects, a
  different null, same input series.

Verified by reading the call site directly (`services/fuzzer.py`, the
`effective_interval` stats-tick block): `self._structure_fn.update(delta)`,
`self._garch.update(delta)`, `self._discovery_uniformity.update(delta)` are
three calls back to back on the identical `delta` value. `structure_fn`'s
own dispersion test additionally already gates a real decision
(`is_overdispersed()` overrides stall detection; `is_underdispersed()`
tightens the stall threshold -- see `_should_stall`-equivalent logic around
`services/fuzzer.py:5933`), so its false-alarm behavior is not purely
cosmetic even before anything new is wired in.

`garch` and `discovery_uniformity` are currently display/log-only (garch
feeds `stats.py`'s printed forecast; discovery_uniformity's verdict only
logs via `log.info`), so nothing decision-affecting changes today from
their multiplicity as such -- but the moment either is wired into a
decision (an obvious next step, since both already compute a p-value
against the same series structure_function uses to gate stalling), that
wiring would compound false-positive rate with no way to see it, because
nothing in the tree tracks or corrects for testing multiple related
hypotheses on the same data.

Also worth naming precisely, because "worth naming" is not "worth
overselling" (see this repo's own P0-T1 lesson about precise vs. inflated
claims): this is **not** a case of a naively repeated single test
inflating false-alarm rate over time (the "multiple looks"/alpha-spending
problem) -- each of the three tests above is evaluated once per tick with
fresh data in its window, not re-run against a static sample. The
multiplicity here is the ordinary one: several *different* named tests,
each with its own nominal alpha, read together.

## What was built

`core/multiple_testing.py` -- pure, dependency-free (matches
`chi_squared.py`'s "no scipy" convention):

- `holm_bonferroni(named_pvalues, alpha) -> list[Correction]` -- FWER
  control, step-down procedure.
- `benjamini_hochberg(named_pvalues, alpha) -> list[Correction]` -- FDR
  control, step-up procedure, also returns BH-adjusted p-values (q-values)
  per test.
- `collect_current_pvalues(fuzzer) -> dict[str, float | None]` -- reads
  `_structure_fn.dispersion_pvalue()` (folded two-sided to match the
  others), `_discovery_uniformity.verdict()["p"]`, and
  `_garch.ljung_box()`'s p-value off a duck-typed fuzzer object. Missing
  attributes and "not enough data yet" (`None`) are both simply absent
  from the result, which is what the two correction functions expect.
- `collect_and_correct(fuzzer, alpha=0.05) -> list[Correction]` --
  convenience wrapper using BH (chosen over Holm as the default: this is a
  monitoring readout across related tests, not a single make-or-break
  call, so FDR's weaker-but-less-conservative guarantee is the better
  fit for the default path; callers wanting FWER can call
  `holm_bonferroni` directly on `collect_current_pvalues`'s output).

`Correction` is a frozen dataclass: `name`, `p_value`, `rejected`,
`adjusted` (`None` for Holm's output, since Holm's procedure is a
decision, not a closed-form adjusted value the way BH's q-value is).

28 tests in `tests/core/test_multiple_testing.py`: worked textbook examples
for both procedures, the "BH rejects a superset of what Holm rejects on
identical input" cross-check, monotonicity of BH's adjusted values, and
duck-typed-fuzzer tests for the collector (missing attributes, `None`
handling, the two-sided fold, and an end-to-end "one borderline p-value
among two null partners must not survive BH's own threshold" check that
verifies the correction is actually being applied rather than passed
through).

## Wiring: display-only, by design

`services/fuzzer.py`'s stats-tick block now also calls
`self._last_dispersion_corrections = collect_and_correct(self)` right
after the existing `discovery_uniformity` verdict log line, and
`services/stats.py` gained `_print_stats_dispersion_corrections_str`,
following the exact defensive shape of `_print_stats_garch_str`
(type-checked, not None-checked, so a test double or partially-restored
object degrades to silence rather than crashing the whole stats line).
It only prints anything when at least one test in the batch survives BH
correction -- a rare-alarm line, not a steady-state one.

**Deliberately not wired into any decision.** `structure_function`'s
`is_overdispersed`/`is_underdispersed` still gate stall detection
uncorrected, exactly as before this patch. Changing that threshold's
effective alpha without first measuring what it does to the existing
stall-detection false-positive/false-negative balance on a real campaign
is precisely the mistake this tree's own P0-T1 audit warned against
(implementing a formula correctly is not the same question as whether
plugging it into an existing decision path, uncalibrated, is an
improvement). That recalibration is a separate, explicit follow-up with
its own before/after measurement -- not bundled into a "add the
correction utility" patch.

## What's still out

- **`periodicity`'s Fisher's g-test is not in the correction batch.** It
  runs on a different (spectral) transform of a related-but-distinct
  series and has no persistent per-fuzzer attribute to read at this call
  site the way `_structure_fn`/`_discovery_uniformity`/`_garch` do --
  folding it in needs its own investigation of where it's actually
  computed (appears to be inline in `report.py`'s `_spectral_diagnostics`
  rather than a standing detector object), not a guess bolted on here.
- **The recalibration question above** -- whether `structure_function`'s
  own two internal tests (`is_overdispersed`/`is_underdispersed`, which
  *do* already gate a decision) should have their effective alpha
  adjusted for being evaluated every tick for the life of a campaign.
  That is the "multiple looks" problem this doc explicitly said the
  current patch does *not* address, and it needs its own before/after
  measurement against a real campaign's stall-detection accuracy, not a
  formula swapped in on paper.
- `CoverageRegimeDetector._classify` still ignores `discovery_rate`,
  `allan_delta`, and `exec_count` (`docs/handover/handover_pending_2026-09-06.md`,
  P2-5) -- unrelated to this patch but worth flagging again: a future
  corrected-significance signal routed toward regime classification would
  land in the same dead spot.

## Verification

`tests/core/test_multiple_testing.py`: 28/28 pass.
Targeted regression sweep (`test_discovery_uniformity.py`, `test_garch.py`,
`test_coverage_regime_garch.py`, `test_regression_analyzer_registry.py`,
`test_regression_stats_eps_stabilization.py`,
`test_regression_stats_model_fragments.py`, `test_stats_reporter.py`,
`tests/core/`): 183/183 pass.
Full `tests/test_regression_*.py` + `tests/services/test_fuzzer_calibration.py`
sweep: 2214 passed, 105 skipped, 16 failed -- all 16 pre-existing
environment gaps (missing vendored ffmpeg/sqlite build artifacts and ASAN
tree, the mypy-ratchet exempt-module list, sandbox filesystem-permission
tests), none touching `fuzzer.py`, `stats.py`, or the new module.
