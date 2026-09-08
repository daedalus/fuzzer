# Analyzer registry migration — handover (2026-09-08, complete)

## Context

Mirroring `core/operator_registry.py` (the single-source-of-truth dispatcher
for mutation operators) for the fuzzer's analyzer/detector components, which
were previously wired ad hoc: a local `from fuzzer_tool.core.<x> import <Y>`
import, construction, optional state-store restore, optional log line,
repeated with slightly different shape at ~20 call sites inside
`Fuzzer.__init__`.

`core/analyzer_registry.py` is now that single source of truth for all 20 of
them. `AnalyzerSpec` declares `available(fuzzer)` / `activate(fuzzer)` /
`deactivate(fuzzer)`, plus two extensions added in this pass:

- `phase: str = "main"` — `"early"` for the one analyzer (`sensitivity`)
  with a real ordering constraint on something later in `__init__`
  (`_init_seed_metadata()`, needed for resume to restore
  `sensitivity.json`). `Fuzzer.__init__` calls
  `REGISTRY.wire_all(self, phase="early")` at sensitivity's original
  position, then the regular `REGISTRY.wire_all(self)` (`phase="main"`,
  the default) later for everything else.
- `swallow_errors: bool = False` — `True` only for `checksum_learner`, whose
  original inline construction was itself wrapped in try/except (recovers
  checksum polynomials via Berlekamp-Massey/GCD, which can legitimately fail
  on inputs that don't fit that model — never meant to fail `Fuzzer()`).
  `wire_all()` catches, logs via `log.debug`, and reports `False` instead of
  propagating, exactly reproducing that one analyzer's original semantics;
  every other analyzer still fails `Fuzzer()` the way its inline
  predecessor would have.

## All 20 analyzers, migrated

| name | phase | flag | source |
|---|---|---|---|
| `fluctuation` | main | `fluctuation` | `core.fluctuation.WorkFunctional` |
| `transfer_entropy` | main | `_use_transfer_entropy` | `core.transfer_entropy.TransferEntropy` |
| `crash_mi` | main | always on | `core.crash_eta.CrashMITracker` |
| `length_tracker` | main | always on | `core.length_mi.LengthEdgeTracker` |
| `allan` | main | always on | `core.allan_variance.AllanVarianceDetector` |
| `sensitivity` | **early** | always on | `core.sensitivity.ByteSensitivityTracker` |
| `execution_time` | main | always on | `core.execution_time.ExecutionTimeTracker` |
| `exec_time_anomaly` | main | always on | `core.exec_time_anomaly.ExecTimeCalibrator` |
| `frameshift` | main | always on | `core.frameshift.FrameShift` |
| `format_learner` | main | `_learn_format_requested` | `core.format_learner.FormatLearner` |
| `corpus_compression` | main | `_corpus_ppmd_requested` | `core.corpus_compression.CorpusCompressor` |
| `elo` | main | `_use_elo` | `core.elo.BayesianEloTracker` |
| `distance` | main | `_distance_targets` | `core.distance.TargetDistance` (+ `adapters.shm.DistanceTableShm`) |
| `trace` | main | `_trace_crashes_requested` | `core.trace.CrashTracer` |
| `checksum_learner` | main | always attempted, **swallow_errors** | `core.checksum_learner.ChecksumLearner` |
| `csd` | main | always on | `core.critical_slowing.CriticalSlowingDown` |
| `coverage_homogeneity` | main | always on | `core.critical_slowing.CoverageHomogeneityDetector` |
| `garch` | main | `_use_garch` | `core.garch.OnlineGarch11` |
| `continuum` | main | `_use_continuum` | `core.navier_stokes.ContinuumField` |
| `coverage_regime` | main | always on, **composite** | `core.coverage_regime.CoverageRegimeDetector` — depends on `csd`, `coverage_homogeneity`, `garch`, `continuum` (registered directly above it; registration order guarantees they run first in the same `wire_all()` pass) |

`kalman` (`core.kalman.RobustKF`) was deliberately **not** migrated: there is
no standalone "the kalman analyzer" construction site in `Fuzzer.__init__`.
Every usage found is embedded in something else's setup (a settle-time
smoother inside the network-adapter/`NetworkRunner` construction; a separate,
unconditional usage in `services/stats.py` for filter-smoothing). Neither is
"an analyzer with a gating flag", so forcing either into this registry would
be a bad fit, not a genuine simplification.

## Call sites in `Fuzzer.__init__`

Two calls, both importing `core.analyzer_registry.REGISTRY` locally (avoids
the circular import — `services.fuzzer` is what several factories import
back, lazily, for module-level constants like `ALLAN_BUFFER_POW` and
`_OPERATOR_STRATEGY_NAMES`):

1. `REGISTRY.wire_all(self, phase="early")` — right where `sensitivity` used
   to be constructed, before `_init_seed_metadata()`.
2. `REGISTRY.wire_all(self)` (`phase="main"`) — right where the original
   five-analyzer cluster was wired (after `_state_store` / `max_len` are
   set), now covering the other 18. All required gating-flag attributes
   (`_use_elo`, `_use_garch`, `_use_continuum`, `_learn_format_requested`,
   `_corpus_ppmd_requested`, `_distance_targets`, `_use_cfg_cache`,
   `_trace_crashes_requested`) are assigned in one block immediately before
   this call.

`self._katz_channel` (K-Scheduler node channel) stays inline, unmigrated —
it's mutually exclusive with `distance` by construction (`if not targets:`),
uses `use_cfg_cache`/`target`/`debug` directly rather than through the
registry, and isn't itself an "analyzer" in the sense the others are (no
detection/estimation role — it's a coverage-channel upload). It sits
immediately after the main `wire_all()` call, unaffected by it.

## Verification

- `tests/test_regression_analyzer_registry.py`: 27 tests (was 10) — registry
  contents (all 20 names, exact set), duplicate-registration guard,
  category partitioning, coverage_regime dependency-order assertion, every
  always-on analyzer constructed with correct sizing (`max_len`, `map_size`),
  every flag-gated analyzer's on/off path, the early-phase/main-phase split,
  the checksum_learner swallow-errors path (both the real one and a
  synthetic forced-failure one), and `wire_all()`'s return-value contract.
- Full test suite diffed against a clean checkout at the same base commit
  (`e2d2bad`, `--timeout=15`): identical 66-failure set, zero diff, on two
  separate runs (one full, one excluding this migration's own test file to
  rule out cross-test interference). One `test_tracecmp_shim.py` flake seen
  in an intermediate run reproduced as a one-off (passed on rerun, not
  present in either the baseline or final diffed run) — confirmed unrelated
  to this change, not chased further.
- `mypy src/`: 270 errors in both the modified tree and a clean checkout of
  the same base commit — identical count, `analyzer_registry.py` (now ~3x
  its original size) contributes zero. Not added to the mypy
  strictness-exemption list.
- `ruff check`: clean on all three changed files.
- Smoke-tested every analyzer's on and off construction path directly
  (not just via the test suite) against both a from-scratch `Fuzzer()` and
  a fully-flagged one (`fluctuation`, `transfer_entropy`, `learn_format`,
  `corpus_ppmd`, `elo`, `trace_crashes`, `garch`, `continuum` all `True`).

## Status: done

All ~20 analyzer/detector components that were previously wired inline in
`Fuzzer.__init__` are now registered in `core/analyzer_registry.py`, the
same pattern `core/operator_registry.py` established for mutation operators.
`kalman` is the one deliberate exception, for the reason above — if a
genuine standalone "kalman analyzer" construction site appears later (as
opposed to its current embedded usages), it can be added the same way.
