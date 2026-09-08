# Analyzer registry migration — handover (2026-09-07)

## Context

Mirroring `core/operator_registry.py` (the single-source-of-truth dispatcher
for mutation operators) for the fuzzer's analyzer/detector components, which
were previously wired ad hoc: a local `from fuzzer_tool.core.<x> import <Y>`
import, construction, optional state-store restore, optional log line,
repeated with slightly different shape at ~20 call sites inside
`Fuzzer.__init__`.

`core/analyzer_registry.py` now exists as that single source of truth.
`AnalyzerSpec` declares `available(fuzzer)` / `activate(fuzzer)` /
`deactivate(fuzzer)`; `REGISTRY.wire_all(fuzzer)` runs every registered spec
in order. See the module docstring for the full design rationale (analyzers
don't share one call signature the way mutation operators do, so each
`activate`/`deactivate` pair does its own construction rather than resolving
through one generic dispatch table).

## Done (this pass)

Migrated the cleanest, self-contained cluster — no cross-dependencies on any
other analyzer, no shared state beyond `self._state_store` / `self.max_len`:

| name | flag | source module | attrs set |
|---|---|---|---|
| `fluctuation` | `fluctuation` (bool) | `core.fluctuation.WorkFunctional` | `_fluctuation` |
| `transfer_entropy` | `_use_transfer_entropy` | `core.transfer_entropy.TransferEntropy` | `_te`, `_te_input_history`, `_te_edge_history`, `_te_history_max` |
| `crash_mi` | always on | `core.crash_eta.CrashMITracker` | `_crash_mi` |
| `length_tracker` | always on | `core.length_mi.LengthEdgeTracker` | `_length_tracker` |
| `allan` | always on | `core.allan_variance.AllanVarianceDetector` | `_allan`, `_last_allan_edge_count` |

Call site: one `_ANALYZER_REGISTRY.wire_all(self)` in `Fuzzer.__init__`,
where the five inline blocks used to be (~line 1841, right after
`_te_byte_edges` / `_use_transfer_entropy` / `_use_renyi_weight` are set).

Verified: full test suite before/after this change shows the identical set
of 61 pre-existing (environment-related — missing build tooling, z3,
ASLR/sandbox restrictions) failures, zero regressions. `mypy src/` and
`ruff check` both clean on the new module (266 pre-existing project errors,
unchanged — `analyzer_registry.py` is NOT on the mypy exemption ratchet
list, per the project's "new modules are strict by default" policy).
10 new tests in `tests/test_regression_analyzer_registry.py`.

## Not done — remaining components and why they're harder

These are still wired inline in `Fuzzer.__init__`. Each is individually
straightforward to migrate, but — unlike the five above — most have a real
dependency on another analyzer's constructed object, so they must move
together or in dependency order, not independently:

- **`checksum_learner`** (~line 1194) — `ChecksumLearner(self)`, wrapped in
  try/except (the only one of the group that swallows construction errors).
  Migrating this one means `AnalyzerSpec` needs an opt-in
  "swallow-construction-errors" mode, since `wire_all()` currently lets
  errors propagate (deliberately, to match every other analyzer's original
  fail-fast behavior) — don't blanket-catch for all specs to fit this one.

- **`execution_time` + `exec_time_anomaly`** (~1436, ~1442) —
  `ExecutionTimeTracker` / `ExecTimeCalibrator`, always-on, self-contained.
  Good next candidates — no known cross-dependency, similar shape to
  `crash_mi`/`length_tracker`.

- **`sensitivity`** (~1547) — `ByteSensitivityTracker`, flag-gated
  (`sensitivity` bool). Self-contained; good next candidate.

- **`frameshift`** (~1850) — `FrameShift(max_relations=64)`, always-on,
  self-contained. Good next candidate.

- **`garch` + `continuum` + `coverage_regime` + `critical_slowing` cluster**
  (~1876–1953) — `OnlineGarch11`, `ContinuumField`, `CriticalSlowingDown`,
  `CoverageHomogeneityDetector`, `CoverageRegimeDetector`. **Not
  independent**: `CoverageRegimeDetector` is constructed from
  `self._csd`, `self._homogeneity`, `self._garch`, `self._continuum` — i.e.
  four other analyzers' instances. This has to become either one composite
  `AnalyzerSpec` (`"coverage_regime"`) whose `activate()` builds all five
  objects together, or five specs registered in dependency order with
  `activate()` reading its dependencies off `fuzzer` (which requires
  `wire_all()` to guarantee registration order — it already does, since
  specs run in insertion order, so this is doable, just needs the four
  sub-specs registered before the composite one).

- **`format_learner` + `corpus_compression`** (~1955, ~1962) — both
  flag-gated, both self-contained as far as this survey found. Good next
  candidates.

- **`elo`** (~1995) — `BayesianEloTracker`, flag-gated, needs checking
  against scheduler wiring (`services/fuzzer.py`'s ELO reporting reads
  `self._elo` from several scheduler-selection sites — confirm none of
  those sites run before `Fuzzer.__init__` finishes, which they can't, but
  double check nothing in `__init__` itself reads `self._elo` between its
  current construction point and where `wire_all()` would move it to).

- **`distance`** (~2126) — `TargetDistance`, flag-gated.

- **`trace`** (~2191) — `CrashTracer`, flag-gated.

- **`kalman`** (~2265) — `RobustKF`, flag-gated. Also constructed
  separately (unconditionally) in `services/stats.py` — that call site is
  outside `Fuzzer.__init__` entirely and is a `Kalman`-specific
  filter-smoothing use, not the same "analyzer component" as this one;
  don't fold that usage into the registry, it's a different concern.

## Suggested order for the next pass

1. `execution_time` / `exec_time_anomaly` / `sensitivity` / `frameshift` /
   `format_learner` / `corpus_compression` — all self-contained, same shape
   as the five already done. Mechanical.
2. `elo` / `distance` / `trace` / `kalman` — self-contained but need the
   "does anything read this attribute before `__init__` finishes"
   double-check called out above.
3. `checksum_learner` — needs the opt-in error-swallowing extension to
   `AnalyzerSpec` first.
4. The `garch`/`continuum`/`coverage_regime`/`critical_slowing` cluster —
   do last; needs the composite-or-ordered-dependency design decided above.

Each pass should repeat the same verification this one did: full test suite
diffed against a clean checkout (not just "tests pass" — diff the *failure
set*, since ~61 failures are pre-existing/environmental and will otherwise
look alarming), `mypy src/` error count unchanged, `ruff check` clean.
