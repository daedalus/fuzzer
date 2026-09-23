# Digital phase-locked loop for online periodicity tracking

**Status:** implemented, standalone diagnostic utility -- same status as
`kuramoto.py`. Not wired into `select_op`, the timeout logic, any
scheduler, `report.py`, or the CLI. HEAD at time of writing: `8e902424`.

## Trigger

User question: does the fuzzer relate to phase-locked loops (PLLs) at
all? A survey turned up the existing Kuramoto module and, initially,
seemed like it might just restate it -- worth being explicit about why
it doesn't (see the module's own docstring, "Relationship to
`core/kuramoto.py`"). Kuramoto is mutual synchronization, no reference;
a PLL locks one oscillator onto an external reference via feedback.
Genuinely different topology, no shared code.

The real, concrete gap: `services/report.py` calls
`periodicity.detect_periodicity` as a whole-history batch FFT recompute
every report tick (on the exec-time series and the discovery-edge delta
series), with no state carried between calls -- so a period that drifts
across a multi-hour campaign (e.g. thermal-throttle-induced GC/JIT
cadence stretching) gets averaged away rather than tracked. A digital
PLL is the standard tool for tracking a drifting frequency online,
cheaply (O(1)/sample vs. `detect_periodicity`'s O(N log N) per call).

## What was built

`core/pll.py`: `PhaseLockedLoop` (I/Q quadrature phase detector +
`PIController` loop filter, reused unmodified, + NCO phase accumulator +
I-channel coherence lock detector with hysteresis), `PLLState` (the
per-tick snapshot dataclass), `PhaseLockedLoop.from_period()` (bootstrap
from a `detect_periodicity(...).dominant_period`-style float).
35 tests in `tests/test_pll.py`. `ruff` clean; `mypy --strict` clean
(module not added to the exemption list in `pyproject.toml`, same as
`kuramoto.py`).

### A real design correction made during implementation

The first draft's lock detector watched a single EMA of `|error|` (the
loop filter's own driven-to-zero quadrature channel) against a
threshold derived from the fully-incoherent baseline
`E[|sin(phi)|] = 2/pi`. Empirically, that statistic cannot distinguish
genuine lock from white noise: a low-pass-filtered product of anything
uncorrelated with the NCO's phase *also* averages toward 0, for the same
reason the phase-error term does at genuine lock. Confirmed with a
synthetic sweep before shipping (matched frequency, an intentional 10%
frequency offset, white noise, a flat constant series): the noise case's
smoothed error came out close to the same magnitude as the genuinely
locked case's steady-state error, not clearly separated by any
threshold.

Fixed by adding a second, in-phase (I) mixer channel purely for lock
detection -- the standard I/Q coherent-detection approach from lock-in
amplifiers and Costas loops. At genuine lock the I channel sits near a
fixed nonzero level (~0.55-0.65 empirically, for a clean sinusoid at the
gains this module ships with); for incoherent input it stays near 0.
Verified against 5 white-noise seeds, a flat series, a matched-frequency
sinusoid, and a 10%-off-frequency sinusoid -- all behave as expected,
including hysteresis (hold state inside the lock/unlock band) and
`min_lock_ticks` (delays first lock, doesn't prevent it).

A second correction: the original docstring claimed the loop filter's
own integral action would low-pass the phase detector's double-frequency
term without a separate filter stage. Empirically false at gains fast
enough to track real drift -- produced a self-sustaining off-frequency
limit cycle (steady nonzero error, frequency estimate stuck ~4% off the
true value) instead of a true lock. Fixed by adding an explicit EMA
low-pass (`detector_alpha`) on both mixer channels before the loop
filter and the coherence statistic; corrected the docstring to match
rather than leave the wrong claim in place.

## What was deliberately not done

- **Not wired into `report.py` or any analyzer/scheduler.** Per the
  user's own framing when confirming this work: ship the tracker as a
  diagnostic first, see whether lock/unlock transitions against a real
  campaign's `f._exec_time_tracker` or `f._discovery_edges` correlate
  with anything the fuzzer already cares about (a stall, a corpus-sync
  artifact, a thermal-throttle window) before it becomes a detector.
  That observation step -- the `analyzer_kuramoto_sync.py` equivalent
  for this module -- is the natural next unit of work, not done here.
- **Gains/thresholds are not empirically tuned against a real
  campaign.** `kp=0.005`, `ki=0.0002`, `detector_alpha=0.05`,
  `lock_threshold=0.45`, `unlock_threshold=0.25` were tuned against this
  module's own synthetic test scenarios (periods in the 15-30 sample
  range). A much shorter or longer real target period likely needs
  different values -- flagged in the module docstring the same way
  `op_kuramoto.py` flags its borrowed `explore_floor=0.06`.
- **A second candidate from the same analysis (parallel-worker
  corpus-sync cadence drift-correction, extending `cadence.py`'s static
  per-site offsets) was raised but not pursued** -- no `parallel.py` or
  equivalent corpus-sync module was found in the current tree to verify
  the idea against, so it wasn't claimed as a grounded gap.

## Next steps (not started)

1. Build the observation layer: instrument a live or replayed campaign's
   exec-time and discovery-edge series through `PhaseLockedLoop`
   (bootstrapped via `from_period` from that series' own
   `detect_periodicity` call) and log lock/unlock transitions alongside
   the fuzzer's existing stall/corpus-sync/thermal signals, purely for
   correlation -- no behavior change.
2. If step 1 finds a real correlation worth acting on, that's the point
   to design an actual consumer (an analyzer, following
   `analyzer_kuramoto_sync.py`'s pattern) and empirically tune gains
   against the real series instead of synthetic ones.
3. Persistence (`to_dict`/`from_dict`, matching `PIController`'s own
   save/load convention used by `analyzer_temperature_control.py`) was
   not added -- not needed until there's a consumer that needs to
   survive a resume, same reasoning `kuramoto.py` itself uses.
