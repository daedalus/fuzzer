# Phase noise theory applied to core/pll.py

**Status:** analysis only, no code change. Same category as
`handover_c_hd_shortest_path_2026-09-*.md` and the C-HD writeup: a
concrete mathematical connection was found and is documented here, but
turning it into a patch needs a real campaign's exec-time/discovery
series, which was not available in this pass. HEAD at time of writing:
`077b369d`.

## Trigger

User question: what mathematical advantage does phase noise theory
(https://en.wikipedia.org/wiki/Phase_noise) offer the fuzzer. A survey
found `core/pll.py` and `core/analyzers/analyzer_pll.py` already exist
(`handover_pll_2026-09-22.md`, HEAD `78aa9499` at the time), so the
question narrows to: does phase-noise theory improve what's already
there, and how.

## What phase noise theory says

Phase noise treats an oscillator's phase deviation `phi(t)` from an
ideal periodic reference as a stochastic process. The key facts used
below:

- Under white frequency noise, `phi(t)` is a pure random walk (Wiener
  process): `Var(phi(t))` grows linearly in `t` with diffusion
  coefficient `D`. The resulting PSD is Lorentzian with FWHM `2D`.
- Under pink (1/f, flicker) noise, the lineshape is Voigt (Lorentzian
  convolved with Gaussian) instead of pure Lorentzian.
- Frequency drift (brown noise) shows up only at long averaging times,
  moving the line center rather than widening it.
- The Allan deviation plot is the standard tool for telling these three
  regimes apart by averaging-time scale: white noise dominates short
  averaging times, flicker gives a flat "floor" at moderate times, drift
  dominates long times.
- Leeson's equation relates loop/oscillator bandwidth choice to the
  noise PSD's slope: the loop filter bandwidth trades off tracking a
  drifting reference (wants wide bandwidth) against rejecting
  high-frequency noise (wants narrow bandwidth), and the optimal
  trade-off point is derived from the noise's spectral shape, not
  chosen by hand.

## Where this maps onto `core/pll.py`

`PhaseLockedLoop`'s own docstring and `handover_pll_2026-09-22.md`
flag two things as empirically tuned against synthetic sinusoids only,
not against a real campaign:

1. **Lock/unlock thresholds** (`lock_threshold=0.45`,
   `unlock_threshold=0.25`, on the I-channel coherence statistic
   `_i_lp`). Currently a bare pair of constants with no derivation.
2. **Loop filter gains** (`kp=0.005`, `ki=0.0002` on the
   `PIController` driven by the Q channel). Currently also bare
   constants.

Phase-noise theory gives closed-form answers for both, contingent on
knowing which noise regime the tracked series (exec-time, discovery-
edge deltas) actually sits in:

1. If the Q-channel residual behaves as a random walk under genuine
   lock (as the white-frequency-noise model predicts), its
   diffusion coefficient `D` can be estimated online from the
   observed growth rate of `Var(q)` between ticks. That gives a
   derived lock threshold (distinguish "residual growing like a
   phase random walk of coefficient `D`" from "residual is white
   noise uncorrelated with the NCO") instead of the current fixed
   0.45/0.25 pair tuned only against synthetic scenarios.
2. Leeson's equation says the loop bandwidth (a function of `kp`,
   `ki`) should be set from the noise PSD's slope: narrow bandwidth
   for pure white noise (better rejection, worse drift tracking),
   wider bandwidth once flicker/drift dominates (worse rejection,
   better tracking). This gives a principled way to choose `kp`/`ki`
   from a measured PSD slope instead of the current fixed values.
3. An online Allan-deviation-style diagnostic (two-sample variance of
   the tracked frequency estimate at increasing averaging times) would
   directly answer the open question already in `pll.py`'s own
   docstring and `kuramoto.py`'s: is a given fuzzer series' drift white
   noise, flicker, or a sustained trend (e.g. thermal throttling)? This
   is a new, independent diagnostic, not a replacement for anything
   existing.

## What this does *not* claim

No claim that the exec-time or discovery-rate series in a real
campaign actually follow a random-walk phase model — that is an
empirical question, same as the open question `pll.py`'s own
docstring already leaves unanswered for the sinusoidal assumption
itself. This analysis only shows that *if* they do (or approximately
do), phase-noise theory gives closed-form replacements for the two
values currently tuned by hand.

## What was deliberately not done

- **No `D` estimator implemented.** Doing this against synthetic data
  only would repeat the same weakness the existing thresholds already
  have (tuned against scenarios that may not resemble a real
  campaign). Per the pattern already used for the PLL module's own
  "step 1" (`analyzer_pll.py`), this should be measured against a real
  or replayed campaign's series first.
- **No Leeson-derived bandwidth selection implemented**, for the same
  reason.
- **No Allan-deviation diagnostic implemented.** Straightforward to add
  as a standalone module (same status as `kuramoto.py`/`pll.py`
  themselves) once there is a use for distinguishing the three noise
  regimes; not built speculatively here.

## Next steps (not started)

1. Run `PLLMonitor` against a real or replayed campaign long enough to
   get lock-state transitions and a Q-channel residual history on at
   least one series (exec-time is the more likely candidate; the
   `analyzer_pll.py` fuzzgoat smoke test noted the discovery series
   rarely reaches the 256-sample warm-up in a short run).
2. From that residual history, check whether `Var(q)` under sustained
   lock actually grows linearly with tick count (the random-walk
   prediction) rather than staying bounded — this is the empirical
   gate before estimating `D` for anything.
3. If step 2 holds, implement the `D`-based lock threshold and/or a
   Leeson-derived bandwidth choice as an opt-in alternative to the
   current fixed constants (same non-default, backward-compatible
   pattern already used for `badness_fn` in `OpKatzScheduler`).
4. Independently of steps 1-3, an Allan-deviation diagnostic module
   could be built and tested against synthetic white/flicker/drift
   series (same bar `pll.py` and `kuramoto.py` themselves were held to
   before any real-campaign validation) as a read-only addition to
   `analyzer_pll.py`'s summary.
