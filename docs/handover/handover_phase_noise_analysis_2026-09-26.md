# Phase noise theory applied to core/pll.py

**Status:** partially implemented. The one piece from the original
analysis that does not need real-campaign data to validate -- an
Allan-deviation noise-regime classifier, tested against synthetic
series the same way `kuramoto.py`/`pll.py` themselves were before any
real-campaign data was available -- is now built as
`core/allan_deviation.py`, standalone, same status as `kuramoto.py` and
`pll.py`: not wired into any scheduler, analyzer, or the CLI. The other
two pieces (a `D`-derived lock threshold, a Leeson-derived loop
bandwidth) are still analysis only -- see "What was deliberately not
done" below; they still need real-campaign data this pass did not have.
HEAD at time of writing: `077b369d`.

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

## What was built (update 2026-09-26)

`core/allan_deviation.py`:

- `allan_deviation(y, tau0=1.0, m_values=None) -> list[AllanPoint]` --
  the standard fully-overlapping Allan-variance estimator
  (`sigma_y^2(tau) = 1/(2*(N-2m)) * sum (ybar_{i+m} - ybar_i)^2`) over
  a frequency-like series, using prefix sums for O(N) work per `m`
  instead of the naive O(N*m). Defaults to power-of-two `m` up to
  `N//4`; degrades (drops the point) rather than raising when an
  explicit `m` would leave fewer than one overlapping window, same
  "degrade rather than fail" pattern `services/stats.py` already uses
  for sparse data.
- `classify_segments(points) -> list[Segment]` -- classifies the
  *local* log-log slope between each adjacent pair of points against
  the five canonical power-law exponents from the module docstring
  (phase noise -1, white FM -1/2, flicker FM 0, random-walk FM +1/2,
  drift +1), each within a 0.25 half-width band (the canonical
  exponents are spaced 0.5 apart, so the bands tile exactly with no
  gaps). Classifies locally rather than fitting one slope through the
  whole curve, since a real series typically shows different regimes
  at different averaging times -- the entire point of an Allan
  deviation plot over a single variance number.
- 23 tests in `tests/test_allan_deviation.py`: input validation,
  `AllanPoint` bookkeeping (tau/m/n_pairs arithmetic, degrade-not-raise
  on an oversized `m`, dedup, default `m_values`), and noise-regime
  classification verified two ways -- against fixed-seed synthetic
  white-frequency-noise and random-walk-frequency-noise series (the
  two regimes with the simplest, least-flaky generators: i.i.d.
  Gaussian samples and their cumulative sum, respectively; textbook
  exponents -1/2 and +1/2 confirmed numerically before writing the
  test, see the sanity check below), and directly against fabricated
  points on each of the five exact canonical slopes so the
  classification boundary itself is pinned down independent of any
  synthetic-noise generator's imperfections.

Numeric sanity check (N=8192, seed 1234, `m` in
{1,2,4,8,16,32,64,128,256,512}, 9 adjacent-pair slopes) run before
writing the test: white-FM slopes came out in [-0.804, -0.287] overall,
but the six segments spanning `m`=1 to 64 -- the range the test suite
uses -- all landed within 0.08 of the canonical -0.5 (-0.505, -0.500,
-0.494, -0.508, -0.455, -0.417); the two outliers (-0.804, -0.287) are
both in the `m`=128-512 tail. Random-walk-FM slopes came out in
[0.292, 0.516] overall, with the same `m`=1-64 range landing within
0.20 of canonical +0.5 (0.299, 0.446, 0.499, 0.506, 0.515, 0.516).
Both tails degrade at the largest `m` values (128-256, 256-512) where
the overlapping-window count is smallest -- expected estimator
behavior, documented in the module docstring, and why the test suite
restricts itself to `m` up to 64 where the window count is still in
the thousands.

`ruff` and `mypy --strict` clean on both new files (module not added to
`pyproject.toml`'s exemption list, same as `kuramoto.py`/`pll.py`); full
related-suite run (`test_allan_deviation.py` + `test_pll.py` +
`test_analyzer_pll.py` + `test_kuramoto.py`, 106 tests) passes clean.

## What was deliberately not done

- **No `D` estimator for the lock threshold, and no Leeson-derived
  bandwidth selection.** Both still need a real (or replayed) fuzzer
  campaign's actual exec-time/discovery series to validate against --
  synthetic-only tuning here would repeat the exact weakness the
  existing fixed thresholds/gains already have. `allan_deviation.py`
  itself was safe to build synthetic-only because it makes no claim
  about the fuzzer's own series (same bar `kuramoto.py`/`pll.py` were
  held to); a lock threshold or bandwidth choice *does* make such a
  claim, so it stays gated on real data.
- **Not wired into `analyzer_pll.py`, `PhaseLockedLoop`, or anything
  else.** Per the same reasoning `pll.py` and `kuramoto.py` themselves
  used: ship the diagnostic standalone first, see whether it says
  anything useful about a real campaign's series before it becomes an
  input to anything.

## Next steps (not started)

1. Run `PLLMonitor` against a real or replayed campaign long enough to
   get lock-state transitions and a Q-channel residual history on at
   least one series (exec-time is the more likely candidate; the
   `analyzer_pll.py` fuzzgoat smoke test noted the discovery series
   rarely reaches the 256-sample warm-up in a short run). Feed the same
   series through `allan_deviation.allan_deviation` +
   `classify_segments` to see which noise regime(s) it actually shows,
   and at which averaging times.
2. From the Q-channel residual history, check whether `Var(q)` under
   sustained lock actually grows linearly with tick count (the
   random-walk prediction) rather than staying bounded -- this is the
   empirical gate before estimating `D` for anything.
3. If step 2 holds, implement the `D`-based lock threshold and/or a
   Leeson-derived bandwidth choice as an opt-in alternative to the
   current fixed constants (same non-default, backward-compatible
   pattern already used for `badness_fn` in `OpKatzScheduler`).
4. If step 1 shows a real series in a clean single regime over a wide
   `tau` range, consider wiring `allan_deviation` into
   `analyzer_pll.py`'s summary as a read-only addition -- not done
   here since there is no real-campaign result yet to confirm it says
   anything the fuzzer doesn't already know.
