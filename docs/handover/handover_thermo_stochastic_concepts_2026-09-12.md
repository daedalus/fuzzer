# Handover: thermal/stochastic-process concepts vs. current tree

**Date:** 2026-09-12, **audited and re-prioritised 2026-09-13**,
**implementation audited 2026-09-13 (later)**
**Base of original analysis:** `f17a3ef`
**Base of the audit revision:** `94d1741`
**Base of this revision:** `73a4996`

**Status: P0-T1 and P2-T4 are closed. P0-T3, P3-T5 and P4-T6 remain open.**
§4 records what landed, what the acceptance tests returned, and the one
cost the fix carried that nobody had measured.
**Companion:** `handover_pending_2026-09-06.md` — the tier scheme below (P0/P1/P2/P3/P4)
is that document's, so these items can fold into it without re-ranking.

Requested analysis: how do thermal equilibrium, Brownian motion, the
equipartition theorem, dynamic equilibrium, the binomial distribution,
quadratic variation, power spectral density, and least squares map onto
this codebase.

**The 2026-09-12 revision of this document answered the question by
inventory — "does a module for this exist" — and four of its five
"already implemented, no new work justified" entries are wrong in the
way that matters: the module exists, and it does not compute the thing
it is named after, or its null hypothesis is contradicted by another
module in the same tree.** Everything below was re-measured in a
container at `94d1741`. Numbers are from that run unless stated.

A `[ ]` is not evidence and neither is a module name. The lesson of this
audit is narrower and worth keeping: **a concept is not "covered"
because a file is named after it.** Three of the four corrections below
were found by deriving what the code computes and comparing it to what
the file claims, not by looking for missing files.

---

## 0. Verdict table

| Concept | 09-12 verdict | 09-13 verdict |
|---|---|---|
| Power spectral density | covered, no work | **covered, but its null is wrong for the series it is applied to** → P0-T3 |
| Least squares | covered, no work | covered; two secondary defects fold into P0-T1 |
| Binomial distribution | covered, no work | **confirmed covered, and better than claimed** — no action |
| Quadratic variation | "is what Allan variance computes" | right about this code, wrong about why → P0-T1, **CLOSED `0e11fd2`** |
| Brownian motion (implicit null) | covered via Allan slope | the estimator could not separate the two hypotheses → P0-T1, **CLOSED `0e11fd2`** |
| Brownian motion on `distance.py` | genuine gap, implement | genuine, but **gated on two facts the document got wrong** → P3-T5 |
| Dynamic equilibrium (corpus flux) | genuine gap, implement | **confirmed sound** → P4-T6 |
| Thermal equilibrium / equipartition | rejected, no action | rejected for a provable reason instead of a metaphorical one; the tree already had the module the rejection overlooked → P2-T4, **CLOSED `155bb54`** |

---

## 1. What the 09-12 revision got right

Kept, re-verified, no action:

- **PSD machinery exists.** `core/periodicity.py` computes the power
  spectrum via `numpy.fft.rfft` and inverts it (Wiener–Khinchin) for
  autocorrelation-based record-size detection, gated by Fisher's
  g-test. `power = power * power.conj()` at `:137-139` is a real PSD,
  not the "forgot to square it" error this shape of code usually has.
  The Hanning window and the `SIGMA_CUTOFF / sqrt(w)` noise floor are
  both correct and both documented.
- **Nonlinear least squares exists.** `edge_tracker.py`
  `bayesian_coverage_growth_model` fits `E(t) = A*(1-exp(-k*t))` by
  Levenberg–Marquardt with a Laplace posterior covariance and a
  delta-method `p_stalled`. Also a closed-form log-log OLS in
  `allan_variance.py::noise_slope`.
- **Binomial distribution is covered, and more completely than the
  09-12 text says.** It credits `report.py` with "Wald and Wilson"
  intervals. The tree is past that: `_wilson_interval` (`report.py:88`)
  exists *specifically because* the Wald interval degenerates at small
  `p` and can emit negative lower bounds, and its docstring says so.
  Discovery probabilities are exactly the small-`p` regime, so this is
  a closed item, not a live one. Do not re-propose a binomial-CI fix.
- **Dynamic equilibrium on corpus size** is a real gap and the 09-12
  framing of it survives audit — see P4-T6.

---

## P0 — defects in shipped code

### P0-T1. `core/allan_variance.py` does not compute the Allan variance, and the fatigue threshold misfires on stationary long-memory noise — **CLOSED `0e11fd2`, see §4.1**

> *Anchors in this section are pre-`29f0d60` and are left as they were
> measured: the file is now `core/structure_function.py`. The defect is
> described here in the past tense it deserves; what replaced it is §4.1.*

`adev(tau)` (`allan_variance.py:112-126`) computes

    sqrt( 0.5 * mean_i (x[i+tau] - x[i])^2 )

That is the **second-order structure function** — the variogram, i.e.
normalised quadratic variation at lag `tau`. It is a *first* difference
of the raw samples. The Allan variance is a first difference of
*block averages*, equivalently a *second* difference of the cumulative
series. They are different estimators with different noise-type
signatures, and separating white from flicker noise is the entire
historical reason the Allan variance exists.

Measured, 200 replicates, N=256, `noise_slope()` vs. a textbook
overlapping Allan variance on the same series:

| series | tree slope | true Allan slope | tree `noise_type()` |
|---|---|---|---|
| white i.i.d. | 0.001 ± 0.021 | −0.532 ± 0.115 | `active` 200/200 |
| flicker 1/f | **0.106 ± 0.034** | −0.041 ± 0.082 | **`fatiguing` 117/200** |
| random walk | 0.441 ± 0.124 | 0.408 ± 0.167 | `fatiguing` 200/200 |
| linear downtrend | 0.632 ± 0.017 | 0.837 ± 0.016 | `fatiguing` 200/200 |

The white-noise column differs by exactly the 0.5 the theory predicts,
which confirms the identification rather than merely suggesting it.

**Why this is P0 and not a naming nit.** `_FATIGUE_SLOPE_THRESHOLD =
0.1` sits precisely where a **stationary** long-memory process lands.
Sweeping the spectral exponent of a stationary 1/f^beta rate series
(400 replicates each):

| beta | mean slope | fraction called `fatiguing` |
|---|---|---|
| 0.00 | −0.001 | 0.000 |
| 0.50 | 0.024 | 0.000 |
| 0.75 | 0.057 | 0.025 |
| **1.00** | **0.105** | **0.593** |
| 1.25 | 0.177 | 0.927 |
| 1.50 | 0.256 | 0.988 |

A campaign whose discovery rate has 1/f structure and **no decline at
all** is classified as approaching stall on a coin flip. That
classification is not a display: `fuzzer.py:5895-5901` halves the stall
threshold on `noise == "fatiguing"` (`max(self._stall_threshold // 2,
100)`), which pulls `_stall_recovery_enter` forward, which is what
drives `--reseed-on-stall` and the bitmap resize. And the detector is
**default-on**: its `AnalyzerSpec` (`analyzer_registry.py:324-330`) has
no `available` gate, so it is in the "always constructed" class.

The class docstring already says "Classification is tailored to
edge-discovery-rate signals, not generic noise theory", which is honest
about the thresholds being empirical. It does not rescue the module
header, which says "Overlapping Allan variance for noise-type
identification" — noise-type identification is the one thing this
estimator cannot do.

**Two secondary defects in the same file, which must move with the
above and not separately, because both change `adev` values and
therefore the classification:**

- `count = n - 2*tau` (`:122`), but a lag-`tau` first difference has
  `n - tau` valid pairs. Discarded pairs: 1.6% at tau=4, 14.3% at
  tau=32, **33.3% at tau=64** — precisely the noisiest point, which the
  unweighted fit then over-weights. `adev` also requires `2*tau+1`
  samples where `tau+1` would do.
- `noise_slope` is an **unweighted** OLS over at most five points
  (tau=4..64). Allan/variogram estimates at large tau have far fewer
  equivalent degrees of freedom, so equal weighting is the textbook
  weighted-least-squares case. Measured on ground truth, 600
  replicates, weights ∝ `count/tau`: Brownian RMSE **0.134 → 0.095**
  and bias −0.044 → −0.023; white noise is a wash (0.0212 → 0.0239).
  So weighting helps exactly in the non-stationary case the classifier
  acts on, and is neutral elsewhere.

**The decision to make on paper first.** Two routes, and they are not
interchangeable:

1. **Rename and recalibrate.** Keep the variogram, fix the name and the
   header formula, and re-derive `_FATIGUE_SLOPE_THRESHOLD` with a
   stated margin against the 1/f null instead of against nothing. Cheap,
   no new estimator, but permanently gives up white-vs-flicker.
2. **Implement the real Allan variance** (second difference of the
   cumulative series) and re-derive all three thresholds
   (`_ADEV_ACTIVE_THRESHOLD`, `_ADEV_STALL_THRESHOLD`,
   `_FATIGUE_SLOPE_THRESHOLD`) against it. Gains the noise-type
   separation the module header claims; costs a full recalibration of a
   default-on stall path.

**Acceptance test, either route:** the beta sweep above, as a table, with
a stated maximum false-`fatiguing` rate at beta=1.0. The current value
is 0.593. A fix that does not report this number has not been tested.
**Falsifier before trusting any replacement:** feed synthetic white,
flicker and random-walk series and confirm the estimator recovers the
slope its own theory predicts — the table above is the harness.

**Open question this audit does not answer:** whether real discovery
series actually have beta near 1. I could not measure it —
`_discovery_edges` reached only 13 snapshots in a 20k-exec run (see §3).
This does not soften the item: a threshold with no margin against the
most plausible stationary alternative is a defect whether or not that
alternative is currently realised.

### P0-T3. Fisher's g-test assumes a constant rate; the series it is given is the one the rest of the tree exists to prove is not constant

`report.py::_spectral_diagnostics` differences cumulative edges and
hands the deltas to `detect_periodicity`, whose null (`fisher_g_pvalue`)
is i.i.d. exponential periodogram ordinates — Gaussian white noise, i.e.
a *constant* discovery rate. `coverage_regime.py`, `critical_slowing.py`
and `garch.py` all exist on the premise that this rate is
non-stationary.

Measured false-positive rate at nominal alpha=0.05, 2000 replicates, on
a Cox process (Poisson counts, slowly drifting OU rate — the honest
model of discovery):

| n | white (control) | Cox, drifting rate | median peak bin |
|---|---|---|---|
| 128 | 0.049 | 0.176 | 2 |
| 256 | 0.048 | 0.378 | 2 |
| 512 | 0.050 | **0.540** | 3 |

At n=512 sync intervals a campaign with no periodicity whatsoever is
told `PERIODIC — possible corpus-sync artifact` more than half the
time, and that message is an active mis-steer toward a cause that is
not there. The existing `peak_bin >= 2` guard is the reason the median
lands at 2–3: it catches bin 1 and nothing above it. AR(1) shows the
same shape and worse — n=512, phi=0.7 → **0.942**.

**Two things this audit expected and falsified, recorded so they are not
re-proposed:**

- **Volatility clustering is not the problem.** GARCH(1,1) work with
  zero mean-level autocorrelation gives 0.044 against a 0.049 control;
  piecewise-constant variance gives 0.043. Variance clustering keeps
  the ordinates exchangeable in expectation, so `garch.py`'s premise is
  **not** in conflict with Fisher's g here. The conflict is mean-level
  rate drift only.
- **Pure 1/f is already handled.** beta=1 and beta=2 both give 0.000,
  because the argmax collapses into bin 1 and the existing guard
  rejects it. The gap is the mid-band, not the red end.

**Fix:** pre-whiten before the periodogram (fit AR(1) and test the
residuals, or difference a second time) or replace the white null with
an autocorrelation-aware one. Ships with the table above as its test.

**Ordered last in P0** because it corrupts a *display* and not a
decision — the same rule that puts `shapley._prune_edges` last in the
companion document's P0.

**Not a defect, checked:** the second scan in the same function, over
`_exec_time_tracker._times`, is fine. Real exec-time series captured
from live in-process runs against `zlib_read` and `png_read` are close
to white (|acf| ≤ 0.13 at lags 1–5), and `detect_periodicity` returned
not-significant on them. No action for that scan.

---

## P2 — shipped but unwired

### P2-T4. `--fluctuation` computes `L*log(L)` and nothing else — **CLOSED `155bb54`, see §4.2**

> *Anchors in this section are pre-`155bb54`. `_op_probability` and
> `crooks_forward_reverse` no longer exist; what replaced them is §4.2.*

The 09-12 revision rejected thermal equilibrium and equipartition as
metaphors with "no natural target" and recommended no action. It did
not notice that **`core/fluctuation.py` already exists** — a
`WorkFunctional` with an inverse temperature `beta`, a Jarzynski
estimator and a `crooks_forward_reverse`, behind `--fluctuation`,
persisted through `state_store` and displayed by `stats.py:1169`. So
the concept is not absent; it shipped, and the question worth asking is
whether its premise holds. It does not, in three independent ways.

**(a) The quantity is not a free energy. It is a Rényi entropy, exactly.**
With `W(tau) = sum_i -log p_i` and a trajectory drawn from those same
`p_i`, `e^{-beta W} = P(tau)^beta`, so

    -log( E[e^{-beta W}] ) / beta  ==  H_{1+beta}(trajectory distribution)

the Rényi entropy of order `1+beta`. Verified numerically against
ground truth on a known product distribution (6 operators, length-3
trajectories, 400k samples per point):

| beta | `jarzynski_estimator` | Rényi_{1+beta} truth |
|---|---|---|
| 0.01 | 5.07781 | 5.07593 |
| 0.25 | 5.01439 | 5.01264 |
| 0.50 | 4.95057 | 4.95016 |
| 1.00 | 4.83393 | 4.83573 |
| 2.00 | 4.64610 | 4.64641 |

The default `beta=1.0` is therefore collision entropy, `beta -> 0` is
Shannon. `core/renyi.py` already exists and already computes Rényi
spectra — on the edge hit-count distribution rather than the
operator-path distribution, so it is an adjacent object, not a
duplicate. This also settles the equipartition question on its own
terms rather than by analogy: `beta` here is an **entropy order**, not
a temperature, so there is no ensemble to equipartition and no
fluctuation–dissipation relation to exploit. That conclusion matches
the 09-12 revision's; the argument does not.

**(b) In production it is not even that.** `self._operators._available`
(`fuzzer.py:3054`) **does not exist anywhere in the tree** — the only
references are the `list(...)`/`hasattr(...)` pair at `:3054-3056`
itself; the sole `_available` assignments in `src/` are on unrelated
objects (`adapters/lbr_trace.py`, `adapters/perf_event.py`,
`core/smt_solver.py`), none of them the operators service. So the
guard always falls through to
`list(self._last_ops_used)`, `available` becomes the trajectory itself,
`_op_probability` returns `1/L` for every step, and

    W = sum_{i=1..L} -log(1/L) = L * log(L)

Confirmed on a real run (`png_read`, 6000 execs, `--fluctuation`):
5,697 work samples, **7 distinct values** — 0, 1.38629, 3.29584,
5.54518, 10.75056, 16.63553, 29.81880 — which are exactly `n*log(n)`
for n ∈ {1,2,3,4,6,8,12}. The recorded signal is the mutation-stack
depth. It carries no operator identity, no scheduler state and no
coverage information, and `jarzynski_delta_f` is a deterministic
function of the distribution of stack depths.

**(c) The scheduler-aware branches are dead, and Crooks is not Crooks.**
Independently of (b): `_op_probability`'s mopt branch (`:3029`) and elo
branch (`:3045`) both `return max(1.0 / max(len(available), 1), 1e-12)`
— byte-identical to the fallback at `:3046`. Only the
`mc.bandit_stats()` branch (`:3019-3023`) returns anything
operator-dependent, and those Beta posterior means `(a+1)/(a+b+2)` are
not normalised across operators, so `-sum log p` is not the
log-probability of anything and the identity in (a) does not hold there
either. And `crooks_forward_reverse` returns a ratio of *mean works*
between two arbitrary state buffers: there is no reverse protocol, no
matched-`W` density ratio, and no crossing at ΔF. Its comment
("for identical work distributions the ratio is centered at 1.0") is
true of any ratio of equal means and tests nothing.

**This is P2 because leaving it in the middle is the worst of the three
states.** Two honest routes:

1. **Wire and rename.** Pass the real normalised selection
   probabilities from the active scheduler (this is work at the call
   site, not in the module — the same shape as `invasion_select`'s
   never-passed `frontier_edges`, P2-3 in the companion document),
   delete the dead branches and the `_available` guard, drop
   `crooks_forward_reverse`, and rename `jarzynski_delta_f` to the
   Rényi-`1+beta` quantity it provably is.
2. **Retire the module.** Defensible: the quantity it would compute is
   the Rényi entropy of the operator-path distribution, which has no
   consumer beyond a stats line, and `core/renyi.py` is the natural
   home if one is ever wanted.

Either is cheap. The reason to do it soon is not the payoff but that
the current state publishes a thermodynamic-sounding number that is
`L*log(L)`.

---

## P3 — design work, gated on a stated question

### P3-T5. Brownian motion / quadratic variation on the seed-distance trajectory

The 09-12 revision's stronger proposal, and it is a real question: the
AFLGo directed-distance signal feeds the `aflgo` power schedule, and
nothing checks whether directed scheduling produces directed
*progress*. The three-way classification it proposes — slope ≈ 1
ballistic, ≈ 0.5 Brownian, < 0.5 sub-diffusive/trapped — is the right
shape, and its falsifier (recover 0.5 on a synthetic random walk over a
reachability graph before trusting real traces) is the right falsifier.

**Three corrections that gate it, all of which the 09-12 text gets
wrong:**

1. **It is blocked by P0-T1.** The proposal is to "reuse the existing
   log-log least-squares slope machinery from `allan_variance.py`". That
   machinery is the misnamed estimator, with the unweighted five-point
   fit and the 33%-discard at tau=64. Building a second consumer on it
   ships the white-vs-flicker confusion into a new decision path. Do
   P0-T1 first; that is the whole reason it is ordered first below.
2. **There is no distance trajectory to apply it to.** `avg_distance`
   is a per-seed scalar in `seed_meta`, written at `fuzzer.py:4618` and
   `:4636` and read at `seed_picker.py:457` and `:1021`. It is
   overwritten, not appended. The series the proposal assumes exists
   has to be built first, and where it is recorded (per seed over its
   own fuzzing history, or per campaign tick over the corpus mean) is a
   design choice that changes what the slope means. Answer that on
   paper before writing code.
3. **The signal is conditional and can be constant.** A live run in
   this container printed `No target functions found, using distance=1
   for all` — with no `--target-functions` the distance channel is a
   constant, the variogram is identically zero, and every tau point is
   excluded as non-positive. Add the feasibility gate that the
   companion document's P2-1 lesson demands: define the behaviour for
   the no-graph case *before* shipping, or this lands, passes its tests,
   and is never read — the `core/target_difficulty.py` failure mode.

---

## P4 — genuine, blocking nothing

### P4-T6. Dynamic equilibrium: gross corpus flux, not net corpus size

Confirmed sound as written on 09-12. `corpus_manager.py` tracks
additions and prunes but nothing distinguishes a corpus that has
plateaued because both processes went to ~0 (stalled) from one that has
plateaued because additions and prunes are active and balanced (healthy,
still finding redundant-but-covering seeds and evicting them at the same
rate). Net size cannot tell these apart; gross flux —
`|additions| + |prunes|` tracked alongside `additions - prunes` — can.
Reuses `DispersionIndex` / `RunningMoments`, already in the tree.

Falsifier, as stated: on a synthetic run with additions and prunes both
forced to zero, the two cases must be distinguishable from the flux log
even though final corpus size is identical.

One note the 09-12 text does not make: `_maybe_prune` already computes
the exact surviving-owner count locally (see the `_edge_owner_count`
rebuild), so the prune side of the counter has a natural home and does
not need a new hook.

---

## 2. Implementation order, and why this order

1. ~~**P0-T1** — `allan_variance.py`.~~ **Done in `0e11fd2`** (route 2:
   the true Allan variance). Verified against the stated acceptance test;
   §4.1. P3-T5 is now unblocked.
2. ~~**P2-T4** — `--fluctuation`.~~ **Done in `155bb54`** (route 1:
   wire and rename), after `6cdba43` did the documentation half only;
   §4.2.
3. **P0-T3** — Fisher's g null. **Next.** Display-only, so it waits behind the
   decision paths, but it is bounded work with its test table already
   written.
4. **P4-T6** — corpus gross flux. Small, independent, no gate.
5. **P3-T5** — distance trajectory. Last: no longer gated on P0-T1,
   which has landed, but still gated on the series-definition question
   and on the no-graph behaviour.
6. **E-T7** (new) — the low-rate sensitivity that P0-T1 cost. §4.1.

Not on this list, deliberately: the PSD machinery itself, the
least-squares *existence* question, and binomial CIs. All three are
genuinely closed and re-proposing them is the failure this document
now exists to prevent.

---

## 3. Environment notes for whoever picks this up

- `pip install -e . --break-system-packages` works; numpy 2.4.4 and
  scipy 1.17.1 are present in the container. The measurements above
  need neither beyond numpy.
- `gcc` builds 26 targets. `clang` is absent, so `tools/build_targets.sh`
  aborts at its trace-cmp stage **after** those 26 are already built —
  the non-zero exit is not a failure of the simple targets.
- `--inprocess-direct` engages correctly against
  `~/fuzzing/builds/png_read_noasan.so` and `zlib_read_noasan.so`.
- **Series lengths are the binding constraint on validating P0-T1 and
  P0-T3 against real data.** `_exec_time_tracker._times` is a deque
  capped at 200. `_discovery_edges` reached 13 entries in a 20k-exec
  run, and `_spectral_diagnostics` needs 51. The n=512 regime where
  P0-T3's false-positive rate hits 0.540 needs a long campaign to reach
  at all, so the synthetic harnesses are the practical test, not a
  substitute that should later be replaced by a live measurement.
- Another Claude surface was operating on the same clone during this
  audit — the reflog shows a `pull --quiet: Fast-forward` this session
  did not issue, which is how the base moved `77d3ab8` → `94d1741`
  mid-work. That commit is docs-only
  (`handover_non_ucb_schedulers_2026-09-13.md`) and touches none of the
  measured files, so no measurement was invalidated. Re-check the base
  before applying patches from this line of work.

---

## 4. Implementation audit (2026-09-13, later the same day)

Four commits landed on top of the audit revision. `f13f7a7` is this
document itself (identical to the audited version — `git patch-id
--stable` matches). The other three are by `Grok Agent <agent@x.ai>`:
`6cdba43` (P0-T1 route 1 plus the documentation half of P2-T4),
`29f0d60` (the rename), `0e11fd2` (P0-T1 route 2). Both routes to
P0-T1 were applied in sequence; the final state is route 2.

Everything below was measured in a container at `0e11fd2` before the
follow-up commits.

### 4.1 P0-T1 — closed, and the acceptance test passes

`adev(tau)` is now the real overlapping Allan deviation, the second
difference of the cumulative series divided by `2*tau^2*(N-2tau)`. The
variogram survives as `sdev()`, documented as a secondary diagnostic.
`noise_slope` uses weighted OLS with weights ∝ `(N-2tau)/tau^2`. The
module is `core/structure_function.py` and the analyzer spec is
`structure_function`; the detector attribute is `_structure_fn`.

The stated acceptance test, 200 replicates, N=256:

| series | slope | `noise_type()` |
|---|---|---|
| white i.i.d. | −0.516 ± 0.101 | `active` 200/200 |
| flicker 1/f | −0.017 ± 0.032 | `active` **200/200** |
| random walk | +0.451 ± 0.113 | `fatiguing` 197/200 |
| linear downtrend | +0.591 ± 0.040 | `fatiguing` 200/200 |

And the beta sweep, 400 replicates, against the 0.593 that opened this
item:

| beta | mean slope | fraction `fatiguing` |
|---|---|---|
| 0.00 | −0.502 | 0.000 |
| 0.75 | −0.142 | 0.000 |
| **1.00** | **−0.016** | **0.000** |
| 1.25 | +0.106 | 0.022 |
| 1.50 | +0.225 | 0.945 |

The white-noise slope recovers the theoretical −0.5, which is the
falsifier this document asked for. Exact 1/f now sits at −0.016 against
a 0.15 threshold, and the crossover has moved out to beta ≈ 1.4. Real
margin where there was none.

**Two things the implementation did not do, and what measurement says
about each.**

*The magnitude thresholds were not re-derived*, against the stated
criterion. Measured, `adev(2)` is uniformly 0.707× the old `sdev(2)` —
exactly 1/√2, as theory predicts — so `_ADEV_STALL_THRESHOLD = 0.01`
now fires at a 41% higher discovery rate than the value was chosen
for. **It does not bite, and the reason is the argument worth keeping:**
edge counts are integers, and a single discovery anywhere in a 256-tick
window already gives `adev(2) = 0.0445`, 4.4× the threshold. Nothing
lives in the shifted gap; only an all-zero window is below it. Verified
across Poisson rates 0.02–1.0/tick. `_ADEV_ACTIVE_THRESHOLD = 0.1` is
now reachable only from the two fallback branches (`len(points) < 2`,
`slope is None`), so it matters even less. **No action — recorded so
nobody spends a day re-deriving it.**

*`update()` rebuilds the whole cumulative deque in a Python loop on
every call once the buffer is full.* The comment justifies it as keeping
`_cum[0] == 0`, which is not needed: second differences are invariant
under an additive constant, so a plain append preserves both alignment
and correctness. It is one call per stats tick (~10 s of work), so this
is cosmetic, not hot. **No action.**

### 4.1a E-T7 (new) — the cost nobody measured

Neither this document nor `0e11fd2` measured what the estimator change
cost, and it cost something. Stepping the true rate down and counting
ticks until `noise_type()` leaves `"active"`, 1024-tick budget, 40
trials:

| step (edges/tick) | median ticks (of those that reacted) | never left `active` |
|---|---|---|
| 20 → 1 | 13 | 0/40 |
| 5 → 0 | 26 | 0/40 |
| 5 → 1 | 23 | **34/40** |
| 3 → 1 | 17 | **39/40** |

A drop to zero is caught reliably at any pre-step rate, because the
Allan deviation collapses under the stall threshold. A *partial*
slowdown at a low pre-step rate is mostly not caught at all, where the
old estimator caught `5 → 1` in a median 6 ticks.

**Read this carefully before calling it a regression.** `5 → 1` means
discovery is continuing at 1 edge/tick, which is not a stall; and
`noise == "active"` only declines to *pre-emptively halve* the stall
threshold (`fuzzer.py`, the `fatiguing` branch). It does not suppress a
genuine stall, since a true stop is caught in a median 26 ticks over
40/40. And the sensitivity the old estimator had at low rates was the
same mechanism as its `p10 = 1` dead-time tail: firing on Poisson noise.
So the trade is a large false-alarm rate for less pre-emptive fatigue
warning at low absolute rates, and it is very likely worth it.

**Why this is an E item and not a P-anything:** deciding it needs
campaign data on how often a low-rate partial slowdown actually
precedes a stall, which no synthetic harness can supply. The before and
after tables are in
`docs/handover/handover_control_theory_loops_2026-09-12.md` §2.1, whose
dead-time characterisation had to be re-measured for the same reason.

### 4.2 P2-T4 — the first pass documented it; the second closed it

`6cdba43` corrected the docstrings, marked `crooks_forward_reverse` as
not Crooks, and exposed the value under a second key, `renyi_entropy`,
while leaving the call site untouched. So `--fluctuation` went on
publishing the same degenerate number, now labelled with the precise
identity it fails to satisfy in production. That is further into the
middle state this tier exists to get out of, not out of it.

`155bb54` closes it via route 1 (wire and rename). The shape of the fix
is a contract rather than a better fallback, which is the part worth
carrying to the next item like this:

- A scheduler opts in by exposing `last_selection_probs()`.
  `TrajectoryRecord.probs_are_true` defaults to **False**, so a caller
  has to assert the property rather than remember to deny it.
  `WorkFunctional` counts but never pools records without it, so
  `jarzynski_estimator` returns None instead of a number shaped like an
  entropy.
- `exp3` is the only scheduler that can answer today; it already keeps
  that mixture for its own importance-weighted estimator. The UCB family
  selects by deterministic argmax and has **no selection law at all**,
  which is why no fallback is the correct answer and not a gap.
- Measured end to end: the default scheduler reports
  `fluc: W=0.00 n=0 unpooled=3919`; with `--exp3`,
  `fluc: W=10.06 n=1000 H2=7.94` — a real Rényi-2 entropy of the
  operator-path distribution.
- `_SELECTION_PROB_SOURCES` holds the opt-in list, and
  `tests/test_regression_fluctuation_probs.py` asserts every name in it
  is a real attribute. That guard is the point: the defect it replaces
  was a `hasattr` on a name nothing assigned, which disabled the feature
  silently instead of failing.

**Two further defects found in the same module while fixing it**, both
in `155bb54`:

1. `state_key` hashed the operator tuple with the builtin `hash()`,
   which is salted per process. Verified: three `PYTHONHASHSEED` values
   give three different keys for the same trajectory, so state restored
   from disk was orphaned under a key the new process cannot reproduce,
   and `--seed` did not determine the keys. Same defect class as the LSH
   banding removed from crash clustering. Now xxhash/sha256, matching
   what the edge branch already did, with no sign leak from the hex of a
   negative int.
2. The two scheduler-aware branches of `_op_probability` (mopt, elo)
   returned the fallback's value verbatim. They went with the function.

`crooks_forward_reverse` is retired rather than kept as a diagnostic:
a ratio of mean works between two arbitrary state buffers has no reverse
protocol, no matched-`W` density ratio and no crossing at ΔF, and its own
comment — "for identical work distributions the ratio is centered at
1.0" — is true of any ratio of equal means.

### 4.3 Collateral from the rename, fixed in `73a4996`

`29f0d60` updated every `.py` and left **nine references** across five
docs plus `docs/architecture.dot` and the generated
`docs/images/architecture.svg`. One was an inlined code block whose
stated purpose is that its numbers are "reproducible from this document
alone", and it no longer imported. Two needed re-measurement rather than
renaming, because the estimator changed underneath their numbers; see
§4.1a. One turned out not to be rename drift at all but a pre-existing
false claim — `handover_ports_pending.md:16` said `allan_variance.py`
feeds `core/seed_quality.py`, which contains no reference to it and never
did.

The analyzer spec name also changed, `allan` → `structure_function`.
Checked: spec names are internal to `AnalyzerRegistry`, not persisted in
state and not exposed as a CLI flag, so the rename crosses no
compatibility boundary. Recorded because it is the kind of thing that
looks like one.

### 4.4 Lesson, for the next document like this

The 09-12 revision answered by inventory and was wrong four times. The
09-13 audit of *that* was right about the defects and still incomplete
in the same direction: it specified an acceptance test for P0-T1 and did
not ask what the fix would cost, so the low-rate sensitivity in §4.1a
had to be found after the fact — and it invalidated a measured table in
a neighbouring document that neither the fix nor the audit thought to
check. **When a recommendation replaces an estimator, the obligation is
not just to state the acceptance test but to name what currently
depends on the old estimator's numbers.** `grep` for the class name
finds the callers; it does not find the measurements.
